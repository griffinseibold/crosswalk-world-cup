"""Decode -> batch -> detect -> write, with latency accounting at each step.

Threads:

- one *source* thread per video stream decodes frames and submits them to the
  :class:`~crosswalk.inference.batcher.Batcher` (in real-time mode it paces
  itself to the file's frame rate and drops frames the GPU cannot keep up with);
- one *worker* thread takes batches, runs the predictor, records metrics and
  writes detections.

One worker is deliberate: there is one GPU, and two threads issuing batches to
it would contend rather than overlap. Decoding runs in parallel because
PyAV releases the GIL while FFmpeg decodes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosswalk.config import BatchingConfig
from crosswalk.inference.batcher import Batch, Batcher, Timing, timings_for
from crosswalk.inference.predictor import Detection, Predictor
from crosswalk.storage import atomic_copy

log = logging.getLogger(__name__)

DETECTION_SCHEMA_VERSION = 1


@dataclass
class FrameMeta:
    video: str
    stream: str
    index: int
    pts_s: float
    width: int
    height: int
    decoded_at: float  # wall clock (time.time()) for the output record


def detection_record(
    meta: FrameMeta, detections: list[Detection], timing: Timing, model_version: str
) -> dict[str, Any]:
    """One JSON-lines record. The schema is documented in docs/benchmarks.md."""
    return {
        "schema": DETECTION_SCHEMA_VERSION,
        "video": Path(meta.video).name,
        "stream": meta.stream,
        "frame": meta.index,
        "pts_s": round(meta.pts_s, 4),
        "width": meta.width,
        "height": meta.height,
        "decoded_at": round(meta.decoded_at, 3),
        "model_version": model_version,
        "batch_size": timing.batch_size,
        "latency_ms": {
            "queue_wait": round(timing.queue_wait * 1000, 3),
            "inference": round(timing.inference * 1000, 3),
            "end_to_end": round(timing.end_to_end * 1000, 3),
        },
        "detections": [d.to_json() for d in detections],
    }


class DetectionWriter:
    """Writes one JSON-lines file per (stream, pass) and publishes it when done.

    Lines go to ``<name>.jsonl.partial``; :meth:`close` renames it to
    ``<name>.jsonl`` and optionally uploads it, so a reader (or a later
    Milestone 3 consumer) never picks up a half-written file.
    """

    def __init__(self, out_dir: str, store=None, prefix: str = ""):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.prefix = prefix.strip("/")
        self._handles: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.records = 0
        self.published: list[Path] = []

    def open(self, name: str) -> None:
        with self._lock:
            if name not in self._handles:
                path = self.out_dir / f"{name}.jsonl.partial"
                # Kept open across many writes and closed in close(); a with
                # block cannot span calls.
                self._handles[name] = open(path, "w", encoding="utf-8")  # noqa: SIM115

    def write(self, name: str, record: dict[str, Any]) -> None:
        with self._lock:
            handle = self._handles.get(name)
            if handle is None:
                handle = self._handles[name] = open(  # noqa: SIM115 (closed in close())
                    self.out_dir / f"{name}.jsonl.partial", "w", encoding="utf-8"
                )
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            self.records += 1

    def close(self, name: str) -> Path | None:
        with self._lock:
            handle = self._handles.pop(name, None)
        if handle is None:
            return None
        handle.flush()
        handle.close()
        partial = self.out_dir / f"{name}.jsonl.partial"
        final = self.out_dir / f"{name}.jsonl"
        atomic_copy(partial, final)
        partial.unlink()
        if self.store is not None:
            key = f"{self.prefix}/{final.name}" if self.prefix else final.name
            self.store.put_file(str(final), key)
        self.published.append(final)
        return final

    def close_all(self) -> None:
        for name in list(self._handles):
            self.close(name)


@dataclass
class RunStats:
    """Raw samples for one run; the benchmark turns these into percentiles."""

    timings: list[Timing] = field(default_factory=list)
    inference_s: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    frames: int = 0
    errors: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, timings: list[Timing], inference_s: float, now: float) -> None:
        with self._lock:
            if self.started_at is None:
                self.started_at = now - inference_s
            self.timings.extend(timings)
            self.inference_s.append(inference_s)
            self.batch_sizes.append(len(timings))
            self.frames += len(timings)
            self.finished_at = now


class InferenceWorker:
    """Consumes batches until the batcher is closed and drained."""

    def __init__(
        self,
        batcher: Batcher,
        predictor: Predictor,
        metrics,
        on_result: Callable[[FrameMeta, list[Detection], Timing], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        stats: RunStats | None = None,
        gpu_memory: Callable[[], int | None] | None = None,
    ):
        self.batcher = batcher
        self.predictor = predictor
        self.metrics = metrics
        self.on_result = on_result
        self.clock = clock
        self.stats = stats or RunStats()
        self.gpu_memory = gpu_memory
        self.ignore_before: float | None = None  # warmup cut-off for stats
        self._thread: threading.Thread | None = None

    def process(self, batch: Batch) -> None:
        started = self.clock()
        try:
            results = self.predictor.predict(batch.payloads)
        except Exception:
            log.exception("inference failed for a batch of %d frames", len(batch))
            self.metrics.errors.labels(stage="inference").inc()
            self.stats.errors += 1
            return
        finished = self.clock()
        timings = timings_for(batch, started, finished)
        counts: dict[str, int] = defaultdict(int)
        for request in batch.requests:
            counts[request.stream] += 1
        self.metrics.observe_batch(counts, timings, finished - started)
        self.metrics.queue_depth.set(len(self.batcher))
        if self.gpu_memory is not None:
            used = self.gpu_memory()
            if used is not None:
                self.metrics.gpu_memory.set(used)
        if self.ignore_before is None or started >= self.ignore_before:
            self.stats.record(timings, finished - started, finished)
        for request, detections, timing in zip(batch.requests, results, timings, strict=True):
            for det in detections:
                self.metrics.detections.labels(label=det.label).inc()
            if self.on_result is not None:
                try:
                    self.on_result(request.meta.get("frame"), detections, timing)
                except Exception:
                    log.exception("writing detections failed")
                    self.metrics.errors.labels(stage="output").inc()

    def run(self) -> None:
        while True:
            batch = self.batcher.next_batch()
            if batch is None:
                return
            self.process(batch)

    def start(self) -> threading.Thread:
        self._thread = threading.Thread(target=self.run, name="inference-worker", daemon=True)
        self._thread.start()
        return self._thread

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)


def feed(
    batcher: Batcher,
    frames: Iterable,
    *,
    stream: str,
    realtime: bool,
    fps: float,
    metrics,
    stop: threading.Event | None = None,
    payload: Callable[[Any], Any] = lambda f: f.image,
    meta: Callable[[Any], FrameMeta] | None = None,
) -> tuple[int, int]:
    """Submit frames from one stream; returns (submitted, dropped).

    In real-time mode frames are released on the file's timeline (frame i at
    start + i / fps), like a camera, and a full queue drops the frame. Decode
    time counts against that budget, exactly as it would for a live source.
    """
    submitted = dropped = 0
    start = time.monotonic()
    iterator = iter(frames)
    index = 0
    while True:
        if stop is not None and stop.is_set():
            break
        t0 = time.monotonic()
        try:
            frame = next(iterator)
        except StopIteration:
            break
        except Exception:
            log.exception("decode failed on stream %s", stream)
            metrics.errors.labels(stage="decode").inc()
            break
        metrics.decode_latency.observe(time.monotonic() - t0)
        if realtime:
            due = start + index / fps
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        frame_meta = meta(frame) if meta else None
        request = batcher.submit(
            payload(frame),
            stream=stream,
            block=not realtime,
            meta={"frame": frame_meta},
        )
        if request is None:
            dropped += 1
            metrics.dropped.labels(stream=stream, reason="queue_full").inc()
        else:
            submitted += 1
        metrics.queue_depth.set(len(batcher))
        index += 1
    return submitted, dropped


def frame_meta_factory(stream: str):
    def make(frame) -> FrameMeta:
        return FrameMeta(
            video=frame.video,
            stream=stream,
            index=frame.index,
            pts_s=frame.pts_s,
            width=frame.width,
            height=frame.height,
            decoded_at=time.time(),
        )

    return make


def make_batcher(config: BatchingConfig) -> Batcher:
    return Batcher(config)
