"""Inference in Kafka mode: frame events in, detection events out.

The GPU path is the same as in file mode: frames go through the same
:class:`~crosswalk.inference.batcher.Batcher` and
:class:`~crosswalk.inference.pipeline.InferenceWorker`, so dynamic batching,
the latency histograms and the dashboard behave identically. What changes is
where frames come from and where results go:

1. :class:`~crosswalk.stream.stage.StageRunner` polls a batch of frame events
   (deduplicated by ``frame_id``).
2. :class:`InferenceHandler` downloads the JPEGs from S3 concurrently, decodes
   them in memory, and submits every frame to the batcher at once, so frames
   from several cameras share GPU batches.
3. The worker's results are matched back to their events and returned as
   detection events; the runner publishes them, then commits the offsets.

Scaling: one inference pod holds the one GPU, and the frames topic has a few
partitions, so a second pod (on a second GPU, or the CPU image) joins the same
consumer group and Kafka splits the partitions between them. More pods than
partitions leaves the extras idle; raise ``CROSSWALK_TOPIC_PARTITIONS`` on the
sampler before the topic is first created, or add partitions with Kafka's
tools (existing cameras may then move partitions, which tracking tolerates
only at stream boundaries; see docs/streaming.md).
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from crosswalk.config import InferenceConfig
from crosswalk.inference.batcher import Batch, Batcher, Timing
from crosswalk.inference.pipeline import InferenceWorker
from crosswalk.inference.predictor import Detection
from crosswalk.stream import events as ev
from crosswalk.stream.bus import Bus, KafkaBus, TopicSpec
from crosswalk.stream.stage import (
    Output,
    RunnerConfig,
    StageMetrics,
    StageRunner,
    TransientError,
    outputs_for,
)

log = logging.getLogger(__name__)


class FrameFetcher:
    """Fetch frame images by S3 reference (or local path in demo mode).

    S3 errors are *transient* (endpoint down, credentials rotated, throttled):
    the runner retries the batch instead of dead-lettering frames that are
    fine. A missing object (404) or an undecodable image is permanent for that
    frame and goes to the dead-letter topic.
    """

    def __init__(self, get_bytes: Callable[[ev.ObjectRef], bytes | None], resize=None):
        self.get_bytes = get_bytes
        self.resize = resize

    def __call__(self, ref: ev.ObjectRef) -> np.ndarray:
        from PIL import Image

        try:
            data = self.get_bytes(ref)
        except Exception as err:
            raise TransientError(f"fetching s3://{ref.bucket}/{ref.key}: {err}") from err
        if data is None:
            raise FileNotFoundError(f"s3://{ref.bucket}/{ref.key} does not exist")
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            if self.resize:
                image = image.resize(tuple(self.resize))
            return np.asarray(image)


def s3_getter(client=None) -> Callable[[ev.ObjectRef], bytes | None]:
    from crosswalk.storage import S3Store

    stores: dict[str, S3Store] = {}

    def get(ref: ev.ObjectRef) -> bytes | None:
        if ref.bucket not in stores:
            stores[ref.bucket] = S3Store(ref.bucket, client=client)
        return stores[ref.bucket].get_bytes(ref.key)

    return get


class InferenceHandler:
    name = "inference"

    def __init__(
        self,
        batcher: Batcher,
        fetch: Callable[[ev.ObjectRef], np.ndarray],
        model_version: str,
        fetch_workers: int = 8,
        timeout_s: float = 120.0,
    ):
        self.batcher = batcher
        self.fetch = fetch
        self.model_version = model_version
        self.pool = ThreadPoolExecutor(max_workers=fetch_workers, thread_name_prefix="fetch")
        self.timeout_s = timeout_s
        # frame_id -> slot filled by the worker thread.
        self._pending: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    # Called on the inference worker thread.
    def on_result(self, meta: dict[str, Any], detections: list[Detection], timing: Timing) -> None:
        with self._lock:
            slot = self._pending.get(meta["frame_id"])
        if slot is not None:
            slot["detections"], slot["timing"] = detections, timing
            slot["done"].set()

    def on_error(self, batch: Batch, error: Exception) -> None:
        for request in batch.requests:
            with self._lock:
                slot = self._pending.get(request.meta["frame"]["frame_id"])
            if slot is not None:
                slot["error"] = error
                slot["done"].set()

    def event_id(self, event: ev.Event) -> str | None:
        if isinstance(event, ev.FrameEvent):
            return event.frame_id
        return None  # end events are cheap and must always pass through

    def process(self, events: Sequence[ev.Event]) -> list[list[Output] | BaseException]:
        results: list[list[Output] | BaseException | None] = [None] * len(events)
        frames = [(i, e) for i, e in enumerate(events) if isinstance(e, ev.FrameEvent)]
        for i, e in enumerate(events):
            if isinstance(e, ev.EndEvent):
                results[i] = outputs_for(ev.TOPIC_DETECTIONS, [e])
            elif not isinstance(e, ev.FrameEvent):
                results[i] = ValueError(f"inference does not consume {e.type!r} events")

        # Download and decode concurrently; a transient S3 error fails the
        # whole batch (raised), a broken frame fails only itself.
        futures = {i: self.pool.submit(self.fetch, e.image) for i, e in frames}
        images: dict[int, np.ndarray] = {}
        for i, future in futures.items():
            try:
                images[i] = future.result()
            except TransientError:
                raise
            except Exception as err:
                results[i] = err

        slots = {}
        for i, e in frames:
            if i not in images:
                continue
            slot = {"done": threading.Event()}
            with self._lock:
                self._pending[e.frame_id] = slot
            slots[i] = slot
            # Blocking submit: back-pressure from the GPU slows the consumer
            # down (and lag grows) rather than dropping frames that Kafka has
            # already durably stored.
            self.batcher.submit(
                images[i],
                stream=e.camera_id,
                block=True,
                meta={"frame": {"frame_id": e.frame_id}},
            )
        deadline = time.monotonic() + self.timeout_s
        for i, slot in slots.items():
            e = events[i]
            finished = slot["done"].wait(max(0.0, deadline - time.monotonic()))
            with self._lock:
                self._pending.pop(e.frame_id, None)
            if not finished:
                results[i] = TimeoutError(
                    f"no inference result for {e.frame_id} in {self.timeout_s}s"
                )
            elif "error" in slot:
                results[i] = slot["error"]
            else:
                results[i] = outputs_for(ev.TOPIC_DETECTIONS, [self._detection_event(e, slot)])
        return [r if r is not None else RuntimeError("unprocessed") for r in results]

    def _detection_event(self, e: ev.FrameEvent, slot: dict[str, Any]) -> ev.DetectionEvent:
        timing: Timing = slot["timing"]
        return ev.DetectionEvent(
            frame_id=e.frame_id,
            stream_id=e.stream_id,
            camera_id=e.camera_id,
            frame_index=e.frame_index,
            pts_s=e.pts_s,
            width=e.width,
            height=e.height,
            decoded_at=e.decoded_at,
            model_version=self.model_version,
            detections=[
                ev.Box(
                    label=d.label, score=round(d.score, 4), box=tuple(round(v, 1) for v in d.box)
                )
                for d in slot["detections"]
            ],
            latency={
                "queue_wait": round(timing.queue_wait, 6),
                "inference": round(timing.inference, 6),
                "end_to_end": round(timing.end_to_end, 6),
            },
        )

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


def build(
    config: InferenceConfig,
    bus: Bus,
    predictor,
    metrics,
    fetch: Callable[[ev.ObjectRef], np.ndarray] | None = None,
    gpu_memory=None,
) -> tuple[StageRunner, InferenceWorker, InferenceHandler, Batcher]:
    """Wire batcher, worker, handler and runner (shared by the service and tests)."""
    kafka = config.kafka
    batcher = Batcher(config.batching)
    handler = InferenceHandler(
        batcher,
        fetch or FrameFetcher(s3_getter(), kafka.resize),
        predictor.model_version,
        fetch_workers=kafka.fetch_workers,
    )

    def on_result(meta, detections, timing):
        handler.on_result(meta, detections, timing)

    worker = InferenceWorker(
        batcher,
        predictor,
        metrics,
        on_result=on_result,
        gpu_memory=gpu_memory,
        on_error=handler.on_error,
    )
    topics = [
        TopicSpec(ev.TOPIC_FRAMES, kafka.partitions, kafka.replication_factor),
        TopicSpec(ev.TOPIC_DETECTIONS, kafka.partitions, kafka.replication_factor),
    ]
    runner = StageRunner(
        handler,
        bus,
        RunnerConfig(
            group=kafka.group or "crosswalk-inference",
            input_topics=[ev.TOPIC_FRAMES],
            topics=topics,
            max_records=kafka.max_records,
            poll_timeout_s=kafka.poll_timeout_s,
            retries=kafka.retries,
            dedupe_capacity=kafka.dedupe_capacity,
            client_id="crosswalk",
        ),
        metrics=StageMetrics(metrics.registry) if hasattr(metrics, "registry") else None,
    )
    return runner, worker, handler, batcher


def run_kafka(
    config: InferenceConfig, metrics, state, stop: threading.Event, server, predictor=None, bus=None
):
    """The inference Deployment's main loop in Kafka mode."""
    from crosswalk.inference.service import load_predictor

    shape = config.kafka.resize or config.source.resize or (1280, 720)
    predictor, gpu_memory = load_predictor(config, metrics, state, shape, predictor)
    bus = bus or KafkaBus()
    runner, worker, handler, batcher = build(config, bus, predictor, metrics, gpu_memory=gpu_memory)
    state["phase"] = "connecting to Kafka"
    # Kafka may still be starting (or Argo CD may sync this app first): keep
    # retrying the topic setup instead of crash-looping.
    while not stop.is_set():
        try:
            runner.start()
            break
        except Exception as err:
            state["kafka_error"] = f"{type(err).__name__}: {err}"
            log.warning("Kafka not reachable yet (%s); retrying in 10 s", err)
            if server is not None:
                server.ready.set()  # answering /status usefully, like "waiting for footage"
            stop.wait(10)
    if stop.is_set():
        return state
    state.pop("kafka_error", None)
    worker.start()
    metrics.ready.set(1)
    if server is not None:
        server.ready.set()
    state["phase"] = "consuming"

    def watch_stop():
        stop.wait()
        runner.stop.set()

    threading.Thread(target=watch_stop, daemon=True, name="stop-watch").start()
    try:
        state["counts"] = runner.counts
        runner.run()
    finally:
        batcher.close()
        worker.join(timeout=30)
        handler.close()
        runner.close()
    state["phase"] = "finished"
    return state
