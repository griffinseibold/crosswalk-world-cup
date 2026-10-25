"""The inference Deployment's main loop and the one-shot `crosswalk infer` command.

``serve`` plays the configured footage (optionally looping, paced like a
camera), keeps the status server up, and writes one JSON-lines file per pass
over each video. ``infer`` is the same pipeline in offline mode: decode as fast
as the GPU accepts, never drop, exit when done.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any

from crosswalk import storage
from crosswalk.config import InferenceConfig
from crosswalk.inference import video
from crosswalk.inference.batcher import Batcher
from crosswalk.inference.metrics import Metrics
from crosswalk.inference.pipeline import (
    DetectionWriter,
    InferenceWorker,
    detection_record,
    feed,
    frame_meta_factory,
)

log = logging.getLogger(__name__)


def resolve_videos(config: InferenceConfig) -> list[str]:
    local: list[str] = []
    for item in config.source.videos:
        if item.startswith("s3://"):
            local.extend(
                storage.expand_s3_videos(item, config.source.cache_dir, video.VIDEO_SUFFIXES)
            )
        else:
            local.extend(video.list_videos([item]))
    if not local:
        raise SystemExit("source.videos lists no video files")
    return local


FOOTAGE_RETRY_S = 60


def _wait_for_videos(
    config: InferenceConfig, state: dict[str, Any], stop: threading.Event, server=None
) -> list[str]:
    """In the service, missing footage is a state, not a crash.

    A fresh install has an empty footage prefix until someone uploads blurred
    clips. Exiting would crash-loop the Deployment; instead the status page
    says what is missing and the source is checked again every minute.
    """
    while not stop.is_set():
        try:
            return resolve_videos(config)
        except (SystemExit, FileNotFoundError, OSError) as err:
            state["phase"] = "waiting for footage"
            state["footage_error"] = str(err)
            log.warning("no footage yet (%s); retrying in %ds", err, FOOTAGE_RETRY_S)
        except Exception as err:  # e.g. S3 credentials or endpoint wrong
            state["phase"] = "waiting for footage"
            state["footage_error"] = f"{type(err).__name__}: {err}"
            log.warning("cannot list footage (%s); retrying in %ds", err, FOOTAGE_RETRY_S)
        if server is not None:
            # Ready means "answering /status and /metrics usefully", which it
            # is while waiting; crosswalk_ready stays 0 until the model loads.
            # Otherwise Argo CD would mark a fresh, footage-less install Degraded.
            server.ready.set()
        stop.wait(FOOTAGE_RETRY_S)
    return []


def _output_store(config: InferenceConfig):
    if config.output.s3_prefix:
        return storage.open_store(config.output.s3_prefix)
    return None


def load_predictor(config: InferenceConfig, metrics, state: dict[str, Any], shape, predictor=None):
    """Load and warm up the detector (unless one is injected, as in tests).

    ``shape`` is (width, height) of the frames it will see, so warmup runs the
    shapes MIOpen/cuDNN will actually meet. Returns (predictor, gpu_memory),
    where gpu_memory reports PyTorch's allocated bytes (None on CPU/fakes).
    """
    if predictor is not None:
        metrics.set_model_info(model="external", weights=predictor.model_version)
        return predictor, None
    from crosswalk import device as devices
    from crosswalk.inference.predictor import TorchPredictor

    state["phase"] = "loading model"
    predictor = TorchPredictor(config.model, config.device, config.precision)
    info = devices.describe(predictor.device)
    sizes = [1, config.batching.effective_batch_size]
    if config.batching.strategy == "dynamic":
        sizes = list(range(1, config.batching.max_batch_size + 1))
    predictor.warmup((shape[1], shape[0]), sizes)

    def gpu_memory():
        return devices.memory_allocated(predictor.device)

    metrics.set_model_info(
        model=config.model.name,
        weights=predictor.model_version,
        device=info.get("gpu_name") or "cpu",
        gpu_arch=info.get("gpu_arch"),
        backend=info.get("gpu_backend") or "cpu",
        torch=info.get("torch"),
        hip=info.get("torch_hip"),
        strategy=config.batching.strategy,
        max_batch_size=config.batching.effective_batch_size,
        precision=config.precision,
    )
    state["runtime"] = info
    return predictor, gpu_memory


def run(config: InferenceConfig, *, serve: bool, predictor=None) -> dict[str, Any]:
    metrics = Metrics()
    stop = threading.Event()
    state: dict[str, Any] = {"phase": "starting", "passes": 0, "videos": []}

    server = None
    if serve:
        from crosswalk.inference.server import StatusServer

        server = StatusServer(config.server.port, metrics, lambda: dict(state))
        server.start()
        log.info("status server on :%d", server.port)

    def handle_signal(signum, _frame):
        log.info("signal %d: finishing the current batch and stopping", signum)
        stop.set()

    previous_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, handle_signal)

    if config.mode == "kafka":
        from crosswalk.inference.stream_mode import run_kafka

        try:
            return run_kafka(config, metrics, state, stop, server, predictor=predictor)
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            metrics.ready.set(0)
            if server is not None:
                server.stop()

    try:
        videos = _wait_for_videos(config, state, stop, server) if serve else resolve_videos(config)
        if not videos:
            return state  # stopped while waiting
        state["videos"] = [Path(v).name for v in videos]

        if config.source.resize:
            shape = config.source.resize
        else:
            first = video.probe(videos[0])
            shape = (first.width, first.height)
        if server is not None and predictor is None:
            server.ready.clear()
        predictor, gpu_memory = load_predictor(config, metrics, state, shape, predictor)

        metrics.ready.set(1)
        if server is not None:
            server.ready.set()
        state["phase"] = "running"

        writer = DetectionWriter(config.output.dir, store=_output_store(config))
        totals = {"submitted": 0, "dropped": 0}
        while not stop.is_set():
            run_id = time.strftime("%Y%m%dT%H%M%S")
            for path in videos:
                if stop.is_set():
                    break
                info = video.probe(path)
                stream = Path(path).stem
                name = f"{stream}-{run_id}"
                batcher = Batcher(config.batching)

                def on_result(meta, detections, timing, name=name):
                    writer.write(
                        name, detection_record(meta, detections, timing, predictor.model_version)
                    )

                worker = InferenceWorker(
                    batcher, predictor, metrics, on_result=on_result, gpu_memory=gpu_memory
                )
                worker.start()
                frames = video.iter_frames(path, config.source.resize, config.source.max_frames)
                submitted, dropped = feed(
                    batcher,
                    frames,
                    stream=stream,
                    realtime=config.source.realtime,
                    fps=info.fps,
                    metrics=metrics,
                    stop=stop,
                    meta=frame_meta_factory(stream),
                )
                batcher.close()
                worker.join()
                writer.close(name)
                totals["submitted"] += submitted
                totals["dropped"] += dropped
                log.info("%s: %d frames, %d dropped", Path(path).name, submitted, dropped)
            state["passes"] += 1
            state.update(totals)
            if not config.source.loop:
                break
        state["phase"] = "finished"
        state["detection_files"] = [str(p) for p in writer.published]

        if serve and not stop.is_set():
            # Footage finished without looping: keep serving metrics and status
            # so the Deployment stays Ready instead of crash-looping.
            log.info("footage finished; serving status until stopped")
            stop.wait()
        return state
    except Exception:
        metrics.errors.labels(stage="startup").inc()
        state["phase"] = "failed"
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        metrics.ready.set(0)
        if server is not None:
            server.stop()
