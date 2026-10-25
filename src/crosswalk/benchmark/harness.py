"""Experiment A: throughput and latency across batching strategies.

Methodology (also in docs/benchmarks.md):

1. Frames are decoded once, resized to each resolution, and held in memory,
   so decode speed and disk do not leak into GPU measurements. (Decode cost is
   measured separately by the service's ``crosswalk_decode_seconds`` metric.)
2. Before each configuration, every batch size it can produce is run
   ``warmup_batches`` times and discarded, so one-off kernel selection
   (MIOpen's find step on ROCm) does not land in the results.
3. ``concurrency`` producer threads each submit an equal share of
   ``frames_per_run`` frames into one batcher feeding one GPU worker. In
   ``offline`` mode producers submit as fast as the queue accepts (throughput
   ceiling); in ``realtime`` mode each producer submits at ``fps`` like a
   camera and drops frames when the queue is full.
4. Every (resolution, strategy, concurrency) cell is repeated ``repeats``
   times. Percentiles come from every per-frame sample, not from histograms.
5. Each result file records the hardware, driver, ROCm, PyTorch, model,
   config hash and git commit, so two files can be compared knowing exactly
   what differed.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from crosswalk import hardware, stats
from crosswalk.config import BenchmarkConfig, config_hash
from crosswalk.inference.batcher import Batcher
from crosswalk.inference.metrics import NullMetrics
from crosswalk.inference.pipeline import InferenceWorker, RunStats
from crosswalk.inference.predictor import Predictor

log = logging.getLogger(__name__)

RESULT_SCHEMA_VERSION = 1

# Required keys of a result file and of each run record. Tests check produced
# files against these; docs/benchmarks.md describes each field.
RESULT_KEYS = {
    "schema",
    "experiment",
    "created_at",
    "config",
    "config_hash",
    "environment",
    "method",
    "runs",
    "summary",
}
ENVIRONMENT_KEYS = {"hardware", "runtime", "model"}
RUN_KEYS = {
    "resolution",
    "strategy",
    "batching",
    "concurrency",
    "mode",
    "repeat",
    "frames_submitted",
    "frames_completed",
    "frames_dropped",
    "errors",
    "wall_s",
    "throughput_fps",
    "latency_ms",
    "queue_wait_ms",
    "inference_ms",
    "batch_size",
    "gpu_memory_peak_bytes",
}
LATENCY_KEYS = {"count", "mean", "p50", "p95", "p99", "max"}

METHOD = {
    "timer": "time.monotonic around predictor.predict, which ends with a GPU synchronize",
    "frames": "decoded once and held in memory; decode excluded",
    "warmup": "every producible batch size run warmup_batches times per configuration, discarded",
    "percentiles": "linear interpolation over all per-frame samples",
    "throughput": "completed frames / wall time from first batch start to last batch end",
}


def validate_result(result: dict[str, Any]) -> list[str]:
    """Return a list of schema problems (empty means valid)."""
    problems = [f"missing key {k!r}" for k in RESULT_KEYS - result.keys()]
    env = result.get("environment", {})
    problems += [f"environment missing {k!r}" for k in ENVIRONMENT_KEYS - env.keys()]
    for i, run in enumerate(result.get("runs", [])):
        problems += [f"run {i} missing {k!r}" for k in RUN_KEYS - run.keys()]
        for field in ("latency_ms", "queue_wait_ms", "inference_ms"):
            missing = LATENCY_KEYS - run.get(field, {}).keys()
            problems += [f"run {i} {field} missing {k!r}" for k in missing]
    return problems


def synthetic_frames(count: int, width: int, height: int, seed: int = 0) -> list[np.ndarray]:
    """Noise frames: same tensor shapes as real footage, no files needed.

    Detection post-processing (NMS) costs depend on how many boxes the model
    proposes, so synthetic numbers are a GPU-throughput reference, not a
    substitute for real footage. Results record which source was used.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    # A few distinct frames cycled, so memory stays bounded for long runs.
    return [np.roll(base, shift=i * 7, axis=1) for i in range(min(count, 16))]


def load_frames(config: BenchmarkConfig, width: int, height: int) -> tuple[list[np.ndarray], float]:
    if config.source.video is None:
        return synthetic_frames(config.source.frames, width, height), config.source.fps or 30.0
    from crosswalk.inference import video

    info = video.probe(config.source.video)
    frames = [
        f.image
        for f in video.iter_frames(config.source.video, (width, height), config.source.frames)
    ]
    if not frames:
        raise SystemExit(f"{config.source.video}: no frames decoded")
    return frames, config.source.fps or info.fps


def _producer(batcher: Batcher, frames, count: int, stream: str, realtime: bool, fps: float, out):
    submitted = dropped = 0
    start = time.monotonic()
    for i in range(count):
        if realtime:
            delay = start + i / fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        request = batcher.submit(frames[i % len(frames)], stream=stream, block=not realtime)
        if request is None:
            dropped += 1
        else:
            submitted += 1
    out[stream] = (submitted, dropped)


def run_cell(
    predictor: Predictor,
    frames: list[np.ndarray],
    batching,
    concurrency: int,
    total_frames: int,
    mode: str,
    fps: float,
    warmup_batches: int,
    peak_memory: Callable[[], int | None] | None = None,
    reset_peak: Callable[[], None] | None = None,
) -> dict[str, Any]:
    sizes = (
        list(range(1, batching.max_batch_size + 1))
        if batching.strategy == "dynamic"
        else [batching.effective_batch_size]
    )
    for size in sizes:
        for _ in range(warmup_batches):
            predictor.predict([frames[0]] * size)
    if reset_peak:
        reset_peak()

    batcher = Batcher(batching)
    run_stats = RunStats()
    worker = InferenceWorker(batcher, predictor, NullMetrics(), stats=run_stats)
    worker.start()
    per_stream = [total_frames // concurrency] * concurrency
    for i in range(total_frames % concurrency):
        per_stream[i] += 1
    results: dict[str, tuple[int, int]] = {}
    threads = [
        threading.Thread(
            target=_producer,
            args=(batcher, frames, n, f"s{i}", mode == "realtime", fps, results),
            daemon=True,
        )
        for i, n in enumerate(per_stream)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    batcher.close()
    worker.join()

    submitted = sum(s for s, _ in results.values())
    dropped = sum(d for _, d in results.values())
    wall = (
        (run_stats.finished_at - run_stats.started_at)
        if run_stats.started_at is not None and run_stats.finished_at is not None
        else 0.0
    )
    ms = 1000.0

    def summary_ms(values):
        s = stats.summarize([v * ms for v in values])
        return s

    return {
        "frames_submitted": submitted,
        "frames_completed": run_stats.frames,
        "frames_dropped": dropped,
        "errors": run_stats.errors,
        "wall_s": wall,
        "throughput_fps": run_stats.frames / wall if wall > 0 else None,
        "latency_ms": summary_ms([t.end_to_end for t in run_stats.timings]),
        "queue_wait_ms": summary_ms([t.queue_wait for t in run_stats.timings]),
        "inference_ms": summary_ms(run_stats.inference_s),
        "batch_size": {
            "mean": (sum(run_stats.batch_sizes) / len(run_stats.batch_sizes))
            if run_stats.batch_sizes
            else None,
            "histogram": {
                str(k): run_stats.batch_sizes.count(k) for k in sorted(set(run_stats.batch_sizes))
            },
        },
        "gpu_memory_peak_bytes": peak_memory() if peak_memory else None,
    }


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median across repeats for each cell (the median resists one noisy repeat)."""
    cells: dict[tuple, list[dict[str, Any]]] = {}
    for run in runs:
        key = (tuple(run["resolution"]), run["strategy"], run["concurrency"])
        cells.setdefault(key, []).append(run)
    out = []
    for (res, strategy, conc), group in cells.items():

        def med(path, group=group):
            values = []
            for r in group:
                v = r
                for p in path:
                    v = v.get(p) if isinstance(v, dict) else None
                if v is not None:
                    values.append(v)
            return stats.percentile(values, 50)

        out.append(
            {
                "resolution": list(res),
                "strategy": strategy,
                "concurrency": conc,
                "repeats": len(group),
                "throughput_fps_median": med(["throughput_fps"]),
                "latency_p50_ms_median": med(["latency_ms", "p50"]),
                "latency_p95_ms_median": med(["latency_ms", "p95"]),
                "latency_p99_ms_median": med(["latency_ms", "p99"]),
                "batch_size_mean_median": med(["batch_size", "mean"]),
                "frames_dropped_total": sum(r["frames_dropped"] for r in group),
            }
        )
    return out


def run_benchmark(
    config: BenchmarkConfig,
    predictor: Predictor | None = None,
    runtime: dict[str, Any] | None = None,
    frame_loader: Callable[[BenchmarkConfig, int, int], tuple[list[np.ndarray], float]] = load_frames,
) -> dict[str, Any]:
    peak_memory = reset_peak = None
    if predictor is None:
        import torch

        from crosswalk import device as devices
        from crosswalk.inference.predictor import TorchPredictor

        tp = TorchPredictor(config.model, config.device, config.precision)
        predictor = tp
        runtime = devices.describe(tp.device)
        if tp.device.type == "cuda":

            def peak_memory():
                return devices.max_memory_allocated(tp.device)

            def reset_peak():
                torch.cuda.reset_peak_memory_stats(tp.device)

    runs: list[dict[str, Any]] = []
    for width, height in config.resolutions:
        frames, fps = frame_loader(config, width, height)
        for spec, concurrency, repeat in itertools.product(
            config.strategies, config.concurrency, range(config.repeats)
        ):
            log.info(
                "%dx%d %s concurrency=%d repeat=%d", width, height, spec.name, concurrency, repeat
            )
            cell = run_cell(
                predictor,
                frames,
                spec.batching,
                concurrency,
                config.frames_per_run,
                config.mode,
                fps,
                config.warmup_batches,
                peak_memory,
                reset_peak,
            )
            runs.append(
                {
                    "resolution": [width, height],
                    "strategy": spec.name,
                    "batching": spec.batching.model_dump(),
                    "concurrency": concurrency,
                    "mode": config.mode,
                    "repeat": repeat,
                    "source_fps": fps,
                    **cell,
                }
            )

    return {
        "schema": RESULT_SCHEMA_VERSION,
        "experiment": config.experiment,
        "created_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": config.model_dump(mode="json"),
        "config_hash": config_hash(config),
        "environment": {
            "hardware": hardware.collect(),
            "runtime": runtime or {},
            "model": {
                "name": config.model.name,
                "weights": config.model.weights,
                "checkpoint": config.model.checkpoint,
                "version": predictor.model_version,
                "precision": config.precision,
                "source": config.source.video or "synthetic",
            },
        },
        "method": METHOD,
        "runs": runs,
        "summary": summarize_runs(runs),
    }


def write_result(result: dict[str, Any], output_dir: str) -> Path:
    from crosswalk.storage import atomic_write_bytes

    stamp = result["created_at"].replace(":", "").replace("-", "")
    path = Path(output_dir) / f"experiment-{result['experiment']}-{stamp}-{result['config_hash']}.json"
    atomic_write_bytes(path, json.dumps(result, indent=2, default=str).encode() + b"\n")
    return path


def markdown_table(result: dict[str, Any]) -> str:
    """Summary rows ready to paste into docs/benchmarks.md."""

    def fmt(v, digits=1):
        return "" if v is None else f"{v:.{digits}f}"

    lines = [
        "| Resolution | Strategy | Concurrency | Throughput (fps) | p50 (ms) | p95 (ms) | p99 (ms) | Mean batch | Dropped |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["summary"]:
        lines.append(
            f"| {row['resolution'][0]}x{row['resolution'][1]} | {row['strategy']} | {row['concurrency']} "
            f"| {fmt(row['throughput_fps_median'])} | {fmt(row['latency_p50_ms_median'])} "
            f"| {fmt(row['latency_p95_ms_median'])} | {fmt(row['latency_p99_ms_median'])} "
            f"| {fmt(row['batch_size_mean_median'], 2)} | {row['frames_dropped_total']} |"
        )
    return "\n".join(lines)
