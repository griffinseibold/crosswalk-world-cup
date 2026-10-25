"""Prometheus metrics for the inference service.

Quantiles (p50/p95/p99) are computed in Prometheus with ``histogram_quantile``
over these histograms, so the bucket edges below decide how precise the
dashboard is: they are dense between 5 ms and 300 ms, where detector latency
on a GPU lives. Benchmark results do not use these histograms; they compute
exact percentiles from raw samples (see crosswalk.stats).

``prometheus_client`` is imported lazily so the metric names can be checked
(tests compare them against the Grafana dashboard) without it installed.
"""

from __future__ import annotations

from collections.abc import Iterable

FRAMES = "crosswalk_frames_total"
DROPPED = "crosswalk_frames_dropped_total"
ERRORS = "crosswalk_errors_total"
INFERENCE_LATENCY = "crosswalk_inference_latency_seconds"
END_TO_END_LATENCY = "crosswalk_end_to_end_latency_seconds"
QUEUE_WAIT = "crosswalk_queue_wait_seconds"
DECODE_LATENCY = "crosswalk_decode_seconds"
BATCH_SIZE = "crosswalk_batch_size"
QUEUE_DEPTH = "crosswalk_queue_depth"
DETECTIONS = "crosswalk_detections_total"
GPU_MEMORY = "crosswalk_gpu_memory_allocated_bytes"
READY = "crosswalk_ready"
MODEL_INFO = "crosswalk_model"  # exposed as crosswalk_model_info

# Every name a dashboard may query (histograms add _bucket/_sum/_count, the
# info metric adds _info, counters are exposed with their _total name).
METRIC_NAMES = {
    FRAMES,
    DROPPED,
    ERRORS,
    INFERENCE_LATENCY,
    END_TO_END_LATENCY,
    QUEUE_WAIT,
    DECODE_LATENCY,
    BATCH_SIZE,
    QUEUE_DEPTH,
    DETECTIONS,
    GPU_MEMORY,
    READY,
    f"{MODEL_INFO}_info",
}

LATENCY_BUCKETS = (
    0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0,
)  # fmt: skip
BATCH_BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64)


def _counter_base(name: str) -> str:
    # prometheus_client appends _total itself.
    return name.removesuffix("_total")


class Metrics:
    def __init__(self, registry=None):
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

        # A private registry per instance keeps tests independent; the server
        # exposes exactly this registry.
        self.registry = registry if registry is not None else CollectorRegistry()
        r = self.registry
        self.frames = Counter(
            _counter_base(FRAMES), "Frames that finished inference", ["stream"], registry=r
        )
        self.dropped = Counter(
            _counter_base(DROPPED),
            "Frames discarded without inference (queue full in real-time mode)",
            ["stream", "reason"],
            registry=r,
        )
        self.errors = Counter(
            _counter_base(ERRORS), "Failures by pipeline stage", ["stage"], registry=r
        )
        self.inference_latency = Histogram(
            INFERENCE_LATENCY,
            "Wall time of one batched model call, including host-to-device copy and GPU synchronize",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.end_to_end_latency = Histogram(
            END_TO_END_LATENCY,
            "Per-frame time from entering the queue to detections ready",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.queue_wait = Histogram(
            QUEUE_WAIT,
            "Per-frame time spent in the batching queue",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.decode_latency = Histogram(
            DECODE_LATENCY, "Time to decode one frame", buckets=LATENCY_BUCKETS, registry=r
        )
        self.batch_size = Histogram(
            BATCH_SIZE, "Frames per model call", buckets=BATCH_BUCKETS, registry=r
        )
        self.queue_depth = Gauge(QUEUE_DEPTH, "Frames waiting for the model", registry=r)
        self.detections = Counter(
            _counter_base(DETECTIONS), "Detections above the score threshold", ["label"], registry=r
        )
        self.gpu_memory = Gauge(
            GPU_MEMORY, "Memory held by PyTorch tensors on the GPU", registry=r
        )
        self.ready = Gauge(READY, "1 once the model is loaded and warmed up", registry=r)
        self.model_info = Info(MODEL_INFO, "Model, weights and runtime serving frames", registry=r)

    def set_model_info(self, **labels: object) -> None:
        self.model_info.info({k: "" if v is None else str(v) for k, v in labels.items()})

    def observe_batch(self, stream_counts: dict[str, int], timings: Iterable, inference_s: float):
        timings = list(timings)
        self.inference_latency.observe(inference_s)
        self.batch_size.observe(len(timings))
        for t in timings:
            self.queue_wait.observe(t.queue_wait)
            self.end_to_end_latency.observe(t.end_to_end)
        for stream, count in stream_counts.items():
            self.frames.labels(stream=stream).inc(count)

    def render(self) -> bytes:
        from prometheus_client import generate_latest

        return generate_latest(self.registry)


class NullMetrics:
    """Same interface, records nothing (tests and the benchmark harness)."""

    def __getattr__(self, name):
        return _Null()

    def set_model_info(self, **labels: object) -> None:
        pass

    def observe_batch(self, stream_counts, timings, inference_s) -> None:
        pass

    def render(self) -> bytes:
        return b""


class _Null:
    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self
