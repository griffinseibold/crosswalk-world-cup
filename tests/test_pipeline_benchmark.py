"""Pipeline, status server, detection output and Experiment A schema, with a fake predictor."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from crosswalk.benchmark import harness
from crosswalk.config import BatchingConfig, BenchmarkConfig, StrategySpec
from crosswalk.inference.batcher import Batcher
from crosswalk.inference.metrics import NullMetrics
from crosswalk.inference.pipeline import (
    DetectionWriter,
    FrameMeta,
    InferenceWorker,
    detection_record,
    feed,
)
from crosswalk.inference.predictor import FakePredictor
from crosswalk.inference.server import StatusServer


class FakeFrame:
    def __init__(self, i):
        self.video = "/footage/clip.mp4"
        self.index = i
        self.pts_s = i / 30
        self.image = np.zeros((48, 64, 3), dtype=np.uint8)
        self.width, self.height = 64, 48


def meta_for(stream):
    def make(f):
        return FrameMeta(f.video, stream, f.index, f.pts_s, f.width, f.height, decoded_at=0.0)

    return make


def run_pipeline(tmp_path, strategy="dynamic", frames=20, realtime=False, predictor=None, queue=64):
    batching = BatchingConfig(strategy=strategy, max_batch_size=4, max_wait_ms=5, queue_size=queue)
    batcher = Batcher(batching)
    predictor = predictor or FakePredictor()
    writer = DetectionWriter(str(tmp_path / "out"))

    def on_result(meta, detections, timing):
        writer.write("clip", detection_record(meta, detections, timing, predictor.model_version))

    worker = InferenceWorker(batcher, predictor, NullMetrics(), on_result=on_result)
    worker.start()
    submitted, dropped = feed(
        batcher,
        (FakeFrame(i) for i in range(frames)),
        stream="clip",
        realtime=realtime,
        fps=1000.0,
        metrics=NullMetrics(),
        meta=meta_for("clip"),
    )
    batcher.close()
    worker.join(5)
    path = writer.close("clip")
    return submitted, dropped, worker, predictor, path


@pytest.mark.parametrize("strategy", ["sequential", "fixed", "dynamic"])
def test_every_frame_is_detected_once_in_order(tmp_path, strategy):
    submitted, dropped, worker, predictor, path = run_pipeline(tmp_path, strategy)
    assert (submitted, dropped) == (20, 0)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["frame"] for r in records] == list(range(20))
    assert worker.stats.frames == 20
    if strategy == "sequential":
        assert set(predictor.batch_sizes) == {1}
    if strategy == "fixed":
        assert predictor.batch_sizes == [4] * 5
    assert max(predictor.batch_sizes) <= 4


def test_detection_record_schema(tmp_path):
    *_, path = run_pipeline(tmp_path, frames=3)
    record = json.loads(path.read_text().splitlines()[0])
    assert set(record) == {
        "schema", "video", "stream", "frame", "pts_s", "width", "height", "decoded_at",
        "model_version", "batch_size", "latency_ms", "detections",
    }  # fmt: skip
    assert record["video"] == "clip.mp4" and record["model_version"] == "fake:0"
    assert set(record["latency_ms"]) == {"queue_wait", "inference", "end_to_end"}
    det = record["detections"][0]
    assert det["label"] == "car" and len(det["box"]) == 4


def test_writer_publishes_only_complete_files(tmp_path):
    writer = DetectionWriter(str(tmp_path))
    writer.write("a", {"x": 1})
    assert not (tmp_path / "a.jsonl").exists()
    assert (tmp_path / "a.jsonl.partial").exists()
    writer.close("a")
    assert (tmp_path / "a.jsonl").read_text() == '{"x":1}\n'
    assert not (tmp_path / "a.jsonl.partial").exists()


def test_writer_uploads_finished_files(tmp_path):
    from crosswalk.storage import LocalStore

    store = LocalStore(str(tmp_path / "bucket"))
    writer = DetectionWriter(str(tmp_path / "out"), store=store, prefix="detections")
    writer.write("a", {"x": 1})
    writer.close("a")
    assert store.list() == ["detections/a.jsonl"]


def test_inference_errors_are_counted_and_do_not_stop_the_worker(tmp_path):
    predictor = FakePredictor(fail_on=2)
    submitted, _, worker, _, path = run_pipeline(tmp_path, "fixed", predictor=predictor)
    assert worker.stats.errors == 1
    assert len(path.read_text().splitlines()) == submitted - 4  # one batch of 4 lost


def test_realtime_mode_drops_when_the_gpu_falls_behind(tmp_path):
    slow = FakePredictor(base_s=0.05, per_image_s=0.0)
    submitted, dropped, worker, *_ = run_pipeline(
        tmp_path, "fixed", frames=60, realtime=True, predictor=slow, queue=4
    )
    assert dropped > 0
    assert submitted + dropped == 60
    assert worker.stats.frames == submitted


# -- status server -----------------------------------------------------------------


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def test_status_server_endpoints():
    server = StatusServer(0, NullMetrics(), lambda: {"phase": "running"}, host="127.0.0.1")
    server.start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        assert get(f"{base}/healthz")[0] == 200
        assert get(f"{base}/readyz")[0] == 503
        server.ready.set()
        assert get(f"{base}/readyz")[0] == 200
        code, body = get(f"{base}/status")
        assert code == 200 and json.loads(body) == {"phase": "running"}
        assert get(f"{base}/metrics")[0] == 200
        assert get(f"{base}/nope")[0] == 404
    finally:
        server.stop()


# -- Experiment A harness ---------------------------------------------------------------


def bench_config(tmp_path, **overrides):
    data = {
        "device": "cpu",
        "resolutions": [[64, 48], [32, 24]],
        "strategies": [
            StrategySpec(name="seq", batching=BatchingConfig(strategy="sequential")).model_dump(),
            StrategySpec(
                name="dyn",
                batching=BatchingConfig(strategy="dynamic", max_batch_size=4, max_wait_ms=2),
            ).model_dump(),
        ],
        "concurrency": [1, 3],
        "frames_per_run": 24,
        "warmup_batches": 1,
        "repeats": 2,
        "output_dir": str(tmp_path),
        "source": {"video": None, "frames": 8},
    }
    data.update(overrides)
    return BenchmarkConfig.model_validate(data)


def test_benchmark_result_schema_and_contents(tmp_path):
    config = bench_config(tmp_path)
    result = harness.run_benchmark(config, predictor=FakePredictor(), runtime={"torch": "fake"})
    assert harness.validate_result(result) == []
    assert len(result["runs"]) == 2 * 2 * 2 * 2  # resolutions x strategies x concurrency x repeats
    for run in result["runs"]:
        assert run["frames_completed"] == run["frames_submitted"] == 24
        assert run["frames_dropped"] == 0 and run["throughput_fps"] > 0
        assert run["latency_ms"]["p50"] <= run["latency_ms"]["p95"] <= run["latency_ms"]["p99"]
        assert sum(run["batch_size"]["histogram"].values()) > 0
    seq = [r for r in result["runs"] if r["strategy"] == "seq"]
    assert all(set(r["batch_size"]["histogram"]) == {"1"} for r in seq)
    assert len(result["summary"]) == 8
    env = result["environment"]
    assert env["runtime"] == {"torch": "fake"} and env["model"]["source"] == "synthetic"
    assert "cpu_count" in env["hardware"]
    assert result["config_hash"] and result["method"]

    path = harness.write_result(result, str(tmp_path))
    assert harness.validate_result(json.loads(path.read_text())) == []
    table = harness.markdown_table(result)
    assert table.count("\n") == 2 + len(result["summary"]) - 1


def test_benchmark_realtime_mode_records_drops(tmp_path):
    config = bench_config(
        tmp_path,
        mode="realtime",
        resolutions=[[64, 48]],
        strategies=[
            StrategySpec(
                name="fixed",
                batching=BatchingConfig(strategy="fixed", max_batch_size=2, queue_size=2),
            ).model_dump()
        ],
        concurrency=[4],
        repeats=1,
        frames_per_run=80,
        source={"video": None, "frames": 4, "fps": 500},
    )
    result = harness.run_benchmark(config, predictor=FakePredictor(base_s=0.02))
    run = result["runs"][0]
    assert run["frames_dropped"] > 0
    assert run["frames_submitted"] + run["frames_dropped"] == 80


def test_validate_result_reports_problems():
    problems = harness.validate_result({"runs": [{"latency_ms": {}}], "environment": {}})
    assert any("missing key 'config'" in p for p in problems)
    assert any("run 0 missing 'throughput_fps'" in p for p in problems)
    assert any("latency_ms missing 'p99'" in p for p in problems)


def test_worker_thread_is_daemon():
    worker = InferenceWorker(Batcher(BatchingConfig()), FakePredictor(), NullMetrics())
    thread = worker.start()
    assert isinstance(thread, threading.Thread) and thread.daemon
    worker.batcher.close()
    worker.join(2)
