import json
import re

import pytest

from crosswalk.inference import metrics as m
from tests.conftest import repo_root

DASHBOARD = repo_root() / "chart/crosswalk/dashboards/inference.json"


def exported_metric_names() -> set[str]:
    """Every metric name any component exports, with histogram suffixes.

    Python: the inference service, every stage and the ranking service.
    Rust: METRIC_NAMES in ingest/src/metrics.rs, read from the source.
    """
    from crosswalk.ranking.service import RANKING_METRIC_NAMES
    from crosswalk.stream.stage import STAGE_METRIC_NAMES

    rust = (repo_root() / "ingest/src/metrics.rs").read_text()
    block = rust.split("pub const METRIC_NAMES")[1].split("];")[0]
    names = (
        set(m.METRIC_NAMES)
        | STAGE_METRIC_NAMES
        | RANKING_METRIC_NAMES
        | set(re.findall(r'"(crosswalk_[a-z_]+)"', block))
    )
    known = set()
    for name in names:
        known |= {name, f"{name}_bucket", f"{name}_sum", f"{name}_count"}
    return known


def test_dashboard_only_queries_metrics_the_services_export():
    dashboard = json.loads(DASHBOARD.read_text())
    exprs = [t["expr"] for p in dashboard["panels"] for t in p.get("targets", [])]
    assert exprs, "dashboard has no queries"
    used = {name for e in exprs for name in re.findall(r"\bcrosswalk_[a-z_]+", e)}
    known = exported_metric_names()
    assert used and used <= known, f"unknown metrics in dashboard: {used - known}"


def test_dashboard_has_the_panels_the_spec_asks_for():
    titles = " ".join(p["title"].lower() for p in json.loads(DASHBOARD.read_text())["panels"])
    for word in ("frames", "latency", "queue", "batch", "errors", "dropped", "model",
                 "lag", "end-to-end", "dead letters", "backpressure"):  # fmt: skip
        assert word in titles, word


def test_metrics_render_and_record():
    pytest.importorskip("prometheus_client")
    from crosswalk.inference.batcher import Timing

    metrics = m.Metrics()
    metrics.set_model_info(model="fcos", weights="COCO_V1", hip="7.2")
    timings = [Timing(0.004, 0.030, 0.034, 2), Timing(0.001, 0.030, 0.031, 2)]
    metrics.observe_batch({"cam": 2}, timings, 0.030)
    metrics.dropped.labels(stream="cam", reason="queue_full").inc()
    metrics.errors.labels(stage="decode").inc()
    metrics.detections.labels(label="car").inc()
    metrics.ready.set(1)
    from prometheus_client.parser import text_string_to_metric_families

    text = metrics.render().decode()
    samples = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            key = (sample.name, tuple(sorted(sample.labels.items())))
            samples[key] = sample.value

    def value(name, **labels):
        return samples[(name, tuple(sorted(labels.items())))]

    assert value("crosswalk_frames_total", stream="cam") == 2
    assert value("crosswalk_inference_latency_seconds_count") == 1
    assert value("crosswalk_queue_wait_seconds_count") == 2
    assert value("crosswalk_batch_size_bucket", le="2.0") == 1
    assert value("crosswalk_batch_size_bucket", le="1.0") == 0
    assert value("crosswalk_frames_dropped_total", stream="cam", reason="queue_full") == 1
    assert value("crosswalk_errors_total", stage="decode") == 1
    assert value("crosswalk_model_info", model="fcos", weights="COCO_V1", hip="7.2") == 1
    assert value("crosswalk_ready") == 1
    exported = {name for name, _ in samples}
    for name in m.METRIC_NAMES:
        assert any(e == name or e.startswith(f"{name}_") for e in exported), name


def test_metrics_instances_are_independent():
    pytest.importorskip("prometheus_client")
    a, b = m.Metrics(), m.Metrics()
    a.errors.labels(stage="x").inc()
    assert 'stage="x"' in a.render().decode()
    assert 'stage="x"' not in b.render().decode()
