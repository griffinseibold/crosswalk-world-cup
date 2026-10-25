"""Promotion gate and rollback against a real (local file store) MLflow registry.

MLflow's file store is in maintenance mode in 3.x and refuses to start unless
``MLFLOW_ALLOW_FILE_STORE=true``; it still supports the model registry and
aliases, which is all these tests need, and it needs no server or database.
"""

from __future__ import annotations

import json
import math

import pytest

from crosswalk import promotion, registry
from crosswalk.promotion import Candidate, GateRules, decide

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def registry_uri(tmp_path, monkeypatch):
    import mlflow.tracking.fluent as fluent

    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    # mlflow.set_experiment() remembers the experiment in a module global and
    # in MLFLOW_EXPERIMENT_ID; each test has a fresh store, so forget both.
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.setattr(fluent, "_active_experiment_id", None)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    yield uri
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    mlflow.set_tracking_uri(None)


def make_versions(n: int, name: str = "crosswalk-detector") -> list[str]:
    c = registry.client()
    c.create_registered_model(name)
    versions = []
    for i in range(n):
        with mlflow.start_run() as run:
            mlflow.log_text(json.dumps({"i": i}), "model/model.json")
        mv = c.create_model_version(name, f"runs:/{run.info.run_id}/model", run_id=run.info.run_id)
        versions.append(str(mv.version))
    return versions


# -- the rules ------------------------------------------------------------------------


def cand(label, map_, ped=None, p95=None) -> Candidate:
    metrics = {"map": map_}
    if ped is not None:
        metrics["ap/pedestrian"] = ped
    return Candidate(label, None, None, metrics, {"p50": p95, "p95": p95})


@pytest.mark.parametrize(
    ("c", "b", "rules", "approved", "fragment"),
    [
        (cand("candidate", 0.42, 0.40, 50), cand("champion", 0.40, 0.40, 50), GateRules(), True, "mAP 0.4200 vs 0.4000"),
        (cand("candidate", 0.39, 0.40, 50), cand("champion", 0.40, 0.40, 50), GateRules(), False, "below the champion"),
        (cand("candidate", 0.39, 0.40, 50), cand("champion", 0.40, 0.40, 50), GateRules(max_map_drop=0.02), True, ""),
        (cand("candidate", 0.405, None, 50), cand("champion", 0.40, None, 50), GateRules(min_map_gain=0.01), False, "below the required"),
        (cand("candidate", 0.45, 0.30, 50), cand("champion", 0.40, 0.40, 50), GateRules(), False, "pedestrian AP fell"),
        (cand("candidate", 0.45, 0.40, 70), cand("champion", 0.40, 0.40, 50), GateRules(), False, "1.40x"),
        (cand("candidate", 0.45, 0.40, 59), cand("champion", 0.40, 0.40, 50), GateRules(), True, "1.18x"),
        (cand("candidate", 0.45, math.nan, 50), cand("champion", 0.40, 0.40, 50), GateRules(), True, ""),
    ],
)  # fmt: skip
def test_decision_rules(c, b, rules, approved, fragment):
    ok, reasons = decide(c, b, rules)
    assert ok is approved, reasons
    assert fragment in " | ".join(reasons)


# -- registry aliases and history ---------------------------------------------------------


def test_aliases_promotion_history_and_rollbacks(registry_uri):
    c = registry.client()
    v1, v2, v3 = make_versions(3)
    assert registry.make_champion(c, "crosswalk-detector", v1) is None
    assert registry.make_champion(c, "crosswalk-detector", v2) == v1
    assert registry.alias_version(c, "crosswalk-detector", registry.PREVIOUS) == v1
    registry.make_champion(c, "crosswalk-detector", v3)
    assert registry.resolve("models:/crosswalk-detector@champion").version == v3

    # Rollback walks the history: v3 -> v2 -> v1, then refuses.
    assert registry.rollback(c, "crosswalk-detector") == (v3, v2)
    assert registry.alias_version(c, "crosswalk-detector", registry.PREVIOUS) == v1
    assert registry.rollback(c, "crosswalk-detector") == (v2, v1)
    assert registry.alias_version(c, "crosswalk-detector", registry.PREVIOUS) is None
    with pytest.raises(RuntimeError, match="no earlier champion"):
        registry.rollback(c, "crosswalk-detector")
    # An explicit target works from anywhere.
    assert registry.rollback(c, "crosswalk-detector", to_version=v3) == (v1, v3)
    assert registry.resolve("models:/crosswalk-detector@champion").version == v3


def test_models_uri_parsing():
    assert registry.parse_models_uri("models:/crosswalk-detector@champion") == (
        "crosswalk-detector",
        "champion",
        None,
    )
    assert registry.parse_models_uri("models:/crosswalk-detector/7") == (
        "crosswalk-detector",
        None,
        "7",
    )
    with pytest.raises(ValueError):
        registry.parse_models_uri("s3://checkpoints/x.pt")
    assert registry.is_registry_uri("models:/x@champion") and not registry.is_registry_uri(
        "/tmp/x.pt"
    )


# -- the gate, end to end with an injected evaluator ------------------------------------------


def eval_config(synthetic_data, tmp_path):
    from crosswalk.config import TrainingConfig

    return TrainingConfig.model_validate(
        {
            "run_name": "promote-test",
            "device": "cpu",
            "model": {"name": "fcos_tiny", "weights": "none", "classes": ["car", "pedestrian"], "min_size": 64, "max_size": 96},
            "data": {
                "train": {"format": "bdd100k", "labels": synthetic_data["bdd100k"], "images": synthetic_data["images"]},
                "eval": {"format": "coco", "labels": synthetic_data["coco"], "images": synthetic_data["images"],
                         "category_map": {"person": "pedestrian"}},
                "hash_images": False,
            },
            "tracking": {"enabled": False},
        }
    )  # fmt: skip


def scripted(scores: dict[str | None, tuple[float, float]]):
    """Evaluator returning fixed (mAP, p95 ms) per URI."""

    def evaluator(uri, *args, **kwargs):
        m, p95 = scores[uri]
        return {"map": m, "ap/pedestrian": m}, {"p50": p95, "p95": p95}

    return evaluator


def test_promote_first_version_against_the_baseline_then_reject_a_worse_one(
    registry_uri, synthetic_data, tmp_path
):
    v1, v2 = make_versions(2)
    config = eval_config(synthetic_data, tmp_path)
    name = "crosswalk-detector"
    scores = {
        None: (0.30, 40.0),
        f"models:/{name}/{v1}": (0.35, 42.0),
        f"models:/{name}/{v2}": (0.33, 41.0),
    }

    d1 = promotion.promote(config, name, v1, evaluator=scripted(scores))
    assert d1.approved and d1.champion.label == "baseline"
    c = registry.client()
    assert registry.alias_version(c, name, registry.CHAMPION) == v1
    assert c.get_model_version(name, v1).tags["promotion.decision"] == "approved"

    d2 = promotion.promote(config, name, None, evaluator=scripted(scores))  # newest = v2
    assert not d2.approved and d2.candidate.version == v2
    assert registry.alias_version(c, name, registry.CHAMPION) == v1  # unchanged
    assert registry.alias_version(c, name, registry.CHALLENGER) == v2
    tags = c.get_model_version(name, v2).tags
    assert tags["promotion.decision"] == "rejected"
    # The full report is an MLflow artifact of the recorded run.
    report = mlflow.artifacts.load_dict(f"runs:/{tags['promotion.report_run']}/promotion.json")
    assert report["decision"] == "rejected" and report["candidate"]["metrics"]["map"] == 0.33
    assert report["eval_dataset_sha256"] == tags["promotion.eval_dataset_sha256"]


def test_dry_run_records_but_does_not_promote_and_rollback_is_recorded(
    registry_uri, synthetic_data, tmp_path
):
    v1, v2 = make_versions(2)
    config = eval_config(synthetic_data, tmp_path)
    name = "crosswalk-detector"
    scores = {
        None: (0.1, 40.0),
        f"models:/{name}/{v1}": (0.3, 40.0),
        f"models:/{name}/{v2}": (0.4, 40.0),
    }
    promotion.promote(config, name, v1, evaluator=scripted(scores))
    dry = promotion.promote(config, name, v2, evaluator=scripted(scores), dry_run=True)
    c = registry.client()
    assert dry.approved and registry.alias_version(c, name, registry.CHAMPION) == v1
    promotion.promote(config, name, v2, evaluator=scripted(scores))
    assert registry.alias_version(c, name, registry.CHAMPION) == v2
    assert promotion.rollback(name, reason="pedestrian misses on night footage") == (v2, v1)
    assert registry.alias_version(c, name, registry.CHAMPION) == v1
    assert "rolled back" in c.get_model_version(name, v2).tags["promotion.decision"]
    with pytest.raises(SystemExit, match="already the champion"):
        promotion.promote(config, name, v1, evaluator=scripted(scores))


# -- with real weights (torch) ----------------------------------------------------------------------


def test_registered_weights_round_trip_and_load_into_the_predictor(registry_uri, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from crosswalk.config import ModelConfig
    from crosswalk.inference.predictor import TorchPredictor
    from crosswalk.models import build_detector

    model_config = ModelConfig(
        name="fcos_tiny", weights="none", classes=["car", "pedestrian"], min_size=48, max_size=80
    )
    model = build_detector(model_config)
    name = "crosswalk-detector"
    with mlflow.start_run() as run:
        version = registry.register_run_model(
            run.info.run_id,
            name,
            model,
            ["car", "pedestrian"],
            model_config.model_dump(mode="json"),
            {"step": 1},
        )
    registry.make_champion(registry.client(), name, version)
    saved = registry.load_weights(f"models:/{name}@champion")
    assert saved["classes"] == ["car", "pedestrian"]
    for key, value in model.state_dict().items():
        assert torch.equal(saved["model"][key], value)
    predictor = TorchPredictor(
        model_config.model_copy(update={"checkpoint": f"models:/{name}@champion"}),
        device_name="cpu",
    )
    assert predictor.model_version == f"models:/{name}/{version} (@champion)"
    import numpy as np

    assert len(predictor.predict([np.zeros((48, 80, 3), dtype=np.uint8)])) == 1


def test_promote_with_real_evaluation_on_cpu(registry_uri, synthetic_data, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pycocotools")
    from crosswalk.models import build_detector

    config = eval_config(synthetic_data, tmp_path)
    name = "crosswalk-detector"
    for seed in (0, 1):
        import torch

        torch.manual_seed(seed)
        model = build_detector(config.model)
        with mlflow.start_run() as run:
            registry.register_run_model(
                run.info.run_id,
                name,
                model,
                config.model.classes,
                config.model.model_dump(mode="json"),
                {},
            )
    # Champion: v1 (forced, as if promoted earlier).
    registry.make_champion(registry.client(), name, "1")
    decision = promotion.promote(
        config,
        name,
        "2",
        GateRules(max_map_drop=1.0, max_latency_ratio=100.0),
        batch_size=2,
        latency_batches=2,
    )
    # Untrained tiny models score ~0 mAP; the point is that both were really
    # evaluated and timed through the same code path.
    assert decision.champion.label == "champion"
    for side in (decision.candidate, decision.champion):
        assert "map" in side.metrics and side.latency_ms["p95"] > 0
    assert decision.approved  # within the deliberately loose rules
