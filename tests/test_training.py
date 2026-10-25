"""Training platform tests with the tiny FCOS on CPU (need torch; CI installs it).

The kill test runs training in a subprocess, lets fault injection SIGKILL it
mid-run (no cleanup, no exception handlers, exactly like an OOM kill or a
node loss), then resumes and checks that the result matches an uninterrupted
run with the same seed.
"""

from __future__ import annotations

import json
import random
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from lightning.pytorch.loggers import Logger  # noqa: E402

from crosswalk.config import ModelConfig, TrainingConfig  # noqa: E402
from crosswalk.training import state  # noqa: E402
from crosswalk.training.trainer import ResumeMismatch, train  # noqa: E402
from tests.conftest import subprocess_env  # noqa: E402

TINY = ModelConfig(
    name="fcos_tiny",
    weights="none",
    classes=["car", "pedestrian"],
    min_size=64,
    max_size=96,
    score_threshold=0.05,
)


def config_dict(data: dict[str, str], tmp_path: Path, **overrides) -> dict:
    base = {
        "run_name": "test-run",
        "seed": 11,
        "deterministic": True,
        "device": "cpu",
        "model": TINY.model_dump(),
        "data": {
            "train": {"format": "bdd100k", "labels": data["bdd100k"], "images": data["images"]},
            "eval": {
                "format": "coco",
                "labels": data["coco"],
                "images": data["images"],
                "category_map": {"person": "pedestrian"},
            },
            "num_workers": 0,
        },
        "optimizer": {"name": "sgd", "lr": 0.01},
        "scheduler": {"warmup_steps": 2},
        "checkpoint": {"every_steps": 3, "keep_last": 2, "local_dir": str(tmp_path / "ckpt")},
        "tracking": {"enabled": False},
        "eval": {"baseline": False, "batch_size": 4},
        "epochs": 3,
        "batch_size": 4,
        "log_every_steps": 1,
    }
    for key, value in overrides.items():
        base[key] = value
    return base


def make_config(data, tmp_path, **overrides) -> TrainingConfig:
    return TrainingConfig.model_validate(config_dict(data, tmp_path, **overrides))


def final_checkpoint(tmp_path: Path, run: str = "test-run") -> dict:
    ckpt_dir = tmp_path / "ckpt" / run
    pointer = json.loads((ckpt_dir / "latest.json").read_text())
    return state.load_checkpoint(str(ckpt_dir / pointer["key"]))


# -- model -------------------------------------------------------------------------


def test_tiny_model_trains_and_predicts():
    from crosswalk.models import build_detector

    model = build_detector(TINY, for_training=True)
    images = [torch.rand(3, 64, 96), torch.rand(3, 64, 96)]
    targets = [
        {"boxes": torch.tensor([[5.0, 5.0, 40.0, 30.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
    ]
    losses = model(images, targets)
    assert set(losses) == {"classification", "bbox_regression", "bbox_ctrness"}
    sum(losses.values()).backward()
    model.eval()
    with torch.inference_mode():
        out = model(images)
    assert len(out) == 2 and set(out[0]) == {"boxes", "scores", "labels"}


def test_classification_head_replacement():
    pytest.importorskip("torchvision")
    from torchvision.models.detection import fcos_resnet50_fpn

    from crosswalk.models import _replace_classification_head

    model = fcos_resnet50_fpn(weights=None, weights_backbone=None, num_classes=91)
    reg_before = model.head.regression_head.bbox_reg.weight.clone()
    _replace_classification_head(model, 4)
    head = model.head.classification_head
    assert head.num_classes == 4 and head.cls_logits.out_channels == 4
    assert torch.equal(model.head.regression_head.bbox_reg.weight, reg_before)
    assert torch.allclose(torch.sigmoid(head.cls_logits.bias), torch.tensor(0.01), atol=1e-4)


def test_device_resolution_on_cpu():
    from crosswalk import device

    assert device.resolve("cpu").type == "cpu"
    assert device.resolve("auto").type in ("cpu", "cuda")
    if not torch.cuda.is_available():
        with pytest.raises(device.DeviceUnavailable):
            device.resolve("gpu")
    info = device.describe(device.resolve("cpu"))
    assert info["torch"] == torch.__version__ and "gpu_backend" in info


def test_validation_runs_on_cpu():
    from crosswalk import validate

    result = validate.run(
        device_name="cpu", steps=2, batch_size=1, height=64, width=96, model_name="fcos_tiny"
    )
    assert result["ok"] and result["steps"] == 2 and result["train_images_per_s"] > 0
    assert "torch" in result["environment"]
    assert "OK" in validate.report(result)


# -- checkpoint contents ----------------------------------------------------------


def test_weights_load_from_lightning_and_old_checkpoints():
    from crosswalk.models import build_detector

    model = build_detector(TINY)
    lightning_ckpt = {
        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
        "crosswalk": {"classes": ["car", "pedestrian"], "model_config": TINY.model_dump()},
    }
    loaded = state.weights(lightning_ckpt)
    assert loaded["classes"] == ["car", "pedestrian"]
    fresh = build_detector(TINY)
    fresh.load_state_dict(loaded["model"])  # strict: every key matches a plain detector
    old = {"format": 1, "model": model.state_dict(), "classes": ["car"]}
    assert state.weights(old)["classes"] == ["car"]
    with pytest.raises(ValueError):
        state.weights({"something": "else"})


def test_rng_round_trip():
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    saved = state.capture_rng()
    expected = (random.random(), np.random.rand(), torch.rand(1).item())
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    state.restore_rng(saved)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected


# -- training with Lightning -----------------------------------------------------------


class RecordingLogger(Logger):
    """An in-memory Lightning logger standing in for MLflow."""

    run_id = "run-123"

    def __init__(self):
        super().__init__()
        self.metrics: list[tuple[int, dict]] = []
        self.params: dict = {}
        self.status = None

    @property
    def name(self):
        return "recording"

    @property
    def version(self):
        return "0"

    def log_hyperparams(self, params, *args, **kwargs):
        self.params = dict(params)

    def log_metrics(self, metrics, step=None):
        self.metrics.append((step, dict(metrics)))

    def finalize(self, status):
        self.status = status


def test_training_runs_checkpoints_and_evaluates(synthetic_data, tmp_path):
    config = make_config(synthetic_data, tmp_path)
    summary = train(config)
    assert summary.final_step == summary.total_steps == 12  # 16 images / 4 per batch * 3 epochs
    assert not summary.resumed
    assert "map" in summary.eval and 0.0 <= summary.eval["map"] <= 1.0
    ckpt = tmp_path / "ckpt" / "test-run"
    names = sorted(p.name for p in ckpt.iterdir())
    assert names == [
        "heartbeat.json",
        "latest.json",
        "step-00000009.pt",
        "step-00000012.pt",
        "summary.json",
    ]
    final = final_checkpoint(tmp_path)
    meta = state.metadata(final)
    assert meta["progress"] == {"step": 12, "epoch": 3, "batch_in_epoch": 0}
    assert meta["classes"] == ["car", "pedestrian"]
    assert final["global_step"] == 12  # Lightning's own record agrees


def test_rerunning_a_finished_run_resumes_and_does_nothing(synthetic_data, tmp_path):
    config = make_config(synthetic_data, tmp_path, epochs=1)
    train(config)
    again = train(config)
    assert again.resumed and again.final_step == 4 and again.resume["checkpoint_step"] == 4


def test_resume_policies_and_config_mismatch(synthetic_data, tmp_path):
    with pytest.raises(ResumeMismatch, match="must"):
        train(
            make_config(
                synthetic_data,
                tmp_path,
                checkpoint={"local_dir": str(tmp_path / "ckpt"), "resume": "must"},
            )
        )
    train(make_config(synthetic_data, tmp_path, epochs=1))
    with pytest.raises(ResumeMismatch, match="different training config"):
        train(make_config(synthetic_data, tmp_path, epochs=1, seed=12))
    with pytest.raises(ResumeMismatch, match="resume=never"):
        train(
            make_config(
                synthetic_data,
                tmp_path,
                epochs=1,
                checkpoint={"local_dir": str(tmp_path / "ckpt"), "resume": "never"},
            )
        )


def test_same_seed_gives_identical_weights(synthetic_data, tmp_path):
    train(make_config(synthetic_data, tmp_path / "a", epochs=1))
    train(make_config(synthetic_data, tmp_path / "b", epochs=1))
    a = state.weights(final_checkpoint(tmp_path / "a"))["model"]
    b = state.weights(final_checkpoint(tmp_path / "b"))["model"]
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_checkpoints_go_to_s3_and_resume_from_there(synthetic_data, tmp_path, s3_client):
    config = make_config(
        synthetic_data, tmp_path, max_steps=6,
        checkpoint={"every_steps": 3, "keep_last": 2, "local_dir": str(tmp_path / "node1"), "s3_uri": "s3://checkpoints/test-run"},
    )  # fmt: skip
    summary = train(config, s3_client=s3_client)
    assert summary.checkpoint_storage.startswith("s3://checkpoints/test-run")
    keys = {o["Key"] for o in s3_client.list_objects_v2(Bucket="checkpoints")["Contents"]}
    assert {"test-run/latest.json", "test-run/step-00000006.pt", "test-run/heartbeat.json"} <= keys
    # A pod on a different node: empty scratch, same S3.
    moved = config.model_copy(
        update={
            "checkpoint": config.checkpoint.model_copy(
                update={"local_dir": str(tmp_path / "node2")}
            )
        }
    )
    again = train(moved, s3_client=s3_client)
    assert again.resumed and again.resume["checkpoint_source"] == "remote"


def test_quality_and_system_metrics_are_separate(synthetic_data, tmp_path):
    logger = RecordingLogger()
    train(make_config(synthetic_data, tmp_path, epochs=1), logger=logger)
    names = {k for _, m in logger.metrics for k in m}
    assert {
        "quality/loss",
        "quality/lr",
        "quality/grad_norm",
        "system/images_per_s",
        "system/checkpoint_bytes",
    } <= names
    assert any(n.startswith("quality/eval/") for n in names)
    # Lightning adds the epoch to every row it logs; everything else is ours.
    assert all(n.split("/")[0] in ("quality", "system") for n in names - {"epoch"})
    assert logger.params["run_name"] == "test-run" and logger.status == "success"
    assert state.metadata(final_checkpoint(tmp_path))["mlflow_run_id"] == "run-123"


def test_metric_logging_drops_missing_values():
    from crosswalk.training import tracking

    logger = RecordingLogger()
    tracking.log_metrics([logger], {"a": 1, "b": None, "c": float("nan")}, step=3)
    assert logger.metrics == [(3, {"a": 1.0})]
    tracking.log_metrics([logger], {"b": None}, step=4)
    assert len(logger.metrics) == 1


def test_unreachable_mlflow_means_no_tracking_not_a_crash(monkeypatch):
    import lightning.pytorch.loggers as loggers

    from crosswalk.training import tracking

    class Broken:
        def __init__(self, **kwargs):
            raise ConnectionError("no server")

    monkeypatch.setattr(loggers, "MLFlowLogger", Broken)
    assert tracking.make_logger("exp", "run", None) is None


# -- Experiment B: kill mid-run, resume --------------------------------------------------


def run_cli(config_path: Path, **env) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "crosswalk", "train", "--config", str(config_path)],
        env=subprocess_env(**env),
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.slow
def test_resume_after_kill_matches_uninterrupted_run(synthetic_data, tmp_path):
    # Reference: one uninterrupted run.
    ref_dir = tmp_path / "ref"
    ref_cfg = tmp_path / "ref.yaml"
    ref_cfg.write_text(yaml.safe_dump(config_dict(synthetic_data, ref_dir)))
    ref = run_cli(ref_cfg)
    assert ref.returncode == 0, ref.stderr

    # Same config, SIGKILLed right after step 7 (checkpoints at 3 and 6).
    run_dir = tmp_path / "killed"
    cfg = tmp_path / "killed.yaml"
    cfg.write_text(yaml.safe_dump(config_dict(synthetic_data, run_dir)))
    killed = run_cli(cfg, CROSSWALK_FAULT_AT_STEP="7")
    assert killed.returncode == -signal.SIGKILL, killed.stderr
    ckpt_dir = run_dir / "ckpt" / "test-run"
    assert json.loads((ckpt_dir / "latest.json").read_text())["step"] == 6
    assert json.loads((ckpt_dir / "heartbeat.json").read_text())["step"] == 7

    # Resume: the fault variable is still set, but a resumed run ignores it.
    resumed = run_cli(cfg, CROSSWALK_FAULT_AT_STEP="7")
    assert resumed.returncode == 0, resumed.stderr
    summary = json.loads((ckpt_dir / "summary.json").read_text())
    assert summary["resumed"] is True
    assert summary["resume"]["checkpoint_step"] == 6
    assert summary["resume"]["lost_steps"] == 1
    assert summary["resume"]["restore_s"] >= 0 and summary["resume"]["recovery_s"] > 0
    assert summary["final_step"] == 12

    ref_ckpt, run_ckpt = final_checkpoint(ref_dir), final_checkpoint(run_dir)
    assert state.metadata(run_ckpt)["progress"] == state.metadata(ref_ckpt)["progress"]
    ref_w, run_w = state.weights(ref_ckpt)["model"], state.weights(run_ckpt)["model"]
    for key in ref_w:
        assert torch.allclose(ref_w[key], run_w[key], atol=1e-6), key
