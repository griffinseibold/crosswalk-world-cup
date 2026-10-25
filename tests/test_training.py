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

from crosswalk.config import ModelConfig, TrainingConfig  # noqa: E402
from crosswalk.training import state  # noqa: E402
from crosswalk.training.loop import ResumeMismatch, train  # noqa: E402
from crosswalk.training.tracking import NullTracker  # noqa: E402
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


def final_weights(tmp_path: Path, run: str = "test-run") -> dict:
    ckpt_dir = tmp_path / "ckpt" / run
    pointer = json.loads((ckpt_dir / "latest.json").read_text())
    return state.load_state(str(ckpt_dir / pointer["key"]))


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


# -- checkpoint state ----------------------------------------------------------------


def test_state_round_trip_restores_everything(tmp_path):
    from crosswalk.models import build_detector
    from crosswalk.training.schedule import WarmupCosine

    def setup():
        model = build_detector(TINY, for_training=True)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, WarmupCosine(50, 5, 0.1))
        return model, opt, sched

    model, opt, sched = setup()
    images = [torch.rand(3, 64, 96)]
    targets = [{"boxes": torch.tensor([[5.0, 5.0, 40.0, 30.0]]), "labels": torch.tensor([1])}]
    for _ in range(3):
        opt.zero_grad()
        sum(model(images, targets).values()).backward()
        opt.step()
        sched.step()
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    progress = state.Progress(step=3, epoch=1, batch_in_epoch=2)
    saved = state.build_state(
        model=model, optimizer=opt, scheduler=sched, scaler=None, progress=progress,
        classes=["car", "pedestrian"], config={"a": 1}, config_hash="h",
    )  # fmt: skip
    path = tmp_path / "s.pt"
    state.save_state(saved, str(path))
    expected_draws = (random.random(), np.random.rand(), torch.rand(1).item())

    model2, opt2, sched2 = setup()
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored = state.restore(
        state.load_state(str(path)), model=model2, optimizer=opt2, scheduler=sched2, scaler=None
    )
    assert restored == progress
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expected_draws
    for p1, p2 in zip(model.state_dict().values(), model2.state_dict().values(), strict=True):
        assert torch.equal(p1, p2)
    momentum = [s["momentum_buffer"] for s in opt.state_dict()["state"].values()]
    momentum2 = [s["momentum_buffer"] for s in opt2.state_dict()["state"].values()]
    assert all(torch.equal(a, b) for a, b in zip(momentum, momentum2, strict=True))
    assert sched2.last_epoch == 3 and sched2.get_last_lr() == sched.get_last_lr()


def test_unknown_checkpoint_format_is_refused():
    with pytest.raises(ValueError, match="format"):
        state.restore({"format": 999}, model=None, optimizer=None, scheduler=None, scaler=None)


# -- the loop ----------------------------------------------------------------------------


def test_training_runs_checkpoints_and_evaluates(synthetic_data, tmp_path):
    config = make_config(synthetic_data, tmp_path)
    summary = train(config, tracker=NullTracker())
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
    final = final_weights(tmp_path)
    assert final["progress"] == {"step": 12, "epoch": 3, "batch_in_epoch": 0}
    assert final["classes"] == ["car", "pedestrian"]


def test_rerunning_a_finished_run_resumes_and_does_nothing(synthetic_data, tmp_path):
    config = make_config(synthetic_data, tmp_path, epochs=1)
    train(config, tracker=NullTracker())
    again = train(config, tracker=NullTracker())
    assert again.resumed and again.final_step == 4 and again.resume["checkpoint_step"] == 4


def test_resume_policies_and_config_mismatch(synthetic_data, tmp_path):
    with pytest.raises(ResumeMismatch, match="must"):
        train(
            make_config(
                synthetic_data,
                tmp_path,
                checkpoint={"local_dir": str(tmp_path / "ckpt"), "resume": "must"},
            ),
            tracker=NullTracker(),
        )
    train(make_config(synthetic_data, tmp_path, epochs=1), tracker=NullTracker())
    with pytest.raises(ResumeMismatch, match="different training config"):
        train(make_config(synthetic_data, tmp_path, epochs=1, seed=12), tracker=NullTracker())
    with pytest.raises(ResumeMismatch, match="resume=never"):
        train(
            make_config(
                synthetic_data,
                tmp_path,
                epochs=1,
                checkpoint={"local_dir": str(tmp_path / "ckpt"), "resume": "never"},
            ),
            tracker=NullTracker(),
        )


def test_same_seed_gives_identical_weights(synthetic_data, tmp_path):
    train(make_config(synthetic_data, tmp_path / "a", epochs=1), tracker=NullTracker())
    train(make_config(synthetic_data, tmp_path / "b", epochs=1), tracker=NullTracker())
    a, b = final_weights(tmp_path / "a"), final_weights(tmp_path / "b")
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k


def test_checkpoints_go_to_s3_and_resume_from_there(synthetic_data, tmp_path, s3_client):
    config = make_config(
        synthetic_data, tmp_path, max_steps=6,
        checkpoint={"every_steps": 3, "keep_last": 2, "local_dir": str(tmp_path / "node1"), "s3_uri": "s3://checkpoints/test-run"},
    )  # fmt: skip
    summary = train(config, s3_client=s3_client, tracker=NullTracker())
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
    again = train(moved, s3_client=s3_client, tracker=NullTracker())
    assert again.resumed and again.resume["checkpoint_source"] == "remote"


class RecordingTracker(NullTracker):
    run_id = "run-123"

    def __init__(self):
        self.metrics: list[tuple[int, dict]] = []
        self.params = {}
        self.ended = None

    def log_params(self, params):
        self.params = params

    def log_metrics(self, metrics, step):
        self.metrics.append((step, metrics))

    def end(self, status="FINISHED"):
        self.ended = status


def test_quality_and_system_metrics_are_separate(synthetic_data, tmp_path):
    tracker = RecordingTracker()
    train(make_config(synthetic_data, tmp_path, epochs=1), tracker=tracker)
    names = {k for _, m in tracker.metrics for k in m}
    assert {"quality/loss", "quality/lr", "system/images_per_s", "system/checkpoint_bytes"} <= names
    assert any(n.startswith("quality/eval/") for n in names)
    assert all(n.split("/")[0] in ("quality", "system") for n in names)
    assert tracker.params["run_name"] == "test-run" and tracker.ended == "FINISHED"
    assert final_weights(tmp_path)["extra"]["mlflow_run_id"] == "run-123"


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

    ref_state, run_state = final_weights(ref_dir), final_weights(run_dir)
    assert run_state["progress"] == ref_state["progress"]
    for key in ref_state["model"]:
        assert torch.allclose(ref_state["model"][key], run_state["model"][key], atol=1e-6), key


def test_mlflow_tracker_calls_with_a_fake_client(monkeypatch):
    """MlflowTracker against a stand-in module (CI's mlflow-skinny has no SQL store)."""
    import types

    calls: list[tuple] = []

    class Run:
        info = types.SimpleNamespace(run_id="abc")

    fake = types.ModuleType("mlflow")
    fake.set_experiment = lambda name: calls.append(("experiment", name))
    fake.start_run = lambda **kw: calls.append(("start", kw)) or Run()
    fake.log_params = lambda p: calls.append(("params", dict(p)))
    fake.log_metrics = lambda m, step: calls.append(("metrics", m, step))
    fake.set_tags = lambda t: calls.append(("tags", t))
    fake.end_run = lambda status: calls.append(("end", status))
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    from crosswalk.training.tracking import MlflowTracker

    tracker = MlflowTracker("exp", "run-a", run_id="abc")
    assert ("start", {"run_id": "abc"}) in calls  # resume reattaches to the same run
    tracker.log_params({f"k{i}": i for i in range(150)} | {"nested": {"x": None}})
    batches = [c[1] for c in calls if c[0] == "params"]
    assert [len(b) for b in batches] == [100, 51] and batches[1]["nested.x"] == ""
    tracker.log_metrics({"quality/loss": 1, "system/x": None}, step=5)
    assert ("metrics", {"quality/loss": 1.0}, 5) in calls
    tracker.end("FAILED")
    assert calls[-1] == ("end", "FAILED")
