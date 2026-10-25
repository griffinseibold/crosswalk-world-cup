"""scripts/submit-training.py builds the right helm/kubectl command lines."""

import datetime as dt
import importlib.util

import pytest

from tests.conftest import repo_root


@pytest.fixture(scope="module")
def submit():
    path = repo_root() / "scripts" / "submit-training.py"
    spec = importlib.util.spec_from_file_location("submit_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dns_label(submit):
    assert submit.dns_label("Crosswalk Train: BDD_v1!") == "crosswalk-train-bdd-v1"
    assert len(submit.dns_label("x" * 100)) == 52


def test_sync_is_derived_from_dataset_paths(submit):
    config = submit.read_config(repo_root() / "configs/train-bdd100k.yaml")
    assert submit.derive_sync(config) == [
        "s3://datasets/bdd100k=/scratch/data/bdd100k",
        "s3://datasets/fixed-camera-eval=/scratch/data/fixed-camera-eval",
    ]
    assert submit.derive_sync({"data": {"train": {"labels": "/elsewhere/x.json"}}}) == []


def test_job_names(submit):
    now = dt.datetime(2026, 10, 25, 9, 30, 0)
    assert (
        submit.job_name("train", {"run_name": "bdd100k-fcos-v1"}, None, now)
        == "crosswalk-train-bdd100k-fcos-v1"
    )
    assert submit.job_name("validate", {}, None, now) == "crosswalk-validate-20261025-093000"
    assert submit.job_name("train", {"run_name": "x"}, "Custom", now) == "custom"


def test_helm_command(submit, tmp_path):
    args = submit.build_parser().parse_args(
        [
            "train",
            "--config",
            "c.yaml",
            "--set",
            "epochs=1",
            "--image-tag",
            "rocm-abc",
            "--synthetic",
        ]
    )
    cmd = submit.helm_command(
        args, "crosswalk-train-x", tmp_path / "c.yaml", ["s3://datasets/a=/scratch/data/a"]
    )
    joined = " ".join(cmd)
    assert cmd[:4] == ["helm", "template", "crosswalk", str(submit.CHART)]
    assert "--show-only templates/job.yaml" in joined
    assert "gpu.enabled=true" in joined and "image.gpu.tag=rocm-abc" in joined
    assert f"job.config={tmp_path / 'c.yaml'}" in joined
    assert "job.sync[0]=s3://datasets/a=/scratch/data/a" in joined
    assert "job.args[0]=--set" in joined and "job.args[1]=epochs=1" in joined
    assert "job.synthetic=true" in joined

    env = submit.build_parser().parse_args(["train", "--env", "CROSSWALK_FAULT_AT_STEP=500"])
    joined = " ".join(submit.helm_command(env, "x", None, []))
    assert "job.env[0].name=CROSSWALK_FAULT_AT_STEP" in joined
    assert "job.env[0].value=500" in joined

    cpu = submit.build_parser().parse_args(["validate", "--cpu", "--image-tag", "0.1.0"])
    joined = " ".join(submit.helm_command(cpu, "v", None, []))
    assert "gpu.enabled=false" in joined and "image.cpu.tag=0.1.0" in joined
    assert "job.config" not in joined


def test_kubectl_context(submit):
    args = submit.build_parser().parse_args(["validate", "--context", ""])
    assert submit.kubectl(args, "get", "pods") == [
        "kubectl",
        "--namespace",
        "crosswalk",
        "get",
        "pods",
    ]
    args = submit.build_parser().parse_args(["validate"])
    assert submit.kubectl(args, "get")[:3] == ["kubectl", "--context", "kind-homelab-dev"]


def test_no_kueue_flag(submit):
    args = submit.build_parser().parse_args(["validate", "--no-kueue"])
    assert "kueue.enabled=false" in " ".join(submit.helm_command(args, "v", None, []))
    args = submit.build_parser().parse_args(["validate"])
    assert "kueue.enabled" not in " ".join(submit.helm_command(args, "v", None, []))
