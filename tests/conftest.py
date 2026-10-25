"""Shared fixtures.

Tests that need an optional heavy dependency (torch, av, prometheus_client,
moto, pycocotools) call ``pytest.importorskip`` so the rest of the suite still
runs on a machine without them. CI installs everything (requirements/cpu.txt
plus dev.txt), so nothing is skipped there; a skip in CI is a failure to
investigate. ``pytest -ra`` (set in pyproject.toml) lists every skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from crosswalk import synthetic


@pytest.fixture
def synthetic_data(tmp_path: Path) -> dict[str, str]:
    return synthetic.make_dataset(tmp_path / "data", images=16, size=(96, 64), seed=3)


@pytest.fixture
def aws_env(monkeypatch):
    """Fake credentials so boto3 never touches real accounts or ~/.aws."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    for name in ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def s3_client(aws_env):
    """An in-memory S3 (moto) with the homelab's three buckets."""
    moto = pytest.importorskip("moto")
    import boto3

    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        for bucket in ("datasets", "checkpoints", "mlflow-artifacts"):
            client.create_bucket(Bucket=bucket)
        yield client


@pytest.fixture(autouse=True)
def _no_fault_injection(monkeypatch):
    monkeypatch.delenv("CROSSWALK_FAULT_AT_STEP", raising=False)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def subprocess_env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra)
    return env
