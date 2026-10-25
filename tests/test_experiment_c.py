"""Experiment C timeline maths, from Job/Workload/Pod objects as kubectl returns them."""

import importlib.util
import sys

import pytest

from tests.conftest import repo_root


@pytest.fixture(scope="module")
def expc():
    path = repo_root() / "scripts" / "experiment-c.py"
    spec = importlib.util.spec_from_file_location("experiment_c", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses look the module up by name while the class is created.
    sys.modules["experiment_c"] = module
    spec.loader.exec_module(module)
    return module


def cond(kind, ts, status="True"):
    return {"type": kind, "status": status, "lastTransitionTime": ts}


def job(name, created, complete=None, failed=None):
    conditions = [
        c
        for c in (
            cond("Complete", complete) if complete else None,
            cond("Failed", failed) if failed else None,
        )
        if c
    ]
    return {
        "metadata": {"name": name, "creationTimestamp": created},
        "status": {"conditions": conditions},
    }


def test_timeline_from_kueue_objects(expc):
    j = job("a", "2026-10-25T10:00:00Z", complete="2026-10-25T10:10:00Z")
    w = {"metadata": {"ownerReferences": [{"kind": "Job", "name": "a"}]},
         "status": {"conditions": [cond("QuotaReserved", "2026-10-25T10:01:30Z"), cond("Admitted", "2026-10-25T10:02:00Z")]}}  # fmt: skip
    pod = {"metadata": {"labels": {"job-name": "a"}},
           "status": {"startTime": "2026-10-25T10:02:05Z", "conditions": [cond("Initialized", "2026-10-25T10:03:00Z")]}}  # fmt: skip
    t = expc.timeline(j, expc.workload_for([w], "a"), [pod])
    assert t.succeeded is True
    assert t.durations() == {
        "queue_wait_s": 120.0,
        "scheduling_wait_s": 60.0,
        "run_s": 420.0,
        "turnaround_s": 600.0,
    }


def test_unqueued_job_and_unfinished_job(expc):
    j = job("b", "2026-10-25T10:00:00Z", failed="2026-10-25T10:00:30Z")
    t = expc.timeline(j, None, [])
    assert t.succeeded is False and t.durations()["queue_wait_s"] == 0.0
    running = expc.timeline(job("c", "2026-10-25T10:00:00Z"), None, [])
    assert running.succeeded is None and running.durations()["turnaround_s"] is None


def test_summary_medians(expc):
    T = expc.Timeline
    ts = [T("a", 0, 10, 12, 100, True), T("b", 0, 50, 51, 140, True), T("c", 0, 90, 95, 180, True)]
    s = expc.summarize(ts)
    assert s["queue_wait_s"] == {"min": 10, "median": 50, "max": 90, "n": 3}
    assert s["run_s"]["median"] == 88  # runs of 88, 89 and 85 s


def test_dry_run_builds_one_queued_submission_per_job(expc, capsys):
    assert (
        expc.main(
            [
                "--jobs",
                "2",
                "--kind",
                "train",
                "--config",
                "configs/train-synthetic.yaml",
                "--synthetic",
                "--dry-run",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert all("submit-training.py train" in line and "--synthetic" in line for line in lines)
    assert all("run_name=crosswalk-expc-" in line for line in lines)
    assert all("--no-kueue" not in line for line in lines)
