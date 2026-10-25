#!/usr/bin/env python3
"""Experiment C: queueing for one GPU with Kueue, with and without a competing workload.

Submits N competing GPU Jobs through scripts/submit-training.py (so each is
queued by Kueue), waits for them to finish, and records for each one when it
was created, admitted by Kueue, actually started (pod running) and
completed. From those it derives:

- **queue wait**: created -> Kueue admission (waiting for quota);
- **scheduling wait**: admission -> pod running (Kueue said yes but the GPU
  was still held by something Kueue does not manage);
- **run time**: running -> completed;
- **turnaround**: created -> completed.

It also writes the PromQL queries for GPU utilisation over the experiment's
window (from node-exporter's drm/hwmon metrics), so the run can
be lined up with what the GPU was doing. It never changes the cluster
beyond submitting the Jobs: starting the competing workload is a manual
step (see docs/benchmarks.md, Experiment C), so this script never takes the
GPU away from something else by accident.

    scripts/experiment-c.py --jobs 3 --kind validate --image-tag rocm-abc1234
    scripts/experiment-c.py --jobs 3 --kind train --config configs/train-synthetic.yaml \\
        --synthetic --set max_steps=200 --image-tag rocm-abc1234 --label gpu-contended

Pure-stdlib apart from kubectl and helm on PATH; the timeline maths is
importable and unit-tested (tests/test_experiment_c.py).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUBMIT = REPO / "scripts" / "submit-training.py"

# Queries for the GPU metrics (node-exporter drm and hwmon
# collectors). Evaluate them over [start, end] in Grafana or the Prometheus
# API; the script records the window.
GPU_QUERIES = {
    "gpu_busy_percent": "max(node_drm_gpu_busy_percent)",
    "vram_used_bytes": "max(node_drm_memory_vram_used_bytes)",
    "gpu_power_watts": 'max(node_hwmon_power_average_watt{chip=~".*amdgpu.*"}) or max(node_hwmon_power_watt)',
    "gpu_temp_celsius": 'max(node_hwmon_temp_celsius{chip=~".*amdgpu.*"})',
    "kueue_pending": 'sum(kueue_pending_workloads{cluster_queue="gpu"})',
    "kueue_admitted_active": 'sum(kueue_admitted_active_workloads{cluster_queue="gpu"})',
}


def parse_time(text: str | None) -> float | None:
    if not text:
        return None
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def condition_time(obj: dict, kind: str, status: str = "True") -> float | None:
    for c in (obj.get("status") or {}).get("conditions") or []:
        if c.get("type") == kind and c.get("status") == status:
            return parse_time(c.get("lastTransitionTime"))
    return None


@dataclass
class Timeline:
    job: str
    created: float | None
    admitted: float | None
    running: float | None
    finished: float | None
    succeeded: bool | None

    def durations(self) -> dict[str, float | None]:
        def gap(a, b):
            return round(b - a, 3) if a is not None and b is not None else None

        return {
            "queue_wait_s": gap(self.created, self.admitted),
            "scheduling_wait_s": gap(self.admitted, self.running),
            "run_s": gap(self.running, self.finished),
            "turnaround_s": gap(self.created, self.finished),
        }


def timeline(job: dict, workload: dict | None, pods: list[dict]) -> Timeline:
    """Build one Job's timeline from its Job, Workload and Pod objects."""
    created = parse_time(job["metadata"].get("creationTimestamp"))
    admitted = condition_time(workload, "Admitted") if workload else None
    if admitted is None and workload is None:
        # Not queued (kueue disabled): admission is the moment it was created.
        admitted = created
    running = None
    for pod in pods:
        started = parse_time((pod.get("status") or {}).get("startTime"))
        ready = condition_time(pod, "PodReadyToStartContainers") or condition_time(
            pod, "Initialized"
        )
        candidate = ready or started
        if candidate is not None and (running is None or candidate < running):
            running = candidate
    finished = condition_time(job, "Complete") or condition_time(job, "Failed")
    succeeded = (
        True
        if condition_time(job, "Complete")
        else (False if condition_time(job, "Failed") else None)
    )
    return Timeline(job["metadata"]["name"], created, admitted, running, finished, succeeded)


def summarize(timelines: list[Timeline]) -> dict:
    def stat(key):
        values = sorted(v for t in timelines if (v := t.durations()[key]) is not None)
        if not values:
            return None
        mid = len(values) // 2
        median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
        return {"min": values[0], "median": round(median, 3), "max": values[-1], "n": len(values)}

    return {k: stat(k) for k in ("queue_wait_s", "scheduling_wait_s", "run_s", "turnaround_s")}


def kubectl_json(context: str, namespace: str, *args: str) -> dict:
    cmd = (
        ["kubectl"]
        + (["--context", context] if context else [])
        + ["-n", namespace, *args, "-o", "json"]
    )
    return json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)


def workload_for(workloads: list[dict], job_name: str) -> dict | None:
    for w in workloads:
        for owner in w["metadata"].get("ownerReferences") or []:
            if owner.get("kind") == "Job" and owner.get("name") == job_name:
                return w
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--jobs", type=int, default=3, help="competing Jobs to submit at once")
    p.add_argument("--kind", default="validate", choices=["validate", "train", "benchmark"])
    p.add_argument("--config", help="crosswalk config for train/benchmark")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], help="config override for every Job"
    )
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--image-tag")
    p.add_argument(
        "--cpu", action="store_true", help="CPU image, no GPU request (dry wiring check)"
    )
    p.add_argument(
        "--label", default="", help="free text recorded with the result, e.g. gpu-contended"
    )
    p.add_argument("--namespace", default="crosswalk")
    p.add_argument(
        "--context",
        default=os.environ.get("CROSSWALK_KUBE_CONTEXT", "kind-homelab-dev"),
        help="kubectl context (default $CROSSWALK_KUBE_CONTEXT or kind-homelab-dev)",
    )
    p.add_argument("--timeout", type=float, default=7200)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--dry-run", action="store_true", help="print the submit commands only")
    args = p.parse_args(argv)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    names = [f"crosswalk-expc-{stamp}-{i}" for i in range(args.jobs)]
    commands = []
    for name in names:
        cmd = [
            sys.executable,
            str(SUBMIT),
            args.kind,
            "--name",
            name,
            "--namespace",
            args.namespace,
            "--context",
            args.context,
        ]
        if args.config:
            cmd += ["--config", args.config]
        for o in args.overrides:
            cmd += ["--set", o]
        if args.kind == "train":
            # Each Job is its own run: never resume another Job's checkpoints.
            cmd += ["--set", f"run_name={name}"]
        if args.synthetic:
            cmd.append("--synthetic")
        if args.image_tag:
            cmd += ["--image-tag", args.image_tag]
        if args.cpu:
            cmd.append("--cpu")
        commands.append(cmd)
    if args.dry_run:
        for cmd in commands:
            print(shlex.join(cmd))
        return 0

    started = time.time()
    for cmd in commands:  # back to back, so they compete for the queue
        subprocess.run(cmd, check=True)
    print(f"submitted {len(names)} Jobs; waiting for them to finish (timeout {args.timeout:.0f}s)")
    deadline = started + args.timeout
    while time.time() < deadline:
        jobs = {
            j["metadata"]["name"]: j
            for j in kubectl_json(args.context, args.namespace, "get", "jobs")["items"]
        }
        done = [
            n
            for n in names
            if n in jobs
            and (condition_time(jobs[n], "Complete") or condition_time(jobs[n], "Failed"))
        ]
        print(f"  {len(done)}/{len(names)} finished", flush=True)
        if len(done) == len(names):
            break
        time.sleep(15)
    ended = time.time()
    jobs = {
        j["metadata"]["name"]: j
        for j in kubectl_json(args.context, args.namespace, "get", "jobs")["items"]
    }
    try:
        workloads = kubectl_json(args.context, args.namespace, "get", "workloads.kueue.x-k8s.io")[
            "items"
        ]
    except subprocess.CalledProcessError:
        workloads = []  # Kueue not installed
    pods = kubectl_json(args.context, args.namespace, "get", "pods")["items"]
    timelines = []
    for name in names:
        if name not in jobs:
            continue
        job_pods = [pd for pd in pods if pd["metadata"].get("labels", {}).get("job-name") == name]
        timelines.append(timeline(jobs[name], workload_for(workloads, name), job_pods))

    result = {
        "schema": 1,
        "experiment": "C",
        "label": args.label,
        "kind": args.kind,
        "jobs": len(names),
        "window": {"start": started, "end": ended},
        "timelines": [asdict(t) | t.durations() for t in timelines],
        "summary": summarize(timelines),
        "gpu_queries": {
            k: {"expr": q, "start": started, "end": ended, "step": "15s"}
            for k, q in GPU_QUERIES.items()
        },
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"experiment-C-{stamp}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
