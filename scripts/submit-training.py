#!/usr/bin/env python3
"""Submit a training, validation or benchmark Job to the cluster.

Argo CD keeps the long-running parts (the inference Deployment, Service,
dashboard) in step with Git. One-off Jobs do not belong there: each run is a
new object with its own config, and Argo CD would either prune it or keep
re-creating it. So Jobs are rendered from the same Helm chart on demand:

    helm template ... --show-only templates/job.yaml --set-file job.config=<file>
      | kubectl create -f -

This script builds that command line, so every run uses the chart's security
context, GPU request, volumes and S3/MLflow wiring, and only the config
differs. ``kubectl create`` (not ``apply``) is deliberate: a training run is
created once. Submitting the same run name twice fails with "already exists"
instead of silently changing a running Job. To resume a run whose Job gave up
(its retries were exhausted), delete the Job and submit again: training resumes
from the newest checkpoint in S3.

Examples:

    scripts/submit-training.py validate
    scripts/submit-training.py train --config configs/train-synthetic.yaml --synthetic --cpu
    scripts/submit-training.py train --config configs/train-bdd100k.yaml --image-tag rocm-abc1234
    scripts/submit-training.py benchmark --config configs/benchmark-a.yaml --image-tag rocm-abc1234
    scripts/submit-training.py train --config configs/train-bdd100k.yaml --dry-run   # print the manifest
    scripts/submit-training.py promote --config configs/train-bdd100k.yaml --image-tag rocm-abc1234

Kueue: with the chart's kueue.enabled (the default) every Job is
created suspended with kueue.x-k8s.io/queue-name: crosswalk and waits for
the gpu ClusterQueue to admit it. --no-kueue submits a plain Job (for a
cluster without Kueue). `kubectl get workloads -n crosswalk` shows the queue.

Data: training configs read datasets from /scratch/data/<name>/...; by
convention each <name> is synced from s3://datasets/<name>/ by an init
container before training (override with --sync, skip with --no-sync).

Needs helm and kubectl on PATH; PyYAML (optional) lets it read run_name and
dataset paths from the config.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CHART = REPO / "chart" / "crosswalk"
SCRATCH_DATA = "/scratch/data/"
DATA_BUCKET = "s3://datasets/"


def dns_label(text: str, limit: int = 52) -> str:
    """Lower-case DNS label; 52 leaves room for the pod-name suffix Kubernetes adds."""
    label = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    return label[:limit].rstrip("-") or "run"


def read_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def derive_sync(config: dict) -> list[str]:
    """s3://datasets/<name>=/scratch/data/<name> for every dataset the config reads."""
    names: list[str] = []
    for split in (config.get("data") or {}).values():
        if not isinstance(split, dict):
            continue
        for key in ("labels", "images"):
            value = str(split.get(key) or "")
            if value.startswith(SCRATCH_DATA):
                name = value[len(SCRATCH_DATA) :].split("/", 1)[0]
                if name and name not in names:
                    names.append(name)
    return [f"{DATA_BUCKET}{n}={SCRATCH_DATA}{n}" for n in names]


def job_name(kind: str, config: dict, explicit: str | None, now: dt.datetime) -> str:
    if explicit:
        return dns_label(explicit)
    if kind == "train" and config.get("run_name"):
        # One Job per training run: the name is the run, so a duplicate submit fails.
        return dns_label(f"crosswalk-train-{config['run_name']}")
    return dns_label(f"crosswalk-{kind}-{now:%Y%m%d-%H%M%S}")


def helm_command(args, name: str, config_file: Path | None, sync: list[str]) -> list[str]:
    cmd = [
        "helm",
        "template",
        "crosswalk",
        str(CHART),
        "--namespace",
        args.namespace,
        "--show-only",
        "templates/job.yaml",
        "--show-only",
        "templates/job-config.yaml",
        "--set",
        "job.enabled=true",
        "--set-string",
        f"job.kind={args.kind}",
        "--set-string",
        f"job.name={name}",
        "--set",
        f"gpu.enabled={'false' if args.cpu else 'true'}",
    ]
    if config_file is not None:
        cmd += ["--set-file", f"job.config={config_file}"]
    if args.image_tag:
        key = "image.cpu.tag" if args.cpu else "image.gpu.tag"
        cmd += ["--set-string", f"{key}={args.image_tag}"]
    if getattr(args, "synthetic", False):
        cmd += ["--set", "job.synthetic=true"]
    for i, pair in enumerate(sync):
        cmd += ["--set-string", f"job.sync[{i}]={pair}"]
    for i, override in enumerate(args.overrides):
        # Passed to the crosswalk command inside the Job as --set KEY=VALUE.
        cmd += [
            "--set-string",
            f"job.args[{2 * i}]=--set",
            "--set-string",
            f"job.args[{2 * i + 1}]={override}",
        ]
    for i, pair in enumerate(args.env):
        name, _, value = pair.partition("=")
        cmd += [
            "--set-string",
            f"job.env[{i}].name={name}",
            "--set-string",
            f"job.env[{i}].value={value}",
        ]
    if getattr(args, "gpus", 1) > 1:
        cmd += ["--set", f"job.gpus={args.gpus}"]
    if getattr(args, "no_kueue", False):
        cmd += ["--set", "kueue.enabled=false"]
    for value_file in args.values:
        cmd += ["--values", value_file]
    for raw in args.helm_set:
        cmd += ["--set", raw]
    return cmd


def kubectl(args, *rest: str) -> list[str]:
    cmd = ["kubectl"]
    if args.context:
        cmd += ["--context", args.context]
    return [*cmd, "--namespace", args.namespace, *rest]


def submit(args) -> int:
    config: dict = {}
    config_file: Path | None = None
    if args.kind != "validate":
        if not args.config:
            print(f"{args.kind} needs --config", file=sys.stderr)
            return 2
        config_file = Path(args.config).resolve()
        config = read_config(config_file)
    name = job_name(args.kind, config, args.name, dt.datetime.now())
    sync: list[str] = []
    if args.kind in ("train", "promote") and not args.no_sync and not args.synthetic:
        sync = args.sync or derive_sync(config)
    elif args.sync:
        sync = args.sync

    cmd = helm_command(args, name, config_file, sync)
    print("+ " + shlex.join(cmd), file=sys.stderr)
    rendered = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    if args.dry_run:
        print(rendered)
        return 0
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(rendered)
        manifest = handle.name
    subprocess.run(kubectl(args, "create", "-f", manifest), check=True)
    # Tie the ConfigMap's lifetime to the Job, so the Job's TTL cleanup (or a
    # manual delete) removes the config with it.
    uid = subprocess.run(
        kubectl(args, "get", "job", name, "-o", "jsonpath={.metadata.uid}"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    owner = (
        '{"metadata":{"ownerReferences":[{"apiVersion":"batch/v1","kind":"Job",'
        f'"name":"{name}","uid":"{uid}","blockOwnerDeletion":false}}]}}}}'
    )
    subprocess.run(
        kubectl(args, "patch", "configmap", f"{name}-config", "--type", "merge", "-p", owner),
        check=True,
    )
    print(f"\nsubmitted job/{name}. Follow it with:")
    if not args.no_kueue:
        print(
            "  "
            + shlex.join(kubectl(args, "get", "workloads", "-w"))
            + "   # Kueue: queued -> admitted"
        )
    print("  " + shlex.join(kubectl(args, "logs", "-f", f"job/{name}")))
    print("  " + shlex.join(kubectl(args, "get", "pods", "-l", f"job-name={name}", "-w")))
    if not args.cpu:
        print(
            "If the pod stays Pending, another workload holds the GPU or the cluster has"
            " no GPU device plugin (kubectl describe pod shows which)."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", choices=["train", "validate", "benchmark", "promote"])
    parser.add_argument("--config", help="crosswalk YAML config (train, benchmark)")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override passed to the Job",
    )
    parser.add_argument("--name", help="Job name (default: from run_name, or kind plus timestamp)")
    parser.add_argument(
        "--cpu", action="store_true", help="no GPU request; CPU image (gpu.enabled=false)"
    )
    parser.add_argument(
        "--image-tag", help="image tag (image.gpu.tag, or image.cpu.tag with --cpu)"
    )
    parser.add_argument(
        "--sync",
        action="append",
        default=[],
        metavar="S3=DIR",
        help="dataset prefix to download first",
    )
    parser.add_argument("--no-sync", action="store_true", help="skip the dataset download")
    parser.add_argument(
        "--synthetic", action="store_true", help="generate the synthetic dataset instead of syncing"
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra environment for the Job, e.g. CROSSWALK_FAULT_AT_STEP=500",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="train only: GPUs for one DistributedDataParallel Job (torchrun, one process each)",
    )
    parser.add_argument(
        "--no-kueue", action="store_true", help="plain Job, not queued (cluster without Kueue)"
    )
    parser.add_argument("--values", action="append", default=[], help="extra Helm values file")
    parser.add_argument("--helm-set", action="append", default=[], help="extra raw Helm --set")
    parser.add_argument("--namespace", default="crosswalk")
    parser.add_argument(
        "--context",
        default=os.environ.get("CROSSWALK_KUBE_CONTEXT", "kind-homelab-dev"),
        help="kubectl context ('' for current; default $CROSSWALK_KUBE_CONTEXT or kind-homelab-dev)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the rendered manifest instead of creating it"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return submit(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
