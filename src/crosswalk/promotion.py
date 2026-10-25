"""``crosswalk promote`` and ``crosswalk rollback``: the gate in front of serving.

A new model version serves only after it has beaten the current one on the
data that matters, the hand-labelled fixed-camera evaluation set, without
getting too slow. The gate:

1. **Resolve** the candidate (default: the newest version) and the champion
   (``@champion``; none yet means the candidate is compared with the
   pretrained COCO baseline, as training already does).
2. **Score both** on the same eval set with pycocotools (the training
   loop's ``evaluate`` code), same score threshold, same device.
3. **Time both** with the same latency check: a fixed number of batches of
   the eval images at the serving batch size, after warmup; p50/p95 of the
   wall time per batch, GPU synchronised.
4. **Decide** with explicit rules (``GateRules``):
   - mAP@[.5:.95] must not drop by more than ``max_map_drop`` (default 0:
     must be at least as good), and must improve by ``min_map_gain`` if set;
   - per-class AP for ``pedestrian`` must not drop by more than
     ``max_class_ap_drop``: an overall gain bought by losing pedestrians is
     the wrong trade for this project;
   - p95 batch latency must not grow by more than ``max_latency_ratio``
     (default 1.2x) over the champion's.
5. **Record** the decision: a JSON report logged to a new MLflow run in the
   ``crosswalk-promotion`` experiment, tags on the candidate version
   (``promotion.decision``, ``promotion.report_run``), and the
   ``challenger`` alias on the candidate. If approved (and not
   ``--dry-run``), ``champion`` moves to the candidate.

Rollback moves ``champion`` back to the previous champion (or a given
version) and records that too. The inference Deployment serves
``models:/crosswalk-detector@champion``; a restart (``kubectl rollout
restart deployment/crosswalk-inference``) picks up the change.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from crosswalk import registry
from crosswalk.stats import percentile

log = logging.getLogger(__name__)

EXPERIMENT = "crosswalk-promotion"


@dataclass(frozen=True)
class GateRules:
    max_map_drop: float = 0.0
    min_map_gain: float = 0.0
    watch_class: str = "pedestrian"
    max_class_ap_drop: float = 0.01
    max_latency_ratio: float = 1.2


@dataclass
class Candidate:
    label: str  # "candidate", "champion" or "baseline"
    uri: str | None
    version: str | None
    metrics: dict[str, float]
    latency_ms: dict[str, float | None]


@dataclass
class Decision:
    approved: bool
    reasons: list[str]
    candidate: Candidate
    champion: Candidate
    rules: GateRules
    eval_dataset_sha256: str
    decided_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = "approved" if self.approved else "rejected"
        return d


def decide(candidate: Candidate, champion: Candidate, rules: GateRules) -> tuple[bool, list[str]]:
    """Pure rule evaluation, so it is easy to test exhaustively."""
    reasons: list[str] = []
    ok = True
    c_map, b_map = candidate.metrics.get("map"), champion.metrics.get("map")
    if c_map is None or b_map is None:
        return False, ["mAP missing for one of the models: cannot compare"]
    delta = c_map - b_map
    if delta < -rules.max_map_drop:
        ok = False
        reasons.append(
            f"mAP {c_map:.4f} is {-delta:.4f} below the {champion.label} ({b_map:.4f}); allowed drop {rules.max_map_drop}"
        )
    elif rules.min_map_gain > 0 and delta < rules.min_map_gain:
        ok = False
        reasons.append(f"mAP gain {delta:+.4f} is below the required {rules.min_map_gain}")
    else:
        reasons.append(f"mAP {c_map:.4f} vs {b_map:.4f} ({delta:+.4f})")
    key = f"ap/{rules.watch_class}"
    if key in candidate.metrics and key in champion.metrics:
        c_ap, b_ap = candidate.metrics[key], champion.metrics[key]
        if c_ap == c_ap and b_ap == b_ap:  # both not NaN
            if c_ap < b_ap - rules.max_class_ap_drop:
                ok = False
                reasons.append(
                    f"{rules.watch_class} AP fell {b_ap - c_ap:.4f} ({b_ap:.4f} -> {c_ap:.4f}); allowed {rules.max_class_ap_drop}"
                )
            else:
                reasons.append(f"{rules.watch_class} AP {c_ap:.4f} vs {b_ap:.4f}")
    c_lat, b_lat = candidate.latency_ms.get("p95"), champion.latency_ms.get("p95")
    if c_lat is not None and b_lat:
        ratio = c_lat / b_lat
        if ratio > rules.max_latency_ratio:
            ok = False
            reasons.append(
                f"p95 latency {c_lat:.1f} ms is {ratio:.2f}x the {champion.label}'s {b_lat:.1f} ms; allowed {rules.max_latency_ratio}x"
            )
        else:
            reasons.append(f"p95 latency {c_lat:.1f} ms vs {b_lat:.1f} ms ({ratio:.2f}x)")
    return ok, reasons


def latency_check(
    model, images, device, batch_size: int, batches: int = 20, warmup: int = 3
) -> dict[str, float | None]:
    """Per-batch wall time (ms) on fixed inputs, GPU synchronised."""
    import torch

    from crosswalk import device as devices

    if not images:
        return {"p50": None, "p95": None, "batches": 0, "batch_size": batch_size}
    batch = [images[i % len(images)].to(device) for i in range(batch_size)]
    model.eval()
    times = []
    with torch.inference_mode():
        for i in range(warmup + batches):
            t0 = time.perf_counter()
            model(batch)
            devices.synchronize(device)
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    return {
        "p50": percentile(times, 50),
        "p95": percentile(times, 95),
        "batches": len(times),
        "batch_size": batch_size,
    }


def _score(model, parsed, image_root, config, dev, classes, mapping=None) -> dict[str, float]:
    from crosswalk.training import datasets, evaluate

    dataset = datasets.DetectionDataset(parsed, image_root, train=False)
    saved = getattr(model, "score_thresh", None)
    if saved is not None:
        model.score_thresh = config.eval.score_threshold
    try:
        preds = evaluate.predict_dataset(model, dataset, dev, config.eval.batch_size)
    finally:
        if saved is not None:
            model.score_thresh = saved
    if mapping is not None:
        preds = [evaluate.map_labels(p, mapping) for p in preds]
    return evaluate.coco_map(parsed.samples, preds, classes)


def evaluate_version(
    uri: str | None, config, parsed, dev, classes, batch_size: int, latency_batches: int
) -> tuple[dict, dict]:
    """mAP and latency of a registered version, or of the COCO baseline (uri None)."""
    from crosswalk.config import ModelConfig
    from crosswalk.models import build_detector
    from crosswalk.training import datasets, evaluate

    if uri is None:
        model_config = ModelConfig(
            name="fcos_resnet50_fpn", weights="COCO_V1", classes=None,
            min_size=config.model.min_size, max_size=config.model.max_size,
            score_threshold=config.eval.score_threshold,
        )  # fmt: skip
        model = build_detector(model_config).to(dev).eval()
        mapping = evaluate.coco_to_project(classes, config.eval.baseline_category_map)
    else:
        saved = registry.load_weights(uri)
        if list(saved["classes"]) != list(classes):
            raise ValueError(f"{uri} predicts {saved['classes']}, the eval set uses {classes}")
        model_config = ModelConfig.model_validate(
            {**saved.get("model_config", {}), "weights": "none", "checkpoint": None}
        )
        model = build_detector(model_config)
        model.load_state_dict(saved["model"])
        model.to(dev).eval()
        mapping = None
    metrics = _score(model, parsed, config.data.eval.images, config, dev, classes, mapping)
    ds = datasets.DetectionDataset(parsed, config.data.eval.images, train=False)
    images = [ds[i][0] for i in range(min(len(ds), batch_size))]
    latency = latency_check(model, images, dev, batch_size, batches=latency_batches)
    return metrics, latency


def promote(
    config,
    name: str,
    candidate_version: str | None = None,
    rules: GateRules | None = None,
    batch_size: int = 4,
    latency_batches: int = 20,
    dry_run: bool = False,
    mlflow_client=None,
    evaluator: Callable[..., tuple[dict, dict]] = evaluate_version,
) -> Decision:
    """Run the gate. ``config`` is a TrainingConfig whose ``data.eval`` is
    the fixed-camera set (the same file training uses)."""
    import mlflow

    from crosswalk import device as devices
    from crosswalk.training import datasets, hashing

    rules = rules or GateRules()
    c = mlflow_client or registry.client()
    # The device is only needed by the real evaluator (an injected one, as
    # in tests, may not need PyTorch at all).
    dev = devices.resolve(config.device) if evaluator is evaluate_version else None
    if not config.data.eval:
        raise SystemExit("the config has no data.eval: the gate needs the fixed-camera eval set")
    classes = list(config.model.classes or [])
    parsed = datasets.load(config.data.eval, classes, config.seed)
    eval_sha = hashing.dataset_manifest(parsed, config.data.eval.images, config.data.hash_images)[
        "sha256"
    ]

    if candidate_version is None:
        versions = c.search_model_versions(f"name='{name}'")
        if not versions:
            raise SystemExit(f"{name} has no registered versions")
        candidate_version = str(max(int(v.version) for v in versions))
    cand_uri = f"models:/{name}/{candidate_version}"
    champ_version = registry.alias_version(c, name, registry.CHAMPION)
    if champ_version == candidate_version:
        raise SystemExit(f"version {candidate_version} is already the champion")

    log.info("evaluating candidate %s", cand_uri)
    c_metrics, c_lat = evaluator(
        cand_uri, config, parsed, dev, classes, batch_size, latency_batches
    )
    candidate = Candidate("candidate", cand_uri, candidate_version, c_metrics, c_lat)
    if champ_version:
        champ_uri = f"models:/{name}/{champ_version}"
        log.info("evaluating champion %s", champ_uri)
        b_metrics, b_lat = evaluator(
            champ_uri, config, parsed, dev, classes, batch_size, latency_batches
        )
        champion = Candidate("champion", champ_uri, champ_version, b_metrics, b_lat)
    else:
        log.info("no champion yet: comparing with the pretrained COCO baseline")
        b_metrics, b_lat = evaluator(
            None, config, parsed, dev, classes, batch_size, latency_batches
        )
        champion = Candidate("baseline", None, None, b_metrics, b_lat)

    approved, reasons = decide(candidate, champion, rules)
    decision = Decision(approved, reasons, candidate, champion, rules, eval_sha)

    # Record: an MLflow run with the full report, tags on the version.
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"promote-{name}-v{candidate_version}") as run:
        mlflow.set_tags({
            "decision": "approved" if approved else "rejected", "candidate": cand_uri,
            "champion": champion.uri or "pretrained COCO baseline", "eval_dataset.sha256": eval_sha,
            "dry_run": str(dry_run),
        })  # fmt: skip
        mlflow.log_metrics(
            {f"candidate/{k.replace('/', '_')}": v for k, v in c_metrics.items() if v == v}
        )
        mlflow.log_metrics(
            {f"champion/{k.replace('/', '_')}": v for k, v in b_metrics.items() if v == v}
        )
        for who, lat in (("candidate", c_lat), ("champion", b_lat)):
            mlflow.log_metrics(
                {
                    f"{who}/latency_{k}_ms": v
                    for k, v in lat.items()
                    if k in ("p50", "p95") and v is not None
                }
            )
        mlflow.log_dict(decision.to_dict(), "promotion.json")
        report_run = run.info.run_id
    c.set_model_version_tag(
        name, candidate_version, "promotion.decision", "approved" if approved else "rejected"
    )
    c.set_model_version_tag(name, candidate_version, "promotion.report_run", report_run)
    c.set_model_version_tag(name, candidate_version, "promotion.eval_dataset_sha256", eval_sha)
    c.set_registered_model_alias(name, registry.CHALLENGER, candidate_version)
    if approved and not dry_run:
        before = registry.make_champion(c, name, candidate_version)
        log.info("champion: %s -> %s", before or "none", candidate_version)
    return decision


def rollback(
    name: str, to_version: str | None = None, mlflow_client=None, reason: str = ""
) -> tuple[str, str]:
    import mlflow

    c = mlflow_client or registry.client()
    before, after = registry.rollback(c, name, to_version)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"rollback-{name}-v{before}-to-v{after}"):
        mlflow.set_tags(
            {"decision": "rollback", "from_version": before, "to_version": after, "reason": reason}
        )
        mlflow.log_dict(
            {"rollback": {"from": before, "to": after, "reason": reason, "at": time.time()}},
            "rollback.json",
        )
    c.set_model_version_tag(name, before, "promotion.decision", f"rolled back to v{after}")
    return before, after


def report(decision: Decision) -> str:
    verdict = "APPROVED" if decision.approved else "REJECTED"
    lines = [
        f"{verdict}: {decision.candidate.uri} vs {decision.champion.uri or 'pretrained COCO baseline'}",
        *(f"  - {r}" for r in decision.reasons),
        f"  eval set sha256 {decision.eval_dataset_sha256[:12]}",
    ]
    return "\n".join(lines)


def to_json(decision: Decision) -> str:
    return json.dumps(decision.to_dict(), indent=2, default=str)
