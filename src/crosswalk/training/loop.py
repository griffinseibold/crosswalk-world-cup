"""The training loop: written out by hand, not delegated to a trainer library.

One call to :func:`train` is one training job. It is safe to kill at any
moment and run again with the same config: the second run finds the newest
checkpoint, restores model, optimizer, scheduler, scaler, progress and RNG
state, reattaches to the same MLflow run and carries on from the exact batch
after the checkpoint.

Experiment B (failure recovery) is measured here:

- A heartbeat (``heartbeat.json``: last completed step and wall time) is
  written next to the checkpoints every ``log_every_steps`` steps.
- On resume, ``lost_steps`` = heartbeat step - checkpoint step (work redone),
  ``downtime_s`` = resume start - heartbeat time (kill, rescheduling, image
  start), ``restore_s`` = time to find, download and load the checkpoint, and
  ``recovery_s`` = process start to the first completed step after resuming.
- ``CROSSWALK_FAULT_AT_STEP=N`` makes a *fresh* run SIGKILL itself right after
  step N (never a resumed run, so it cannot loop). On the cluster, deleting
  the pod does the same from outside; see docs/benchmarks.md.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosswalk.config import ModelConfig, TrainingConfig, config_hash
from crosswalk.storage import atomic_write_bytes
from crosswalk.training import datasets, evaluate, hashing, state
from crosswalk.training.checkpoint import CheckpointManager, make_manager
from crosswalk.training.sampling import ResumableSampler, steps_per_epoch
from crosswalk.training.schedule import WarmupCosine
from crosswalk.training.tracking import make_tracker, write_json_artifact

log = logging.getLogger(__name__)

HEARTBEAT = "heartbeat.json"
FAULT_ENV = "CROSSWALK_FAULT_AT_STEP"

# Config sections that must match for a checkpoint to be resumed. Changing any
# of them mid-run would silently produce a model that no single config
# describes. Checkpoint location, tracking and logging may change freely.
RESUME_KEYS = (
    "seed",
    "model",
    "data",
    "optimizer",
    "scheduler",
    "epochs",
    "batch_size",
    "max_steps",
    "amp",
    "grad_clip_norm",
)


class ResumeMismatch(RuntimeError):
    pass


def resume_hash(config: TrainingConfig) -> str:
    import hashlib

    subset = {k: v for k, v in config.model_dump(mode="json").items() if k in RESUME_KEYS}
    # The checkpoint to *start* from is irrelevant once a run has its own.
    subset["model"].pop("checkpoint", None)
    # Loader parallelism and hashing effort do not change what is trained on.
    subset["data"].pop("num_workers", None)
    subset["data"].pop("hash_images", None)
    return hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class RunSummary:
    run_name: str
    final_step: int
    total_steps: int
    resumed: bool
    resume: dict[str, Any] | None
    checkpoint_storage: str
    dataset_sha256: str
    mlflow_run_id: str | None
    eval: dict[str, float] = field(default_factory=dict)
    baseline_eval: dict[str, float] = field(default_factory=dict)
    upload_failures: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _write_heartbeat(manager: CheckpointManager, step: int, epoch: int) -> None:
    data = json.dumps({"step": step, "epoch": epoch, "time": time.time()}).encode()
    atomic_write_bytes(Path(manager.local_dir) / HEARTBEAT, data)
    if manager.remote is not None:
        # Best effort: a missed heartbeat only blurs Experiment B's numbers.
        with contextlib.suppress(Exception):
            manager.remote.put_bytes(data, HEARTBEAT)


def _read_heartbeat(manager: CheckpointManager) -> dict[str, Any] | None:
    candidates = []
    for store in (manager.remote, manager.local):
        if store is None:
            continue
        try:
            raw = store.get_bytes(HEARTBEAT)
            if raw:
                candidates.append(json.loads(raw))
        except Exception:
            continue
    return max(candidates, key=lambda h: h.get("step", -1)) if candidates else None


def _maybe_inject_fault(step: int, resumed: bool) -> None:
    target = os.environ.get(FAULT_ENV)
    if target and not resumed and step == int(target):
        log.warning("fault injection: SIGKILL at step %d", step)
        logging.shutdown()
        os.kill(os.getpid(), signal.SIGKILL)


def _datasets(config: TrainingConfig, classes: list[str]):
    train = datasets.load(config.data.train, classes, config.seed)
    val = datasets.load(config.data.val, classes, config.seed) if config.data.val else None
    ev = datasets.load(config.data.eval, classes, config.seed) if config.data.eval else None
    return train, val, ev


def _evaluate(
    model, parsed, image_root, config: TrainingConfig, device, classes
) -> dict[str, float]:
    dataset = datasets.DetectionDataset(parsed, image_root, train=False)
    # Keep low-score boxes for mAP; restore the serving threshold afterwards.
    saved = model.score_thresh
    model.score_thresh = config.eval.score_threshold
    try:
        preds = evaluate.predict_dataset(model, dataset, device, config.eval.batch_size)
    finally:
        model.score_thresh = saved
        model.train()
    return evaluate.coco_map(parsed.samples, preds, classes)


def baseline_evaluate(
    config: TrainingConfig, parsed, image_root, device, classes
) -> dict[str, float]:
    """Pretrained COCO FCOS on the same eval set, labels mapped to project classes."""
    from crosswalk.models import build_detector

    base_cfg = ModelConfig(
        name="fcos_resnet50_fpn",
        weights="COCO_V1",
        classes=None,
        min_size=config.model.min_size,
        max_size=config.model.max_size,
        score_threshold=config.eval.score_threshold,
    )
    model = build_detector(base_cfg).to(device).eval()
    dataset = datasets.DetectionDataset(parsed, image_root, train=False)
    preds = evaluate.predict_dataset(model, dataset, device, config.eval.batch_size)
    mapping = evaluate.coco_to_project(classes, config.eval.baseline_category_map)
    mapped = [evaluate.map_labels(p, mapping) for p in preds]
    del model
    return evaluate.coco_map(parsed.samples, mapped, classes)


def train(config: TrainingConfig, *, s3_client=None, tracker=None) -> RunSummary:
    import torch
    from torch.utils.data import DataLoader

    from crosswalk import device as devices
    from crosswalk.models import build_detector

    process_start = time.monotonic()
    wall_start = time.time()
    classes = list(config.model.classes or [])
    state.seed_everything(config.seed, config.deterministic)
    dev = devices.resolve(config.device)
    runtime = devices.describe(dev)
    log.info("device %s (%s)", dev, runtime.get("gpu_name") or "cpu")

    # -- data -----------------------------------------------------------------
    train_parsed, val_parsed, eval_parsed = _datasets(config, classes)
    manifest = hashing.dataset_manifest(
        train_parsed, config.data.train.images, config.data.hash_images
    )
    eval_manifest = (
        hashing.dataset_manifest(eval_parsed, config.data.eval.images, config.data.hash_images)
        if eval_parsed
        else None
    )
    log.info("train set: %s (sha256 %s)", train_parsed.summary(), manifest["sha256"][:12])
    train_ds = datasets.DetectionDataset(
        train_parsed, config.data.train.images, train=True, seed=config.seed
    )
    sampler = ResumableSampler(len(train_ds), config.batch_size, config.seed)
    per_epoch = steps_per_epoch(len(train_ds), config.batch_size)
    total_steps = per_epoch * config.epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    # -- model and optimizer ----------------------------------------------------
    model = build_detector(config.model, for_training=True)
    if config.model.checkpoint:
        # Initialise from an earlier run's weights (not a resume).
        state.load_model_weights(model, config.model.checkpoint)
    model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt_cfg = config.optimizer
    if opt_cfg.name == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=opt_cfg.lr, momentum=opt_cfg.momentum, weight_decay=opt_cfg.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(params, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        WarmupCosine(total_steps, config.scheduler.warmup_steps, config.scheduler.min_lr_ratio),
    )
    use_amp = config.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -- resume ----------------------------------------------------------------
    manager, storage_desc = make_manager(
        config.run_name,
        config.checkpoint.s3_uri,
        config.checkpoint.local_dir,
        config.checkpoint.keep_last,
        s3_client=s3_client,
    )
    log.info("checkpoints: %s", storage_desc)
    heartbeat = _read_heartbeat(manager)
    restore_t0 = time.monotonic()
    found = manager.find_latest()
    if config.checkpoint.resume == "never" and found is not None:
        raise ResumeMismatch(
            f"run {config.run_name!r} already has checkpoints and checkpoint.resume=never; "
            "pick a new run_name or delete them"
        )
    if config.checkpoint.resume == "must" and found is None:
        raise ResumeMismatch(
            f"checkpoint.resume=must but run {config.run_name!r} has no checkpoint"
        )

    progress = state.Progress()
    mlflow_run_id = None
    resume_info: dict[str, Any] | None = None
    rhash = resume_hash(config)
    if found is not None:
        saved = state.load_state(str(found.path))
        if saved.get("extra", {}).get("resume_hash") != rhash:
            raise ResumeMismatch(
                f"checkpoint step {found.step} was written with a different training config "
                "(seed, model, data, optimizer, schedule or batch size changed); use a new run_name"
            )
        progress = state.restore(
            saved, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler
        )
        mlflow_run_id = saved.get("extra", {}).get("mlflow_run_id")
        restore_s = time.monotonic() - restore_t0
        lost = None
        downtime = None
        if heartbeat and heartbeat.get("step", -1) >= progress.step:
            lost = int(heartbeat["step"]) - progress.step
            downtime = wall_start - float(heartbeat["time"])
        resume_info = {
            "checkpoint_step": progress.step,
            "checkpoint_source": found.source,
            "heartbeat_step": heartbeat.get("step") if heartbeat else None,
            "lost_steps": lost,
            "downtime_s": downtime,
            "restore_s": restore_s,
        }
        log.info("resumed from step %d (%s): %s", progress.step, found.source, resume_info)
    resumed = found is not None

    # -- tracking ----------------------------------------------------------------
    if tracker is None:
        tracker = make_tracker(
            config.tracking.enabled, config.tracking.experiment, config.run_name, mlflow_run_id
        )
    if not resumed:
        tracker.log_params(config.model_dump(mode="json"))
        tracker.set_tags(
            {
                "dataset.sha256": manifest["sha256"],
                "eval_dataset.sha256": eval_manifest["sha256"] if eval_manifest else "",
                "config_hash": config_hash(config),
                "device": runtime.get("gpu_name") or "cpu",
                "gpu_arch": runtime.get("gpu_arch") or "",
                "torch": runtime.get("torch"),
                "hip": runtime.get("torch_hip") or "",
                "image": runtime.get("container_image") or "",
                "git_commit": runtime.get("git_commit") or "",
            }
        )
        tracker.log_dict(manifest, "dataset/train_manifest.json")
        if eval_manifest:
            tracker.log_dict(eval_manifest, "dataset/eval_manifest.json")
        tracker.log_dict(runtime, "system/runtime.json")
    else:
        tracker.log_dict(runtime, f"system/runtime-resume-step-{progress.step}.json")

    def save_checkpoint(reason: str) -> None:
        result = manager.save(
            progress.step,
            lambda path: state.save_state(
                state.build_state(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    progress=progress,
                    classes=classes,
                    config=config.model_dump(mode="json"),
                    config_hash=config_hash(config),
                    extra={"mlflow_run_id": tracker.run_id, "resume_hash": rhash, "reason": reason},
                ),
                path,
            ),
            {"epoch": progress.epoch, "batch_in_epoch": progress.batch_in_epoch, "reason": reason},
        )
        tracker.log_metrics(
            {
                "system/checkpoint_serialize_s": result.serialize_s,
                "system/checkpoint_upload_s": result.upload_s,
                "system/checkpoint_bytes": result.bytes,
                "system/checkpoint_upload_failures": manager.upload_failures,
            },
            progress.step,
        )
        log.info("checkpoint step %d (%s, %.1f MB)", progress.step, reason, result.bytes / 1e6)

    # -- loop ------------------------------------------------------------------
    first_step_logged = False
    window_t0 = time.monotonic()
    window_images = 0
    window_data_wait = 0.0
    last_saved = progress.step if resumed else -1
    done = progress.step >= total_steps
    try:
        while not done and progress.epoch < config.epochs:
            epoch = progress.epoch
            sampler.set_position(epoch, progress.batch_in_epoch)
            train_ds.set_epoch(epoch)
            loader = DataLoader(
                train_ds,
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=config.data.num_workers,
                collate_fn=datasets.collate,
                drop_last=True,
                pin_memory=dev.type == "cuda",
            )
            fetch_t0 = time.monotonic()
            for images, targets in loader:
                window_data_wait += time.monotonic() - fetch_t0
                images = [img.to(dev, non_blocking=True) for img in images]
                targets = [
                    {"boxes": t["boxes"].to(dev), "labels": t["labels"].to(dev)} for t in targets
                ]
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    losses = model(images, targets)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss {loss.item()} at step {progress.step}; "
                        "the last checkpoint is intact, lower the learning rate and resume"
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                grad_norm = None
                if config.grad_clip_norm:
                    scaler.unscale_(optimizer)
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, config.grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                progress.step += 1
                progress.batch_in_epoch += 1
                window_images += len(images)

                if resumed and not first_step_logged:
                    resume_info["recovery_s"] = time.monotonic() - process_start
                    tracker.log_metrics(
                        {
                            f"system/resume_{k}": v
                            for k, v in resume_info.items()
                            if isinstance(v, (int, float)) and v is not None
                        },
                        progress.step,
                    )
                    first_step_logged = True

                if progress.step % config.log_every_steps == 0 or progress.step == total_steps:
                    devices.synchronize(dev)
                    elapsed = time.monotonic() - window_t0
                    tracker.log_metrics(
                        {
                            "quality/loss": float(loss.item()),
                            **{f"quality/{k}": float(v.item()) for k, v in losses.items()},
                            "quality/lr": scheduler.get_last_lr()[0],
                            "quality/grad_norm": grad_norm,
                            "system/images_per_s": window_images / elapsed if elapsed > 0 else None,
                            "system/step_s": elapsed / max(1, window_images / config.batch_size),
                            "system/data_wait_fraction": window_data_wait / elapsed
                            if elapsed > 0
                            else None,
                            "system/gpu_memory_peak_bytes": devices.max_memory_allocated(dev),
                            "system/epoch": epoch,
                        },
                        progress.step,
                    )
                    log.info(
                        "step %d/%d epoch %d loss %.4f lr %.2e",
                        progress.step, total_steps, epoch, loss.item(), scheduler.get_last_lr()[0],
                    )  # fmt: skip
                    _write_heartbeat(manager, progress.step, epoch)
                    window_t0, window_images, window_data_wait = time.monotonic(), 0, 0.0

                if progress.step % config.checkpoint.every_steps == 0:
                    save_checkpoint("periodic")
                    last_saved = progress.step
                _maybe_inject_fault(progress.step, resumed)
                if progress.step >= total_steps:
                    done = True
                    break
                fetch_t0 = time.monotonic()

            if progress.batch_in_epoch >= per_epoch:
                progress.epoch += 1
                progress.batch_in_epoch = 0
                if val_parsed and progress.epoch % config.eval.every_epochs == 0:
                    metrics = _evaluate(
                        model, val_parsed, config.data.val.images, config, dev, classes
                    )
                    tracker.log_metrics(
                        evaluate.finite(evaluate.prefixed(metrics, "quality/val").items()),
                        progress.step,
                    )

        if last_saved != progress.step:
            save_checkpoint("final")

        summary = RunSummary(
            run_name=config.run_name,
            final_step=progress.step,
            total_steps=total_steps,
            resumed=resumed,
            resume=resume_info,
            checkpoint_storage=storage_desc,
            dataset_sha256=manifest["sha256"],
            mlflow_run_id=tracker.run_id,
            upload_failures=manager.upload_failures,
        )

        # -- final evaluation: fine-tuned vs pretrained baseline ------------------
        if eval_parsed:
            summary.eval = _evaluate(
                model, eval_parsed, config.data.eval.images, config, dev, classes
            )
            tracker.log_metrics(
                evaluate.finite(evaluate.prefixed(summary.eval, "quality/eval").items()),
                progress.step,
            )
            if config.eval.baseline and config.model.name == "fcos_resnet50_fpn":
                summary.baseline_eval = baseline_evaluate(
                    config, eval_parsed, config.data.eval.images, dev, classes
                )
                tracker.log_metrics(
                    evaluate.finite(
                        evaluate.prefixed(summary.baseline_eval, "quality/baseline").items()
                    ),
                    progress.step,
                )
        write_json_artifact(tracker, summary.to_dict(), "run_summary.json")
        if config.tracking.registered_model_name:
            tracker.register_model(
                model,
                config.tracking.registered_model_name,
                {
                    "classes": classes,
                    "dataset_sha256": manifest["sha256"],
                    "eval_map": summary.eval.get("map"),
                    "step": progress.step,
                },
            )
        tracker.end("FINISHED")
        _write_summary(manager, summary)
        return summary
    except BaseException:
        # SIGKILL never reaches here; SIGTERM/exceptions leave the MLflow run
        # open as FAILED so a resume reopens it and the UI shows the gap.
        tracker.end("FAILED")
        raise


def _write_summary(manager: CheckpointManager, summary: RunSummary) -> None:
    data = json.dumps(summary.to_dict(), indent=2, default=_json_default).encode()
    atomic_write_bytes(Path(manager.local_dir) / "summary.json", data)
    if manager.remote is not None:
        try:
            manager.remote.put_bytes(data, "summary.json")
        except Exception as err:
            log.warning("summary upload failed: %s", err)


def _json_default(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def evaluate_only(config: TrainingConfig, checkpoint: str | None) -> dict[str, Any]:
    """Score a checkpoint (and the baseline) on the eval set without training."""
    from crosswalk import device as devices
    from crosswalk.models import build_detector

    if not config.data.eval:
        raise SystemExit("data.eval is not set in this config")
    classes = list(config.model.classes or [])
    dev = devices.resolve(config.device)
    _, _, eval_parsed = _datasets(config, classes)
    result: dict[str, Any] = {
        "eval_dataset": hashing.dataset_manifest(eval_parsed, config.data.eval.images, False)
    }
    if checkpoint:
        model = build_detector(config.model)
        state.load_model_weights(model, checkpoint)
        model.to(dev)
        result["model"] = _evaluate(
            model, eval_parsed, config.data.eval.images, config, dev, classes
        )
    if config.eval.baseline:
        result["baseline"] = baseline_evaluate(
            config, eval_parsed, config.data.eval.images, dev, classes
        )
    return result
