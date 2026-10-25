"""One training job: fine-tune the detector with PyTorch Lightning.

Lightning's ``Trainer`` runs the loop (mixed precision, gradient clipping,
optimizer and scheduler steps, DistributedDataParallel under torchrun, and
saving and restoring the full training state). Around it, this module:

- loads the datasets and records their content hashes;
- finds the newest checkpoint (S3 first, local fallback), refuses to resume
  one written with an incompatible config, and hands it to ``Trainer.fit``;
- sets up MLflow tracking on the same run as before a crash;
- after training, scores the fine-tuned model and the pretrained baseline on
  the fixed-camera evaluation set, registers the model and writes a summary.

It is safe to kill at any moment and run again with the same config: the
second run resumes from the newest checkpoint on the exact batch after it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crosswalk.config import ModelConfig, TrainingConfig, config_hash
from crosswalk.storage import atomic_write_bytes
from crosswalk.training import datasets, distributed, evaluate, hashing, state, tracking
from crosswalk.training.checkpoint import CheckpointManager, make_manager
from crosswalk.training.sampling import steps_per_epoch

log = logging.getLogger(__name__)

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


def resume_hash(config: TrainingConfig, world_size: int = 1) -> str:
    subset = {k: v for k, v in config.model_dump(mode="json").items() if k in RESUME_KEYS}
    if world_size > 1:
        # The global batch is batch_size * world_size: resuming with another
        # process count would silently change it.
        subset["world_size"] = world_size
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


def _datasets(config: TrainingConfig, classes: list[str]):
    train = datasets.load(config.data.train, classes, config.seed)
    val = datasets.load(config.data.val, classes, config.seed) if config.data.val else None
    ev = datasets.load(config.data.eval, classes, config.seed) if config.data.eval else None
    return train, val, ev


def score(model, parsed, image_root, config: TrainingConfig, device, classes) -> dict[str, float]:
    """COCO mAP of ``model`` on a labelled set (pycocotools)."""
    model.to(device)
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


def baseline_score(config: TrainingConfig, parsed, image_root, device, classes) -> dict[str, float]:
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


def _find_checkpoint(config: TrainingConfig, manager: CheckpointManager, rhash: str):
    """The checkpoint to resume from (or None), after the resume policy checks."""
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
    if found is None:
        return None, {}
    meta = state.metadata(state.load_checkpoint(str(found.path)))
    if meta.get("resume_hash") != rhash:
        raise ResumeMismatch(
            f"checkpoint step {found.step} was written with a different training config "
            "(seed, model, data, optimizer, schedule, batch size or number of processes changed); "
            "use a new run_name"
        )
    return found, meta


def train(config: TrainingConfig, *, s3_client=None, logger=None) -> RunSummary:
    """Run (or resume) one training job. ``logger``: a Lightning logger to use
    instead of MLflow (tests)."""
    import lightning.pytorch as pl

    from crosswalk import device as devices
    from crosswalk.models import build_detector
    from crosswalk.training.callbacks import MetricsCallback, RecoveryCallback, read_heartbeat
    from crosswalk.training.module import DetectorModule

    process_start = time.monotonic()
    wall_start = time.time()
    classes = list(config.model.classes or [])
    pl.seed_everything(config.seed, verbose=False)
    dev = devices.resolve(config.device)  # fails early if a GPU was asked for and is missing
    # torchrun sets RANK/WORLD_SIZE; Lightning picks them up too, but the
    # step count and the resume check need the world size first.
    dist = distributed.from_env()
    main = dist.is_main
    runtime = devices.describe(dev)
    if dist.enabled:
        runtime["world_size"] = dist.world_size
    log.info(
        "device %s (%s)%s", dev, runtime.get("gpu_name") or "cpu",
        f", rank {dist.rank}/{dist.world_size}" if dist.enabled else "",
    )  # fmt: skip

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
    global_batch = config.batch_size * dist.world_size
    if len(train_ds) < global_batch:
        raise ValueError(
            f"dataset has {len(train_ds)} images, fewer than one global batch ({global_batch})"
        )
    per_epoch = steps_per_epoch(len(train_ds), global_batch)
    total_steps = per_epoch * config.epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    # -- checkpoint to resume from ------------------------------------------------
    local_run_dir = str(Path(config.checkpoint.local_dir) / config.run_name)
    manager, storage_desc = make_manager(
        config.run_name,
        config.checkpoint.s3_uri,
        config.checkpoint.local_dir,
        config.checkpoint.keep_last,
        s3_client=s3_client,
        # Other ranks download into their own directory (never the same file
        # as rank 0); local checkpoints they read from rank 0's directory.
        download_dir=None if main else str(Path(local_run_dir) / f".rank-{dist.rank}"),
    )
    log.info("checkpoints: %s", storage_desc)
    heartbeat = read_heartbeat(manager)
    restore_t0 = time.monotonic()
    rhash = resume_hash(config, dist.world_size)
    found, meta = _find_checkpoint(config, manager, rhash)
    resume_info: dict[str, Any] | None = None
    if found is not None:
        lost = downtime = None
        if heartbeat and heartbeat.get("step", -1) >= found.step:
            lost = int(heartbeat["step"]) - found.step
            downtime = wall_start - float(heartbeat["time"])
        resume_info = {
            "checkpoint_step": found.step,
            "checkpoint_source": found.source,
            "heartbeat_step": heartbeat.get("step") if heartbeat else None,
            "lost_steps": lost,
            "downtime_s": downtime,
            "restore_s": time.monotonic() - restore_t0,
        }
        log.info("resuming from step %d (%s): %s", found.step, found.source, resume_info)

    # -- tracking ----------------------------------------------------------------
    if logger is None and config.tracking.enabled and main:
        # One MLflow run per training run, reopened on resume; only rank 0 tracks.
        logger = tracking.make_logger(
            config.tracking.experiment, config.run_name, meta.get("mlflow_run_id")
        )
    if logger is not None and main:
        if found is None:
            logger.log_hyperparams(config.model_dump(mode="json"))
            tracking.set_tags(
                logger,
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
                },
            )
            tracking.log_dict(logger, manifest, "dataset/train_manifest.json")
            if eval_manifest:
                tracking.log_dict(logger, eval_manifest, "dataset/eval_manifest.json")
            tracking.log_dict(logger, runtime, "system/runtime.json")
        else:
            tracking.log_dict(logger, runtime, f"system/runtime-resume-step-{found.step}.json")

    # -- model ---------------------------------------------------------------------
    model = build_detector(config.model, for_training=True)
    if config.model.checkpoint and found is None:
        # Initialise from an earlier run's weights (not a resume).
        state.load_model_weights(model, config.model.checkpoint)
    module = DetectorModule(
        config,
        model,
        train_ds,
        total_steps,
        {
            "classes": classes,
            "model_config": config.model.model_dump(mode="json"),
            "config": config.model_dump(mode="json"),
            "config_hash": config_hash(config),
            "resume_hash": rhash,
            "mlflow_run_id": tracking.run_id(logger),
        },
    )

    def validate(m) -> dict[str, float]:
        return score(m, val_parsed, config.data.val.images, config, dev, classes)

    recovery = RecoveryCallback(
        manager,
        config.checkpoint.every_steps,
        config.log_every_steps,
        resume_info,
        process_start,
    )
    on_gpu = dev.type == "cuda"
    trainer = pl.Trainer(
        accelerator="gpu" if on_gpu else "cpu",
        devices=distributed.local_world_size(),
        num_nodes=max(1, dist.world_size // distributed.local_world_size()),
        strategy="ddp" if dist.enabled else "auto",
        precision="16-mixed" if config.amp and on_gpu else "32-true",
        max_epochs=config.epochs,
        max_steps=total_steps,
        gradient_clip_val=config.grad_clip_norm,
        gradient_clip_algorithm="norm",
        # Warn-only determinism: some detection ops have no deterministic GPU
        # kernel; raising would make training impossible.
        deterministic="warn" if config.deterministic else False,
        logger=logger if logger is not None else False,
        log_every_n_steps=config.log_every_steps,
        callbacks=[
            recovery,
            MetricsCallback(
                config.log_every_steps,
                per_epoch,
                validate if val_parsed else None,
                config.eval.every_epochs,
                dist.world_size,
            ),
        ],
        # Checkpoints are RecoveryCallback's (S3 + local fallback + pointer).
        enable_checkpointing=False,
        # No validation loop: validation is a COCO-mAP pass at epoch end.
        limit_val_batches=0,
        num_sanity_val_steps=0,
        # The resumable sampler shards global batches itself (distributed.py).
        use_distributed_sampler=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tempfile.mkdtemp(prefix="crosswalk-lightning-"),
    )
    trainer.fit(module, ckpt_path=str(found.path) if found else None)

    summary = RunSummary(
        run_name=config.run_name,
        final_step=trainer.global_step,
        total_steps=total_steps,
        resumed=found is not None,
        resume=resume_info,
        checkpoint_storage=storage_desc,
        dataset_sha256=manifest["sha256"],
        mlflow_run_id=tracking.run_id(logger),
        upload_failures=manager.upload_failures,
    )
    if not trainer.is_global_zero:
        # Evaluation, registration and the summary are rank 0's.
        return summary

    # -- final evaluation: fine-tuned vs pretrained baseline --------------------------
    loggers = trainer.loggers
    if eval_parsed:
        summary.eval = score(model, eval_parsed, config.data.eval.images, config, dev, classes)
        tracking.log_metrics(
            loggers,
            evaluate.finite(evaluate.prefixed(summary.eval, "quality/eval").items()),
            summary.final_step,
        )
        if config.eval.baseline and config.model.name == "fcos_resnet50_fpn":
            summary.baseline_eval = baseline_score(
                config, eval_parsed, config.data.eval.images, dev, classes
            )
            tracking.log_metrics(
                loggers,
                evaluate.finite(
                    evaluate.prefixed(summary.baseline_eval, "quality/baseline").items()
                ),
                summary.final_step,
            )
    tracking.log_dict(logger, json.loads(_dumps(summary)), "run_summary.json")
    if config.tracking.registered_model_name:
        tracking.register_model(
            logger,
            model,
            config.tracking.registered_model_name,
            {
                "classes": classes,
                "dataset_sha256": manifest["sha256"],
                "eval_map": summary.eval.get("map"),
                "step": summary.final_step,
                "run_name": config.run_name,
            },
            model_config=config.model.model_dump(mode="json"),
        )
    _write_summary(manager, summary)
    return summary


def _dumps(summary: RunSummary) -> str:
    return json.dumps(summary.to_dict(), indent=2, default=_json_default)


def _write_summary(manager: CheckpointManager, summary: RunSummary) -> None:
    data = _dumps(summary).encode()
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
        result["model"] = score(model, eval_parsed, config.data.eval.images, config, dev, classes)
    if config.eval.baseline:
        result["baseline"] = baseline_score(
            config, eval_parsed, config.data.eval.images, dev, classes
        )
    return result
