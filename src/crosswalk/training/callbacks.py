"""Lightning callbacks for what Lightning does not do on its own.

:class:`RecoveryCallback` keeps checkpoints on S3 (with a local fallback, via
:class:`~crosswalk.training.checkpoint.CheckpointManager`) and measures
Experiment B, failure recovery:

- A heartbeat (``heartbeat.json``: last completed step and wall time) is
  written next to the checkpoints every ``log_every_steps`` steps.
- On resume, ``lost_steps`` = heartbeat step - checkpoint step (work redone),
  ``downtime_s`` = resume start - heartbeat time (kill, rescheduling, image
  start), ``restore_s`` = time to find and download the checkpoint, and
  ``recovery_s`` = process start to the first completed step after resuming.
- ``CROSSWALK_FAULT_AT_STEP=N`` makes a *fresh* run SIGKILL itself right after
  step N (never a resumed run, so it cannot loop). On the cluster, deleting
  the pod does the same from outside; see docs/benchmarks.md.

:class:`MetricsCallback` logs the ``system/`` metrics (throughput, step time,
data-loader wait, GPU memory) and scores the validation set at the end of
an epoch.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning.pytorch as pl

from crosswalk import device as devices
from crosswalk.storage import atomic_write_bytes
from crosswalk.training.checkpoint import CheckpointManager
from crosswalk.training.tracking import log_metrics

log = logging.getLogger(__name__)

HEARTBEAT = "heartbeat.json"
FAULT_ENV = "CROSSWALK_FAULT_AT_STEP"


def write_heartbeat(manager: CheckpointManager, step: int, epoch: int) -> None:
    data = json.dumps({"step": step, "epoch": epoch, "time": time.time()}).encode()
    atomic_write_bytes(Path(manager.local_dir) / HEARTBEAT, data)
    if manager.remote is not None:
        # Best effort: a missed heartbeat only blurs Experiment B's numbers.
        with contextlib.suppress(Exception):
            manager.remote.put_bytes(data, HEARTBEAT)


def read_heartbeat(manager: CheckpointManager) -> dict[str, Any] | None:
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


class RecoveryCallback(pl.Callback):
    def __init__(
        self,
        manager: CheckpointManager,
        every_steps: int,
        log_every_steps: int,
        resume_info: dict[str, Any] | None,
        process_start: float,
    ):
        self.manager = manager
        self.every_steps = every_steps
        self.log_every_steps = log_every_steps
        self.resume_info = resume_info  # None for a fresh run; filled in here
        self.process_start = process_start
        self.last_saved = -1
        self.position: tuple[int, int] | None = None
        self.first_step_logged = False

    @property
    def resumed(self) -> bool:
        return self.resume_info is not None

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        step = trainer.global_step
        if self.resumed:
            self.last_saved = step
        # Every rank must have restored the same step, or the replicas diverge
        # from the first update (e.g. a rank on another node without S3).
        chosen = trainer.strategy.broadcast(step, src=0)
        if chosen != step:
            raise RuntimeError(
                f"rank {trainer.global_rank} restored step {step} but rank 0 restored "
                f"{chosen}; give every rank the same checkpoint storage"
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        step = trainer.global_step
        # Lightning's batch_idx counts from the start of the epoch, including
        # batches done before a resume.
        self.position = (trainer.current_epoch, batch_idx + 1)
        if self.resumed and not self.first_step_logged:
            self.first_step_logged = True
            self.resume_info["recovery_s"] = time.monotonic() - self.process_start
            if trainer.is_global_zero:
                log_metrics(
                    trainer.loggers,
                    {
                        f"system/resume_{k}": v
                        for k, v in self.resume_info.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)
                    },
                    step,
                )
        if step % self.log_every_steps == 0 and trainer.is_global_zero:
            write_heartbeat(self.manager, step, trainer.current_epoch)
        if step % self.every_steps == 0 and step != self.last_saved:
            self.save(trainer, pl_module, "periodic")
        self._maybe_inject_fault(step)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_step != self.last_saved:
            self.save(trainer, pl_module, "final")

    def save(self, trainer: pl.Trainer, pl_module, reason: str) -> None:
        step = trainer.global_step
        sampler = getattr(pl_module, "sampler", None)
        if sampler is not None and self.position is not None:
            sampler.mark(*self.position)
        position = sampler.state_dict() if sampler is not None else {}
        if trainer.is_global_zero:
            # Lightning gathers the state on every rank and writes it on rank 0;
            # the manager makes that write atomic, then uploads it.
            result = self.manager.save(
                step,
                trainer.save_checkpoint,
                {
                    "epoch": position.get("epoch"),
                    "batch_in_epoch": position.get("start_batch"),
                    "reason": reason,
                },
            )
            log_metrics(
                trainer.loggers,
                {
                    "system/checkpoint_serialize_s": result.serialize_s,
                    "system/checkpoint_upload_s": result.upload_s,
                    "system/checkpoint_bytes": result.bytes,
                    "system/checkpoint_upload_failures": self.manager.upload_failures,
                },
                step,
            )
            log.info("checkpoint step %d (%s, %.1f MB)", step, reason, result.bytes / 1e6)
        else:
            # Same call, so every rank reaches Lightning's collective state
            # gathering and barrier; only rank 0 writes anything.
            trainer.save_checkpoint(os.path.join(tempfile.gettempdir(), "crosswalk-unused.ckpt"))
        # Nobody runs ahead while rank 0 uploads.
        trainer.strategy.barrier("crosswalk-checkpoint")
        self.last_saved = step

    def _maybe_inject_fault(self, step: int) -> None:
        target = os.environ.get(FAULT_ENV)
        if target and not self.resumed and step == int(target):
            log.warning("fault injection: SIGKILL at step %d", step)
            logging.shutdown()
            os.kill(os.getpid(), signal.SIGKILL)


class MetricsCallback(pl.Callback):
    def __init__(
        self,
        log_every_steps: int,
        steps_per_epoch: int,
        validate: Callable[[Any], dict[str, float]] | None = None,
        validate_every_epochs: int = 1,
        expected_world_size: int = 1,
    ):
        self.log_every_steps = log_every_steps
        self.steps_per_epoch = steps_per_epoch
        self.validate = validate
        self.validate_every_epochs = validate_every_epochs
        self.expected_world_size = expected_world_size
        self._reset_window()
        self._last_batch_end: float | None = None
        self._batches_done = 0

    def _reset_window(self) -> None:
        self._window_t0 = time.monotonic()
        self._window_images = 0
        self._window_data_wait = 0.0

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if trainer.world_size != self.expected_world_size:
            raise RuntimeError(
                f"Lightning runs {trainer.world_size} processes but the environment says "
                f"{self.expected_world_size}; the step count and resume check assume the latter"
            )
        backend = ""
        if trainer.world_size > 1:
            import torch.distributed as torch_dist

            backend = f" ({torch_dist.get_backend()})" if torch_dist.is_initialized() else ""
            log.info("rank %d/%d%s", trainer.global_rank, trainer.world_size, backend)

    def on_train_start(self, trainer, pl_module) -> None:
        self._reset_window()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        if self._last_batch_end is not None:
            self._window_data_wait += time.monotonic() - self._last_batch_end

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        self._batches_done = batch_idx + 1
        self._window_images += len(batch[0])
        step = trainer.global_step
        if step % self.log_every_steps == 0 and trainer.is_global_zero:
            devices.synchronize(pl_module.device)
            elapsed = time.monotonic() - self._window_t0
            steps = max(1, self._window_images // max(1, len(batch[0])))
            log_metrics(
                trainer.loggers,
                {
                    # All processes together (each sees its own slice).
                    "system/images_per_s": self._window_images * trainer.world_size / elapsed
                    if elapsed > 0
                    else None,
                    "system/world_size": trainer.world_size,
                    "system/step_s": elapsed / steps,
                    "system/data_wait_fraction": self._window_data_wait / elapsed
                    if elapsed > 0
                    else None,
                    "system/gpu_memory_peak_bytes": devices.max_memory_allocated(pl_module.device),
                    "system/epoch": trainer.current_epoch,
                },
                step,
            )
            log.info("step %d epoch %d", step, trainer.current_epoch)
            self._reset_window()
        self._last_batch_end = time.monotonic()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._last_batch_end = None
        completed = self._batches_done >= self.steps_per_epoch
        epochs_done = trainer.current_epoch + 1
        if not (completed and self.validate and epochs_done % self.validate_every_epochs == 0):
            return
        if trainer.is_global_zero:
            from crosswalk.training import evaluate

            metrics = self.validate(pl_module.model)
            log_metrics(
                trainer.loggers,
                dict(evaluate.finite(evaluate.prefixed(metrics, "quality/val").items())),
                trainer.global_step,
            )
        trainer.strategy.barrier("crosswalk-validation")  # the others wait for rank 0
