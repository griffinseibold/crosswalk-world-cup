"""The detector as a Lightning module.

Lightning runs the training loop: mixed precision, gradient clipping,
optimizer and LR-scheduler steps, DistributedDataParallel, and saving and
restoring model, optimizer, scheduler and loop state. This module says only
what a training step is, how the optimizer and schedule are built, and how
batches are drawn, so that a checkpoint resumes on the exact next batch.
"""

from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from crosswalk.config import TrainingConfig
from crosswalk.training import datasets, state
from crosswalk.training.distributed import DistInfo, ShardedSampler
from crosswalk.training.sampling import ResumableSampler
from crosswalk.training.schedule import WarmupCosine


class ResumableDataLoader(DataLoader):
    """A DataLoader whose position Lightning saves with every checkpoint.

    Lightning treats any loader with ``state_dict``/``load_state_dict`` as
    resumable and restores it before creating the first iterator of a resumed
    run; the position itself lives in the sampler.
    """

    def state_dict(self) -> dict[str, int]:
        return self.sampler.state_dict()

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.sampler.load_state_dict(state_dict)


class DetectorModule(pl.LightningModule):
    def __init__(
        self,
        config: TrainingConfig,
        model,
        train_dataset: datasets.DetectionDataset,
        total_steps: int,
        checkpoint_meta: dict[str, Any],
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.total_steps = total_steps
        # Written into every checkpoint (see state.py): classes, configs,
        # resume hash, MLflow run id.
        self.checkpoint_meta = checkpoint_meta
        self.sampler: ResumableSampler | ShardedSampler | None = None

    # -- data -------------------------------------------------------------------

    def train_dataloader(self) -> ResumableDataLoader:
        world = self.trainer.world_size
        # The order of *global* batches (batch_size per process x processes);
        # each process takes its slice. With one process both are the same.
        base = ResumableSampler(
            len(self.train_dataset),
            self.config.batch_size * world,
            self.config.seed,
            on_epoch=self.train_dataset.set_epoch,
        )
        sampler = base
        if world > 1:
            info = DistInfo(self.trainer.global_rank, self.trainer.local_rank, world)
            sampler = ShardedSampler(base, info)
        self.sampler = sampler
        return ResumableDataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.data.num_workers,
            collate_fn=datasets.collate,
            drop_last=True,
            pin_memory=self.device.type == "cuda",
        )

    # -- training ---------------------------------------------------------------

    def training_step(self, batch, batch_idx: int):
        images, targets = batch
        targets = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        losses = self.model(images, targets)
        loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss {loss.item()} at step {self.global_step}; "
                "the last checkpoint is intact, lower the learning rate and resume"
            )
        size = len(images)
        self.log("quality/loss", loss, batch_size=size)
        for name, value in losses.items():
            self.log(f"quality/{name}", value, batch_size=size)
        self.log("quality/lr", self.lr_schedulers().get_last_lr()[0], batch_size=size)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        # Before clipping (and after AMP unscaling): the norm clipping acts on.
        from lightning.pytorch.utilities import grad_norm

        norms = grad_norm(self.model, norm_type=2)
        if "grad_2.0_norm_total" in norms:
            self.log("quality/grad_norm", norms["grad_2.0_norm_total"])

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = self.config.optimizer
        if opt.name == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=opt.weight_decay)
        schedule = WarmupCosine(
            self.total_steps,
            self.config.scheduler.warmup_steps,
            self.config.scheduler.min_lr_ratio,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # -- checkpoints ------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        position = self.sampler.state_dict() if self.sampler is not None else {}
        checkpoint[state.META_KEY] = {
            **self.checkpoint_meta,
            "format": state.FORMAT_VERSION,
            "progress": {
                "step": self.global_step,
                "epoch": position.get("epoch"),
                "batch_in_epoch": position.get("start_batch"),
            },
            "rng": state.capture_rng(),
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        meta = checkpoint.get(state.META_KEY) or {}
        if "rng" in meta:
            state.restore_rng(meta["rng"])
