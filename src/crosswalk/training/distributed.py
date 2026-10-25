"""Data-parallel training across GPUs (DistributedDataParallel), launched by torchrun.

``torchrun --nproc-per-node=N -m crosswalk train --config ...`` starts N
processes; each sees ``RANK``, ``LOCAL_RANK`` and ``WORLD_SIZE`` and trains a
replica on its own slice of every batch, with gradients averaged across
processes after each backward pass. Without those variables (plain
``crosswalk train``) nothing here does anything and the single-process path
is exactly what it was.

Backend by device, never by vendor:

- GPU: ``nccl``. On ROCm builds PyTorch's ``nccl`` backend is RCCL (AMD's
  implementation of the same API), so the same code runs on both.
- CPU: ``gloo``, which is how CI tests this module.

What changes with N processes, and what does not:

- ``batch_size`` in the config stays the **per-process** batch; the global
  batch is ``batch_size * world_size`` (the LR is not rescaled automatically:
  set it for the global batch you intend). The resume hash includes the
  world size, so a run cannot silently resume with a different global batch.
- The data order is the single-process order of global batches; rank r takes
  the r-th slice of each (``ShardedSampler``), so a checkpoint's (epoch,
  batch) position means the same on every rank and on resume.
- Only rank 0 writes checkpoints, heartbeats, the summary and MLflow; the
  others wait at a barrier so nobody runs ahead of a checkpoint.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def from_env(env: dict[str, str] | None = None) -> DistInfo:
    env = os.environ if env is None else env
    world = int(env.get("WORLD_SIZE", "1"))
    if world <= 1:
        return DistInfo()
    return DistInfo(int(env["RANK"]), int(env.get("LOCAL_RANK", env["RANK"])), world)


def backend_for(device_type: str) -> str:
    return "nccl" if device_type == "cuda" else "gloo"


def init(info: DistInfo, device_type: str, timeout_s: float = 1800.0) -> str | None:
    """Start the process group (torchrun has set MASTER_ADDR/PORT). Returns
    the backend, or None when not distributed."""
    if not info.enabled:
        return None
    import torch
    import torch.distributed as torch_dist

    if device_type == "cuda":
        # One process per GPU: bind before any CUDA/HIP work.
        torch.cuda.set_device(info.local_rank)
    backend = backend_for(device_type)
    if not torch_dist.is_initialized():
        torch_dist.init_process_group(backend, timeout=dt.timedelta(seconds=timeout_s))
    return backend


def wrap(model, info: DistInfo, device_type: str):
    if not info.enabled:
        return model
    from torch.nn.parallel import DistributedDataParallel

    if device_type == "cuda":
        return DistributedDataParallel(
            model, device_ids=[info.local_rank], output_device=info.local_rank
        )
    return DistributedDataParallel(model)


def unwrap(model):
    """The underlying module (for state_dict, evaluation and registration)."""
    return getattr(model, "module", model)


def barrier(info: DistInfo) -> None:
    if info.enabled:
        import torch.distributed as torch_dist

        torch_dist.barrier()


def broadcast_object(value, info: DistInfo):
    """Rank 0's value on every rank."""
    if not info.enabled:
        return value
    import torch.distributed as torch_dist

    box = [value if info.is_main else None]
    torch_dist.broadcast_object_list(box, src=0)
    return box[0]


def shutdown(info: DistInfo) -> None:
    if info.enabled:
        import torch.distributed as torch_dist

        if torch_dist.is_initialized():
            torch_dist.destroy_process_group()


class ShardedSampler:
    """Rank ``rank``'s slice of each global batch of a ResumableSampler.

    Global batch b covers indices [b*G, (b+1)*G) of the epoch order (G =
    per-rank batch * world); rank r takes [b*G + r*B, b*G + (r+1)*B).
    ``set_position`` takes the same (epoch, global batch) as the base sampler.
    """

    def __init__(self, base, info: DistInfo):
        self.base = base
        self.info = info
        self.per_rank = base.batch_size // info.world_size

    def set_position(self, epoch: int, start_batch: int = 0) -> None:
        self.base.set_position(epoch, start_batch)

    def __iter__(self):
        order = list(iter(self.base))
        g, b, r = self.base.batch_size, self.per_rank, self.info.rank
        for start in range(0, len(order) - g + 1, g):
            yield from order[start + r * b : start + (r + 1) * b]

    def __len__(self) -> int:
        return len(self.base) // self.info.world_size
