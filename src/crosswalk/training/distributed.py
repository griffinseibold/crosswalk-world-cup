"""Data-parallel training across GPUs: what Lightning does not decide for us.

``torchrun --nproc-per-node=N -m crosswalk train --config ...`` starts N
processes. Lightning's DDP strategy picks up torchrun's environment, starts
the process group (NCCL on GPUs, which is RCCL on ROCm builds; gloo on CPU),
wraps the model in DistributedDataParallel and averages gradients. Two
things stay here:

- :func:`from_env` reads the world size before Lightning starts, because the
  number of steps and the resume check depend on the global batch;
- :class:`ShardedSampler` splits each *global* batch of the resumable epoch
  order between the ranks, so a checkpoint's (epoch, batch) position means
  the same on every rank and on resume. (Lightning's own
  ``DistributedSampler`` would shuffle independently of that order, so the
  trainer runs with ``use_distributed_sampler=False``.)

``batch_size`` in the config stays the **per-process** batch; the global batch
is ``batch_size * world_size``, and the learning rate is not rescaled. The
resume hash includes the world size, so a run cannot silently resume with a
different global batch.
"""

from __future__ import annotations

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


def local_world_size(env: dict[str, str] | None = None) -> int:
    """Processes on this node (torchrun's LOCAL_WORLD_SIZE), 1 without torchrun."""
    env = os.environ if env is None else env
    return int(env.get("LOCAL_WORLD_SIZE", env.get("WORLD_SIZE", "1")))


class ShardedSampler:
    """Rank ``rank``'s slice of each global batch of a ResumableSampler.

    Global batch b covers indices [b*G, (b+1)*G) of the epoch order (G =
    per-rank batch * world); rank r takes [b*G + r*B, b*G + (r+1)*B). Epoch,
    position and checkpoint state are the base sampler's.
    """

    def __init__(self, base, info: DistInfo):
        self.base = base
        self.info = info
        self.per_rank = base.batch_size // info.world_size

    @property
    def per_epoch(self) -> int:
        return self.base.per_epoch

    def set_position(self, epoch: int, start_batch: int = 0) -> None:
        self.base.set_position(epoch, start_batch)

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def mark(self, epoch: int, batches_done: int) -> None:
        self.base.mark(epoch, batches_done)

    def state_dict(self) -> dict[str, int]:
        return self.base.state_dict()

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.base.load_state_dict(state)

    def __iter__(self):
        order = list(iter(self.base))
        g, b, r = self.base.batch_size, self.per_rank, self.info.rank
        for start in range(0, len(order) - g + 1, g):
            yield from order[start + r * b : start + (r + 1) * b]

    def __len__(self) -> int:
        return len(self.base) // self.info.world_size
