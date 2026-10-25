"""Epoch order that can be resumed from the middle of an epoch.

PyTorch's ``RandomSampler`` draws its order from the global RNG when an epoch
starts, so resuming mid-epoch would need that exact RNG state *and* a way to
skip the batches already done. Instead, the order of epoch ``e`` is a pure
function of ``(seed, e)``, and the sampler starts ``start_batch`` batches in.
A run resumed at (epoch 3, batch 17) therefore sees exactly the batches the
uninterrupted run would have seen next.

Lightning drives it through three methods:

- ``set_epoch(e)`` at the start of every epoch (Lightning calls it on any
  sampler that has it);
- ``state_dict()`` / ``load_state_dict()`` through the DataLoader, which
  Lightning saves with every checkpoint and restores before it creates the
  first iterator of a resumed run.

``__len__`` always reports the full epoch. Lightning counts the batches of a
resumed epoch from where the checkpoint left off, so a shorter length would
end that epoch early.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def epoch_order(num_samples: int, seed: int, epoch: int, shuffle: bool = True) -> list[int]:
    if not shuffle:
        return list(range(num_samples))
    rng = np.random.default_rng([seed, epoch])
    return rng.permutation(num_samples).tolist()


def steps_per_epoch(num_samples: int, batch_size: int) -> int:
    # drop_last: a final short batch would change the gradients' scale and
    # make step counts depend on dataset size modulo batch size.
    return num_samples // batch_size


class ResumableSampler:
    """Yields indices for one epoch, starting ``start_batch`` batches in.

    Implements the ``__iter__``/``__len__`` protocol torch's DataLoader
    accepts, without importing torch. ``on_epoch`` is called with the epoch
    whenever it changes (the dataset's augmentation depends on it).
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        seed: int,
        shuffle: bool = True,
        on_epoch: Callable[[int], None] | None = None,
    ):
        if num_samples < batch_size:
            raise ValueError(
                f"dataset has {num_samples} images, fewer than batch_size={batch_size}"
            )
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.on_epoch = on_epoch
        self.epoch = 0
        self.start_batch = 0
        # (epoch, batches done) of the last completed batch, recorded just
        # before a checkpoint is written; that is the position to resume at.
        self.marked: tuple[int, int] | None = None

    @property
    def per_epoch(self) -> int:
        return steps_per_epoch(self.num_samples, self.batch_size)

    def set_position(self, epoch: int, start_batch: int = 0) -> None:
        # A position at the very end of an epoch is the start of the next.
        if start_batch >= self.per_epoch:
            epoch, start_batch = epoch + 1, 0
        self.epoch = epoch
        self.start_batch = start_batch
        if self.on_epoch is not None:
            self.on_epoch(epoch)

    def set_epoch(self, epoch: int) -> None:
        # Lightning calls this for the epoch a checkpoint resumed into as
        # well; that epoch keeps its start batch. Any other epoch starts at 0.
        if epoch != self.epoch:
            self.set_position(epoch, 0)

    def mark(self, epoch: int, batches_done: int) -> None:
        self.marked = (epoch, batches_done)

    def state_dict(self) -> dict[str, int]:
        epoch, batch = self.marked if self.marked else (self.epoch, self.start_batch)
        if batch >= self.per_epoch:  # the end of an epoch is the start of the next
            epoch, batch = epoch + 1, 0
        return {"epoch": epoch, "start_batch": batch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.set_position(int(state["epoch"]), int(state["start_batch"]))
        self.marked = None

    def __iter__(self):
        order = epoch_order(self.num_samples, self.seed, self.epoch, self.shuffle)
        usable = self.per_epoch * self.batch_size
        return iter(order[self.start_batch * self.batch_size : usable])

    def __len__(self) -> int:
        return self.per_epoch * self.batch_size
