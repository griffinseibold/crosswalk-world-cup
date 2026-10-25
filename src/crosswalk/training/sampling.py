"""Epoch order that can be resumed from the middle of an epoch.

PyTorch's ``RandomSampler`` draws its order from the global RNG when an epoch
starts, so resuming mid-epoch would need that exact RNG state *and* a way to
skip the batches already done. Instead, the order of epoch ``e`` is a pure
function of ``(seed, e)``, and the sampler starts ``offset`` samples in. A run
resumed at (epoch 3, batch 17) therefore sees exactly the batches the
uninterrupted run would have seen next.
"""

from __future__ import annotations

import numpy as np


def epoch_order(num_samples: int, seed: int, epoch: int, shuffle: bool = True) -> list[int]:
    if not shuffle:
        return list(range(num_samples))
    rng = np.random.default_rng([seed, epoch])
    return rng.permutation(num_samples).tolist()


def steps_per_epoch(num_samples: int, batch_size: int) -> int:
    # drop_last: a final short batch would change BatchNorm-free gradients'
    # scale and make step counts depend on dataset size modulo batch size.
    return num_samples // batch_size


class ResumableSampler:
    """Yields indices for one epoch, starting ``start_batch`` batches in.

    Implements the ``__iter__``/``__len__`` protocol torch's DataLoader
    accepts, without importing torch.
    """

    def __init__(self, num_samples: int, batch_size: int, seed: int, shuffle: bool = True):
        if num_samples < batch_size:
            raise ValueError(
                f"dataset has {num_samples} images, fewer than batch_size={batch_size}"
            )
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.start_batch = 0

    def set_position(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch = epoch
        self.start_batch = start_batch

    def __iter__(self):
        order = epoch_order(self.num_samples, self.seed, self.epoch, self.shuffle)
        usable = steps_per_epoch(self.num_samples, self.batch_size) * self.batch_size
        return iter(order[self.start_batch * self.batch_size : usable])

    def __len__(self) -> int:
        return (
            steps_per_epoch(self.num_samples, self.batch_size) - self.start_batch
        ) * self.batch_size
