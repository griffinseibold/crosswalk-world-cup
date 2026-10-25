"""What a checkpoint contains, and how to put it back.

A checkpoint is everything needed for the resumed run to continue as if it had
never stopped:

- model weights, optimizer state (SGD momentum buffers / AdamW moments),
  LR scheduler state, AMP grad-scaler state;
- progress: global step, epoch, batch index within the epoch;
- RNG state: Python ``random``, NumPy's global generator, the PyTorch CPU
  generator and every GPU generator;
- the class list, config and config hash, so a checkpoint is self-describing
  and resuming with an incompatible config is refused rather than silently
  mixing runs.

The data order and augmentation do not depend on RNG state at all (see
sampling.py and DetectionDataset): they are functions of (seed, epoch, index).
RNG state is still saved so any other randomness (dropout, future
augmentations) resumes faithfully.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

FORMAT_VERSION = 1


@dataclass
class Progress:
    step: int = 0  # optimizer steps completed
    epoch: int = 0  # epoch currently in progress
    batch_in_epoch: int = 0  # batches of `epoch` already done

    def to_dict(self) -> dict[str, int]:
        return {"step": self.step, "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch}


def capture_rng() -> dict[str, Any]:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_gpu": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    return state


def restore_rng(state: dict[str, Any]) -> None:
    import torch

    random.setstate(_as_python_state(state["python"]))
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    gpu = state.get("torch_gpu") or []
    if gpu and torch.cuda.is_available() and len(gpu) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(gpu)


def _as_python_state(value):
    # random.getstate() is (int, tuple, None); serialization can turn the
    # inner tuple into a list, which setstate rejects.
    version, internal, gauss = value
    return (version, tuple(internal), gauss)


def seed_everything(seed: int, deterministic: bool) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every GPU
    if deterministic:
        # warn_only: detection models use ops (e.g. the scatter/index_put in
        # loss computation, upsampling backward) without deterministic GPU
        # kernels. Raising would make training impossible; warning keeps the
        # rest deterministic and logs which ops are not.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False  # MIOpen honours this on ROCm too


def build_state(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    progress: Progress,
    classes: list[str],
    config: dict[str, Any],
    config_hash: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "progress": progress.to_dict(),
        "rng": capture_rng(),
        "classes": classes,
        "config": config,
        "config_hash": config_hash,
        "extra": extra or {},
    }


def save_state(state: dict[str, Any], path: str) -> None:
    import torch

    torch.save(state, path)


def load_state(path: str) -> dict[str, Any]:
    import torch

    # weights_only=False: the file holds optimizer state and Python RNG
    # tuples, not just tensors. Only load checkpoints this project wrote.
    return torch.load(path, map_location="cpu", weights_only=False)


def restore(
    state: dict[str, Any], *, model, optimizer, scheduler, scaler, restore_rng_state: bool = True
) -> Progress:
    if state.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {state.get('format')}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if restore_rng_state:
        restore_rng(state["rng"])
    return Progress(**state["progress"])


def load_model_weights(
    model, uri: str, cache_dir: str = "/tmp/crosswalk-models"
) -> list[str] | None:
    """Load only the weights from a training checkpoint (local path or s3://)
    or a registered model version (``models:/name@alias``).

    Returns the label names (background first) stored with them.
    """
    from crosswalk.registry import load_any

    state = load_any(uri, cache_dir)
    model.load_state_dict(state["model"])
    classes = state.get("classes")
    return ["__background__", *classes] if classes else None
