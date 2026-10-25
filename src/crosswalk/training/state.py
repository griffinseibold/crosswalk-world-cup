"""What a training checkpoint contains, and how to read the weights back out.

Checkpoints are Lightning checkpoints (``trainer.save_checkpoint``): model
weights under ``state_dict`` (keys prefixed ``model.``), optimizer and LR
scheduler state, AMP scaler state, Lightning's loop progress, and the
DataLoader position (see sampling.py). The detector module adds one entry,
``checkpoint["crosswalk"]``:

- ``classes``, ``model_config``, ``config``, ``config_hash``: so a checkpoint
  is self-describing and loads into a plain detector for inference;
- ``resume_hash``: resuming with an incompatible config is refused rather
  than silently mixing runs;
- ``mlflow_run_id``: a resumed run reattaches to the same MLflow run;
- ``progress``: step, epoch and batch within the epoch, for people reading it;
- ``rng``: Python, NumPy and PyTorch RNG state. Data order and augmentation
  do not depend on it (they are functions of seed, epoch and index), but any
  other randomness resumes faithfully.

Checkpoints written before the move to Lightning (``format`` 1, weights under
``model``) still load for inference and evaluation.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

META_KEY = "crosswalk"
FORMAT_VERSION = 2
MODEL_PREFIX = "model."


def capture_rng() -> dict[str, Any]:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_gpu": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


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


def load_checkpoint(path: str) -> dict[str, Any]:
    import torch

    # weights_only=False: the file holds optimizer state and Python RNG
    # tuples, not just tensors. Only load checkpoints this project wrote.
    return torch.load(path, map_location="cpu", weights_only=False)


def metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The project's entry of a checkpoint (empty for foreign files)."""
    if META_KEY in checkpoint:
        return checkpoint[META_KEY]
    if checkpoint.get("format") == 1:  # pre-Lightning checkpoints
        return {**checkpoint.get("extra", {}), "classes": checkpoint.get("classes")}
    return {}


def weights(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """``{"model": detector state dict, "classes": [...]}`` from any checkpoint."""
    if "state_dict" in checkpoint:
        model = {
            k[len(MODEL_PREFIX) :]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith(MODEL_PREFIX)
        }
        meta = metadata(checkpoint)
        return {
            "model": model,
            "classes": meta.get("classes"),
            "model_config": meta.get("model_config"),
        }
    if checkpoint.get("format") == 1:
        return {"model": checkpoint["model"], "classes": checkpoint.get("classes")}
    raise ValueError("not a crosswalk training checkpoint (no state_dict and no format 1 weights)")


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
