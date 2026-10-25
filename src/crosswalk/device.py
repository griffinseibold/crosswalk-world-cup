"""Choose the compute device and describe it.

PyTorch's ROCm build exposes AMD GPUs through the ``torch.cuda`` API (HIP is
mapped onto it), so the rest of the code only ever sees ``torch.device("cuda")``
or ``torch.device("cpu")`` and never branches on the vendor. The vendor shows up
in exactly two places: the image the chart runs, and the descriptive fields
returned by :func:`describe`.

Config values:

- ``auto``: a GPU if PyTorch can see one, else the CPU.
- ``gpu``: a GPU from either vendor; fail if there is none.
- ``rocm`` / ``cuda``: a GPU *and* a PyTorch build for that vendor; fail
  otherwise. Useful in validation jobs, where silently falling back to the CPU
  would make a broken GPU setup look like a slow one.
- ``cpu``: always the CPU.
"""

from __future__ import annotations

from typing import Any

from crosswalk import hardware


class DeviceUnavailable(RuntimeError):
    pass


def gpu_backend() -> str | None:
    """'rocm', 'cuda' or None for the installed PyTorch build."""
    import torch

    if getattr(torch.version, "hip", None):
        return "rocm"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return None


def resolve(name: str = "auto"):
    import torch

    if name == "cpu":
        return torch.device("cpu")
    available = torch.cuda.is_available()
    backend = gpu_backend()
    if name == "auto":
        return torch.device("cuda" if available else "cpu")
    if not available:
        raise DeviceUnavailable(
            f"device {name!r} requested but PyTorch sees no GPU (build: {backend or 'cpu-only'}). "
            "Check that the pod has an amd.com/gpu limit and that /dev/kfd and /dev/dri are present."
        )
    if name in ("rocm", "cuda") and backend != name:
        raise DeviceUnavailable(f"device {name!r} requested but this PyTorch is a {backend} build")
    return torch.device("cuda")


def synchronize(device) -> None:
    """Wait for queued GPU work so wall-clock timings include it."""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_allocated(device) -> int | None:
    import torch

    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    return None


def max_memory_allocated(device) -> int | None:
    import torch

    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def describe(device=None) -> dict[str, Any]:
    """Host facts plus PyTorch, HIP/CUDA and GPU details for result records."""
    import torch
    import torchvision

    info: dict[str, Any] = dict(hardware.collect())
    info.update(
        {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "torch_hip": getattr(torch.version, "hip", None),
            "torch_cuda": getattr(torch.version, "cuda", None),
            "gpu_backend": gpu_backend(),
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device": str(device) if device is not None else None,
            "torch_num_threads": torch.get_num_threads(),
        }
    )
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_memory_bytes": int(props.total_memory),
                # gcnArchName (e.g. "gfx1030") exists on ROCm builds only.
                "gpu_arch": getattr(props, "gcnArchName", None)
                or f"sm_{props.major}{props.minor}",
                "gpu_multiprocessors": props.multi_processor_count,
            }
        )
    try:
        info["miopen"] = torch.backends.cudnn.version() if gpu_backend() == "rocm" else None
        info["cudnn"] = torch.backends.cudnn.version() if gpu_backend() == "cuda" else None
    except Exception:  # pragma: no cover - depends on the build
        pass
    return info
