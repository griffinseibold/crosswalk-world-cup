"""Facts about the machine a run happened on, recorded next to every result.

Benchmark numbers are only comparable when the hardware, driver and library
versions behind them are known, so every benchmark, validation and training
run stores this dictionary. Everything here reads files or the standard
library; the PyTorch half lives in :mod:`crosswalk.device`.

Vendor-specific probing (AMD sysfs paths, ``/opt/rocm``, NVIDIA's ``/proc``
driver file) is confined to this module and ``device.py``.
"""

from __future__ import annotations

import os
import platform
import socket
from pathlib import Path


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip() or None
    except OSError:
        return None


def cpu_model() -> str | None:
    text = _read("/proc/cpuinfo")
    if text:
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def memory_total_bytes() -> int | None:
    text = _read("/proc/meminfo")
    if text:
        for line in text.splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return None


def cgroup_memory_limit_bytes() -> int | None:
    """The container's memory limit, which matters more than host RAM in a pod."""
    raw = _read("/sys/fs/cgroup/memory.max")
    if raw and raw != "max":
        return int(raw)
    return None


def rocm_userspace_version() -> str | None:
    # The ROCm libraries inside the image, not the host kernel driver.
    return _read("/opt/rocm/.info/version")


def amdgpu_driver_version() -> str | None:
    # Present for AMD's packaged (DKMS) driver; the in-tree kernel driver has
    # no version file, in which case the kernel release identifies it.
    return _read("/sys/module/amdgpu/version")


def nvidia_driver_version() -> str | None:
    text = _read("/proc/driver/nvidia/version")
    return text.splitlines()[0] if text else None


def collect() -> dict[str, object]:
    return {
        "hostname": socket.gethostname(),
        "node_name": os.environ.get("NODE_NAME"),
        "kernel": platform.release(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": memory_total_bytes(),
        "cgroup_memory_limit_bytes": cgroup_memory_limit_bytes(),
        "rocm_userspace_version": rocm_userspace_version(),
        "amdgpu_driver_version": amdgpu_driver_version(),
        "nvidia_driver_version": nvidia_driver_version(),
        # Set by the chart and the image build so results name the exact image.
        "container_image": os.environ.get("CROSSWALK_IMAGE"),
        "git_commit": os.environ.get("CROSSWALK_GIT_COMMIT"),
        "hsa_override_gfx_version": os.environ.get("HSA_OVERRIDE_GFX_VERSION"),
    }
