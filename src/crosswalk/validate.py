"""GPU check: prove PyTorch can train a TorchVision detector on this GPU.

Run before anything else on new hardware or a new image. It needs no dataset
and no downloads: a randomly initialised FCOS-ResNet50-FPN takes a few
training steps on random tensors shaped like 720p frames. If this works, the
ROCm runtime, MIOpen convolutions, the backward pass and the optimizer all
work on the card; if it fails, nothing later will.

It prints (and optionally writes as JSON):

- device, GPU name and architecture (gfx1030 for the RX 6900 XT), VRAM;
- PyTorch, TorchVision and HIP versions, ROCm userspace and driver versions;
- the first step's time (includes MIOpen kernel selection) separately from the
  steady-state steps, training images/s, and peak GPU memory;
- a small matmul check comparing the GPU against the CPU result.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

log = logging.getLogger(__name__)


def run(
    device_name: str = "gpu",
    steps: int = 10,
    batch_size: int = 2,
    height: int = 720,
    width: int = 1280,
    model_name: str = "fcos_resnet50_fpn",
    seed: int = 0,
) -> dict[str, Any]:
    import torch

    from crosswalk import device as devices
    from crosswalk.config import ModelConfig
    from crosswalk.models import build_detector

    torch.manual_seed(seed)
    dev = devices.resolve(device_name)
    info = devices.describe(dev)

    # Numerical sanity: a wrong-architecture kernel can "succeed" with garbage.
    a = torch.randn(256, 256)
    b = torch.randn(256, 256)
    expected = a @ b
    got = (a.to(dev) @ b.to(dev)).cpu()
    matmul_max_error = float((expected - got).abs().max())

    config = ModelConfig(
        name=model_name,
        weights="none",
        classes=["car", "pedestrian"],
        min_size=height,
        max_size=max(width, height),
    )
    model = build_detector(config, for_training=True).to(dev).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def batch():
        images = [torch.rand(3, height, width, device=dev) for _ in range(batch_size)]
        targets = []
        for _ in range(batch_size):
            x1 = torch.randint(0, width // 2, (3,), device=dev).float()
            y1 = torch.randint(0, height // 2, (3,), device=dev).float()
            boxes = torch.stack([x1, y1, x1 + width / 4, y1 + height / 4], dim=1)
            targets.append({"boxes": boxes, "labels": torch.tensor([1, 2, 1], device=dev)})
        return images, targets

    step_times: list[float] = []
    losses: list[float] = []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for i in range(steps):
        images, targets = batch()
        devices.synchronize(dev)
        t0 = time.monotonic()
        loss = sum(model(images, targets).values())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        devices.synchronize(dev)
        step_times.append(time.monotonic() - t0)
        losses.append(float(loss.item()))
        log.info("step %d: %.3fs loss %.3f", i, step_times[-1], losses[-1])

    steady = step_times[1:] or step_times
    steady_mean = sum(steady) / len(steady)
    result = {
        "ok": all(map(_finite, losses)) and matmul_max_error < 1e-2,
        "device": str(dev),
        "model": model_name,
        "input": [batch_size, 3, height, width],
        "steps": steps,
        "first_step_s": step_times[0],
        "steady_step_s": steady_mean,
        "train_images_per_s": batch_size / steady_mean if steady_mean > 0 else None,
        "peak_gpu_memory_bytes": devices.max_memory_allocated(dev),
        "losses": losses,
        "matmul_max_abs_error": matmul_max_error,
        "environment": info,
    }
    return result


def _finite(value: float) -> bool:
    import math

    return math.isfinite(value)


def report(result: dict[str, Any]) -> str:
    env = result["environment"]
    mib = (result["peak_gpu_memory_bytes"] or 0) / 2**20
    lines = [
        f"result:          {'OK' if result['ok'] else 'FAILED'}",
        f"device:          {result['device']} {env.get('gpu_name') or ''} {env.get('gpu_arch') or ''}".rstrip(),
        f"gpu memory:      {(env.get('gpu_memory_bytes') or 0) / 2**30:.1f} GiB",
        f"torch:           {env.get('torch')}  torchvision {env.get('torchvision')}",
        f"hip:             {env.get('torch_hip')}  rocm userspace {env.get('rocm_userspace_version')}",
        f"kernel driver:   {env.get('amdgpu_driver_version') or 'in-tree amdgpu'} (kernel {env.get('kernel')})",
        f"model:           {result['model']} input {result['input']}",
        f"first step:      {result['first_step_s']:.2f}s (includes kernel selection)",
        f"steady step:     {result['steady_step_s']:.3f}s",
        f"throughput:      {result['train_images_per_s'] or 0:.2f} training images/s",
        f"peak gpu memory: {mib:.0f} MiB",
        f"matmul error:    {result['matmul_max_abs_error']:.2e}",
    ]
    return "\n".join(lines)


def to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, default=str)
