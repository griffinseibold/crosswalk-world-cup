"""Run the detector on a batch of frames.

The pipeline and the benchmark harness depend only on the :class:`Predictor`
protocol, so their batching, timing and output logic is tested with a fake
predictor on machines without PyTorch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from crosswalk.config import ModelConfig


@dataclass
class Detection:
    label: str
    label_id: int
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels

    def to_json(self) -> dict:
        return {
            "label": self.label,
            "label_id": self.label_id,
            "score": round(self.score, 4),
            "box": [round(v, 1) for v in self.box],
        }


class Predictor(Protocol):
    model_version: str

    def predict(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Detections for each H x W x 3 uint8 RGB image. Returns after the GPU finishes."""
        ...


class TorchPredictor:
    def __init__(
        self,
        model_config: ModelConfig,
        device_name: str = "auto",
        precision: str = "fp32",
    ):
        import torch

        from crosswalk import device as devices
        from crosswalk.models import build_detector, describe_weights, label_names

        self._torch = torch
        self.device = devices.resolve(device_name)
        self.model_config = model_config
        if model_config.checkpoint:
            from crosswalk import registry

            # A training checkpoint (path or s3://) or a registered version
            # (models:/crosswalk-detector@champion). Either decides the class
            # list (and so the head's shape). weights="none" skips
            # downloading COCO weights that would be overwritten anyway.
            saved = registry.load_any(model_config.checkpoint)
            model_config = model_config.model_copy(
                update={"classes": saved["classes"], "weights": "none"}
            )
            model = build_detector(model_config)
            model.load_state_dict(saved["model"])
        else:
            model = build_detector(model_config)
        self.names = label_names(model_config)
        self.model = model.to(self.device).eval()
        # fp16 autocast halves activation memory and uses the GPU's faster
        # half-precision paths; detection post-processing stays in fp32.
        self.autocast = precision == "fp16" and self.device.type == "cuda"
        self.model_version = describe_weights(model_config)
        if model_config.checkpoint and model_config.checkpoint.startswith("models:/"):
            # Name the exact version an alias resolved to, so detections and
            # the dashboard say what actually served them.
            try:
                from crosswalk import registry

                self.model_version = registry.resolve(model_config.checkpoint).describe()
            except Exception:  # pragma: no cover - registry unreachable; keep the URI
                pass
        self._sync = devices.synchronize

    def _to_tensor(self, image: np.ndarray):
        torch = self._torch
        tensor = torch.from_numpy(np.ascontiguousarray(image)).to(self.device, non_blocking=True)
        return tensor.permute(2, 0, 1).float().div_(255.0)

    def predict(self, images: list[np.ndarray]) -> list[list[Detection]]:
        torch = self._torch
        with torch.inference_mode():
            batch = [self._to_tensor(img) for img in images]
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.autocast):
                outputs = self.model(batch)
            self._sync(self.device)
        threshold = self.model_config.score_threshold
        results: list[list[Detection]] = []
        for out in outputs:
            boxes = out["boxes"].float().cpu().tolist()
            scores = out["scores"].float().cpu().tolist()
            labels = out["labels"].cpu().tolist()
            frame: list[Detection] = []
            for box, score, label in zip(boxes, scores, labels, strict=True):
                # TorchVision's FCOS counts background among its classes and
                # can emit label 0; it is never a detection.
                if score < threshold or label == 0:
                    continue
                name = self.names[label] if label < len(self.names) else str(label)
                frame.append(Detection(name, int(label), float(score), tuple(box)))
            results.append(frame)
        return results

    def warmup(self, shape: tuple[int, int], batch_sizes: list[int]) -> None:
        """Run each batch size once so kernel selection and caching happen up front.

        MIOpen (ROCm) and cuDNN pick and sometimes compile convolution kernels
        the first time they see a shape; without warmup the first real frames
        would carry seconds of one-off latency.
        """
        height, width = shape
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        for size in sorted(set(batch_sizes)):
            self.predict([dummy] * size)


class FakePredictor:
    """Deterministic stand-in: sleeps ``base_s + per_image_s * N`` per batch."""

    def __init__(
        self, base_s: float = 0.002, per_image_s: float = 0.001, fail_on: int | None = None
    ):
        self.base_s = base_s
        self.per_image_s = per_image_s
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.fail_on = fail_on
        self.model_version = "fake:0"

    def predict(self, images: list[np.ndarray]) -> list[list[Detection]]:
        self.calls += 1
        self.batch_sizes.append(len(images))
        if self.fail_on is not None and self.calls == self.fail_on:
            raise RuntimeError("injected predictor failure")
        time.sleep(self.base_s + self.per_image_s * len(images))
        results = []
        for img in images:
            h, w = img.shape[:2]
            results.append([Detection("car", 3, 0.9, (w * 0.1, h * 0.1, w * 0.3, h * 0.3))])
        return results
