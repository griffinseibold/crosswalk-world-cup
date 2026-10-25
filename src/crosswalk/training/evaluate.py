"""COCO-style mAP with pycocotools, the reference implementation.

pycocotools is what the COCO leaderboard and TorchVision's own reference
scripts use, so "mAP@[.5:.95]" here means exactly what it means in papers and
in TorchVision's published model numbers. torchmetrics wraps the same
algorithm (and can call pycocotools as its backend) but adds a dependency for
no gain here.

Both the fine-tuned model and the pretrained COCO baseline are scored on the
same eval set. The baseline predicts COCO categories, so its labels are mapped
onto the project classes by name through ``coco_to_project`` (e.g. COCO
"person" -> "pedestrian"); anything that does not map is discarded before
scoring, so the baseline is not penalized for correctly finding classes the
project does not label.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from crosswalk.training.datasets import Sample


@dataclass
class Prediction:
    boxes: list[tuple[float, float, float, float]]  # x1, y1, x2, y2
    scores: list[float]
    labels: list[int]  # project label ids (1..N)


def to_coco_gt(samples: list[Sample], classes: list[str]) -> dict[str, Any]:
    images, annotations = [], []
    ann_id = 1
    for image_id, s in enumerate(samples, start=1):
        images.append(
            {"id": image_id, "file_name": s.image, "width": s.width or 0, "height": s.height or 0}
        )
        for boxes, labels, crowd in ((s.boxes, s.labels, 0), (s.crowd_boxes, s.crowd_labels, 1)):
            for (x1, y1, x2, y2), label in zip(boxes, labels, strict=True):
                w, h = x2 - x1, y2 - y1
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": crowd,
                    }
                )
                ann_id += 1
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(classes)]
    return {"images": images, "annotations": annotations, "categories": categories}


def to_coco_dt(
    predictions: list[Prediction], num_classes: int | None = None
) -> list[dict[str, Any]]:
    out = []
    for image_id, p in enumerate(predictions, start=1):
        for (x1, y1, x2, y2), score, label in zip(p.boxes, p.scores, p.labels, strict=True):
            # Label 0 is background (FCOS can emit it); ids past the class
            # list cannot match any ground truth.
            if label < 1 or (num_classes is not None and label > num_classes):
                continue
            out.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return out


STAT_NAMES = [
    "map", "map_50", "map_75", "map_small", "map_medium", "map_large",
    "mar_1", "mar_10", "mar_100", "mar_small", "mar_medium", "mar_large",
]  # fmt: skip


def coco_map(
    samples: list[Sample], predictions: list[Prediction], classes: list[str]
) -> dict[str, float]:
    """mAP and friends; -1 from pycocotools (no ground truth for a slice) becomes NaN."""
    if len(samples) != len(predictions):
        raise ValueError("one prediction per sample is required")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO()
    gt.dataset = to_coco_gt(samples, classes)
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    detections = to_coco_dt(predictions, len(classes))
    results: dict[str, float] = {}
    if not detections:
        return dict.fromkeys(STAT_NAMES, 0.0) | {f"ap/{c}": 0.0 for c in classes}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(detections)
        evaluator = COCOeval(gt, dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    for name, value in zip(STAT_NAMES, evaluator.stats, strict=True):
        results[name] = float("nan") if value < 0 else float(value)
    # Per-class AP@[.5:.95]: precision has shape [iou, recall, class, area, maxdets].
    precision = evaluator.eval["precision"]
    for k, cat_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, k, 0, -1]
        values = values[values > -1]
        results[f"ap/{classes[cat_id - 1]}"] = float(values.mean()) if values.size else float("nan")
    return results


def map_labels(prediction: Prediction, mapping: dict[int, int]) -> Prediction:
    keep = [i for i, label in enumerate(prediction.labels) if label in mapping]
    return Prediction(
        [prediction.boxes[i] for i in keep],
        [prediction.scores[i] for i in keep],
        [mapping[prediction.labels[i]] for i in keep],
    )


def coco_to_project(classes: list[str], category_map: dict[str, str]) -> dict[int, int]:
    """COCO label id -> project label id, via names and the dataset's category map."""
    from crosswalk.models import COCO_CATEGORIES

    lookup = {name: i + 1 for i, name in enumerate(classes)}
    mapping = {}
    for coco_id, name in enumerate(COCO_CATEGORIES):
        target = category_map.get(name, name)
        if target in lookup:
            mapping[coco_id] = lookup[target]
    return mapping


def predict_dataset(
    model, dataset, device, batch_size: int, num_workers: int = 0
) -> list[Prediction]:
    import torch
    from torch.utils.data import DataLoader

    from crosswalk.training.datasets import collate

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate
    )
    model.eval()
    predictions: list[Prediction] = []
    with torch.inference_mode():
        for images, _targets in loader:
            outputs = model([img.to(device) for img in images])
            for out in outputs:
                predictions.append(
                    Prediction(
                        [tuple(b) for b in out["boxes"].float().cpu().tolist()],
                        out["scores"].float().cpu().tolist(),
                        out["labels"].cpu().tolist(),
                    )
                )
    return predictions


def prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{k}": v for k, v in metrics.items()}


def finite(metrics: Iterable[tuple[str, float]]) -> dict[str, float]:
    import math

    return {k: v for k, v in metrics if not math.isnan(v)}
