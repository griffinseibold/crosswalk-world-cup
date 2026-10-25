"""Tiny synthetic road-scene datasets in BDD100K and COCO format.

Coloured rectangles on a grey "road", labelled as cars and pedestrians. The
images are meaningless to a pretrained detector, but they exercise every part
of the platform (parsing, hashing, data loading, the training loop,
checkpoints, resume, evaluation, MLflow) in seconds on a CPU, in CI and as a
smoke test on the cluster before BDD100K is in place.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLASSES = ["car", "pedestrian"]
# Category names as they appear in each format, so the category_map path is
# exercised the same way real data does.
BDD_NAMES = {"car": "car", "pedestrian": "pedestrian"}
COCO_NAMES = {"car": "car", "pedestrian": "person"}


def _draw(rng: np.random.Generator, width: int, height: int, max_objects: int):
    image = np.full((height, width, 3), 90, dtype=np.uint8)
    image[height // 2 :, :] = 60  # darker road surface
    objects = []
    for _ in range(int(rng.integers(1, max_objects + 1))):
        cls = CLASSES[int(rng.integers(0, len(CLASSES)))]
        if cls == "car":
            w, h, colour = int(rng.integers(width // 6, width // 3)), int(rng.integers(height // 8, height // 5)), (200, 30, 30)
        else:
            w, h, colour = int(rng.integers(width // 20, width // 12)), int(rng.integers(height // 6, height // 3)), (30, 30, 220)
        x1 = int(rng.integers(0, width - w))
        y1 = int(rng.integers(height // 4, height - h))
        image[y1 : y1 + h, x1 : x1 + w] = colour
        objects.append((cls, float(x1), float(y1), float(x1 + w), float(y1 + h)))
    return image, objects


def make_dataset(
    out_dir: str | Path,
    images: int = 16,
    size: tuple[int, int] = (96, 64),
    seed: int = 0,
    max_objects: int = 3,
) -> dict[str, str]:
    """Write images plus BDD100K and COCO label files; returns their paths."""
    from PIL import Image

    out = Path(out_dir)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    width, height = size
    bdd, coco_images, coco_annotations = [], [], []
    ann_id = 1
    for i in range(images):
        array, objects = _draw(rng, width, height, max_objects)
        name = f"frame-{i:05d}.png"  # PNG: lossless, so hashes are stable
        Image.fromarray(array).save(image_dir / name)
        bdd.append(
            {
                "name": name,
                "attributes": {"weather": "clear", "timeofday": "daytime", "scene": "city street"},
                "labels": [
                    {
                        "id": str(j),
                        "category": BDD_NAMES[cls],
                        "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                    for j, (cls, x1, y1, x2, y2) in enumerate(objects)
                ],
            }
        )
        coco_images.append({"id": i + 1, "file_name": name, "width": width, "height": height})
        for cls, x1, y1, x2, y2 in objects:
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": i + 1,
                    "category_id": CLASSES.index(cls) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    bdd_path = out / "det_train.json"
    bdd_path.write_text(json.dumps(bdd, indent=1), encoding="utf-8")
    coco_path = out / "instances.json"
    coco_path.write_text(
        json.dumps(
            {
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": [{"id": i + 1, "name": COCO_NAMES[c]} for i, c in enumerate(CLASSES)],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return {"images": str(image_dir), "bdd100k": str(bdd_path), "coco": str(coco_path)}
