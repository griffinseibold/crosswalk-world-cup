"""Parse BDD100K and COCO detection labels into one in-memory form.

Two formats, one :class:`Sample` type:

- **BDD100K** (training data): the ``det_20`` list files
  (``det_train.json``: a list of ``{"name", "attributes", "labels": [{"category",
  "box2d": {"x1","y1","x2","y2"}}]}``) and the older 2018 release
  (``bdd100k_labels_images_*.json``, same shape, plus lane/drivable-area
  polygons, which have no ``box2d`` and are skipped). A directory of per-image
  JSON files (each one entry, or the Scalabel ``frames[0].objects`` form) is
  also accepted.
- **COCO** (the hand-labeled fixed-camera eval set exported from CVAT or
  Label Studio): ``images``, ``annotations`` with ``[x, y, w, h]`` boxes,
  ``categories``.

Category names are mapped onto the project's class list through
``category_map``; anything that does not map is dropped and counted, so a
config mistake shows up in the logged dataset summary rather than as silently
missing classes.

Parsing is pure Python so it is tested without PyTorch;
:class:`DetectionDataset` (below) is the PyTorch view over the samples.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from crosswalk.config import DatasetConfig


@dataclass
class Sample:
    image: str  # path relative to the dataset's image root
    boxes: list[tuple[float, float, float, float]]  # x1, y1, x2, y2 pixels
    labels: list[int]  # project class index + 1 (0 is background)
    width: int | None = None
    height: int | None = None
    # Regions marked iscrowd in COCO: not trained on; pycocotools ignores
    # detections inside them during evaluation.
    crowd_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    crowd_labels: list[int] = field(default_factory=list)


@dataclass
class ParsedDataset:
    samples: list[Sample]
    classes: list[str]
    format: str
    labels_files: list[str]
    dropped_categories: Counter
    skipped_boxes: int  # degenerate (zero-area) boxes

    def class_counts(self) -> dict[str, int]:
        counts = Counter()
        for s in self.samples:
            for label in s.labels:
                counts[self.classes[label - 1]] += 1
        return {name: counts.get(name, 0) for name in self.classes}

    def summary(self) -> dict:
        return {
            "format": self.format,
            "images": len(self.samples),
            "boxes": sum(len(s.labels) for s in self.samples),
            "class_counts": self.class_counts(),
            "dropped_categories": dict(self.dropped_categories),
            "skipped_degenerate_boxes": self.skipped_boxes,
        }


def _class_index(classes: list[str], category_map: dict[str, str]):
    lookup = {name: i + 1 for i, name in enumerate(classes)}

    def index(category: str) -> int | None:
        target = category_map.get(category, category)
        return lookup.get(target) if target else None

    return index


def _valid(box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return x2 - x1 >= 1.0 and y2 - y1 >= 1.0


def _bdd_entries(path: Path) -> tuple[list[dict], list[str]]:
    if path.is_dir():
        files = sorted(p for p in path.rglob("*.json"))
        entries = []
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            entries.extend(data if isinstance(data, list) else [data])
        return entries, [str(f) for f in files]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "frames" in data:
        data = [data]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of BDD100K image entries")
    return data, [str(path)]


def _bdd_objects(entry: dict) -> list[dict]:
    if "labels" in entry:
        return entry.get("labels") or []
    frames = entry.get("frames") or []
    return (frames[0].get("objects") or []) if frames else []


def parse_bdd100k(config: DatasetConfig, classes: list[str]) -> ParsedDataset:
    entries, files = _bdd_entries(Path(config.labels))
    index = _class_index(classes, config.category_map)
    dropped: Counter = Counter()
    skipped = 0
    samples = []
    for entry in entries:
        attrs = entry.get("attributes") or {}
        if any(attrs.get(k) not in allowed for k, allowed in config.attribute_filter.items()):
            continue
        boxes, labels = [], []
        for obj in _bdd_objects(entry):
            box2d = obj.get("box2d")
            if not box2d:
                continue  # lanes and drivable areas are polygons, not boxes
            label = index(obj.get("category", ""))
            if label is None:
                dropped[obj.get("category", "")] += 1
                continue
            box = (float(box2d["x1"]), float(box2d["y1"]), float(box2d["x2"]), float(box2d["y2"]))
            if not _valid(box):
                skipped += 1
                continue
            boxes.append(box)
            labels.append(label)
        name = entry["name"]
        if not Path(name).suffix:
            name = f"{name}.jpg"  # video-frame entries sometimes omit it
        # BDD100K images are all 1280x720.
        samples.append(Sample(name, boxes, labels, width=1280, height=720))
    return ParsedDataset(samples, classes, "bdd100k", files, dropped, skipped)


def parse_coco(config: DatasetConfig, classes: list[str]) -> ParsedDataset:
    path = Path(config.labels)
    data = json.loads(path.read_text(encoding="utf-8"))
    index = _class_index(classes, config.category_map)
    category_label = {c["id"]: index(c["name"]) for c in data.get("categories", [])}
    category_name = {c["id"]: c["name"] for c in data.get("categories", [])}
    by_image: dict[int, Sample] = {}
    for image in data.get("images", []):
        by_image[image["id"]] = Sample(
            image["file_name"], [], [], width=image.get("width"), height=image.get("height")
        )
    dropped: Counter = Counter()
    skipped = 0
    for ann in data.get("annotations", []):
        sample = by_image.get(ann["image_id"])
        if sample is None:
            raise ValueError(f"annotation {ann.get('id')} refers to unknown image {ann['image_id']}")
        label = category_label.get(ann["category_id"])
        if label is None:
            dropped[category_name.get(ann["category_id"], str(ann["category_id"]))] += 1
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        box = (x, y, x + w, y + h)
        if not _valid(box):
            skipped += 1
            continue
        if ann.get("iscrowd"):
            sample.crowd_boxes.append(box)
            sample.crowd_labels.append(label)
        else:
            sample.boxes.append(box)
            sample.labels.append(label)
    samples = sorted(by_image.values(), key=lambda s: s.image)
    return ParsedDataset(samples, classes, "coco", [str(path)], dropped, skipped)


def select_subset(samples: list[Sample], max_images: int | None, seed: int) -> list[Sample]:
    """Deterministic subset: sort by image name, then sample without replacement."""
    ordered = sorted(samples, key=lambda s: s.image)
    if max_images is None or max_images >= len(ordered):
        return ordered
    rng = np.random.default_rng(seed)
    picks = sorted(rng.choice(len(ordered), size=max_images, replace=False).tolist())
    return [ordered[i] for i in picks]


def load(config: DatasetConfig, classes: list[str], seed: int = 0) -> ParsedDataset:
    if config.format == "bdd100k":
        parsed = parse_bdd100k(config, classes)
    else:
        parsed = parse_coco(config, classes)
    parsed.samples = select_subset(parsed.samples, config.max_images, seed)
    return parsed


class DetectionDataset:
    """PyTorch dataset over parsed samples.

    Augmentation is a horizontal flip whose coin toss comes from an RNG seeded
    by (seed, epoch, index), not from global or per-worker state. The same
    sample in the same epoch is therefore flipped the same way regardless of
    DataLoader worker count or whether the run was resumed, which is what makes
    resume-after-kill reproduce the uninterrupted run.
    """

    def __init__(
        self,
        parsed: ParsedDataset,
        image_root: str,
        train: bool,
        seed: int = 0,
        flip_prob: float = 0.5,
    ):
        self.samples = parsed.samples
        self.image_root = Path(image_root)
        self.train = train
        self.seed = seed
        self.flip_prob = flip_prob if train else 0.0
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch
        from torchvision.io import ImageReadMode, decode_image, read_file

        sample = self.samples[index]
        data = read_file(str(self.image_root / sample.image))
        image = decode_image(data, mode=ImageReadMode.RGB).float().div_(255.0)
        boxes = torch.tensor(sample.boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(sample.labels, dtype=torch.int64)
        if self.flip_prob > 0:
            rng = np.random.default_rng([self.seed, self.epoch, index])
            if rng.random() < self.flip_prob:
                width = image.shape[-1]
                image = image.flip(-1)
                if boxes.numel():
                    boxes = torch.stack(
                        [width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=1
                    )
        return image, {"boxes": boxes, "labels": labels, "index": torch.tensor(index)}


def collate(batch):
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)
