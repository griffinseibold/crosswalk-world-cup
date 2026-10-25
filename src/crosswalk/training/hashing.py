"""Content hashes for datasets, recorded in MLflow with every run.

A run's dataset hash answers "was this model trained on exactly the same data
as that one?" without trusting file names or dates. The hash covers:

- the bytes of every labels file,
- the parsed, selected samples (image name, boxes, labels), so a different
  ``max_images`` subset, category map or attribute filter changes the hash
  even with identical files,
- optionally the bytes of every selected image (``hash_images``), so a
  re-encoded or blurred image changes it too.

Files are hashed in sorted order, so directory listing order never matters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from crosswalk.training.datasets import ParsedDataset

CHUNK = 1024 * 1024


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def samples_sha256(parsed: ParsedDataset) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(parsed.classes).encode())
    for s in sorted(parsed.samples, key=lambda s: s.image):
        record = [s.image, [list(b) for b in s.boxes], s.labels, [list(b) for b in s.crowd_boxes]]
        digest.update(json.dumps(record, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def images_sha256(parsed: ParsedDataset, image_root: str | Path) -> str:
    """Hash of (name, content hash) pairs for every selected image."""
    root = Path(image_root)
    digest = hashlib.sha256()
    for name in sorted({s.image for s in parsed.samples}):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(file_sha256(root / name).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_manifest(parsed: ParsedDataset, image_root: str | Path, hash_images: bool) -> dict:
    labels = {Path(f).name: file_sha256(f) for f in sorted(parsed.labels_files)}
    parts = {
        "labels_sha256": hashlib.sha256(
            json.dumps(labels, sort_keys=True).encode()
        ).hexdigest(),
        "samples_sha256": samples_sha256(parsed),
        "images_sha256": images_sha256(parsed, image_root) if hash_images else None,
    }
    combined = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
    return {
        "sha256": combined,
        **parts,
        "labels_files": labels,
        "images_hashed": hash_images,
        **parsed.summary(),
    }
