import json
from pathlib import Path

import pytest

from crosswalk import synthetic
from crosswalk.config import DatasetConfig
from crosswalk.training import datasets, hashing
from crosswalk.training.sampling import ResumableSampler, epoch_order, steps_per_epoch

CLASSES = ["car", "pedestrian"]


def bdd_entry(name, objects, **attributes):
    return {
        "name": name,
        "attributes": attributes,
        "labels": [
            {"category": cat, "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
            for cat, x1, y1, x2, y2 in objects
        ],
    }


def write_json(path: Path, data) -> str:
    path.write_text(json.dumps(data))
    return str(path)


def test_bdd100k_list_file_with_mapping_drops_and_filters(tmp_path):
    entries = [
        bdd_entry(
            "a.jpg",
            [("car", 0, 0, 10, 10), ("rider", 5, 5, 8, 20), ("traffic sign", 1, 1, 4, 4)],
            timeofday="daytime",
        ),
        bdd_entry("b", [("car", 3, 3, 3, 9)], timeofday="daytime"),  # zero width: skipped
        bdd_entry("c.jpg", [("car", 0, 0, 5, 5)], timeofday="night"),
    ]
    entries[0]["labels"].append({"category": "lane", "poly2d": [[0, 0]]})  # no box2d
    config = DatasetConfig(
        format="bdd100k",
        labels=write_json(tmp_path / "det.json", entries),
        images=str(tmp_path),
        category_map={"rider": "pedestrian"},
        attribute_filter={"timeofday": ["daytime"]},
    )
    parsed = datasets.load(config, CLASSES)
    assert [s.image for s in parsed.samples] == ["a.jpg", "b.jpg"]
    a = parsed.samples[0]
    assert a.labels == [1, 2]
    assert a.boxes == [(0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 8.0, 20.0)]
    assert parsed.dropped_categories == {"traffic sign": 1}
    assert parsed.skipped_boxes == 1
    assert parsed.class_counts() == {"car": 1, "pedestrian": 1}


def test_bdd100k_directory_of_scalabel_frames(tmp_path):
    folder = tmp_path / "labels"
    folder.mkdir()
    write_json(
        folder / "x.json",
        {
            "name": "x.jpg",
            "frames": [
                {"objects": [{"category": "car", "box2d": {"x1": 1, "y1": 1, "x2": 9, "y2": 9}}]}
            ],
        },
    )
    write_json(folder / "y.json", bdd_entry("y.jpg", [("pedestrian", 2, 2, 4, 9)]))
    parsed = datasets.load(DatasetConfig(format="bdd100k", labels=str(folder), images="."), CLASSES)
    assert [(s.image, s.labels) for s in parsed.samples] == [("x.jpg", [1]), ("y.jpg", [2])]


def test_coco_parsing_converts_boxes_and_handles_crowd(tmp_path):
    coco = {
        "images": [
            {"id": 1, "file_name": "f1.png", "width": 100, "height": 50},
            {"id": 2, "file_name": "f0.png"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 5, 20, 10], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 30], "iscrowd": 1},
            {"id": 3, "image_id": 1, "category_id": 9, "bbox": [0, 0, 5, 5]},
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 3, "name": "car"},
            {"id": 9, "name": "dog"},
        ],
    }
    config = DatasetConfig(
        format="coco",
        labels=write_json(tmp_path / "c.json", coco),
        images=str(tmp_path),
        category_map={"person": "pedestrian"},
    )
    parsed = datasets.load(config, CLASSES)
    assert [s.image for s in parsed.samples] == ["f0.png", "f1.png"]  # sorted, empty image kept
    f1 = parsed.samples[1]
    assert f1.boxes == [(10.0, 5.0, 30.0, 15.0)] and f1.labels == [1]
    assert f1.crowd_boxes == [(0.0, 0.0, 5.0, 30.0)] and f1.crowd_labels == [2]
    assert (f1.width, f1.height) == (100, 50)
    assert parsed.dropped_categories == {"dog": 1}


def test_coco_rejects_annotations_for_missing_images(tmp_path):
    coco = {
        "images": [],
        "annotations": [{"id": 1, "image_id": 7, "category_id": 1, "bbox": [0, 0, 2, 2]}],
        "categories": [{"id": 1, "name": "car"}],
    }
    config = DatasetConfig(format="coco", labels=write_json(tmp_path / "c.json", coco), images=".")
    with pytest.raises(ValueError, match="unknown image"):
        datasets.load(config, CLASSES)


def test_synthetic_bdd_and_coco_describe_the_same_boxes(synthetic_data):
    bdd = datasets.load(
        DatasetConfig(
            format="bdd100k", labels=synthetic_data["bdd100k"], images=synthetic_data["images"]
        ),
        CLASSES,
    )
    coco = datasets.load(
        DatasetConfig(
            format="coco",
            labels=synthetic_data["coco"],
            images=synthetic_data["images"],
            category_map={"person": "pedestrian"},
        ),
        CLASSES,
    )
    assert [(s.image, s.boxes, s.labels) for s in bdd.samples] == [
        (s.image, s.boxes, s.labels) for s in coco.samples
    ]


def test_subset_is_deterministic_and_seed_dependent(synthetic_data):
    config = DatasetConfig(
        format="bdd100k",
        labels=synthetic_data["bdd100k"],
        images=synthetic_data["images"],
        max_images=5,
    )
    a = [s.image for s in datasets.load(config, CLASSES, seed=1).samples]
    b = [s.image for s in datasets.load(config, CLASSES, seed=1).samples]
    c = [s.image for s in datasets.load(config, CLASSES, seed=2).samples]
    assert a == b and len(a) == 5 and a == sorted(a)
    assert a != c


# -- hashing --------------------------------------------------------------------


def manifest(data, **overrides):
    config = DatasetConfig(
        format="bdd100k", labels=data["bdd100k"], images=data["images"], **overrides
    )
    return hashing.dataset_manifest(
        datasets.load(config, CLASSES, seed=0), data["images"], hash_images=True
    )


def test_hash_is_stable(synthetic_data):
    assert manifest(synthetic_data)["sha256"] == manifest(synthetic_data)["sha256"]


def test_hash_changes_with_image_bytes(synthetic_data):
    before = manifest(synthetic_data)
    image = sorted(Path(synthetic_data["images"]).iterdir())[0]
    image.write_bytes(image.read_bytes() + b"\0")
    after = manifest(synthetic_data)
    assert after["images_sha256"] != before["images_sha256"]
    assert after["samples_sha256"] == before["samples_sha256"]
    assert after["sha256"] != before["sha256"]


def test_hash_changes_with_selection_but_not_file_order(synthetic_data):
    full = manifest(synthetic_data)
    subset = manifest(synthetic_data, max_images=4)
    assert subset["labels_sha256"] == full["labels_sha256"]
    assert subset["sha256"] != full["sha256"]
    # Reordering entries in the labels file changes its bytes (labels hash)
    # but not which samples were selected (samples hash).
    path = Path(synthetic_data["bdd100k"])
    path.write_text(json.dumps(list(reversed(json.loads(path.read_text())))))
    reordered = manifest(synthetic_data)
    assert reordered["samples_sha256"] == full["samples_sha256"]
    assert reordered["labels_sha256"] != full["labels_sha256"]


def test_manifest_contents(synthetic_data):
    m = manifest(synthetic_data)
    assert m["images"] == 16 and m["images_hashed"] is True
    assert set(m["class_counts"]) == set(CLASSES)
    assert list(m["labels_files"]) == ["det_train.json"]


# -- resumable sampling ------------------------------------------------------------


def test_epoch_order_is_a_function_of_seed_and_epoch():
    assert epoch_order(20, 1, 0) == epoch_order(20, 1, 0)
    assert epoch_order(20, 1, 0) != epoch_order(20, 1, 1)
    assert sorted(epoch_order(20, 1, 3)) == list(range(20))
    assert epoch_order(5, 1, 0, shuffle=False) == [0, 1, 2, 3, 4]


def test_sampler_resumes_mid_epoch_on_the_same_batches():
    full = ResumableSampler(22, 4, seed=9)
    full.set_position(2)
    all_indices = list(full)
    assert len(all_indices) == len(full) == steps_per_epoch(22, 4) * 4 == 20
    resumed = ResumableSampler(22, 4, seed=9)
    resumed.set_position(2, start_batch=3)
    assert list(resumed) == all_indices[12:]
    # The length stays the full epoch: Lightning counts a resumed epoch's
    # batches from where the checkpoint left off.
    assert len(resumed) == 20


def test_sampler_state_round_trip_and_epoch_changes():
    epochs_seen = []
    sampler = ResumableSampler(22, 4, seed=9, on_epoch=epochs_seen.append)
    sampler.mark(1, 3)
    assert sampler.state_dict() == {"epoch": 1, "start_batch": 3}
    restored = ResumableSampler(22, 4, seed=9, on_epoch=epochs_seen.append)
    restored.load_state_dict(sampler.state_dict())
    assert (restored.epoch, restored.start_batch) == (1, 3)
    # Lightning then calls set_epoch for the restored epoch: the position stays.
    restored.set_epoch(1)
    assert restored.start_batch == 3
    # The next epoch starts from its first batch.
    restored.set_epoch(2)
    assert (restored.epoch, restored.start_batch) == (2, 0)
    assert epochs_seen == [1, 2]
    # A position at the end of an epoch is the start of the next one.
    sampler.mark(1, 5)
    assert sampler.state_dict() == {"epoch": 2, "start_batch": 0}


def test_sampler_rejects_tiny_datasets():
    with pytest.raises(ValueError):
        ResumableSampler(3, 4, seed=0)


def test_synthetic_dataset_is_reproducible(tmp_path):
    a = synthetic.make_dataset(tmp_path / "a", images=4, seed=5)
    b = synthetic.make_dataset(tmp_path / "b", images=4, seed=5)
    assert Path(a["bdd100k"]).read_text() == Path(b["bdd100k"]).read_text()
