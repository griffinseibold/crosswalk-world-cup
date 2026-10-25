import itertools
import math
import pickle

import pytest

from crosswalk.training.datasets import Sample
from crosswalk.training.evaluate import (
    Prediction,
    coco_map,
    coco_to_project,
    map_labels,
    to_coco_dt,
    to_coco_gt,
)
from crosswalk.training.schedule import WarmupCosine

CLASSES = ["car", "pedestrian"]


def test_warmup_then_cosine():
    s = WarmupCosine(total_steps=110, warmup_steps=10, min_ratio=0.1)
    assert s(0) == pytest.approx(0.1)
    assert s(9) == pytest.approx(1.0)
    assert s(10) == pytest.approx(1.0)
    assert s(60) == pytest.approx(0.1 + 0.9 * 0.5)
    assert s(110) == pytest.approx(0.1)
    assert s(500) == pytest.approx(0.1)
    values = [s(i) for i in range(10, 111)]
    assert all(a >= b for a, b in itertools.pairwise(values))


def test_schedule_pickles_for_scheduler_state():
    s = WarmupCosine(100, 5, 0.05)
    assert pickle.loads(pickle.dumps(s))(50) == s(50)


def toy():
    samples = [
        Sample("a.png", [(10, 10, 50, 50), (60, 10, 70, 40)], [1, 2], width=100, height=100),
        Sample("b.png", [(5, 5, 45, 45)], [1], width=100, height=100),
    ]
    return samples


def test_coco_conversion():
    gt = to_coco_gt(toy(), CLASSES)
    assert [c["name"] for c in gt["categories"]] == CLASSES
    assert gt["annotations"][0]["bbox"] == [10, 10, 40, 40] and gt["annotations"][0]["area"] == 1600
    dt = to_coco_dt([Prediction([(1, 2, 4, 6)], [0.5], [2])])
    assert dt == [{"image_id": 1, "category_id": 2, "bbox": [1, 2, 3, 4], "score": 0.5}]
    # Background (0) and out-of-range labels are dropped.
    mixed = Prediction([(0, 0, 1, 1)] * 3, [0.9, 0.8, 0.7], [0, 1, 3])
    assert [d["category_id"] for d in to_coco_dt([mixed], num_classes=2)] == [1]


def test_perfect_predictions_score_one():
    pytest.importorskip("pycocotools")
    samples = toy()
    preds = [Prediction(s.boxes, [0.99] * len(s.boxes), s.labels) for s in samples]
    result = coco_map(samples, preds, CLASSES)
    assert result["map"] == pytest.approx(1.0)
    assert result["map_50"] == pytest.approx(1.0)
    assert result["ap/car"] == pytest.approx(1.0) and result["ap/pedestrian"] == pytest.approx(1.0)


def test_wrong_and_missing_predictions_score_lower():
    pytest.importorskip("pycocotools")
    samples = toy()
    # Cars found on image a only; pedestrian boxed in the wrong place.
    preds = [
        Prediction([(10, 10, 50, 50), (0, 80, 10, 95)], [0.9, 0.8], [1, 2]),
        Prediction([], [], []),
    ]
    result = coco_map(samples, preds, CLASSES)
    assert 0 < result["map"] < 1
    assert result["ap/pedestrian"] == pytest.approx(0.0)
    assert result["ap/car"] == pytest.approx(0.5, abs=0.02)  # one of two cars, perfect precision


def test_no_detections_at_all():
    pytest.importorskip("pycocotools")
    result = coco_map(toy(), [Prediction([], [], [])] * 2, CLASSES)
    assert result["map"] == 0.0


def test_prediction_count_must_match():
    with pytest.raises(ValueError):
        coco_map(toy(), [], CLASSES)


def test_baseline_label_mapping():
    mapping = coco_to_project(CLASSES, {"person": "pedestrian"})
    assert mapping == {1: 2, 3: 1}  # COCO person -> pedestrian, COCO car -> car
    pred = Prediction([(0, 0, 1, 1), (0, 0, 2, 2), (0, 0, 3, 3)], [0.9, 0.8, 0.7], [3, 18, 1])
    mapped = map_labels(pred, mapping)
    assert mapped.labels == [1, 2] and mapped.scores == [0.9, 0.7]  # the dog (18) is dropped


def test_empty_size_slices_are_nan_not_minus_one():
    pytest.importorskip("pycocotools")
    samples = toy()  # every box is "small" or "medium"; no large objects
    preds = [Prediction(s.boxes, [0.9] * len(s.boxes), s.labels) for s in samples]
    result = coco_map(samples, preds, CLASSES)
    assert math.isnan(result["map_large"])
