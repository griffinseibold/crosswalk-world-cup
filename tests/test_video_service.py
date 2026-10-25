"""PyAV decoding and the end-to-end inference service on a synthetic clip."""

import json
from pathlib import Path

import pytest

av = pytest.importorskip("av")

from crosswalk.config import InferenceConfig  # noqa: E402
from crosswalk.inference import video  # noqa: E402
from crosswalk.inference.predictor import FakePredictor  # noqa: E402


@pytest.fixture
def clip(tmp_path) -> str:
    path = str(tmp_path / "clip.mp4")
    video.write_test_video(path, frames=24, size=(160, 96), fps=24)
    return path


def test_probe_and_decode(clip):
    info = video.probe(clip)
    assert (info.width, info.height) == (160, 96) and info.fps == pytest.approx(24)
    frames = list(video.iter_frames(clip))
    assert len(frames) == 24
    assert frames[0].image.shape == (96, 160, 3) and frames[0].image.dtype.name == "uint8"
    assert [f.index for f in frames] == list(range(24))
    assert frames[12].pts_s == pytest.approx(0.5, abs=0.05)


def test_decode_resize_and_limit(clip):
    frames = list(video.iter_frames(clip, resize=(80, 48), max_frames=5))
    assert len(frames) == 5 and frames[0].image.shape == (48, 80, 3)


def test_list_videos(tmp_path, clip):
    (tmp_path / "notes.txt").write_text("x")
    assert video.list_videos([str(tmp_path)]) == [clip]
    with pytest.raises(FileNotFoundError):
        video.list_videos([str(tmp_path / "missing.mp4")])


def test_offline_service_writes_one_record_per_frame(tmp_path, clip):
    pytest.importorskip("prometheus_client")
    from crosswalk.inference import service

    config = InferenceConfig.model_validate(
        {
            "source": {"videos": [clip], "realtime": False, "loop": False, "resize": [80, 48]},
            "output": {"dir": str(tmp_path / "out")},
            "batching": {"strategy": "dynamic", "max_batch_size": 4, "max_wait_ms": 2},
        }
    )
    state = service.run(config, serve=False, predictor=FakePredictor(base_s=0.0, per_image_s=0.0))
    assert state["phase"] == "finished" and state["submitted"] == 24 and state["dropped"] == 0
    [path] = state["detection_files"]
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert [r["frame"] for r in records] == list(range(24))
    assert records[0]["width"] == 80 and records[0]["video"] == "clip.mp4"


def test_torch_predictor_on_cpu(clip):
    pytest.importorskip("torch")
    from crosswalk.config import ModelConfig
    from crosswalk.inference.predictor import TorchPredictor

    predictor = TorchPredictor(
        ModelConfig(
            name="fcos_tiny",
            weights="none",
            classes=["car", "pedestrian"],
            min_size=48,
            max_size=80,
            score_threshold=0.0,
        ),
        device_name="cpu",
    )
    frames = [f.image for f in video.iter_frames(clip, resize=(80, 48), max_frames=3)]
    predictor.warmup((48, 80), [1, 3])
    results = predictor.predict(frames)
    assert len(results) == 3
    for dets in results:
        for d in dets:
            assert d.label in ("car", "pedestrian") and 0 <= d.score <= 1
