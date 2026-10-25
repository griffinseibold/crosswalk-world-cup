"""End-to-end: the whole streaming pipeline on generated footage, in process.

Runs ``crosswalk demo`` (memory bus, local storage, colour detector) and
checks every stage's output, idempotency of a replay, and the leaderboard.
The ``--video`` and Rust-sampler variants need PyAV and a built
``crosswalk-ingest`` binary; CI has PyAV, and the ingest job builds the
binary in the same workflow (the binary test runs when
``CROSSWALK_INGEST_BIN`` points at it).
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from crosswalk import demo, scene
from crosswalk.stream import events as ev

pytestmark = pytest.mark.slow


def check_result(result, crossings=2):
    counts = result.counts
    for stage in ("inference", "tracking", "indicators", "ranking"):
        assert counts[stage]["dead_lettered"] == 0, (stage, counts[stage])
        assert counts[stage]["invalid"] == 0
    # Every frame produced a detection event, and every detection a track event.
    assert result.topics[ev.TOPIC_FRAMES] == result.topics[ev.TOPIC_DETECTIONS], (
        result.topics,
        counts,
    )
    assert result.topics[ev.TOPIC_DETECTIONS] == result.topics[ev.TOPIC_TRACKS], (
        result.topics,
        counts,
    )
    assert result.topics[ev.TOPIC_INDICATORS] > 0
    board = result.leaderboard
    assert board["crossings"] == crossings
    ranked = board["tables"]["metric"]
    assert ranked, (board["not_ranked"], counts)
    for entry in ranked:
        assert entry["space"] == "metric"
        assert entry["observation"]["pedestrians"] > 0 and entry["observation"]["vehicles"] > 0
        assert "Generated" in entry["crossing"]
    assert "not safety advice" in board["disclaimer"]
    files = {p.name for p in result.out_dir.iterdir()}
    assert {"leaderboard.json", "index.html", "ranking.sqlite3", "metrics.prom"} <= files
    on_disk = json.loads((result.out_dir / "leaderboard.json").read_text())
    assert on_disk["tables"]["metric"][0]["camera_id"] == ranked[0]["camera_id"]


def test_the_scene_detector_finds_the_drawn_objects():
    detector = scene.SceneDetector()
    found = truth = 0
    for frame in scene.frames(scene.SceneConfig(seconds=20, seed=3)):
        truth += len(frame.truth)
        found += len(detector.predict([frame.image])[0])
    # Occlusion (a car over a pedestrian) and objects entering the frame
    # make it less than perfect, which is realistic enough for the demo.
    assert truth > 0 and 0.8 * truth <= found <= truth


def test_demo_runs_the_whole_pipeline(tmp_path):
    result = demo.run(tmp_path / "out", seconds=60, fps=5, window_s=15)
    check_result(result)
    # The impatient scene (drivers do not yield) should show more conflict
    # events than the yielding one: a sanity check that the indicators
    # respond to behaviour, on generated data with fixed seeds.
    by_id = {e["camera_id"]: e for t in result.leaderboard["tables"].values() for e in t} | {
        e["camera_id"]: e for e in result.leaderboard["not_ranked"]
    }
    yielding, impatient = by_id["synthetic-demo"], by_id["synthetic-demo-impatient"]
    assert impatient["counts"]["ttc_events"] + impatient["counts"]["close_approaches"] > (
        yielding["counts"]["ttc_events"] + yielding["counts"]["close_approaches"]
    )


def test_demo_with_video_decoding(tmp_path):
    pytest.importorskip("av")
    result = demo.run(tmp_path / "out", seconds=30, fps=5, window_s=10, video=True)
    check_result(result)


@pytest.mark.skipif(
    not (os.environ.get("CROSSWALK_INGEST_BIN") and shutil.which("ffmpeg")),
    reason="set CROSSWALK_INGEST_BIN to a built crosswalk-ingest (and install ffmpeg)",
)
def test_demo_through_the_rust_sampler(tmp_path):
    result = demo.run(
        tmp_path / "out",
        seconds=30,
        fps=5,
        window_s=10,
        ingest_bin=os.environ["CROSSWALK_INGEST_BIN"],
    )
    check_result(result)
    # 30 s of 10 fps video sampled at 5 fps: 150 frames per crossing.
    assert result.topics[ev.TOPIC_FRAMES] == 2 * 150 + 2, result.topics  # plus one end event each
