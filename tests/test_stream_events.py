"""Event schemas, including the contract with the Rust sampler."""

import json

import pytest

from crosswalk.stream import events as ev
from tests.conftest import repo_root

RUST_FIXTURE = repo_root() / "ingest" / "testdata" / "frame-event.json"


def test_the_samplers_frame_event_parses_here():
    # ingest/src/event.rs checks that its FrameEvent serializes to the same
    # set of fields as this file; this side checks the file parses.
    event = ev.parse(RUST_FIXTURE.read_bytes())
    assert isinstance(event, ev.FrameEvent)
    assert event.camera_id == "intersection-01"
    assert event.frame_id == f"{event.stream_id}/{event.frame_index}"
    assert event.image.bucket == "datasets" and event.image.key.endswith(".jpg")


def test_round_trip_and_discriminator():
    det = ev.DetectionEvent(
        frame_id="c/s/r-0/1", stream_id="c/s/r-0", camera_id="c", frame_index=1, pts_s=0.2,
        width=64, height=48, decoded_at=1.0, model_version="fake:0",
        detections=[ev.Box(label="person", score=0.9, box=(1, 2, 3, 4))],
    )  # fmt: skip
    data = json.loads(det.to_bytes())
    assert data["schema"] == 1 and data["type"] == "detections"
    again = ev.parse(det.to_bytes())
    assert again == det
    end = ev.parse({"type": "end", "stream_id": "s", "camera_id": "c", "clip": "c.mp4"})
    assert isinstance(end, ev.EndEvent) and end.complete


def test_unknown_fields_are_ignored_but_bad_events_are_rejected():
    raw = json.loads(RUST_FIXTURE.read_text())
    raw["added_in_a_later_version"] = 1
    assert isinstance(ev.parse(raw), ev.FrameEvent)
    with pytest.raises(ValueError):
        ev.parse({"type": "frame", "frame_id": "x"})  # required fields missing
    with pytest.raises(ValueError):
        ev.parse({"type": "mystery"})
    with pytest.raises(ValueError):
        ev.parse(b"not json")


@pytest.mark.parametrize(
    ("name", "camera"),
    [
        ("intersection-01.mp4", "intersection-01"),
        ("intersection-01__2026-10-01_0800.mp4", "intersection-01"),
        ("Main St & 3rd.mkv", "Main-St---3rd"),
        ("noext", "noext"),
        (".mp4", "mp4"),
        ("__x.mp4", "unnamed"),
    ],
)
def test_camera_ids_match_the_rust_rule(name, camera):
    # Same cases as ingest/src/event.rs::camera_ids_come_from_the_stem.
    assert ev.camera_id_from_clip(name) == camera


def test_dlq_topics_are_per_stage():
    assert ev.dlq_topic("inference") == "crosswalk.inference.dlq"
