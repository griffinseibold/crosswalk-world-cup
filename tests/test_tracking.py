"""Tracker tests on synthetic trajectories whose right answers are known."""

from __future__ import annotations

import numpy as np
import pytest

from crosswalk.stream import events as ev
from crosswalk.tracking.assignment import (
    brute_force_assignment,
    iou_matrix,
    linear_sum_assignment,
    match,
)
from crosswalk.tracking.kalman import KalmanBox
from crosswalk.tracking.stage import TrackingHandler
from crosswalk.tracking.tracker import Detection, Tracker, TrackerConfig

FPS = 5.0


def box_at(x: float, y: float, w: float = 20, h: float = 40) -> tuple[float, float, float, float]:
    """Box whose bottom centre (the "foot") is at (x, y)."""
    return (x - w / 2, y - h, x + w / 2, y)


def run(tracker, frames):
    """frames: list of (t, [Detection]); returns {t: {track_id: TrackState}}."""
    out = {}
    for t, dets in frames:
        visible, _ = tracker.update(dets, t)
        out[t] = {s.track_id: s for s in visible}
    return out


# -- assignment -------------------------------------------------------------------


def test_hungarian_matches_brute_force_on_random_costs():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n, m = rng.integers(1, 6, size=2)
        cost = rng.random((n, m))
        pairs = linear_sum_assignment(cost)
        assert len(pairs) == min(n, m)
        assert len({r for r, _ in pairs}) == len(pairs) == len({c for _, c in pairs})
        assert sum(cost[r, c] for r, c in pairs) == pytest.approx(brute_force_assignment(cost))


def test_iou_and_thresholded_match():
    a = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    b = np.array([[5, 0, 15, 10], [0, 0, 10, 10], [200, 200, 210, 210]])
    iou = iou_matrix(a, b)
    assert iou[0, 1] == pytest.approx(1.0)
    assert iou[0, 0] == pytest.approx(50 / 150)
    assert iou[1].max() == 0
    pairs, rows, cols = match(iou, 0.3)
    assert pairs == [(0, 1)] and rows == [1] and cols == [0, 2]
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


# -- Kalman -----------------------------------------------------------------------


def test_kalman_recovers_constant_velocity_with_irregular_time_steps():
    rng = np.random.default_rng(1)
    vx, vy = 60.0, -15.0  # px/s
    t, kf = 0.0, KalmanBox(box_at(100, 300))
    for _ in range(40):
        dt = rng.choice([0.2, 0.2, 0.4, 0.1])  # dropped and bunched frames
        t += dt
        kf.predict(dt)
        noisy = box_at(100 + vx * t + rng.normal(0, 0.5), 300 + vy * t + rng.normal(0, 0.5))
        kf.update(noisy)
    est_vx, est_vy = kf.foot_velocity
    assert est_vx == pytest.approx(vx, abs=3.0)
    assert est_vy == pytest.approx(vy, abs=3.0)


# -- tracker ----------------------------------------------------------------------


def test_one_object_keeps_one_id_and_its_velocity():
    frames = [
        (i / FPS, [Detection(box_at(50 + 10 * i, 200), 0.9, "pedestrian")]) for i in range(20)
    ]
    tracks = run(Tracker(), frames)
    ids = {tid for frame in tracks.values() for tid in frame}
    assert ids == {1}
    assert 0.0 not in tracks or not tracks[0.0]  # tentative on the first frame (min_hits=2)
    last = tracks[19 / FPS][1]
    assert last.velocity[0] == pytest.approx(50.0, abs=5.0)  # 10 px per 0.2 s


def test_a_one_frame_false_positive_never_becomes_a_track():
    frames = [(0.0, [Detection(box_at(300, 300), 0.95, "vehicle")])] + [
        (i / FPS, []) for i in range(1, 6)
    ]
    tracks = run(Tracker(), frames)
    assert all(not f for f in tracks.values())


def test_pedestrian_and_vehicle_crossing_never_swap_identities():
    # A pedestrian walks right, a vehicle drives left along the same line;
    # their boxes overlap completely mid-way.
    frames = []
    for i in range(30):
        t = i / FPS
        frames.append(
            (t, [Detection(box_at(100 + 8 * i, 300), 0.9, "pedestrian"),
                 Detection(box_at(340 - 8 * i, 300, w=60, h=40), 0.9, "vehicle")])
        )  # fmt: skip
    tracks = run(Tracker(), frames)
    labels: dict[int, set[str]] = {}
    for frame in tracks.values():
        for tid, s in frame.items():
            labels.setdefault(tid, set()).add(s.label)
    assert len(labels) == 2
    assert all(len(v) == 1 for v in labels.values())


def test_two_pedestrians_passing_each_other_keep_their_ids():
    # Same class, opposite directions, vertically offset so the boxes overlap
    # partly at the crossing. The Kalman prediction carries each through.
    frames = []
    for i in range(30):
        t = i / FPS
        frames.append(
            (t, [Detection(box_at(100 + 10 * i, 300), 0.9, "pedestrian"),
                 Detection(box_at(390 - 10 * i, 320), 0.9, "pedestrian")])
        )  # fmt: skip
    tracks = run(Tracker(), frames)
    # Track 1 started on the left: it must stay the one moving right.
    for frame in tracks.values():
        for tid, s in frame.items():
            moving_right = s.velocity[0] > 0
            if s.hits > 4:
                assert moving_right == (tid == 1), (tid, s)
    assert {tid for f in tracks.values() for tid in f} == {1, 2}


def test_low_score_detections_carry_a_track_through_occlusion():
    frames = []
    for i in range(20):
        score = 0.3 if 8 <= i < 14 else 0.9  # partly hidden: low confidence
        frames.append((i / FPS, [Detection(box_at(50 + 6 * i, 250), score, "pedestrian")]))
    tracks = run(Tracker(), frames)
    assert {tid for f in tracks.values() for tid in f} == {1}
    assert all(1 in tracks[i / FPS] for i in range(1, 20))  # visible throughout


def test_short_gaps_keep_the_id_long_gaps_start_a_new_one():
    cfg = TrackerConfig(max_age_s=1.0)

    def frames(gap_frames):
        out = []
        for i in range(30):
            missing = 10 <= i < 10 + gap_frames
            out.append(
                (
                    i / FPS,
                    [] if missing else [Detection(box_at(50 + 4 * i, 250), 0.9, "pedestrian")],
                )
            )
        return out

    short = run(Tracker(cfg), frames(3))  # 0.6 s gap
    assert {tid for f in short.values() for tid in f} == {1}
    tracker = Tracker(cfg)
    long = run(tracker, frames(8))  # 1.6 s gap
    # The first track ended after max_age_s without a match; the object's
    # return started track 2.
    assert {tid for f in long.values() for tid in f} == {1, 2}
    assert [t.track_id for t in tracker.finished] == [1]


def test_out_of_order_timestamps_are_ignored_by_the_tracker():
    tracker = Tracker()
    tracker.update([Detection(box_at(10, 100), 0.9, "pedestrian")], 1.0)
    assert tracker.update([Detection(box_at(10, 100), 0.9, "pedestrian")], 0.5) == ([], [])


def test_flush_ends_every_confirmed_track():
    tracker = Tracker()
    for i in range(3):
        tracker.update([Detection(box_at(10 + i, 100), 0.9, "pedestrian")], i / FPS)
    ended = tracker.flush()
    assert [t.track_id for t in ended] == [1] and not tracker.tracks


# -- stage --------------------------------------------------------------------------


def det_event(i: int, x: float, stream="cam/clip/r-0") -> ev.DetectionEvent:
    return ev.DetectionEvent(
        frame_id=f"{stream}/{i}", stream_id=stream, camera_id="cam", frame_index=i, pts_s=i / FPS,
        width=640, height=480, decoded_at=1.0, model_version="fake",
        detections=[ev.Box(label="person", score=0.9, box=box_at(x, 300)),
                    ev.Box(label="traffic light", score=0.9, box=(0, 0, 5, 5))],
    )  # fmt: skip


def outputs(results):
    return [o.event for r in results for o in r]


def test_stage_reorders_within_the_window_and_counts_late_frames():
    late = []
    handler = TrackingHandler(reorder_window=3, on_late=late.append)
    order = [0, 2, 1, 3, 5, 4, 6, 7, 8]
    out = outputs(handler.process([det_event(i, 50 + 5 * i) for i in order]))
    assert [e.frame_index for e in out] == sorted(e.frame_index for e in out)
    # A frame arriving after its successors were fed is too late.
    out2 = outputs(handler.process([det_event(2, 60)]))
    assert out2 == [] and late == [1]


def test_stage_maps_labels_and_flushes_on_end():
    handler = TrackingHandler(reorder_window=2)
    events = [det_event(i, 50 + 5 * i) for i in range(6)]
    end = ev.EndEvent(stream_id="cam/clip/r-0", camera_id="cam", clip="clip.mp4", frames=6)
    out = outputs(handler.process([*events, end]))
    tracks = [e for e in out if isinstance(e, ev.TrackEvent)]
    assert [e.frame_index for e in tracks] == list(range(6))  # all released by the end event
    assert isinstance(out[-1], ev.EndEvent)
    labels = {p.label for e in tracks for p in e.tracks}
    assert labels == {"pedestrian"}  # the traffic light is not tracked
    assert handler.streams == {}


def test_stage_flushes_streams_that_go_quiet():
    now = [0.0]
    handler = TrackingHandler(reorder_window=4, idle_timeout_s=60, clock=lambda: now[0])
    handler.process([det_event(i, 50 + 5 * i, stream="a/x/r-0") for i in range(3)])
    now[0] = 120.0
    out = outputs(handler.process([det_event(0, 10, stream="b/y/r-0")]))
    assert {e.stream_id for e in out} == {"a/x/r-0"}
    assert set(handler.streams) == {"b/y/r-0"}


def test_tracking_stage_end_to_end_on_the_memory_bus():
    from crosswalk.config import TrackingStageConfig
    from crosswalk.stream.bus import MemoryBus
    from crosswalk.tracking.stage import build_runner

    bus = MemoryBus()
    runner = build_runner(TrackingStageConfig(reorder_window=2), bus)
    runner.start()
    producer = bus.producer()
    for i in [0, 1, 3, 2, 4, 5]:
        producer.send(ev.TOPIC_DETECTIONS, "cam", det_event(i, 50 + 5 * i).to_bytes())
    producer.send(
        ev.TOPIC_DETECTIONS,
        "cam",
        ev.EndEvent(stream_id="cam/clip/r-0", camera_id="cam", clip="clip.mp4").to_bytes(),
    )
    runner.config.poll_timeout_s = 0.01
    counts = runner.run(until_idle=True)
    assert counts["processed"] == 7
    out = [ev.parse(m.value) for m in bus.messages(ev.TOPIC_TRACKS)]
    assert [e.frame_index for e in out if isinstance(e, ev.TrackEvent)] == [0, 1, 2, 3, 4, 5]
    assert isinstance(out[-1], ev.EndEvent)
