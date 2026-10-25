"""Conflict indicators on synthetic trajectories whose answers are known.

Each scenario is built so the expected value can be worked out by hand, and
the comment next to each assertion shows the arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from crosswalk.indicators import geometry as geo
from crosswalk.indicators.camera import Calibration, CameraConfig, Thresholds, load_cameras
from crosswalk.indicators.compute import Window, window_bounds
from crosswalk.indicators.stage import IndicatorsHandler
from crosswalk.stream import events as ev
from tests.conftest import repo_root

# A calibration where 1 pixel = 0.05 m and image y grows with world y: the
# image is a plan view of the road (scale 20 px per metre).
PLAN_VIEW = Calibration(
    image_points=[(0, 0), (400, 0), (400, 200), (0, 200)],
    world_points=[(0, 0), (20, 0), (20, 10), (0, 10)],
    reference="synthetic plan view, 20 px per metre",
)


def metric_camera(**thresholds) -> CameraConfig:
    return CameraConfig(
        camera_id="m",
        frame_size=(400, 200),
        calibration=PLAN_VIEW,
        thresholds=Thresholds(**thresholds),
    )


def pixel_camera(**thresholds) -> CameraConfig:
    return CameraConfig(camera_id="p", thresholds=Thresholds(**thresholds))


def foot_box(x_px: float, y_px: float, h: float = 40, w: float = 16):
    return (x_px - w / 2, y_px - h, x_px + w / 2, y_px)


def metres_to_px(x_m, y_m):
    return x_m * 20, y_m * 20


def run_window(camera, frames, size=(400, 200)):
    """frames: list of (t, [(track_id, label, (x_m, y_m) or (x_px, y_px))]).
    Positions are metres for a calibrated camera, pixels otherwise."""
    w = Window(camera, size[0], size[1], 0.0, 60.0)
    for t, objs in frames:
        boxes = []
        for tid, label, (x, y) in objs:
            px = metres_to_px(x, y) if camera.metric else (x, y)
            boxes.append((tid, label, foot_box(*px)))
        w.add_frame(t, 1000.0 + t, boxes)
    return w, w.compute()


# -- geometry ---------------------------------------------------------------------


def test_homography_maps_the_reference_points_exactly_and_others_projectively():
    # A trapezoid in the image (perspective) to a 12 m x 3 m crosswalk.
    image = [(420, 430), (860, 430), (940, 560), (340, 560)]
    world = [(0, 0), (12, 0), (12, 3), (0, 3)]
    H = geo.Homography.from_points(image, world)
    for p, q in zip(image, world, strict=True):
        assert H.to_world(p) == pytest.approx(q, abs=1e-9)
    # The intersection of the image diagonals maps to the centre of the
    # rectangle: diagonals are preserved by any homography.
    (center_img, _, _) = geo.segment_intersection(image[0], image[2], image[1], image[3])
    assert H.to_world(center_img) == pytest.approx((6, 1.5), abs=1e-9)


def test_homography_rejects_collinear_points_and_scales_with_resizing():
    with pytest.raises(geo.DegenerateCalibration):
        geo.Homography.from_points(
            [(0, 0), (1, 1), (2, 2), (0, 5)], [(0, 0), (1, 0), (1, 1), (0, 1)]
        )
    H = geo.Homography.from_points(PLAN_VIEW.image_points, PLAN_VIEW.world_points)
    half = H.scaled(0.5, 0.5)  # frames resized to 200 x 100
    assert half.to_world((100, 50)) == pytest.approx(H.to_world((200, 100)))


def test_points_beyond_the_horizon_are_not_on_the_road_whatever_the_normalisation():
    # Perspective: image rows converge towards a horizon above y = 0.
    H = geo.Homography.from_points(
        [(100, 100), (300, 100), (400, 300), (0, 300)], [(0, 10), (4, 10), (4, 0), (0, 0)]
    )
    assert H.to_world((200, 300)) is not None
    assert H.to_world((200, -500)) is None
    # The same mapping with H multiplied by -1 (W changes sign everywhere).
    flipped = geo.Homography(-H.H, road_sign=-H.road_sign)
    assert flipped.to_world((200, 300)) == pytest.approx(H.to_world((200, 300)))
    assert flipped.to_world((200, -500)) is None


def test_segment_intersection():
    hit = geo.segment_intersection((0, 0), (10, 0), (5, -5), (5, 5))
    assert (
        hit[0] == pytest.approx((5, 0))
        and hit[1] == pytest.approx(0.5)
        and hit[2] == pytest.approx(0.5)
    )
    assert geo.segment_intersection((0, 0), (10, 0), (11, -5), (11, 5)) is None
    assert geo.segment_intersection((0, 0), (10, 0), (0, 1), (10, 1)) is None  # parallel
    assert geo.segment_intersection((0, 0), (10, 0), (2, 0), (8, 0)) is None  # collinear overlap


def test_time_to_collision_closed_form():
    # Two objects 10 m apart closing at 4 m/s, collision radius 1 m:
    # they touch when the gap is 1 m, after (10 - 1) / 4 = 2.25 s.
    assert geo.time_to_collision((10, 0), (-4, 0), 1.0) == pytest.approx(2.25)
    # Already within the radius.
    assert geo.time_to_collision((0.5, 0), (-1, 0), 1.0) == 0.0
    # Moving apart.
    assert geo.time_to_collision((10, 0), (4, 0), 1.0) is None
    # Passing with a 3 m lateral offset never comes within 1 m.
    assert geo.time_to_collision((10, 3), (-4, 0), 1.0) is None
    # Offset 0.6 m: contact when the along-track gap is sqrt(1 - 0.36) = 0.8,
    # after (10 - 0.8) / 4 = 2.3 s.
    assert geo.time_to_collision((10, 0.6), (-4, 0), 1.0) == pytest.approx(2.3)


def test_velocity_is_exact_for_constant_motion():
    points = [(1.0 + 1.5 * t, 2.0 - 0.5 * t, t) for t in np.arange(0, 2, 0.2)]
    assert geo.velocity(points) == pytest.approx((1.5, -0.5))
    assert geo.velocity(points[:1]) is None


def test_first_crossing_and_post_encroachment_time():
    # Pedestrian walks north along x = 5 at 1 m/s, at (5, 5) at t = 5.
    ped = [(5.0, float(y), float(y)) for y in range(11)]
    # Vehicle drives east along y = 5 at 5 m/s, at (5, 5) at t = 7.
    veh = [(5.0 * (t - 6), 5.0, float(t)) for t in range(6, 10)]
    point, t_ped, t_veh = geo.first_crossing(ped, veh)
    assert point == pytest.approx((5, 5))
    assert t_ped == pytest.approx(5.0) and t_veh == pytest.approx(7.0)  # PET = 2 s


def test_point_in_polygon():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert geo.point_in_polygon((5, 5), square)
    assert not geo.point_in_polygon((15, 5), square)


# -- windows: metric space ------------------------------------------------------------


def crossing_scenario(veh_delay_s: float, fps: float = 10.0, duration: float = 10.0):
    """Pedestrian: north along x = 10 m from y = 2 to y = 8 at 1 m/s (t 0-6),
    so at (10, 5) at t = 3. Vehicle: east along y = 5 m at 4 m/s, at
    (10, 5) at t = 3 + veh_delay_s."""
    frames = []
    for i in range(int(duration * fps) + 1):
        t = i / fps
        objs = []
        if t <= 6:
            objs.append((1, "pedestrian", (10.0, 2.0 + t)))
        x_v = 10.0 + 4.0 * (t - 3 - veh_delay_s)
        if 0 <= x_v <= 19.5:
            objs.append((2, "vehicle", (x_v, 5.0)))
        frames.append((t, objs))
    return frames


def test_counts_and_exposure():
    _, r = run_window(metric_camera(), crossing_scenario(veh_delay_s=4.0))
    assert (r.pedestrians, r.vehicles) == (1, 1)
    # Pedestrian visible t = 0..6.
    assert r.pedestrian_seconds == pytest.approx(6.0)
    # Vehicle on screen while 0 <= 10 + 4 (t - 7) <= 19.5, i.e.
    # 4.5 <= t <= 9.375; at 10 fps the frames run 4.5 .. 9.3 -> 4.8 s.
    assert r.vehicle_seconds == pytest.approx(4.8)
    assert r.space == "metric"


def test_post_encroachment_time_is_the_gap_at_the_conflict_point():
    _, r = run_window(metric_camera(pet_s=3.0), crossing_scenario(veh_delay_s=2.0))
    # Pedestrian at the crossing point at t = 3, vehicle at t = 5: PET = 2 s.
    assert r.min_pet_s == pytest.approx(2.0, abs=0.05)
    assert r.pet_events == 1 and r.path_crossings == 1
    _, late = run_window(metric_camera(pet_s=3.0), crossing_scenario(veh_delay_s=3.5))
    assert late.min_pet_s == pytest.approx(3.5, abs=0.05)
    assert late.pet_events == 0  # 3.5 s is not below the 3 s threshold
    assert late.path_crossings == 0  # nor within the 3 s crossing window


def test_close_approach_and_minimum_separation():
    # Vehicle passes the crossing point 0.5 s after the pedestrian. At time t
    # the pedestrian is at (10, 2 + t) and the vehicle at (10 + 4 (t - 3.5), 5):
    # separation^2 = (4 (t - 3.5))^2 + (t - 3)^2, minimised at
    # t = (16 * 3.5 + 3) / 17 = 59 / 17, giving 4 * 0.5 / sqrt(17) = 0.485 m.
    _, r = run_window(
        metric_camera(close_approach_m=2.0), crossing_scenario(veh_delay_s=0.5, fps=100)
    )
    assert r.min_separation == pytest.approx(2 / math.sqrt(17), abs=0.01)
    assert r.close_approaches == 1  # one pair, counted once however many frames


def test_ttc_flags_a_closing_course_and_ignores_a_safe_pass():
    # Head-on at walking + driving speeds along the same line: a vehicle
    # coming straight at a pedestrian. Positions in metres.
    frames = []
    for i in range(31):
        t = i / 10
        frames.append(
            (t, [(1, "pedestrian", (5.0 + 1.0 * t, 5.0)), (2, "vehicle", (19.0 - 3.0 * t, 5.0))])
        )
    _, r = run_window(metric_camera(ttc_s=1.5, collision_radius_m=1.0), frames)
    # Closing speed 4 m/s, gap 14 - 4 t; TTC = (gap - 1) / 4 drops below
    # 1.5 s once gap < 7 m, i.e. after t = 1.75 s: one pair flagged.
    assert r.ttc_events == 1
    assert r.min_ttc_s is not None and r.min_ttc_s < 1.5
    # Same speeds, 4 m lateral offset: never within 1 m, so no TTC.
    frames_safe = [
        (t, [(1, "pedestrian", (5.0 + 1.0 * t, 1.0)), (2, "vehicle", (19.0 - 3.0 * t, 5.0))])
        for t in (i / 10 for i in range(31))
    ]
    _, safe = run_window(metric_camera(), frames_safe)
    assert safe.ttc_events == 0 and safe.min_ttc_s is None


def test_a_pedestrian_standing_by_a_parked_car_is_one_close_approach_not_hundreds():
    frames = [
        (i / 10, [(1, "pedestrian", (5.0, 5.0)), (2, "vehicle", (6.0, 5.0))]) for i in range(300)
    ]
    _, r = run_window(metric_camera(), frames)
    assert r.close_approaches == 1
    assert r.ttc_events == 0  # neither moves: no closing course
    assert r.path_crossings == 0


def test_roi_excludes_points_outside_it():
    cam = CameraConfig(
        camera_id="r", frame_size=(400, 200), calibration=PLAN_VIEW,
        roi=[(0, 0), (200, 0), (200, 200), (0, 200)],  # left half of the frame (x < 10 m)
    )  # fmt: skip
    frames = [
        (i / 10, [(1, "pedestrian", (5.0, 5.0)), (2, "pedestrian", (15.0, 5.0))]) for i in range(20)
    ]
    _, r = run_window(cam, frames)
    assert r.pedestrians == 1


# -- windows: pixel space ---------------------------------------------------------------


def test_pixel_space_never_reports_ttc_or_pet():
    # The same crossing, but the camera has no calibration: positions are pixels.
    frames = [
        (t, [(tid, lab, (x * 20, y * 20)) for tid, lab, (x, y) in objs])
        for t, objs in crossing_scenario(2.0)
    ]
    _, r = run_window(pixel_camera(), frames)
    assert r.space == "pixel"
    assert r.ttc_events == 0 and r.min_ttc_s is None
    assert r.pet_events == 0 and r.min_pet_s is None
    assert r.path_crossings == 1  # same rule on image paths
    assert r.pedestrians == 1 and r.vehicles == 1


def test_pixel_close_approach_scales_with_apparent_height():
    # Boxes are 40 px tall: "close" means within 40 px (1 height).
    near = [(i / 10, [(1, "pedestrian", (100, 100)), (2, "vehicle", (130, 100))]) for i in range(5)]
    far = [(i / 10, [(1, "pedestrian", (100, 100)), (2, "vehicle", (150, 100))]) for i in range(5)]
    assert run_window(pixel_camera(), near)[1].close_approaches == 1
    assert run_window(pixel_camera(), far)[1].close_approaches == 0
    assert run_window(pixel_camera(), near)[1].min_separation == pytest.approx(30)


def test_window_bounds_are_aligned_to_the_clip_start():
    assert window_bounds(0.0, 60) == (0, 0.0, 60.0)
    assert window_bounds(59.99, 60) == (0, 0.0, 60.0)
    assert window_bounds(61.0, 60) == (1, 60.0, 120.0)


# -- stage ----------------------------------------------------------------------------


def track_event(i: int, t: float, tracks, stream="m/clip/r-0", camera="m") -> ev.TrackEvent:
    return ev.TrackEvent(
        frame_id=f"{stream}/{i}", stream_id=stream, camera_id=camera, frame_index=i, pts_s=t,
        width=400, height=200, decoded_at=1000.0 + t,
        tracks=[ev.TrackPoint(track_id=tid, label=lab, box=foot_box(*metres_to_px(x, y)), score=0.9)
                for tid, lab, (x, y) in tracks],
    )  # fmt: skip


def test_stage_emits_one_event_per_window_and_on_end():
    handler = IndicatorsHandler({"m": metric_camera()}, window_s=5.0)
    frames = crossing_scenario(veh_delay_s=2.0, fps=5, duration=12)
    events = [track_event(i, t, objs) for i, (t, objs) in enumerate(frames)]
    end = ev.EndEvent(stream_id="m/clip/r-0", camera_id="m", clip="clip.mp4")
    out = [o.event for r in handler.process([*events, end]) for o in r]
    assert [e.window_id for e in out] == ["m/clip/r-0/0", "m/clip/r-0/1", "m/clip/r-0/2"]
    first = out[0]
    assert first.window_start_s == 0 and first.window_end_s == 5
    assert first.observed_s == pytest.approx(4.8)  # frames at 0.0 .. 4.8
    assert first.frames == 25 and first.space == "metric"
    assert first.decoded_at_min == pytest.approx(1000.0)
    # Replaying the same events emits nothing new (ids already emitted).
    again = [o.event for r in handler.process([*events, end]) for o in r]
    assert again == []


def test_stage_uses_pixel_space_for_unknown_cameras():
    handler = IndicatorsHandler({}, window_s=60)
    out = handler.process(
        [track_event(0, 0.0, [(1, "pedestrian", (5, 5))], camera="new", stream="new/c/r-0"),
         ev.EndEvent(stream_id="new/c/r-0", camera_id="new", clip="c.mp4")]
    )  # fmt: skip
    [event] = [o.event for r in out for o in r]
    assert event.space == "pixel" and event.camera_id == "new"


def test_example_camera_configs_load():
    cameras = load_cameras(repo_root() / "configs" / "cameras")
    assert cameras, "configs/cameras/ should ship at least one example"
    for camera in cameras.values():
        if camera.calibration:
            assert camera.homography_for(*camera.frame_size) is not None


def test_calibration_needs_a_frame_size_and_valid_points():
    with pytest.raises(ValueError, match="frame_size"):
        CameraConfig(camera_id="x", calibration=PLAN_VIEW)
    with pytest.raises(ValueError):
        Calibration(
            image_points=[(0, 0), (1, 1), (2, 2), (3, 3)],
            world_points=PLAN_VIEW.world_points,
            reference="bad",
        )


def test_chart_ships_the_same_camera_configs_as_the_repository():
    # chart/crosswalk/files/cameras/ is a copy (Helm cannot read files
    # outside the chart); keep them identical.
    repo = {p.name: p.read_text() for p in (repo_root() / "configs" / "cameras").glob("*.yaml")}
    chart = {
        p.name: p.read_text()
        for p in (repo_root() / "chart" / "crosswalk" / "files" / "cameras").glob("*.yaml")
    }
    assert repo == chart
