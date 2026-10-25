"""Rule-based conflict indicators over windows of tracked video.

These are *observable indicators*, not a risk model. Nothing here predicts
crashes or says a crossing is safe. Each number is a count or a duration
that anyone could re-measure from the same footage with the same rules, and
docs/indicators.md defines every one. The leaderboard compares crossings on
these numbers as an experiment, never as advice.

Per window of video (default 60 s) and camera:

| Indicator | Metric space (calibrated) | Pixel space (no calibration) |
| --- | --- | --- |
| pedestrians, vehicles | distinct confirmed tracks seen | same |
| pedestrian/vehicle seconds | sum of time each track was visible: exposure | same |
| close approaches | pedestrian-vehicle pairs closer than 2 m | closer than 1 pedestrian height (apparent) |
| min separation | metres | pixels (apparent, not comparable across cameras) |
| path crossings | ped and vehicle paths intersect within 3 s of each other | same rule on image paths |
| TTC events | constant-velocity time to collision below 1.5 s | not computed |
| PET events | post-encroachment time below 3 s | not computed |

A pair of tracks counts at most once per window for each indicator, so a
pedestrian standing next to a parked car for 30 seconds is one close
approach, not 150.

Only foot points inside the camera's region of interest count, which keeps
the far side of the street and the pavement out of the numbers.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from crosswalk.indicators import geometry as geo
from crosswalk.indicators.camera import CameraConfig

PEDESTRIAN, VEHICLE = "pedestrian", "vehicle"


@dataclass
class TrackObs:
    label: str
    # (x, y, t) foot points in the window's space (metres or pixels).
    points: list[geo.TimedPoint] = field(default_factory=list)
    heights: list[float] = field(default_factory=list)  # box heights, pixels
    first_t: float | None = None
    last_t: float | None = None


@dataclass
class WindowResult:
    space: str
    pedestrians: int
    vehicles: int
    pedestrian_seconds: float
    vehicle_seconds: float
    close_approaches: int
    min_separation: float | None
    path_crossings: int
    ttc_events: int
    min_ttc_s: float | None
    pet_events: int
    min_pet_s: float | None


class Window:
    """Accumulates one camera's tracks over one window of video time."""

    def __init__(self, camera: CameraConfig, width: int, height: int, start_s: float, end_s: float):
        self.camera = camera
        self.start_s, self.end_s = start_s, end_s
        self.homography = camera.homography_for(width, height)
        self.roi = camera.roi_for(width, height)
        self.space = "metric" if self.homography is not None else "pixel"
        self.tracks: dict[int, TrackObs] = {}
        self.frames = 0
        self.frame_times: list[float] = []
        # Per frame: [(track_id, label, point, pixel_height)] for proximity/TTC.
        self.snapshots: list[tuple[float, list[tuple[int, str, geo.Point, float]]]] = []
        self.decoded_at_min: float | None = None

    def add_frame(
        self,
        t: float,
        decoded_at: float,
        tracks: list[tuple[int, str, tuple[float, float, float, float]]],
    ) -> None:
        self.frames += 1
        self.frame_times.append(t)
        if decoded_at and (self.decoded_at_min is None or decoded_at < self.decoded_at_min):
            self.decoded_at_min = decoded_at
        snapshot = []
        for track_id, label, box in tracks:
            x1, y1, x2, y2 = box
            foot_px = ((x1 + x2) / 2, y2)
            if self.roi is not None and not geo.point_in_polygon(foot_px, self.roi):
                continue
            point = self.homography.to_world(foot_px) if self.homography else foot_px
            if point is None:
                continue  # above the horizon: not on the road plane
            obs = self.tracks.setdefault(track_id, TrackObs(label))
            obs.points.append((point[0], point[1], t))
            obs.heights.append(y2 - y1)
            obs.first_t = t if obs.first_t is None else obs.first_t
            obs.last_t = t
            snapshot.append((track_id, label, point, y2 - y1))
        self.snapshots.append((t, snapshot))

    def observed_s(self) -> float:
        """Video time covered by frames: the window's real exposure. A window
        cut short by the end of a clip, or with frames lost to a rebalance,
        covers less than its nominal length."""
        if len(self.frame_times) < 2:
            return 0.0
        return max(self.frame_times) - min(self.frame_times)

    def compute(self) -> WindowResult:
        th = self.camera.thresholds
        metric = self.space == "metric"
        peds = {i: o for i, o in self.tracks.items() if o.label == PEDESTRIAN}
        vehs = {i: o for i, o in self.tracks.items() if o.label == VEHICLE}

        def exposure(group):
            return sum((o.last_t - o.first_t) for o in group.values() if o.first_t is not None)

        # Close approaches and minimum separation, from simultaneous positions.
        close_pairs: set[tuple[int, int]] = set()
        ttc_pairs: set[tuple[int, int]] = set()
        min_sep: float | None = None
        min_ttc: float | None = None
        history: dict[int, list[geo.TimedPoint]] = defaultdict(list)
        for t, snapshot in self.snapshots:
            for track_id, _label, point, _h in snapshot:
                history[track_id].append((point[0], point[1], t))
            ped_now = [s for s in snapshot if s[1] == PEDESTRIAN]
            veh_now = [s for s in snapshot if s[1] == VEHICLE]
            for pid, _, ppt, ph in ped_now:
                for vid, _, vpt, _vh in veh_now:
                    d = geo.distance(ppt, vpt)
                    min_sep = d if min_sep is None else min(min_sep, d)
                    limit = th.close_approach_m if metric else th.close_approach_heights * ph
                    if d < limit:
                        close_pairs.add((pid, vid))
                    if metric:
                        pv = geo.velocity(history[pid])
                        vv = geo.velocity(history[vid])
                        if pv is None or vv is None:
                            continue
                        rel_pos = (vpt[0] - ppt[0], vpt[1] - ppt[1])
                        rel_vel = (vv[0] - pv[0], vv[1] - pv[1])
                        ttc = geo.time_to_collision(rel_pos, rel_vel, th.collision_radius_m)
                        # TTC 0 means "already within the radius": that is a
                        # close approach, not a closing course, so it is not
                        # a TTC event.
                        if ttc is not None and 0 < ttc < th.ttc_s:
                            ttc_pairs.add((pid, vid))
                            min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)

        # Path crossings and PET, from whole paths.
        crossings = 0
        pet_events = 0
        min_pet: float | None = None
        for po in peds.values():
            for vo in vehs.values():
                hit = geo.first_crossing(po.points, vo.points)
                if hit is None:
                    continue
                _point, t_p, t_v = hit
                gap = abs(t_p - t_v)
                if gap <= th.crossing_window_s:
                    crossings += 1
                if metric:
                    min_pet = gap if min_pet is None else min(min_pet, gap)
                    if gap < th.pet_s:
                        pet_events += 1

        return WindowResult(
            space=self.space,
            pedestrians=len(peds),
            vehicles=len(vehs),
            pedestrian_seconds=round(exposure(peds), 3),
            vehicle_seconds=round(exposure(vehs), 3),
            close_approaches=len(close_pairs),
            min_separation=None if min_sep is None else round(min_sep, 3),
            path_crossings=crossings,
            ttc_events=len(ttc_pairs) if metric else 0,
            min_ttc_s=None if (min_ttc is None or not metric) else round(min_ttc, 3),
            pet_events=pet_events if metric else 0,
            min_pet_s=None if (min_pet is None or not metric) else round(min_pet, 3),
        )


def window_bounds(t: float, length_s: float) -> tuple[int, float, float]:
    """Index and [start, end) of the fixed window containing time ``t``.

    Windows are aligned to multiples of ``length_s`` from the start of the
    clip, so a replayed clip produces windows with the same ids.
    """
    index = math.floor(t / length_s) if t >= 0 else 0
    return index, index * length_s, (index + 1) * length_s
