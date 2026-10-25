"""Multi-object tracking: ByteTrack-style association over Kalman-predicted boxes.

Tracking follows a box from frame to frame within one stream (one pass over
one clip) so the indicators can count distinct pedestrians and vehicles and
reason about their paths. It never links tracks across streams or clips, and
it never tries to recognise anyone (docs/data.md, "No re-identification").

The algorithm is ByteTrack's (Zhang et al., 2022), which is simple and does
well on crowded scenes without an appearance model:

1. Predict every track's box to the new frame's timestamp (Kalman filter).
2. Split detections into **high** (score >= ``high_threshold``) and **low**.
3. Match high detections to all active tracks on IoU (Hungarian algorithm).
4. Match low detections to the tracks still unmatched. Low-score boxes are
   usually real objects that are partly hidden; using them keeps tracks alive
   through occlusion instead of fragmenting them. Low boxes never start tracks.
5. Unmatched high detections start **tentative** tracks, which become
   **confirmed** after ``min_hits`` consecutive matches (filters one-frame
   false positives).
6. Tracks unmatched for longer than ``max_age_s`` seconds of video are ended.

Classes are matched separately per group (pedestrians with pedestrians,
vehicles with vehicles), so a person walking past a car never swaps identity
with it.

Times are seconds of video (``pts_s``), not frame counts, because the sampler
keeps frames by time and may drop some.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from crosswalk.tracking.assignment import iou_matrix, match
from crosswalk.tracking.kalman import KalmanBox

# Detector labels (COCO names from the pretrained model, project names from a
# fine-tuned one) grouped into the two classes the indicators care about.
CLASS_GROUPS: dict[str, str] = {
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",  # treated as a vehicle: it is what a pedestrian crosses in front of
}


def class_group(label: str) -> str | None:
    return CLASS_GROUPS.get(label)


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float
    label: str  # already grouped: "pedestrian" or "vehicle"


@dataclass(frozen=True)
class TrackerConfig:
    high_threshold: float = 0.5
    low_threshold: float = 0.1
    # IoU needed to continue a track with a high / low detection.
    match_iou: float = 0.3
    low_match_iou: float = 0.5
    min_hits: int = 2
    max_age_s: float = 1.0
    # Ignore boxes smaller than this (pixels of height): far-away noise.
    min_box_height: float = 4.0


@dataclass
class Track:
    track_id: int
    label: str
    kf: KalmanBox
    score: float
    first_seen_s: float
    last_seen_s: float
    hits: int = 1
    confirmed: bool = False
    # Bottom-centre points (x, y, t) of the matched boxes: the path on the ground.
    history: list[tuple[float, float, float]] = field(default_factory=list)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.kf.box

    def foot(self) -> tuple[float, float]:
        x1, _y1, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)


@dataclass(frozen=True)
class TrackState:
    """What the tracker reports for one confirmed track in one frame."""

    track_id: int
    label: str
    box: tuple[float, float, float, float]
    score: float
    velocity: tuple[float, float]  # bottom centre, pixels per second
    hits: int
    age_s: float


class Tracker:
    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self.next_id = 1
        self.last_t: float | None = None
        self.finished: list[Track] = []

    def update(
        self, detections: Iterable[Detection], t: float
    ) -> tuple[list[TrackState], list[Track]]:
        """Advance to time ``t`` (seconds) with this frame's detections.

        Returns (confirmed tracks visible in this frame, tracks that ended).
        A frame with an earlier timestamp than the last one is ignored (it is
        a replay or out-of-order delivery; the stage reorders before this).
        """
        cfg = self.config
        if self.last_t is not None and t < self.last_t:
            return [], []
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        for track in self.tracks:
            track.kf.predict(dt)

        dets = [
            d
            for d in detections
            if d.score >= cfg.low_threshold and d.box[3] - d.box[1] >= cfg.min_box_height
        ]
        matched_tracks: set[int] = set()
        for label in sorted({d.label for d in dets} | {tr.label for tr in self.tracks}):
            tracks = [i for i, tr in enumerate(self.tracks) if tr.label == label]
            group = [d for d in dets if d.label == label]
            high = [d for d in group if d.score >= cfg.high_threshold]
            low = [d for d in group if d.score < cfg.high_threshold]

            # Stage 1: high-score detections against every track of this class.
            pairs, left_tracks, left_high = self._associate(tracks, high, cfg.match_iou)
            for ti, di in pairs:
                self._hit(self.tracks[ti], high[di], t)
                matched_tracks.add(ti)
            # Stage 2: low-score detections keep the remaining tracks alive.
            pairs, _still_left, _unused_low = self._associate(left_tracks, low, cfg.low_match_iou)
            for ti, di in pairs:
                self._hit(self.tracks[ti], low[di], t)
                matched_tracks.add(ti)
            # Unmatched confident detections start new (tentative) tracks.
            for di in left_high:
                self._start(high[di], t)

        # Tentative tracks must be matched in consecutive frames; confirmed
        # ones survive misses until max_age_s.
        ended: list[Track] = []
        alive: list[Track] = []
        for i, track in enumerate(self.tracks):
            if i in matched_tracks or track.last_seen_s == t:
                alive.append(track)
            elif not track.confirmed or t - track.last_seen_s > cfg.max_age_s:
                if track.confirmed:
                    ended.append(track)
            else:
                alive.append(track)
        self.tracks = alive
        self.finished.extend(ended)
        visible = [self._state(tr, t) for tr in self.tracks if tr.confirmed and tr.last_seen_s == t]
        return visible, ended

    def flush(self) -> list[Track]:
        """End every confirmed track (end of stream)."""
        ended = [tr for tr in self.tracks if tr.confirmed]
        self.finished.extend(ended)
        self.tracks = []
        return ended

    # -- internals ------------------------------------------------------------------

    def _associate(self, track_indices: Sequence[int], dets: Sequence[Detection], threshold: float):
        if not track_indices or not dets:
            return [], list(track_indices), list(range(len(dets)))
        boxes_t = np.array([self.tracks[i].box for i in track_indices])
        boxes_d = np.array([d.box for d in dets])
        pairs, left_t, left_d = match(iou_matrix(boxes_t, boxes_d), threshold)
        return [(track_indices[r], c) for r, c in pairs], [track_indices[r] for r in left_t], left_d

    def _hit(self, track: Track, det: Detection, t: float) -> None:
        track.kf.update(det.box)
        track.score = det.score
        track.hits += 1
        track.last_seen_s = t
        if track.hits >= self.config.min_hits:
            track.confirmed = True
        x1, _y1, x2, y2 = det.box
        track.history.append(((x1 + x2) / 2, y2, t))

    def _start(self, det: Detection, t: float) -> None:
        x1, _y1, x2, y2 = det.box
        track = Track(
            track_id=self.next_id,
            label=det.label,
            kf=KalmanBox(det.box),
            score=det.score,
            first_seen_s=t,
            last_seen_s=t,
            confirmed=self.config.min_hits <= 1,
            history=[((x1 + x2) / 2, y2, t)],
        )
        self.next_id += 1
        self.tracks.append(track)

    def _state(self, track: Track, t: float) -> TrackState:
        return TrackState(
            track_id=track.track_id,
            label=track.label,
            box=tuple(round(v, 1) for v in track.box),
            score=round(track.score, 4),
            velocity=tuple(round(v, 2) for v in track.kf.foot_velocity),
            hits=track.hits,
            age_s=round(t - track.first_seen_s, 4),
        )
