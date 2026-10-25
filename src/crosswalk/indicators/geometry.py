"""Geometry for the conflict indicators: homography, paths, TTC and PET.

Everything works on *foot points*: the bottom centre of a tracked box, which
is roughly where a pedestrian or a wheel meets the road. With a homography
those points map to metres on the road plane; without one they stay in
pixels, and only quantities that make sense in pixels are computed.

All functions are pure and small, so each is tested against an independent
calculation (tests/test_indicators.py).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Point = tuple[float, float]


class DegenerateCalibration(ValueError):
    pass


class Homography:
    """Image (pixels) to road plane (metres), from four point pairs.

    Four correspondences determine a plane-to-plane homography exactly
    (eight unknowns, ``h33 = 1``). The usual reference is the four corners of
    a crosswalk whose width and length are known or measured, e.g. on a
    public map. No three points may be collinear.
    """

    def __init__(self, matrix: np.ndarray, road_sign: float = 1.0):
        self.H = np.asarray(matrix, dtype=float).reshape(3, 3)
        # The horizon is the image line where the homogeneous W is 0. Points
        # on the road side share the sign W has at the reference points, which
        # depends on how H happens to be normalised, so it is recorded here.
        self.road_sign = 1.0 if road_sign >= 0 else -1.0

    @classmethod
    def from_points(cls, image: Sequence[Point], world: Sequence[Point]) -> Homography:
        if len(image) != 4 or len(world) != 4:
            raise DegenerateCalibration("a homography needs exactly four point pairs")
        for pts, name in ((image, "image"), (world, "world")):
            for i in range(4):
                a, b, c = (np.asarray(pts[j], dtype=float) for j in (i, (i + 1) % 4, (i + 2) % 4))
                area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
                if area < 1e-6:
                    raise DegenerateCalibration(f"three {name} points are collinear")
        A, rhs = [], []
        for (x, y), (X, Y) in zip(image, world, strict=True):
            A.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
            A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
            rhs.extend([X, Y])
        try:
            h = np.linalg.solve(np.array(A, dtype=float), np.array(rhs, dtype=float))
        except np.linalg.LinAlgError as err:
            raise DegenerateCalibration(
                f"calibration points do not define a homography: {err}"
            ) from err
        matrix = np.append(h, 1.0).reshape(3, 3)
        cx = sum(p[0] for p in image) / 4
        cy = sum(p[1] for p in image) / 4
        w_center = float((matrix @ np.array([cx, cy, 1.0]))[2])
        return cls(matrix, road_sign=w_center)

    def to_world(self, point: Point) -> Point | None:
        """Map one image point. ``None`` if it lies at or beyond the horizon
        (the homography sends it to infinity or behind the camera)."""
        x, y = point
        X, Y, W = self.H @ np.array([x, y, 1.0])
        if W * self.road_sign <= 1e-9:
            return None
        return (float(X / W), float(Y / W))

    def scaled(self, sx: float, sy: float) -> Homography:
        """The same homography for frames resized by (sx, sy) from the
        calibration frame size."""
        S_inv = np.diag([1 / sx, 1 / sy, 1.0])
        return Homography(self.H @ S_inv, road_sign=self.road_sign)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray casting; points on an edge may go either way, which is fine for a
    region of interest drawn by hand."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < cross:
                inside = not inside
    return inside


def segment_intersection(
    p1: Point, p2: Point, q1: Point, q2: Point
) -> tuple[Point, float, float] | None:
    """Where segments p1-p2 and q1-q2 cross, with the fractions along each.

    Returns ``(point, s, u)`` with ``point = p1 + s (p2 - p1) = q1 + u (q2 - q1)``
    and ``0 <= s, u <= 1``; ``None`` if they do not cross (parallel segments
    never count, even if they overlap: two objects moving along the same line
    are a following situation, not a crossing).
    """
    r = (p2[0] - p1[0], p2[1] - p1[1])
    d = (q2[0] - q1[0], q2[1] - q1[1])
    denom = r[0] * d[1] - r[1] * d[0]
    if abs(denom) < 1e-12:
        return None
    qp = (q1[0] - p1[0], q1[1] - p1[1])
    s = (qp[0] * d[1] - qp[1] * d[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if not (0.0 <= s <= 1.0 and 0.0 <= u <= 1.0):
        return None
    return (p1[0] + s * r[0], p1[1] + s * r[1]), s, u


TimedPoint = tuple[float, float, float]  # x, y, t


def first_crossing(
    a: Sequence[TimedPoint], b: Sequence[TimedPoint]
) -> tuple[Point, float, float] | None:
    """First point where path ``a`` crosses path ``b``, with the time each
    object was there (linear interpolation within the crossing segments).

    Returns ``(point, t_a, t_b)``; the post-encroachment time is
    ``abs(t_a - t_b)``.
    """
    best = None
    for i in range(len(a) - 1):
        for j in range(len(b) - 1):
            hit = segment_intersection(a[i][:2], a[i + 1][:2], b[j][:2], b[j + 1][:2])
            if hit is None:
                continue
            point, s, u = hit
            t_a = a[i][2] + s * (a[i + 1][2] - a[i][2])
            t_b = b[j][2] + u * (b[j + 1][2] - b[j][2])
            when = max(t_a, t_b)  # the crossing is complete once both passed
            if best is None or when < best[0]:
                best = (when, (point, t_a, t_b))
    return best[1] if best else None


def time_to_collision(rel_pos: Point, rel_vel: Point, radius: float) -> float | None:
    """Time until two objects come within ``radius`` of each other if both
    keep their current velocities.

    ``rel_pos`` and ``rel_vel`` are the second object's position and velocity
    relative to the first. Solves ``|p + v t| = radius`` for the smallest
    ``t >= 0``. Returns 0 if they are already within ``radius``, and ``None``
    if, at constant velocity, they never will be (not on a collision course).
    """
    px, py = rel_pos
    vx, vy = rel_vel
    c = px * px + py * py - radius * radius
    if c <= 0:
        return 0.0
    a = vx * vx + vy * vy
    b = 2 * (px * vx + py * vy)
    if a < 1e-12 or b >= 0:
        return None  # not moving relative to each other, or moving apart
    disc = b * b - 4 * a * c
    if disc < 0:
        return None  # closest approach is still outside the radius
    return (-b - math.sqrt(disc)) / (2 * a)


def velocity(
    points: Sequence[TimedPoint], window_s: float = 1.0, max_points: int = 6
) -> Point | None:
    """Least-squares velocity over the last ``max_points`` points within
    ``window_s`` seconds: exact for straight constant-speed motion, and much
    less noisy than the last difference on real detections."""
    if len(points) < 2:
        return None
    t_end = points[-1][2]
    recent = [p for p in points[-max_points:] if t_end - p[2] <= window_s]
    if len(recent) < 2:
        recent = list(points[-2:])
    t = np.array([p[2] for p in recent])
    if np.ptp(t) <= 0:
        return None
    tc = t - t.mean()
    denom = float(tc @ tc)
    vx = float(tc @ np.array([p[0] for p in recent])) / denom
    vy = float(tc @ np.array([p[1] for p in recent])) / denom
    return (vx, vy)


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
