"""A generated crossing for demos and integration tests.

A plan view (looking straight down) of a two-lane road with a zebra
crossing, drawn at 20 pixels per metre: 320 x 192 pixels is 16 m x 9.6 m.
Pedestrians walk across, vehicles drive along the road, and ``yielding``
decides whether vehicles stop for people on the crossing. Two scenes with
different settings give the leaderboard something to compare.

Everything is exact by construction, which is the point: the
``synthetic-demo`` camera's calibration (``configs/cameras/synthetic-demo.yaml``)
maps these pixels to metres exactly, so the metric indicators (TTC, PET) are
exercised end to end, and :class:`SceneDetector` finds the drawn objects by
colour, so the pipeline can run without a GPU or model weights.

This is *not* footage and the numbers it produces mean nothing about any
real place. The demo labels its crossings as generated.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from crosswalk.inference.predictor import Detection

PX_PER_M = 20.0
WIDTH, HEIGHT = 320, 192
ROAD_TOP_M, ROAD_BOTTOM_M = 3.0, 7.2
CROSSWALK_X_M = (7.0, 9.0)
LANES = {"east": 7.0, "west": 5.0}  # bottom edge (y, metres) of each lane's vehicles

PAVEMENT = (128, 128, 124)
ROAD = (84, 86, 90)
STRIPE = (226, 226, 222)
PEDESTRIAN = (40, 90, 235)
VEHICLE = (235, 55, 40)

PED_W, PED_H = 10, 24  # pixels
CAR_W, CAR_H = 60, 28


@dataclass
class Walker:
    x: float
    y: float
    vy: float
    start_s: float


@dataclass
class Car:
    lane: str
    x: float  # centre, metres
    speed: float  # m/s, signed by direction
    start_s: float
    stopped: bool = False


@dataclass
class SceneConfig:
    name: str = "synthetic-demo"
    seconds: float = 60.0
    fps: float = 10.0
    seed: int = 0
    pedestrians_per_min: float = 8.0
    vehicles_per_min: float = 14.0
    yielding: bool = True


@dataclass
class Frame:
    index: int
    pts_s: float
    image: np.ndarray
    truth: list[tuple[str, tuple[float, float, float, float]]] = field(default_factory=list)


def _box_ped(w: Walker) -> tuple[float, float, float, float]:
    cx, by = w.x * PX_PER_M, w.y * PX_PER_M
    return (cx - PED_W / 2, by - PED_H, cx + PED_W / 2, by)


def _box_car(c: Car) -> tuple[float, float, float, float]:
    cx, by = c.x * PX_PER_M, LANES[c.lane] * PX_PER_M
    return (cx - CAR_W / 2, by - CAR_H, cx + CAR_W / 2, by)


def _background() -> np.ndarray:
    img = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    img[:] = PAVEMENT
    top, bottom = int(ROAD_TOP_M * PX_PER_M), int(ROAD_BOTTOM_M * PX_PER_M)
    img[top:bottom] = ROAD
    x0, x1 = (int(v * PX_PER_M) for v in CROSSWALK_X_M)
    for y in range(top + 4, bottom - 4, 12):  # zebra bars across the walking direction
        img[y : y + 6, x0:x1] = STRIPE
    return img


def _draw(img: np.ndarray, box, colour) -> None:
    x1, y1, x2, y2 = (round(v) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, WIDTH), min(y2, HEIGHT)
    if x2 > x1 and y2 > y1:
        img[y1:y2, x1:x2] = colour


def frames(config: SceneConfig) -> Iterator[Frame]:
    """Simulate and render the scene, one frame every 1/fps seconds."""
    rng = np.random.default_rng(config.seed)
    background = _background()
    dt = 1.0 / config.fps
    walkers: list[Walker] = []
    cars: list[Car] = []
    p_ped = config.pedestrians_per_min / 60.0 * dt
    p_car = config.vehicles_per_min / 60.0 * dt
    n = int(config.seconds * config.fps)
    for i in range(n):
        t = i * dt
        # Arrivals (Bernoulli per step approximates a Poisson process).
        if rng.random() < p_ped:
            north = rng.random() < 0.5
            walkers.append(
                Walker(
                    x=float(rng.uniform(CROSSWALK_X_M[0] + 0.3, CROSSWALK_X_M[1] - 0.3)),
                    y=9.4 if north else 1.2,
                    vy=float(rng.uniform(1.1, 1.6)) * (-1 if north else 1),
                    start_s=t,
                )
            )
        if rng.random() < p_car:
            lane = "east" if rng.random() < 0.5 else "west"
            speed = float(rng.uniform(7.0, 11.0))
            entry = -2.0 if lane == "east" else 18.0
            same_lane = [c for c in cars if c.lane == lane]
            if all(abs(c.x - entry) > 6.0 for c in same_lane):  # room to enter
                cars.append(Car(lane, entry, speed if lane == "east" else -speed, t))

        # Move pedestrians.
        for w in walkers:
            w.y += w.vy * dt
        walkers = [w for w in walkers if 0.6 <= w.y <= 10.0]

        # Move vehicles: keep a gap to the car ahead, and (if yielding) stop
        # before the crossing while anyone is on the road part of it.
        on_road = [w for w in walkers if ROAD_TOP_M - 0.5 <= w.y <= ROAD_BOTTOM_M + 0.5]
        for c in cars:
            front = c.x + np.sign(c.speed) * CAR_W / PX_PER_M / 2
            stop = False
            if config.yielding and on_road:
                if c.speed > 0 and CROSSWALK_X_M[0] - 5.0 < front < CROSSWALK_X_M[0]:
                    stop = True
                if c.speed < 0 and CROSSWALK_X_M[1] < front < CROSSWALK_X_M[1] + 5.0:
                    stop = True
            ahead = [
                o
                for o in cars
                if o is not c and o.lane == c.lane and (o.x - c.x) * np.sign(c.speed) > 0
            ]
            if ahead and min(abs(o.x - c.x) for o in ahead) < 4.0:
                stop = True
            c.stopped = stop
            if not stop:
                c.x += c.speed * dt
        cars = [c for c in cars if -4.0 <= c.x <= 20.0]

        img = background.copy()
        truth = []
        for w in walkers:
            box = _box_ped(w)
            _draw(img, box, PEDESTRIAN)
            truth.append(("pedestrian", box))
        for c in cars:  # vehicles drawn last: they hide people behind them
            box = _box_car(c)
            _draw(img, box, VEHICLE)
            truth.append(("vehicle", box))
        yield Frame(i, round(t, 6), img, truth)


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Connected components of a boolean mask (4-connectivity) as
    (x1, y1, x2, y2, area), by iterated max-label propagation."""
    if not mask.any():
        return []
    labels = np.where(mask, np.arange(1, mask.size + 1).reshape(mask.shape), 0)
    while True:
        grown = labels.copy()
        grown[1:] = np.maximum(grown[1:], labels[:-1])
        grown[:-1] = np.maximum(grown[:-1], labels[1:])
        grown[:, 1:] = np.maximum(grown[:, 1:], labels[:, :-1])
        grown[:, :-1] = np.maximum(grown[:, :-1], labels[:, 1:])
        grown = np.where(mask, grown, 0)
        if np.array_equal(grown, labels):
            break
        labels = grown
    out = []
    for label in np.unique(labels[labels > 0]):
        ys, xs = np.nonzero(labels == label)
        out.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, len(xs)))
    return out


class SceneDetector:
    """Finds the generated scene's objects by colour: a stand-in for the
    detector so the pipeline runs on CPU in seconds. It is only meaningful
    on frames from :func:`frames`; on real footage use the FCOS predictor.

    Implements the same :class:`~crosswalk.inference.predictor.Predictor`
    protocol as ``TorchPredictor``, labelling with COCO names (``person``,
    ``car``) like the pretrained model, so tracking's class mapping is
    exercised as in production.
    """

    model_version = "scene-colour-detector:0"

    def __init__(self, min_area: int = 30):
        self.min_area = min_area

    def predict(self, images: list[np.ndarray]) -> list[list[Detection]]:
        results = []
        for img in images:
            rgb = img.astype(np.int16)
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            # Margins absorb JPEG and video compression of the flat colours.
            blue = (b > 160) & (r < 110) & (g < 150)
            red = (r > 160) & (b < 110) & (g < 120)
            dets = []
            for mask, label, label_id in ((blue, "person", 1), (red, "car", 3)):
                for x1, y1, x2, y2, area in _components(mask):
                    if area >= self.min_area:
                        dets.append(
                            Detection(
                                label, label_id, 0.9, (float(x1), float(y1), float(x2), float(y2))
                            )
                        )
            results.append(dets)
        return results


def write_video(path: str, config: SceneConfig, encoder: str = "av") -> str:
    """Encode the scene as a short clip (for the video-decoding demo paths).

    ``encoder="av"`` uses PyAV (a Python dependency already); ``"ffmpeg"``
    pipes raw frames to the ffmpeg binary, which the Rust sampler path needs
    on PATH anyway.
    """
    if encoder == "ffmpeg":
        import subprocess

        proc = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(config.fps),
             "-i", "pipe:0", "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE,
        )  # fmt: skip
        for f in frames(config):
            proc.stdin.write(f.image.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {path}")
        return path

    from fractions import Fraction

    import av

    with av.open(path, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=int(config.fps))
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "yuv420p"
        # Generous bitrate: the colour detector needs the flat colours intact.
        stream.bit_rate = 4_000_000
        for f in frames(config):
            frame = av.VideoFrame.from_ndarray(f.image, format="rgb24")
            frame.pts = f.index
            frame.time_base = Fraction(1, int(config.fps))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def now() -> float:
    return time.time()
