"""Per-camera configuration: which crossing a camera watches, and how to
turn its pixels into metres when that is possible.

One YAML file per camera in ``configs/cameras/``. Every field except
``camera_id`` is optional. A camera with no ``calibration`` still gets
counts, exposure and pixel-space indicators; TTC and PET need metric
calibration and are simply not computed without it (never estimated from
pixels).

Example (``configs/cameras/example-crossing.yaml``)::

    camera_id: example-crossing
    crossing: Example St & 1st Ave, north crosswalk
    frame_size: [1280, 720]       # size the points below were picked on
    roi: [[100, 400], [1180, 400], [1280, 720], [0, 720]]
    calibration:
      image_points: [[420, 430], [860, 430], [940, 560], [340, 560]]
      world_points: [[0, 0], [12.0, 0], [12.0, 3.0], [0, 3.0]]   # metres
      reference: "crosswalk corners, 12.0 m x 3.0 m, measured on the city's open map"
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, model_validator

from crosswalk.config import StrictModel
from crosswalk.indicators.geometry import Homography

Pt = tuple[float, float]


class Calibration(StrictModel):
    # Four image points (pixels, on a frame of `frame_size`) and the same
    # four points on the road plane in metres. Any consistent origin works;
    # a crosswalk corner is convenient.
    image_points: list[Pt] = Field(min_length=4, max_length=4)
    world_points: list[Pt] = Field(min_length=4, max_length=4)
    # Where the world measurements came from. Required so a reader can judge
    # how far to trust metric numbers.
    reference: str = Field(min_length=3)

    @model_validator(mode="after")
    def _valid(self) -> Calibration:
        Homography.from_points(self.image_points, self.world_points)  # raises if degenerate
        return self


class Thresholds(StrictModel):
    """Where an observation becomes an *event*. Defaults are conventional
    starting points from the surrogate-safety literature, not tuned values;
    docs/indicators.md explains each."""

    # Metric space (with calibration).
    close_approach_m: float = Field(2.0, gt=0)
    ttc_s: float = Field(1.5, gt=0)  # FHWA SSAM's default maximum TTC
    pet_s: float = Field(3.0, gt=0)
    collision_radius_m: float = Field(1.0, gt=0)
    # Pixel space (no calibration): proximity relative to the pedestrian's
    # apparent height, so it scales with distance from the camera.
    close_approach_heights: float = Field(1.0, gt=0)
    # Pixel-space PET-like window: two paths crossing within this many
    # seconds counts as a path crossing event.
    crossing_window_s: float = Field(3.0, gt=0)


class CameraConfig(StrictModel):
    camera_id: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    crossing: str = ""  # human-readable name for the leaderboard
    location: str = ""  # city/region shown on the leaderboard
    frame_size: tuple[int, int] | None = None
    # Region of interest: only foot points inside this polygon count. Draw it
    # around the crosswalk and its approaches; leave it out to use the frame.
    roi: list[Pt] | None = Field(None, min_length=3)
    calibration: Calibration | None = None
    thresholds: Thresholds = Field(default_factory=Thresholds)
    # Footage provenance shown on the leaderboard (see docs/data-sources.md).
    source: str = ""
    license: str = ""
    attribution: str = ""

    @model_validator(mode="after")
    def _calibration_needs_a_frame_size(self) -> CameraConfig:
        if self.calibration is not None and self.frame_size is None:
            raise ValueError("calibration needs frame_size (the size the points were picked on)")
        return self

    @property
    def metric(self) -> bool:
        return self.calibration is not None

    def homography_for(self, width: int, height: int) -> Homography | None:
        """The homography for frames of this size (frames may be resized
        between calibration and inference)."""
        if self.calibration is None:
            return None
        H = Homography.from_points(self.calibration.image_points, self.calibration.world_points)
        fw, fh = self.frame_size
        if (fw, fh) == (width, height):
            return H
        return H.scaled(width / fw, height / fh)

    def roi_for(self, width: int, height: int) -> list[Pt] | None:
        if self.roi is None:
            return None
        if self.frame_size is None or tuple(self.frame_size) == (width, height):
            return list(self.roi)
        sx, sy = width / self.frame_size[0], height / self.frame_size[1]
        return [(x * sx, y * sy) for x, y in self.roi]


def load_cameras(directory: str | Path | None) -> dict[str, CameraConfig]:
    """Every ``*.yaml`` in a directory, keyed by camera id (duplicates fail)."""
    cameras: dict[str, CameraConfig] = {}
    if directory is None or not Path(directory).is_dir():
        return cameras
    for path in sorted(Path(directory).glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        camera = CameraConfig.model_validate(data)
        if camera.camera_id in cameras:
            raise ValueError(f"{path}: camera_id {camera.camera_id!r} is defined twice")
        cameras[camera.camera_id] = camera
    return cameras
