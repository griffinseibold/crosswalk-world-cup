"""Event schemas for every topic, validated with pydantic.

Each event is a small JSON object with ``schema`` (format version) and
``type``. Frame events come from the Rust sampler (``ingest/src/event.rs``);
``tests/test_stream_events.py`` parses the sampler's checked-in example so the
two definitions cannot drift apart silently.

Identity rules, which idempotent processing relies on:

- ``stream_id``: one pass over one clip by one sampler process.
- ``frame_id``: ``<stream_id>/<frame_index>``; unique per frame. Detections
  carry the frame id they came from, so a redelivered frame produces a
  detection event with the same id and downstream stages can drop it.
- Every topic is keyed by ``camera_id``, so all events of one camera share a
  partition and arrive in order relative to each other.

Unknown fields are ignored (not rejected) on purpose: a producer can add a
field without breaking consumers that do not know it yet. A breaking change
gets a new topic version (``.v2``) instead.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SCHEMA_VERSION = 1

TOPIC_FRAMES = "crosswalk.frames.v1"
TOPIC_DETECTIONS = "crosswalk.detections.v1"
TOPIC_TRACKS = "crosswalk.tracks.v1"
TOPIC_INDICATORS = "crosswalk.indicators.v1"
# Dead-letter topics: events a stage could not process after its retries,
# wrapped with the error. One per consuming stage, so a DLQ count points at
# the stage that failed.
TOPIC_DLQ_SUFFIX = ".dlq"


def dlq_topic(stage: str) -> str:
    return f"crosswalk.{stage}{TOPIC_DLQ_SUFFIX}"


class Event(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    schema_version: int = Field(SCHEMA_VERSION, alias="schema")

    def to_bytes(self) -> bytes:
        return self.model_dump_json(by_alias=True).encode()


class ObjectRef(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    bucket: str
    key: str


class FrameEvent(Event):
    type: Literal["frame"] = "frame"
    frame_id: str
    stream_id: str
    camera_id: str
    clip: str
    frame_index: int
    sample_index: int = 0
    pts_s: float
    width: int
    height: int
    decoded_at: float
    published_at: float = 0.0
    image: ObjectRef
    content_type: str = "image/jpeg"
    bytes: int = 0
    sha256: str = ""


class EndEvent(Event):
    """End of one stream (pass over a clip); passed along by every stage."""

    type: Literal["end"] = "end"
    stream_id: str
    camera_id: str
    clip: str
    frames: int = 0
    dropped: int = 0
    last_pts_s: float = 0.0
    complete: bool = True
    published_at: float = 0.0


class Box(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    label: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2, frame pixels


class DetectionEvent(Event):
    type: Literal["detections"] = "detections"
    frame_id: str
    stream_id: str
    camera_id: str
    frame_index: int
    pts_s: float
    width: int
    height: int
    decoded_at: float
    model_version: str
    detections: list[Box]
    # Seconds spent in the inference stage: queue wait, model, total.
    latency: dict[str, float] = Field(default_factory=dict)


class TrackPoint(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    track_id: int
    label: str  # "pedestrian" or "vehicle" (see tracking.CLASS_GROUPS)
    box: tuple[float, float, float, float]
    score: float
    # Kalman-filtered velocity of the box's bottom centre, pixels per second.
    velocity: tuple[float, float] = (0.0, 0.0)
    age: int = 1  # frames this track has been confirmed for
    hits: int = 1


class TrackEvent(Event):
    """Confirmed tracks visible in one frame (after the tracker's update)."""

    type: Literal["tracks"] = "tracks"
    frame_id: str
    stream_id: str
    camera_id: str
    frame_index: int
    pts_s: float
    width: int
    height: int
    decoded_at: float
    tracks: list[TrackPoint]
    ended: list[int] = Field(default_factory=list)  # track ids that ended here


class IndicatorEvent(Event):
    """Rule-based conflict indicators for one camera over one window of video.

    ``space`` says what the distances and speeds are measured in: ``metric``
    only when the camera has a hand-set homography, otherwise ``pixel``.
    Pixel-space values are apparent quantities and are labelled as such
    everywhere they are shown.
    """

    type: Literal["indicators"] = "indicators"
    window_id: str  # <stream_id>/<window index>; idempotency key
    stream_id: str
    camera_id: str
    window_start_s: float
    window_end_s: float
    observed_s: float  # video seconds actually covered by frames
    frames: int
    space: Literal["metric", "pixel"]
    pedestrians: int  # distinct pedestrian tracks seen
    vehicles: int  # distinct vehicle tracks seen
    pedestrian_seconds: float  # exposure: sum over pedestrian tracks of time visible
    vehicle_seconds: float
    close_approaches: int  # pedestrian-vehicle pairs closer than the threshold
    min_separation: float | None  # metres (metric) or pixels (pixel)
    path_crossings: int  # trajectory intersection events
    ttc_events: int = 0  # time-to-collision below threshold (metric only)
    min_ttc_s: float | None = None
    pet_events: int = 0  # post-encroachment time below threshold (metric only)
    min_pet_s: float | None = None
    decoded_at_min: float = 0.0  # oldest frame's decode time, for end-to-end latency
    produced_at: float = 0.0


AnyEvent = Annotated[
    FrameEvent | EndEvent | DetectionEvent | TrackEvent | IndicatorEvent,
    Field(discriminator="type"),
]
_ADAPTER: TypeAdapter[Any] = TypeAdapter(AnyEvent)


def parse(data: bytes | str | dict[str, Any]) -> Event:
    """Parse any event by its ``type``. Raises ``ValueError`` if invalid."""
    if isinstance(data, bytes | str):
        data = json.loads(data)
    return _ADAPTER.validate_python(data)


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def sanitize(text: str) -> str:
    cleaned = _UNSAFE.sub("-", text).strip("-.")
    return cleaned or "unnamed"


def camera_id_from_clip(file_name: str) -> str:
    """Same rule as the Rust sampler: the stem up to the first ``__``."""
    stem, dot, _ext = file_name.rpartition(".")
    if not dot or not stem:
        stem = file_name
    return sanitize(stem.split("__", 1)[0])
