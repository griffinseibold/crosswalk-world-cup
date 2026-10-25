"""The tracking stage: detection events in, track events out.

One :class:`~crosswalk.tracking.tracker.Tracker` per stream (one pass over one
clip). Two details make it correct on a real event stream:

- **Reordering.** The sampler publishes with several parallel tasks, so a
  camera's frames can reach Kafka slightly out of order. Each stream keeps a
  small buffer (``reorder_window`` frames) and feeds the tracker in
  ``frame_index`` order. A frame older than one already fed is too late: it
  is counted and skipped, never fed backwards in time.
- **Stream end.** An ``end`` event flushes the buffer, ends every track and
  forgets the stream, then is forwarded so the indicators stage can close its
  windows. A stream that goes quiet without an end event (a sampler killed
  hard) is flushed after ``idle_timeout_s`` of wall-clock time, so memory
  does not grow without bound.

Frames waiting in the reorder buffer have had their input offsets committed
before their track events are produced. A crash at that moment loses tracking
for those few frames (at most ``reorder_window`` per stream). That is the
price of reordering without a persistent state store; docs/streaming.md
discusses it.
"""

from __future__ import annotations

import heapq
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from crosswalk.stream import events as ev
from crosswalk.stream.stage import Output, outputs_for
from crosswalk.tracking.tracker import Detection, Tracker, TrackerConfig, class_group

log = logging.getLogger(__name__)


@dataclass
class _Stream:
    tracker: Tracker
    buffer: list[tuple[int, int, ev.DetectionEvent]] = field(default_factory=list)  # heap
    last_fed: int = -1
    last_activity: float = 0.0
    seq: int = 0
    late: int = 0


class TrackingHandler:
    name = "tracking"

    def __init__(
        self,
        tracker_config: TrackerConfig | None = None,
        reorder_window: int = 8,
        idle_timeout_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
        on_late: Callable[[int], None] | None = None,
    ):
        self.tracker_config = tracker_config or TrackerConfig()
        self.reorder_window = reorder_window
        self.idle_timeout_s = idle_timeout_s
        self.clock = clock
        self.on_late = on_late
        self.streams: dict[str, _Stream] = {}

    def event_id(self, event: ev.Event) -> str | None:
        if isinstance(event, ev.DetectionEvent):
            return event.frame_id
        return None

    def _stream(self, stream_id: str) -> _Stream:
        if stream_id not in self.streams:
            self.streams[stream_id] = _Stream(Tracker(self.tracker_config))
        state = self.streams[stream_id]
        state.last_activity = self.clock()
        return state

    def _feed(self, state: _Stream, det: ev.DetectionEvent) -> ev.TrackEvent:
        detections = []
        for box in det.detections:
            group = class_group(box.label)
            if group is not None:
                detections.append(Detection(tuple(box.box), box.score, group))
        visible, ended = state.tracker.update(detections, det.pts_s)
        state.last_fed = det.frame_index
        return ev.TrackEvent(
            frame_id=det.frame_id,
            stream_id=det.stream_id,
            camera_id=det.camera_id,
            frame_index=det.frame_index,
            pts_s=det.pts_s,
            width=det.width,
            height=det.height,
            decoded_at=det.decoded_at,
            tracks=[
                ev.TrackPoint(
                    track_id=s.track_id,
                    label=s.label,
                    box=s.box,
                    score=s.score,
                    velocity=s.velocity,
                    age=s.hits,
                    hits=s.hits,
                )
                for s in visible
            ],
            ended=[t.track_id for t in ended],
        )

    def _release(self, state: _Stream, keep: int) -> list[ev.TrackEvent]:
        out = []
        while len(state.buffer) > keep:
            _, _, det = heapq.heappop(state.buffer)
            out.append(self._feed(state, det))
        return out

    def _finish(self, stream_id: str, end: ev.EndEvent | None) -> list[ev.Event]:
        state = self.streams.pop(stream_id, None)
        out: list[ev.Event] = []
        if state is not None:
            out.extend(self._release(state, 0))
            state.tracker.flush()
        if end is not None:
            out.append(end)
        return out

    def process(self, events: Sequence[ev.Event]) -> list[list[Output] | BaseException]:
        results: list[list[Output] | BaseException] = []
        for event in events:
            try:
                if isinstance(event, ev.DetectionEvent):
                    state = self._stream(event.stream_id)
                    if event.frame_index <= state.last_fed:
                        state.late += 1
                        if self.on_late:
                            self.on_late(1)
                        results.append([])
                        continue
                    state.seq += 1
                    heapq.heappush(state.buffer, (event.frame_index, state.seq, event))
                    released = self._release(state, self.reorder_window)
                    results.append(outputs_for(ev.TOPIC_TRACKS, released))
                elif isinstance(event, ev.EndEvent):
                    results.append(
                        outputs_for(ev.TOPIC_TRACKS, self._finish(event.stream_id, event))
                    )
                else:
                    results.append(ValueError(f"tracking does not consume {event.type!r} events"))
            except Exception as err:
                results.append(err)
        # Streams idle for too long are flushed onto the last result list.
        idle = [
            sid
            for sid, s in self.streams.items()
            if self.clock() - s.last_activity > self.idle_timeout_s
        ]
        for sid in idle:
            log.warning(
                "stream %s idle for %.0fs without an end event; flushing", sid, self.idle_timeout_s
            )
            flushed = outputs_for(ev.TOPIC_TRACKS, self._finish(sid, None))
            for i in range(len(results) - 1, -1, -1):
                if isinstance(results[i], list):
                    results[i].extend(flushed)
                    break
        return results


def build_runner(config, bus, metrics=None):
    """Wire the tracking stage (shared by the service, the demo and tests)."""
    from crosswalk.stream.bus import TopicSpec
    from crosswalk.stream.stage import RunnerConfig, StageRunner

    k = config.kafka
    handler = TrackingHandler(
        TrackerConfig(
            high_threshold=config.high_threshold,
            low_threshold=config.low_threshold,
            match_iou=config.match_iou,
            low_match_iou=config.low_match_iou,
            min_hits=config.min_hits,
            max_age_s=config.max_age_s,
            min_box_height=config.min_box_height,
        ),
        reorder_window=config.reorder_window,
        idle_timeout_s=config.idle_timeout_s,
    )
    if metrics is not None:
        late = metrics.events.labels(stage="tracking", outcome="late")
        handler.on_late = late.inc
    return StageRunner(
        handler,
        bus,
        RunnerConfig(
            group=k.group or "crosswalk-tracking",
            input_topics=[ev.TOPIC_DETECTIONS],
            topics=[
                TopicSpec(ev.TOPIC_DETECTIONS, k.partitions, k.replication_factor),
                TopicSpec(ev.TOPIC_TRACKS, k.partitions, k.replication_factor),
            ],
            max_records=k.max_records,
            poll_timeout_s=k.poll_timeout_s,
            retries=k.retries,
            dedupe_capacity=k.dedupe_capacity,
        ),
        metrics=metrics,
    )
