"""The indicators stage: track events in, one indicator event per window out.

For each stream, frames are accumulated into fixed windows of video time
(``window_s``, default 60 s, aligned to the start of the clip). A window is
emitted when a frame from a later window arrives, or when the stream ends.
Its id is ``<stream_id>/<window index>``, so a replay of the same pass gives
the same ids and the ranking store upserts instead of double-counting.

Cameras without a config file still get indicators: in pixel space, with no
region of interest. That makes a newly added clip visible on the leaderboard
before anyone has calibrated it, with its numbers clearly labelled as pixel
indicators.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from crosswalk.indicators.camera import CameraConfig
from crosswalk.indicators.compute import Window, window_bounds
from crosswalk.stream import events as ev
from crosswalk.stream.stage import Output, outputs_for

log = logging.getLogger(__name__)


@dataclass
class _Open:
    index: int
    window: Window


class IndicatorsHandler:
    name = "indicators"

    def __init__(self, cameras: dict[str, CameraConfig], window_s: float = 60.0, clock=time.time):
        self.cameras = cameras
        self.window_s = window_s
        self.clock = clock
        self.open: dict[str, _Open] = {}
        self.emitted: set[str] = set()

    def camera(self, camera_id: str) -> CameraConfig:
        if camera_id not in self.cameras:
            log.info("camera %s has no config: pixel-space indicators, whole frame", camera_id)
            self.cameras[camera_id] = CameraConfig(camera_id=camera_id)
        return self.cameras[camera_id]

    def event_id(self, event: ev.Event) -> str | None:
        if isinstance(event, ev.TrackEvent):
            return event.frame_id
        return None

    def _emit(self, stream_id: str, camera_id: str, state: _Open) -> ev.IndicatorEvent | None:
        w = state.window
        window_id = f"{stream_id}/{state.index}"
        if w.frames == 0 or window_id in self.emitted:
            return None
        self.emitted.add(window_id)
        r = w.compute()
        return ev.IndicatorEvent(
            window_id=window_id,
            stream_id=stream_id,
            camera_id=camera_id,
            window_start_s=w.start_s,
            window_end_s=w.end_s,
            observed_s=round(w.observed_s(), 3),
            frames=w.frames,
            space=r.space,
            pedestrians=r.pedestrians,
            vehicles=r.vehicles,
            pedestrian_seconds=r.pedestrian_seconds,
            vehicle_seconds=r.vehicle_seconds,
            close_approaches=r.close_approaches,
            min_separation=r.min_separation,
            path_crossings=r.path_crossings,
            ttc_events=r.ttc_events,
            min_ttc_s=r.min_ttc_s,
            pet_events=r.pet_events,
            min_pet_s=r.min_pet_s,
            decoded_at_min=w.decoded_at_min or 0.0,
        )

    def process(self, events: Sequence[ev.Event]) -> list[list[Output] | BaseException]:
        results: list[list[Output] | BaseException] = []
        for event in events:
            try:
                out: list[ev.Event] = []
                if isinstance(event, ev.TrackEvent):
                    index, start, end = window_bounds(event.pts_s, self.window_s)
                    state = self.open.get(event.stream_id)
                    if state is not None and index != state.index:
                        if index < state.index:
                            results.append([])  # late frame from a closed window
                            continue
                        emitted = self._emit(event.stream_id, event.camera_id, state)
                        if emitted:
                            out.append(emitted)
                        state = None
                    if state is None:
                        window = Window(
                            self.camera(event.camera_id), event.width, event.height, start, end
                        )
                        state = self.open[event.stream_id] = _Open(index, window)
                    state.window.add_frame(
                        event.pts_s,
                        event.decoded_at,
                        [(p.track_id, p.label, p.box) for p in event.tracks],
                    )
                elif isinstance(event, ev.EndEvent):
                    state = self.open.pop(event.stream_id, None)
                    if state is not None:
                        emitted = self._emit(event.stream_id, event.camera_id, state)
                        if emitted:
                            out.append(emitted)
                else:
                    results.append(ValueError(f"indicators do not consume {event.type!r} events"))
                    continue
                results.append(outputs_for(ev.TOPIC_INDICATORS, out))
            except Exception as err:
                results.append(err)
        return results


def build_runner(config, bus, metrics=None):
    from crosswalk.indicators.camera import load_cameras
    from crosswalk.stream.bus import TopicSpec
    from crosswalk.stream.stage import RunnerConfig, StageRunner

    k = config.kafka
    handler = IndicatorsHandler(load_cameras(config.cameras_dir), window_s=config.window_s)
    return StageRunner(
        handler,
        bus,
        RunnerConfig(
            group=k.group or "crosswalk-indicators",
            input_topics=[ev.TOPIC_TRACKS],
            topics=[
                TopicSpec(ev.TOPIC_TRACKS, k.partitions, k.replication_factor),
                TopicSpec(ev.TOPIC_INDICATORS, k.partitions, k.replication_factor),
            ],
            max_records=k.max_records,
            poll_timeout_s=k.poll_timeout_s,
            retries=k.retries,
            dedupe_capacity=k.dedupe_capacity,
        ),
        metrics=metrics,
    )
