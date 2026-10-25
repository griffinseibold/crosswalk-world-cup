"""The replay load tool: pacing, identity rewriting, backpressure accounting."""

from __future__ import annotations

import json

import pytest

from crosswalk.stream import events as ev
from crosswalk.stream import replay
from crosswalk.stream.bus import MemoryBus


def recorded(n=10, fps=5.0, camera="cam", t0=1000.0):
    stream = f"{camera}/clip/r-0"
    frames = [
        {"schema": 1, "type": "frame", "frame_id": f"{stream}/{i}", "stream_id": stream, "camera_id": camera,
         "clip": "clip.mp4", "frame_index": i, "pts_s": i / fps, "width": 8, "height": 8,
         "decoded_at": t0 + i / fps, "image": {"bucket": "b", "key": f"k/{i}.jpg"}}
        for i in range(n)
    ]  # fmt: skip
    return [
        *frames,
        {"schema": 1, "type": "end", "stream_id": stream, "camera_id": camera, "clip": "clip.mp4"},
    ]


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def test_paces_at_n_times_real_speed_and_rewrites_identity():
    bus = MemoryBus()
    clock = FakeClock()
    records = recorded(n=10, fps=5.0)  # 1.8 s of recording
    result = replay.replay(
        bus, records, speed=2.0, clock=clock.now, wall=lambda: 5000.0, sleep=clock.sleep
    )
    assert result.events == 11 and result.recorded_s == pytest.approx(1.8)
    assert clock.t == pytest.approx(0.9)  # 1.8 s at 2x
    assert result.offered_eps == pytest.approx(11 / 0.9)
    out = [ev.parse(m.value) for m in bus.messages(ev.TOPIC_FRAMES)]
    frames = [e for e in out if isinstance(e, ev.FrameEvent)]
    assert all(f.stream_id == "cam/clip/r-0/replay-5000-0" for f in frames)
    assert all(f.frame_id == f"{f.stream_id}/{f.frame_index}" for f in frames)
    assert all(f.decoded_at == 5000.0 for f in frames)
    assert isinstance(out[-1], ev.EndEvent)  # end after the stream's frames


def test_several_passes_are_distinct_streams_and_speed_zero_does_not_sleep():
    bus = MemoryBus()
    clock = FakeClock()
    result = replay.replay(
        bus, recorded(n=3), speed=0, passes=3, clock=clock.now, sleep=clock.sleep
    )
    assert result.events == 12 and clock.slept == []
    streams = {ev.parse(m.value).stream_id for m in bus.messages(ev.TOPIC_FRAMES)}
    assert len(streams) == 3


def test_blocked_sends_are_counted_as_backpressure():
    bus = MemoryBus()
    clock = FakeClock()
    producer = bus.producer()
    real_send = producer.send

    def slow_send(*args):
        clock.t += 0.05  # each send blocks 50 ms (queue full)
        real_send(*args)

    bus.producer = lambda client_id="x": type(
        "P", (), {"send": staticmethod(slow_send), "flush": lambda self=None: None}
    )()
    result = replay.replay(
        bus, recorded(n=10, fps=5.0), speed=4.0, clock=clock.now, sleep=clock.sleep
    )
    assert result.send_blocked_s == pytest.approx(11 * 0.05)
    assert result.behind_schedule_s > 0  # 50 ms per send cannot keep 4x pace at 5 fps
    assert result.send_latency_ms["p50"] == pytest.approx(50.0)


def test_lag_samples_are_collected():
    bus = MemoryBus()
    clock = FakeClock()
    result = replay.replay(
        bus, recorded(n=10), speed=1.0, clock=clock.now, sleep=clock.sleep,
        lag_probe=lambda: {"crosswalk-inference": 7}, lag_every_s=0.5,
    )  # fmt: skip
    assert result.lag_samples and result.lag_samples[-1]["crosswalk-inference"] == 7


def test_jsonl_and_topic_sources(tmp_path):
    path = tmp_path / "events.jsonl"
    lines = [json.dumps(r | {"_key": "cam"}) for r in recorded(n=4)]
    lines.append(json.dumps({"type": "detections", "frame_id": "x"}))  # not replayed
    path.write_text("\n".join(lines) + "\n")
    records = replay.load_jsonl(path)
    assert len(records) == 5 and all("_key" not in r for r in records)
    bus = MemoryBus()
    replay.replay(bus, records, speed=0)
    assert len(replay.from_topic(bus, idle_polls=1)) == 5
