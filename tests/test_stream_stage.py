"""The shared stage loop: dedupe, retries, dead letters, commit-after-flush."""

from __future__ import annotations

import json

import pytest

from crosswalk.stream import events as ev
from crosswalk.stream.bus import MemoryBus, TopicSpec
from crosswalk.stream.stage import Output, RunnerConfig, StageRunner, TransientError


def frame(i: int, camera: str = "cam") -> ev.FrameEvent:
    return ev.FrameEvent(
        frame_id=f"{camera}/clip/r-0/{i}", stream_id=f"{camera}/clip/r-0", camera_id=camera,
        clip="clip.mp4", frame_index=i, pts_s=i / 5, width=8, height=8, decoded_at=1.0,
        image=ev.ObjectRef(bucket="b", key=f"k/{i}.jpg"),
    )  # fmt: skip


def end(camera: str = "cam") -> ev.EndEvent:
    return ev.EndEvent(stream_id=f"{camera}/clip/r-0", camera_id=camera, clip="clip.mp4")


class Echo:
    """Re-publishes frames to 'out'; can be told to fail on chosen frames."""

    name = "echo"

    def __init__(self):
        self.calls = 0
        self.seen: list[str] = []
        self.fail: dict[int, int] = {}  # frame index -> failures left (-1: forever)
        self.transient_failures = 0

    def event_id(self, event):
        return getattr(event, "frame_id", None)

    def process(self, events):
        self.calls += 1
        if self.transient_failures:
            self.transient_failures -= 1
            raise TransientError("S3 down")
        out = []
        for e in events:
            left = self.fail.get(getattr(e, "frame_index", -99), 0)
            if left:
                self.fail[e.frame_index] = left - 1 if left > 0 else -1
                out.append(ValueError(f"cannot handle frame {e.frame_index}"))
                continue
            self.seen.append(self.event_id(e) or e.type)
            out.append([Output("out", e.camera_id, e)])
        return out


def setup(handler=None, **config):
    bus = MemoryBus()
    bus.ensure_topics([TopicSpec("in", 2)])
    handler = handler or Echo()
    runner = StageRunner(
        handler,
        bus,
        RunnerConfig(
            group="g",
            input_topics=["in"],
            topics=[TopicSpec("in", 2), TopicSpec("out", 2)],
            poll_timeout_s=0.01,
            retry_backoff_s=0.0,
            **config,
        ),
        sleep=lambda s: None,
    )
    runner.start()
    return bus, handler, runner


def publish(bus, events):
    p = bus.producer()
    for e in events:
        p.send("in", e.camera_id, e.to_bytes())


def test_events_flow_through_and_offsets_are_committed():
    bus, _handler, runner = setup()
    publish(bus, [frame(i) for i in range(5)] + [end()])
    counts = runner.run(until_idle=True)
    assert counts["processed"] == 6
    out = [ev.parse(m.value) for m in bus.messages("out")]
    assert [getattr(e, "frame_index", "end") for e in out] == [0, 1, 2, 3, 4, "end"]
    assert bus.lag_for("g", "in") == 0


def test_duplicates_are_skipped():
    bus, _handler, runner = setup()
    publish(bus, [frame(0), frame(1), frame(0), frame(1), frame(2)])
    counts = runner.run(until_idle=True)
    assert counts["duplicate"] == 2 and counts["processed"] == 3
    assert len(bus.messages("out")) == 3


def test_a_redelivered_batch_after_a_crash_is_not_processed_twice():
    bus, _handler, runner = setup()
    publish(bus, [frame(i) for i in range(3)])
    runner.run(until_idle=True)
    # Simulate a crash after producing but before committing: rewind the
    # group's committed offsets and replay.
    bus._groups["g"].committed.clear()
    runner._consumer.rewind()
    counts = runner.run(until_idle=True)
    assert counts["duplicate"] == 3
    assert len(bus.messages("out")) == 3


def test_a_failing_event_is_retried_then_dead_lettered_and_the_rest_continue():
    handler = Echo()
    handler.fail = {2: -1, 3: 1}  # frame 2 always fails; frame 3 fails once
    bus, _, runner = setup(handler, retries=2)
    publish(bus, [frame(i) for i in range(5)])
    counts = runner.run(until_idle=True)
    assert counts["processed"] == 4 and counts["dead_lettered"] == 1
    assert sorted(ev.parse(m.value).frame_index for m in bus.messages("out")) == [0, 1, 3, 4]
    [dead] = bus.messages("crosswalk.echo.dlq")
    record = json.loads(dead.value)
    assert record["stage"] == "echo" and record["attempts"] == 3
    assert "cannot handle frame 2" in record["error"]
    assert ev.parse(record["value"]).frame_index == 2  # the original is kept
    assert bus.lag_for("g", "in") == 0  # the poison event does not block the partition


def test_invalid_records_go_straight_to_the_dead_letter_topic():
    bus, _handler, runner = setup()
    p = bus.producer()
    p.send("in", "cam", b"{not json")
    p.send("in", "cam", json.dumps({"type": "frame"}).encode())
    publish(bus, [frame(0)])
    counts = runner.run(until_idle=True)
    assert counts["invalid"] == 2 and counts["processed"] == 1
    assert len(bus.messages("crosswalk.echo.dlq")) == 2


def test_transient_errors_retry_the_batch_without_committing_or_dead_lettering():
    handler = Echo()
    handler.transient_failures = 3
    bus, _, runner = setup(handler)
    publish(bus, [frame(i) for i in range(3)])
    counts = runner.run(until_idle=True)
    assert counts["transient_retry"] == 3
    assert counts["processed"] == 3 and counts["dead_lettered"] == 0
    assert len(bus.messages("out")) == 3


def test_outputs_are_flushed_before_commit():
    order = []
    bus, _handler, runner = setup()
    real_flush, real_commit = runner._producer.flush, runner._consumer.commit
    runner._producer.flush = lambda *a, **k: (order.append("flush"), real_flush())
    runner._consumer.commit = lambda m: (order.append("commit"), real_commit(m))
    publish(bus, [frame(0)])
    runner.run_once()
    assert order == ["flush", "commit"]


def test_a_failed_flush_leaves_the_batch_uncommitted():
    bus, _handler, runner = setup()

    def broken_flush(*a, **k):
        raise TimeoutError("broker did not acknowledge")

    runner._producer.flush = broken_flush
    publish(bus, [frame(0)])
    with pytest.raises(TimeoutError):
        runner.run_once()
    assert bus.lag_for("g", "in") == 1


def test_stage_metrics_are_recorded():
    pytest.importorskip("prometheus_client")
    from prometheus_client import CollectorRegistry, generate_latest

    from crosswalk.stream.stage import STAGE_METRIC_NAMES, StageMetrics

    registry = CollectorRegistry()
    bus, handler, runner = setup()
    runner.metrics = StageMetrics(registry)
    handler.fail = {1: -1}
    publish(bus, [frame(0), frame(1), frame(0)])
    runner.run(until_idle=True)
    text = generate_latest(registry).decode()
    assert 'crosswalk_stage_events_total{outcome="processed",stage="echo"} 1.0' in text
    assert 'crosswalk_stage_events_total{outcome="duplicate",stage="echo"} 1.0' in text
    assert 'crosswalk_stage_dead_letters_total{stage="echo"} 1.0' in text
    assert "crosswalk_stage_consumer_lag{partition=" in text
    for name in STAGE_METRIC_NAMES:
        assert name in text, name
