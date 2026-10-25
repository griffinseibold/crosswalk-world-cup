"""Inference in Kafka mode: frame events in, detections out, through the real
batcher and worker with a fake predictor, frames fetched from (moto) S3."""

from __future__ import annotations

import io
import json
import threading

import numpy as np
import pytest

from crosswalk.config import InferenceConfig
from crosswalk.inference.predictor import FakePredictor
from crosswalk.stream import events as ev
from crosswalk.stream.bus import MemoryBus, TopicSpec

pytest.importorskip("prometheus_client")


def jpeg(width=32, height=24, value=128) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8)).save(buf, format="JPEG")
    return buf.getvalue()


def frame_event(i: int, camera: str = "cam-a", key: str | None = None) -> ev.FrameEvent:
    return ev.FrameEvent(
        frame_id=f"{camera}/clip/r-0/{i}", stream_id=f"{camera}/clip/r-0", camera_id=camera,
        clip="clip.mp4", frame_index=i, pts_s=i / 5, width=32, height=24, decoded_at=100.0,
        image=ev.ObjectRef(bucket="datasets", key=key or f"frames/{camera}/{i}.jpg"),
    )  # fmt: skip


def config(**kafka) -> InferenceConfig:
    return InferenceConfig.model_validate(
        {
            "mode": "kafka",
            "batching": {"strategy": "dynamic", "max_batch_size": 4, "max_wait_ms": 5},
            "kafka": {"poll_timeout_s": 0.05, "retries": 1, **kafka},
        }
    )


class Rig:
    """Wires the stage like run_kafka does, on a MemoryBus."""

    def __init__(self, fetch, predictor=None, cfg=None):
        from crosswalk.inference import stream_mode
        from crosswalk.inference.metrics import Metrics

        self.bus = MemoryBus()
        self.bus.ensure_topics([TopicSpec(ev.TOPIC_FRAMES, 3)])
        self.metrics = Metrics()
        self.predictor = predictor or FakePredictor(base_s=0.0, per_image_s=0.0)
        self.runner, self.worker, self.handler, self.batcher = stream_mode.build(
            cfg or config(), self.bus, self.predictor, self.metrics, fetch=fetch
        )
        self.runner.sleep = lambda s: None
        self.runner.start()
        self.worker.start()

    def publish(self, events):
        p = self.bus.producer()
        for e in events:
            p.send(ev.TOPIC_FRAMES, e.camera_id, e.to_bytes())

    def run(self):
        counts = self.runner.run(until_idle=True)
        return counts

    def detections(self):
        return [ev.parse(m.value) for m in self.bus.messages(ev.TOPIC_DETECTIONS)]

    def close(self):
        self.batcher.close()
        self.worker.join(5)
        self.handler.close()
        self.runner.close()


@pytest.fixture
def s3_frames(s3_client):
    for camera in ("cam-a", "cam-b"):
        for i in range(10):
            s3_client.put_object(
                Bucket="datasets", Key=f"frames/{camera}/{i}.jpg", Body=jpeg(value=i * 10)
            )
    return s3_client


def test_frames_from_s3_become_detection_events(s3_frames):
    from crosswalk.inference.stream_mode import FrameFetcher, s3_getter

    rig = Rig(FrameFetcher(s3_getter(s3_frames)))
    try:
        events = [frame_event(i, c) for i in range(10) for c in ("cam-a", "cam-b")]
        rig.publish(
            [*events, ev.EndEvent(stream_id="cam-a/clip/r-0", camera_id="cam-a", clip="clip.mp4")]
        )
        counts = rig.run()
        assert counts["processed"] == 21 and counts["dead_lettered"] == 0
        out = rig.detections()
        dets = [e for e in out if isinstance(e, ev.DetectionEvent)]
        assert sorted(d.frame_id for d in dets) == sorted(e.frame_id for e in events)
        assert any(isinstance(e, ev.EndEvent) for e in out)
        d = dets[0]
        assert d.model_version == "fake:0" and d.decoded_at == 100.0
        assert d.detections[0].label == "car" and set(d.latency) == {
            "queue_wait",
            "inference",
            "end_to_end",
        }
        # Several cameras' frames shared GPU batches (max 4).
        assert max(rig.predictor.batch_sizes) > 1
        # Per camera, detections keep frame order on their partition.
        for camera in ("cam-a", "cam-b"):
            order = [
                e.frame_index
                for e in out
                if isinstance(e, ev.DetectionEvent) and e.camera_id == camera
            ]
            assert order == sorted(order)
        assert rig.bus.lag_for("crosswalk-inference", ev.TOPIC_FRAMES) == 0
        text = rig.metrics.render().decode()
        assert "crosswalk_frames_total" in text and "crosswalk_stage_consumer_lag" in text
    finally:
        rig.close()


def test_redelivered_frames_are_not_inferred_twice(s3_frames):
    from crosswalk.inference.stream_mode import FrameFetcher, s3_getter

    rig = Rig(FrameFetcher(s3_getter(s3_frames)))
    try:
        rig.publish([frame_event(i) for i in range(4)] + [frame_event(1), frame_event(2)])
        counts = rig.run()
        assert counts["duplicate"] == 2
        assert sum(rig.predictor.batch_sizes) == 4
        assert len(rig.detections()) == 4
    finally:
        rig.close()


def test_a_missing_frame_is_dead_lettered_and_the_stream_continues(s3_frames):
    from crosswalk.inference.stream_mode import FrameFetcher, s3_getter

    rig = Rig(FrameFetcher(s3_getter(s3_frames)))
    try:
        rig.publish([frame_event(0), frame_event(1, key="frames/nope.jpg"), frame_event(2)])
        counts = rig.run()
        assert counts["processed"] == 2 and counts["dead_lettered"] == 1
        [dead] = rig.bus.messages("crosswalk.inference.dlq")
        assert "does not exist" in json.loads(dead.value)["error"]
        assert len(rig.detections()) == 2
    finally:
        rig.close()


def test_s3_outage_retries_instead_of_dead_lettering():
    from crosswalk.inference.stream_mode import FrameFetcher

    calls = {"n": 0}
    lock = threading.Lock()

    def flaky_get(ref):
        with lock:
            calls["n"] += 1
            if calls["n"] <= 3:
                raise ConnectionError("S3 endpoint refused the connection")
        return jpeg()

    rig = Rig(FrameFetcher(flaky_get))
    try:
        rig.publish([frame_event(0)])
        counts = rig.run()
        assert counts["transient_retry"] >= 1
        assert counts["processed"] == 1 and counts["dead_lettered"] == 0
    finally:
        rig.close()


def test_a_failing_gpu_batch_is_retried_then_dead_lettered():
    from crosswalk.inference.stream_mode import FrameFetcher

    predictor = FakePredictor(base_s=0.0, per_image_s=0.0, fail_on=1)
    rig = Rig(FrameFetcher(lambda ref: jpeg()), predictor=predictor)
    try:
        rig.publish([frame_event(0)])
        counts = rig.run()
        # First call fails; the per-event retry succeeds.
        assert counts["processed"] == 1 and counts["dead_lettered"] == 0
        assert predictor.calls == 2
    finally:
        rig.close()


def test_file_mode_is_still_the_default():
    assert InferenceConfig().mode == "file"
    assert InferenceConfig.model_validate({"mode": "kafka"}).kafka.group == "crosswalk-inference"
