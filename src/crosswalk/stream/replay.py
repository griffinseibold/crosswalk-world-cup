"""``crosswalk replay``: load-test the pipeline with recorded frame events.

Frame events are small (metadata and an S3 key), so a recorded stream of them
can be replayed at many times its real speed without re-decoding any video:
the frames are already in S3. The source is either

- a JSON-lines file of frame events (the Rust sampler's local-mode
  ``events.jsonl``), or
- a Kafka topic, read from the beginning with a throwaway consumer group.

Each replay gets a fresh ``stream_id`` suffix (``.../replay-<n>``) and frame
ids, so the stages treat it as new traffic instead of deduplicating it. The
original pacing is kept, compressed by ``--speed``: ``--speed 4`` sends four
seconds of recorded traffic per second. ``--speed 0`` sends as fast as the
broker accepts.

Measured and printed (and written as JSON with ``--output``):

- offered rate (events/s the replay tried to send) and achieved rate;
- **producer backpressure**: time the send call spent blocked because the
  client's local queue was full (the broker or network could not keep up);
- **pipeline lag** for each stage's consumer group, sampled during the run
  when ``--watch-groups`` is given (Kafka only), which shows which stage
  falls behind first as the speed goes up.

Only frame and end events are replayed. Detections and later events are what
the pipeline under test produces.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from crosswalk.stats import summarize
from crosswalk.stream import events as ev
from crosswalk.stream.bus import Bus, TopicSpec

log = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    events: int = 0
    passes: int = 0
    speed: float = 1.0
    wall_s: float = 0.0
    recorded_s: float = 0.0
    offered_eps: float | None = None
    achieved_eps: float | None = None
    send_blocked_s: float = 0.0
    behind_schedule_s: float = 0.0
    send_latency_ms: dict = field(default_factory=dict)
    lag_samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record.pop("_key", None)
        if record.get("type") in ("frame", "end"):
            records.append(record)
    return records


def from_topic(bus: Bus, topic: str = ev.TOPIC_FRAMES, idle_polls: int = 3) -> list[dict]:
    """Read every frame/end event currently in a topic (throwaway group)."""
    consumer = bus.consumer(
        f"crosswalk-replay-reader-{time.time_ns()}", [topic], "crosswalk-replay"
    )
    out, empty = [], 0
    try:
        while empty < idle_polls:
            batch = consumer.poll(500, 1.0)
            empty = 0 if batch else empty + 1
            for m in batch:
                record = json.loads(m.value)
                if record.get("type") in ("frame", "end"):
                    out.append(record)
    finally:
        consumer.close()
    return out


def schedule(records: list[dict]) -> list[tuple[float, dict]]:
    """(offset seconds from the start of the recording, record), in send order.

    Offsets come from each frame's ``decoded_at`` (wall-clock at capture).
    End events take the time of the last frame of their stream.
    """
    t0 = min((r["decoded_at"] for r in records if r.get("type") == "frame"), default=0.0)
    last_by_stream: dict[str, float] = {}
    timed = []
    for r in records:
        if r.get("type") == "frame":
            t = r["decoded_at"] - t0
            last_by_stream[r["stream_id"]] = max(last_by_stream.get(r["stream_id"], 0.0), t)
        else:
            t = last_by_stream.get(r["stream_id"], 0.0)
        timed.append((t, r))
    timed.sort(key=lambda x: (x[0], x[1].get("type") == "end"))
    return timed


def rewrite(record: dict, run: str, now: float) -> dict:
    """New identity and timestamps for a replayed event."""
    r = dict(record)
    r["stream_id"] = f"{record['stream_id']}/{run}"
    if r.get("type") == "frame":
        r["frame_id"] = f"{r['stream_id']}/{record['frame_index']}"
        r["decoded_at"] = now  # end-to-end latency is measured from the replay
        r["published_at"] = now
    else:
        r["published_at"] = now
    return r


def replay(
    bus: Bus,
    records: list[dict],
    *,
    speed: float = 1.0,
    passes: int = 1,
    topic: str = ev.TOPIC_FRAMES,
    partitions: int = 3,
    clock: Callable[[], float] = time.monotonic,
    wall: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    lag_probe: Callable[[], dict] | None = None,
    lag_every_s: float = 5.0,
) -> ReplayResult:
    if speed < 0:
        raise ValueError("speed must be >= 0 (0 = as fast as possible)")
    bus.ensure_topics([TopicSpec(topic, partitions)])
    producer = bus.producer("crosswalk-replay")
    timed = schedule(records)
    recorded_s = timed[-1][0] if timed else 0.0
    result = ReplayResult(speed=speed, recorded_s=recorded_s, passes=passes)
    send_ms: list[float] = []
    start = clock()
    last_lag = start
    offset = 0.0
    for p in range(passes):
        run = f"replay-{int(wall())}-{p}"
        for t, record in timed:
            due = start + (offset + t) / speed if speed > 0 else clock()
            ahead = due - clock()
            if ahead > 0:
                sleep(ahead)
            elif speed > 0:
                result.behind_schedule_s = max(result.behind_schedule_s, -ahead)
            t_send = clock()
            producer.send(
                topic, record.get("camera_id"), json.dumps(rewrite(record, run, wall())).encode()
            )
            spent = clock() - t_send
            send_ms.append(spent * 1000)
            result.send_blocked_s += spent
            result.events += 1
            if lag_probe is not None and clock() - last_lag >= lag_every_s:
                result.lag_samples.append({"t": round(clock() - start, 3), **lag_probe()})
                last_lag = clock()
        offset += recorded_s
    producer.flush()
    result.wall_s = clock() - start
    if lag_probe is not None:
        result.lag_samples.append({"t": round(result.wall_s, 3), **lag_probe()})
    if result.wall_s > 0:
        result.achieved_eps = result.events / result.wall_s
    if speed > 0 and recorded_s > 0:
        result.offered_eps = (len(timed) * passes) / (recorded_s * passes / speed)
    result.send_latency_ms = summarize(send_ms)
    return result


def kafka_lag_probe(bootstrap: str, groups: Iterable[str]) -> Callable[[], dict]:
    """Committed-offset lag per consumer group, read through the admin API."""
    from confluent_kafka import Consumer, TopicPartition

    groups = list(groups)

    def probe() -> dict:
        out = {}
        for group in groups:
            c = Consumer(
                {"bootstrap.servers": bootstrap, "group.id": group, "enable.auto.commit": False}
            )
            try:
                md = c.list_topics(timeout=10)
                total = 0
                for name, topic in md.topics.items():
                    if not name.startswith("crosswalk."):
                        continue
                    tps = [TopicPartition(name, p) for p in topic.partitions]
                    for tp in c.committed(tps, timeout=10):
                        if tp.offset < 0:
                            continue  # this group does not read the topic
                        _lo, hi = c.get_watermark_offsets(tp, timeout=10)
                        total += max(0, hi - tp.offset)
                out[group] = total
            finally:
                c.close()
        return out

    return probe
