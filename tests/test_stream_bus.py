"""The message bus: Kafka semantics in memory, and the confluent-kafka adapter.

The KafkaBus tests run against librdkafka's built-in mock cluster
(``test.mock.num.brokers``), a real Kafka protocol server inside the client
library: real producers, consumer groups, commits and watermarks, no broker
to install. The mock does not implement CreateTopics, so topic creation is
tested with a stand-in admin client and the mock auto-creates topics on first
produce.
"""

from __future__ import annotations

import time
import types

import pytest

from crosswalk.stream.bus import MemoryBus, TopicSpec, murmur2, partition_for


def test_murmur2_matches_kafkas_partitioner():
    # librdkafka's unittest_murmur2 vectors (the Java client's results); the
    # Rust sampler is tested against the same table.
    short, unaligned = "1234", "PreAmbleWillBeRemoved,ThePrePartThatIs"
    cases = {
        "kafka": 0xD067CF64, "giberish123456789": 0x8F552B0C,
        short: 0x9FC97B14, short[1:]: 0xE7C009CA, short[2:]: 0x873930DA, short[3:]: 0x5A4B5CA1,
        unaligned: 0x78424F1C, unaligned[1:]: 0x4A62B377, unaligned[2:]: 0xE0E4E09E,
        unaligned[3:]: 0x62B8B43F, "": 0x106E08D9,
    }  # fmt: skip
    for text, expected in cases.items():
        assert murmur2(text.encode()) == expected, text
    assert partition_for(None, 6) == 0
    assert all(0 <= partition_for(f"cam-{i}", 3) < 3 for i in range(50))


def test_memory_bus_orders_by_key_and_tracks_offsets():
    bus = MemoryBus()
    bus.ensure_topics([TopicSpec("t", partitions=3)])
    producer = bus.producer()
    for i in range(12):
        producer.send("t", f"cam-{i % 3}", str(i).encode())
    consumer = bus.consumer("g", ["t"])
    got = consumer.poll(100, timeout=0)
    assert len(got) == 12
    by_key: dict[str, list[int]] = {}
    for m in got:
        by_key.setdefault(m.key, []).append(int(m.value))
        assert m.partition == partition_for(m.key, 3)
    # Per key, records arrive in the order produced.
    assert all(v == sorted(v) for v in by_key.values())
    assert sum(consumer.lag().values()) == 12  # nothing committed yet
    consumer.commit(got[:5])
    assert bus.lag_for("g", "t") == 12 - sum(1 for m in got[:5])


def test_uncommitted_records_are_redelivered_after_a_restart():
    bus = MemoryBus()
    bus.ensure_topics([TopicSpec("t", partitions=1)])
    p = bus.producer()
    for i in range(4):
        p.send("t", "k", str(i).encode())
    first = bus.consumer("g", ["t"])
    batch = first.poll(2, timeout=0)
    first.commit(batch)
    first.poll(10, timeout=0)  # read but never committed
    first.close()
    second = bus.consumer("g", ["t"])
    assert [int(m.value) for m in second.poll(10, timeout=0)] == [2, 3]


def test_consumer_group_members_split_partitions():
    bus = MemoryBus()
    bus.ensure_topics([TopicSpec("t", partitions=4)])
    a, b = bus.consumer("g", ["t"]), bus.consumer("g", ["t"])
    other = bus.consumer("other", ["t"])
    p = bus.producer()
    for i in range(40):
        p.send("t", f"k{i}", b"x")
    got_a, got_b = a.poll(100, 0), b.poll(100, 0)
    assert len(got_a) + len(got_b) == 40
    assert not {m.partition for m in got_a} & {m.partition for m in got_b}
    assert len(other.poll(100, 0)) == 40  # another group gets everything


def test_producing_to_an_unknown_topic_fails_like_auto_create_off():
    with pytest.raises(KeyError):
        MemoryBus().producer().send("missing", None, b"x")


def test_poll_waits_for_new_records():
    import threading

    bus = MemoryBus()
    bus.ensure_topics([TopicSpec("t", partitions=1)])
    consumer = bus.consumer("g", ["t"])
    threading.Timer(0.05, lambda: bus.producer().send("t", None, b"late")).start()
    started = time.monotonic()
    got = consumer.poll(10, timeout=2.0)
    assert [m.value for m in got] == [b"late"]
    assert time.monotonic() - started < 1.5


# -- KafkaBus against librdkafka's mock cluster ---------------------------------------


@pytest.fixture
def mock_cluster():
    ck = pytest.importorskip("confluent_kafka")
    # The first client with test.mock.num.brokers starts the mock cluster and
    # keeps it alive; other clients connect to its bootstrap address.
    holder = ck.Producer({"test.mock.num.brokers": 1, "log_level": 2})
    brokers = holder.list_topics(timeout=10).brokers.values()
    yield ",".join(f"{b.host}:{b.port}" for b in brokers)
    holder.flush(1)


def test_kafka_bus_produces_consumes_commits_and_reports_lag(mock_cluster):
    from crosswalk.stream.bus import KafkaBus

    bus = KafkaBus(mock_cluster, extra={"log_level": 2})
    producer = bus.producer("test")
    for i in range(10):
        producer.send("crosswalk.test.v1", f"cam-{i % 2}", f"v{i}".encode())
    producer.flush(10)
    consumer = bus.consumer("g", ["crosswalk.test.v1"], "test")
    got = []
    deadline = time.monotonic() + 20
    while len(got) < 10 and time.monotonic() < deadline:
        got.extend(consumer.poll(10, 0.5))
    assert sorted(m.value for m in got) == sorted(f"v{i}".encode() for i in range(10))
    # Keyed records stay in order per key (murmur2_random partitioner).
    for key in ("cam-0", "cam-1"):
        values = [int(m.value[1:]) for m in got if m.key == key]
        assert values == sorted(values)
        assert len({m.partition for m in got if m.key == key}) == 1
    assert sum(consumer.lag().values()) == 10
    consumer.commit(got)
    consumer._lag_at = 0  # skip the refresh interval
    assert sum(consumer.lag().values()) == 0
    consumer.close()


def test_kafka_bus_creates_missing_topics_and_accepts_already_exists():
    ck = pytest.importorskip("confluent_kafka")
    from crosswalk.stream.bus import KafkaBus

    class Future:
        def __init__(self, error=None):
            self.error = error

        def result(self, timeout=None):
            if self.error:
                raise ck.KafkaException(self.error)

    class Admin:
        def __init__(self):
            self.created = []

        def list_topics(self, timeout=None):
            return types.SimpleNamespace(topics={"exists": None})

        def create_topics(self, topics, **kwargs):
            self.created.extend((t.topic, t.num_partitions, t.replication_factor) for t in topics)
            return {
                t.topic: Future(
                    ck.KafkaError(ck.KafkaError.TOPIC_ALREADY_EXISTS)
                    if t.topic == "raced"
                    else None
                )
                for t in topics
            }

    admin = Admin()
    KafkaBus("unused:9092").ensure_topics(
        [TopicSpec("exists"), TopicSpec("new", 6, 1), TopicSpec("raced")], admin=admin
    )
    assert admin.created == [("new", 6, 1), ("raced", 3, 1)]

    class Broken(Admin):
        def create_topics(self, topics, **kwargs):
            return {
                t.topic: Future(ck.KafkaError(ck.KafkaError.TOPIC_AUTHORIZATION_FAILED))
                for t in topics
            }

    with pytest.raises(ck.KafkaException):
        KafkaBus("unused:9092").ensure_topics([TopicSpec("new")], admin=Broken())
