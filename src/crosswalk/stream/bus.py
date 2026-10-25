"""The message bus behind the stages: Kafka in the cluster, memory in tests.

Both implementations offer the same small interface, modelled on Kafka:
topics with partitions, keyed records (same key, same partition, in order),
consumer groups with committed offsets, and lag (high watermark minus the
committed position). Stages are written against this interface only, so the
same stage code runs in ``crosswalk demo`` (``MemoryBus``) and in the cluster
(``KafkaBus``).

Delivery is at least once: a consumer commits offsets only after everything
it produced for those records has been acknowledged by the broker. A crash
between producing and committing replays those records, which is why every
stage deduplicates by event id (``stream/stage.py``).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP = "kafka.kafka.svc.cluster.local:9092"


def bootstrap_servers() -> str:
    """From ``KAFKA_BOOTSTRAP_SERVERS`` (set by the chart), like the S3 and
    MLflow endpoints: config files stay the same everywhere."""
    return os.environ.get("KAFKA_BOOTSTRAP_SERVERS") or DEFAULT_BOOTSTRAP


@dataclass(frozen=True)
class TopicSpec:
    name: str
    partitions: int = 3
    replication_factor: int = 1


@dataclass
class Message:
    topic: str
    partition: int
    offset: int
    key: str | None
    value: bytes
    timestamp: float = 0.0


def murmur2(data: bytes) -> int:
    """Kafka's murmur2 (Java ``Utils.murmur2``), as in the Rust sampler.

    The memory bus partitions with it so that tests see the same
    camera-to-partition mapping as the real cluster (the Kafka producer is
    configured with ``partitioner=murmur2_random``, the Java default).
    """
    seed, m, r = 0x9747B28C, 0x5BD1E995, 24
    mask = 0xFFFFFFFF
    length = len(data)
    h = (seed ^ length) & mask
    for i in range(length // 4):
        k = int.from_bytes(data[4 * i : 4 * i + 4], "little")
        k = (k * m) & mask
        k ^= k >> r
        k = (k * m) & mask
        h = (h * m) & mask
        h ^= k
    tail = data[(length // 4) * 4 :]
    if len(tail) == 3:
        h ^= tail[2] << 16
    if len(tail) >= 2:
        h ^= tail[1] << 8
    if len(tail) >= 1:
        h ^= tail[0]
        h = (h * m) & mask
    h ^= h >> 13
    h = (h * m) & mask
    h ^= h >> 15
    return h


def partition_for(key: str | None, partitions: int) -> int:
    if key is None or partitions == 1:
        return 0
    return (murmur2(key.encode()) & 0x7FFFFFFF) % partitions


class Producer(Protocol):
    def send(self, topic: str, key: str | None, value: bytes) -> None: ...
    def flush(self, timeout: float = 30.0) -> None: ...


class Consumer(Protocol):
    def poll(self, max_records: int, timeout: float) -> list[Message]: ...
    def commit(self, messages: Iterable[Message]) -> None: ...
    def lag(self) -> dict[tuple[str, int], int]: ...
    def close(self) -> None: ...


class Bus(Protocol):
    def ensure_topics(self, specs: Iterable[TopicSpec]) -> None: ...
    def producer(self, client_id: str) -> Producer: ...
    def consumer(self, group: str, topics: list[str], client_id: str) -> Consumer: ...


# -- in memory -----------------------------------------------------------------


@dataclass
class _Topic:
    partitions: list[list[Message]]


@dataclass
class _Group:
    committed: dict[tuple[str, int], int] = field(default_factory=dict)
    members: list[MemoryConsumer] = field(default_factory=list)


class MemoryBus:
    """Thread-safe in-process bus with Kafka's ordering and offset semantics.

    Topics must be created first (``ensure_topics``), like the homelab broker
    with auto-creation off; producing to an unknown topic raises.
    """

    def __init__(self):
        self._lock = threading.Condition()
        self._topics: dict[str, _Topic] = {}
        self._groups: dict[str, _Group] = defaultdict(_Group)

    def ensure_topics(self, specs: Iterable[TopicSpec]) -> None:
        with self._lock:
            for spec in specs:
                if spec.name not in self._topics:
                    self._topics[spec.name] = _Topic([[] for _ in range(spec.partitions)])

    def topics(self) -> list[str]:
        with self._lock:
            return sorted(self._topics)

    def messages(self, topic: str) -> list[Message]:
        """Every record in a topic, partition by partition (for tests)."""
        with self._lock:
            return [m for p in self._topics[topic].partitions for m in p]

    def _append(self, topic: str, key: str | None, value: bytes) -> Message:
        with self._lock:
            if topic not in self._topics:
                raise KeyError(f"unknown topic {topic!r} (auto-creation is off)")
            parts = self._topics[topic].partitions
            p = partition_for(key, len(parts))
            message = Message(topic, p, len(parts[p]), key, value, time.time())
            parts[p].append(message)
            self._lock.notify_all()
            return message

    def producer(self, client_id: str = "memory") -> MemoryProducer:
        return MemoryProducer(self)

    def consumer(self, group: str, topics: list[str], client_id: str = "memory") -> MemoryConsumer:
        consumer = MemoryConsumer(self, group, topics)
        with self._lock:
            self._groups[group].members.append(consumer)
        return consumer

    def _assignment(self, consumer: MemoryConsumer) -> list[tuple[str, int]]:
        """Partitions are split across the group's members round-robin, like
        Kafka's range/round-robin assignors: one partition, one consumer."""
        with self._lock:
            group = self._groups[consumer.group]
            members = group.members
            index = members.index(consumer)
            pairs = [
                (t, p)
                for t in consumer.topics
                if t in self._topics
                for p in range(len(self._topics[t].partitions))
            ]
            return [pair for i, pair in enumerate(pairs) if i % len(members) == index]

    def lag_for(self, group: str, topic: str) -> int:
        with self._lock:
            committed = self._groups[group].committed
            return sum(
                len(part) - committed.get((topic, p), 0)
                for p, part in enumerate(self._topics[topic].partitions)
            )


class MemoryProducer:
    def __init__(self, bus: MemoryBus):
        self.bus = bus
        self.sent = 0

    def send(self, topic: str, key: str | None, value: bytes) -> None:
        self.bus._append(topic, key, value)
        self.sent += 1

    def flush(self, timeout: float = 30.0) -> None:
        pass  # appends are synchronous


class MemoryConsumer:
    def __init__(self, bus: MemoryBus, group: str, topics: list[str]):
        self.bus = bus
        self.group = group
        self.topics = list(topics)
        # Position = next offset to hand out (may run ahead of committed).
        self._position: dict[tuple[str, int], int] = {}
        self._closed = False

    def _start(self, tp: tuple[str, int]) -> int:
        if tp not in self._position:
            self._position[tp] = self.bus._groups[self.group].committed.get(tp, 0)
        return self._position[tp]

    def poll(self, max_records: int = 100, timeout: float = 0.0) -> list[Message]:
        deadline = time.monotonic() + timeout
        with self.bus._lock:
            while True:
                out: list[Message] = []
                for tp in self.bus._assignment(self):
                    part = self.bus._topics[tp[0]].partitions[tp[1]]
                    start = self._start(tp)
                    take = part[start : start + max_records - len(out)]
                    out.extend(take)
                    self._position[tp] = start + len(take)
                    if len(out) >= max_records:
                        break
                if out or self._closed:
                    return out
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self.bus._lock.wait(remaining)

    def commit(self, messages: Iterable[Message]) -> None:
        with self.bus._lock:
            committed = self.bus._groups[self.group].committed
            for m in messages:
                tp = (m.topic, m.partition)
                committed[tp] = max(committed.get(tp, 0), m.offset + 1)

    def rewind(self) -> None:
        """Forget uncommitted progress, as a restarted consumer would (tests)."""
        self._position.clear()

    def lag(self) -> dict[tuple[str, int], int]:
        with self.bus._lock:
            committed = self.bus._groups[self.group].committed
            return {
                tp: len(self.bus._topics[tp[0]].partitions[tp[1]]) - committed.get(tp, 0)
                for tp in self.bus._assignment(self)
            }

    def close(self) -> None:
        with self.bus._lock:
            self._closed = True
            members = self.bus._groups[self.group].members
            if self in members:
                members.remove(self)
            self.bus._lock.notify_all()


# -- Kafka -----------------------------------------------------------------------


class KafkaBus:
    """confluent-kafka (librdkafka) against a real broker.

    Producer settings: ``acks=all`` and ``enable.idempotence=true`` so a
    retried send never duplicates a record within the producer's lifetime;
    ``partitioner=murmur2_random`` so a keyed record lands on the same
    partition the Rust sampler (and the Java client) would choose.

    Consumer settings: auto-commit off (commits happen after outputs are
    flushed, see stage.py) and ``auto.offset.reset=earliest`` so a new
    consumer group starts from the oldest retained record instead of
    silently skipping everything produced before it existed.
    """

    def __init__(self, bootstrap: str | None = None, extra: dict[str, object] | None = None):
        self.bootstrap = bootstrap or bootstrap_servers()
        self.extra = dict(extra or {})

    def _config(self, **values) -> dict[str, object]:
        return {"bootstrap.servers": self.bootstrap, **self.extra, **values}

    def ensure_topics(self, specs: Iterable[TopicSpec], admin=None, timeout: float = 30.0) -> None:
        """Create missing topics; "already exists" counts as success.

        The homelab broker has auto-creation off, so each stage creates the
        topics it produces to and consumes from at startup. Several pods
        doing this at once is safe: the broker serialises creation and the
        losers get TOPIC_ALREADY_EXISTS.
        """
        from confluent_kafka import KafkaError
        from confluent_kafka.admin import AdminClient, NewTopic

        specs = list(specs)
        admin = admin or AdminClient(self._config())
        existing = set(admin.list_topics(timeout=timeout).topics)
        missing = [s for s in specs if s.name not in existing]
        if not missing:
            return
        futures = admin.create_topics(
            [
                NewTopic(
                    s.name, num_partitions=s.partitions, replication_factor=s.replication_factor
                )
                for s in missing
            ],
            operation_timeout=timeout,
            request_timeout=timeout,
        )
        for name, future in futures.items():
            try:
                future.result(timeout=timeout)
                log.info("created topic %s", name)
            except Exception as err:
                kafka_error = err.args[0] if err.args else None
                if getattr(kafka_error, "code", lambda: None)() == KafkaError.TOPIC_ALREADY_EXISTS:
                    continue
                raise

    def producer(self, client_id: str) -> KafkaProducer:
        from confluent_kafka import Producer as CProducer

        return KafkaProducer(
            CProducer(
                self._config(
                    **{
                        "client.id": client_id,
                        "acks": "all",
                        "enable.idempotence": True,
                        "partitioner": "murmur2_random",
                        "linger.ms": 5,
                    }
                )
            )
        )

    def consumer(self, group: str, topics: list[str], client_id: str) -> KafkaConsumer:
        from confluent_kafka import Consumer as CConsumer

        consumer = CConsumer(
            self._config(
                **{
                    "group.id": group,
                    "client.id": client_id,
                    "enable.auto.commit": False,
                    "auto.offset.reset": "earliest",
                    # A stage that takes longer than this between polls (a
                    # slow GPU batch) is considered dead and its partitions
                    # move to another member; 5 min leaves plenty of room.
                    "max.poll.interval.ms": 300_000,
                }
            )
        )
        consumer.subscribe(topics)
        return KafkaConsumer(consumer)


class KafkaProducer:
    def __init__(self, producer):
        self._producer = producer
        self._errors: list[str] = []

    def _on_delivery(self, err, _msg) -> None:
        if err is not None:
            self._errors.append(str(err))

    def send(self, topic: str, key: str | None, value: bytes) -> None:
        while True:
            try:
                self._producer.produce(topic, key=key, value=value, on_delivery=self._on_delivery)
                break
            except BufferError:
                # The local queue is full: serve delivery reports to make room.
                self._producer.poll(0.1)
        self._producer.poll(0)

    def flush(self, timeout: float = 30.0) -> None:
        remaining = self._producer.flush(timeout)
        if remaining:
            raise TimeoutError(f"{remaining} records not acknowledged within {timeout}s")
        if self._errors:
            errors, self._errors = self._errors, []
            raise RuntimeError(f"{len(errors)} records failed delivery: {errors[0]}")


class KafkaConsumer:
    # Asking the broker for watermarks costs a round trip per partition, so
    # lag is refreshed at most this often.
    LAG_INTERVAL_S = 10.0

    def __init__(self, consumer):
        self._consumer = consumer
        self._lag: dict[tuple[str, int], int] = {}
        self._lag_at = 0.0

    def poll(self, max_records: int, timeout: float) -> list[Message]:
        out = []
        for m in self._consumer.consume(num_messages=max_records, timeout=timeout):
            if m.error():
                # Partition EOF and similar informational events are not
                # records; real errors are logged and the poll continues.
                log.warning("consumer: %s", m.error())
                continue
            key = m.key()
            ts = m.timestamp()[1] / 1000.0 if m.timestamp() else 0.0
            out.append(
                Message(
                    m.topic(),
                    m.partition(),
                    m.offset(),
                    key.decode() if key is not None else None,
                    m.value() or b"",
                    ts,
                )
            )
        return out

    def commit(self, messages: Iterable[Message]) -> None:
        from confluent_kafka import TopicPartition

        latest: dict[tuple[str, int], int] = {}
        for m in messages:
            tp = (m.topic, m.partition)
            latest[tp] = max(latest.get(tp, -1), m.offset)
        if latest:
            self._consumer.commit(
                offsets=[TopicPartition(t, p, o + 1) for (t, p), o in latest.items()],
                asynchronous=False,
            )

    def lag(self) -> dict[tuple[str, int], int]:
        now = time.monotonic()
        if now - self._lag_at < self.LAG_INTERVAL_S:
            return self._lag
        lag = {}
        assignment = self._consumer.assignment()
        committed = (
            {
                (tp.topic, tp.partition): tp.offset
                for tp in self._consumer.committed(assignment, timeout=10)
            }
            if assignment
            else {}
        )
        for tp in assignment:
            _low, high = self._consumer.get_watermark_offsets(tp, timeout=10)
            done = committed.get((tp.topic, tp.partition), -1)
            # No commit yet (-1001): everything retained counts as lag.
            lag[(tp.topic, tp.partition)] = max(0, high - max(done, _low))
        self._lag, self._lag_at = lag, now
        return lag

    def close(self) -> None:
        self._consumer.close()
