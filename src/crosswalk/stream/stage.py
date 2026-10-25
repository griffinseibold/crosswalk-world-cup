"""The consume -> process -> produce -> commit loop every Python stage shares.

A stage (inference, tracking, indicators, ranking) only implements
:class:`Handler`: turn a batch of input events into output events. This
module does the rest, the same way for every stage:

1. **Poll** up to ``max_records`` from the input topic(s).
2. **Parse** each record. A record that is not valid JSON or does not match
   a known schema can never succeed, so it goes straight to the stage's
   dead-letter topic.
3. **Deduplicate** on the event's id (``frame_id``, ``window_id``, ...).
   Delivery is at least once: after a crash, records since the last commit
   are delivered again. An id seen recently (bounded LRU) is skipped and
   counted as a duplicate.
4. **Process** the batch. Per event, the handler returns output events or
   raises:

   - :class:`TransientError` (S3 or a downstream service unreachable): the
     whole batch is retried with exponential backoff and nothing is
     committed, so the partition stalls instead of losing data. Lag grows,
     and the lag alert fires if it does not recover.
   - any other exception (a corrupt frame, a bug triggered by one input):
     the event is retried on its own ``retries`` times, then written to the
     dead-letter topic with the error, and the stream carries on.

5. **Produce** the outputs (keyed by camera, so per-camera order holds) and
   flush, which waits for the broker to acknowledge them.
6. **Commit** the input offsets, only after step 5. A crash in between
   replays the batch, which step 3 (and the handlers' own idempotent ids)
   absorbs.

Metrics (one registry per process, shared with the stage's own metrics):
consumer lag per partition, events by outcome, dead-lettered events, batch
processing time, and event age (now minus the frame's decode time), which is
end-to-end latency up to this stage.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from crosswalk.stream import events as ev
from crosswalk.stream.bus import Bus, Message, TopicSpec

log = logging.getLogger(__name__)

# Metric names, checked against the dashboard and alert rules by the tests.
EVENTS = "crosswalk_stage_events_total"
DLQ = "crosswalk_stage_dead_letters_total"
LAG = "crosswalk_stage_consumer_lag"
BATCH_SECONDS = "crosswalk_stage_batch_seconds"
EVENT_AGE = "crosswalk_stage_event_age_seconds"
STAGE_UP = "crosswalk_stage_up"
LAST_COMMIT = "crosswalk_stage_last_commit_timestamp_seconds"
STAGE_METRIC_NAMES = {EVENTS, DLQ, LAG, BATCH_SECONDS, EVENT_AGE, STAGE_UP, LAST_COMMIT}

AGE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 900.0)
BATCH_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class TransientError(Exception):
    """Retry the batch later; do not dead-letter (dependency unavailable)."""


@dataclass
class Output:
    topic: str
    key: str | None
    event: ev.Event


class Handler(Protocol):
    """What a stage implements."""

    name: str  # "inference", "tracking", ...

    def event_id(self, event: ev.Event) -> str | None:
        """Idempotency key for an input event (None: never deduplicate)."""
        ...

    def process(self, events: Sequence[ev.Event]) -> list[list[Output] | BaseException]:
        """One entry per input event, in order: its outputs, or the error."""
        ...


class StageMetrics:
    def __init__(self, registry):
        from prometheus_client import Counter, Gauge, Histogram

        r = registry
        self.events = Counter(
            EVENTS.removesuffix("_total"),
            "Input events by outcome: processed, duplicate, dead_lettered, invalid, transient_retry",
            ["stage", "outcome"],
            registry=r,
        )
        self.dlq = Counter(
            DLQ.removesuffix("_total"),
            "Events written to the stage's dead-letter topic",
            ["stage"],
            registry=r,
        )
        self.lag = Gauge(
            LAG,
            "Records in the input partition not yet committed by this consumer group",
            ["stage", "topic", "partition"],
            registry=r,
        )
        self.batch_seconds = Histogram(
            BATCH_SECONDS,
            "Time to process one polled batch",
            ["stage"],
            buckets=BATCH_BUCKETS,
            registry=r,
        )
        self.event_age = Histogram(
            EVENT_AGE,
            "Now minus the source frame's decode time, when this stage finished an event",
            ["stage"],
            buckets=AGE_BUCKETS,
            registry=r,
        )
        self.up = Gauge(
            STAGE_UP, "1 while the stage's consume loop is running", ["stage"], registry=r
        )
        self.last_commit = Gauge(
            LAST_COMMIT, "Unix time of the last successful offset commit", ["stage"], registry=r
        )


class _Seen:
    """Bounded set of recently processed ids (least recently used evicted)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items: OrderedDict[str, None] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        if key in self._items:
            self._items.move_to_end(key)
            return True
        return False

    def add(self, key: str) -> None:
        self._items[key] = None
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)


@dataclass
class RunnerConfig:
    group: str
    input_topics: list[str]
    topics: list[TopicSpec]  # every topic this stage reads or writes (created at startup)
    max_records: int = 64
    poll_timeout_s: float = 1.0
    retries: int = 2  # extra attempts per failing event before dead-lettering
    retry_backoff_s: float = 0.2
    transient_backoff_max_s: float = 30.0
    dedupe_capacity: int = 200_000
    client_id: str = "crosswalk"
    produced_at_field: bool = True
    extra: dict = field(default_factory=dict)


def _decoded_at(event: ev.Event) -> float | None:
    for name in ("decoded_at", "decoded_at_min"):
        value = getattr(event, name, None)
        if value:
            return float(value)
    return None


class StageRunner:
    def __init__(
        self,
        handler: Handler,
        bus: Bus,
        config: RunnerConfig,
        metrics: StageMetrics | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ):
        self.handler = handler
        self.bus = bus
        self.config = config
        self.metrics = metrics
        self.sleep = sleep
        self.clock = clock
        self.seen = _Seen(config.dedupe_capacity)
        self.dlq_topic = ev.dlq_topic(handler.name)
        self.stop = threading.Event()
        self.counts = {
            "processed": 0,
            "duplicate": 0,
            "dead_lettered": 0,
            "invalid": 0,
            "transient_retry": 0,
        }
        self._producer = None
        self._consumer = None

    # -- setup -----------------------------------------------------------------

    def start(self) -> None:
        specs = list(self.config.topics)
        if not any(s.name == self.dlq_topic for s in specs):
            specs.append(TopicSpec(self.dlq_topic, partitions=1))
        self.bus.ensure_topics(specs)
        self._producer = self.bus.producer(f"{self.config.client_id}-{self.handler.name}")
        self._consumer = self.bus.consumer(
            self.config.group,
            self.config.input_topics,
            f"{self.config.client_id}-{self.handler.name}",
        )

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
        if self.metrics is not None:
            self.metrics.up.labels(stage=self.handler.name).set(0)

    def _count(self, outcome: str, n: int = 1) -> None:
        self.counts[outcome] = self.counts.get(outcome, 0) + n
        if self.metrics is not None and n:
            self.metrics.events.labels(stage=self.handler.name, outcome=outcome).inc(n)

    # -- dead letters ------------------------------------------------------------

    def _dead_letter(self, message: Message, error: BaseException | str, attempts: int) -> None:
        record = {
            "schema": 1,
            "type": "dead_letter",
            "stage": self.handler.name,
            "error": f"{type(error).__name__}: {error}"
            if isinstance(error, BaseException)
            else error,
            "attempts": attempts,
            "source": {
                "topic": message.topic,
                "partition": message.partition,
                "offset": message.offset,
            },
            "key": message.key,
            # The original record, as text when it is text (it normally is JSON).
            "value": message.value.decode("utf-8", errors="replace"),
            "failed_at": self.clock(),
        }
        self._producer.send(self.dlq_topic, message.key, json.dumps(record).encode())
        self._count("dead_lettered")
        if self.metrics is not None:
            self.metrics.dlq.labels(stage=self.handler.name).inc()
        log.warning(
            "%s: dead-lettered %s[%d]@%d after %d attempt(s): %s",
            self.handler.name, message.topic, message.partition, message.offset, attempts, record["error"],
        )  # fmt: skip

    # -- one batch ------------------------------------------------------------------

    def _process_with_retries(self, items: list[tuple[Message, ev.Event]]) -> list[Output]:
        """Run the handler; isolate and retry failing events; DLQ the hopeless."""
        events = [e for _, e in items]
        try:
            results = self.handler.process(events)
        except TransientError:
            raise
        except Exception as err:  # the handler failed as a whole: retry each event alone
            log.warning(
                "%s: batch of %d failed (%s); retrying events one by one",
                self.handler.name,
                len(events),
                err,
            )
            results = [err] * len(events)
        if len(results) != len(events):
            raise RuntimeError(
                f"{self.handler.name} returned {len(results)} results for {len(events)} events"
            )
        outputs: list[Output] = []
        for (message, event), result in zip(items, results, strict=True):
            attempts = 1
            while isinstance(result, BaseException) and not isinstance(result, TransientError):
                if attempts > self.config.retries:
                    break
                self.sleep(self.config.retry_backoff_s * (2 ** (attempts - 1)))
                attempts += 1
                try:
                    [result] = self.handler.process([event])
                except TransientError:
                    raise
                except Exception as err:
                    result = err
            if isinstance(result, TransientError):
                raise result
            if isinstance(result, BaseException):
                self._dead_letter(message, result, attempts)
                continue
            outputs.extend(result)
            event_id = self.handler.event_id(event)
            if event_id is not None:
                self.seen.add(event_id)
            self._count("processed")
            decoded_at = _decoded_at(event)
            if self.metrics is not None and decoded_at:
                self.metrics.event_age.labels(stage=self.handler.name).observe(
                    max(0.0, self.clock() - decoded_at)
                )
        return outputs

    def run_once(self) -> int:
        """Poll, process, produce, commit one batch. Returns records polled."""
        messages = self._consumer.poll(self.config.max_records, self.config.poll_timeout_s)
        self._update_lag()
        if not messages:
            return 0
        started = time.monotonic()
        items: list[tuple[Message, ev.Event]] = []
        in_batch: set[str] = set()  # the same id twice in one poll counts too
        for message in messages:
            try:
                event = ev.parse(message.value)
            except Exception as err:
                self._count("invalid")
                self._dead_letter(message, err, attempts=0)
                continue
            event_id = self.handler.event_id(event)
            if event_id is not None and (event_id in self.seen or event_id in in_batch):
                self._count("duplicate")
                continue
            if event_id is not None:
                in_batch.add(event_id)
            items.append((message, event))

        backoff = self.config.retry_backoff_s
        while True:
            try:
                outputs = self._process_with_retries(items) if items else []
                break
            except TransientError as err:
                self._count("transient_retry")
                log.warning(
                    "%s: dependency unavailable (%s); retrying in %.1fs",
                    self.handler.name,
                    err,
                    backoff,
                )
                if self.stop.is_set():
                    return len(messages)  # nothing committed: replayed on restart
                self.sleep(backoff)
                backoff = min(backoff * 2, self.config.transient_backoff_max_s)

        for output in outputs:
            event = output.event
            if (
                self.config.produced_at_field
                and hasattr(event, "produced_at")
                and not event.produced_at
            ):
                event = event.model_copy(update={"produced_at": self.clock()})
            self._producer.send(output.topic, output.key, event.to_bytes())
        # Outputs (and dead letters) must be acknowledged before the inputs
        # are committed; otherwise a crash here would lose them.
        self._producer.flush()
        self._consumer.commit(messages)
        if self.metrics is not None:
            self.metrics.batch_seconds.labels(stage=self.handler.name).observe(
                time.monotonic() - started
            )
            self.metrics.last_commit.labels(stage=self.handler.name).set(self.clock())
        return len(messages)

    def _update_lag(self) -> None:
        if self.metrics is None:
            return
        try:
            for (topic, partition), lag in self._consumer.lag().items():
                self.metrics.lag.labels(
                    stage=self.handler.name, topic=topic, partition=str(partition)
                ).set(lag)
        except Exception as err:  # lag is diagnostics; never stop the stage for it
            log.debug("lag unavailable: %s", err)

    def run(self, until_idle: bool = False, idle_polls: int = 2) -> dict[str, int]:
        """Loop until ``stop`` is set (or, with ``until_idle``, until the input
        has been empty for ``idle_polls`` polls in a row: demos and tests)."""
        if self._consumer is None:
            self.start()
        if self.metrics is not None:
            self.metrics.up.labels(stage=self.handler.name).set(1)
        empty = 0
        try:
            while not self.stop.is_set():
                if self.run_once():
                    empty = 0
                else:
                    empty += 1
                    if until_idle and empty >= idle_polls:
                        break
        finally:
            if self.metrics is not None:
                self.metrics.up.labels(stage=self.handler.name).set(0)
        return dict(self.counts)


def outputs_for(topic: str, events: Iterable[ev.Event]) -> list[Output]:
    """Key every output by its camera (per-camera order on one partition)."""
    return [Output(topic, getattr(e, "camera_id", None), e) for e in events]
