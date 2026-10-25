"""Group frames into batches for the GPU.

A detector processes a batch of N frames in less than N times the cost of one
frame, because the GPU is underused by a single 720p image. Waiting to fill a
batch adds latency, though, so the trade-off is configurable and is exactly
what Experiment A measures:

- ``sequential``: every frame is its own batch; no waiting.
- ``fixed``: wait until ``max_batch_size`` frames are queued. Highest GPU
  efficiency, but a slow stream can wait a long time; only end of stream
  flushes a partial batch.
- ``dynamic``: dispatch when ``max_batch_size`` frames are queued *or* the
  oldest queued frame has waited ``max_wait_ms``, whichever comes first. This
  bounds the added latency while still batching under load.

:class:`BatchPolicy` is the pure decision rule (easy to test with a fake
clock). :class:`Batcher` wraps it around a bounded, thread-safe queue.
Nothing here knows about PyTorch or video; payloads are opaque.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crosswalk.config import BatchingConfig


@dataclass(frozen=True)
class Decision:
    take: int  # frames to dispatch now (0: keep waiting)
    wait_s: float | None  # how long to wait before deciding again; None: until notified


class BatchPolicy:
    def __init__(self, config: BatchingConfig):
        self.strategy = config.strategy
        self.batch_size = config.effective_batch_size
        self.max_wait_s = config.max_wait_ms / 1000.0

    def decide(self, queued: int, oldest_wait_s: float, closed: bool) -> Decision:
        if queued >= self.batch_size:
            return Decision(self.batch_size, None)
        if queued == 0:
            return Decision(0, None)
        if closed:
            # End of stream: never strand frames, whatever the strategy.
            return Decision(queued, None)
        if self.strategy == "dynamic":
            remaining = self.max_wait_s - oldest_wait_s
            if remaining <= 0:
                return Decision(queued, None)
            return Decision(0, remaining)
        # fixed: wait for more frames (or close).
        return Decision(0, None)


@dataclass
class Request:
    payload: Any
    stream: str
    seq: int
    enqueued_at: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    requests: list[Request]
    formed_at: float

    def __len__(self) -> int:
        return len(self.requests)

    @property
    def payloads(self) -> list[Any]:
        return [r.payload for r in self.requests]

    def queue_waits(self) -> list[float]:
        return [self.formed_at - r.enqueued_at for r in self.requests]


class Batcher:
    """Bounded queue that hands out batches according to a :class:`BatchPolicy`.

    Producers call :meth:`submit`; one consumer calls :meth:`next_batch` in a
    loop until it returns ``None`` (closed and drained).
    """

    def __init__(self, config: BatchingConfig, clock: Callable[[], float] = time.monotonic):
        self.config = config
        self.policy = BatchPolicy(config)
        self._clock = clock
        self._queue: deque[Request] = deque()
        self._capacity = config.queue_size
        self._closed = False
        self._seq = 0
        self._cond = threading.Condition()
        self.dropped = 0

    def __len__(self) -> int:
        with self._cond:
            return len(self._queue)

    @property
    def closed(self) -> bool:
        return self._closed

    def submit(
        self,
        payload: Any,
        *,
        stream: str = "default",
        block: bool = True,
        timeout: float | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Request | None:
        """Queue a payload. Returns None if it was dropped.

        ``block=False`` is real-time behaviour: a full queue means the GPU is
        behind, and dropping the newest frame keeps latency bounded instead of
        letting it grow without limit. ``block=True`` applies back-pressure to
        the producer (offline processing, where every frame matters).
        """
        with self._cond:
            if self._closed:
                raise RuntimeError("batcher is closed")
            if len(self._queue) >= self._capacity:
                if not block:
                    self.dropped += 1
                    return None
                deadline = None if timeout is None else time.monotonic() + timeout
                while len(self._queue) >= self._capacity and not self._closed:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        self.dropped += 1
                        return None
                    self._cond.wait(remaining)
                if self._closed:
                    raise RuntimeError("batcher is closed")
            request = Request(payload, stream, self._seq, self._clock(), meta or {})
            self._seq += 1
            self._queue.append(request)
            self._cond.notify_all()
            return request

    def close(self) -> None:
        """No more submissions; queued frames are still handed out."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def try_next_batch(self) -> tuple[Batch | None, float | None]:
        """One non-blocking decision: (batch, None) or (None, seconds to wait)."""
        with self._cond:
            return self._decide_locked()

    def _decide_locked(self) -> tuple[Batch | None, float | None]:
        now = self._clock()
        oldest = now - self._queue[0].enqueued_at if self._queue else 0.0
        decision = self.policy.decide(len(self._queue), oldest, self._closed)
        if decision.take:
            requests = [self._queue.popleft() for _ in range(decision.take)]
            self._cond.notify_all()  # wake blocked producers
            return Batch(requests, now), None
        return None, decision.wait_s

    def next_batch(self, timeout: float | None = None) -> Batch | None:
        """Block until a batch is ready. None means closed and drained (or timeout)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                batch, wait_s = self._decide_locked()
                if batch is not None:
                    return batch
                if self._closed and not self._queue:
                    return None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    wait_s = remaining if wait_s is None else min(wait_s, remaining)
                self._cond.wait(wait_s)


@dataclass
class Timing:
    """Latency accounting for one frame, all in seconds.

    queue_wait + inference ~= end_to_end. The batch's inference time is
    charged in full to every frame in it, because each frame really does wait
    for the whole batch to finish.
    """

    queue_wait: float
    inference: float
    end_to_end: float
    batch_size: int


def timings_for(batch: Batch, started_at: float, finished_at: float) -> list[Timing]:
    inference = finished_at - started_at
    return [
        Timing(
            queue_wait=started_at - r.enqueued_at,
            inference=inference,
            end_to_end=finished_at - r.enqueued_at,
            batch_size=len(batch),
        )
        for r in batch.requests
    ]
