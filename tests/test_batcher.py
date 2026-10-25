import threading
import time

import pytest

from crosswalk.config import BatchingConfig
from crosswalk.inference.batcher import Batch, Batcher, BatchPolicy, Request, timings_for


class FakeClock:
    def __init__(self, now: float = 100.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def cfg(strategy, size=4, wait_ms=10.0, queue=16):
    return BatchingConfig(
        strategy=strategy, max_batch_size=size, max_wait_ms=wait_ms, queue_size=queue
    )


# -- the pure policy -----------------------------------------------------------


def test_sequential_dispatches_every_frame_immediately():
    p = BatchPolicy(cfg("sequential"))
    assert p.decide(1, 0.0, False).take == 1
    assert p.decide(5, 0.0, False).take == 1


def test_fixed_waits_for_a_full_batch_until_close():
    p = BatchPolicy(cfg("fixed", size=4))
    assert p.decide(3, 10.0, False).take == 0  # waiting longer never helps
    assert p.decide(3, 10.0, False).wait_s is None
    assert p.decide(4, 0.0, False).take == 4
    assert p.decide(3, 0.0, True).take == 3  # end of stream flushes


def test_dynamic_dispatches_on_size_or_deadline():
    p = BatchPolicy(cfg("dynamic", size=4, wait_ms=10))
    assert p.decide(4, 0.0, False).take == 4
    d = p.decide(2, 0.004, False)
    assert d.take == 0 and d.wait_s == pytest.approx(0.006)
    assert p.decide(2, 0.010, False).take == 2
    assert p.decide(9, 0.0, False).take == 4  # never more than the max


def test_empty_queue_waits():
    for strategy in ("sequential", "fixed", "dynamic"):
        assert BatchPolicy(cfg(strategy)).decide(0, 0.0, False).take == 0


# -- the queue, with a fake clock ---------------------------------------------


def test_dynamic_batcher_with_fake_clock():
    clock = FakeClock()
    b = Batcher(cfg("dynamic", size=4, wait_ms=10), clock=clock)
    b.submit("a")
    clock.advance(0.003)
    b.submit("b")
    batch, wait = b.try_next_batch()
    assert batch is None and wait == pytest.approx(0.007)
    clock.advance(0.007)
    batch, _ = b.try_next_batch()
    assert batch.payloads == ["a", "b"]
    assert batch.queue_waits() == pytest.approx([0.010, 0.007])


def test_fixed_batcher_flushes_partial_batch_on_close():
    b = Batcher(cfg("fixed", size=3, queue=8))
    for i in range(7):
        b.submit(i)
    b.close()
    sizes = [len(x) for x in iter(b.next_batch, None)]
    assert sizes == [3, 3, 1]


def test_order_and_sequence_numbers_are_preserved():
    b = Batcher(cfg("fixed", size=2))
    reqs = [b.submit(i, stream="s") for i in range(4)]
    assert [r.seq for r in reqs] == [0, 1, 2, 3]
    b.close()
    flat = [r.payload for batch in iter(b.next_batch, None) for r in batch.requests]
    assert flat == [0, 1, 2, 3]


def test_non_blocking_submit_drops_when_full():
    b = Batcher(cfg("fixed", size=2, queue=2))
    assert b.submit(1, block=False) is not None
    assert b.submit(2, block=False) is not None
    assert b.submit(3, block=False) is None
    assert b.dropped == 1
    assert len(b) == 2


def test_blocking_submit_times_out_and_counts_a_drop():
    b = Batcher(cfg("fixed", size=1, queue=1))
    b.submit(1)
    t0 = time.monotonic()
    assert b.submit(2, timeout=0.05) is None
    assert time.monotonic() - t0 >= 0.04
    assert b.dropped == 1


def test_blocking_submit_resumes_when_consumer_takes_a_batch():
    b = Batcher(cfg("sequential", queue=1))
    b.submit(1)
    done = threading.Event()

    def producer():
        b.submit(2)  # blocks until the consumer makes room
        done.set()

    t = threading.Thread(target=producer)
    t.start()
    time.sleep(0.02)
    assert not done.is_set()
    assert b.next_batch(timeout=1).payloads == [1]
    assert done.wait(1)
    t.join()


def test_submit_after_close_raises():
    b = Batcher(cfg("dynamic"))
    b.close()
    with pytest.raises(RuntimeError):
        b.submit(1)


def test_next_batch_returns_none_when_closed_and_drained():
    b = Batcher(cfg("dynamic"))
    b.close()
    assert b.next_batch(timeout=1) is None


def test_next_batch_timeout():
    b = Batcher(cfg("fixed", size=4))
    b.submit(1)
    t0 = time.monotonic()
    assert b.next_batch(timeout=0.05) is None
    assert time.monotonic() - t0 < 1


def test_dynamic_deadline_with_real_threads():
    b = Batcher(cfg("dynamic", size=8, wait_ms=20))
    t0 = time.monotonic()
    b.submit("only")
    batch = b.next_batch(timeout=2)
    elapsed = time.monotonic() - t0
    assert batch.payloads == ["only"]
    assert 0.015 <= elapsed < 0.5  # waited for the deadline, not forever


def test_concurrent_producers_lose_nothing():
    b = Batcher(cfg("dynamic", size=8, wait_ms=2, queue=32))
    received = []

    def consumer():
        while (batch := b.next_batch()) is not None:
            received.extend(batch.payloads)

    c = threading.Thread(target=consumer)
    c.start()
    producers = [
        threading.Thread(target=lambda s=s: [b.submit((s, i), stream=str(s)) for i in range(200)])
        for s in range(4)
    ]
    for p in producers:
        p.start()
    for p in producers:
        p.join()
    b.close()
    c.join(timeout=5)
    assert sorted(received) == sorted((s, i) for s in range(4) for i in range(200))


# -- latency accounting --------------------------------------------------------


def test_timings_split_queue_wait_and_inference():
    reqs = [Request(i, "s", i, enqueued_at=t) for i, t in enumerate([10.0, 10.5, 11.0])]
    batch = Batch(reqs, formed_at=11.0)
    timings = timings_for(batch, started_at=11.2, finished_at=11.5)
    assert [t.queue_wait for t in timings] == pytest.approx([1.2, 0.7, 0.2])
    assert all(t.inference == pytest.approx(0.3) for t in timings)
    assert [t.end_to_end for t in timings] == pytest.approx([1.5, 1.0, 0.5])
    for t in timings:
        assert t.queue_wait + t.inference == pytest.approx(t.end_to_end)
        assert t.batch_size == 3
