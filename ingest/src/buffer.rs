//! A bounded queue between decoding and publishing, with an explicit overflow
//! policy.
//!
//! Decoding (FFmpeg) and publishing (S3 upload, then a Kafka record) run at
//! different speeds. If publishing falls behind, something has to give, and
//! which thing is a deliberate choice:
//!
//! - [`Overflow::Block`]: the decoder waits for room. Nothing is lost; the
//!   clip simply takes longer than real time. Right for offline processing
//!   and replays, where every sampled frame matters.
//! - [`Overflow::DropNewest`]: the frame that does not fit is discarded and
//!   counted. Latency stays bounded, like a camera that cannot stall the
//!   world. Right when the sampler paces a clip in real time.
//!
//! Either way the depth, the time spent blocked and the number dropped are
//! exported as metrics, so backpressure is visible rather than silent.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tokio::sync::Notify;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overflow {
    Block,
    DropNewest,
}

impl Overflow {
    pub fn parse(text: &str) -> Result<Self, String> {
        match text.trim().to_ascii_lowercase().as_str() {
            "block" => Ok(Overflow::Block),
            "drop" | "drop-newest" => Ok(Overflow::DropNewest),
            other => Err(format!(
                "unknown overflow policy {other:?} (expected block or drop)"
            )),
        }
    }
}

/// What happened to an item offered with [`Buffer::push`].
#[derive(Debug, PartialEq, Eq)]
pub enum Pushed {
    /// Queued; `waited` is how long the producer was blocked for room.
    Queued { waited: Duration },
    /// Discarded because the buffer was full (drop policy).
    Dropped,
    /// The buffer is closed; the item was not queued.
    Closed,
}

struct State<T> {
    items: VecDeque<T>,
    closed: bool,
}

/// Multi-producer, multi-consumer bounded buffer for async tasks.
pub struct Buffer<T> {
    capacity: usize,
    overflow: Overflow,
    state: Mutex<State<T>>,
    // Two notifiers so producers waiting for room and consumers waiting for
    // items do not wake each other needlessly.
    room: Notify,
    items: Notify,
}

impl<T> Buffer<T> {
    pub fn new(capacity: usize, overflow: Overflow) -> Self {
        assert!(capacity >= 1, "buffer capacity must be at least 1");
        Self {
            capacity,
            overflow,
            state: Mutex::new(State {
                items: VecDeque::with_capacity(capacity),
                closed: false,
            }),
            room: Notify::new(),
            items: Notify::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.state.lock().expect("buffer lock").items.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Offer an item, applying the overflow policy when full.
    pub async fn push(&self, item: T) -> Pushed {
        let started = Instant::now();
        let mut item = Some(item);
        loop {
            // Register interest before checking, so a pop that happens
            // between the check and the await still wakes this producer.
            let room = self.room.notified();
            {
                let mut state = self.state.lock().expect("buffer lock");
                if state.closed {
                    return Pushed::Closed;
                }
                if state.items.len() < self.capacity {
                    state.items.push_back(item.take().expect("item present"));
                    drop(state);
                    self.items.notify_one();
                    return Pushed::Queued {
                        waited: started.elapsed(),
                    };
                }
                if self.overflow == Overflow::DropNewest {
                    return Pushed::Dropped;
                }
            }
            room.await;
        }
    }

    /// Take the oldest item, waiting for one. `None` once closed and drained.
    pub async fn pop(&self) -> Option<T> {
        loop {
            let items = self.items.notified();
            {
                let mut state = self.state.lock().expect("buffer lock");
                if let Some(item) = state.items.pop_front() {
                    drop(state);
                    self.room.notify_one();
                    return Some(item);
                }
                if state.closed {
                    return None;
                }
            }
            items.await;
        }
    }

    /// No more pushes. Items already queued are still handed out, so a
    /// graceful shutdown publishes what was decoded instead of losing it.
    pub fn close(&self) {
        self.state.lock().expect("buffer lock").closed = true;
        self.room.notify_waiters();
        self.items.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn fifo_order() {
        let buffer = Buffer::new(3, Overflow::Block);
        for i in 0..3 {
            assert!(matches!(buffer.push(i).await, Pushed::Queued { .. }));
        }
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.pop().await, Some(0));
        assert_eq!(buffer.pop().await, Some(1));
        assert_eq!(buffer.pop().await, Some(2));
    }

    #[tokio::test]
    async fn drop_policy_discards_the_newest_when_full() {
        let buffer = Buffer::new(2, Overflow::DropNewest);
        assert!(matches!(buffer.push(1).await, Pushed::Queued { .. }));
        assert!(matches!(buffer.push(2).await, Pushed::Queued { .. }));
        assert_eq!(buffer.push(3).await, Pushed::Dropped);
        assert_eq!(buffer.pop().await, Some(1));
        assert_eq!(buffer.pop().await, Some(2));
        assert!(buffer.is_empty());
    }

    #[tokio::test]
    async fn block_policy_waits_for_room_and_reports_the_wait() {
        let buffer = Arc::new(Buffer::new(1, Overflow::Block));
        assert!(matches!(buffer.push(1).await, Pushed::Queued { .. }));
        let producer = {
            let buffer = Arc::clone(&buffer);
            tokio::spawn(async move { buffer.push(2).await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        // Still blocked: the buffer is full and nobody has popped.
        assert!(!producer.is_finished());
        assert_eq!(buffer.pop().await, Some(1));
        match producer.await.unwrap() {
            Pushed::Queued { waited } => assert!(waited >= Duration::from_millis(40)),
            other => panic!("expected Queued, got {other:?}"),
        }
        assert_eq!(buffer.pop().await, Some(2));
    }

    #[tokio::test]
    async fn close_drains_then_ends() {
        let buffer = Buffer::new(4, Overflow::Block);
        buffer.push("a").await;
        buffer.push("b").await;
        buffer.close();
        assert_eq!(buffer.push("c").await, Pushed::Closed);
        assert_eq!(buffer.pop().await, Some("a"));
        assert_eq!(buffer.pop().await, Some("b"));
        assert_eq!(buffer.pop().await, None);
    }

    #[tokio::test]
    async fn close_wakes_a_waiting_consumer() {
        let buffer: Arc<Buffer<u8>> = Arc::new(Buffer::new(1, Overflow::Block));
        let consumer = {
            let buffer = Arc::clone(&buffer);
            tokio::spawn(async move { buffer.pop().await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        buffer.close();
        let result = tokio::time::timeout(Duration::from_secs(1), consumer)
            .await
            .expect("consumer woke up")
            .unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn close_wakes_a_blocked_producer() {
        let buffer = Arc::new(Buffer::new(1, Overflow::Block));
        buffer.push(1).await;
        let producer = {
            let buffer = Arc::clone(&buffer);
            tokio::spawn(async move { buffer.push(2).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        buffer.close();
        let result = tokio::time::timeout(Duration::from_secs(1), producer)
            .await
            .expect("producer woke up")
            .unwrap();
        assert_eq!(result, Pushed::Closed);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn many_producers_and_consumers_lose_nothing_under_block() {
        let buffer = Arc::new(Buffer::new(4, Overflow::Block));
        let mut producers = Vec::new();
        for p in 0..4u32 {
            let buffer = Arc::clone(&buffer);
            producers.push(tokio::spawn(async move {
                for i in 0..250u32 {
                    assert!(matches!(
                        buffer.push(p * 1000 + i).await,
                        Pushed::Queued { .. }
                    ));
                }
            }));
        }
        let mut consumers = Vec::new();
        for _ in 0..3 {
            let buffer = Arc::clone(&buffer);
            consumers.push(tokio::spawn(async move {
                let mut got = Vec::new();
                while let Some(item) = buffer.pop().await {
                    got.push(item);
                }
                got
            }));
        }
        for p in producers {
            p.await.unwrap();
        }
        buffer.close();
        let mut all = Vec::new();
        for c in consumers {
            all.extend(c.await.unwrap());
        }
        all.sort_unstable();
        let mut expected: Vec<u32> = (0..4)
            .flat_map(|p| (0..250).map(move |i| p * 1000 + i))
            .collect();
        expected.sort_unstable();
        assert_eq!(all, expected);
    }

    #[test]
    fn overflow_parse() {
        assert_eq!(Overflow::parse("block"), Ok(Overflow::Block));
        assert_eq!(Overflow::parse("DROP"), Ok(Overflow::DropNewest));
        assert!(Overflow::parse("spill").is_err());
    }
}
