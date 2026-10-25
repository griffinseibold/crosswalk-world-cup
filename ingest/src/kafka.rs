//! Publishing frame events to Kafka with rskafka.
//!
//! Why rskafka (pure Rust) and not rdkafka (bindings to librdkafka): rdkafka
//! compiles librdkafka from C with cmake and wants OpenSSL, zlib and friends
//! on the build machine or vendored into the build. rskafka builds with
//! nothing but cargo, so the CI job and the image are plain `cargo build`.
//! What rskafka lacks (consumer groups, a built-in partitioner, linger
//! batching) the sampler does not need: it is a producer only, and does the
//! two things it needs itself:
//!
//! - **Partitioning.** Records are keyed by camera, and the partition is
//!   chosen with Kafka's default partitioner algorithm (murmur2 of the key,
//!   positive, modulo partition count), so a Java or librdkafka producer
//!   keyed the same way would land on the same partition. All frames of one
//!   camera stay on one partition, in order, which tracking depends on.
//! - **Topic creation.** Auto-creation is off on the homelab broker, so the
//!   sampler creates its topics at startup and treats "already exists" as
//!   success. Consumers (Python) do the same for the topics they own.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::Utc;
use rskafka::BackoffConfig;
use rskafka::client::error::{Error as KafkaError, ProtocolError};
use rskafka::client::partition::{Compression, PartitionClient, UnknownTopicHandling};
use rskafka::client::{Client, ClientBuilder};
use rskafka::record::Record;
use tokio::sync::Mutex;

/// Kafka's murmur2 (the Java client's `Utils.murmur2`), as used by the
/// default partitioner. Seed 0x9747b28c, m = 0x5bd1e995, r = 24.
pub fn murmur2(data: &[u8]) -> u32 {
    const SEED: u32 = 0x9747_b28c;
    const M: u32 = 0x5bd1_e995;
    const R: u32 = 24;
    let len = data.len();
    let mut h: u32 = SEED ^ (len as u32);
    let chunks = len / 4;
    for i in 0..chunks {
        let b = &data[i * 4..i * 4 + 4];
        let mut k = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h = h.wrapping_mul(M);
        h ^= k;
    }
    let tail = &data[chunks * 4..];
    match tail.len() {
        3 => {
            h ^= (tail[2] as u32) << 16;
            h ^= (tail[1] as u32) << 8;
            h ^= tail[0] as u32;
            h = h.wrapping_mul(M);
        }
        2 => {
            h ^= (tail[1] as u32) << 8;
            h ^= tail[0] as u32;
            h = h.wrapping_mul(M);
        }
        1 => {
            h ^= tail[0] as u32;
            h = h.wrapping_mul(M);
        }
        _ => {}
    }
    h ^= h >> 13;
    h = h.wrapping_mul(M);
    h ^= h >> 15;
    h
}

/// The default partitioner's choice for a keyed record.
pub fn partition_for(key: &[u8], partitions: i32) -> i32 {
    assert!(partitions > 0);
    ((murmur2(key) & 0x7fff_ffff) % partitions as u32) as i32
}

#[derive(Debug, Clone)]
pub struct TopicSpec {
    pub name: String,
    pub partitions: i32,
    pub replication_factor: i16,
}

pub struct Publisher {
    client: Client,
    topic: String,
    partitions: i32,
    clients: Mutex<HashMap<i32, Arc<PartitionClient>>>,
}

fn already_exists(err: &KafkaError) -> bool {
    matches!(
        err,
        KafkaError::ServerError {
            protocol_error: ProtocolError::TopicAlreadyExists,
            ..
        }
    )
}

impl Publisher {
    pub async fn connect(brokers: Vec<String>, client_id: &str) -> Result<Client> {
        ClientBuilder::new(brokers.clone())
            .client_id(client_id.to_string())
            .backoff_config(BackoffConfig {
                init_backoff: Duration::from_millis(100),
                max_backoff: Duration::from_secs(5),
                base: 2.0,
                // Give up on one operation after a minute so a broker outage
                // surfaces as errors and metrics instead of a silent hang.
                deadline: Some(Duration::from_secs(60)),
            })
            .build()
            .await
            .with_context(|| format!("connecting to Kafka at {}", brokers.join(",")))
    }

    /// Create every topic in `specs` that does not exist yet (idempotent).
    pub async fn ensure_topics(client: &Client, specs: &[TopicSpec]) -> Result<()> {
        let existing: Vec<String> = client
            .list_topics()
            .await
            .context("listing Kafka topics")?
            .into_iter()
            .map(|t| t.name)
            .collect();
        let controller = client
            .controller_client()
            .context("Kafka controller client")?;
        for spec in specs {
            if existing.iter().any(|t| t == &spec.name) {
                continue;
            }
            match controller
                .create_topic(&spec.name, spec.partitions, spec.replication_factor, 10_000)
                .await
            {
                Ok(()) => {
                    tracing::info!(topic = %spec.name, partitions = spec.partitions, "created topic")
                }
                // Another process created it between the list and the create.
                Err(err) if already_exists(&err) => {}
                Err(err) => {
                    return Err(err).with_context(|| format!("creating topic {}", spec.name));
                }
            }
        }
        Ok(())
    }

    pub async fn new(client: Client, topic: &str) -> Result<Self> {
        let partitions = client
            .list_topics()
            .await
            .context("listing Kafka topics")?
            .into_iter()
            .find(|t| t.name == topic)
            .map(|t| t.partitions.len() as i32)
            .filter(|&n| n > 0)
            .with_context(|| format!("topic {topic} does not exist"))?;
        Ok(Self {
            client,
            topic: topic.to_string(),
            partitions,
            clients: Mutex::new(HashMap::new()),
        })
    }

    pub fn partitions(&self) -> i32 {
        self.partitions
    }

    async fn partition_client(&self, partition: i32) -> Result<Arc<PartitionClient>> {
        let mut clients = self.clients.lock().await;
        if let Some(client) = clients.get(&partition) {
            return Ok(Arc::clone(client));
        }
        let client = Arc::new(
            self.client
                .partition_client(self.topic.clone(), partition, UnknownTopicHandling::Retry)
                .await
                .with_context(|| format!("partition client for {}:{partition}", self.topic))?,
        );
        clients.insert(partition, Arc::clone(&client));
        Ok(client)
    }

    /// Produce one keyed record and wait for the broker's acknowledgement.
    /// Returns (partition, offset).
    pub async fn send(
        &self,
        key: &str,
        value: Vec<u8>,
        headers: BTreeMap<String, Vec<u8>>,
    ) -> Result<(i32, i64)> {
        let partition = partition_for(key.as_bytes(), self.partitions);
        let client = self.partition_client(partition).await?;
        let record = Record {
            key: Some(key.as_bytes().to_vec()),
            value: Some(value),
            headers,
            timestamp: Utc::now(),
        };
        let offsets = client
            .produce(vec![record], Compression::NoCompression)
            .await
            .with_context(|| format!("producing to {}:{partition}", self.topic))?;
        Ok((partition, offsets.first().copied().unwrap_or(-1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test vectors from librdkafka's `unittest_murmur2` (src/rdmurmur2.c),
    /// which in turn are the Java client's results. Matching them means a
    /// Java or Python (librdkafka) producer would pick the same partition.
    #[test]
    fn murmur2_matches_the_java_client() {
        let short_unaligned = "1234";
        let unaligned = "PreAmbleWillBeRemoved,ThePrePartThatIs";
        let cases: [(&str, u32); 11] = [
            ("kafka", 0xd067cf64),
            ("giberish123456789", 0x8f552b0c),
            (short_unaligned, 0x9fc97b14),
            (&short_unaligned[1..], 0xe7c009ca),
            (&short_unaligned[2..], 0x873930da),
            (&short_unaligned[3..], 0x5a4b5ca1),
            (unaligned, 0x78424f1c),
            (&unaligned[1..], 0x4a62b377),
            (&unaligned[2..], 0xe0e4e09e),
            (&unaligned[3..], 0x62b8b43f),
            ("", 0x106e08d9),
        ];
        for (input, expected) in cases {
            assert_eq!(murmur2(input.as_bytes()), expected, "murmur2({input:?})");
        }
    }

    #[test]
    fn partitions_are_stable_and_in_range() {
        for n in [1, 3, 6, 12] {
            for key in ["intersection-01", "cam-b", "x"] {
                let p = partition_for(key.as_bytes(), n);
                assert!((0..n).contains(&p));
                assert_eq!(p, partition_for(key.as_bytes(), n));
            }
        }
        assert_eq!(partition_for(b"anything", 1), 0);
    }

    #[test]
    fn topic_already_exists_is_recognised() {
        let err = KafkaError::ServerError {
            protocol_error: ProtocolError::TopicAlreadyExists,
            error_message: None,
            request: rskafka::client::error::RequestContext::Topic("t".into()),
            response: None,
            is_virtual: false,
        };
        assert!(already_exists(&err));
        let other = KafkaError::InvalidResponse("x".into());
        assert!(!already_exists(&other));
    }
}
