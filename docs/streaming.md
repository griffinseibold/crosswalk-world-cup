# Streaming pipeline

How the Milestone 3 stages talk to each other through Kafka: the topics, the
event formats, the delivery guarantees, and how each stage scales.

```text
recorded clips
   │
   ▼
crosswalk-ingest (Rust) ──frames.v1──► inference ──detections.v1──► tracking ──tracks.v1──► indicators ──indicators.v1──► ranking ──► leaderboard
   │ JPEGs                            (GPU)                                                                                (SQLite, HTTP)
   ▼
s3://datasets/frames/<camera>/<clip>/<frame>.jpg
```

## Topics

| Topic | Producer | Consumer group | Key | Value |
| --- | --- | --- | --- | --- |
| `crosswalk.frames.v1` | `crosswalk-ingest` | `crosswalk-inference` | camera | frame event: metadata + S3 key (never pixels) |
| `crosswalk.detections.v1` | inference | `crosswalk-tracking` | camera | boxes, labels, scores per frame |
| `crosswalk.tracks.v1` | tracking | `crosswalk-indicators` | camera | confirmed tracks per frame |
| `crosswalk.indicators.v1` | indicators | `crosswalk-ranking` | camera | conflict indicators per window |
| `crosswalk.<stage>.dlq` | each stage | (people) | camera | events that stage could not process, with the error |

Every stage also forwards `end` events, which mark the end of one pass over a
clip, so the next stage can flush its state for that stream.

- **Auto-creation is off** on the homelab broker. Each process creates the
  topics it reads and writes at startup, if missing, with replication factor 1
  (one broker). Several pods doing this at once is safe: the losers get
  "already exists", which counts as success.
- **Partitions**: 3 by default (`CROSSWALK_TOPIC_PARTITIONS` on the sampler,
  `kafka.partitions` in the stage configs). This is set only when a topic is
  first created.
- **Retention** is the broker default. Frame events are small (about 600
  bytes), and the JPEGs they point to live in S3 under a lifecycle you
  control, not in Kafka.

## Delivery guarantees

**At least once, with idempotent stages.** That is the honest guarantee for
this design:

1. A stage polls a batch, processes it, produces its outputs and waits for
   the broker to acknowledge them (`acks=all`, idempotent producer), and only
   then commits the input offsets.
2. A crash before the commit replays the batch from the last commit. Nothing
   is lost, and some events are processed twice.
3. Each stage deduplicates on the event's id (`frame_id` for frames and
   detections, `window_id` for indicators) in a bounded in-memory set, and
   the ranking store upserts by `window_id`. A replay therefore does not
   double-count anything that reaches the leaderboard.

Exactly-once (Kafka transactions) would remove the duplicates at the source.
It needs a transactional producer, read-committed consumers and a
`transaction.state.log` the single-broker homelab would have to be configured
for. Idempotent consumers give the same visible result for this pipeline with
far less machinery.

The deduplication set is in memory. A restarted pod has an empty set, so
records replayed after a restart are skipped by the next idempotent step,
not this one. For tracking, that step is the tracker's reorder window, which
ignores frames older than its position (`stream/tracking_stage.py`). For
ranking, it is the store's upsert.

## Errors, retries and the dead-letter topics

| Failure | What happens |
| --- | --- |
| A record is not valid JSON or matches no schema | Straight to `crosswalk.<stage>.dlq`. It can never succeed |
| One event fails (corrupt JPEG, missing S3 object, a bug triggered by one input) | Retried on its own up to `retries` times (default 2) with backoff, then dead-lettered. The rest of the batch carries on, and the partition does not stall on a poison event |
| A dependency is down (S3 unreachable, throttled) | The handler raises `TransientError`. The whole batch is retried with exponential backoff (up to 30 s) and nothing is committed, so lag grows instead of frames being dead-lettered for a problem that is not theirs. The lag alert fires if it does not recover |
| The broker does not acknowledge outputs | Nothing is committed. The batch is replayed after the pod restarts |

A dead-letter record keeps the original value, the error, the number of
attempts and the source topic, partition and offset. To replay one after
fixing the cause, produce its `value` back to the source topic. Dead-lettered
frames are not counted by `crosswalk_stage_events_total{outcome="processed"}`.
`CrosswalkDeadLetters` alerts when any appear.

## Scaling

A consumer group splits a topic's partitions between its members: one
partition is read by one member at a time. So:

- **Pods per stage ≤ partitions of its input topic.** Extra pods sit idle as
  hot standbys, which is still useful for failover.
- **Per-camera order holds** because every topic is keyed by camera, and a
  key always maps to the same partition (murmur2, the same algorithm in the
  Rust sampler, librdkafka's `murmur2_random` and the in-memory bus).
- **Stateful stages** (tracking, indicators) keep per-stream state in memory.
  When a rebalance moves a partition to another pod, the new owner starts
  that camera's open streams from scratch. Tracks in flight at that moment
  are cut in two. The indicators stage marks windows that started before the
  move as partial (`frames` lower than expected), and the ranking excludes
  windows with too little observed time.
- **Adding partitions** to an existing topic moves some cameras to other
  partitions (the hash is taken modulo the partition count). Do it between
  passes, when no stream is open. For a clean start, delete the topics and
  their consumer groups and let the stages recreate them.
- **Inference** is bound by the one GPU. A second inference pod (another
  GPU, or the CPU image for a smoke test) joins `crosswalk-inference` and
  takes some partitions. Dynamic batching runs inside each pod over frames
  from all the cameras it owns.

## Lag and latency

| Metric | Meaning |
| --- | --- |
| `crosswalk_stage_consumer_lag{stage,topic,partition}` | Records not yet committed by the stage's group (high watermark minus committed offset, refreshed every 10 s). A steadily growing value means the stage cannot keep up |
| `crosswalk_stage_event_age_seconds{stage}` | Now minus the source frame's decode time when the stage finishes an event: end-to-end latency up to that stage. The indicators stage's value is the full pipeline latency |
| `crosswalk_stage_batch_seconds{stage}` | Time to process one polled batch |
| `crosswalk_stage_events_total{stage,outcome}` | `processed`, `duplicate`, `dead_lettered`, `invalid`, `transient_retry` |
| `crosswalk_ingest_frame_age_seconds` | Decode to Kafka acknowledgement in the sampler |

The Grafana dashboard's streaming row shows these per stage.

## Running without Kafka

`crosswalk demo` runs every stage in one process on `MemoryBus`, an
in-memory implementation of the same interface with Kafka's ordering, offset
and consumer-group semantics. The tests use it too. The Kafka adapter itself
is tested against librdkafka's built-in mock cluster, a real Kafka protocol
server inside the client library.
