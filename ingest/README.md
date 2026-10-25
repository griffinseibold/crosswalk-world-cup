# crosswalk-ingest

The frame sampler: the first stage of the streaming pipeline. It plays
recorded clips through FFmpeg, keeps a sample of the frames, uploads each kept
frame to S3 as a JPEG and publishes a small *frame event* to Kafka that points
at it.

```text
clip.mp4 ─► ffmpeg (subprocess) ─RGB─► sampler ─► bounded buffer ─► publishers
                                         │            block | drop     │ JPEG encode
                                         │                             │ S3 PUT  s3://datasets/frames/<camera>/<clip>/<n>.jpg
                                         │                             └ Kafka   crosswalk.frames.v1 (key = camera)
                                         └ /metrics on :9100
```

Recorded footage only. Inputs are files or directories, never camera URLs
([data handling](../docs/data.md)).

## Why it looks like this

| Choice | Why |
| --- | --- |
| FFmpeg as a subprocess | Every codec FFmpeg supports works, with no libav bindings to build or keep in step. Raw RGB frames come out of stdout; the `showinfo` filter prints each frame's timestamp to stderr. |
| Rust | The sampler is a long-running I/O pipeline with a hard memory bound. Tokio gives cheap concurrency for uploads, and the bounded buffer makes backpressure explicit instead of an ever-growing queue. |
| [rskafka](https://github.com/influxdata/rskafka) 0.6.0, not rdkafka | Pure Rust: the build needs no C toolchain, cmake or OpenSSL. rdkafka wraps librdkafka, which is compiled from C. The sampler only produces, so rskafka's missing consumer groups do not matter. It partitions with Kafka's own murmur2 algorithm (tested against librdkafka's vectors), so all frames of one camera stay on one partition, in order. |
| Hand-written S3 SigV4 over [ureq](https://github.com/algesten/ureq) 3.5.0 | Two calls (PUT object, HEAD bucket) against one in-cluster endpoint. `aws-sdk-s3` brings around 200 crates and a TLS stack for that. The signer is tested against AWS's published worked examples, signature for signature. |
| Image in S3, pointer in Kafka | Kafka is built for many small records, not 100 kB images (broker memory, the 1 MB default message limit). The image is uploaded first, so an event never points at a missing object. |
| No TLS | The cluster's S3 and Kafka are plain-text Services inside the cluster. An `https://` S3 endpoint is refused at startup with an explanation. Adding TLS means enabling ureq's `rustls` feature and rskafka's `transport-tls`. |

## Behaviour

- **Sampling** (`--sampling`): `fps:5` (default) keeps one frame per 0.2 s of
  *video time*, which also works for variable-frame-rate clips. `every:N`
  keeps every N-th decoded frame, and `all` keeps everything.
  `motion:IDLE-BUSY[@CELLS]` adapts to the scene: each frame is reduced to a
  32x18 grid of average brightness and compared with the last kept frame; if
  at least `CELLS` cells (default 1) changed by more than 10 grey levels, the
  frame is kept at `BUSY` fps, otherwise only at `IDLE` fps. `motion:2-10`
  watches an empty crossing at 2 fps and a busy one at 10. Keep `IDLE` at
  2 fps or more with the default tracker (`max_age_s: 1.0`): at 1 fps a
  pedestrian standing still at the kerb is seen once a second, right at the
  tracker's limit, and their track can break into pieces.
- **Backpressure** (`--overflow`, `--buffer`): when publishing falls behind,
  `block` (default) makes the decoder wait, so nothing is lost and the clip
  plays slower than real time. `drop` discards the frame that does not fit
  and counts it, which keeps latency bounded. Time spent blocked is exported
  as `crosswalk_ingest_backpressure_seconds_total`.
- **Real-time pacing** (`--realtime`): FFmpeg reads at the clip's native
  rate (`-re`), like a camera. Combine it with `--overflow drop` to see what
  a live-rate source would lose.
- **Identity**: every pass over a clip is a new stream,
  `<camera>/<clip>/<run>-<pass>`. Frame ids are `<stream>/<frame index>`, and
  consumers deduplicate on them. The camera is the clip name up to the first
  `__`, so `corner-a__monday.mp4` and `corner-a__tuesday.mp4` rank as one
  crossing.
- **End of stream**: after the last frame of a pass has been acknowledged, an
  `end` event goes to the same partition. Downstream stages use it to flush
  tracks and close windows.
- **Graceful shutdown**: on SIGTERM, decoding stops, queued frames are still
  published, and the end event is sent with `"complete": false`. Draining
  stops after `--grace-seconds` (default 20), inside the pod's 30 s
  termination grace.
- **Failures**: a frame that fails to encode, upload or publish is logged,
  counted (`crosswalk_ingest_errors_total{stage}`) and skipped; the stream
  carries on. A clip FFmpeg cannot read is skipped. Startup fails fast if S3
  or Kafka is unreachable, and the pod restarts.
- **Topics**: the broker's auto-creation is off, so the sampler creates
  `crosswalk.frames.v1` itself at startup (default 3 partitions, replication
  factor 1). It treats "already exists" as success.

## Event schema

`testdata/frame-event.json` is a complete example. The Python consumers parse
that same file in their tests (`tests/test_stream_events.py`).

| Field | Meaning |
| --- | --- |
| `type` | `frame` (or `end`) |
| `frame_id`, `stream_id`, `camera_id`, `clip` | Identity, see above |
| `frame_index` | Position in decode order (every frame counts, sampled or not) |
| `sample_index` | Position among the frames this stream kept |
| `pts_s` | Presentation time in the clip, seconds |
| `width`, `height` | Frame size after `--resize` |
| `decoded_at`, `published_at` | Unix wall-clock seconds |
| `image.bucket`, `image.key` | Where the JPEG is |
| `bytes`, `sha256` | Size and digest of the JPEG |

## Running it

```bash
# Local: no Kafka or S3. JPEGs go under out/, events to out/events.jsonl.
cargo run --release -- /path/to/clips --local-dir out --sampling fps:5

# Against the cluster (the chart sets all of this from values.yaml)
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
cargo run --release -- /footage --brokers kafka.kafka.svc.cluster.local:9092 \
  --s3-endpoint http://s3.ml-storage.svc.cluster.local:9000 --loop --realtime
```

Every option has an environment variable, listed by `crosswalk-ingest --help`.

## Metrics

On `--listen` (default `:9100`): `/metrics` (OpenMetrics), `/healthz`,
`/readyz` (Kafka and S3 reached, topic exists).

| Metric | Type | Meaning |
| --- | --- | --- |
| `crosswalk_ingest_frames_decoded_total{camera}` | counter | Frames FFmpeg produced |
| `crosswalk_ingest_frames_sampled_total{camera}` | counter | Frames the policy kept |
| `crosswalk_ingest_frames_published_total{camera}` | counter | Events Kafka acknowledged |
| `crosswalk_ingest_frames_dropped_total{camera,reason}` | counter | `buffer_full` (drop policy) or `shutdown` |
| `crosswalk_ingest_errors_total{stage}` | counter | `decode`, `encode`, `upload`, `publish`, `startup` |
| `crosswalk_ingest_buffer_depth` / `_capacity` | gauge | Frames queued between decode and publish |
| `crosswalk_ingest_backpressure_seconds_total` | counter | Decoder time spent blocked |
| `crosswalk_ingest_{encode,upload,publish}_seconds` | histogram | Per-frame stage latency |
| `crosswalk_ingest_frame_age_seconds` | histogram | Decode to Kafka acknowledgement |
| `crosswalk_ingest_streams_completed_total{camera}` | counter | End events published |
| `crosswalk_ingest_ready` | gauge | 1 once dependencies were reached |

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked          # decoder tests need ffmpeg and ffprobe on PATH
```

The toolchain is pinned in `rust-toolchain.toml` (1.98.1). The crate
supports Rust 1.87 and newer. Dependencies are exact versions in `Cargo.toml`,
and `Cargo.lock` pins everything beneath them.
