//! `crosswalk-ingest`: the frame sampler.
//!
//! Plays recorded clips through FFmpeg, keeps a sample of frames, uploads
//! each kept frame to S3 as a JPEG and publishes a frame event (metadata plus
//! the S3 key) to Kafka. Recorded footage only: inputs are files, never
//! camera URLs (see docs/data.md).
//!
//! Every option is also an environment variable (`CROSSWALK_INGEST_*`) so the
//! Helm chart configures it without a config file.

mod buffer;
mod decode;
mod event;
mod kafka;
mod metrics;
mod pipeline;
mod s3;
mod sampling;
#[cfg(test)]
mod testutil;

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::Parser;
use tokio::sync::watch;

use crate::buffer::Overflow;
use crate::metrics::{Health, Metrics};
use crate::pipeline::{Clip, EventSink, FrameStore, Outputs, Settings};
use crate::sampling::Policy;

const VIDEO_SUFFIXES: &[&str] = &["mp4", "mkv", "mov", "avi", "webm", "ts"];

#[derive(Debug, Parser)]
#[command(
    name = "crosswalk-ingest",
    version,
    about = "Frame sampler: FFmpeg -> S3 + Kafka"
)]
pub struct Args {
    /// Video files or directories of them (recorded footage only).
    #[arg(
        required = true,
        env = "CROSSWALK_INGEST_INPUTS",
        value_delimiter = ','
    )]
    inputs: Vec<PathBuf>,

    /// all | fps:<rate> | every:<n>
    #[arg(long, env = "CROSSWALK_INGEST_SAMPLING", default_value = "fps:5", value_parser = Policy::parse)]
    sampling: Policy,

    /// What to do when publishing falls behind: block (lose nothing) or drop (bound latency).
    #[arg(long, env = "CROSSWALK_INGEST_OVERFLOW", default_value = "block", value_parser = Overflow::parse)]
    overflow: Overflow,

    /// Frames held between the decoder and the publishers.
    #[arg(long, env = "CROSSWALK_INGEST_BUFFER", default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..=10_000))]
    buffer: u32,

    /// Concurrent encode/upload/publish tasks.
    #[arg(long, env = "CROSSWALK_INGEST_PUBLISHERS", default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=64))]
    publishers: u32,

    /// Pace decoding at the clip's own frame rate, like a camera.
    #[arg(long, env = "CROSSWALK_INGEST_REALTIME", default_value_t = false)]
    realtime: bool,

    /// Start again from the first clip after the last one.
    #[arg(long = "loop", env = "CROSSWALK_INGEST_LOOP", default_value_t = false)]
    looping: bool,

    /// Resize to WIDTHxHEIGHT before sampling (e.g. 1280x720).
    #[arg(long, env = "CROSSWALK_INGEST_RESIZE", value_parser = parse_size)]
    resize: Option<(u32, u32)>,

    #[arg(long, env = "CROSSWALK_INGEST_JPEG_QUALITY", default_value_t = 85, value_parser = clap::value_parser!(u8).range(30..=100))]
    jpeg_quality: u8,

    /// Kafka bootstrap servers, comma-separated (PLAINTEXT).
    #[arg(
        long,
        env = "KAFKA_BOOTSTRAP_SERVERS",
        default_value = "kafka.kafka.svc.cluster.local:9092",
        value_delimiter = ','
    )]
    brokers: Vec<String>,

    #[arg(
        long,
        env = "CROSSWALK_TOPIC_FRAMES",
        default_value = "crosswalk.frames.v1"
    )]
    topic: String,

    /// Partitions to create the topic with (if it does not exist). Frames of
    /// one camera always share a partition.
    #[arg(long, env = "CROSSWALK_TOPIC_PARTITIONS", default_value_t = 3)]
    partitions: i32,

    #[arg(long, env = "CROSSWALK_TOPIC_REPLICATION", default_value_t = 1)]
    replication_factor: i16,

    #[arg(
        long,
        env = "AWS_ENDPOINT_URL_S3",
        default_value = "http://s3.ml-storage.svc.cluster.local:9000"
    )]
    s3_endpoint: String,

    #[arg(long, env = "AWS_DEFAULT_REGION", default_value = "us-east-1")]
    s3_region: String,

    #[arg(long, env = "CROSSWALK_FRAMES_BUCKET", default_value = "datasets")]
    bucket: String,

    #[arg(long, env = "CROSSWALK_FRAMES_PREFIX", default_value = "frames")]
    prefix: String,

    /// Write frames under this local directory instead of S3, and events as
    /// JSON lines to <dir>/events.jsonl instead of Kafka (local runs, demos).
    #[arg(long, env = "CROSSWALK_INGEST_LOCAL_DIR")]
    local_dir: Option<PathBuf>,

    /// Address for /metrics, /healthz and /readyz.
    #[arg(long, env = "CROSSWALK_INGEST_LISTEN", default_value = "0.0.0.0:9100")]
    listen: SocketAddr,

    #[arg(long, env = "CROSSWALK_FFMPEG", default_value = "ffmpeg")]
    ffmpeg: String,

    #[arg(long, env = "CROSSWALK_FFPROBE", default_value = "ffprobe")]
    ffprobe: String,

    /// Seconds to keep publishing queued frames after SIGTERM.
    #[arg(long, env = "CROSSWALK_INGEST_GRACE", default_value_t = 20)]
    grace_seconds: u64,
}

fn parse_size(text: &str) -> Result<(u32, u32), String> {
    let (w, h) = text
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected WIDTHxHEIGHT, got {text:?}"))?;
    let w: u32 = w
        .trim()
        .parse()
        .map_err(|_| format!("bad width in {text:?}"))?;
    let h: u32 = h
        .trim()
        .parse()
        .map_err(|_| format!("bad height in {text:?}"))?;
    // Even sizes: yuv420 sources and most encoders need them, and FFmpeg's
    // scaler would otherwise round silently and break the frame size.
    if w == 0 || h == 0 || w % 2 == 1 || h % 2 == 1 || w > 7680 || h > 4320 {
        return Err(format!(
            "size must be even and at most 7680x4320, got {text:?}"
        ));
    }
    Ok((w, h))
}

/// Expand inputs into a sorted list of video files.
fn list_clips(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut clips = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let mut found: Vec<PathBuf> = std::fs::read_dir(input)
                .with_context(|| format!("reading {}", input.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file() && is_video(p))
                .collect();
            found.sort();
            clips.extend(found);
        } else if input.is_file() {
            clips.push(input.clone());
        } else {
            bail!("{} does not exist", input.display());
        }
    }
    Ok(clips)
}

fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| VIDEO_SUFFIXES.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

// -- real storage and sink ----------------------------------------------------

struct S3Store {
    client: s3::Client,
    bucket: String,
}

impl FrameStore for S3Store {
    fn bucket(&self) -> &str {
        &self.bucket
    }
    fn put(&self, key: &str, body: &[u8]) -> Result<()> {
        self.client
            .put_object(&self.bucket, key, body, "image/jpeg")
    }
}

struct KafkaSink(kafka::Publisher);

impl EventSink for KafkaSink {
    async fn send(
        &self,
        key: &str,
        value: Vec<u8>,
        headers: BTreeMap<String, Vec<u8>>,
    ) -> Result<()> {
        self.0.send(key, value, headers).await.map(|_| ())
    }
}

/// Local mode: frames as files, events as JSON lines (the `bucket` is the
/// directory). Used by `crosswalk demo`-style runs without a cluster.
struct LocalStore {
    root: PathBuf,
    name: String,
}

impl FrameStore for LocalStore {
    fn bucket(&self) -> &str {
        &self.name
    }
    fn put(&self, key: &str, body: &[u8]) -> Result<()> {
        let path = self.root.join(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Write then rename, so a reader never sees half a JPEG.
        let tmp = path.with_extension("jpg.tmp");
        std::fs::write(&tmp, body)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }
}

struct JsonlSink {
    file: tokio::sync::Mutex<tokio::fs::File>,
}

impl EventSink for JsonlSink {
    async fn send(
        &self,
        key: &str,
        value: Vec<u8>,
        _headers: BTreeMap<String, Vec<u8>>,
    ) -> Result<()> {
        use tokio::io::AsyncWriteExt;
        let mut record: serde_json::Value = serde_json::from_slice(&value)?;
        record["_key"] = serde_json::Value::String(key.to_string());
        let mut line = serde_json::to_vec(&record)?;
        line.push(b'\n');
        let mut file = self.file.lock().await;
        file.write_all(&line).await?;
        file.flush().await?;
        Ok(())
    }
}

// -- main loop ------------------------------------------------------------------

async fn play<S: FrameStore, K: EventSink>(
    args: &Args,
    clips: &[PathBuf],
    settings: &Settings,
    outputs: &Outputs<S, K>,
    shutdown: watch::Receiver<bool>,
) -> Result<()> {
    let metrics = &outputs.metrics;
    let mut pass: u64 = 0;
    loop {
        for path in clips {
            if *shutdown.borrow() {
                return Ok(());
            }
            let clip = Clip::new(path);
            let info = match decode::probe(&args.ffprobe, path).await {
                Ok(info) => info,
                Err(err) => {
                    // One bad file should not stop the others.
                    metrics
                        .errors
                        .get_or_create(&vec![("stage", "decode".into())])
                        .inc();
                    tracing::error!(clip = %path.display(), error = %format!("{err:#}"), "skipping clip");
                    continue;
                }
            };
            tracing::info!(clip = %clip.file_name, camera = %clip.camera_id, width = info.width, height = info.height, fps = info.fps, pass, "playing");
            let decoder = decode::Decoder::start(
                &args.ffmpeg,
                path,
                info.width,
                info.height,
                args.resize,
                args.realtime,
            )
            .await?;
            let (frames, process) = decoder.split();
            let report =
                pipeline::run_pass(&clip, pass, frames, settings, outputs, shutdown.clone()).await;
            let _ = process.stop().await;
            tracing::info!(clip = %clip.file_name, ?report, "pass finished");
        }
        pass += 1;
        if !args.looping || clips.is_empty() {
            return Ok(());
        }
    }
}

fn run_id() -> String {
    // A sortable UTC timestamp: new for every process start, so a restarted
    // sampler's streams never collide with the previous run's frame ids.
    s3::amz_date(decode::unix_now() as i64)
}

async fn run(args: Args) -> Result<()> {
    let clips = list_clips(&args.inputs)?;
    if clips.is_empty() {
        bail!("no video files in {:?}", args.inputs);
    }
    let metrics = Metrics::new();
    let health = Arc::new(Health::default());
    let (stop_tx, stop_rx) = watch::channel(false);

    let listener = tokio::net::TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("binding {}", args.listen))?;
    let server = tokio::spawn(metrics::serve(
        listener,
        metrics.clone(),
        Arc::clone(&health),
        stop_rx.clone(),
    ));

    // SIGTERM (Kubernetes) and SIGINT (Ctrl-C): stop decoding, publish what
    // is queued, send end events, exit within the grace period.
    let signal_tx = stop_tx.clone();
    tokio::spawn(async move {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = signal(SignalKind::terminate()).expect("SIGTERM handler");
        tokio::select! {
            _ = term.recv() => tracing::info!("SIGTERM: draining"),
            _ = tokio::signal::ctrl_c() => tracing::info!("SIGINT: draining"),
        }
        let _ = signal_tx.send(true);
    });

    let settings = Settings {
        policy: args.sampling,
        overflow: args.overflow,
        buffer_size: args.buffer as usize,
        publishers: args.publishers as usize,
        jpeg_quality: args.jpeg_quality,
        key_prefix: args.prefix.clone(),
        run_id: run_id(),
    };
    tracing::info!(clips = clips.len(), ?settings, "starting");

    let work = async {
        if let Some(dir) = &args.local_dir {
            std::fs::create_dir_all(dir)?;
            let store = Arc::new(LocalStore {
                root: dir.clone(),
                name: dir.display().to_string(),
            });
            let file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(dir.join("events.jsonl"))
                .await?;
            let sink = Arc::new(JsonlSink {
                file: tokio::sync::Mutex::new(file),
            });
            health.ready.store(true, Ordering::Relaxed);
            metrics.ready.set(1);
            let outputs = Outputs {
                store,
                sink,
                metrics: metrics.clone(),
            };
            play(&args, &clips, &settings, &outputs, stop_rx.clone()).await
        } else {
            let creds = s3::Credentials {
                access_key: std::env::var("AWS_ACCESS_KEY_ID")
                    .context("AWS_ACCESS_KEY_ID is not set")?,
                secret_key: std::env::var("AWS_SECRET_ACCESS_KEY")
                    .context("AWS_SECRET_ACCESS_KEY is not set")?,
            };
            let client = s3::Client::new(&args.s3_endpoint, &args.s3_region, creds)?;
            {
                let client = client.clone();
                let bucket = args.bucket.clone();
                tokio::task::spawn_blocking(move || client.check_bucket(&bucket)).await??;
            }
            let kafka = kafka::Publisher::connect(args.brokers.clone(), "crosswalk-ingest").await?;
            kafka::Publisher::ensure_topics(
                &kafka,
                &[kafka::TopicSpec {
                    name: args.topic.clone(),
                    partitions: args.partitions,
                    replication_factor: args.replication_factor,
                }],
            )
            .await?;
            let publisher = kafka::Publisher::new(kafka, &args.topic).await?;
            tracing::info!(topic = %args.topic, partitions = publisher.partitions(), "Kafka ready");
            health.ready.store(true, Ordering::Relaxed);
            metrics.ready.set(1);
            let store = Arc::new(S3Store {
                client,
                bucket: args.bucket.clone(),
            });
            let sink = Arc::new(KafkaSink(publisher));
            let outputs = Outputs {
                store,
                sink,
                metrics: metrics.clone(),
            };
            play(&args, &clips, &settings, &outputs, stop_rx.clone()).await
        }
    };

    // The grace period bounds how long draining may take after a signal.
    let grace = Duration::from_secs(args.grace_seconds);
    let mut stop_watch = stop_rx.clone();
    let result = tokio::select! {
        r = work => r,
        _ = async {
            let _ = stop_watch.wait_for(|stopped| *stopped).await;
            tokio::time::sleep(grace).await;
        } => {
            metrics.dropped.get_or_create(&vec![("camera", "*".into()), ("reason", "shutdown".into())]).inc();
            tracing::warn!("grace period over; exiting with frames still queued");
            Ok(())
        }
    };
    metrics.ready.set(0);
    if let Err(err) = &result {
        metrics
            .errors
            .get_or_create(&vec![("stage", "startup".into())])
            .inc();
        tracing::error!(error = %format!("{err:#}"), "stopping");
    }
    let _ = stop_tx.send(true);
    let _ = server.await;
    result
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(args))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(parse_size("1280x720"), Ok((1280, 720)));
        assert_eq!(parse_size("640X360"), Ok((640, 360)));
        assert!(parse_size("641x360").is_err());
        assert!(parse_size("0x0").is_err());
        assert!(parse_size("big").is_err());
    }

    #[test]
    fn cli_defaults_and_env_style_lists() {
        let args = Args::try_parse_from(["crosswalk-ingest", "/footage"]).unwrap();
        assert_eq!(args.sampling, Policy::TargetFps(5.0));
        assert_eq!(args.overflow, Overflow::Block);
        assert_eq!(args.topic, "crosswalk.frames.v1");
        assert_eq!(args.brokers, vec!["kafka.kafka.svc.cluster.local:9092"]);
        let args = Args::try_parse_from([
            "crosswalk-ingest",
            "a.mp4,b.mp4",
            "--sampling",
            "every:2",
            "--overflow",
            "drop",
            "--brokers",
            "k1:9092,k2:9092",
            "--resize",
            "640x360",
        ])
        .unwrap();
        assert_eq!(args.inputs.len(), 2);
        assert_eq!(args.brokers.len(), 2);
        assert_eq!(args.overflow, Overflow::DropNewest);
        assert_eq!(args.resize, Some((640, 360)));
        assert!(
            Args::try_parse_from(["crosswalk-ingest", "x", "--sampling", "rtsp://cam"]).is_err()
        );
    }

    #[test]
    fn clip_listing_filters_and_sorts() {
        let dir = std::env::temp_dir().join(format!("cw-ingest-list-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["b.mp4", "a.MKV", "notes.txt"] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        let clips = list_clips(std::slice::from_ref(&dir)).unwrap();
        let names: Vec<_> = clips
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, vec!["a.MKV", "b.mp4"]);
        assert!(list_clips(&[dir.join("missing.mp4")]).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The whole binary path in local mode: FFmpeg decodes a generated clip,
    /// frames land as JPEG files and events as JSON lines.
    #[tokio::test]
    async fn local_mode_end_to_end() {
        let Some(dir) = testutil::clip_dir() else {
            return;
        };
        let clip = testutil::make_clip(&dir, "cam-x__test.mp4", 25, 50, (160, 96));
        let out = dir.join("out");
        let args = Args::try_parse_from([
            "crosswalk-ingest",
            clip.to_str().unwrap(),
            "--sampling",
            "fps:5",
            "--local-dir",
            out.to_str().unwrap(),
            "--listen",
            "127.0.0.1:0",
            "--resize",
            "80x48",
        ])
        .unwrap();
        run(args).await.unwrap();
        let events: Vec<serde_json::Value> = std::fs::read_to_string(out.join("events.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let frames: Vec<_> = events.iter().filter(|e| e["type"] == "frame").collect();
        assert_eq!(frames.len(), 10, "2 s at 5 fps");
        assert_eq!(events.last().unwrap()["type"], "end");
        assert_eq!(events.last().unwrap()["complete"], true);
        for frame in frames {
            assert_eq!(frame["camera_id"], "cam-x");
            assert_eq!(frame["width"], 80);
            let key = frame["image"]["key"].as_str().unwrap();
            let jpeg = std::fs::read(out.join(key)).unwrap();
            assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
        }
    }
}
