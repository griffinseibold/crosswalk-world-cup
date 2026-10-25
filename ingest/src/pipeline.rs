//! Decode -> sample -> buffer -> (encode, upload, publish), for one clip pass.
//!
//! ```text
//!  ffmpeg ──RGB──► sampler ──► Buffer (bounded, block|drop) ──► N publisher tasks
//!                                                                 │ JPEG encode
//!                                                                 │ S3 PUT (frame asset)
//!                                                                 └ Kafka produce (frame event)
//! ```
//!
//! The image is always in S3 before its event is in Kafka, so a consumer that
//! sees an event can always fetch the frame. Publishers run in parallel for
//! throughput, so frames of one stream may be *published* slightly out of
//! order; every event carries its `frame_index` and `pts_s`, and the tracking
//! stage orders by them (it buffers a small reorder window). The end-of-stream
//! event is sent only after every frame of the stream has been acknowledged.
//!
//! Storage and the event bus are traits so the whole pipeline is tested
//! in-process with fakes (no Kafka, no S3), including backpressure and
//! shutdown behaviour.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use tokio::sync::watch;

use crate::buffer::{Buffer, Overflow, Pushed};
use crate::decode::{RawFrame, unix_now};
use crate::event::{self, EndEvent, FrameEvent, ObjectRef};
use crate::metrics::{Metrics, camera};
use crate::s3::sha256_hex;
use crate::sampling::{Policy, Sampler};

/// Where frame JPEGs go.
pub trait FrameStore: Send + Sync + 'static {
    fn bucket(&self) -> &str;
    /// Blocking upload (called from `spawn_blocking`).
    fn put(&self, key: &str, body: &[u8]) -> Result<()>;
}

/// Where events go. `key` decides ordering (same key, same partition).
#[allow(async_fn_in_trait)]
pub trait EventSink: Send + Sync + 'static {
    fn send(
        &self,
        key: &str,
        value: Vec<u8>,
        headers: BTreeMap<String, Vec<u8>>,
    ) -> impl std::future::Future<Output = Result<()>> + Send;
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub policy: Policy,
    pub overflow: Overflow,
    pub buffer_size: usize,
    pub publishers: usize,
    pub jpeg_quality: u8,
    pub key_prefix: String,
    pub run_id: String,
}

/// One sampled frame waiting to be published.
struct Job {
    frame: RawFrame,
    sample_index: u64,
}

/// Counts for one pass, returned to the caller and put in the end event.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PassReport {
    pub decoded: u64,
    pub sampled: u64,
    pub published: u64,
    pub dropped: u64,
    pub failed: u64,
    pub last_pts_s: f64,
    pub complete: bool,
}

pub fn encode_jpeg(frame: &RawFrame, quality: u8) -> Result<Vec<u8>> {
    let width = u16::try_from(frame.width).context("frame wider than 65535 px")?;
    let height = u16::try_from(frame.height).context("frame taller than 65535 px")?;
    let mut out = Vec::with_capacity(frame.rgb.len() / 8);
    jpeg_encoder::Encoder::new(&mut out, quality)
        .encode(&frame.rgb, width, height, jpeg_encoder::ColorType::Rgb)
        .context("JPEG encoding")?;
    Ok(out)
}

pub struct Clip {
    pub camera_id: String,
    pub stem: String,
    pub file_name: String,
}

impl Clip {
    pub fn new(path: &Path) -> Self {
        let file_name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "clip".into());
        let stem = event::sanitize(
            path.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .as_deref()
                .unwrap_or("clip"),
        );
        Self {
            camera_id: event::camera_id_from_clip(&file_name),
            stem,
            file_name,
        }
    }
}

/// Where a pass sends its output, and what it reports to.
pub struct Outputs<S, K> {
    pub store: Arc<S>,
    pub sink: Arc<K>,
    pub metrics: Metrics,
}

impl<S, K> Clone for Outputs<S, K> {
    fn clone(&self) -> Self {
        Self {
            store: Arc::clone(&self.store),
            sink: Arc::clone(&self.sink),
            metrics: self.metrics.clone(),
        }
    }
}

/// Run one pass over a clip, reading frames from `frames` until it ends or
/// `shutdown` becomes true. Generic over storage and sink so tests use fakes.
pub async fn run_pass<S: FrameStore, K: EventSink>(
    clip: &Clip,
    pass: u64,
    mut frames: tokio::sync::mpsc::Receiver<Result<RawFrame>>,
    settings: &Settings,
    outputs: &Outputs<S, K>,
    mut shutdown: watch::Receiver<bool>,
) -> PassReport {
    let Outputs {
        store,
        sink,
        metrics,
    } = outputs;
    let stream_id = event::stream_id(&clip.camera_id, &clip.stem, &settings.run_id, pass);
    let labels = camera(&clip.camera_id);
    let buffer: Arc<Buffer<Job>> = Arc::new(Buffer::new(settings.buffer_size, settings.overflow));
    metrics.buffer_capacity.set(settings.buffer_size as i64);
    let published = Arc::new(AtomicU64::new(0));
    let failed = Arc::new(AtomicU64::new(0));

    // Publisher tasks: encode, upload, publish.
    let mut workers = Vec::new();
    for _ in 0..settings.publishers.max(1) {
        let buffer = Arc::clone(&buffer);
        let store = Arc::clone(store);
        let sink = Arc::clone(sink);
        let metrics = metrics.clone();
        let labels = labels.clone();
        let published = Arc::clone(&published);
        let failed = Arc::clone(&failed);
        let stream_id = stream_id.clone();
        let camera_id = clip.camera_id.clone();
        let clip_name = clip.file_name.clone();
        let stem = clip.stem.clone();
        let quality = settings.jpeg_quality;
        let prefix = settings.key_prefix.clone();
        workers.push(tokio::spawn(async move {
            while let Some(job) = buffer.pop().await {
                metrics.buffer_depth.set(buffer.len() as i64);
                let index = job.frame.index;
                let target = Target {
                    stream_id: &stream_id,
                    camera_id: &camera_id,
                    clip_name: &clip_name,
                    stem: &stem,
                    quality,
                    prefix: &prefix,
                };
                let result = publish_one(job, &target, &store, sink.as_ref(), &metrics).await;
                match result {
                    Ok(()) => {
                        published.fetch_add(1, Ordering::Relaxed);
                        metrics.published.get_or_create(&labels).inc();
                    }
                    Err(err) => {
                        // A frame that cannot be uploaded or published is
                        // logged and counted, and the stream carries on: one
                        // lost frame matters less than stalling the camera.
                        failed.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(stream = %stream_id, frame = index, error = %format!("{err:#}"), "frame not published");
                    }
                }
            }
        }));
    }

    // Decode loop: sample and push into the buffer.
    let mut sampler = Sampler::new(settings.policy);
    let mut report = PassReport::default();
    let mut decoder_failed = false;
    loop {
        let next = tokio::select! {
            biased;
            _ = shutdown.changed() => break,
            next = frames.recv() => next,
        };
        let Some(next) = next else { break };
        let frame = match next {
            Ok(frame) => frame,
            Err(err) => {
                metrics
                    .errors
                    .get_or_create(&vec![("stage", "decode".into())])
                    .inc();
                tracing::error!(clip = %clip.file_name, error = %format!("{err:#}"), "decoding failed");
                decoder_failed = true;
                break;
            }
        };
        report.decoded += 1;
        report.last_pts_s = frame.pts_s;
        metrics.decoded.get_or_create(&labels).inc();
        if !sampler.keep_frame(frame.pts_s, &frame.rgb, frame.width, frame.height) {
            continue;
        }
        let sample_index = report.sampled;
        report.sampled += 1;
        metrics.sampled.get_or_create(&labels).inc();
        match buffer
            .push(Job {
                frame,
                sample_index,
            })
            .await
        {
            Pushed::Queued { waited } => {
                metrics.backpressure.inc_by(waited.as_secs_f64());
            }
            Pushed::Dropped => {
                report.dropped += 1;
                metrics
                    .dropped
                    .get_or_create(&vec![
                        ("camera", clip.camera_id.clone()),
                        ("reason", "buffer_full".into()),
                    ])
                    .inc();
            }
            Pushed::Closed => break,
        }
        metrics.buffer_depth.set(buffer.len() as i64);
    }
    let stopped = *shutdown.borrow();
    // Let the publishers finish what is queued (graceful shutdown: decoded
    // frames are published, not lost), then send the end event.
    buffer.close();
    for worker in workers {
        let _ = worker.await;
    }
    metrics.buffer_depth.set(0);
    report.published = published.load(Ordering::Relaxed);
    report.failed = failed.load(Ordering::Relaxed);
    report.complete = !stopped && !decoder_failed;

    let end = EndEvent {
        schema: event::SCHEMA_VERSION,
        kind: "end",
        stream_id: stream_id.clone(),
        camera_id: clip.camera_id.clone(),
        clip: clip.file_name.clone(),
        frames: report.published,
        dropped: report.dropped,
        last_pts_s: report.last_pts_s,
        complete: report.complete,
        published_at: unix_now(),
    };
    let value = serde_json::to_vec(&end).expect("end event serializes");
    match sink.send(&clip.camera_id, value, headers("end")).await {
        Ok(()) => {
            metrics.streams_completed.get_or_create(&labels).inc();
        }
        Err(err) => {
            metrics
                .errors
                .get_or_create(&vec![("stage", "publish".into())])
                .inc();
            tracing::error!(stream = %stream_id, error = %format!("{err:#}"), "end event not published");
        }
    }
    report
}

fn headers(kind: &str) -> BTreeMap<String, Vec<u8>> {
    BTreeMap::from([
        ("type".to_string(), kind.as_bytes().to_vec()),
        (
            "schema".to_string(),
            event::SCHEMA_VERSION.to_string().into_bytes(),
        ),
    ])
}

/// The per-stream facts every frame event of a pass shares.
#[derive(Clone, Copy)]
struct Target<'a> {
    stream_id: &'a str,
    camera_id: &'a str,
    clip_name: &'a str,
    stem: &'a str,
    quality: u8,
    prefix: &'a str,
}

async fn publish_one<S: FrameStore, K: EventSink>(
    job: Job,
    target: &Target<'_>,
    store: &Arc<S>,
    sink: &K,
    metrics: &Metrics,
) -> Result<()> {
    let Target {
        stream_id,
        camera_id,
        clip_name,
        stem,
        quality,
        prefix,
    } = *target;
    let Job {
        frame,
        sample_index,
    } = job;
    let (width, height, index, pts_s, decoded_at) = (
        frame.width,
        frame.height,
        frame.index,
        frame.pts_s,
        frame.decoded_at,
    );

    // JPEG encoding is CPU-bound: keep it off the async worker threads.
    let started = Instant::now();
    let jpeg = tokio::task::spawn_blocking(move || encode_jpeg(&frame, quality))
        .await
        .context("encoder task")?
        .inspect_err(|_| {
            metrics
                .errors
                .get_or_create(&vec![("stage", "encode".into())])
                .inc();
        })?;
    metrics
        .encode_seconds
        .observe(started.elapsed().as_secs_f64());

    let key = event::frame_key(prefix, camera_id, stem, index);
    let digest = sha256_hex(&jpeg);
    let size = jpeg.len();
    let started = Instant::now();
    {
        let store = Arc::clone(store);
        let key = key.clone();
        tokio::task::spawn_blocking(move || store.put(&key, &jpeg))
            .await
            .context("upload task")?
            .inspect_err(|_| {
                metrics
                    .errors
                    .get_or_create(&vec![("stage", "upload".into())])
                    .inc();
            })?;
    }
    metrics
        .upload_seconds
        .observe(started.elapsed().as_secs_f64());

    let event = FrameEvent {
        schema: event::SCHEMA_VERSION,
        kind: "frame",
        frame_id: event::frame_id(stream_id, index),
        stream_id: stream_id.to_string(),
        camera_id: camera_id.to_string(),
        clip: clip_name.to_string(),
        frame_index: index,
        sample_index,
        pts_s,
        width,
        height,
        decoded_at,
        published_at: unix_now(),
        image: ObjectRef {
            bucket: store.bucket().to_string(),
            key,
        },
        content_type: "image/jpeg",
        bytes: size,
        sha256: digest,
    };
    let value = serde_json::to_vec(&event).context("serializing frame event")?;
    let started = Instant::now();
    sink.send(camera_id, value, headers("frame"))
        .await
        .inspect_err(|_| {
            metrics
                .errors
                .get_or_create(&vec![("stage", "publish".into())])
                .inc();
        })?;
    metrics
        .publish_seconds
        .observe(started.elapsed().as_secs_f64());
    metrics
        .frame_age_seconds
        .observe((unix_now() - decoded_at).max(0.0));
    Ok(())
}

#[cfg(test)]
pub mod fakes {
    use super::*;
    use std::sync::Mutex;
    use std::time::Duration;

    #[derive(Default)]
    pub struct MemoryStore {
        pub objects: Mutex<BTreeMap<String, Vec<u8>>>,
        pub delay: Duration,
        pub fail_every: Option<u64>,
        pub calls: AtomicU64,
    }

    impl MemoryStore {
        pub fn slow(delay: Duration) -> Self {
            Self {
                delay,
                ..Default::default()
            }
        }
    }

    impl FrameStore for MemoryStore {
        fn bucket(&self) -> &str {
            "datasets"
        }
        fn put(&self, key: &str, body: &[u8]) -> Result<()> {
            let n = self.calls.fetch_add(1, Ordering::Relaxed) + 1;
            if let Some(every) = self.fail_every
                && n.is_multiple_of(every)
            {
                anyhow::bail!("injected upload failure");
            }
            std::thread::sleep(self.delay);
            self.objects
                .lock()
                .unwrap()
                .insert(key.to_string(), body.to_vec());
            Ok(())
        }
    }

    #[derive(Default)]
    pub struct MemorySink {
        pub records: Mutex<Vec<(String, serde_json::Value)>>,
    }

    impl EventSink for MemorySink {
        async fn send(
            &self,
            key: &str,
            value: Vec<u8>,
            _headers: BTreeMap<String, Vec<u8>>,
        ) -> Result<()> {
            let parsed = serde_json::from_slice(&value)?;
            self.records.lock().unwrap().push((key.to_string(), parsed));
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fakes::*;
    use super::*;
    use std::time::Duration;
    use tokio::sync::mpsc;

    fn frame(index: u64, fps: f64) -> RawFrame {
        let (w, h) = (16u32, 8u32);
        RawFrame {
            index,
            pts_s: index as f64 / fps,
            width: w,
            height: h,
            rgb: vec![(index % 255) as u8; (w * h * 3) as usize],
            decoded_at: unix_now(),
        }
    }

    fn settings(policy: Policy, overflow: Overflow, buffer: usize, publishers: usize) -> Settings {
        Settings {
            policy,
            overflow,
            buffer_size: buffer,
            publishers,
            jpeg_quality: 80,
            key_prefix: "frames".into(),
            run_id: "run".into(),
        }
    }

    /// Feed `n` frames at `fps` through a channel, like the decoder does.
    fn feed(n: u64, fps: f64) -> mpsc::Receiver<Result<RawFrame>> {
        let (tx, rx) = mpsc::channel(2);
        tokio::spawn(async move {
            for i in 0..n {
                if tx.send(Ok(frame(i, fps))).await.is_err() {
                    return;
                }
            }
        });
        rx
    }

    fn clip() -> Clip {
        Clip::new(Path::new("/footage/cam-a__morning.mp4"))
    }

    #[tokio::test]
    async fn every_sampled_frame_is_uploaded_then_published_then_ended() {
        let store = Arc::new(MemoryStore::default());
        let sink = Arc::new(MemorySink::default());
        let metrics = Metrics::new();
        let (_tx, rx) = watch::channel(false);
        let report = run_pass(
            &clip(),
            0,
            feed(30, 30.0),
            &settings(Policy::TargetFps(10.0), Overflow::Block, 4, 3),
            &Outputs {
                store: Arc::clone(&store),
                sink: Arc::clone(&sink),
                metrics: metrics.clone(),
            },
            rx,
        )
        .await;
        assert_eq!(report.decoded, 30);
        assert_eq!(report.sampled, 10);
        assert_eq!(report.published, 10);
        assert_eq!(report.dropped, 0);
        assert!(report.complete);
        let records = sink.records.lock().unwrap();
        assert_eq!(records.len(), 11);
        let (key, end) = records.last().unwrap();
        assert_eq!(key, "cam-a");
        assert_eq!(end["type"], "end");
        assert_eq!(end["frames"], 10);
        assert_eq!(end["complete"], true);
        let objects = store.objects.lock().unwrap();
        let mut indexes = Vec::new();
        for (key, event) in records.iter().take(10) {
            assert_eq!(key, "cam-a");
            assert_eq!(event["type"], "frame");
            assert_eq!(event["camera_id"], "cam-a");
            let object = event["image"]["key"].as_str().unwrap();
            // The image was in the store before the event was published.
            let body = objects.get(object).expect("uploaded before published");
            assert_eq!(&body[..2], &[0xFF, 0xD8], "JPEG magic");
            assert_eq!(event["sha256"], sha256_hex(body));
            indexes.push(event["frame_index"].as_u64().unwrap());
        }
        indexes.sort_unstable();
        assert_eq!(indexes, vec![0, 3, 6, 9, 12, 15, 18, 21, 24, 27]);
    }

    #[tokio::test]
    async fn drop_policy_counts_frames_that_did_not_fit() {
        // A slow store (20 ms per upload) and a tiny buffer: frames arriving
        // faster than one publisher can upload are dropped and counted.
        let store = Arc::new(MemoryStore::slow(Duration::from_millis(20)));
        let sink = Arc::new(MemorySink::default());
        let metrics = Metrics::new();
        let (_tx, rx) = watch::channel(false);
        let report = run_pass(
            &clip(),
            0,
            feed(40, 30.0),
            &settings(Policy::All, Overflow::DropNewest, 1, 1),
            &Outputs {
                store,
                sink: Arc::clone(&sink),
                metrics: metrics.clone(),
            },
            rx,
        )
        .await;
        assert_eq!(report.sampled, 40);
        assert!(report.dropped > 0, "{report:?}");
        assert_eq!(report.published + report.dropped, 40);
        let text = metrics.render();
        assert!(text.contains("reason=\"buffer_full\""), "{text}");
    }

    #[tokio::test]
    async fn block_policy_loses_nothing_and_records_backpressure() {
        let store = Arc::new(MemoryStore::slow(Duration::from_millis(5)));
        let sink = Arc::new(MemorySink::default());
        let metrics = Metrics::new();
        let (_tx, rx) = watch::channel(false);
        let report = run_pass(
            &clip(),
            0,
            feed(20, 30.0),
            &settings(Policy::All, Overflow::Block, 1, 1),
            &Outputs {
                store,
                sink,
                metrics: metrics.clone(),
            },
            rx,
        )
        .await;
        assert_eq!((report.published, report.dropped), (20, 0));
        assert!(metrics.backpressure.get() > 0.0);
    }

    #[tokio::test]
    async fn upload_failures_skip_the_frame_but_not_the_stream() {
        let store = Arc::new(MemoryStore {
            fail_every: Some(3),
            ..Default::default()
        });
        let sink = Arc::new(MemorySink::default());
        let metrics = Metrics::new();
        let (_tx, rx) = watch::channel(false);
        let report = run_pass(
            &clip(),
            0,
            feed(9, 30.0),
            &settings(Policy::All, Overflow::Block, 4, 1),
            &Outputs {
                store,
                sink: Arc::clone(&sink),
                metrics: metrics.clone(),
            },
            rx,
        )
        .await;
        assert_eq!(report.failed, 3);
        assert_eq!(report.published, 6);
        assert!(metrics.render().contains("stage=\"upload\""));
        // No event was published for a frame whose upload failed.
        assert_eq!(sink.records.lock().unwrap().len(), 7);
    }

    #[tokio::test]
    async fn shutdown_publishes_what_was_queued_and_marks_the_stream_incomplete() {
        let store = Arc::new(MemoryStore::slow(Duration::from_millis(10)));
        let sink = Arc::new(MemorySink::default());
        let metrics = Metrics::new();
        let (tx, rx) = watch::channel(false);
        // An endless decoder.
        let (ftx, frames) = mpsc::channel(2);
        tokio::spawn(async move {
            for i in 0.. {
                if ftx.send(Ok(frame(i, 30.0))).await.is_err() {
                    return;
                }
            }
        });
        let stopper = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            tx.send(true).unwrap();
        });
        let report = run_pass(
            &clip(),
            0,
            frames,
            &settings(Policy::All, Overflow::Block, 4, 2),
            &Outputs {
                store,
                sink: Arc::clone(&sink),
                metrics: metrics.clone(),
            },
            rx,
        )
        .await;
        stopper.await.unwrap();
        assert!(!report.complete);
        assert!(report.published > 0);
        assert_eq!(
            report.published, report.sampled,
            "queued frames were published"
        );
        let records = sink.records.lock().unwrap();
        let end = &records.last().unwrap().1;
        assert_eq!(end["type"], "end");
        assert_eq!(end["complete"], false);
    }

    #[tokio::test]
    async fn a_decoder_error_ends_the_pass_as_incomplete() {
        let (tx, frames) = mpsc::channel(2);
        tokio::spawn(async move {
            tx.send(Ok(frame(0, 30.0))).await.unwrap();
            tx.send(Err(anyhow::anyhow!("corrupt packet")))
                .await
                .unwrap();
        });
        let sink = Arc::new(MemorySink::default());
        let metrics = Metrics::new();
        let (_tx, rx) = watch::channel(false);
        let report = run_pass(
            &clip(),
            0,
            frames,
            &settings(Policy::All, Overflow::Block, 4, 1),
            &Outputs {
                store: Arc::new(MemoryStore::default()),
                sink,
                metrics: metrics.clone(),
            },
            rx,
        )
        .await;
        assert_eq!(report.published, 1);
        assert!(!report.complete);
        assert!(metrics.render().contains("stage=\"decode\""));
    }

    #[test]
    fn jpeg_encoding_produces_a_jpeg() {
        let jpeg = encode_jpeg(&frame(1, 30.0), 85).unwrap();
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8]);
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9]);
    }

    #[test]
    fn clip_names() {
        let c = clip();
        assert_eq!(c.camera_id, "cam-a");
        assert_eq!(c.stem, "cam-a__morning");
        assert_eq!(c.file_name, "cam-a__morning.mp4");
    }
}
