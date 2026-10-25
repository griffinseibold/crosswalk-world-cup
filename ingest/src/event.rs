//! The records the sampler publishes to Kafka.
//!
//! A frame event carries metadata and a pointer to the JPEG in S3, never the
//! image itself: Kafka is good at many small ordered records and bad at
//! multi-hundred-kilobyte blobs (broker memory, replication traffic, the
//! default 1 MB message limit). The Python schema of the same records lives
//! in `src/crosswalk/stream/events.py`; `tests/test_stream_events.py` checks
//! that a record written here (`testdata/frame-event.json`) parses there.
//!
//! Identity, which every downstream stage relies on for idempotency:
//!
//! - `stream_id` = `<camera>/<clip>/<run>-<pass>`: one pass over one clip by
//!   one sampler process. A restarted sampler gets a new `run`, so a replay
//!   is a new stream, while a Kafka redelivery carries the same ids.
//! - `frame_id` = `<stream_id>/<frame index>`: unique per decoded frame.
//!   Consumers deduplicate on it.
//! - The S3 key does not contain the run or pass, so looping over a clip
//!   overwrites identical objects instead of filling the bucket.

use serde::Serialize;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ObjectRef {
    pub bucket: String,
    pub key: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct FrameEvent {
    pub schema: u32,
    #[serde(rename = "type")]
    pub kind: &'static str, // "frame"
    pub frame_id: String,
    pub stream_id: String,
    pub camera_id: String,
    pub clip: String,
    /// Index of the frame in decode order within the clip (every frame
    /// counts, sampled or not), so gaps show how much was skipped.
    pub frame_index: u64,
    /// Index among the frames this stream kept.
    pub sample_index: u64,
    /// Presentation time in the clip, seconds.
    pub pts_s: f64,
    pub width: u32,
    pub height: u32,
    /// Wall clock (Unix seconds) when FFmpeg handed over the frame.
    pub decoded_at: f64,
    /// Wall clock when the record was produced; with `decoded_at` this shows
    /// time spent in the buffer and the upload.
    pub published_at: f64,
    pub image: ObjectRef,
    pub content_type: &'static str,
    pub bytes: usize,
    pub sha256: String,
}

/// Published after the last frame of a stream, on the same partition, so a
/// consumer knows the stream is complete (tracking flushes its tracks,
/// indicators close their last window).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EndEvent {
    pub schema: u32,
    #[serde(rename = "type")]
    pub kind: &'static str, // "end"
    pub stream_id: String,
    pub camera_id: String,
    pub clip: String,
    /// Frames published for this stream (dropped frames are not counted).
    pub frames: u64,
    /// Frames decoded but discarded by the overflow policy.
    pub dropped: u64,
    /// Last presentation time seen, seconds.
    pub last_pts_s: f64,
    /// false if the stream was cut short (shutdown, decoder failure).
    pub complete: bool,
    pub published_at: f64,
}

/// `<camera>` from a clip file name: the stem up to the first `__`.
///
/// `intersection-01__2026-10-01_0800.mp4` and `intersection-01.mp4` both
/// belong to camera `intersection-01`, so several recordings from one camera
/// rank as one crossing. The Python side uses the same rule
/// (`crosswalk.stream.events.camera_id_from_clip`).
pub fn camera_id_from_clip(file_name: &str) -> String {
    let stem = match file_name.rsplit_once('.') {
        Some((stem, _)) if !stem.is_empty() => stem,
        _ => file_name,
    };
    let camera = stem.split("__").next().unwrap_or(stem);
    sanitize(camera)
}

/// Keep ids safe for S3 keys, Kafka keys and URLs: `[A-Za-z0-9._-]`.
pub fn sanitize(text: &str) -> String {
    let cleaned: String = text
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    // Leading dots would make hidden files and odd S3 keys (`./..`).
    let trimmed = cleaned.trim_matches(|c| c == '-' || c == '.');
    if trimmed.is_empty() {
        "unnamed".to_string()
    } else {
        trimmed.to_string()
    }
}

pub fn stream_id(camera_id: &str, clip_stem: &str, run_id: &str, pass: u64) -> String {
    format!("{camera_id}/{clip_stem}/{run_id}-{pass}")
}

pub fn frame_id(stream_id: &str, frame_index: u64) -> String {
    format!("{stream_id}/{frame_index}")
}

/// Deterministic S3 key: `<prefix>/<camera>/<clip>/<frame index>.jpg`.
pub fn frame_key(prefix: &str, camera_id: &str, clip_stem: &str, frame_index: u64) -> String {
    let prefix = prefix.trim_matches('/');
    let tail = format!("{camera_id}/{clip_stem}/{frame_index:08}.jpg");
    if prefix.is_empty() {
        tail
    } else {
        format!("{prefix}/{tail}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_ids_come_from_the_stem() {
        assert_eq!(
            camera_id_from_clip("intersection-01.mp4"),
            "intersection-01"
        );
        assert_eq!(
            camera_id_from_clip("intersection-01__2026-10-01_0800.mp4"),
            "intersection-01"
        );
        assert_eq!(camera_id_from_clip("Main St & 3rd.mkv"), "Main-St---3rd");
        assert_eq!(camera_id_from_clip("noext"), "noext");
        assert_eq!(camera_id_from_clip(".mp4"), "mp4");
        assert_eq!(camera_id_from_clip("__x.mp4"), "unnamed");
    }

    #[test]
    fn ids_and_keys() {
        let stream = stream_id("cam-a", "cam-a__morning", "20261025T101500Z", 2);
        assert_eq!(stream, "cam-a/cam-a__morning/20261025T101500Z-2");
        assert_eq!(frame_id(&stream, 41), format!("{stream}/41"));
        assert_eq!(
            frame_key("/frames/", "cam-a", "cam-a__morning", 41),
            "frames/cam-a/cam-a__morning/00000041.jpg"
        );
        assert_eq!(frame_key("", "c", "k", 1), "c/k/00000001.jpg");
    }

    #[test]
    fn frame_event_serializes_to_the_documented_shape() {
        let event = FrameEvent {
            schema: SCHEMA_VERSION,
            kind: "frame",
            frame_id: "cam/clip/run-0/12".into(),
            stream_id: "cam/clip/run-0".into(),
            camera_id: "cam".into(),
            clip: "clip.mp4".into(),
            frame_index: 12,
            sample_index: 2,
            pts_s: 0.4,
            width: 1280,
            height: 720,
            decoded_at: 1.5,
            published_at: 1.75,
            image: ObjectRef {
                bucket: "datasets".into(),
                key: "frames/cam/clip/00000012.jpg".into(),
            },
            content_type: "image/jpeg",
            bytes: 100,
            sha256: "ab".into(),
        };
        let value: serde_json::Value = serde_json::to_value(&event).unwrap();
        assert_eq!(value["type"], "frame");
        assert_eq!(value["image"]["key"], "frames/cam/clip/00000012.jpg");
        // The checked-in fixture the Python tests read must stay in step with
        // this struct: same set of fields.
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/frame-event.json")).unwrap();
        let keys = |v: &serde_json::Value| {
            let mut k: Vec<String> = v.as_object().unwrap().keys().cloned().collect();
            k.sort();
            k
        };
        assert_eq!(keys(&value), keys(&fixture));
    }
}
