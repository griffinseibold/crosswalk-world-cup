//! Helpers for tests that need real FFmpeg.

use std::path::{Path, PathBuf};
use std::process::Command;

/// A fresh temporary directory, or `None` (with a note on stderr) when
/// ffmpeg/ffprobe are not installed. CI installs both, so these tests run
/// there; a developer machine without FFmpeg still runs everything else.
pub fn clip_dir() -> Option<PathBuf> {
    let available = ["ffmpeg", "ffprobe"].iter().all(|tool| {
        Command::new(tool)
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    });
    if !available {
        eprintln!("ffmpeg/ffprobe not on PATH: skipping a test that decodes video");
        return None;
    }
    let dir = std::env::temp_dir().join(format!(
        "cw-ingest-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    Some(dir)
}

static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Encode `frames` frames of FFmpeg's built-in test pattern at `fps`.
pub fn make_clip(dir: &Path, name: &str, fps: u32, frames: u32, size: (u32, u32)) -> PathBuf {
    let path = dir.join(name);
    let status = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
        ])
        .arg(format!("testsrc=size={}x{}:rate={fps}", size.0, size.1))
        .args([
            "-frames:v",
            &frames.to_string(),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "mpeg4",
        ])
        .arg(&path)
        .status()
        .expect("running ffmpeg");
    assert!(status.success(), "ffmpeg could not generate a test clip");
    path
}
