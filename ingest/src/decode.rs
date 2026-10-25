//! Decoding with FFmpeg as a subprocess.
//!
//! FFmpeg does all the decoding: every codec and container it supports works,
//! and the sampler never links libav (no C build, no ABI to keep in step). The
//! sampler runs
//!
//! ```text
//! ffmpeg -i <clip> -vf [scale=W:H,]showinfo -fps_mode passthrough \
//!        -pix_fmt rgb24 -f rawvideo pipe:1
//! ```
//!
//! and reads fixed-size RGB frames (`width * height * 3` bytes) from stdout.
//! The `showinfo` filter prints one line per frame to stderr with its
//! presentation timestamp (`pts_time:`), which is how each frame gets an exact
//! time without parsing the container. `-fps_mode passthrough` stops FFmpeg
//! from duplicating or dropping frames to hit a constant rate, so frame N on
//! stdout is the N-th `showinfo` line on stderr.
//!
//! Frame size comes from `ffprobe` before decoding starts (or from the
//! requested resize), because raw video has no header to read it from.

use std::collections::VecDeque;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

/// FFmpeg log lines (other than per-frame showinfo) kept for error messages.
const LOG_TAIL: usize = 20;

/// One decoded frame, RGB24, row-major.
#[derive(Debug)]
pub struct RawFrame {
    pub index: u64,
    pub pts_s: f64,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub decoded_at: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
}

pub fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Parse `ffprobe -show_entries stream=width,height,avg_frame_rate -of csv=p=0`
/// output, e.g. `1280,720,30000/1001`.
pub fn parse_probe(text: &str) -> Result<ProbeInfo> {
    let line = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .ok_or_else(|| anyhow!("ffprobe printed nothing"))?;
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    if fields.len() < 3 {
        bail!("unexpected ffprobe output {line:?}");
    }
    let width: u32 = fields[0]
        .parse()
        .with_context(|| format!("width in {line:?}"))?;
    let height: u32 = fields[1]
        .parse()
        .with_context(|| format!("height in {line:?}"))?;
    let fps = parse_rate(fields[2]).unwrap_or(0.0);
    if width == 0 || height == 0 {
        bail!("ffprobe reported a {width}x{height} video stream");
    }
    Ok(ProbeInfo { width, height, fps })
}

/// `30000/1001` -> 29.97; `0/0` (unknown) -> None.
pub fn parse_rate(text: &str) -> Option<f64> {
    let (num, den) = match text.split_once('/') {
        Some((n, d)) => (n.parse::<f64>().ok()?, d.parse::<f64>().ok()?),
        None => (text.parse::<f64>().ok()?, 1.0),
    };
    (num > 0.0 && den > 0.0).then(|| num / den)
}

/// Extract `pts_time` from one `showinfo` stderr line, if it is a frame line.
///
/// Frame lines look like
/// `[Parsed_showinfo_1 @ 0x...] n:  12 pts: 6144 pts_time:0.48 duration: ...`.
/// Configuration lines from the same filter (`config in time_base: ...`) and
/// FFmpeg's other log output are ignored.
pub fn parse_showinfo_pts(line: &str) -> Option<f64> {
    if !line.contains("showinfo") || !line.contains(" n:") {
        return None;
    }
    let start = line.find("pts_time:")? + "pts_time:".len();
    let rest = &line[start..];
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let value = &rest[..end];
    if value == "NOPTS" {
        return Some(f64::NAN);
    }
    value.parse().ok()
}

pub async fn probe(ffprobe: &str, path: &Path) -> Result<ProbeInfo> {
    let output = Command::new(ffprobe)
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .await
        .with_context(|| format!("running {ffprobe} (is FFmpeg installed?)"))?;
    if !output.status.success() {
        bail!(
            "ffprobe failed on {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    parse_probe(&String::from_utf8_lossy(&output.stdout))
        .with_context(|| format!("probing {}", path.display()))
}

/// The ffmpeg argument list for one clip (separate for testing).
pub fn ffmpeg_args(path: &Path, resize: Option<(u32, u32)>, realtime: bool) -> Vec<String> {
    let mut args: Vec<String> = vec![
        "-nostdin".into(),
        "-hide_banner".into(),
        "-nostats".into(),
        // showinfo logs at "info"; everything else at info is a few header
        // lines per clip, which are skipped by the parser.
        "-loglevel".into(),
        "info".into(),
    ];
    if realtime {
        // Read input at its native frame rate, like a camera.
        args.push("-re".into());
    }
    args.push("-i".into());
    args.push(path.to_string_lossy().into_owned());
    args.extend(["-an", "-sn", "-dn"].map(String::from));
    let filter = match resize {
        Some((w, h)) => format!("scale={w}:{h},showinfo"),
        None => "showinfo".to_string(),
    };
    args.extend([
        "-vf".to_string(),
        filter,
        "-fps_mode".to_string(),
        "passthrough".to_string(),
        "-pix_fmt".to_string(),
        "rgb24".to_string(),
        "-f".to_string(),
        "rawvideo".to_string(),
        "pipe:1".to_string(),
    ]);
    args
}

/// A running decode of one clip. Frames arrive on `frames`; dropping or
/// calling [`Decoder::stop`] ends FFmpeg.
pub struct Decoder {
    child: Child,
    pub frames: mpsc::Receiver<Result<RawFrame>>,
}

impl Decoder {
    /// Start FFmpeg on `path`. `info` comes from [`probe`]; frames come out
    /// at `resize` if given, else at the probed size.
    pub async fn start(
        ffmpeg: &str,
        path: &Path,
        info: &ProbeInfo,
        resize: Option<(u32, u32)>,
        realtime: bool,
    ) -> Result<Self> {
        let (width, height) = resize.unwrap_or((info.width, info.height));
        let fps = info.fps;
        // Used only if a frame's timestamp never arrives (see below).
        let fallback_fps = if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            30.0
        };
        let mut child = Command::new(ffmpeg)
            .args(ffmpeg_args(path, resize, realtime))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("starting {ffmpeg} (is FFmpeg installed?)"))?;
        let mut stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");

        // Timestamps (stderr) and pixels (stdout) arrive on separate pipes.
        // An unbounded channel for timestamps is fine: one f64 per frame,
        // and stdout is read as fast as the frame channel allows. Other
        // stderr lines are kept in a small ring for error messages.
        let (pts_tx, mut pts_rx) = mpsc::unbounded_channel::<f64>();
        let log_tail = Arc::new(Mutex::new(VecDeque::<String>::with_capacity(LOG_TAIL)));
        let tail_writer = Arc::clone(&log_tail);
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                match parse_showinfo_pts(&line) {
                    Some(pts) => {
                        let _ = pts_tx.send(pts);
                    }
                    None => {
                        let mut tail = tail_writer.lock().expect("log tail lock");
                        if tail.len() == LOG_TAIL {
                            tail.pop_front();
                        }
                        tail.push_back(line);
                    }
                }
            }
        });

        // One frame of slack between FFmpeg and the sampler: the real buffer
        // with an overflow policy is further downstream (buffer.rs).
        let (tx, frames) = mpsc::channel::<Result<RawFrame>>(2);
        let frame_bytes = width as usize * height as usize * 3;
        tokio::spawn(async move {
            let mut index: u64 = 0;
            loop {
                let mut rgb = vec![0u8; frame_bytes];
                match stdout.read_exact(&mut rgb).await {
                    Ok(_) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::UnexpectedEof => break,
                    Err(err) => {
                        let _ = tx
                            .send(Err(anyhow!("reading frames from ffmpeg: {err}")))
                            .await;
                        break;
                    }
                }
                // The showinfo line for a frame is printed before the frame
                // is written, but over a different pipe; wait briefly for it.
                // 30 s is generous: the line is written before the frame, and only a
                // stalled stderr reader could delay it.
                let pts = match tokio::time::timeout(Duration::from_secs(30), pts_rx.recv()).await {
                    Ok(Some(p)) if p.is_finite() => p,
                    // Missing or NOPTS: fall back to the frame index at the
                    // probed rate, so time still increases at the right pace.
                    _ => {
                        tracing::warn!(
                            frame = index,
                            "no timestamp from ffmpeg; using index / fps"
                        );
                        index as f64 / fallback_fps
                    }
                };
                let frame = RawFrame {
                    index,
                    pts_s: pts,
                    width,
                    height,
                    rgb,
                    decoded_at: unix_now(),
                };
                index += 1;
                if tx.send(Ok(frame)).await.is_err() {
                    break; // receiver gone: stopping
                }
            }
            // Report the last FFmpeg log lines if nothing was decoded, which
            // is almost always a bad file or an unsupported codec.
            if index == 0 {
                // stderr may still be draining; give it a moment.
                tokio::time::sleep(Duration::from_millis(200)).await;
                let tail: Vec<String> = log_tail
                    .lock()
                    .expect("log tail lock")
                    .iter()
                    .filter(|l| !l.trim().is_empty())
                    .cloned()
                    .collect();
                let last: Vec<&str> = tail
                    .iter()
                    .rev()
                    .take(3)
                    .rev()
                    .map(String::as_str)
                    .collect();
                let _ = tx
                    .send(Err(anyhow!(
                        "ffmpeg produced no frames: {}",
                        last.join(" | ")
                    )))
                    .await;
            }
        });
        Ok(Self { child, frames })
    }

    /// Separate the frame stream from the process handle, so the frames can
    /// be handed to the pipeline while the caller keeps control of FFmpeg.
    pub fn split(self) -> (mpsc::Receiver<Result<RawFrame>>, Process) {
        (self.frames, Process(self.child))
    }

    /// Stop FFmpeg (if still running) and reap it.
    #[cfg(test)]
    pub async fn stop(self) -> Result<Option<i32>> {
        let (mut frames, process) = self.split();
        frames.close();
        process.stop().await
    }

    /// Wait for FFmpeg to exit on its own (end of clip).
    #[cfg(test)]
    pub async fn wait(self) -> Result<Option<i32>> {
        self.split().1.wait().await
    }
}

/// The FFmpeg child process for one clip.
pub struct Process(Child);

impl Process {
    /// Kill FFmpeg if it is still running (shutdown, or the pipeline stopped
    /// reading) and reap it so no zombie is left behind.
    pub async fn stop(mut self) -> Result<Option<i32>> {
        if self.0.try_wait()?.is_none() {
            let _ = self.0.start_kill();
        }
        Ok(self.0.wait().await?.code())
    }

    #[cfg(test)]
    pub async fn wait(mut self) -> Result<Option<i32>> {
        Ok(self.0.wait().await?.code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_output_parses() {
        assert_eq!(
            parse_probe("1280,720,30000/1001\n").unwrap(),
            ProbeInfo {
                width: 1280,
                height: 720,
                fps: 30000.0 / 1001.0
            }
        );
        assert_eq!(parse_probe("\n640,360,0/0\n").unwrap().fps, 0.0);
        assert!(parse_probe("").is_err());
        assert!(parse_probe("0,0,25/1").is_err());
        assert!(parse_probe("abc").is_err());
    }

    #[test]
    fn rates() {
        assert_eq!(parse_rate("25/1"), Some(25.0));
        assert_eq!(parse_rate("30"), Some(30.0));
        assert_eq!(parse_rate("0/0"), None);
        assert_eq!(parse_rate("x"), None);
    }

    #[test]
    fn showinfo_lines() {
        let frame = "[Parsed_showinfo_1 @ 0x5623d39f5200] n:   0 pts:      0 pts_time:0       duration:    512 duration_time:0.04    fmt:rgb24";
        assert_eq!(parse_showinfo_pts(frame), Some(0.0));
        let later = "[Parsed_showinfo_1 @ 0x1] n:  12 pts:   6144 pts_time:0.48    duration: 512";
        assert_eq!(parse_showinfo_pts(later), Some(0.48));
        let config = "[Parsed_showinfo_1 @ 0x1] config in time_base: 1/12800, frame_rate: 25/1";
        assert_eq!(parse_showinfo_pts(config), None);
        let colour = "[Parsed_showinfo_1 @ 0x1] color_range:pc color_space:gbr";
        assert_eq!(parse_showinfo_pts(colour), None);
        assert_eq!(parse_showinfo_pts("Input #0, mov,mp4 from 'x.mp4':"), None);
        let nopts = "[Parsed_showinfo_0 @ 0x1] n:   3 pts:NOPTS pts_time:NOPTS duration: 1";
        assert!(parse_showinfo_pts(nopts).unwrap().is_nan());
    }

    #[test]
    fn args_include_scale_and_passthrough() {
        let args = ffmpeg_args(Path::new("/c/clip.mp4"), Some((640, 360)), false);
        let joined = args.join(" ");
        assert!(joined.contains("-i /c/clip.mp4"));
        assert!(joined.contains("-vf scale=640:360,showinfo"));
        assert!(joined.contains("-fps_mode passthrough"));
        assert!(joined.ends_with("-pix_fmt rgb24 -f rawvideo pipe:1"));
        assert!(!joined.contains("-re"));
        let rt = ffmpeg_args(Path::new("x.mp4"), None, true).join(" ");
        assert!(rt.contains("-re -i x.mp4") && rt.contains("-vf showinfo"));
    }

    /// Decodes a clip generated by FFmpeg's own test source. Needs ffmpeg and
    /// ffprobe on PATH (CI installs them); skipped with a message otherwise.
    #[tokio::test]
    async fn decodes_a_generated_clip_with_timestamps() {
        let Some(dir) = crate::testutil::clip_dir() else {
            return;
        };
        let clip = crate::testutil::make_clip(&dir, "decode.mp4", 25, 50, (160, 96));
        let info = probe("ffprobe", &clip).await.unwrap();
        assert_eq!((info.width, info.height), (160, 96));
        assert!((info.fps - 25.0).abs() < 1e-6);
        let mut decoder = Decoder::start("ffmpeg", &clip, &info, Some((80, 48)), false)
            .await
            .unwrap();
        let mut frames = Vec::new();
        while let Some(frame) = decoder.frames.recv().await {
            frames.push(frame.unwrap());
        }
        assert_eq!(decoder.wait().await.unwrap(), Some(0));
        assert_eq!(frames.len(), 50);
        assert_eq!(frames[0].rgb.len(), 80 * 48 * 3);
        assert_eq!(
            frames.iter().map(|f| f.index).collect::<Vec<_>>(),
            (0..50).collect::<Vec<_>>()
        );
        assert!(
            (frames[25].pts_s - 1.0).abs() < 1e-6,
            "pts {}",
            frames[25].pts_s
        );
    }

    #[tokio::test]
    async fn a_bad_file_reports_ffmpegs_error() {
        let Some(dir) = crate::testutil::clip_dir() else {
            return;
        };
        let bad = dir.join("not-a-video.mp4");
        std::fs::write(&bad, b"definitely not a video").unwrap();
        assert!(probe("ffprobe", &bad).await.is_err());
        let mut decoder = Decoder::start(
            "ffmpeg",
            &bad,
            &ProbeInfo {
                width: 16,
                height: 16,
                fps: 0.0,
            },
            None,
            false,
        )
        .await
        .unwrap();
        let first = decoder.frames.recv().await.expect("an error is reported");
        let err = first.expect_err("decoding garbage fails");
        assert!(err.to_string().contains("no frames"), "{err}");
        let _ = decoder.stop().await;
    }
}
