//! Which decoded frames become frame events.
//!
//! FFmpeg hands the sampler every frame of a recorded clip at its native rate
//! (often 25 or 30 fps). Detection, tracking and the conflict indicators do
//! not need all of them: a pedestrian moves a few centimetres between
//! consecutive frames at 30 fps, and every frame kept costs an S3 upload, a
//! Kafka record and a GPU slot. The sampler keeps a configurable subset.
//!
//! Sampling is decided on the frame's *presentation timestamp*, not on a
//! frame counter, so a clip with a variable frame rate (common for phone
//! recordings) is still sampled evenly in time.
//!
//! The `motion` policy adapts the rate to the scene. A crosswalk camera spends
//! much of its time looking at an empty road, where a frame a second is
//! plenty, and the rest watching people and vehicles move, where every frame
//! narrows the gap in which a close approach can be missed. Each candidate
//! frame is reduced to a coarse grid of average brightness (32 x 18 cells)
//! and compared with the last kept frame. If enough cells changed, the frame
//! is kept at the busy rate; otherwise only at the idle rate. The comparison
//! is cheap (a strided pass over the pixels, only for frames that are due at
//! the busy rate) and needs nothing beyond the RGB frame already decoded.

/// How to choose frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Policy {
    /// Keep every frame.
    All,
    /// Keep one frame every `1 / fps` seconds of video time.
    TargetFps(f64),
    /// Keep every n-th decoded frame (n >= 1).
    EveryNth(u64),
    /// Keep frames at `busy_fps` while the scene changes and at `idle_fps`
    /// while it does not. A frame counts as changed when at least
    /// `min_cells` cells of the brightness grid differ from the last kept
    /// frame by more than [`CELL_DELTA`] grey levels.
    Motion {
        idle_fps: f64,
        busy_fps: f64,
        min_cells: u32,
    },
}

/// Brightness grid used by the motion policy (cells across, cells down).
/// 16:9 like most road footage; each cell of a 1280x720 frame is 40x40 pixels.
pub const GRID_W: usize = 32;
pub const GRID_H: usize = 18;
/// Grey levels (0-255) a cell's average must move to count as changed.
/// Compression noise and sensor grain move a 40x40 average by one or two
/// levels; a pedestrian or a car entering a cell moves it by tens.
pub const CELL_DELTA: u8 = 10;
/// Sample every STRIDE-th pixel in each direction when averaging cells.
const STRIDE: usize = 4;

fn parse_rate(text: &str, whole: &str) -> Result<f64, String> {
    let rate: f64 = text
        .parse()
        .map_err(|_| format!("bad frame rate in {whole:?}"))?;
    if !(rate.is_finite() && rate > 0.0) {
        return Err(format!("frame rate must be positive, got {whole:?}"));
    }
    Ok(rate)
}

impl Policy {
    /// Parse `all`, `fps:<rate>`, `every:<n>` or
    /// `motion:<idle fps>-<busy fps>[@<min cells>]` (the CLI and env format).
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        if text.eq_ignore_ascii_case("all") {
            return Ok(Policy::All);
        }
        if let Some(rate) = text.strip_prefix("fps:") {
            return Ok(Policy::TargetFps(parse_rate(rate, text)?));
        }
        if let Some(spec) = text.strip_prefix("motion:") {
            let (rates, cells) = match spec.split_once('@') {
                Some((rates, cells)) => (rates, Some(cells)),
                None => (spec, None),
            };
            let (idle, busy) = rates
                .split_once('-')
                .ok_or_else(|| format!("motion needs <idle fps>-<busy fps>, got {text:?}"))?;
            let idle_fps = parse_rate(idle, text)?;
            let busy_fps = parse_rate(busy, text)?;
            if idle_fps > busy_fps {
                return Err(format!(
                    "motion idle rate must not exceed the busy rate, got {text:?}"
                ));
            }
            let min_cells = match cells {
                None => 1,
                Some(c) => c
                    .parse::<u32>()
                    .ok()
                    .filter(|&n| (1..=GRID_W * GRID_H).contains(&(n as usize)))
                    .ok_or_else(|| {
                        format!(
                            "motion cell count must be 1-{}, got {text:?}",
                            GRID_W * GRID_H
                        )
                    })?,
            };
            return Ok(Policy::Motion {
                idle_fps,
                busy_fps,
                min_cells,
            });
        }
        if let Some(n) = text.strip_prefix("every:") {
            let n: u64 = n.parse().map_err(|_| format!("bad count in {text:?}"))?;
            if n == 0 {
                return Err("every:<n> needs n >= 1".into());
            }
            return Ok(Policy::EveryNth(n));
        }
        Err(format!(
            "unknown sampling policy {text:?} (expected all, fps:<rate>, every:<n> or motion:<idle>-<busy>)"
        ))
    }
}

/// Average brightness of each grid cell, from a packed RGB24 frame.
///
/// Returns `None` if the buffer does not hold `width * height * 3` bytes or
/// the frame is smaller than the grid (nothing sensible to compare).
pub fn brightness_grid(rgb: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let (w, h) = (width as usize, height as usize);
    if w < GRID_W || h < GRID_H || rgb.len() != w * h * 3 {
        return None;
    }
    let mut sums = vec![0u32; GRID_W * GRID_H];
    let mut counts = vec![0u32; GRID_W * GRID_H];
    for y in (0..h).step_by(STRIDE) {
        let gy = y * GRID_H / h;
        let row = &rgb[y * w * 3..(y + 1) * w * 3];
        for x in (0..w).step_by(STRIDE) {
            let gx = x * GRID_W / w;
            let p = &row[x * 3..x * 3 + 3];
            // Integer BT.601 luma: (77 R + 150 G + 29 B) / 256.
            let luma = (77 * p[0] as u32 + 150 * p[1] as u32 + 29 * p[2] as u32) >> 8;
            sums[gy * GRID_W + gx] += luma;
            counts[gy * GRID_W + gx] += 1;
        }
    }
    Some(
        sums.iter()
            .zip(&counts)
            .map(|(&s, &c)| s.checked_div(c).unwrap_or(0) as u8)
            .collect(),
    )
}

/// Number of cells whose brightness moved by more than [`CELL_DELTA`].
pub fn changed_cells(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b)
        .filter(|&(&x, &y)| x.abs_diff(y) > CELL_DELTA)
        .count() as u32
}

/// Stateful sampler for one pass over one clip.
#[derive(Debug, Clone)]
pub struct Sampler {
    policy: Policy,
    /// Video time at which the next frame is due (TargetFps).
    next_due_s: Option<f64>,
    seen: u64,
    /// Motion: time and brightness grid of the last kept frame.
    last_kept_s: Option<f64>,
    reference: Option<Vec<u8>>,
    /// Motion: frames kept because the scene changed (for logs and tests).
    pub busy_kept: u64,
}

impl Sampler {
    pub fn new(policy: Policy) -> Self {
        Self {
            policy,
            next_due_s: None,
            seen: 0,
            last_kept_s: None,
            reference: None,
            busy_kept: 0,
        }
    }

    /// Like [`Sampler::keep`], with the decoded pixels for the motion policy.
    /// Other policies ignore the pixels.
    pub fn keep_frame(&mut self, pts_s: f64, rgb: &[u8], width: u32, height: u32) -> bool {
        match self.policy {
            Policy::Motion {
                idle_fps,
                busy_fps,
                min_cells,
            } => {
                self.seen += 1;
                self.keep_motion(pts_s, idle_fps, busy_fps, min_cells, || {
                    brightness_grid(rgb, width, height)
                })
            }
            _ => self.keep(pts_s),
        }
    }

    fn keep_motion(
        &mut self,
        pts_s: f64,
        idle_fps: f64,
        busy_fps: f64,
        min_cells: u32,
        grid: impl FnOnce() -> Option<Vec<u8>>,
    ) -> bool {
        let Some(last) = self.last_kept_s else {
            self.last_kept_s = Some(pts_s);
            self.reference = grid();
            return true;
        };
        let elapsed = pts_s - last;
        // Same 1% tolerance as TargetFps, for container timestamp rounding.
        let due = |fps: f64| elapsed + 0.01 / fps >= 1.0 / fps;
        // Backwards timestamps (elapsed < 0) are never due.
        if elapsed < 0.0 || !due(busy_fps) {
            return false;
        }
        let current = grid();
        let busy = match (&self.reference, &current) {
            (Some(r), Some(c)) => changed_cells(r, c) >= min_cells,
            // No pixels to compare: fall back to the idle schedule.
            _ => false,
        };
        if busy || due(idle_fps) {
            if busy && !due(idle_fps) {
                self.busy_kept += 1;
            }
            self.last_kept_s = Some(pts_s);
            self.reference = current;
            return true;
        }
        false
    }

    /// Should the frame at `pts_s` (seconds from the start of the clip) be kept?
    ///
    /// Frames must be offered in decode order. Timestamps that go backwards
    /// (a broken file) never cause a frame to be kept twice for one slot.
    /// Without pixels, the motion policy sees no change and samples at its
    /// idle rate; use [`Sampler::keep_frame`] for it.
    pub fn keep(&mut self, pts_s: f64) -> bool {
        let index = self.seen;
        self.seen += 1;
        match self.policy {
            Policy::All => true,
            Policy::Motion {
                idle_fps,
                busy_fps,
                min_cells,
            } => self.keep_motion(pts_s, idle_fps, busy_fps, min_cells, || None),
            Policy::EveryNth(n) => index.is_multiple_of(n),
            Policy::TargetFps(fps) => {
                let interval = 1.0 / fps;
                match self.next_due_s {
                    None => {
                        self.next_due_s = Some(pts_s + interval);
                        true
                    }
                    // A small tolerance (1% of the interval) absorbs rounding
                    // in container timestamps: at 30 fps a frame stamped
                    // 0.19999 for a slot due at 0.2 is still "on time".
                    Some(due) if pts_s + interval * 0.01 >= due => {
                        // Advance by whole intervals so the long-run rate is
                        // exactly `fps` even if a slot falls between frames;
                        // after a gap (seek, dropped frames), restart from now.
                        let mut next = due + interval;
                        if next <= pts_s {
                            next = pts_s + interval;
                        }
                        self.next_due_s = Some(next);
                        true
                    }
                    Some(_) => false,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kept(policy: Policy, fps: f64, frames: usize) -> Vec<usize> {
        let mut sampler = Sampler::new(policy);
        (0..frames)
            .filter(|&i| sampler.keep(i as f64 / fps))
            .collect()
    }

    #[test]
    fn parse_accepts_the_three_forms() {
        assert_eq!(Policy::parse("all"), Ok(Policy::All));
        assert_eq!(Policy::parse(" fps:5 "), Ok(Policy::TargetFps(5.0)));
        assert_eq!(Policy::parse("every:3"), Ok(Policy::EveryNth(3)));
        assert!(Policy::parse("fps:0").is_err());
        assert!(Policy::parse("fps:-1").is_err());
        assert!(Policy::parse("every:0").is_err());
        assert!(Policy::parse("sometimes").is_err());
    }

    #[test]
    fn all_keeps_everything() {
        assert_eq!(kept(Policy::All, 30.0, 5), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn every_nth_keeps_by_index() {
        assert_eq!(kept(Policy::EveryNth(3), 30.0, 10), vec![0, 3, 6, 9]);
    }

    #[test]
    fn target_fps_divides_evenly() {
        // 30 fps down to 5 fps: every 6th frame.
        assert_eq!(
            kept(Policy::TargetFps(5.0), 30.0, 31),
            vec![0, 6, 12, 18, 24, 30]
        );
    }

    #[test]
    fn target_fps_keeps_the_long_run_rate_when_it_does_not_divide() {
        // 25 fps down to 10 fps: 2.5 source frames per kept frame. Over 10 s
        // of video exactly 100 frames must be kept, not 125 or 84.
        let kept = kept(Policy::TargetFps(10.0), 25.0, 250);
        assert_eq!(kept.len(), 100);
        // Gaps alternate between 2 and 3 frames.
        for pair in kept.windows(2) {
            let gap = pair[1] - pair[0];
            assert!(gap == 2 || gap == 3, "gap {gap}");
        }
    }

    #[test]
    fn target_fps_above_the_source_rate_keeps_every_frame() {
        assert_eq!(kept(Policy::TargetFps(60.0), 30.0, 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn variable_frame_rate_is_sampled_in_time() {
        // Frames bunched at the start, then sparse: sampling follows time.
        let pts = [0.0, 0.01, 0.02, 0.03, 0.5, 1.0, 1.01, 1.5, 2.0];
        let mut sampler = Sampler::new(Policy::TargetFps(2.0));
        let kept: Vec<f64> = pts.iter().copied().filter(|&t| sampler.keep(t)).collect();
        assert_eq!(kept, vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn a_gap_restarts_the_schedule_instead_of_bursting() {
        let mut sampler = Sampler::new(Policy::TargetFps(1.0));
        assert!(sampler.keep(0.0));
        // Ten seconds missing from the file: keep one frame, not ten.
        assert!(sampler.keep(10.0));
        assert!(!sampler.keep(10.5));
        assert!(sampler.keep(11.0));
    }

    #[test]
    fn timestamps_going_backwards_do_not_duplicate_slots() {
        let mut sampler = Sampler::new(Policy::TargetFps(1.0));
        assert!(sampler.keep(5.0));
        assert!(!sampler.keep(4.0));
        assert!(!sampler.keep(5.5));
        assert!(sampler.keep(6.0));
    }

    /// A flat grey frame, optionally with a bright square at (x, y).
    fn frame(w: u32, h: u32, square: Option<(u32, u32)>) -> Vec<u8> {
        let mut rgb = vec![100u8; (w * h * 3) as usize];
        if let Some((sx, sy)) = square {
            for y in sy..(sy + 60).min(h) {
                for x in sx..(sx + 60).min(w) {
                    let i = ((y * w + x) * 3) as usize;
                    rgb[i..i + 3].copy_from_slice(&[250, 250, 250]);
                }
            }
        }
        rgb
    }

    #[test]
    fn parse_accepts_motion() {
        assert_eq!(
            Policy::parse("motion:1-5"),
            Ok(Policy::Motion {
                idle_fps: 1.0,
                busy_fps: 5.0,
                min_cells: 1
            })
        );
        assert_eq!(
            Policy::parse("motion:0.5-10@3"),
            Ok(Policy::Motion {
                idle_fps: 0.5,
                busy_fps: 10.0,
                min_cells: 3
            })
        );
        assert!(Policy::parse("motion:5-1").is_err());
        assert!(Policy::parse("motion:5").is_err());
        assert!(Policy::parse("motion:1-5@0").is_err());
        assert!(Policy::parse("motion:1-5@9999").is_err());
        assert!(Policy::parse("motion:0-5").is_err());
    }

    #[test]
    fn brightness_grid_sees_a_square_and_ignores_bad_buffers() {
        let (w, h) = (320, 180);
        let still = brightness_grid(&frame(w, h, None), w, h).unwrap();
        assert_eq!(still.len(), GRID_W * GRID_H);
        assert!(still.iter().all(|&v| v == still[0]));
        let moved = brightness_grid(&frame(w, h, Some((100, 60))), w, h).unwrap();
        assert!(changed_cells(&still, &moved) >= 1);
        assert_eq!(changed_cells(&still, &still), 0);
        assert!(brightness_grid(&[0; 10], w, h).is_none());
        assert!(brightness_grid(&frame(16, 9, None), 16, 9).is_none());
    }

    #[test]
    fn motion_samples_at_the_idle_rate_on_a_still_scene() {
        let (w, h) = (320, 180);
        let still = frame(w, h, None);
        let mut sampler = Sampler::new(Policy::parse("motion:1-5").unwrap());
        // 4 s at 30 fps, nothing moving: one frame per second (t = 0, 1, 2, 3).
        let kept: Vec<usize> = (0..120)
            .filter(|&i| sampler.keep_frame(i as f64 / 30.0, &still, w, h))
            .collect();
        assert_eq!(kept, vec![0, 30, 60, 90]);
        assert_eq!(sampler.busy_kept, 0);
    }

    #[test]
    fn motion_samples_at_the_busy_rate_while_something_moves() {
        let (w, h) = (320, 180);
        let mut sampler = Sampler::new(Policy::parse("motion:1-5").unwrap());
        // A square crossing the frame 4 pixels per source frame for 2 s:
        // every kept frame (0.2 s apart) differs from the previous one.
        let kept: Vec<u32> = (0..60u32)
            .filter(|&i| {
                let rgb = frame(w, h, Some((i * 4, 60)));
                sampler.keep_frame(i as f64 / 30.0, &rgb, w, h)
            })
            .collect();
        assert_eq!(kept, vec![0, 6, 12, 18, 24, 30, 36, 42, 48, 54]);
        assert!(sampler.busy_kept >= 8);
    }

    #[test]
    fn motion_without_pixels_falls_back_to_the_idle_rate() {
        assert_eq!(
            kept(Policy::parse("motion:2-10").unwrap(), 30.0, 61),
            vec![0, 15, 30, 45, 60]
        );
    }

    #[test]
    fn other_policies_ignore_the_pixels() {
        let rgb = frame(320, 180, None);
        let mut sampler = Sampler::new(Policy::TargetFps(5.0));
        let kept: Vec<usize> = (0..31)
            .filter(|&i| sampler.keep_frame(i as f64 / 30.0, &rgb, 320, 180))
            .collect();
        assert_eq!(kept, vec![0, 6, 12, 18, 24, 30]);
    }
}
