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

/// How to choose frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Policy {
    /// Keep every frame.
    All,
    /// Keep one frame every `1 / fps` seconds of video time.
    TargetFps(f64),
    /// Keep every n-th decoded frame (n >= 1).
    EveryNth(u64),
}

impl Policy {
    /// Parse `all`, `fps:<rate>` or `every:<n>` (the CLI and env format).
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        if text.eq_ignore_ascii_case("all") {
            return Ok(Policy::All);
        }
        if let Some(rate) = text.strip_prefix("fps:") {
            let rate: f64 = rate
                .parse()
                .map_err(|_| format!("bad frame rate in {text:?}"))?;
            if !(rate.is_finite() && rate > 0.0) {
                return Err(format!("frame rate must be positive, got {text:?}"));
            }
            return Ok(Policy::TargetFps(rate));
        }
        if let Some(n) = text.strip_prefix("every:") {
            let n: u64 = n.parse().map_err(|_| format!("bad count in {text:?}"))?;
            if n == 0 {
                return Err("every:<n> needs n >= 1".into());
            }
            return Ok(Policy::EveryNth(n));
        }
        Err(format!(
            "unknown sampling policy {text:?} (expected all, fps:<rate> or every:<n>)"
        ))
    }
}

/// Stateful sampler for one pass over one clip.
#[derive(Debug, Clone)]
pub struct Sampler {
    policy: Policy,
    /// Video time at which the next frame is due (TargetFps).
    next_due_s: Option<f64>,
    seen: u64,
}

impl Sampler {
    pub fn new(policy: Policy) -> Self {
        Self {
            policy,
            next_due_s: None,
            seen: 0,
        }
    }

    /// Should the frame at `pts_s` (seconds from the start of the clip) be kept?
    ///
    /// Frames must be offered in decode order. Timestamps that go backwards
    /// (a broken file) never cause a frame to be kept twice for one slot.
    pub fn keep(&mut self, pts_s: f64) -> bool {
        let index = self.seen;
        self.seen += 1;
        match self.policy {
            Policy::All => true,
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
}
