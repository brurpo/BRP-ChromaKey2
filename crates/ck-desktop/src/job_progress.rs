//! One steady progress value for a worker job.
//!
//! The worker reports several independent counters: foundation passes (`rvm`, `cradio`,
//! `moge` per frame, staged mode only), Stage1 windows, Stage2 frames, plus frequent
//! `runtime_phase` diagnostics that usually carry `0/total`. Drawing whichever counter
//! arrived last made the bar flicker and, in resident mode where Stage1 windows and
//! Stage2 frames interleave, jump back and forth. Here every work counter contributes a
//! weighted share of one job fraction, diagnostics are ignored, and the shown value never
//! moves backwards within an attempt.

use std::collections::BTreeMap;

/// Approximate share of job time per work counter (CorridorKey foundation + stages).
fn weight(phase: &str) -> f32 {
    match phase {
        "rvm" | "moge" => 0.04,
        "cradio" => 0.08,
        "stage1" => 0.5,
        "stage2" => 0.34,
        "decode" | "decode_scenes" => 0.05,
        _ => 1.0,
    }
}

/// Counters that only describe the worker's internals, not progress through the job.
fn is_diagnostic(phase: &str) -> bool {
    matches!(phase, "runtime_phase" | "window_policy" | "memory_retry")
}

/// Counters implied by one that has been seen, so early phases cannot fill the bar alone.
fn implied(phase: &str) -> &'static [&'static str] {
    match phase {
        "rvm" | "cradio" | "moge" => &["rvm", "cradio", "moge", "stage1", "stage2"],
        "stage1" | "stage2" => &["stage1", "stage2"],
        "decode" => &["sam_video"],
        "decode_scenes" => &["transnet"],
        _ => &[],
    }
}

fn display_name(phase: &str) -> &str {
    match phase {
        "rvm" => "Motion features",
        "cradio" => "Image features",
        "moge" => "Depth features",
        "stage1" => "Stage 1 windows",
        "stage2" => "Stage 2 frames",
        "decode" | "decode_scenes" => "Decoding",
        "transnet" => "Scene detection",
        "sam_video" => "Tracking",
        other => other,
    }
}

#[derive(Debug, Clone, Default)]
pub struct JobProgress {
    counters: BTreeMap<String, (u64, u64)>,
    /// Staged (Low VRAM) attempts run per-frame foundation passes before Stage1.
    staged: bool,
    shown: f32,
    last_work: Option<String>,
}

impl JobProgress {
    pub fn observe(&mut self, phase: &str, current: u64, total: u64, message: &str) {
        match phase {
            "window_policy" => {
                // A new attempt starts: counters restart, the bar may restart too.
                self.counters.clear();
                self.shown = 0.0;
                self.last_work = None;
                self.staged = message.contains("staged");
            }
            "memory_retry" => {
                self.counters.clear();
                self.shown = 0.0;
                self.last_work = None;
            }
            _ if is_diagnostic(phase) || total == 0 => {}
            _ => {
                self.counters
                    .insert(phase.to_owned(), (current.min(total), total));
                self.last_work = Some(phase.to_owned());
                self.shown = self.shown.max(self.target());
            }
        }
    }

    fn target(&self) -> f32 {
        let mut expected: Vec<&str> = self.counters.keys().map(String::as_str).collect();
        for phase in self.counters.keys() {
            for implied in implied(phase) {
                let foundation = matches!(*implied, "rvm" | "cradio" | "moge");
                if (!foundation || self.staged) && !expected.contains(implied) {
                    expected.push(implied);
                }
            }
        }
        if self.staged {
            for phase in ["rvm", "cradio", "moge"] {
                if expected.contains(&"stage1") && !expected.contains(&phase) {
                    expected.push(phase);
                }
            }
        }
        let total_weight: f32 = expected.iter().map(|p| weight(p)).sum();
        if total_weight <= 0.0 {
            return 0.0;
        }
        let done: f32 = expected
            .iter()
            .map(|p| {
                self.counters.get(*p).map_or(0.0, |&(current, total)| {
                    weight(p) * current as f32 / total as f32
                })
            })
            .sum();
        (done / total_weight).clamp(0.0, 1.0)
    }

    /// Job fraction 0..=1; never decreases within one attempt.
    pub fn fraction(&self) -> f32 {
        self.shown
    }

    /// Steady text for the most recent work counter, e.g. "Stage 2 frames 5/16".
    pub fn stage_text(&self) -> String {
        match &self.last_work {
            Some(phase) => {
                let (current, total) = self.counters[phase];
                format!("{} {current}/{total}", display_name(phase))
            }
            None => "Starting".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resident() -> JobProgress {
        let mut p = JobProgress::default();
        p.observe(
            "window_policy",
            1,
            1,
            "Resident streaming: window 8/stride 4",
        );
        p
    }

    #[test]
    fn diagnostics_do_not_move_the_bar() {
        let mut p = resident();
        p.observe("stage1", 1, 3, "");
        let before = p.fraction();
        p.observe("runtime_phase", 0, 16, "stage1_window_begin");
        p.observe("runtime_phase", 7, 16, "moge_end");
        assert_eq!(p.fraction(), before);
        assert_eq!(p.stage_text(), "Stage 1 windows 1/3");
    }

    #[test]
    fn interleaved_resident_counters_never_go_backwards() {
        let mut p = resident();
        let mut last = 0.0;
        let events = [
            ("stage1", 1, 3),
            ("stage2", 1, 16),
            ("stage2", 2, 16),
            ("stage1", 2, 3),
            ("stage2", 6, 16),
            ("stage1", 3, 3),
            ("stage2", 16, 16),
        ];
        for (phase, current, total) in events {
            p.observe(phase, current, total, "");
            assert!(p.fraction() >= last, "{phase} {current}/{total}");
            last = p.fraction();
        }
        assert!((p.fraction() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn staged_foundation_does_not_fill_the_bar_alone() {
        let mut p = JobProgress::default();
        p.observe(
            "window_policy",
            1,
            1,
            "Low VRAM staged offload: window 8/stride 4",
        );
        p.observe("rvm", 16, 16, "");
        assert!(p.fraction() < 0.1, "{}", p.fraction());
        p.observe("cradio", 16, 16, "");
        p.observe("moge", 16, 16, "");
        let foundation = p.fraction();
        p.observe("stage1", 1, 3, "");
        assert!(p.fraction() > foundation);
        p.observe("stage1", 3, 3, "");
        p.observe("stage2", 16, 16, "");
        assert!((p.fraction() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_retry_restarts_the_attempt() {
        let mut p = resident();
        p.observe("stage1", 2, 3, "");
        assert!(p.fraction() > 0.0);
        p.observe("memory_retry", 1, 2, "host_capacity");
        assert_eq!(p.fraction(), 0.0);
        assert_eq!(p.stage_text(), "Starting");
    }

    #[test]
    fn preparation_phases_imply_their_main_phase() {
        let mut p = JobProgress::default();
        p.observe("decode_scenes", 1, 1, "");
        assert!(p.fraction() < 0.1);
        p.observe("transnet", 5, 10, "");
        assert!(p.fraction() > 0.4 && p.fraction() < 0.6);
    }
}
