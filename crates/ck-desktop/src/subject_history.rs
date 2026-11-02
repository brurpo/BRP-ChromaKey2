//! Undo/redo for subject (SAM) edits: points, boxes, layers and text prompts.
//!
//! Subject state is edited from many places (viewer clicks and drags, point text, the layer
//! list, the per-layer text prompt), so instead of recording at every call site the studio
//! calls [`SubjectHistory::observe`] once per frame: any difference from the last observed
//! state becomes one undo step. A change made while the pointer is held or a text field has
//! focus extends the current step, so a drag or a typed word undoes as a whole. Undo restores
//! only the subject state, never scenes, settings or garbage/despill masks.

use ck_desktop_core::{SamBox, SamLayer, SamPrompt};
use std::time::{Duration, Instant};

pub const DEFAULT_SUBJECT_HISTORY_LIMIT: usize = 64;
/// A continuing gesture (held pointer, focused text field) idle for longer starts a new step.
pub const COALESCE_WINDOW: Duration = Duration::from_millis(900);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SubjectState {
    pub prompts: Vec<SamPrompt>,
    pub boxes: Vec<SamBox>,
    pub layers: Vec<SamLayer>,
}

impl SubjectState {
    /// Layers whose points, boxes or settings differ between `self` and `other`.
    pub fn changed_layers(&self, other: &SubjectState) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .layers
            .iter()
            .chain(&other.layers)
            .map(|layer| layer.id)
            .chain(self.prompts.iter().chain(&other.prompts).map(|p| p.layer))
            .chain(self.boxes.iter().chain(&other.boxes).map(|b| b.layer))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids.retain(|&id| {
            let layer = |state: &SubjectState| state.layers.iter().find(|l| l.id == id).cloned();
            let prompts = |state: &SubjectState| {
                state
                    .prompts
                    .iter()
                    .filter(|p| p.layer == id)
                    .cloned()
                    .collect::<Vec<_>>()
            };
            let boxes = |state: &SubjectState| {
                state
                    .boxes
                    .iter()
                    .filter(|b| b.layer == id)
                    .cloned()
                    .collect::<Vec<_>>()
            };
            layer(self) != layer(other)
                || prompts(self) != prompts(other)
                || boxes(self) != boxes(other)
        });
        ids
    }
}

pub struct SubjectHistory {
    undo: Vec<SubjectState>,
    redo: Vec<SubjectState>,
    /// The state as of the last observation.
    baseline: Option<SubjectState>,
    /// When the current step last changed, while its gesture may still continue.
    open_step: Option<Instant>,
    limit: usize,
}

impl Default for SubjectHistory {
    fn default() -> Self {
        Self::new(DEFAULT_SUBJECT_HISTORY_LIMIT)
    }
}

impl SubjectHistory {
    pub fn new(limit: usize) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            baseline: None,
            open_step: None,
            limit,
        }
    }

    /// Records a step when `current` differs from the last observed state. `continuing` is
    /// true while a gesture can still be in progress (pointer held, text field focused).
    pub fn observe(&mut self, current: &SubjectState, continuing: bool, now: Instant) {
        let Some(baseline) = &self.baseline else {
            self.baseline = Some(current.clone());
            return;
        };
        if baseline == current {
            if !continuing {
                self.open_step = None;
            }
            return;
        }
        let extend = continuing
            && !self.undo.is_empty()
            && self
                .open_step
                .is_some_and(|at| now.saturating_duration_since(at) < COALESCE_WINDOW);
        if !extend && self.limit > 0 {
            if self.undo.len() == self.limit {
                self.undo.remove(0);
            }
            self.undo.push(baseline.clone());
        }
        self.redo.clear();
        self.baseline = Some(current.clone());
        self.open_step = continuing.then_some(now);
    }

    /// Accepts `current` without recording (changes made by scene edits, loads or an
    /// app-wide undo). Older steps would refer to another scene layout, so they are dropped.
    pub fn reset(&mut self, current: &SubjectState) {
        self.undo.clear();
        self.redo.clear();
        self.baseline = Some(current.clone());
        self.open_step = None;
    }

    pub fn undo(&mut self, current: &SubjectState) -> Option<SubjectState> {
        let previous = self.undo.pop()?;
        self.redo.push(current.clone());
        self.baseline = Some(previous.clone());
        self.open_step = None;
        Some(previous)
    }

    pub fn redo(&mut self, current: &SubjectState) -> Option<SubjectState> {
        let next = self.redo.pop()?;
        self.undo.push(current.clone());
        self.baseline = Some(next.clone());
        self.open_step = None;
        Some(next)
    }

    #[cfg(test)]
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(layer: u64, x: f32) -> SamPrompt {
        SamPrompt {
            frame: 0,
            x,
            y: 1.0,
            positive: true,
            layer,
            text: None,
        }
    }

    fn state(points: &[(u64, f32)]) -> SubjectState {
        SubjectState {
            prompts: points.iter().map(|&(l, x)| point(l, x)).collect(),
            ..SubjectState::default()
        }
    }

    #[test]
    fn each_separate_edit_is_one_step() {
        let t = Instant::now();
        let mut history = SubjectHistory::default();
        history.observe(&state(&[]), false, t);
        history.observe(&state(&[(1, 1.0)]), false, t);
        history.observe(&state(&[(1, 1.0), (1, 2.0)]), false, t);
        let current = state(&[(1, 1.0), (1, 2.0)]);
        let back = history.undo(&current).unwrap();
        assert_eq!(back, state(&[(1, 1.0)]));
        history.observe(&back, false, t);
        let back2 = history.undo(&back).unwrap();
        assert_eq!(back2, state(&[]));
        assert!(!history.can_undo());
        assert_eq!(history.redo(&back2).unwrap(), back);
    }

    #[test]
    fn a_drag_is_one_step_and_a_pause_starts_another() {
        let t = Instant::now();
        let mut history = SubjectHistory::default();
        history.observe(&state(&[(1, 0.0)]), false, t);
        for (i, x) in [1.0, 2.0, 3.0].into_iter().enumerate() {
            history.observe(
                &state(&[(1, x)]),
                true,
                t + Duration::from_millis(16 * i as u64),
            );
        }
        history.observe(&state(&[(1, 3.0)]), false, t + Duration::from_millis(60));
        let back = history.undo(&state(&[(1, 3.0)])).unwrap();
        assert_eq!(back, state(&[(1, 0.0)]));
        assert!(!history.can_undo());
    }

    #[test]
    fn reset_accepts_external_changes_without_a_step() {
        let t = Instant::now();
        let mut history = SubjectHistory::default();
        history.observe(&state(&[]), false, t);
        history.observe(&state(&[(1, 1.0)]), false, t);
        history.reset(&state(&[(2, 1.0)]));
        history.observe(&state(&[(2, 1.0)]), false, t);
        assert!(!history.can_undo());
    }

    #[test]
    fn changed_layers_lists_only_layers_that_differ() {
        let a = state(&[(1, 1.0), (2, 1.0)]);
        let b = state(&[(1, 1.0), (2, 5.0), (3, 1.0)]);
        assert_eq!(a.changed_layers(&b), vec![2, 3]);
    }
}
