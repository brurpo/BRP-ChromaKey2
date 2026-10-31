//! Bounded undo/redo history for the editable garbage/despill mask stacks.
//!
//! One snapshot is recorded before a user gesture mutates the stacks (layer
//! add/remove/visibility/mode/copy, bezier edits, keyframe moves). Continuous
//! drags record a single snapshot at gesture start. The history is per Session
//! and memory is bounded by a fixed entry count.

use ck_desktop_core::MaskStacks;
use std::collections::VecDeque;

/// Default number of retained undo steps.
pub const DEFAULT_MASK_HISTORY_LIMIT: usize = 24;

#[derive(Debug)]
pub struct MaskHistory {
    undo: VecDeque<MaskStacks>,
    redo: Vec<MaskStacks>,
    limit: usize,
}

impl Default for MaskHistory {
    fn default() -> Self {
        Self::new(DEFAULT_MASK_HISTORY_LIMIT)
    }
}

impl MaskHistory {
    pub fn new(limit: usize) -> Self {
        Self {
            undo: VecDeque::new(),
            redo: Vec::new(),
            limit,
        }
    }

    /// Snapshot the stacks before a mutation; any redo branch is discarded.
    pub fn record(&mut self, stacks: &MaskStacks) {
        if self.limit == 0 {
            return;
        }
        if self.undo.len() == self.limit {
            self.undo.pop_front();
        }
        self.undo.push_back(stacks.clone());
        self.redo.clear();
    }

    /// Drop the most recent snapshot (used when an edit turned out to be a
    /// no-op, so undo/redo stays free of empty steps).
    #[cfg(test)]
    pub fn discard_last(&mut self) {
        self.undo.pop_back();
    }

    /// Restore the previous snapshot, pushing `current` onto the redo stack.
    pub fn undo(&mut self, current: &MaskStacks) -> Option<MaskStacks> {
        let previous = self.undo.pop_back()?;
        self.redo.push(current.clone());
        Some(previous)
    }

    /// Reapply the next redo snapshot, pushing `current` back onto undo.
    pub fn redo(&mut self, current: &MaskStacks) -> Option<MaskStacks> {
        let next = self.redo.pop()?;
        if self.undo.len() == self.limit && self.limit > 0 {
            self.undo.pop_front();
        }
        self.undo.push_back(current.clone());
        Some(next)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    #[cfg(test)]
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ck_desktop_core::MaskStacks;

    fn stacks(width: u32, height: u32) -> MaskStacks {
        MaskStacks::new(width, height)
    }

    #[test]
    fn undo_and_redo_round_trip_a_snapshot() {
        let mut history = MaskHistory::new(8);
        let before = stacks(4, 4);
        history.record(&before);
        assert!(history.can_undo());
        let current = stacks(6, 6);
        let restored = history.undo(&current).expect("undo");
        assert_eq!(restored.raster_size(), (4, 4));
        assert!(history.can_redo());
        let reapplied = history.redo(&restored).expect("redo");
        assert_eq!(reapplied.raster_size(), (6, 6));
        assert!(!history.can_redo());
    }

    #[test]
    fn a_new_edit_clears_the_redo_branch() {
        let mut history = MaskHistory::new(8);
        history.record(&stacks(4, 4));
        let current = stacks(6, 6);
        history.undo(&current);
        assert!(history.can_redo());
        history.record(&current);
        assert!(!history.can_redo());
    }

    #[test]
    fn history_is_bounded_and_discard_removes_a_noop_snapshot() {
        let mut history = MaskHistory::new(2);
        history.record(&stacks(1, 1));
        history.record(&stacks(2, 2));
        history.record(&stacks(3, 3));
        assert_eq!(history.undo_len(), 2);
        history.discard_last();
        assert_eq!(history.undo_len(), 1);
    }

    #[test]
    fn empty_history_returns_none() {
        let mut history = MaskHistory::default();
        assert!(!history.can_undo());
        assert!(!history.can_redo());
        assert!(history.undo(&stacks(2, 2)).is_none());
        assert!(history.redo(&stacks(2, 2)).is_none());
    }
}
