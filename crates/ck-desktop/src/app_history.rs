//! App-wide undo/redo snapshots with pointer-context selection.
//!
//! Mask edits live in the per-session `MaskHistory`; this manager covers the
//! rest of the editable document state (scenes, playback range, keyer/despill
//! settings, embed-sources flag) and lets the Ctrl+Z target follow the pointer:
//! over the viewer the mask history is used, over the timeline the timeline
//! stack, over the inspector the keyer stack, and elsewhere the newest edit.

use ck_desktop_core::{FrameRange, Project};
use std::time::{Duration, Instant};

pub const DEFAULT_APP_HISTORY_LIMIT: usize = 32;
/// Edits with the same context/label inside this window are one undo step.
pub const COALESCE_WINDOW: Duration = Duration::from_millis(900);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppContext {
    /// Newest edit anywhere (used when the pointer is over neutral chrome).
    Global,
    /// Mask editing in the viewer / mask tabs (handled by `MaskHistory`).
    Mask,
    /// Timeline and scene edits.
    Timeline,
    /// Keyer/despill/Hann settings.
    Keyer,
    /// Media/project options such as embed sources.
    Media,
}

#[derive(Clone, Debug)]
pub struct AppSnapshot {
    pub session_key: String,
    pub project: Project,
    pub range: FrameRange,
    pub playhead: u64,
    pub embed_sources: bool,
}

#[derive(Clone, Debug)]
struct Entry {
    context: AppContext,
    label: &'static str,
    snapshot: AppSnapshot,
}

pub struct AppHistory {
    undo: Vec<Entry>,
    redo: Vec<Entry>,
    limit: usize,
    last: Option<(AppContext, &'static str, Instant)>,
}

impl Default for AppHistory {
    fn default() -> Self {
        Self::new(DEFAULT_APP_HISTORY_LIMIT)
    }
}

impl AppHistory {
    pub fn new(limit: usize) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            limit,
            last: None,
        }
    }

    pub fn record(&mut self, context: AppContext, label: &'static str, snapshot: AppSnapshot) {
        self.record_at(context, label, snapshot, Instant::now());
    }

    /// Coalesce rapid same-label edits (slider drags, resize handles) into one
    /// undo step while keeping the snapshot from before the gesture.
    pub fn record_at(
        &mut self,
        context: AppContext,
        label: &'static str,
        snapshot: AppSnapshot,
        now: Instant,
    ) {
        if self.limit == 0 {
            return;
        }
        if let Some((last_context, last_label, at)) = self.last
            && last_context == context
            && last_label == label
            && now.saturating_duration_since(at) < COALESCE_WINDOW
        {
            self.last = Some((context, label, now));
            self.redo.clear();
            return;
        }
        if self.undo.len() == self.limit {
            self.undo.remove(0);
        }
        self.undo.push(Entry {
            context,
            label,
            snapshot,
        });
        self.redo.clear();
        self.last = Some((context, label, now));
    }

    pub fn undo_for(&mut self, context: AppContext, current: AppSnapshot) -> Option<AppSnapshot> {
        let index = if context == AppContext::Global {
            self.undo.len().checked_sub(1)?
        } else {
            self.undo
                .iter()
                .rposition(|entry| entry.context == context)?
        };
        let entry = self.undo.remove(index);
        self.redo.push(Entry {
            context: entry.context,
            label: entry.label,
            snapshot: current,
        });
        Some(entry.snapshot)
    }

    pub fn redo_for(&mut self, context: AppContext, current: AppSnapshot) -> Option<AppSnapshot> {
        let index = if context == AppContext::Global {
            self.redo.len().checked_sub(1)?
        } else {
            self.redo
                .iter()
                .rposition(|entry| entry.context == context)?
        };
        let entry = self.redo.remove(index);
        if self.undo.len() == self.limit {
            self.undo.remove(0);
        }
        self.undo.push(Entry {
            context: entry.context,
            label: entry.label,
            snapshot: current,
        });
        Some(entry.snapshot)
    }

    #[cfg(test)]
    pub fn can_undo_for(&self, context: AppContext) -> bool {
        if context == AppContext::Global {
            !self.undo.is_empty()
        } else {
            self.undo.iter().any(|entry| entry.context == context)
        }
    }

    #[cfg(test)]
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ck_desktop_core::Project;
    use std::time::{Duration, Instant};

    fn snapshot(session: &str) -> AppSnapshot {
        AppSnapshot {
            session_key: session.into(),
            project: Project::default(),
            range: FrameRange {
                start: 0,
                end_exclusive: 10,
            },
            playhead: 0,
            embed_sources: false,
        }
    }

    #[test]
    fn undo_and_redo_round_trip() {
        let mut history = AppHistory::new(4);
        let now = Instant::now();
        history.record_at(AppContext::Timeline, "timeline edit", snapshot("a"), now);
        let current = snapshot("b");
        let restored = history
            .undo_for(AppContext::Timeline, current.clone())
            .expect("undo");
        assert_eq!(restored.session_key, "a");
        let redone = history
            .redo_for(AppContext::Timeline, restored)
            .expect("redo");
        assert_eq!(redone.session_key, "b");
    }

    #[test]
    fn context_specific_undo_picks_the_matching_edit() {
        let mut history = AppHistory::new(8);
        let now = Instant::now();
        history.record_at(AppContext::Timeline, "timeline", snapshot("t"), now);
        history.record_at(
            AppContext::Keyer,
            "settings",
            snapshot("k"),
            now + Duration::from_secs(2),
        );
        let restored = history
            .undo_for(AppContext::Timeline, snapshot("current"))
            .expect("timeline undo");
        assert_eq!(restored.session_key, "t");
        let restored = history
            .undo_for(AppContext::Keyer, snapshot("current2"))
            .expect("keyer undo");
        assert_eq!(restored.session_key, "k");
    }

    #[test]
    fn rapid_same_label_edits_coalesce_into_one_step() {
        let mut history = AppHistory::new(8);
        let now = Instant::now();
        history.record_at(AppContext::Keyer, "slider", snapshot("before"), now);
        history.record_at(
            AppContext::Keyer,
            "slider",
            snapshot("middle"),
            now + Duration::from_millis(100),
        );
        assert_eq!(history.undo_len(), 1);
        let restored = history
            .undo_for(AppContext::Keyer, snapshot("after"))
            .expect("undo");
        assert_eq!(restored.session_key, "before");
    }

    #[test]
    fn history_is_bounded_and_context_readiness_is_reported() {
        let mut history = AppHistory::new(2);
        let now = Instant::now();
        for index in 0..3 {
            history.record_at(
                AppContext::Media,
                "media",
                snapshot(&format!("s{index}")),
                now + Duration::from_secs(2 * index as u64),
            );
        }
        assert_eq!(history.undo_len(), 2);
        assert!(history.can_undo_for(AppContext::Media));
        assert!(!history.can_undo_for(AppContext::Timeline));
    }
}
