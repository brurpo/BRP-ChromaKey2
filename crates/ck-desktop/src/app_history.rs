//! App-wide undo/redo of the editable document (scenes, In/Out range, scene overrides,
//! keyer/despill settings, media options).
//!
//! The studio calls [`AppHistory::observe`] once per frame with the current [`DocSnapshot`], so
//! every edit is recorded whichever control made it. Each step remembers which parts of the
//! document it changed ([`group`]) and undo restores only those parts, so undoing a keyer
//! setting never reverts a later timeline edit. Ctrl+Z follows the pointer: the timeline undoes
//! timeline edits, the inspector keyer edits, and so on. Subject points and garbage/despill
//! masks have their own histories (`subject_history`, `mask_history`).

use ck_desktop_core::{
    CkSettings, DespillSettings, FrameRange, HintQuality, SamBox, SamLayer, SamPrompt,
    SceneOverrides, SceneRange,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub const DEFAULT_APP_HISTORY_LIMIT: usize = 64;
/// A continuing gesture (held pointer, focused text field) idle for longer starts a new step.
pub const COALESCE_WINDOW: Duration = Duration::from_millis(900);

/// Parts of the document a step can change.
pub mod group {
    /// Scene boundaries and enables. Subject layers move with scenes, so they are restored too.
    pub const SCENES: u8 = 1;
    /// In/Out playback range.
    pub const RANGE: u8 = 2;
    /// Per-scene overrides (despill).
    pub const OVERRIDES: u8 = 4;
    /// Keyer, despill and hint settings, output folder.
    pub const KEYER: u8 = 8;
    /// Media options (embed sources).
    pub const MEDIA: u8 = 16;
    pub const ALL: u8 = SCENES | RANGE | OVERRIDES | KEYER | MEDIA;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AppContext {
    /// Newest document edit anywhere (pointer over neutral chrome).
    Global,
    /// Garbage/despill mask editing (handled by `MaskHistory`).
    Mask,
    /// Subject points/boxes/layers on the Subject tab (handled by `SubjectHistory`).
    Subject,
    /// Timeline and scene edits.
    Timeline,
    /// Keyer/despill/Hann settings and scene overrides in the inspector.
    Keyer,
    /// Media/project options such as embed sources and hint footage.
    Media,
}

impl AppContext {
    /// Document parts Ctrl+Z may undo in this context.
    pub fn groups(self) -> u8 {
        match self {
            AppContext::Timeline => group::SCENES | group::RANGE | group::OVERRIDES,
            AppContext::Keyer => group::KEYER | group::OVERRIDES,
            AppContext::Media => group::MEDIA | group::KEYER,
            AppContext::Global => group::ALL,
            AppContext::Mask | AppContext::Subject => 0,
        }
    }
}

/// The editable document state covered by this history.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DocSnapshot {
    pub scenes: Vec<SceneRange>,
    pub scene_overrides: BTreeMap<u32, SceneOverrides>,
    pub ck_settings: CkSettings,
    pub despill_settings: DespillSettings,
    pub hint_quality: HintQuality,
    pub output_dir: PathBuf,
    pub range: FrameRange,
    pub embed_sources: bool,
    /// Carried along so undoing a scene split/merge also restores the subject layers it moved.
    pub prompts: Vec<SamPrompt>,
    pub sam_boxes: Vec<SamBox>,
    pub sam_layers: Vec<SamLayer>,
}

impl DocSnapshot {
    /// The groups whose fields differ between `self` and `other`.
    pub fn changed_groups(&self, other: &DocSnapshot) -> u8 {
        let mut groups = 0;
        if self.scenes != other.scenes {
            groups |= group::SCENES;
        }
        if self.range != other.range {
            groups |= group::RANGE;
        }
        if self.scene_overrides != other.scene_overrides {
            groups |= group::OVERRIDES;
        }
        if self.ck_settings != other.ck_settings
            || self.despill_settings != other.despill_settings
            || self.hint_quality != other.hint_quality
            || self.output_dir != other.output_dir
        {
            groups |= group::KEYER;
        }
        if self.embed_sources != other.embed_sources {
            groups |= group::MEDIA;
        }
        groups
    }

    /// `self` with the fields of `groups` taken from `from`.
    fn merged(mut self, from: &DocSnapshot, groups: u8) -> DocSnapshot {
        if groups & group::SCENES != 0 {
            self.scenes = from.scenes.clone();
            self.prompts = from.prompts.clone();
            self.sam_boxes = from.sam_boxes.clone();
            self.sam_layers = from.sam_layers.clone();
        }
        if groups & group::RANGE != 0 {
            self.range = from.range;
        }
        if groups & group::OVERRIDES != 0 {
            self.scene_overrides = from.scene_overrides.clone();
        }
        if groups & group::KEYER != 0 {
            self.ck_settings = from.ck_settings.clone();
            self.despill_settings = from.despill_settings;
            self.hint_quality = from.hint_quality;
            self.output_dir = from.output_dir.clone();
        }
        if groups & group::MEDIA != 0 {
            self.embed_sources = from.embed_sources;
        }
        self
    }
}

#[derive(Clone, Debug)]
struct Entry {
    groups: u8,
    /// The document before this step (only the fields of `groups` are restored).
    before: DocSnapshot,
}

pub struct AppHistory {
    undo: Vec<Entry>,
    redo: Vec<Entry>,
    baseline: Option<DocSnapshot>,
    /// Groups and time of the step a continuing gesture may still extend.
    open_step: Option<(u8, Instant)>,
    limit: usize,
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
            baseline: None,
            open_step: None,
            limit,
        }
    }

    /// Records a step when the document differs from the last observation. `continuing` is
    /// true while a gesture can still be in progress (slider drag, text field focus).
    pub fn observe(&mut self, current: &DocSnapshot, continuing: bool, now: Instant) {
        let Some(baseline) = self.baseline.take() else {
            self.baseline = Some(current.clone());
            return;
        };
        let groups = baseline.changed_groups(current);
        if groups == 0 {
            // Subject fields may still have changed; keep them current for scene steps.
            self.baseline = Some(current.clone());
            if !continuing {
                self.open_step = None;
            }
            return;
        }
        let extend = continuing
            && self.open_step.is_some_and(|(open, at)| {
                open & groups == groups && now.saturating_duration_since(at) < COALESCE_WINDOW
            })
            && !self.undo.is_empty();
        if !extend && self.limit > 0 {
            if self.undo.len() == self.limit {
                self.undo.remove(0);
            }
            self.undo.push(Entry {
                groups,
                before: baseline,
            });
            self.open_step = continuing.then_some((groups, now));
        } else if extend {
            self.open_step = Some((self.open_step.map_or(groups, |(g, _)| g), now));
        }
        self.redo.clear();
        self.baseline = Some(current.clone());
    }

    /// Undoes the newest step this context covers. Newer steps that changed the same parts are
    /// undone with it, so the document never mixes states. Returns the restored document and
    /// the groups it changed.
    pub fn undo(
        &mut self,
        context: AppContext,
        current: &DocSnapshot,
    ) -> Option<(DocSnapshot, u8)> {
        let (state, groups) = transfer(&mut self.undo, &mut self.redo, context.groups(), current)?;
        self.baseline = Some(state.clone());
        self.open_step = None;
        Some((state, groups))
    }

    pub fn redo(
        &mut self,
        context: AppContext,
        current: &DocSnapshot,
    ) -> Option<(DocSnapshot, u8)> {
        let (state, groups) = transfer(&mut self.redo, &mut self.undo, context.groups(), current)?;
        self.baseline = Some(state.clone());
        self.open_step = None;
        Some((state, groups))
    }

    #[cfg(test)]
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }
}

/// Moves the newest entry of `from` matching `accepted` (plus every newer entry overlapping
/// it) to `to`, applying their snapshots to `current`.
fn transfer(
    from: &mut Vec<Entry>,
    to: &mut Vec<Entry>,
    accepted: u8,
    current: &DocSnapshot,
) -> Option<(DocSnapshot, u8)> {
    let first = from
        .iter()
        .rposition(|entry| entry.groups & accepted != 0)?;
    let mut union = from[first].groups;
    let mut picked = vec![first];
    loop {
        let before = picked.len();
        for (index, entry) in from.iter().enumerate().skip(first + 1) {
            if !picked.contains(&index) && entry.groups & union != 0 {
                union |= entry.groups;
                picked.push(index);
            }
        }
        if picked.len() == before {
            break;
        }
    }
    picked.sort_unstable();
    let mut state = current.clone();
    for &index in picked.iter().rev() {
        let entry = from.remove(index);
        to.push(Entry {
            groups: entry.groups,
            before: state.clone(),
        });
        state = state.merged(&entry.before, entry.groups);
    }
    Some((state, union))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(range_end: u64, despill: f32, scenes: usize) -> DocSnapshot {
        DocSnapshot {
            range: FrameRange {
                start: 0,
                end_exclusive: range_end,
            },
            despill_settings: DespillSettings { amount: despill },
            scenes: (0..scenes as u64)
                .map(|i| SceneRange {
                    index: i as u32,
                    start: i * 10,
                    end_exclusive: i * 10 + 10,
                    enabled: true,
                })
                .collect(),
            ..DocSnapshot::default()
        }
    }

    #[test]
    fn undo_restores_only_the_edited_part() {
        let t = Instant::now();
        let mut history = AppHistory::default();
        history.observe(&doc(50, 0.0, 1), false, t);
        history.observe(&doc(50, 0.5, 1), false, t); // keyer edit
        history.observe(&doc(40, 0.5, 1), false, t); // timeline edit
        let current = doc(40, 0.5, 1);
        let (restored, groups) = history.undo(AppContext::Keyer, &current).unwrap();
        assert_eq!(groups, group::KEYER);
        // The despill change is undone, the later range edit stays.
        assert_eq!(restored, doc(40, 0.0, 1));
        let (redone, _) = history.redo(AppContext::Keyer, &restored).unwrap();
        assert_eq!(redone, current);
    }

    #[test]
    fn a_slider_drag_is_one_step() {
        let t = Instant::now();
        let mut history = AppHistory::default();
        history.observe(&doc(50, 0.0, 1), false, t);
        for (i, value) in [0.1, 0.2, 0.3].into_iter().enumerate() {
            history.observe(
                &doc(50, value, 1),
                true,
                t + Duration::from_millis(16 * i as u64),
            );
        }
        history.observe(&doc(50, 0.3, 1), false, t + Duration::from_millis(80));
        assert_eq!(history.undo_len(), 1);
        let (restored, _) = history.undo(AppContext::Global, &doc(50, 0.3, 1)).unwrap();
        assert_eq!(restored.despill_settings.amount, 0.0);
    }

    #[test]
    fn contexts_without_matching_steps_undo_nothing() {
        let t = Instant::now();
        let mut history = AppHistory::default();
        history.observe(&doc(50, 0.0, 1), false, t);
        history.observe(&doc(50, 0.0, 2), false, t); // split
        assert!(history.undo(AppContext::Keyer, &doc(50, 0.0, 2)).is_none());
        assert!(
            history
                .undo(AppContext::Subject, &doc(50, 0.0, 2))
                .is_none()
        );
        assert!(
            history
                .undo(AppContext::Timeline, &doc(50, 0.0, 2))
                .is_some()
        );
    }

    #[test]
    fn undoing_a_split_restores_the_subject_layers_it_moved() {
        let t = Instant::now();
        let mut history = AppHistory::default();
        let mut before = doc(50, 0.0, 1);
        before.sam_layers = vec![SamLayer {
            scene: Some(0),
            ..SamLayer::subject()
        }];
        history.observe(&before, false, t);
        let mut after = doc(50, 0.0, 2);
        after.sam_layers = vec![SamLayer {
            scene: Some(1),
            ..SamLayer::subject()
        }];
        history.observe(&after, false, t);
        let (restored, groups) = history.undo(AppContext::Timeline, &after).unwrap();
        assert_eq!(groups, group::SCENES);
        assert_eq!(restored, before);
    }
}
