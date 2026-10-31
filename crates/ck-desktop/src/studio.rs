use crate::asset_setup::AssetSetup;
use crate::hann_advice::{HannAdviceKind, custom_hann_advice};
use crate::icons::{Icon, Icons, checkbox_widget};
use crate::mask_editor::MaskUi;
use crate::preview::{self, Background, PreviewService, ViewMode};
use crate::project_store::{
    self, ChromaProject, Document, FrameFiles, HintFootage, MaskBatch, TemporaryWorkspace,
};
use crate::range_export::{ExportEvent, ExportFormat, ExportRequest, ExportService};
use crate::viewport::{ViewerZoom, display_path};
use crate::{theme, timeline};
use ck_core::mask_layers::BezierShape;
use ck_desktop_core::{
    AssetGroup, CkSettings, CkWindowMode, FrameRange, HintInput, JobArtifacts, JobEvent, JobKind,
    JobRequest, MaskKind, MaskLayer, MaskMode, MaskStacks, MediaInfo, Project, Provider,
    RuntimePaths, SamModel, SamPrompt, SceneRange, Worker,
};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::PathBuf,
    sync::{Arc, mpsc},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const ACCENT: Color32 = theme::ACCENT;
const GREEN: Color32 = theme::GOOD;
const MUTED: Color32 = theme::TEXT_DIM;
const ORANGE: Color32 = theme::WARN;
const TIMELINE_PANEL_ID: &str = "timeline";

/// Key that identifies the fully composited preview frame, including the
/// mask/despill state, so the RAM result cache never reuses a frame rendered
/// with a different despill amount or mask revision.
fn preview_composition_signature(
    session: &Session,
    mode: crate::preview::ViewMode,
    overlay: bool,
    background: crate::preview::Background,
) -> String {
    let overrides = session
        .project
        .scene_overrides
        .iter()
        .filter_map(|(index, overrides)| {
            let entry = overrides.despill?;
            entry
                .enabled
                .then(|| format!("{index}:{:08x}", entry.amount.to_bits()))
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{}:{:?}:{}:{}:{:?}:{:?}:{:08x}:{}",
        session.id,
        mode,
        session.revision,
        overlay,
        session.media().source_identity,
        background,
        session.project.despill_settings.amount.to_bits(),
        overrides
    )
}

fn preview_composition_key(
    session: &Session,
    frame: u64,
    mode: crate::preview::ViewMode,
    overlay: bool,
    background: crate::preview::Background,
) -> String {
    // Only Result compositions apply masks; the fingerprint keeps entries for frames a
    // mask edit did not touch valid instead of invalidating the whole clip.
    let masks = if mode == crate::preview::ViewMode::Composite {
        crate::mask_composite::mask_fingerprint(&session.mask_stacks, frame as i64).unwrap_or(0)
    } else {
        0
    };
    format!(
        "{}:{}:{:016x}",
        preview_composition_signature(session, mode, overlay, background),
        frame,
        masks
    )
}

/// `Session::sam_layer` when the playhead scene has no SAM layer yet.
const NO_LAYER: u64 = u64::MAX;

struct Session {
    id: u64,
    project: Project,
    playhead: u64,
    range: FrameRange,
    selected_scene: usize,
    frames: BTreeMap<u64, FrameFiles>,
    timeline_view: timeline::TimelineView,
    masks: Vec<MaskBatch>,
    revision: u64,
    key: String,
    hint: Option<HintFootage>,
    /// Editable runtime garbage/despill stacks (saved back into the project).
    mask_stacks: MaskStacks,
    /// Window-only mask editor state (selection memory, timeline visibility).
    mask_ui: MaskUi,
    /// Bounded undo/redo snapshots of the mask stacks.
    mask_history: crate::mask_history::MaskHistory,
    /// Bumped by every mask edit. Kept apart from `revision` so a mask edit only
    /// invalidates the frames whose masks changed (see `mask_fingerprint`).
    mask_revision: u64,
    /// Read-only copy of `mask_stacks` at `mask_revision`, shared with the preview worker.
    mask_snapshot: Option<(u64, std::sync::Arc<MaskStacks>)>,
    /// SAM layer that viewer clicks and Segment/Propagate act on.
    sam_layer: u64,
    /// Combined Remove-layer overlays per frame, keyed by the layer files they were built from.
    removal_masks: BTreeMap<u64, (Vec<PathBuf>, PathBuf)>,
}

impl Session {
    fn new(mut project: Project) -> Self {
        static NEXT_SESSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        project.ensure_sam_layers();
        let first_scene = project.scene_of_frame(0);
        let sam_layer = project
            .layers_in_scene(first_scene)
            .first()
            .map_or(NO_LAYER, |layer| layer.id);
        let count = project.media.as_ref().map_or(1, |media| media.frame_count);
        let mask_stacks = project.mask_stacks();
        Self {
            sam_layer,
            removal_masks: BTreeMap::new(),
            id: NEXT_SESSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            project,
            playhead: 0,
            range: FrameRange {
                start: 0,
                end_exclusive: count,
            },
            selected_scene: 0,
            frames: BTreeMap::new(),
            timeline_view: timeline::TimelineView::default(),
            masks: vec![],
            revision: 0,
            key: project_store::unique_key(),
            hint: None,
            mask_stacks,
            mask_ui: MaskUi::default(),
            mask_history: crate::mask_history::MaskHistory::default(),
            mask_revision: 0,
            mask_snapshot: None,
        }
    }
    fn media(&self) -> &MediaInfo {
        self.project
            .media
            .as_ref()
            .expect("only imported projects become sessions")
    }
    fn ck_job(&self, range: FrameRange, provider: Provider, hints: HintInput) -> JobKind {
        JobKind::CorridorKey {
            range,
            provider,
            hints,
            settings: self.project.ck_settings.clone(),
        }
    }
    fn seek(&mut self, frame: u64) {
        self.playhead = frame.min(self.media().frame_count.saturating_sub(1));
        if let Some(index) = timeline::scene_index_at_frame(&self.project.scenes, self.playhead) {
            self.selected_scene = index;
        }
        self.sync_mask_selection();
    }
    fn current_scene_range(&self) -> Result<FrameRange, timeline::SceneRunError> {
        timeline::enabled_scene_range_at_frame(&self.project.scenes, self.playhead)
    }
    fn select_scene(&mut self, index: usize) {
        if let Some(scene) = self.project.scenes.get(index) {
            self.range = FrameRange {
                start: scene.start,
                end_exclusive: scene.end_exclusive,
            };
            self.playhead = scene.start;
            self.selected_scene = index;
        }
        self.sync_mask_selection();
    }
    /// Replace scenes from a detection job without moving the playback range.
    ///
    /// Detection re-segments the clip, but the user's current In/Out range and
    /// playhead stay exactly where they were; only the selected-scene highlight
    /// is recomputed from the playhead.
    fn apply_detected_scenes(&mut self, scenes: Vec<SceneRange>) {
        let old = self
            .project
            .scenes
            .iter()
            .map(|scene| (scene.start, scene.end_exclusive))
            .collect::<Vec<_>>();
        let new = scenes
            .iter()
            .map(|scene| (scene.start, scene.end_exclusive))
            .collect::<Vec<_>>();
        if old != new {
            self.frames.clear();
            self.masks.clear();
        }
        self.project.scenes = scenes;
        self.selected_scene =
            timeline::scene_index_at_frame(&self.project.scenes, self.playhead).unwrap_or(0);
        self.rehome_sam_layers();
    }
    /// Scene under the playhead (falls back to the selected scene).
    fn mask_scene_index(&self) -> usize {
        timeline::scene_index_at_frame(&self.project.scenes, self.playhead)
            .unwrap_or(self.selected_scene)
    }

    /// Keep the selected garbage/despill layer in sync with the scene under the
    /// playhead, restoring v1's per-scene layer memory.
    fn sync_mask_selection(&mut self) {
        let scene = self.mask_scene_index() as u32;
        for kind in [MaskKind::Garbage, MaskKind::Despill] {
            let remembered = self.mask_ui.remembered(kind, scene);
            let stack = self.mask_stacks.stack_mut(kind);
            let valid = remembered.filter(|id| {
                stack
                    .layers
                    .iter()
                    .any(|layer| layer.layer_id == *id && layer.scene_index == scene)
            });
            let selected = valid.or_else(|| {
                stack
                    .layers
                    .iter()
                    .find(|layer| layer.scene_index == scene)
                    .map(|layer| layer.layer_id)
            });
            stack.select(selected);
            let selected = stack.selected;
            self.mask_ui.remember(kind, scene, selected);
        }
    }

    /// Snapshot the stacks before a user edit so it can be undone.
    fn record_mask_edit(&mut self) {
        self.mask_history.record(&self.mask_stacks);
    }

    #[cfg(test)]
    fn discard_mask_edit_record(&mut self) {
        self.mask_history.discard_last();
    }

    fn undo_mask_edit(&mut self) -> bool {
        let Some(previous) = self.mask_history.undo(&self.mask_stacks) else {
            return false;
        };
        self.mask_stacks = previous;
        self.mask_revision = self.mask_revision.wrapping_add(1);
        self.sync_mask_selection();
        true
    }

    fn redo_mask_edit(&mut self) -> bool {
        let Some(next) = self.mask_history.redo(&self.mask_stacks) else {
            return false;
        };
        self.mask_stacks = next;
        self.mask_revision = self.mask_revision.wrapping_add(1);
        self.sync_mask_selection();
        true
    }

    /// Create a new layer scoped to the current scene and select it.
    fn add_mask_layer(&mut self, kind: MaskKind) -> Option<u32> {
        self.record_mask_edit();
        let scene = self.mask_scene_index();
        let (start, end) = self
            .project
            .scene_inclusive_range(scene as u32)
            .unwrap_or((-1, -1));
        let stack = self.mask_stacks.stack_mut(kind);
        let layer = stack.add(scene as u32, None);
        layer.set_scene_range(start, end);
        let id = layer.layer_id;
        stack.select(Some(id));
        self.mask_ui.remember(kind, scene as u32, Some(id));
        self.mask_revision = self.mask_revision.wrapping_add(1);
        Some(id)
    }

    /// Remove one layer and select the next layer of the current scene.
    fn remove_mask_layer(&mut self, kind: MaskKind, layer_id: u32) -> bool {
        if !self
            .mask_stacks
            .stack(kind)
            .layers
            .iter()
            .any(|layer| layer.layer_id == layer_id)
        {
            return false;
        }
        self.record_mask_edit();
        let scene = self.mask_scene_index() as u32;
        let stack = self.mask_stacks.stack_mut(kind);
        stack.remove(layer_id);
        let next = stack
            .layers
            .iter()
            .find(|layer| layer.scene_index == scene)
            .map(|layer| layer.layer_id);
        stack.select(next);
        let next = stack.selected;
        self.mask_ui.remember(kind, scene, next);
        self.mask_revision = self.mask_revision.wrapping_add(1);
        true
    }

    fn set_mask_layer_visible(&mut self, kind: MaskKind, layer_id: u32, visible: bool) {
        let changed = self
            .mask_stacks
            .stack(kind)
            .layers
            .iter()
            .any(|layer| layer.layer_id == layer_id && layer.visible != visible);
        if !changed {
            return;
        }
        self.record_mask_edit();
        if let Some(layer) = self
            .mask_stacks
            .stack_mut(kind)
            .layers
            .iter_mut()
            .find(|layer| layer.layer_id == layer_id)
        {
            layer.visible = visible;
            self.mask_revision = self.mask_revision.wrapping_add(1);
        }
    }

    fn set_mask_layer_mode(&mut self, kind: MaskKind, layer_id: u32, mode: MaskMode) -> bool {
        if !mode.is_valid_for(kind) {
            return false;
        }
        let changed = self
            .mask_stacks
            .stack(kind)
            .layers
            .iter()
            .any(|layer| layer.layer_id == layer_id && layer.mode != mode);
        if !changed {
            return self
                .mask_stacks
                .stack(kind)
                .layers
                .iter()
                .any(|layer| layer.layer_id == layer_id);
        }
        self.record_mask_edit();
        if let Some(layer) = self
            .mask_stacks
            .stack_mut(kind)
            .layers
            .iter_mut()
            .find(|layer| layer.layer_id == layer_id)
        {
            layer.mode = mode;
            self.mask_revision = self.mask_revision.wrapping_add(1);
            return true;
        }
        false
    }

    /// Feather radius in source pixels. `record` adds one undo step (once per slider drag).
    fn set_mask_layer_softness(
        &mut self,
        kind: MaskKind,
        layer_id: u32,
        softness: u32,
        record: bool,
    ) -> bool {
        let current = self
            .mask_stacks
            .stack(kind)
            .layers
            .iter()
            .find(|layer| layer.layer_id == layer_id)
            .map(|layer| layer.softness);
        if current.is_none_or(|value| value == softness) {
            return false;
        }
        if record {
            self.record_mask_edit();
        }
        if let Some(layer) = self
            .mask_stacks
            .stack_mut(kind)
            .layers
            .iter_mut()
            .find(|layer| layer.layer_id == layer_id)
        {
            layer.softness = softness;
            self.mask_revision = self.mask_revision.wrapping_add(1);
        }
        true
    }

    fn select_mask_layer(&mut self, kind: MaskKind, layer_id: u32) {
        let scene = self.mask_scene_index() as u32;
        let stack = self.mask_stacks.stack_mut(kind);
        if stack
            .layers
            .iter()
            .any(|layer| layer.layer_id == layer_id && layer.scene_index == scene)
        {
            stack.select(Some(layer_id));
            self.mask_ui.remember(kind, scene, Some(layer_id));
        }
    }

    /// Copy every layer of the current scene in one bank onto `target_scene`.
    #[cfg(test)]
    fn copy_mask_layers_to_scene(&mut self, kind: MaskKind, target_scene: usize) -> usize {
        let source = self.mask_scene_index();
        if source == target_scene {
            return 0;
        }
        let Some((start, end)) = self.project.scene_inclusive_range(target_scene as u32) else {
            return 0;
        };
        self.record_mask_edit();
        let copied = self
            .mask_stacks
            .copy_layers_from_scene(kind, source as u32, target_scene as u32, start, end)
            .len();
        if copied > 0 {
            self.mask_revision = self.mask_revision.wrapping_add(1);
        } else {
            self.discard_mask_edit_record();
        }
        copied
    }

    /// Rasterise the current frame's garbage/despill masks for the preview.
    fn mask_application(
        &self,
        frame: u64,
    ) -> Option<std::sync::Arc<crate::mask_composite::MaskApplication>> {
        let media = self.media();
        crate::mask_composite::MaskApplication::build(
            &self.mask_stacks,
            frame as i64,
            media.width,
            media.height,
        )
        .map(std::sync::Arc::new)
    }

    /// Refresh the worker's read-only mask copy after edits (cheap when unchanged).
    fn refresh_mask_snapshot(&mut self) {
        if self
            .mask_snapshot
            .as_ref()
            .is_none_or(|(revision, _)| *revision != self.mask_revision)
        {
            self.mask_snapshot = Some((
                self.mask_revision,
                std::sync::Arc::new(self.mask_stacks.clone()),
            ));
        }
    }

    /// Masks for `frame`, to be rasterised on the preview worker. Call
    /// `refresh_mask_snapshot` first in the same frame.
    fn mask_job(&self, frame: u64) -> Option<preview::MaskJob> {
        let fingerprint = crate::mask_composite::mask_fingerprint(&self.mask_stacks, frame as i64)?;
        let (revision, stacks) = self.mask_snapshot.as_ref()?;
        debug_assert_eq!(*revision, self.mask_revision, "stale mask snapshot");
        let media = self.media();
        Some(preview::MaskJob {
            key: crate::mask_cache::MaskCacheKey {
                session: self.id,
                fingerprint,
                frame: frame as i64,
            },
            stacks: std::sync::Arc::clone(stacks),
            width: media.width,
            height: media.height,
        })
    }

    fn active_mask_layer(&self, kind: MaskKind) -> Option<&MaskLayer> {
        let selected = self.mask_stacks.stack(kind).selected?;
        self.mask_stacks
            .stack(kind)
            .layers
            .iter()
            .find(|layer| layer.layer_id == selected)
    }

    fn refresh_source(&mut self, media: MediaInfo) {
        let changed = self.media().source_identity != media.source_identity
            || self.media().width != media.width
            || self.media().height != media.height
            || self.media().frame_count != media.frame_count
            || self.media().fps != media.fps;
        self.project.prompts.retain(|point| {
            point.frame < media.frame_count
                && point.x.is_finite()
                && point.y.is_finite()
                && point.x >= 0.0
                && point.y >= 0.0
                && point.x < media.width as f32
                && point.y < media.height as f32
        });
        if self
            .project
            .scenes
            .first()
            .is_none_or(|scene| scene.start != 0)
            || self
                .project
                .scenes
                .last()
                .is_none_or(|scene| scene.end_exclusive != media.frame_count)
            || self.project.scenes.iter().any(|scene| {
                scene.start >= scene.end_exclusive || scene.end_exclusive > media.frame_count
            })
        {
            self.project.scenes = vec![SceneRange {
                index: 0,
                start: 0,
                end_exclusive: media.frame_count,
                enabled: true,
            }];
            self.selected_scene = 0;
        }
        self.playhead = self.playhead.min(media.frame_count - 1);
        self.range.start = self.range.start.min(media.frame_count - 1);
        self.range.end_exclusive = self
            .range
            .end_exclusive
            .min(media.frame_count)
            .max(self.range.start + 1);
        self.project.media = Some(media);
        if changed {
            self.frames.clear();
            self.masks.clear();
        }
        self.revision += 1;
    }
    /// Projects saved before SAM layers kept one subject mask per frame: it becomes the
    /// first layer's result.
    fn migrate_sam_layers(&mut self) {
        let legacy_masks = self
            .frames
            .values()
            .any(|files| files.sam_layers.is_empty() && files.mask.is_some());
        if self.project.sam_layers.is_empty() && legacy_masks {
            self.project
                .sam_layers
                .push(ck_desktop_core::SamLayer::subject());
        }
        let Some(first) = self.project.sam_layers.first().map(|l| l.id) else {
            return;
        };
        for files in self.frames.values_mut() {
            if files.sam_layers.is_empty()
                && let Some(mask) = &files.mask
            {
                files.sam_layers.insert(first, mask.clone());
            }
        }
        self.bind_legacy_sam_layers();
    }

    /// Layers from projects saved before scene-bound layers belong to no scene: bind each one to
    /// the scene(s) its points, boxes and masks are in (one layer per scene), or scene 1.
    fn bind_legacy_sam_layers(&mut self) {
        let mut legacy = false;
        for layer in &mut self.project.sam_layers {
            if layer.scene.is_none() {
                layer.scene = Some(0);
                legacy = true;
            }
        }
        if legacy {
            self.rehome_sam_layers();
        }
    }
    fn sam_layer(&self, id: u64) -> Option<&ck_desktop_core::SamLayer> {
        self.project.sam_layers.iter().find(|layer| layer.id == id)
    }
    /// The combined subject mask of `frame`, built from the enabled layers when not cached.
    fn subject_mask(
        &mut self,
        frame: u64,
        archive: Option<&project_store::EmbeddedArchive>,
    ) -> Option<PathBuf> {
        let output = self.project.output_dir.join("sam-composite");
        let layers = self.project.layers_for_frame(frame);
        let files = self.frames.get_mut(&frame)?;
        if files.mask.is_some() {
            return files.mask.clone();
        }
        let built = match crate::sam_layers::plan(&layers, &files.sam_layers) {
            crate::sam_layers::Composite::None => return None,
            crate::sam_layers::Composite::Single(path) => Ok(path),
            crate::sam_layers::Composite::Combine(inputs) => (|| {
                if let Some(archive) = archive {
                    let paths: Vec<PathBuf> = inputs.iter().map(|(p, _)| p.clone()).collect();
                    archive.materialize(&paths)?;
                }
                crate::sam_layers::write_combined(&inputs, &output, frame)
            })(),
        };
        match built {
            Ok(path) => {
                files.mask = Some(path.clone());
                Some(path)
            }
            Err(error) => {
                eprintln!("combining SAM layers for frame {frame}: {error:#}");
                None
            }
        }
    }
    /// Union of the enabled Remove layers' results on `frame` (drawn red on the overlay). One
    /// Remove layer is used as-is; several are combined once and cached by their inputs.
    fn removal_mask(
        &mut self,
        frame: u64,
        archive: Option<&project_store::EmbeddedArchive>,
    ) -> Option<PathBuf> {
        let files = self.frames.get(&frame)?;
        let inputs: Vec<PathBuf> = self
            .project
            .layers_for_frame(frame)
            .iter()
            .filter(|layer| layer.enabled && layer.mode == ck_desktop_core::SamLayerMode::Remove)
            .filter_map(|layer| files.sam_layers.get(&layer.id).cloned())
            .collect();
        match inputs.as_slice() {
            [] => None,
            [single] => Some(single.clone()),
            _ => {
                if let Some((cached_inputs, path)) = self.removal_masks.get(&frame)
                    && *cached_inputs == inputs
                {
                    return Some(path.clone());
                }
                let union: Vec<(PathBuf, ck_desktop_core::SamLayerMode)> = inputs
                    .iter()
                    .map(|path| (path.clone(), ck_desktop_core::SamLayerMode::Keep))
                    .collect();
                let output = self.project.output_dir.join("sam-removal");
                let built = (|| {
                    if let Some(archive) = archive {
                        archive.materialize(&inputs)?;
                    }
                    crate::sam_layers::write_combined(&union, &output, frame)
                })();
                match built {
                    Ok(path) => {
                        self.removal_masks.insert(frame, (inputs, path.clone()));
                        Some(path)
                    }
                    Err(error) => {
                        eprintln!("combining Remove layers for frame {frame}: {error:#}");
                        None
                    }
                }
            }
        }
    }
    /// True when any enabled layer has a result on `frame` (the timeline mask lane).
    fn has_subject_mask(&self, files: &FrameFiles) -> bool {
        files.mask.is_some()
            || self
                .project
                .sam_layers
                .iter()
                .any(|layer| layer.enabled && files.sam_layers.contains_key(&layer.id))
    }
    /// Layer order, mode, visibility or results changed: drop every cached combined mask.
    fn invalidate_sam_composites(&mut self) {
        for files in self.frames.values_mut() {
            if !files.sam_layers.is_empty() || files.mask.is_some() {
                files.mask = None;
            }
        }
        self.revision += 1;
    }
    /// Points of `layer` changed on `frame`: drop that layer's results for the scene (the keyed
    /// CorridorKey results stay viewable; re-run the scene to use the new hint).
    fn invalidate_prompt_scene(&mut self, frame: u64, layer: u64) {
        let range = self
            .project
            .scenes
            .iter()
            .find(|scene| frame >= scene.start && frame < scene.end_exclusive)
            .map(|scene| (scene.start, scene.end_exclusive))
            .unwrap_or((0, self.media().frame_count));
        self.frames.retain(|&frame, files| {
            if frame < range.0 || frame >= range.1 {
                return true;
            }
            files.sam_layers.remove(&layer);
            files.mask = None;
            files.rgba.is_some()
                || files.alpha.is_some()
                || files.checker.is_some()
                || !files.sam_layers.is_empty()
        });
        self.masks
            .retain(|batch| batch.range.end_exclusive <= range.0 || batch.range.start >= range.1);
        self.revision += 1;
    }
    /// Remove a layer with its points and results.
    fn remove_sam_layer(&mut self, id: u64) {
        let Some(layer) = self.sam_layer(id).cloned() else {
            return;
        };
        let _ = layer;
        self.project.sam_layers.retain(|layer| layer.id != id);
        self.project.prompts.retain(|point| point.layer != id);
        self.project.sam_boxes.retain(|b| b.layer != id);
        self.frames.retain(|_, files| {
            files.sam_layers.remove(&id);
            files.rgba.is_some()
                || files.alpha.is_some()
                || files.checker.is_some()
                || !files.sam_layers.is_empty()
        });
        if self.sam_layer == id {
            self.sam_layer = NO_LAYER;
            self.sync_sam_layer();
        }
        self.invalidate_sam_composites();
    }

    /// The layer a new point, box or text prompt goes to: the selected layer of the playhead
    /// scene, created (and selected) when the scene has none yet.
    fn layer_for_edit(&mut self) -> u64 {
        let scene = self.project.scene_of_frame(self.playhead);
        if self
            .sam_layer(self.sam_layer)
            .is_some_and(|layer| layer.in_scene(scene))
        {
            return self.sam_layer;
        }
        self.sam_layer = self.project.ensure_scene_layer(scene);
        self.sam_layer
    }

    /// Source frames of the scene under the playhead (the whole clip without scenes).
    fn scene_frames_at_playhead(&self) -> std::ops::Range<u64> {
        self.project
            .scene_index_containing(self.playhead)
            .and_then(|index| self.project.scenes.get(index))
            .map_or(0..self.media().frame_count, |scene| {
                scene.start..scene.end_exclusive
            })
    }

    /// Keep the selected SAM layer inside the playhead's scene (subject layers are scene
    /// bound, like garbage/despill layers); `NO_LAYER` when the scene has none.
    fn sync_sam_layer(&mut self) {
        let scene = self.project.scene_of_frame(self.playhead);
        if !self
            .sam_layer(self.sam_layer)
            .is_some_and(|layer| layer.in_scene(scene))
        {
            self.sam_layer = self
                .project
                .layers_in_scene(scene)
                .first()
                .map_or(NO_LAYER, |layer| layer.id);
        }
    }

    /// After scene changes (split, merge, detection), give every scene-bound layer the scene
    /// its points, boxes and results are in. Content spread over several scenes is split into
    /// one layer per scene (same name, mode and text), so nothing moves between scenes.
    fn rehome_sam_layers(&mut self) {
        let scene_count = self.project.scenes.len().max(1) as u32;
        let ids: Vec<u64> = self
            .project
            .sam_layers
            .iter()
            .filter(|layer| layer.scene.is_some())
            .map(|layer| layer.id)
            .collect();
        for id in ids {
            let mut used = BTreeSet::new();
            for point in self.project.prompts.iter().filter(|p| p.layer == id) {
                used.insert(self.project.scene_of_frame(point.frame));
            }
            for b in self.project.sam_boxes.iter().filter(|b| b.layer == id) {
                used.insert(self.project.scene_of_frame(b.frame));
            }
            for (&frame, files) in &self.frames {
                if files.sam_layers.contains_key(&id) {
                    used.insert(self.project.scene_of_frame(frame));
                }
            }
            let Some(pos) = self.project.sam_layers.iter().position(|l| l.id == id) else {
                continue;
            };
            let own = self.project.sam_layers[pos].scene.unwrap_or(0);
            let Some(&first) = used.iter().next() else {
                // Text-only or empty layer: keep its index inside the scene list.
                self.project.sam_layers[pos].scene = Some(own.min(scene_count - 1));
                continue;
            };
            let home = if used.contains(&own) { own } else { first };
            self.project.sam_layers[pos].scene = Some(home);
            for &scene in used.iter().filter(|&&scene| scene != home) {
                let new_id = self
                    .project
                    .sam_layers
                    .iter()
                    .map(|l| l.id + 1)
                    .max()
                    .unwrap_or(0);
                let mut copy = self.project.sam_layers[pos].clone();
                copy.id = new_id;
                copy.scene = Some(scene);
                self.project.sam_layers.insert(pos + 1, copy);
                let scene_of: Vec<u32> = self
                    .project
                    .prompts
                    .iter()
                    .map(|p| self.project.scene_of_frame(p.frame))
                    .collect();
                for (point, &s) in self.project.prompts.iter_mut().zip(&scene_of) {
                    if point.layer == id && s == scene {
                        point.layer = new_id;
                    }
                }
                let box_scene: Vec<u32> = self
                    .project
                    .sam_boxes
                    .iter()
                    .map(|b| self.project.scene_of_frame(b.frame))
                    .collect();
                for (b, &s) in self.project.sam_boxes.iter_mut().zip(&box_scene) {
                    if b.layer == id && s == scene {
                        b.layer = new_id;
                    }
                }
                let frames: Vec<u64> = self
                    .frames
                    .keys()
                    .copied()
                    .filter(|&f| self.project.scene_of_frame(f) == scene)
                    .collect();
                for frame in frames {
                    if let Some(files) = self.frames.get_mut(&frame)
                        && let Some(path) = files.sam_layers.remove(&id)
                    {
                        files.sam_layers.insert(new_id, path);
                    }
                }
            }
        }
        self.sync_sam_layer();
        self.invalidate_sam_composites();
    }
    fn discard_render_attempt(&mut self, rendered_frames: &BTreeSet<u64>) {
        for frame in rendered_frames {
            let remove = self.frames.get_mut(frame).is_some_and(|files| {
                files.rgba = None;
                files.alpha = None;
                files.checker = None;
                files.mask.is_none() && files.sam_layers.is_empty()
            });
            if remove {
                self.frames.remove(frame);
            }
        }
        self.revision += 1;
    }
}

enum IoMessage {
    Loaded(PathBuf, Result<ChromaProject, String>),
    Saved(PathBuf, Vec<(String, u64)>, Result<ChromaProject, String>),
    Relinked(u64, Result<MediaInfo, String>),
    Hint(u64, Result<HintFootage, String>),
    HintPrepared(Vec<QueuedJob>, Result<Option<HintInput>, String>),
    FrameExported(Result<PathBuf, String>),
    Materialized(Result<(), String>),
    ExportReady(ExportRequest, Result<(), String>),
}

enum MediaAction {
    Activate(usize),
    Reveal(usize),
    Reconnect(usize),
    Hint(usize),
    ClearHint(usize),
    Remove(usize),
}

struct QueuedJob {
    request: JobRequest,
    label: String,
    session_id: u64,
    range: FrameRange,
    device_id: u32,
    /// SAM jobs: the layer whose points they segment and whose results they produce.
    sam_layer: Option<u64>,
    /// CorridorKey jobs with a SAM hint: built from the layers before the job starts.
    hint_plan: Option<HintPlan>,
}

/// Progress of a scene subject-mask export (written on a background thread).
struct MaskExportStatus {
    total: usize,
    written: std::sync::atomic::AtomicUsize,
    done: std::sync::Mutex<Option<(Result<usize, String>, u64)>>,
}

impl MaskExportStatus {
    fn new(total: usize) -> Self {
        Self {
            total,
            written: std::sync::atomic::AtomicUsize::new(0),
            done: std::sync::Mutex::new(None),
        }
    }
    fn progress(&self, written: usize) {
        self.written
            .store(written, std::sync::atomic::Ordering::Relaxed);
    }
    fn finish(&self, result: Result<usize, String>, first: u64) {
        if let Ok(mut done) = self.done.lock() {
            *done = Some((result, first));
        }
    }
    fn label(&self) -> String {
        match self.done.lock().ok().and_then(|done| done.clone()) {
            None => format!(
                "Exporting mask {}/{}…",
                self.written.load(std::sync::atomic::Ordering::Relaxed),
                self.total
            ),
            Some((Ok(count), first)) => {
                format!("Exported {count} mask frames (from frame {}).", first + 1)
            }
            Some((Err(error), _)) => format!("Mask export failed: {error}"),
        }
    }
}

/// A CorridorKey hint made of combined SAM layer masks, written to its own folder.
struct HintPlan {
    frames: Vec<(u64, crate::sam_layers::Composite)>,
    directory: PathBuf,
    first_frame: bool,
    quality: f32,
}

impl HintPlan {
    /// Write `{frame:06}.png` for every frame and return the hint input that reads them.
    fn build(self) -> anyhow::Result<HintInput> {
        use crate::sam_layers::Composite;
        std::fs::create_dir_all(&self.directory)?;
        for (frame, composite) in &self.frames {
            let target = self.directory.join(format!("{frame:06}.png"));
            match composite {
                Composite::None => anyhow::bail!("no SAM mask for frame {}", frame + 1),
                Composite::Single(path) => {
                    if std::fs::hard_link(path, &target).is_err() {
                        std::fs::copy(path, &target)?;
                    }
                }
                Composite::Combine(inputs) => crate::sam_layers::combine(inputs)?.save(&target)?,
            }
        }
        let (first, _) = self
            .frames
            .first()
            .ok_or_else(|| anyhow::anyhow!("empty SAM hint"))?;
        Ok(if self.first_frame {
            HintInput::FirstFrame {
                mask_file: self.directory.join(format!("{first:06}.png")),
                frame: *first,
                quality: self.quality,
            }
        } else {
            HintInput::Video {
                mask_directory: self.directory,
                frame_offset: 0,
                quality: self.quality,
            }
        })
    }
}
struct ActiveJob {
    worker: Worker,
    queued: QueuedJob,
    progress: crate::job_progress::JobProgress,
    message: String,
    policy_message: Option<String>,
    memory_retry_message: Option<String>,
    rendered_frames: BTreeSet<u64>,
}
/// Background fill of the RAM result cache after a calculation completes.
/// Refilled automatically when masks/despill/settings change the composition.
struct AutoResultCache {
    session_id: u64,
    /// Composition signature the queued/cached frames belong to.
    signature: String,
    revision: u64,
    despill_bits: u32,
    mode: ViewMode,
    background: Background,
    frames: std::collections::VecDeque<u64>,
    inflight: usize,
    /// Playhead when the queue was last rebuilt (recentres the fill window).
    last_playhead: u64,
    /// Reached after a full scan found nothing left to cache.
    exhausted: bool,
    /// Result-cache bytes at the previous pump (detects eviction/shrink).
    last_cache_bytes: usize,
    /// Calculated frames already inserted by the automatic filler.
    cached: std::collections::BTreeSet<u64>,
    /// When the current background request was sent (watchdog for lost work).
    inflight_since: Option<std::time::Instant>,
    /// Frames whose background composition failed; skipped until the scan restarts.
    failed: BTreeSet<u64>,
    /// Calculated frames seen at the last scan; new results from a running job restart it.
    available: usize,
    /// Session mask revision at the last scan; edits re-check which frames still match.
    mask_revision: u64,
}

/// Order calculated frames for cache filling: nearest to the playhead first.
fn auto_cache_frame_order(
    frames: impl Iterator<Item = u64>,
    playhead: u64,
) -> std::collections::VecDeque<u64> {
    let mut ordered = frames.collect::<Vec<_>>();
    ordered.sort_by_key(|frame| (frame.abs_diff(playhead), *frame));
    ordered.into()
}

struct LoadedFrame {
    key: String,
    frame: u64,
    mode: ViewMode,
    texture: egui::TextureHandle,
    path: Option<PathBuf>,
    background_applied: bool,
    source_rgba: Option<Arc<[u8]>>,
    despill_applied: bool,
    despill_unavailable: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HintMode {
    None,
    First,
    Video,
    External,
}
#[derive(Clone, Copy)]
enum Action {
    Detect,
    Segment,
    Propagate,
    PropagateAll,
    RenderRange,
    RenderScene,
    RenderAll,
}

/// What a click/drag in the viewer does on the Subject tab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PointTool {
    Subject,
    Background,
    Box,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum LeftTab {
    Media,
    Subject,
    Garbage,
    Despill,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum RightTab {
    Keyer,
    Scenes,
    Jobs,
    Info,
}

pub struct Studio {
    assets: AssetSetup,
    icons: Icons,
    app_directory: PathBuf,
    pending_imports: Vec<PathBuf>,
    runtime: RuntimePaths,
    /// SAM 3.1 text prompts typed in Portuguese go to jobs in English (see `translation.rs`).
    translator: crate::translation::TranslationService,
    /// Running or last scene mask export (Subject tab).
    export_masks: Option<std::sync::Arc<MaskExportStatus>>,
    /// GitHub release check and in-app update (see `update_ui.rs`).
    updater: crate::update_ui::Updater,
    /// Screen rects of the layer rows' enable checkboxes (last frame; tests and hit checks).
    layer_checkbox_rects: Vec<Rect>,
    work_root: PathBuf,
    temporary: Arc<TemporaryWorkspace>,
    io_sender: mpsc::Sender<IoMessage>,
    io_messages: mpsc::Receiver<IoMessage>,
    project_io_busy: bool,
    calculated_dirty: bool,
    archive: Option<Arc<project_store::EmbeddedArchive>>,
    embed_sources: bool,
    materializing: bool,
    materialize_failed: bool,
    sessions: Vec<Session>,
    selected: usize,
    preview: PreviewService,
    loaded: Option<LoadedFrame>,
    requested_key: Option<String>,
    mode: ViewMode,
    background: Background,
    left_tab: LeftTab,
    right_tab: RightTab,
    viewer_scale: f32,
    viewer_fit: f32,
    startup_maximize: bool,
    ui_only: bool,
    overlay: bool,
    /// (frame, layer) whose subject points changed; segmented again once no job is running.
    segment_pending: Option<(u64, u64)>,
    /// Subject point being dragged in the viewer: index into `project.prompts`.
    point_drag: Option<usize>,
    /// SAM layer row being dragged in the Layers list (display index).
    layer_drag: Option<usize>,
    /// SAM layer being renamed in place: (layer id, edited text).
    layer_rename: Option<(u64, String)>,
    /// Layer text prompt as it was when its field gained focus (applied only if changed).
    layer_prompt_focus: Option<String>,
    /// Viewer click tool on the Subject tab.
    point_tool: PointTool,
    /// Point whose text prompt is being edited beside it: (index into prompts, text).
    point_text: Option<(usize, String)>,
    /// Box being dragged in the viewer: (frame, layer, start, current) in source pixels.
    box_drag: Option<(u64, u64, Pos2, Pos2)>,
    /// Subject segmentation model (SAM 3.1 only when its optional pack is installed).
    sam_model: SamModel,
    zoom: ViewerZoom,
    pan: Vec2,
    playing: bool,
    playback_tick: Instant,
    playback_fraction: f64,
    hints: HintMode,
    provider: Provider,
    device_id: u32,
    devices: Vec<ck_desktop_core::GpuDevice>,
    repaint: egui::Context,
    scene_threshold: f64,
    min_scene_frames: u64,
    jobs: VecDeque<QueuedJob>,
    active: Option<ActiveJob>,
    log: VecDeque<String>,
    log_open: bool,
    error: Option<String>,
    can_select_cpu: bool,
    toast: String,
    open_settings: bool,
    project_path: Option<PathBuf>,
    screenshot: Option<PathBuf>,
    ui_report: Option<PathBuf>,
    last_ui_report: Instant,
    presented_frames: u64,
    viewer_canvas: Rect,
    viewer_image: Rect,
    mask_edit: crate::mask_overlay::MaskEditState,
    /// Ctrl+C/Ctrl+V mask layer clipboard (kind + deep layer snapshot).
    mask_clipboard: Option<(MaskKind, MaskLayer)>,
    cache_settings: crate::memory_budget::CacheSettings,
    ram_total: u64,
    result_cache: crate::frame_cache::ResultRamCache,
    preview_cache: std::sync::Arc<preview::PreviewCacheState>,
    app_history: crate::app_history::AppHistory,
    auto_cache: Option<AutoResultCache>,
    left_panel_rect: Rect,
    inspector_rect: Rect,
    timeline_rect: Rect,
    screenshot_started: bool,
    started: Instant,
    export_dialog: bool,
    export_format: ExportFormat,
    export: Option<ExportService>,
    export_lease: Option<Arc<TemporaryWorkspace>>,
    export_progress: (u64, u64),
}

impl Studio {
    pub fn new(
        creation: &eframe::CreationContext<'_>,
        _workspace: PathBuf,
        open: Option<PathBuf>,
        screenshot: Option<PathBuf>,
        ui_only: bool,
        startup_maximize: bool,
        ui_report: Option<PathBuf>,
    ) -> Self {
        let ctx = &creation.egui_ctx;
        theme::configure(ctx);
        let app_directory = RuntimePaths::application_directory();
        let updater = crate::update_ui::Updater::start(
            app_directory.clone(),
            screenshot.is_none() && !ui_only,
            ctx.clone(),
        );
        let temporary = Arc::new(
            TemporaryWorkspace::new().expect("Create app temporary calculations directory"),
        );
        let work_root = temporary.root.clone();
        let (io_sender, io_messages) = mpsc::channel();
        let devices = ck_desktop_core::enumerate_gpus().unwrap_or_default();
        let default_device = devices.first().map_or(0, |device| device.id);
        let provider = if cfg!(windows) && !devices.is_empty() {
            Provider::DirectMl
        } else {
            Provider::Cpu
        };
        let runtime = RuntimePaths::installed(&app_directory);
        let assets = AssetSetup::new(app_directory.clone(), ctx.clone());
        let preview = PreviewService::new(
            runtime.clone(),
            work_root.join("source-preview"),
            ctx.clone(),
        );
        let cache_settings = crate::memory_budget::load();
        let ram_total = crate::memory_budget::total_ram_bytes();
        let cache_limits = crate::memory_budget::cache_limits(ram_total, cache_settings.percent);
        let preview_cache = preview.cache_state();
        preview_cache.set_limits(cache_limits.source_bytes, cache_limits.source_items);
        let mut app = Self {
            assets,
            icons: Icons::new(ctx),
            app_directory,
            pending_imports: vec![],
            export_masks: None,
            updater,
            layer_checkbox_rects: Vec::new(),
            translator: {
                let mut translator = crate::translation::TranslationService::new(runtime.clone());
                translator.set_repaint(ctx);
                translator
            },
            runtime,
            work_root,
            temporary,
            io_sender,
            io_messages,
            project_io_busy: false,
            calculated_dirty: false,
            archive: None,
            embed_sources: false,
            materializing: false,
            materialize_failed: false,
            sessions: vec![],
            selected: 0,
            preview,
            loaded: None,
            requested_key: None,
            mode: ViewMode::Composite,
            background: Background::default(),
            left_tab: LeftTab::Media,
            right_tab: RightTab::Keyer,
            viewer_scale: 0.0,
            viewer_fit: 1.0,
            startup_maximize,
            ui_only,
            overlay: true,
            segment_pending: None,
            point_drag: None,
            layer_drag: None,
            layer_rename: None,
            layer_prompt_focus: None,
            point_tool: PointTool::Subject,
            point_text: None,
            box_drag: None,
            sam_model: SamModel::Sam31,
            zoom: ViewerZoom::default(),
            pan: Vec2::ZERO,
            playing: false,
            playback_tick: Instant::now(),
            playback_fraction: 0.0,
            hints: HintMode::None,
            provider,
            device_id: default_device,
            devices,
            repaint: ctx.clone(),
            scene_threshold: 0.35,
            min_scene_frames: 6,
            jobs: VecDeque::new(),
            active: None,
            log: VecDeque::new(),
            log_open: false,
            error: None,
            can_select_cpu: false,
            toast: "Drop footage here or open a clip to begin".into(),
            open_settings: false,
            project_path: None,
            screenshot,
            ui_report,
            last_ui_report: Instant::now(),
            presented_frames: 0,
            viewer_canvas: Rect::NOTHING,
            viewer_image: Rect::NOTHING,
            mask_edit: crate::mask_overlay::MaskEditState::default(),
            mask_clipboard: None,
            cache_settings,
            ram_total,
            result_cache: crate::frame_cache::ResultRamCache::new(
                cache_limits.result_bytes,
                cache_limits.result_items,
            ),
            preview_cache,
            app_history: crate::app_history::AppHistory::default(),
            auto_cache: None,
            left_panel_rect: Rect::NOTHING,
            inspector_rect: Rect::NOTHING,
            timeline_rect: Rect::NOTHING,
            screenshot_started: false,
            started: Instant::now(),
            export_dialog: false,
            export_format: ExportFormat::ProRes4444,
            export: None,
            export_lease: None,
            export_progress: (0, 0),
        };
        // Screenshot QA only: `BRP_CHROMA_QA_VIEW=garbage,scenes` picks the starting tabs
        // so CPU screenshots can cover panels that otherwise need a click.
        if app.screenshot.is_some()
            && let Ok(view) = std::env::var("BRP_CHROMA_QA_VIEW")
        {
            for part in view.split(',').map(str::trim) {
                match part {
                    "subject" => app.left_tab = LeftTab::Subject,
                    "garbage" => app.left_tab = LeftTab::Garbage,
                    "despill" => app.left_tab = LeftTab::Despill,
                    "scenes" => app.right_tab = RightTab::Scenes,
                    "jobs" => app.right_tab = RightTab::Jobs,
                    "info" => app.right_tab = RightTab::Info,
                    _ => {}
                }
            }
        }
        if let Some(path) = open {
            app.import(path);
        }
        app
    }

    fn current(&self) -> Option<&Session> {
        self.sessions.get(self.selected)
    }
    fn current_mut(&mut self) -> Option<&mut Session> {
        self.sessions.get_mut(self.selected)
    }
    /// Geometry and estimated GPU budget for the Custom Hann advisory.
    ///
    /// DirectML jobs only: the worker resolves the authoritative live budget,
    /// so this estimate only drives pre-run UI guidance.
    fn hann_advice_input(&self) -> Option<(usize, usize, usize, f64)> {
        if self.provider != Provider::DirectMl {
            return None;
        }
        let session = self.current()?;
        let media = session.media();
        let frames = session
            .range
            .end_exclusive
            .saturating_sub(session.range.start) as usize;
        let device = self
            .devices
            .iter()
            .find(|device| device.id == self.device_id)?;
        let budget_gib = crate::hann_advice::estimated_budget_gib(device.dedicated_bytes)?;
        Some((
            media.width as usize,
            media.height as usize,
            frames,
            budget_gib,
        ))
    }
    fn push_log(&mut self, text: impl Into<String>) {
        self.log.push_back(text.into());
        while self.log.len() > 180 {
            self.log.pop_front();
        }
    }
    fn fail(&mut self, text: impl Into<String>) {
        let text = text.into();
        self.push_log(format!("Error: {text}"));
        self.error = Some(text);
        self.can_select_cpu = false;
    }
    fn import(&mut self, path: PathBuf) {
        if self.active.is_some() || !self.jobs.is_empty() || self.project_io_busy {
            self.fail("Finish or cancel the current job before importing another clip.");
            return;
        }
        if !path.is_file() {
            self.fail(format!("Footage does not exist: {}", path.display()));
            return;
        }
        if path.extension().is_some_and(|extension| {
            extension.eq_ignore_ascii_case("brpnative") || extension.eq_ignore_ascii_case("chroma")
        }) {
            self.load_project_path(path);
            return;
        }
        if !self.assets.require(&[AssetGroup::Core]) {
            if !self.pending_imports.contains(&path) {
                self.pending_imports.push(path);
            }
            self.toast = "Preparing required files before opening footage".into();
            return;
        }
        self.toast = format!(
            "Opening {}…",
            path.file_name().unwrap_or_default().to_string_lossy()
        );
        self.preview.import(path, self.runtime.clone());
    }
    fn open_footage(&mut self) {
        if let Some(paths) = rfd::FileDialog::new()
            .set_title("Open footage")
            .add_filter("Video", &["mp4", "mov", "mkv", "avi", "mxf", "webm"])
            .pick_files()
        {
            for path in paths {
                self.import(path);
            }
        }
    }
    fn accept_messages(&mut self, ctx: &egui::Context) {
        while let Ok(message) = self.preview.messages.try_recv() {
            match message {
                preview::Message::Imported(Ok(media)) => {
                    if self.active.is_some() || !self.jobs.is_empty() {
                        self.fail("Footage probing finished while a job was running. Finish or cancel the job, then reopen the clip.");
                        continue;
                    }
                    if let Some(index) = self
                        .sessions
                        .iter()
                        .position(|s| s.media().path == media.path)
                    {
                        self.sessions[index].refresh_source(media);
                        self.selected = index;
                    } else {
                        let count = media.frame_count;
                        let project = Project {
                            schema_version: 1,
                            media: Some(media),
                            scenes: vec![SceneRange {
                                index: 0,
                                start: 0,
                                end_exclusive: count,
                                enabled: true,
                            }],
                            prompts: vec![],
                            sam_layers: vec![ck_desktop_core::SamLayer::subject()],
                            hint_quality: Default::default(),
                            sam_boxes: Vec::new(),
                            output_dir: self.work_root.join(project_store::unique_key()),
                            ck_settings: CkSettings::default(),
                            despill_settings: Default::default(),
                            mask_layers: Default::default(),
                            scene_overrides: Default::default(),
                        };
                        self.sessions.push(Session::new(project));
                        self.selected = self.sessions.len() - 1;
                    }
                    self.loaded = None;
                    self.requested_key = None;
                    self.zoom = ViewerZoom::default();
                    self.pan = Vec2::ZERO;
                    self.toast = "Footage ready".into();
                }
                preview::Message::Imported(Err(error)) => self.fail(error),
                preview::Message::Cached { key, frame, result } => {
                    if let Some(state) = self.auto_cache.as_mut() {
                        state.inflight = state.inflight.saturating_sub(1);
                        state.inflight_since = None;
                        let expected = format!("{}:", state.signature);
                        if !key.starts_with(&expected) {
                            ctx.request_repaint();
                            continue;
                        }
                    }
                    if result.is_err() {
                        // Skip only this frame; the rest of the range keeps filling.
                        // A playhead move or composition change retries it.
                        if let Some(state) = self.auto_cache.as_mut() {
                            state.failed.insert(frame);
                        }
                        ctx.request_repaint();
                        continue;
                    }
                    if let Ok(pixels) = result {
                        // Never retain a no-despill fallback frame: the plate was
                        // unavailable only for this background request.
                        if pixels.despill_unavailable
                            && self.current().is_some_and(|session| {
                                session.project.despill_settings.is_enabled()
                            })
                        {
                            // Stop only when the plate is really offline; a transient read
                            // failure skips just this frame (retried after a playhead move).
                            let offline = self.current().is_some_and(|session| {
                                let source = &session.media().path;
                                !source.is_file()
                                    && !self
                                        .archive
                                        .as_ref()
                                        .is_some_and(|archive| archive.is_embedded(source))
                            });
                            if let Some(state) = self.auto_cache.as_mut() {
                                if offline {
                                    state.frames.clear();
                                    state.exhausted = true;
                                } else {
                                    state.failed.insert(frame);
                                }
                            }
                            ctx.request_repaint();
                            continue;
                        }
                        let cached = crate::frame_cache::CachedPreview {
                            width: pixels.width,
                            height: pixels.height,
                            rgba: pixels.rgba.clone(),
                            path: pixels.path.clone(),
                            background_applied: pixels.background_applied,
                            despill_applied: pixels.despill_applied,
                            despill_unavailable: pixels.despill_unavailable,
                            source_rgba: pixels.source_rgba.clone(),
                        };
                        let retained = self.result_cache.insert(key, frame, cached);
                        let evicted = self.result_cache.take_evicted_frames();
                        if let Some(state) = self.auto_cache.as_mut() {
                            for frame in evicted {
                                state.cached.remove(&frame);
                            }
                            if retained {
                                state.cached.insert(frame);
                            } else {
                                state.frames.clear();
                                state.exhausted = true;
                            }
                        }
                    }
                    ctx.request_repaint();
                }
                preview::Message::Frame {
                    key,
                    frame,
                    mode,
                    result,
                } => {
                    let exact = self.requested_key.as_ref() == Some(&key);
                    let playback_frame = self.playing
                        && self
                            .requested_key
                            .as_ref()
                            .is_some_and(|requested| same_preview_context(requested, &key))
                        && self.current().is_some_and(|session| {
                            playback_frame_order(
                                frame,
                                session.playhead,
                                self.loaded.as_ref().map(|loaded| loaded.frame),
                            )
                        });
                    if !exact && (!playback_frame || result.is_err()) {
                        continue;
                    }
                    match result {
                        Ok(pixels) => {
                            self.presented_frames += 1;
                            let cached = crate::frame_cache::CachedPreview {
                                width: pixels.width,
                                height: pixels.height,
                                rgba: pixels.rgba.clone(),
                                path: pixels.path.clone(),
                                background_applied: pixels.background_applied,
                                despill_applied: pixels.despill_applied,
                                despill_unavailable: pixels.despill_unavailable,
                                source_rgba: pixels.source_rgba.clone(),
                            };
                            self.result_cache.insert(key.clone(), frame, cached);
                            let image = egui::ColorImage::from_rgba_unmultiplied(
                                [pixels.width, pixels.height],
                                &pixels.rgba,
                            );
                            let texture = if let Some(mut previous) = self.loaded.take() {
                                previous.texture.set(image, egui::TextureOptions::LINEAR);
                                previous.texture
                            } else {
                                ctx.load_texture(
                                    "current-video-frame",
                                    image,
                                    egui::TextureOptions::LINEAR,
                                )
                            };
                            self.loaded = Some(LoadedFrame {
                                key,
                                frame,
                                mode,
                                texture,
                                path: pixels.path,
                                background_applied: pixels.background_applied,
                                despill_applied: pixels.despill_applied,
                                despill_unavailable: pixels.despill_unavailable,
                                source_rgba: pixels.source_rgba,
                            });
                        }
                        Err(error) => self.fail(error),
                    }
                }
            }
        }
    }

    fn cache_limits(&self) -> crate::memory_budget::CacheLimits {
        crate::memory_budget::cache_limits(self.ram_total, self.cache_settings.percent)
    }

    fn apply_cache_settings(&mut self) {
        let limits = self.cache_limits();
        self.preview_cache
            .set_limits(limits.source_bytes, limits.source_items);
        if let Ok(mut masks) = self.preview.mask_cache().lock() {
            masks.retune(limits.mask_bytes, limits.mask_items);
        }
        self.result_cache
            .retune(limits.result_bytes, limits.result_items);
        let _ = crate::memory_budget::save(&self.cache_settings);
    }

    fn clear_ram_caches(&mut self) {
        self.auto_cache = None;
        self.preview.clear_cache_queue();
        self.preview_cache.request_clear();
        if let Ok(mut masks) = self.preview.mask_cache().lock() {
            masks.clear();
        }
        self.result_cache.clear();
    }

    fn mask_raster_bytes(&self) -> usize {
        self.preview
            .mask_cache()
            .lock()
            .map_or(0, |masks| masks.bytes())
    }

    fn ram_cache_usage(&self) -> (usize, usize) {
        let used = self
            .preview_cache
            .bytes()
            .saturating_add(self.result_cache.bytes())
            .saturating_add(self.mask_raster_bytes());
        let budget = crate::memory_budget::budget_bytes(self.ram_total, self.cache_settings.percent)
            as usize;
        (used, budget)
    }

    fn present_cached_preview(
        &mut self,
        key: String,
        frame: u64,
        cached: &crate::frame_cache::CachedPreview,
    ) -> bool {
        let image =
            egui::ColorImage::from_rgba_unmultiplied([cached.width, cached.height], &cached.rgba);
        let texture = if let Some(mut previous) = self.loaded.take() {
            previous.texture.set(image, egui::TextureOptions::LINEAR);
            previous.texture
        } else {
            self.repaint
                .load_texture("current-video-frame", image, egui::TextureOptions::LINEAR)
        };
        self.loaded = Some(LoadedFrame {
            key,
            frame,
            mode: self.mode,
            texture,
            path: cached.path.clone(),
            background_applied: cached.background_applied,
            source_rgba: cached.source_rgba.clone(),
            despill_applied: cached.despill_applied,
            despill_unavailable: cached.despill_unavailable,
        });
        self.presented_frames = self.presented_frames.saturating_add(1);
        true
    }

    fn copy_selected_mask_layer(&mut self) -> bool {
        let Some(kind) = self.active_mask_tab() else {
            return false;
        };
        let Some(session) = self.current() else {
            return false;
        };
        let Some(layer) = session.mask_stacks.stack(kind).selected_layer().cloned() else {
            return false;
        };
        let name = layer.name.clone();
        self.mask_clipboard = Some((kind, layer));
        self.toast = format!("Copied mask layer {name}");
        true
    }

    fn paste_mask_layer(&mut self) -> bool {
        let Some((kind, layer)) = self.mask_clipboard.clone() else {
            return false;
        };
        let Some(session) = self.current_mut() else {
            return false;
        };
        let scene = session.mask_scene_index() as u32;
        let Some((start, end)) = session.project.scene_inclusive_range(scene) else {
            return false;
        };
        session.record_mask_edit();
        let id = session
            .mask_stacks
            .stack_mut(kind)
            .add_copy(&layer, scene, start, end);
        session.mask_ui.set_active_kind(kind);
        session.mask_ui.remember(kind, scene, Some(id));
        session.mask_revision = session.mask_revision.wrapping_add(1);
        self.left_tab = match kind {
            MaskKind::Garbage => LeftTab::Garbage,
            MaskKind::Despill => LeftTab::Despill,
        };
        self.requested_key = None;
        self.toast = format!("Pasted mask layer into scene {:02}", scene + 1);
        true
    }

    /// The SAM model jobs use: SAM 3.1 (the default) when its pack is installed, else SAM 2.1.
    fn effective_sam_model(&self) -> SamModel {
        if self.sam_model == SamModel::Sam31 && self.runtime.sam31_available() {
            SamModel::Sam31
        } else {
            SamModel::Sam21
        }
    }

    /// Segment the frame whose subject points changed, once nothing else is running. A
    /// change made while a job runs is picked up by the next pass.
    fn pump_auto_segment(&mut self) {
        let Some((frame, layer)) = self.segment_pending else {
            return;
        };
        if self.active.is_some()
            || !self.jobs.is_empty()
            || self.project_io_busy
            || self.export.is_some()
        {
            return;
        }
        self.segment_pending = None;
        let sam31 = self.effective_sam_model() == SamModel::Sam31;
        let ready =
            self.current().is_some_and(|session| {
                let project = &session.project;
                session.playhead == frame
                    && session.sam_layer == layer
                    && (project.prompts.iter().any(|point| {
                        point.frame == frame && point.positive && point.layer == layer
                    }) || (sam31
                        && (project
                            .sam_boxes
                            .iter()
                            .any(|b| b.frame == frame && b.layer == layer)
                            || session
                                .sam_layer(layer)
                                .is_some_and(|l| !l.prompt.trim().is_empty()))))
            });
        if ready {
            self.enqueue(Action::Segment);
        }
    }

    fn queue_auto_result_cache(&mut self, session_id: u64, _range: FrameRange) {
        let Some(session) = self.current() else {
            return;
        };
        if session.id != session_id {
            return;
        }
        self.preview.clear_cache_queue();
        self.auto_cache = Some(AutoResultCache {
            session_id,
            signature: preview_composition_signature(
                session,
                ViewMode::Composite,
                self.overlay,
                self.background,
            ),
            revision: session.revision,
            despill_bits: session.project.despill_settings.amount.to_bits(),
            mode: self.mode,
            background: self.background,
            frames: std::collections::VecDeque::new(),
            inflight: 0,
            last_playhead: session.playhead,
            exhausted: false,
            last_cache_bytes: self.result_cache.bytes(),
            cached: std::collections::BTreeSet::new(),
            inflight_since: None,
            failed: BTreeSet::new(),
            available: 0,
            mask_revision: 0,
        });
    }

    fn auto_cache_request(
        &self,
        session: &Session,
        frame: u64,
        background: Background,
        overlay: bool,
    ) -> Option<preview::Request> {
        let artifact = session.frames.get(&frame)?.rgba.clone()?;
        if !artifact.is_file() {
            return None;
        }
        if session.project.despill_settings.is_enabled() {
            let source = &session.media().path;
            let source_available = source.is_file()
                || self
                    .archive
                    .as_ref()
                    .is_some_and(|archive| archive.is_embedded(source));
            if !source_available {
                return None;
            }
        }
        let _ = overlay;
        let key = preview_composition_key(session, frame, ViewMode::Composite, false, background);
        Some(preview::Request {
            key,
            media: session.media().clone(),
            frame,
            mode: ViewMode::Composite,
            artifact: Some(artifact),
            artifact_is_straight_rgba: true,
            overlay: None,
            overlay_remove: None,
            background,
            despill: session.project.despill_settings_for_frame(frame),
            masks: session.mask_job(frame),
        })
    }

    fn pump_auto_result_cache(&mut self) {
        if self.auto_cache.is_none() {
            // Start collecting without waiting for a new calculation: projects
            // opened with calculated frames (or after despill/mask edits) should
            // still auto-fill the result cache, also while other scenes calculate.
            if self.mode != ViewMode::Composite {
                return;
            }
            let Some(session) = self.current() else {
                return;
            };
            if !session
                .frames
                .values()
                .any(|files| files.rgba.as_ref().is_some_and(|path| path.is_file()))
            {
                return;
            }
            self.auto_cache = Some(AutoResultCache {
                session_id: session.id,
                signature: preview_composition_signature(
                    session,
                    ViewMode::Composite,
                    self.overlay,
                    self.background,
                ),
                revision: session.revision,
                despill_bits: session.project.despill_settings.amount.to_bits(),
                mode: self.mode,
                background: self.background,
                frames: std::collections::VecDeque::new(),
                inflight: 0,
                last_playhead: session.playhead,
                exhausted: false,
                last_cache_bytes: self.result_cache.bytes(),
                cached: std::collections::BTreeSet::new(),
                inflight_since: None,
                failed: BTreeSet::new(),
                available: 0,
                mask_revision: 0,
            });
        }
        if let Some(session) = self.current_mut() {
            session.refresh_mask_snapshot();
        }
        let Some(mut state) = self.auto_cache.take() else {
            return;
        };
        // Keeps filling while calculations run: each new result invalidates its own
        // cached frame (FrameReady) and restarts the scan below.
        if self.mode != ViewMode::Composite {
            self.auto_cache = Some(state);
            return;
        }
        // Read the session values first, then drop the session borrow so the
        // result cache can be purged for the new composition.
        let (signature, composition_changed, session_revision, session_despill, session_playhead) = {
            let Some(session) = self.current() else {
                return;
            };
            if session.id != state.session_id {
                return;
            }
            let signature = preview_composition_signature(
                session,
                ViewMode::Composite,
                self.overlay,
                self.background,
            );
            let changed = state.signature != signature
                || state.revision != session.revision
                || state.despill_bits != session.project.despill_settings.amount.to_bits()
                || state.mode != self.mode
                || state.background != self.background;
            (
                signature,
                changed,
                session.revision,
                session.project.despill_settings.amount.to_bits(),
                session.playhead,
            )
        };
        let cache_bytes = self.result_cache.bytes();
        if composition_changed {
            self.result_cache.purge_except_prefix(&signature);
            state.signature = signature.clone();
            state.revision = session_revision;
            state.despill_bits = session_despill;
            state.mode = self.mode;
            state.background = self.background;
            state.frames.clear();
            state.inflight = 0;
            state.inflight_since = None;
            state.exhausted = false;
            state.last_playhead = session_playhead;
            state.cached.clear();
            state.failed.clear();
            self.preview.clear_cache_queue();
        }
        let Some(session) = self.current() else {
            return;
        };
        let evicted = cache_bytes < state.last_cache_bytes;
        state.last_cache_bytes = cache_bytes;
        if evicted {
            state.cached.clear();
        }
        if state.inflight > 0
            && state
                .inflight_since
                .is_some_and(|started| started.elapsed() > std::time::Duration::from_secs(15))
        {
            // A lost/crashed background request must never stop the filler.
            state.inflight = 0;
            state.inflight_since = None;
        }
        let playhead_moved = session.playhead.abs_diff(state.last_playhead) >= 8 && !self.playing;
        if playhead_moved {
            // Moving the playhead while paused recentres the fill immediately so
            // the cache keeps working around the frame the user is looking at.
            state.last_playhead = session.playhead;
            state.frames.clear();
            state.exhausted = false;
            state.failed.clear();
            // A queued request dropped here never reports back; stop waiting for it.
            if self.preview.clear_cache_queue() > 0 {
                state.inflight = 0;
                state.inflight_since = None;
            }
        }
        if session.mask_revision != state.mask_revision {
            // Only frames whose mask fingerprint changed need recomposing; the key check
            // below skips every frame that is still cached under its current key.
            state.mask_revision = session.mask_revision;
            state.frames.clear();
            state.exhausted = false;
            state.failed.clear();
            if self.preview.clear_cache_queue() > 0 {
                state.inflight = 0;
                state.inflight_since = None;
            }
        }
        let available = session
            .frames
            .values()
            .filter(|files| files.rgba.is_some())
            .count();
        if available != state.available {
            state.available = available;
            state.exhausted = false;
        }
        let needs_rebuild = state.frames.is_empty()
            && state.inflight == 0
            && (!state.exhausted || evicted || playhead_moved);
        if needs_rebuild {
            state.last_playhead = session.playhead;
            let overlay = self.overlay;
            let background = self.background;
            let mut candidates = Vec::new();
            for (&frame, files) in &session.frames {
                if files.rgba.as_ref().is_some_and(|path| path.is_file()) {
                    let key = preview_composition_key(
                        session,
                        frame,
                        ViewMode::Composite,
                        overlay,
                        background,
                    );
                    candidates.push((frame, key));
                }
            }
            candidates.retain(|(frame, key)| {
                !state.failed.contains(frame) && !self.result_cache.contains(key)
            });
            state.frames = auto_cache_frame_order(
                candidates.into_iter().map(|(frame, _)| frame),
                session.playhead,
            );
            state.exhausted = state.frames.is_empty();
        }
        if state.inflight == 0 && self.preview.cache_queue_len() == 0 {
            while let Some(frame) = state.frames.pop_front() {
                if let Some(request) =
                    self.auto_cache_request(session, frame, self.background, self.overlay)
                {
                    if self.preview.cache_request(request) {
                        state.inflight = 1;
                        state.inflight_since = Some(std::time::Instant::now());
                    }
                    break;
                }
            }
        }
        self.auto_cache = Some(state);
    }

    fn request_preview(&mut self) {
        if !self.assets.ready_for(&[AssetGroup::Core])
            || self.materializing
            || self.materialize_failed
        {
            return;
        }
        let needs_subject_mask = self.mode == ViewMode::Sam
            || (self.overlay
                && (self.mode == ViewMode::Source || self.left_tab == LeftTab::Subject));
        let archive = self.archive.clone();
        // Same places as the blue subject overlay (not on the SAM mask view itself).
        let show_removal = self.overlay
            && self.mode != ViewMode::Sam
            && (self.mode == ViewMode::Source || self.left_tab == LeftTab::Subject);
        let Some(session) = self.current_mut() else {
            return;
        };
        session.refresh_mask_snapshot();
        let mut overlay_remove = None;
        if needs_subject_mask {
            let frame = session.playhead;
            session.subject_mask(frame, archive.as_deref());
            if show_removal {
                overlay_remove = session.removal_mask(frame, archive.as_deref());
            }
        }
        let Some(session) = self.current() else {
            return;
        };
        let media = session.media().clone();
        let frame = session.playhead;
        let files = session.frames.get(&frame);
        let known = |path: &std::path::Path| {
            path.is_file()
                || self
                    .archive
                    .as_ref()
                    .is_some_and(|archive| archive.is_embedded(path))
        };
        let artifact = files.and_then(|files| match self.mode {
            ViewMode::Source => None,
            ViewMode::Composite => files
                .rgba
                .clone()
                .filter(|path| known(path))
                .or_else(|| files.checker.clone().filter(|path| known(path))),
            ViewMode::Alpha => files.alpha.clone().filter(|path| known(path)),
            ViewMode::Sam => files.mask.clone().filter(|path| known(path)),
        });
        let actual_mode = if artifact.is_some() {
            self.mode
        } else {
            ViewMode::Source
        };
        let straight = self.mode == ViewMode::Composite
            && artifact
                .as_ref()
                .is_some_and(|path| files.and_then(|files| files.rgba.as_ref()) == Some(path));
        let despill = session.project.despill_settings_for_frame(frame);
        let masks = if straight {
            session.mask_job(frame)
        } else {
            None
        };
        let overlay = if self.overlay
            && (self.mode == ViewMode::Source || self.left_tab == LeftTab::Subject)
        {
            files.and_then(|files| files.mask.clone())
        } else {
            None
        };
        let key = preview_composition_key(
            session,
            frame,
            self.mode,
            overlay.is_some() || overlay_remove.is_some(),
            self.background,
        );
        if self.requested_key.as_ref() == Some(&key) {
            return;
        }
        if let Some(cached) = self.result_cache.get(&key) {
            // A frame cached while the original plate was briefly unavailable
            // must not keep showing the no-despill fallback after the source is
            // back: drop it and render a fresh composition instead.
            let stale_fallback =
                cached.despill_unavailable && despill.is_enabled() && known(&media.path);
            if !stale_fallback {
                self.present_cached_preview(key.clone(), frame, &cached);
                self.requested_key = Some(key);
                return;
            }
            self.result_cache.remove(&key);
        }
        let mut needed = vec![artifact.clone().unwrap_or_else(|| media.path.clone())];
        if straight
            && despill.is_enabled()
            && self
                .archive
                .as_ref()
                .is_some_and(|archive| archive.is_embedded(&media.path))
        {
            needed.push(media.path.clone());
        }
        if let Some(path) = &overlay {
            needed.push(path.clone());
        }
        if let Some(path) = &overlay_remove {
            needed.push(path.clone());
        }
        if let Some(archive) = self.archive.clone()
            && needed
                .iter()
                .any(|path| archive.needs_materialization(path))
        {
            let sender = self.io_sender.clone();
            let repaint = self.repaint.clone();
            self.materializing = true;
            std::thread::spawn(move || {
                let result = archive
                    .materialize(&needed)
                    .map_err(|error| format!("{error:#}"));
                let _ = sender.send(IoMessage::Materialized(result));
                repaint.request_repaint();
            });
            return;
        }
        if artifact.is_none() && !media.path.is_file() {
            self.toast =
                "Missing source footage · right-click the media entry to reconnect it".into();
            return;
        }
        let request = preview::Request {
            key: key.clone(),
            media,
            frame,
            mode: actual_mode,
            artifact,
            artifact_is_straight_rgba: straight,
            overlay,
            overlay_remove,
            background: self.background,
            despill,
            masks,
        };
        if self.preview.request(request) {
            self.requested_key = Some(key);
        }
    }
    fn enqueue(&mut self, action: Action) {
        self.auto_cache = None;
        self.preview.clear_cache_queue();
        if self.project_io_busy || self.export.is_some() {
            self.toast = "Wait for the project or export task to finish.".into();
            return;
        }
        if self.current().is_none() {
            return;
        }
        if self.ui_only {
            self.toast = "Processing is disabled in this interface preview session.".into();
            return;
        }
        if matches!(action, Action::RenderAll)
            && !self
                .current()
                .expect("existing session")
                .project
                .scenes
                .iter()
                .any(|scene| scene.enabled)
        {
            self.fail("Enable at least one scene using its checkbox on the timeline.");
            return;
        }
        let scene_range = if matches!(action, Action::RenderScene) {
            match self
                .current()
                .expect("existing session")
                .current_scene_range()
            {
                Ok(range) => Some(range),
                Err(timeline::SceneRunError::Disabled(index)) => {
                    self.fail(format!(
                        "Scene {} is disabled. Enable it using its checkbox on the timeline.",
                        index + 1
                    ));
                    return;
                }
                Err(timeline::SceneRunError::NoScene) => {
                    self.fail("The playhead is not inside a scene.");
                    return;
                }
            }
        } else {
            None
        };
        let sam31 = matches!(
            action,
            Action::Segment | Action::Propagate | Action::PropagateAll
        ) && self.effective_sam_model() == SamModel::Sam31;
        let model_group = match action {
            Action::Detect => Some(AssetGroup::Transnet),
            // The SAM 3.1 pack is optional and not managed by Required files.
            Action::Segment | Action::Propagate | Action::PropagateAll if sam31 => None,
            Action::Segment | Action::Propagate | Action::PropagateAll => Some(AssetGroup::Sam),
            Action::RenderRange | Action::RenderScene | Action::RenderAll => {
                Some(AssetGroup::Ck1080)
            }
        };
        let mut groups = vec![AssetGroup::Core, AssetGroup::Runtime];
        groups.extend(model_group);
        if !self.assets.require(&groups) {
            self.toast = "Prepare the required files, then run this action again".into();
            return;
        }
        let session = self
            .current()
            .expect("existing session after asset preflight");
        if let Err(error) = self.runtime.validate_common() {
            self.assets.recheck();
            self.push_log(format!("Native runtime preflight: {error:#}"));
            self.fail("A required runtime file is unavailable. Recheck Required files to restore it automatically.");
            return;
        }
        let required_model = match action {
            Action::Detect => Some(("TransNetV2 model", self.runtime.transnet_model.clone())),
            Action::Segment | Action::Propagate | Action::PropagateAll if sam31 => Some((
                "SAM 3.1 pack",
                self.runtime.sam31_models().join("tracker/config.json"),
            )),
            Action::Segment | Action::Propagate | Action::PropagateAll => Some((
                "SAM video pack",
                self.runtime.sam_models.join("video_config.json"),
            )),
            Action::RenderRange | Action::RenderScene | Action::RenderAll => self
                .runtime
                .ck_models_for(session.media().width, session.media().height)
                .ok()
                .map(|path| ("CorridorKey model pack", path.join("stage1/model.onnx"))),
        };
        if let Some((label, path)) = required_model
            && !path.is_file()
        {
            self.assets.recheck();
            self.push_log(format!("{label} was not found: {}", path.display()));
            self.fail(format!(
                "{label} is unavailable. Recheck Required files to restore it automatically."
            ));
            return;
        }
        let session_id = session.id;
        let ranges = if matches!(action, Action::RenderAll) {
            session
                .project
                .scenes
                .iter()
                .filter(|scene| scene.enabled)
                .map(|scene| FrameRange {
                    start: scene.start,
                    end_exclusive: scene.end_exclusive,
                })
                .collect::<Vec<_>>()
        } else if let Some(range) = scene_range {
            vec![range]
        } else if matches!(action, Action::Propagate | Action::PropagateAll) {
            // Subject layers belong to a scene: propagate through the playhead's scene.
            let scene = session
                .project
                .scene_index_containing(session.playhead)
                .and_then(|index| session.project.scenes.get(index));
            vec![scene.map_or(session.range, |scene| FrameRange {
                start: scene.start,
                end_exclusive: scene.end_exclusive,
            })]
        } else if matches!(action, Action::Segment) {
            vec![FrameRange {
                start: session.playhead,
                end_exclusive: session.playhead + 1,
            }]
        } else {
            vec![session.range]
        };
        let media = session.media().clone();
        let prompts = session.project.prompts.clone();
        let boxes = session.project.sam_boxes.clone();
        let output_root = session.project.output_dir.clone();
        let quality = session.project.hint_quality.value();
        let sam_action = matches!(
            action,
            Action::Segment | Action::Propagate | Action::PropagateAll
        );
        let layer_text = |layer: u64| -> Option<String> {
            session
                .sam_layer(layer)
                .map(|l| l.prompt.trim().to_owned())
                .filter(|t| sam31 && !t.is_empty())
        };
        // SAM jobs run once per layer: the selected one, or every enabled layer with a start. SAM 3.1
        // starts from the earliest subject point/box anywhere in the range (tracking both ways) or
        // a text prompt; SAM 2.1 needs a subject point on the first frame of the range.
        let mut units: Vec<(FrameRange, Option<u64>)> = Vec::new();
        for range in ranges {
            if !sam_action {
                units.push((range, None));
                continue;
            }
            let starts = |layer: u64| {
                if sam31 {
                    prompts
                        .iter()
                        .any(|p| p.layer == layer && range.contains(p.frame) && p.positive)
                        || boxes
                            .iter()
                            .any(|b| b.layer == layer && range.contains(b.frame))
                        || layer_text(layer).is_some()
                } else {
                    prompts
                        .iter()
                        .any(|p| p.layer == layer && p.frame == range.start && p.positive)
                }
            };
            if matches!(action, Action::PropagateAll) {
                let scene = session.project.scene_of_frame(range.start);
                let ready: Vec<u64> = session
                    .project
                    .sam_layers
                    .iter()
                    .filter(|layer| layer.in_scene(scene) && layer.enabled && starts(layer.id))
                    .map(|layer| layer.id)
                    .collect();
                if ready.is_empty() {
                    self.fail(if sam31 {
                        "Add a subject point, a box or a text prompt to at least one enabled layer."
                            .to_owned()
                    } else {
                        format!(
                            "Add a subject point on frame {} in at least one enabled layer.",
                            range.start + 1
                        )
                    });
                    return;
                }
                units.extend(ready.into_iter().map(|layer| (range, Some(layer))));
            } else {
                let layer = session.sam_layer;
                if !starts(layer) {
                    let name = session
                        .sam_layer(layer)
                        .map_or("this layer", |l| l.name.as_str());
                    self.fail(if sam31 {
                        format!(
                            "Add a subject point, a box or a text prompt to {name} in this range."
                        )
                    } else {
                        format!(
                            "Add a positive point to {name} on frame {} to start this mask range.",
                            range.start + 1
                        )
                    });
                    return;
                }
                units.push((range, Some(layer)));
            }
        }
        let mut queue = vec![];
        for (range, sam_layer) in units {
            let mut hint_plan = None;
            let kind = match action {
                Action::Detect => JobKind::DetectScenes {
                    threshold: self.scene_threshold,
                    min_scene_frames: self.min_scene_frames,
                },
                Action::Segment | Action::Propagate | Action::PropagateAll => {
                    let layer = sam_layer.expect("SAM jobs name a layer");
                    let mut selected = prompts
                        .iter()
                        .filter(|point| {
                            point.layer == layer
                                && point.frame >= range.start
                                && point.frame < range.end_exclusive
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    let layer_boxes: Vec<ck_desktop_core::SamBox> = boxes
                        .iter()
                        .filter(|b| b.layer == layer && range.contains(b.frame))
                        .cloned()
                        .collect();
                    if !sam31 {
                        // SAM 2.1 has no text or box prompts: points stay plain points.
                        if !layer_boxes.is_empty() {
                            self.fail("Boxes need SAM 3.1. Switch the model or clear the boxes.");
                            return;
                        }
                        for point in &mut selected {
                            point.text = None;
                        }
                    }
                    JobKind::SamVideo {
                        range,
                        provider: self.provider,
                        prompts: selected,
                        model: self.effective_sam_model(),
                        boxes: layer_boxes,
                        text: layer_text(layer),
                    }
                }
                Action::RenderRange | Action::RenderScene | Action::RenderAll => {
                    // SAM hints are built from the enabled layers just before the job starts.
                    let frames: Vec<u64> = match self.hints {
                        HintMode::None | HintMode::External => Vec::new(),
                        HintMode::Video => (range.start..range.end_exclusive).collect(),
                        HintMode::First => vec![range.start],
                    };
                    let mut composites = Vec::new();
                    for frame in &frames {
                        let composite = session.frames.get(frame).map_or(
                            crate::sam_layers::Composite::None,
                            |files| {
                                crate::sam_layers::plan(
                                    &session.project.layers_for_frame(*frame),
                                    &files.sam_layers,
                                )
                            },
                        );
                        if composite == crate::sam_layers::Composite::None {
                            self.fail(if self.hints == HintMode::First {
                                format!(
                                    "Segment frame {} before using its mask as a hint.",
                                    range.start + 1
                                )
                            } else {
                                format!("Propagate SAM masks for frames {}–{} first, or select No mask.", range.start + 1, range.end_exclusive)
                            });
                            return;
                        }
                        composites.push((*frame, composite));
                    }
                    if !composites.is_empty() {
                        hint_plan = Some(HintPlan {
                            frames: composites,
                            directory: output_root
                                .join(format!("sam-hint-{}", project_store::unique_key())),
                            first_frame: self.hints == HintMode::First,
                            quality,
                        });
                    }
                    session.ck_job(range, self.provider, HintInput::None)
                }
            };
            let label = match action {
                Action::Detect => "Detect scenes".to_string(),
                Action::Segment | Action::Propagate | Action::PropagateAll => {
                    let name = sam_layer
                        .and_then(|layer| session.sam_layer(layer))
                        .map_or("SAM", |l| l.name.as_str());
                    if matches!(action, Action::Segment) {
                        format!("Segment frame · {name}")
                    } else {
                        format!("Propagate SAM · {name}")
                    }
                }
                _ => "CorridorKey render".to_string(),
            };
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros();
            let id = format!(
                "{}-{stamp}-{}-{}{}",
                match action {
                    Action::Detect => "scenes",
                    Action::Segment | Action::Propagate | Action::PropagateAll => "sam",
                    _ => "ck",
                },
                range.start,
                range.end_exclusive,
                sam_layer.map_or(String::new(), |layer| format!("-l{layer}"))
            );
            let request = JobRequest {
                schema_version: 1,
                job_id: id.clone(),
                media: media.clone(),
                output_dir: output_root.join(id),
                runtime: self.runtime.clone(),
                kind,
            };
            if self.archive.is_none()
                && let Err(error) = request.validate()
            {
                self.fail(format!("{error:#}"));
                return;
            }
            queue.push(QueuedJob {
                request,
                label,
                session_id,
                range,
                device_id: self.device_id,
                sam_layer,
                hint_plan,
            });
        }
        // SAM 3.1's text encoder understands English only: send translated text prompts.
        if self.translator.available() {
            for job in &mut queue {
                if let JobKind::SamVideo {
                    model: SamModel::Sam31,
                    text,
                    prompts,
                    ..
                } = &mut job.request.kind
                {
                    if let Some(text) = text.as_mut() {
                        *text = self.translator.english_for_job(text);
                    }
                    for point in prompts.iter_mut() {
                        if let Some(text) = point.text.as_mut() {
                            *text = self.translator.english_for_job(text);
                        }
                    }
                }
            }
        }
        let external = self.hints == HintMode::External
            && matches!(
                action,
                Action::RenderRange | Action::RenderScene | Action::RenderAll
            );
        let layered_hints = queue.iter().any(|job| job.hint_plan.is_some());
        if external || layered_hints || self.archive.is_some() {
            let hint = if external {
                self.current().and_then(|session| session.hint.clone())
            } else {
                None
            };
            if external && hint.is_none() {
                self.fail("Add hint footage using the media item's right-click menu.");
                return;
            }
            let start = queue.iter().map(|job| job.range.start).min().unwrap_or(0);
            let end_exclusive = queue
                .iter()
                .map(|job| job.range.end_exclusive)
                .max()
                .unwrap_or(1);
            let root = output_root.join(format!("external-hint-{}", project_store::unique_key()));
            let sender = self.io_sender.clone();
            let runtime = self.runtime.clone();
            let archive = self.archive.clone();
            let lease = self.temporary.clone();
            let repaint = self.repaint.clone();
            self.project_io_busy = true;
            self.toast = "Preparing project media…".into();
            std::thread::spawn(move || {
                let _lease = lease;
                let result = (|| -> anyhow::Result<Option<HintInput>> {
                    let mut paths = vec![media.path.clone()];
                    if let Some(hint) = &hint {
                        paths.push(hint.path.clone());
                    }
                    for job in &queue {
                        if let JobKind::CorridorKey { hints, .. } = &job.request.kind {
                            match hints {
                                HintInput::FirstFrame { mask_file, .. } => {
                                    paths.push(mask_file.clone())
                                }
                                HintInput::Video { mask_directory, .. } => {
                                    paths.push(mask_directory.clone())
                                }
                                HintInput::None => {}
                            }
                        }
                    }
                    for job in &queue {
                        if let Some(plan) = &job.hint_plan {
                            for (_, composite) in &plan.frames {
                                match composite {
                                    crate::sam_layers::Composite::None => {}
                                    crate::sam_layers::Composite::Single(path) => {
                                        paths.push(path.clone())
                                    }
                                    crate::sam_layers::Composite::Combine(inputs) => {
                                        paths.extend(inputs.iter().map(|(p, _)| p.clone()))
                                    }
                                }
                            }
                        }
                    }
                    if let Some(archive) = &archive {
                        archive.materialize(&paths)?;
                    }
                    // Build each job's SAM hint from its layers (combined masks per frame).
                    for job in &mut queue {
                        if let Some(plan) = job.hint_plan.take()
                            && let JobKind::CorridorKey { hints, .. } = &mut job.request.kind
                        {
                            *hints = plan.build()?;
                        }
                    }
                    let mut media = media;
                    refresh_embedded_identity(&mut media, archive.as_deref())?;
                    if let Some(hint) = hint {
                        crate::media_tasks::prepare_external_hint(
                            &hint,
                            &media,
                            FrameRange {
                                start,
                                end_exclusive,
                            },
                            &root,
                            &runtime.ffmpeg,
                        )
                        .map(Some)
                    } else {
                        Ok(None)
                    }
                })()
                .map_err(|error| format!("{error:#}"));
                let _ = sender.send(IoMessage::HintPrepared(queue, result));
                repaint.request_repaint();
            });
        } else {
            for job in queue {
                self.jobs.push_back(job);
            }
        }
        self.playing = false;
    }

    fn poll_jobs(&mut self) {
        if self.active.is_none()
            && let Some(queued) = self.jobs.pop_front()
        {
            match Worker::spawn_on_device(&queued.request, queued.device_id) {
                Ok(worker) => {
                    self.push_log(format!(
                        "{}: {}–{}",
                        queued.label,
                        queued.range.start + 1,
                        queued.range.end_exclusive
                    ));
                    self.active = Some(ActiveJob {
                        worker,
                        queued,
                        progress: Default::default(),
                        message: String::new(),
                        policy_message: None,
                        memory_retry_message: None,
                        rendered_frames: BTreeSet::new(),
                    });
                }
                Err(error) => self.fail(format!("{error:#}")),
            }
        }
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let events = match active.worker.poll() {
            Ok(events) => events,
            Err(error) => {
                self.fail(format!("{error:#}"));
                self.active = None;
                return;
            }
        };
        for event in events {
            if self
                .active
                .as_ref()
                .is_none_or(|active| event.job_id() != active.queued.request.job_id)
            {
                continue;
            }
            match event {
                JobEvent::Started { .. } => {}
                JobEvent::Progress {
                    phase,
                    current,
                    total,
                    message,
                    ..
                } => {
                    if phase == "memory_retry" {
                        self.discard_failed_attempt();
                    }
                    if matches!(phase.as_str(), "window_policy" | "memory_retry") {
                        self.push_log(message.clone());
                    }
                    if let Some(active) = self.active.as_mut() {
                        match phase.as_str() {
                            "window_policy" => active.policy_message = Some(message.clone()),
                            "memory_retry" => active.memory_retry_message = Some(message.clone()),
                            _ => {}
                        }
                        active.progress.observe(&phase, current, total, &message);
                        active.message = message;
                    }
                }
                JobEvent::FrameReady {
                    frame,
                    rgba_path,
                    checker_path,
                    alpha_path,
                    mask_path,
                    ..
                } => {
                    if (rgba_path.is_some() || alpha_path.is_some() || checker_path.is_some())
                        && let Some(active) = self.active.as_mut()
                    {
                        active.rendered_frames.insert(frame);
                    }
                    if let Some(active) = self.active.as_ref()
                        && let Some(session) = self
                            .sessions
                            .iter_mut()
                            .find(|session| session.id == active.queued.session_id)
                    {
                        let entry = session.frames.entry(frame).or_default();
                        if rgba_path.is_some() {
                            entry.rgba = rgba_path;
                            // A recalculated frame must not keep showing its old RAM copy.
                            self.result_cache.remove_frame(frame);
                            if let Some(state) = self.auto_cache.as_mut() {
                                state.cached.remove(&frame);
                                state.failed.remove(&frame);
                            }
                        }
                        if checker_path.is_some() {
                            entry.checker = checker_path;
                        }
                        if alpha_path.is_some() {
                            entry.alpha = alpha_path;
                        }
                        if let Some(mask) = mask_path {
                            if let Some(layer) = active.queued.sam_layer {
                                entry.sam_layers.insert(layer, mask);
                                entry.mask = None; // rebuilt from the layers when shown
                            } else {
                                entry.mask = Some(mask);
                            }
                        }
                        if frame == session.playhead {
                            session.revision += 1;
                        }
                    }
                }
                JobEvent::Completed { artifacts, .. } => {
                    if let Some(active) = self.active.take() {
                        self.accept_artifacts(
                            active.queued.session_id,
                            &active.queued.range,
                            &artifacts,
                            active.queued.sam_layer,
                        );
                        self.toast = format!("{} complete", active.queued.label);
                        self.push_log(self.toast.clone());
                        self.calculated_dirty = true;
                        self.queue_auto_result_cache(active.queued.session_id, active.queued.range);
                    }
                }
                JobEvent::Failed { message, .. } => {
                    self.calculated_dirty |= self
                        .active
                        .as_ref()
                        .is_some_and(|active| !active.rendered_frames.is_empty());
                    let directml_failed = self.active.as_ref().is_some_and(|active| {
                        matches!(
                            active.queued.request.kind,
                            JobKind::CorridorKey {
                                provider: Provider::DirectMl,
                                ..
                            } | JobKind::SamVideo {
                                provider: Provider::DirectMl,
                                ..
                            }
                        )
                    });
                    self.active = None;
                    self.jobs.clear();
                    self.fail(message);
                    self.can_select_cpu = directml_failed;
                }
                JobEvent::Cancelled { .. } => {
                    self.active = None;
                    self.jobs.clear();
                    self.toast = "Job cancelled; completed frames remain available".into();
                    self.calculated_dirty = true;
                }
            }
        }
    }

    fn accept_artifacts(
        &mut self,
        session_id: u64,
        range: &FrameRange,
        artifacts: &JobArtifacts,
        sam_layer: Option<u64>,
    ) {
        if !artifacts.scenes.is_empty() {
            self.record_app_edit(crate::app_history::AppContext::Timeline, "scene detection");
        }
        let visible = self
            .current()
            .is_some_and(|session| session.id == session_id);
        let Some(session) = self
            .sessions
            .iter_mut()
            .find(|session| session.id == session_id)
        else {
            return;
        };
        if !artifacts.scenes.is_empty() {
            session.apply_detected_scenes(artifacts.scenes.clone());
        }
        for frame in range.start..range.end_exclusive {
            let files = session.frames.entry(frame).or_default();
            for (directory, target) in [
                (&artifacts.frames_dir, &mut files.rgba),
                (&artifacts.checker_dir, &mut files.checker),
                (&artifacts.alpha_dir, &mut files.alpha),
            ] {
                if let Some(directory) = directory {
                    let path = directory.join(format!("{frame:06}.png"));
                    if path.is_file() {
                        *target = Some(path);
                    }
                }
            }
            if let Some(directory) = &artifacts.masks_dir {
                let path = directory.join(format!("{frame:06}.png"));
                if path.is_file() {
                    match sam_layer {
                        Some(layer) => {
                            files.sam_layers.insert(layer, path);
                            files.mask = None;
                        }
                        None => files.mask = Some(path),
                    }
                }
            }
        }
        if artifacts.masks_dir.is_some() {
            // The view stays where the user put it; only offer the new masks as a hint.
            if visible {
                if range.len() > 1 {
                    self.hints = HintMode::Video;
                } else if self.hints == HintMode::None {
                    self.hints = HintMode::First;
                }
            }
        } else if visible && artifacts.frames_dir.is_some() {
            self.mode = ViewMode::Composite;
        }
        session.revision += 1;
    }

    fn cancel(&mut self) {
        self.jobs.clear();
        if let Some(active) = self.active.as_mut() {
            if let Err(error) = active.worker.cancel() {
                self.fail(format!("Cancellation failed: {error:#}"));
            } else {
                self.toast = "Cancelling native job…".into();
            }
        }
    }

    fn export_frame(&mut self) {
        if self.project_io_busy || self.playing {
            return;
        }
        if self.mode == ViewMode::Sam {
            let archive = self.archive.clone();
            if let Some(session) = self.current_mut() {
                let frame = session.playhead;
                session.subject_mask(frame, archive.as_deref());
            }
        }
        let Some(session) = self.current() else {
            return;
        };
        let media = session.media().clone();
        let frame = session.playhead;
        let mode = self.mode;
        let despill = session.project.despill_settings_for_frame(frame);
        let masks = session.mask_application(frame);
        let artifact = session.frames.get(&frame).and_then(|files| match mode {
            ViewMode::Composite => files.rgba.clone(),
            ViewMode::Alpha => files.alpha.clone(),
            ViewMode::Sam => files.mask.clone(),
            ViewMode::Source => None,
        });
        let Some(target) = rfd::FileDialog::new()
            .set_title("Export current frame")
            .add_filter("PNG", &["png"])
            .set_file_name(format!("frame-{:06}.png", frame + 1))
            .save_file()
        else {
            return;
        };
        let sender = self.io_sender.clone();
        let ffmpeg = self.runtime.ffmpeg.clone();
        let archive = self.archive.clone();
        let lease = self.temporary.clone();
        let repaint = self.repaint.clone();
        self.project_io_busy = true;
        std::thread::spawn(move || {
            let _lease = lease;
            let result = (|| -> anyhow::Result<PathBuf> {
                let needs_source =
                    artifact.is_none() || mode == ViewMode::Composite && despill.is_enabled();
                if let Some(archive) = &archive {
                    let mut paths = vec![artifact.clone().unwrap_or_else(|| media.path.clone())];
                    if needs_source {
                        paths.push(media.path.clone());
                    }
                    archive.materialize(&paths)?;
                }
                let mut media = media;
                if needs_source {
                    refresh_embedded_identity(&mut media, archive.as_deref())?;
                }
                crate::media_tasks::export_single_frame_masked(
                    &media,
                    frame,
                    mode,
                    artifact,
                    &target,
                    &ffmpeg,
                    crate::media_tasks::ExportComposition {
                        settings: despill,
                        masks: masks.as_deref(),
                    },
                )?;
                Ok(target)
            })()
            .map_err(|error| format!("{error:#}"));
            let _ = sender.send(IoMessage::FrameExported(result));
            repaint.request_repaint();
        });
    }

    fn export_window(&mut self, ctx: &egui::Context) {
        self.export_progress_window(ctx);
        if !self.export_dialog {
            return;
        }
        let mut open = true;
        let mut start = false;
        egui::Window::new("Export playback range").open(&mut open).resizable(false).default_width(400.0).show(ctx, |ui| {
            if let Some(session) = self.current() {
                ui.label(format!("Frames {}–{} · {} frames", session.range.start + 1, session.range.end_exclusive, session.range.len()));
                ui.label("Rendered frames use their alpha; unrendered frames use the original footage.");
            }
            ui.add_space(8.0);
            for format in ExportFormat::ALL { ui.radio_value(&mut self.export_format, format, format.label()); }
            ui.add_space(8.0);
            start = ui.add_enabled(self.current().is_some() && self.export.is_none() && !self.project_io_busy, egui::Button::new("Choose destination and export…")).clicked();
        });
        self.export_dialog = open;
        if start {
            let Some(session) = self.current() else {
                return;
            };
            let format = self.export_format;
            let target = if format.is_sequence() {
                rfd::FileDialog::new()
                    .set_title("Choose an empty sequence folder")
                    .pick_folder()
            } else {
                rfd::FileDialog::new()
                    .set_title("Export ProRes 4444 alpha")
                    .add_filter("QuickTime", &["mov"])
                    .set_file_name("output.mov")
                    .save_file()
            };
            let Some(target) = target else {
                return;
            };
            let request = ExportRequest {
                media: session.media().clone(),
                range: session.range,
                rendered: session
                    .frames
                    .range(session.range.start..session.range.end_exclusive)
                    .filter_map(|(&frame, files)| files.rgba.clone().map(|path| (frame, path)))
                    .collect(),
                target,
                format,
                despill: session.project.despill_settings,
                despills: (session.range.start..session.range.end_exclusive)
                    .map(|frame| (frame, session.project.despill_settings_for_frame(frame)))
                    .collect(),
                masks: Some(std::sync::Arc::new(session.mask_stacks.clone())),
            };
            self.export_dialog = false;
            self.playing = false;
            if let Some(archive) = self.archive.clone() {
                let sender = self.io_sender.clone();
                let repaint = self.repaint.clone();
                let mut request = request;
                self.project_io_busy = true;
                std::thread::spawn(move || {
                    let result = (|| -> anyhow::Result<()> {
                        let mut paths: Vec<_> = request.rendered.values().cloned().collect();
                        let needs_source = request.rendered.len() as u64 != request.range.len()
                            || request.despill.is_enabled();
                        if needs_source {
                            paths.push(request.media.path.clone());
                        }
                        archive.materialize(&paths)?;
                        if needs_source {
                            refresh_embedded_identity(&mut request.media, Some(&archive))?;
                        }
                        Ok(())
                    })()
                    .map_err(|error| format!("{error:#}"));
                    let _ = sender.send(IoMessage::ExportReady(request, result));
                    repaint.request_repaint();
                });
            } else {
                self.start_export(request, ctx);
            }
        }
    }

    fn export_progress_window(&self, ctx: &egui::Context) {
        if let Some(service) = &self.export {
            egui::Window::new("Exporting playback range")
                .resizable(false)
                .show(ctx, |ui| {
                    let (current, total) = self.export_progress;
                    ui.add(
                        egui::ProgressBar::new(if total == 0 {
                            0.0
                        } else {
                            current as f32 / total as f32
                        })
                        .fill(ACCENT)
                        .desired_width(300.0)
                        .show_percentage(),
                    );
                    ui.label(format!("{current} / {total} frames"));
                    if ui.button("Cancel export").clicked() {
                        service.cancel();
                    }
                });
        }
    }

    fn start_export(&mut self, request: ExportRequest, ctx: &egui::Context) {
        self.export_progress = (0, request.range.len());
        match ExportService::start(request, self.runtime.ffmpeg.clone(), ctx.clone()) {
            Ok(service) => {
                self.export = Some(service);
                self.export_lease = Some(self.temporary.clone());
                self.toast = "Exporting playback range…".into();
            }
            Err(error) => self.fail(format!("{error:#}")),
        }
    }

    fn poll_export(&mut self) {
        loop {
            let event = self
                .export
                .as_ref()
                .and_then(|service| service.messages.try_recv().ok());
            let Some(event) = event else {
                break;
            };
            match event {
                ExportEvent::Progress { current, total } => self.export_progress = (current, total),
                ExportEvent::Finished {
                    target,
                    frames,
                    rendered_frames,
                    source_frames,
                } => {
                    self.export.take();
                    self.export_lease.take();
                    self.toast = format!(
                        "Exported {frames} frames ({rendered_frames} rendered, {source_frames} original) to {}",
                        display_path(&target)
                    );
                }
                ExportEvent::Failed(error) => {
                    self.export.take();
                    self.export_lease.take();
                    self.fail(error);
                }
                ExportEvent::Cancelled => {
                    self.export.take();
                    self.export_lease.take();
                    self.toast = "Export cancelled".into();
                }
            }
        }
    }
    fn discard_failed_attempt(&mut self) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        let session_id = active.queued.session_id;
        let rendered_frames = std::mem::take(&mut active.rendered_frames);
        active.policy_message = None;
        if let Some(session) = self
            .sessions
            .iter_mut()
            .find(|session| session.id == session_id)
        {
            session.discard_render_attempt(&rendered_frames);
        }
        if self
            .current()
            .is_some_and(|session| session.id == session_id)
        {
            self.loaded = None;
            self.requested_key = None;
        }
    }

    fn project_snapshot(&self) -> ChromaProject {
        ChromaProject {
            chroma_version: 1,
            selected: self.selected,
            archive: self.archive.clone(),
            documents: self
                .sessions
                .iter()
                .map(|session| {
                    let mut project = session.project.clone();
                    project.set_mask_stacks(&session.mask_stacks);
                    project.sanitize_scene_overrides();
                    Document {
                        key: session.key.clone(),
                        project,
                        range: session.range,
                        playhead: session.playhead,
                        frames: session.frames.clone(),
                        masks: session.masks.clone(),
                        hint: session.hint.clone(),
                    }
                })
                .collect(),
        }
    }

    fn save_project(&mut self) {
        if let Some(path) = self.project_path.clone() {
            self.save_project_to(path);
        } else {
            self.save_project_as();
        }
    }

    fn save_project_as(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .set_title("Save Chroma project")
            .add_filter("Chroma project", &["chroma"])
            .set_file_name("project.chroma")
            .save_file()
        {
            self.save_project_to(path.with_extension("chroma"));
        }
    }

    fn save_project_to(&mut self, path: PathBuf) {
        if self.project_io_busy || self.active.is_some() || !self.jobs.is_empty() {
            self.toast = "Finish calculations before saving the project.".into();
            return;
        }
        if self.sessions.is_empty() {
            return;
        }
        let snapshot = self.project_snapshot();
        let embed_sources = self.embed_sources;
        let versions = self
            .sessions
            .iter()
            .map(|s| (s.key.clone(), s.revision))
            .collect();
        let sender = self.io_sender.clone();
        let lease = self.temporary.clone();
        self.project_io_busy = true;
        self.calculated_dirty = false;
        self.toast = "Saving project and calculated segments…".into();
        std::thread::spawn(move || {
            let _lease = lease;
            let result = project_store::save(&path, snapshot, embed_sources)
                .map_err(|error| format!("{error:#}"));
            let _ = sender.send(IoMessage::Saved(path, versions, result));
        });
    }

    fn load_project(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .set_title("Open Chroma project")
            .add_filter("Chroma projects", &["chroma", "brpnative"])
            .pick_file()
        {
            self.load_project_path(path);
        }
    }

    fn load_project_path(&mut self, path: PathBuf) {
        if self.active.is_some()
            || !self.jobs.is_empty()
            || self.project_io_busy
            || self.export.is_some()
        {
            self.fail("Finish the current job before opening another project.");
            return;
        }
        let sender = self.io_sender.clone();
        self.project_io_busy = true;
        self.toast = "Opening project index…".into();
        std::thread::spawn(move || {
            let result = project_store::load(&path).map_err(|error| format!("{error:#}"));
            let _ = sender.send(IoMessage::Loaded(path, result));
        });
    }

    fn poll_io(&mut self) {
        while let Ok(message) = self.io_messages.try_recv() {
            match message {
                IoMessage::Loaded(path, result) => {
                    self.project_io_busy = false;
                    match result {
                        Ok(project) => {
                            // Native FFmpeg may hold files inside the previous
                            // archive's temporary cache. Close it before that
                            // archive's last lease can be released.
                            self.preview.shutdown();
                            self.preview = PreviewService::new(
                                self.runtime.clone(),
                                self.work_root.join("preview"),
                                self.repaint.clone(),
                            );
                            self.archive = project.archive.clone();
                            self.calculated_dirty = false;
                            self.materialize_failed = false;
                            self.sessions = project
                                .documents
                                .into_iter()
                                .map(|document| {
                                    let mut session = Session::new(document.project);
                                    session.key = document.key;
                                    session.range = document.range;
                                    session.playhead = document.playhead;
                                    session.frames = document.frames;
                                    session.masks = document.masks;
                                    session.migrate_sam_layers();
                                    session.hint = document.hint;
                                    // A short work range would be a sliver of a long clip;
                                    // open framed on it (wheel/middle-drag still navigate).
                                    let frames = session.media().frame_count;
                                    if session.range.len() * 2 < frames {
                                        session.timeline_view.frame_range(
                                            frames,
                                            session.range.start,
                                            session.range.end_exclusive,
                                        );
                                    }
                                    session
                                })
                                .collect();
                            self.selected = project.selected;
                            self.project_path = Some(path);
                            self.loaded = None;
                            self.requested_key = None;
                            self.playing = false;
                            self.zoom = ViewerZoom::default();
                            self.pan = Vec2::ZERO;
                            self.mode = ViewMode::Composite;
                            self.toast = "Project opened · media is loaded on demand".into();
                        }
                        Err(error) => self.fail(error),
                    }
                }
                IoMessage::Saved(path, versions, result) => {
                    self.project_io_busy = false;
                    match result {
                        Ok(project) => {
                            self.archive = project.archive.clone();
                            self.materialize_failed = false;
                            for document in project.documents {
                                if let Some(session) =
                                    self.sessions.iter_mut().find(|s| s.key == document.key)
                                {
                                    session.project.output_dir = document.project.output_dir;
                                    session.project.media = document.project.media;
                                    session.hint = document.hint;
                                    if versions.iter().any(|(key, revision)| {
                                        *key == session.key && *revision == session.revision
                                    }) {
                                        session.frames = document.frames;
                                        session.masks = document.masks;
                                    }
                                }
                            }
                            self.project_path = Some(path);
                            self.toast = "Project and calculated segments saved".into();
                        }
                        Err(error) => self.fail(error),
                    }
                }
                IoMessage::Relinked(id, result) => {
                    self.project_io_busy = false;
                    match result {
                        Ok(media) => {
                            if let Some(session) = self.sessions.iter_mut().find(|s| s.id == id) {
                                let compatible = project_store::source_metadata_compatible(
                                    session.media(),
                                    &media,
                                );
                                let identity_matches =
                                    session.media().source_identity == media.source_identity;
                                session.refresh_source(media);
                                if !compatible || !identity_matches {
                                    session.frames.clear();
                                    session.masks.clear();
                                }
                                self.loaded = None;
                                self.requested_key = None;
                                self.toast = if compatible && identity_matches {
                                    "Footage reconnected"
                                } else {
                                    "Footage replaced; stale calculations cleared"
                                }
                                .into();
                            }
                        }
                        Err(error) => self.fail(error),
                    }
                }
                IoMessage::Hint(id, result) => {
                    self.project_io_busy = false;
                    match result {
                        Ok(hint) => {
                            if let Some(session) = self.sessions.iter_mut().find(|s| s.id == id) {
                                session.hint = Some(hint);
                                session.revision += 1;
                            }
                            self.hints = HintMode::External;
                            self.toast = "Hint footage connected".into();
                        }
                        Err(error) => self.fail(error),
                    }
                }
                IoMessage::HintPrepared(mut queue, result) => {
                    self.project_io_busy = false;
                    match result {
                        Ok(hint) => {
                            for job in &mut queue {
                                if let Some(hint) = &hint
                                    && let JobKind::CorridorKey { hints, .. } =
                                        &mut job.request.kind
                                {
                                    *hints = hint.clone();
                                    if let HintInput::FirstFrame { frame, .. } = hints {
                                        *frame = job.range.start;
                                    }
                                    // External footage uses the project's hint quality too.
                                    let chosen = self
                                        .sessions
                                        .iter()
                                        .find(|s| s.id == job.session_id)
                                        .map(|s| s.project.hint_quality.value());
                                    if let Some(chosen) = chosen
                                        && let HintInput::FirstFrame { quality, .. }
                                        | HintInput::Video { quality, .. } = hints
                                    {
                                        *quality = chosen;
                                    }
                                }
                                if let Err(error) = refresh_embedded_identity(
                                    &mut job.request.media,
                                    self.archive.as_deref(),
                                ) {
                                    self.fail(format!("{error:#}"));
                                    return;
                                }
                                if let Err(error) = job.request.validate() {
                                    self.fail(format!("{error:#}"));
                                    return;
                                }
                            }
                            self.jobs.extend(queue);
                            self.toast = "Calculations queued".into();
                        }
                        Err(error) => self.fail(error),
                    }
                }
                IoMessage::FrameExported(result) => {
                    self.project_io_busy = false;
                    match result {
                        Ok(path) => self.toast = format!("Exported {}", display_path(&path)),
                        Err(error) => self.fail(error),
                    }
                }
                IoMessage::Materialized(result) => {
                    self.materializing = false;
                    match result {
                        Ok(()) => {
                            if let Some(archive) = &self.archive {
                                for session in &mut self.sessions {
                                    if let Some(media) = &mut session.project.media
                                        && let Err(error) =
                                            refresh_embedded_identity(media, Some(archive))
                                    {
                                        self.materialize_failed = true;
                                        self.fail(format!("{error:#}"));
                                        return;
                                    }
                                }
                            }
                            self.requested_key = None;
                        }
                        Err(error) => {
                            self.materialize_failed = true;
                            self.fail(error);
                        }
                    }
                }
                IoMessage::ExportReady(request, result) => {
                    self.project_io_busy = false;
                    match result {
                        Ok(()) => {
                            let ctx = self.repaint.clone();
                            self.start_export(request, &ctx);
                        }
                        Err(error) => self.fail(error),
                    }
                }
            }
        }
    }

    fn reconnect_media(&mut self, index: usize) {
        let Some(id) = self.sessions.get(index).map(|session| session.id) else {
            return;
        };
        if let Some(path) = rfd::FileDialog::new()
            .set_title("Find source footage")
            .add_filter("Video", &["mp4", "mov", "mkv", "avi", "mxf", "webm"])
            .pick_file()
        {
            let sender = self.io_sender.clone();
            let runtime = self.runtime.clone();
            self.project_io_busy = true;
            std::thread::spawn(move || {
                let _ = sender.send(IoMessage::Relinked(
                    id,
                    MediaInfo::probe(path, &runtime).map_err(|e| format!("{e:#}")),
                ));
            });
        }
    }

    fn add_hint_footage(&mut self, index: usize) {
        let Some(id) = self.sessions.get(index).map(|session| session.id) else {
            return;
        };
        if let Some(path) = rfd::FileDialog::new()
            .set_title("Add hint footage or mask image")
            .add_filter(
                "Video / image",
                &["mp4", "mov", "mkv", "avi", "png", "exr", "jpg", "jpeg"],
            )
            .pick_file()
        {
            let sender = self.io_sender.clone();
            let runtime = self.runtime.clone();
            self.project_io_busy = true;
            std::thread::spawn(move || {
                let first_frame_only = path.extension().is_some_and(|extension| {
                    ["png", "exr", "jpg", "jpeg"]
                        .iter()
                        .any(|candidate| extension.eq_ignore_ascii_case(candidate))
                });
                let result = if first_frame_only {
                    Ok(HintFootage {
                        path,
                        first_frame_only,
                    })
                } else {
                    MediaInfo::probe(&path, &runtime)
                        .map(|_| HintFootage {
                            path,
                            first_frame_only,
                        })
                        .map_err(|e| format!("{e:#}"))
                };
                let _ = sender.send(IoMessage::Hint(id, result));
            });
        }
    }
    /// Custom window title bar (the native frame is disabled in `main`): brand, project
    /// name and toolbar controls, window buttons on the right. Empty bar space drags the
    /// window; double-click toggles maximise.
    fn toolbar(&mut self, ctx: &egui::Context) {
        let width = ctx.content_rect().width();
        // Icon-only secondary buttons below 1600 pt; a second row only when even the
        // compact controls cannot share one row with the window buttons.
        let compact = width < 1600.0;
        let narrow = width < 900.0;
        let maximized = ctx.input(|input| input.viewport().maximized.unwrap_or(false));
        egui::TopBottomPanel::top("studio_toolbar")
            .exact_height(if narrow { 80.0 } else { TITLE_BAR_HEIGHT })
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_APP)
                    .inner_margin(egui::Margin {
                        left: 8,
                        right: 0,
                        top: 0,
                        bottom: 0,
                    }),
            )
            .show_separator_line(true)
            .show(ctx, |ui| {
                // Added first so every control drawn later takes the pointer instead.
                let drag = ui.interact(
                    ui.max_rect(),
                    egui::Id::new("title_bar_drag"),
                    egui::Sense::click_and_drag(),
                );
                if drag.drag_started_by(egui::PointerButton::Primary) {
                    ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                }
                if drag.double_clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized));
                }
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), TITLE_BAR_HEIGHT),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        self.toolbar_open_controls(ui, compact);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            window_buttons(ui, maximized);
                            ui.add_space(6.0);
                            self.updater.title_button(ui);
                            if !narrow {
                                self.toolbar_output_controls(ui, compact);
                            }
                            self.title_project_name(ui);
                        });
                    },
                );
                if narrow {
                    ui.allocate_ui_with_layout(
                        Vec2::new(ui.available_width() - 8.0, 36.0),
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| self.toolbar_output_controls(ui, true),
                    );
                }
            });
    }

    /// Muted project/clip name in the title bar's free space (it also keeps dragging).
    fn title_project_name(&self, ui: &mut egui::Ui) {
        let name = self
            .project_path
            .as_ref()
            .and_then(|path| path.file_name())
            .map(|name| name.to_string_lossy().into_owned())
            .or_else(|| {
                self.current()
                    .and_then(|session| session.media().path.file_name())
                    .map(|name| name.to_string_lossy().into_owned())
            });
        if let Some(name) = name {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                ui.add_space(12.0);
                ui.add(
                    egui::Label::new(RichText::new(name).color(MUTED))
                        .truncate()
                        .selectable(false)
                        .sense(egui::Sense::hover()),
                );
            });
        }
    }

    /// Resize from the window edges while the native frame is hidden. Runs after all
    /// panels so the edge cursor wins over widgets underneath it.
    fn window_edge_resize(&self, ctx: &egui::Context) {
        let (maximized, fullscreen) = ctx.input(|input| {
            let viewport = input.viewport();
            (
                viewport.maximized.unwrap_or(false),
                viewport.fullscreen.unwrap_or(false),
            )
        });
        if maximized || fullscreen {
            return;
        }
        let Some(pointer) = ctx.input(|input| input.pointer.hover_pos()) else {
            return;
        };
        let Some(direction) = edge_direction(ctx.content_rect(), pointer) else {
            return;
        };
        use egui::viewport::ResizeDirection as Dir;
        ctx.set_cursor_icon(match direction {
            Dir::North | Dir::South => egui::CursorIcon::ResizeVertical,
            Dir::East | Dir::West => egui::CursorIcon::ResizeHorizontal,
            Dir::NorthWest | Dir::SouthEast => egui::CursorIcon::ResizeNwSe,
            Dir::NorthEast | Dir::SouthWest => egui::CursorIcon::ResizeNeSw,
        });
        if ctx.input(|input| input.pointer.primary_pressed()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::BeginResize(direction));
        }
    }

    fn toolbar_open_controls(&mut self, ui: &mut egui::Ui, _compact: bool) {
        ui.add(
            egui::Label::new(
                RichText::new("BRP CHROMA")
                    .size(15.0)
                    .strong()
                    .color(theme::TEXT),
            )
            .selectable(false)
            .sense(egui::Sense::hover()),
        );
        ui.add_space(6.0);
        ui.spacing_mut().item_spacing.x = 4.0;
        if ui
            .add(self.icons.tool_button(Icon::OpenMedia))
            .on_hover_text("Open media (Ctrl+O)")
            .clicked()
        {
            self.open_footage();
        }
        if ui
            .add(self.icons.tool_button(Icon::OpenProject))
            .on_hover_text("Open a .chroma project")
            .clicked()
        {
            self.load_project();
        }
        let save = ui
            .add_enabled(
                self.current().is_some() && !self.project_io_busy,
                self.icons.tool_button(Icon::SaveProject),
            )
            .on_hover_text("Save project (Ctrl+S) · right-click: Save as…");
        if save.clicked() {
            self.save_project();
        }
        save.context_menu(|ui| {
            if ui.button("Save project as…").clicked() {
                self.save_project_as();
                ui.close();
            }
        });
        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);
        if ui
            .add_enabled(
                self.current().is_some() && self.active.is_none() && self.jobs.is_empty(),
                self.icons.tool_button(Icon::DetectScenes),
            )
            .on_hover_text("Detect scenes with TransNetV2 (Ctrl+D)")
            .clicked()
        {
            self.enqueue(Action::Detect);
        }
        ui.spacing_mut().item_spacing.x = 8.0;
    }

    fn toolbar_output_controls(&mut self, ui: &mut egui::Ui, compact: bool) {
        ui.spacing_mut().item_spacing.x = 4.0;
        if ui
            .add(self.icons.tool_button(Icon::Settings))
            .on_hover_text("Settings")
            .clicked()
        {
            self.open_settings = true;
        }
        // During playback this stays consistently disabled, instead of flashing
        // each time the decoder and playhead briefly agree on one frame.
        if ui
            .add_enabled(
                self.current().is_some() && !self.playing && self.export.is_none(),
                self.icons.tool_button(Icon::ExportFrame),
            )
            .on_hover_text("Export the current frame as PNG")
            .clicked()
        {
            self.export_frame();
        }
        if ui
            .add_enabled(
                self.current().is_some() && self.export.is_none(),
                self.icons.tool_button(Icon::ExportRange),
            )
            .on_hover_text("Export the In/Out range: ProRes 4444 alpha, PNG or EXR (Ctrl+E)")
            .clicked()
        {
            self.export_dialog = true;
        }
        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);
        ui.spacing_mut().item_spacing.x = 8.0;
        let previous = self.provider;
        let device_text = if self.provider == Provider::Cpu {
            "System CPU".to_owned()
        } else if self.provider == Provider::Cuda {
            format!("CUDA ordinal {}", self.device_id)
        } else {
            self.devices
                .iter()
                .find(|device| device.id == self.device_id)
                .map_or_else(
                    || format!("GPU {}", self.device_id),
                    |device| format!("{} · {}", device.id, device.name),
                )
        };
        egui::ComboBox::from_id_salt("compute_device")
            .width(if compact { 180.0 } else { 220.0 })
            .selected_text(device_text)
            .show_ui(ui, |ui| {
                if self.provider == Provider::DirectMl {
                    for device in &self.devices {
                        ui.selectable_value(
                            &mut self.device_id,
                            device.id,
                            format!(
                                "GPU {} · {} ({:.1} GB)",
                                device.id,
                                device.name,
                                device.dedicated_bytes as f64 / 1073741824.0
                            ),
                        );
                    }
                } else if self.provider == Provider::Cuda {
                    ui.selectable_value(&mut self.device_id, 0, "CUDA ordinal 0");
                } else {
                    ui.label("System CPU");
                }
            });
        egui::ComboBox::from_id_salt("toolbar_compute")
            .width(100.0)
            .selected_text(match self.provider {
                Provider::Cpu => "CPU",
                Provider::DirectMl => "GPU / DirectML",
                Provider::Cuda => "CUDA",
            })
            .show_ui(ui, |ui| {
                if cfg!(windows) {
                    ui.selectable_value(&mut self.provider, Provider::DirectMl, "GPU / DirectML");
                }
                ui.selectable_value(&mut self.provider, Provider::Cpu, "CPU");
            });
        if self.provider != previous {
            self.device_id = if self.provider == Provider::DirectMl {
                self.devices.first().map_or(0, |device| device.id)
            } else {
                0
            };
        }
    }
    fn sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("media_subject")
            .default_width(270.0)
            .min_width(200.0)
            .max_width((ctx.content_rect().width() * 0.28).clamp(200.0, 440.0))
            .resizable(true)
            .show_separator_line(true)
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(10))
            .show(ctx, |ui| {
                let previous_tab = self.left_tab;
                let tab = theme::tab_bar(
                    ui,
                    "left_tabs",
                    &["Media", "Subject", "Garbage", "Despill"],
                    match self.left_tab {
                        LeftTab::Media => 0,
                        LeftTab::Subject => 1,
                        LeftTab::Garbage => 2,
                        LeftTab::Despill => 3,
                    },
                );
                if tab == 1 && previous_tab != LeftTab::Subject {
                    self.mode = ViewMode::Source;
                }
                self.left_tab = match tab {
                    0 => LeftTab::Media,
                    1 => LeftTab::Subject,
                    2 => LeftTab::Garbage,
                    _ => LeftTab::Despill,
                };
                let new_tab = self.left_tab;
                let mask_kind = match new_tab {
                    LeftTab::Garbage => Some(MaskKind::Garbage),
                    LeftTab::Despill => Some(MaskKind::Despill),
                    LeftTab::Media | LeftTab::Subject => None,
                };
                if let Some(session) = self.current_mut() {
                    if let Some(kind) = mask_kind {
                        session.mask_ui.set_active_kind(kind);
                    }
                    if previous_tab != new_tab {
                        // v1 shows the editable overlay only while a mask tab is
                        // active; switching back hides it again.
                        session.mask_ui.editing = mask_kind.is_some();
                    }
                }
                if new_tab == LeftTab::Media {
                    self.media_list(ui);
                    self.left_panel_rect = ui.min_rect();
                    return;
                }
                let left_tab = new_tab;
                if left_tab == LeftTab::Subject {
                    egui::TopBottomPanel::bottom("subject_actions")
                        .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(
                            egui::Margin {
                                left: 0,
                                right: 0,
                                top: 8,
                                bottom: 2,
                            },
                        ))
                        .show_separator_line(false)
                        .show_inside(ui, |ui| self.subject_actions(ui));
                }
                egui::ScrollArea::vertical()
                    .id_salt("left_tools")
                    .auto_shrink([false, false])
                    .show(ui, |ui| match left_tab {
                        LeftTab::Media => unreachable!(),
                        LeftTab::Subject => self.subject_controls(ui),
                        LeftTab::Garbage => self.mask_layers_controls(ui, MaskKind::Garbage),
                        LeftTab::Despill => self.mask_layers_controls(ui, MaskKind::Despill),
                    });
                self.left_panel_rect = ui.min_rect();
            });
    }

    fn media_list(&mut self, ui: &mut egui::Ui) {
        // Leave room below the list for Embed sources and the Source card.
        let list_height = (ui.available_height() - 290.0).max(60.0);
        let mut choose = None;
        let mut media_action = None;
        ui.allocate_ui_with_layout(
            Vec2::new(ui.available_width(), list_height),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                ui.set_min_height(list_height);
                ui.set_max_height(list_height);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Media")
                            .size(13.5)
                            .strong()
                            .color(theme::TEXT),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(egui::Button::new("+ Add").corner_radius(6))
                            .on_hover_text("Add footage to this project (Ctrl+O)")
                            .clicked()
                        {
                            self.open_footage();
                        }
                    });
                });
                ui.add_space(4.0);
                egui::ScrollArea::vertical()
                    .id_salt("media_entries")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 4.0;
                        for (index, session) in self.sessions.iter().enumerate() {
                            let selected = index == self.selected;
                            let name = session
                                .media()
                                .path
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy();
                            let media = session.media();
                            let meta = format!(
                                "{} × {} · {} frames{}",
                                media.width,
                                media.height,
                                media.frame_count,
                                if session.hint.is_some() {
                                    " · hint"
                                } else {
                                    ""
                                }
                            );
                            let response = media_list_row(ui, &name, &meta, selected);
                            response
                                .clone()
                                .on_hover_text(display_path(&session.media().path));
                            if response.clicked() {
                                choose = Some(index);
                            }
                            response.context_menu(|ui| {
                                if ui.button("Open in timeline").clicked() {
                                    media_action = Some(MediaAction::Activate(index));
                                    ui.close();
                                }
                                if ui.button("Show in file manager").clicked() {
                                    media_action = Some(MediaAction::Reveal(index));
                                    ui.close();
                                }
                                let idle = self.active.is_none()
                                    && self.jobs.is_empty()
                                    && !self.project_io_busy;
                                if ui
                                    .add_enabled(
                                        idle,
                                        egui::Button::new("Find / reconnect footage…"),
                                    )
                                    .clicked()
                                {
                                    media_action = Some(MediaAction::Reconnect(index));
                                    ui.close();
                                }
                                ui.separator();
                                if ui
                                    .add_enabled(
                                        idle,
                                        egui::Button::new(if session.hint.is_some() {
                                            "Replace hint footage…"
                                        } else {
                                            "Add hint footage…"
                                        }),
                                    )
                                    .clicked()
                                {
                                    media_action = Some(MediaAction::Hint(index));
                                    ui.close();
                                }
                                if ui
                                    .add_enabled(
                                        idle && session.hint.is_some(),
                                        egui::Button::new("Remove hint footage"),
                                    )
                                    .clicked()
                                {
                                    media_action = Some(MediaAction::ClearHint(index));
                                    ui.close();
                                }
                                ui.separator();
                                if ui
                                    .add_enabled(idle, egui::Button::new("Remove from project"))
                                    .clicked()
                                {
                                    media_action = Some(MediaAction::Remove(index));
                                    ui.close();
                                }
                            });
                        }
                        if self.sessions.is_empty() {
                            theme::hint(ui, "Open or drop a clip to begin.");
                        }
                    });
            },
        );
        let embed_before = self.app_snapshot();
        self.icons
            .checkbox(ui, &mut self.embed_sources, "Embed sources")
            .on_hover_text(
                "Rendered footage is always inside .chroma. Include original and hint media too; existing embedded media is retained.",
            );
        if let Some(before) = embed_before
            && before.embed_sources != self.embed_sources
        {
            self.app_history.record(
                crate::app_history::AppContext::Media,
                "embed sources",
                before,
            );
        }
        if let Some(index) = choose {
            self.selected = index;
            self.loaded = None;
            self.requested_key = None;
            self.playing = false;
            self.pan = Vec2::ZERO;
            self.zoom = ViewerZoom::default();
        }
        match media_action {
            Some(MediaAction::Activate(index)) => {
                self.selected = index;
                self.loaded = None;
                self.requested_key = None;
                self.mode = ViewMode::Composite;
            }
            Some(MediaAction::Reconnect(index)) => self.reconnect_media(index),
            Some(MediaAction::Hint(index)) => self.add_hint_footage(index),
            Some(MediaAction::ClearHint(index)) => {
                self.sessions[index].hint = None;
                self.sessions[index].revision += 1;
                if self.selected == index {
                    self.hints = HintMode::None;
                }
            }
            Some(MediaAction::Remove(index)) => {
                self.sessions.remove(index);
                self.selected = self.selected.min(self.sessions.len().saturating_sub(1));
                self.loaded = None;
                self.requested_key = None;
            }
            Some(MediaAction::Reveal(index)) => {
                let path = self.sessions[index].media().path.clone();
                if path.is_file() {
                    reveal_path(&path);
                } else {
                    self.toast = "Footage is offline or embedded; reconnect it or load it in the viewport first.".into();
                }
            }
            None => {}
        }
        ui.add_space(6.0);
        let source = self.current().map(|session| {
            let media = session.media();
            let path = self
                .archive
                .as_ref()
                .and_then(|archive| archive.original_filename(&media.path))
                .map_or_else(
                    || display_path(&media.path),
                    |name| format!("Embedded · {name}"),
                );
            (
                format!("{} × {}", media.width, media.height),
                format!("{:.3} fps", fps(media)),
                format!(
                    "{} · {} scenes",
                    media.frame_count,
                    session.project.scenes.len()
                ),
                timecode(media.frame_count, fps(media)),
                path,
            )
        });
        theme::card(ui, "Source", None, |ui| match &source {
            Some((size, rate, frames, duration, path)) => {
                theme::field_row(ui, "Resolution", |ui| ui.label(size));
                theme::field_row(ui, "Frame rate", |ui| ui.label(rate));
                theme::field_row(ui, "Frames", |ui| ui.label(frames));
                theme::field_row(ui, "Duration", |ui| ui.label(duration));
                ui.add(egui::Label::new(RichText::new(path).size(11.5).color(MUTED)).truncate())
                    .on_hover_text(path);
            }
            None => {
                theme::hint(ui, "No source selected.");
            }
        });
    }

    /// SAM layers: listed top (applied last) to bottom (applied first). Click selects, the
    /// checkbox enables, Keep/Remove sets how the layer combines, the handle drags to reorder.
    fn sam_layer_controls(&mut self, ui: &mut egui::Ui, idle: bool) {
        enum LayerEdit {
            Select(u64),
            Enable(u64, bool),
            Mode(u64, ck_desktop_core::SamLayerMode),
            Delete(u64),
            Add,
            Move(usize, usize),
            Rename(u64, String),
        }
        let Some(session) = self.current() else {
            return;
        };
        let scene = session.project.scene_of_frame(session.playhead);
        let layers: Vec<ck_desktop_core::SamLayer> = session
            .project
            .layers_in_scene(scene)
            .into_iter()
            .rev()
            .cloned()
            .collect();
        let display_ids: Vec<u64> = layers.iter().map(|layer| layer.id).collect();
        let selected = session.sam_layer;
        let checkbox_on = self.icons.checkbox_texture(true);
        let checkbox_off = self.icons.checkbox_texture(false);
        let mut edits = Vec::new();
        let mut rows: Vec<Rect> = Vec::new();
        let mut drag = self.layer_drag;
        let mut rename = self.layer_rename.take();
        let mut checkbox_rects = Vec::new();
        let trash = self.icons.texture(Icon::Trash);
        let badge = format!("{}", layers.len());
        theme::card(ui, "Layers", Some(&badge), |ui| {
            if layers.is_empty() {
                theme::hint(
                    ui,
                    "No layers in this scene. A point, a box or a text prompt creates one.",
                );
            }
            for (display, layer) in layers.iter().enumerate() {
                let row = ui.horizontal(|ui| {
                    let handle = ui
                        .add(
                            egui::Label::new(egui::RichText::new("≡").color(MUTED))
                                .sense(Sense::drag()),
                        )
                        .on_hover_text("Drag to reorder");
                    if handle.hovered() || drag == Some(display) {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                    }
                    if handle.drag_started() && idle {
                        drag = Some(display);
                    }
                    let mut enabled = layer.enabled;
                    let toggle = checkbox_widget(ui, &mut enabled, "", checkbox_on, checkbox_off)
                        .on_hover_text("Use this layer in the combined mask");
                    checkbox_rects.push(toggle.rect);
                    if toggle.clicked() {
                        edits.push(LayerEdit::Enable(layer.id, enabled));
                    }
                    // Right side first (fixed widths), the name takes whatever is left.
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        if ui
                            .add_enabled(
                                idle,
                                egui::Button::new(
                                    egui::Image::from_texture((trash, Vec2::splat(14.0)))
                                        .alt_text("Delete layer"),
                                )
                                .frame(false),
                            )
                            .on_hover_text("Delete the layer, its points and its masks")
                            .clicked()
                        {
                            edits.push(LayerEdit::Delete(layer.id));
                        }
                        let remove = layer.mode == ck_desktop_core::SamLayerMode::Remove;
                        let chip = egui::Button::new(
                            RichText::new(if remove { "− Remove" } else { "+ Keep" })
                                .size(11.5)
                                .color(if remove { theme::BAD } else { theme::TEXT }),
                        )
                        .corner_radius(4)
                        .min_size(Vec2::new(66.0, 20.0));
                        if ui
                            .add_enabled(idle, chip)
                            .on_hover_text(if remove {
                                "Remove: subtracts this mask from the layers below. Click for Keep."
                            } else {
                                "Keep: adds this mask. Click for Remove."
                            })
                            .clicked()
                        {
                            edits.push(LayerEdit::Mode(
                                layer.id,
                                if remove {
                                    ck_desktop_core::SamLayerMode::Keep
                                } else {
                                    ck_desktop_core::SamLayerMode::Remove
                                },
                            ));
                        }
                        let size = Vec2::new(ui.available_width().max(24.0), 22.0);
                        let renaming = matches!(&rename, Some((id, _)) if *id == layer.id);
                        if renaming {
                            let (_, text) = rename.as_mut().expect("renaming this layer");
                            let field = ui.add_sized(size, egui::TextEdit::singleline(text));
                            if !field.has_focus() && !field.lost_focus() {
                                field.request_focus();
                            }
                            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                rename = None;
                            } else if field.lost_focus() {
                                let (id, text) = rename.take().expect("renaming this layer");
                                edits.push(LayerEdit::Rename(id, text));
                            }
                        } else {
                            let (rect, response) = ui.allocate_exact_size(size, Sense::click());
                            let is_selected = layer.id == selected;
                            if is_selected || response.hovered() {
                                ui.painter().rect_filled(
                                    rect,
                                    4.0,
                                    if is_selected {
                                        theme::SEGMENT_ON
                                    } else {
                                        theme::BG_PANEL_ALT
                                    },
                                );
                            }
                            ui.painter_at(rect.shrink2(Vec2::new(6.0, 0.0))).text(
                                rect.left_center() + Vec2::new(6.0, 0.0),
                                egui::Align2::LEFT_CENTER,
                                &layer.name,
                                FontId::proportional(12.5),
                                if layer.enabled { theme::TEXT } else { MUTED },
                            );
                            let response = response.on_hover_text(format!(
                                "{}\nClick to select · double-click to rename",
                                layer.name
                            ));
                            if response.double_clicked() && idle {
                                rename = Some((layer.id, layer.name.clone()));
                            } else if response.clicked() {
                                edits.push(LayerEdit::Select(layer.id));
                            }
                        }
                    });
                });
                rows.push(row.response.rect);
            }
            if let Some(from) = drag {
                let pointer = ui.ctx().pointer_latest_pos();
                let to = pointer.map_or(from, |p| {
                    rows.iter()
                        .position(|r| p.y < r.center().y)
                        .unwrap_or(rows.len().saturating_sub(1))
                });
                if let Some(target) = rows.get(to) {
                    let y = if to <= from {
                        target.top()
                    } else {
                        target.bottom()
                    };
                    ui.painter()
                        .hline(target.x_range(), y, Stroke::new(2.0_f32, theme::ACCENT));
                }
                if ui.ctx().input(|i| i.pointer.any_released()) {
                    if to != from {
                        edits.push(LayerEdit::Move(from, to));
                    }
                    drag = None;
                }
            }
            if ui
                .add_enabled(
                    idle,
                    theme::secondary_button("+ Add layer", ui.available_width()),
                )
                .on_hover_text("New Keep layer on top, with its own points and masks")
                .clicked()
            {
                edits.push(LayerEdit::Add);
            }
        });
        self.layer_drag = drag;
        self.layer_rename = rename;
        self.layer_checkbox_rects = checkbox_rects;
        if edits.is_empty() {
            return;
        }
        let Some(session) = self.current_mut() else {
            return;
        };
        let mut composites_changed = false;
        for edit in edits {
            match edit {
                LayerEdit::Select(id) => session.sam_layer = id,
                LayerEdit::Rename(id, name) => {
                    let name = name.trim();
                    if !name.is_empty()
                        && let Some(l) = session.project.sam_layers.iter_mut().find(|l| l.id == id)
                    {
                        l.name = name.chars().take(48).collect();
                    }
                }
                LayerEdit::Enable(id, on) => {
                    if let Some(l) = session.project.sam_layers.iter_mut().find(|l| l.id == id) {
                        l.enabled = on;
                        composites_changed = true;
                    }
                }
                LayerEdit::Mode(id, mode) => {
                    if let Some(l) = session.project.sam_layers.iter_mut().find(|l| l.id == id) {
                        l.mode = mode;
                        composites_changed = true;
                    }
                }
                LayerEdit::Delete(id) => session.remove_sam_layer(id),
                LayerEdit::Add => {
                    session.sam_layer = session.project.add_sam_layer(scene);
                }
                LayerEdit::Move(from, to) => {
                    // Reorder this scene's layers (display order is reversed: top = applied
                    // last) and write them back into the slots they occupy in the project.
                    if from < display_ids.len() && to < display_ids.len() {
                        let mut order = display_ids.clone();
                        let id = order.remove(from);
                        order.insert(to, id);
                        order.reverse();
                        let slots: Vec<usize> = session
                            .project
                            .sam_layers
                            .iter()
                            .enumerate()
                            .filter(|(_, l)| l.in_scene(scene))
                            .map(|(i, _)| i)
                            .collect();
                        let reordered: Vec<ck_desktop_core::SamLayer> = order
                            .iter()
                            .filter_map(|id| {
                                session
                                    .project
                                    .sam_layers
                                    .iter()
                                    .find(|l| l.id == *id)
                                    .cloned()
                            })
                            .collect();
                        if reordered.len() == slots.len() {
                            for (slot, layer) in slots.into_iter().zip(reordered) {
                                session.project.sam_layers[slot] = layer;
                            }
                            composites_changed = true;
                        }
                    }
                }
            }
        }
        if composites_changed {
            session.invalidate_sam_composites();
        }
        self.requested_key = None;
    }

    fn subject_controls(&mut self, ui: &mut egui::Ui) {
        let frame = self.current().map_or(0, |session| session.playhead);
        let layer = self.current().map_or(0, |session| session.sam_layer);
        let points = self.current().map_or(0, |session| {
            session
                .project
                .prompts
                .iter()
                .filter(|p| p.frame == frame && p.layer == layer)
                .count()
        });
        let has_box = self.current().is_some_and(|session| {
            session
                .project
                .sam_boxes
                .iter()
                .any(|b| b.frame == frame && b.layer == layer)
        });
        let idle = self.active.is_none() && self.jobs.is_empty();
        let sam31 = self.effective_sam_model() == SamModel::Sam31;
        let mut tool = self.point_tool;
        if !sam31 && tool == PointTool::Box {
            tool = PointTool::Subject;
        }
        let mut overlay = self.overlay;
        let mut undo = false;
        let mut clear = false;
        let checkbox_on = self.icons.checkbox_texture(true);
        let checkbox_off = self.icons.checkbox_texture(false);
        let badge = if has_box {
            format!("{points} + box on frame {}", frame + 1)
        } else {
            format!("{points} on frame {}", frame + 1)
        };
        theme::card(ui, "Points", Some(&badge), |ui| {
            theme::hint(
                ui,
                if sam31 {
                    "Click to add a point to the selected layer (Shift: background). Drag moves a point, right-click removes it, click a point to give it a text prompt. Box: drag a rectangle."
                } else {
                    "Click to add a point to the selected layer (Shift: background). Drag moves a point, right-click removes it."
                },
            );
            let mut options = vec![
                (PointTool::Subject, "+ Subject", "Clicks add subject points"),
                (
                    PointTool::Background,
                    "− Background",
                    "Clicks add background points",
                ),
            ];
            if sam31 {
                options.push((
                    PointTool::Box,
                    "▭ Box",
                    "Drag a box around the subject (one per layer and frame)",
                ));
            }
            theme::segmented_fill(ui, &mut tool, &options);
            ui.columns(2, |columns| {
                undo = columns[0]
                    .add_enabled(
                        points > 0 && idle,
                        theme::secondary_button("Undo point", columns[0].available_width()),
                    )
                    .clicked();
                clear = columns[1]
                    .add_enabled(
                        (points > 0 || has_box) && idle,
                        theme::secondary_button("Clear frame", columns[1].available_width()),
                    )
                    .clicked();
            });
            theme::field_row(ui, "Show mask overlay", |ui| {
                checkbox_widget(ui, &mut overlay, "", checkbox_on, checkbox_off)
                    .on_hover_text("Show the SAM subject mask on the viewer.");
            });
        });
        self.point_tool = tool;
        self.overlay = overlay;
        if undo
            && let Some(session) = self.current_mut()
            && let Some(index) = session
                .project
                .prompts
                .iter()
                .rposition(|p| p.frame == frame && p.layer == layer)
        {
            session.project.prompts.remove(index);
            session.invalidate_prompt_scene(frame, layer);
            self.segment_pending = Some((frame, layer));
        }
        if clear && let Some(session) = self.current_mut() {
            session
                .project
                .prompts
                .retain(|p| !(p.frame == frame && p.layer == layer));
            session
                .project
                .sam_boxes
                .retain(|b| !(b.frame == frame && b.layer == layer));
            session.invalidate_prompt_scene(frame, layer);
        }
        self.sam_layer_controls(ui, idle);
        let start = self.current().map(|session| session.range.start + 1);
        let original_prompt = self
            .current()
            .and_then(|session| session.sam_layer(session.sam_layer))
            .map(|l| l.prompt.clone())
            .unwrap_or_default();
        let mut layer_prompt = original_prompt.clone();
        let mut prompt_committed = false;
        let mut prompt_focus = self.layer_prompt_focus.take();
        // Read-only translation line (SAM 3.1 understands English only); not while typing.
        let prompt_caption =
            if prompt_focus.is_none() && self.effective_sam_model() == SamModel::Sam31 {
                self.translator.request(&original_prompt);
                crate::translation::caption(&self.translator, &original_prompt)
            } else {
                None
            };
        let sam31_available = self.runtime.sam31_available();
        let mut sam_model = if sam31_available {
            self.sam_model
        } else {
            SamModel::Sam21
        };
        theme::card(ui, "Segmentation", None, |ui| {
            theme::field_row(ui, "Model", |ui| {
                ui.add_enabled_ui(idle && sam31_available, |ui| {
                    theme::segmented(
                        ui,
                        &mut sam_model,
                        &[
                            (
                                SamModel::Sam21,
                                "SAM 2.1",
                                "Smaller and faster. Points only.",
                            ),
                            (
                                SamModel::Sam31,
                                "SAM 3.1",
                                "Higher quality tracking with points, boxes and text prompts.",
                            ),
                        ],
                    );
                })
                .response
                .on_disabled_hover_text(if sam31_available {
                    "Wait for the current job to finish."
                } else {
                    "Install the SAM 3.1 pack in models/sam3.1 to enable it."
                });
            });
            if sam_model == SamModel::Sam31 {
                theme::field_row(ui, "Text prompt", |ui| {
                    let response = ui.add(
                        egui::TextEdit::singleline(&mut layer_prompt)
                            .hint_text("e.g. person")
                            .desired_width(ui.available_width()),
                    );
                    if response.gained_focus() {
                        prompt_focus = Some(layer_prompt.clone());
                    }
                    if response.lost_focus() {
                        prompt_committed = prompt_focus
                            .take()
                            .is_none_or(|before| before != layer_prompt);
                    }
                    response.on_hover_text(
                        "Every match of this text joins the layer (found on the earliest prompted frame, or the first frame of the range). Enter to apply.",
                    );
                });
                if let Some(caption) = &prompt_caption {
                    ui.label(egui::RichText::new(caption).small().color(MUTED))
                        .on_hover_text("What SAM 3.1 receives (it understands English only).");
                }
            }
            if let Some(start) = start {
                theme::hint(
                    ui,
                    if sam_model == SamModel::Sam31 {
                        "Add a subject point, a box or a text prompt on any frame. Tracking starts from the earliest prompted frame and runs forward and backward; prompts on other frames correct it."
                            .to_owned()
                    } else {
                        format!(
                            "Begin with a subject point on frame {start}. Later points correct the propagation after them."
                        )
                    },
                );
            }
        });
        self.sam_model = sam_model;
        self.layer_prompt_focus = prompt_focus;
        // The layer's text prompt: stored as typed, applied (results dropped, frame re-segmented)
        // when the field loses focus with a changed value.
        if layer_prompt != original_prompt
            && let Some(session) = self.current_mut()
        {
            let id = if layer_prompt.trim().is_empty() {
                session.sam_layer
            } else {
                session.layer_for_edit()
            };
            if let Some(l) = session.project.sam_layers.iter_mut().find(|l| l.id == id) {
                l.prompt = layer_prompt.clone();
            }
        }
        if prompt_committed && let Some(session) = self.current_mut() {
            let id = session.sam_layer;
            let frame = session.playhead;
            session.invalidate_prompt_scene(frame, id);
            if !layer_prompt.trim().is_empty() {
                self.segment_pending = Some((frame, id));
            }
        }
        theme::hint(ui, "Wheel: zoom · Middle drag: pan");
    }

    /// Segment / Propagate / All layers / Export mask, pinned to the bottom of the Subject tab.
    fn subject_actions(&mut self, ui: &mut egui::Ui) {
        let idle = self.active.is_none() && self.jobs.is_empty() && !self.project_io_busy;
        let available = self.current().is_some() && idle;
        let layer_name = self
            .current()
            .and_then(|session| session.sam_layer(session.sam_layer))
            .map_or_else(|| "layer".to_owned(), |l| l.name.clone());
        let scene_layers = self.current().map_or(0, |session| {
            let scene = session.project.scene_of_frame(session.playhead);
            session.project.layers_in_scene(scene).len()
        });
        let can_export = self.export_masks.is_none()
            && self.current().is_some_and(|session| {
                session.scene_frames_at_playhead().any(|frame| {
                    session
                        .frames
                        .get(&frame)
                        .is_some_and(|files| session.has_subject_mask(files))
                })
            });
        let gap = ui.spacing().item_spacing.x;
        let size = Vec2::new(((ui.available_width() - 3.0 * gap) / 4.0).max(30.0), 30.0);
        let (mut segment, mut propagate, mut propagate_all, mut export) =
            (false, false, false, false);
        ui.horizontal(|ui| {
            segment = ui
                .add_enabled(available, self.icons.sized_button(Icon::SegmentFrame, size))
                .on_hover_text(format!(
                    "Segment frame: the playhead frame for {layer_name}."
                ))
                .clicked();
            propagate = ui
                .add_enabled(
                    available,
                    self.icons
                        .sized_button(Icon::Propagate, size)
                        .fill(theme::ACCENT),
                )
                .on_hover_text(format!("Propagate {layer_name} through this scene."))
                .clicked();
            propagate_all = ui
                .add_enabled(
                    available && scene_layers > 1,
                    self.icons.sized_button(Icon::PropagateAll, size),
                )
                .on_hover_text("Propagate all layers: every enabled layer of this scene.")
                .on_disabled_hover_text("Propagate all layers (needs two or more layers).")
                .clicked();
            export = ui
                .add_enabled(can_export, self.icons.sized_button(Icon::ExportMask, size))
                .on_hover_text(
                    "Export mask: this scene's combined subject mask as a PNG sequence \
                     (white = subject, black = background).",
                )
                .on_disabled_hover_text("Export mask (segment or propagate this scene first).")
                .clicked();
        });
        if let Some(status) = &self.export_masks {
            ui.label(egui::RichText::new(status.label()).small().color(MUTED));
        }
        if segment {
            self.enqueue(Action::Segment);
        }
        if propagate {
            self.enqueue(Action::Propagate);
        }
        if propagate_all {
            self.enqueue(Action::PropagateAll);
        }
        if export {
            self.export_scene_masks();
        }
    }

    /// Writes the playhead scene's combined subject mask, one 8-bit grayscale PNG per frame
    /// (`<clip>_mask_<frame>.png`, source frame numbers, black where nothing was segmented).
    fn export_scene_masks(&mut self) {
        use crate::sam_layers::Composite;
        let Some(session) = self.current() else {
            return;
        };
        let media = session.media().clone();
        let frames: Vec<(u64, Composite)> = session
            .scene_frames_at_playhead()
            .map(|frame| {
                let layers = session.project.layers_for_frame(frame);
                let composite = session.frames.get(&frame).map_or(Composite::None, |files| {
                    crate::sam_layers::plan(&layers, &files.sam_layers)
                });
                (frame, composite)
            })
            .collect();
        let Some(&(first, _)) = frames.first() else {
            return;
        };
        let stem = media
            .path
            .file_stem()
            .map_or_else(|| "clip".to_owned(), |s| s.to_string_lossy().into_owned());
        let Some(folder) = rfd::FileDialog::new()
            .set_title("Export scene subject mask (PNG sequence)")
            .pick_folder()
        else {
            return;
        };
        let mut paths = Vec::new();
        for (_, composite) in &frames {
            match composite {
                Composite::None => {}
                Composite::Single(path) => paths.push(path.clone()),
                Composite::Combine(inputs) => paths.extend(inputs.iter().map(|(p, _)| p.clone())),
            }
        }
        let archive = self.archive.clone();
        let status = std::sync::Arc::new(MaskExportStatus::new(frames.len()));
        self.export_masks = Some(status.clone());
        let repaint = self.repaint.clone();
        let (width, height) = (media.width, media.height);
        std::thread::spawn(move || {
            let result = (|| -> anyhow::Result<usize> {
                if let Some(archive) = archive {
                    archive.materialize(&paths)?;
                }
                let mut written = 0;
                for (frame, composite) in frames {
                    let target = folder.join(format!("{stem}_mask_{frame:06}.png"));
                    let image = match composite {
                        Composite::None => image::GrayImage::new(width, height),
                        Composite::Single(path) => image::open(&path)?.into_luma8(),
                        Composite::Combine(inputs) => crate::sam_layers::combine(&inputs)?,
                    };
                    image.save(&target)?;
                    written += 1;
                    status.progress(written);
                    repaint.request_repaint();
                }
                Ok(written)
            })();
            status.finish(result.map_err(|e| format!("{e:#}")), first);
            repaint.request_repaint();
        });
    }

    fn tools(&mut self, ctx: &egui::Context) {
        let history_before = self.app_snapshot();
        let settings_before = self.current().map(|session| {
            (
                session.project.ck_settings.clone(),
                session.project.despill_settings,
            )
        });
        egui::SidePanel::right("studio_inspector")
            .default_width(330.0)
            .min_width(285.0)
            .max_width((ctx.content_rect().width() * 0.31).clamp(285.0, 460.0))
            .resizable(true)
            .show_separator_line(true)
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(10))
            .show(ctx, |ui| {
                let tabs = [
                    RightTab::Keyer,
                    RightTab::Scenes,
                    RightTab::Jobs,
                    RightTab::Info,
                ];
                let selected = tabs
                    .iter()
                    .position(|tab| *tab == self.right_tab)
                    .unwrap_or(0);
                self.right_tab = tabs[theme::tab_bar(
                    ui,
                    "right_tabs",
                    &["Keyer", "Scenes", "Jobs", "Info"],
                    selected,
                )];
                ui.add_space(8.0);
                egui::TopBottomPanel::bottom("fixed_scene_actions")
                    .exact_height(if matches!(self.right_tab, RightTab::Keyer) {
                        232.0
                    } else {
                        116.0
                    })
                    .frame(
                        egui::Frame::new()
                            .fill(theme::BG_PANEL)
                            .inner_margin(egui::Margin {
                                left: 0,
                                right: 0,
                                top: 8,
                                bottom: 2,
                            }),
                    )
                    .show_separator_line(false)
                    .show_inside(ui, |ui| {
                        if matches!(self.right_tab, RightTab::Keyer) {
                            self.despill_controls(ui);
                        }
                        self.keyer_actions(ui);
                    });
                egui::ScrollArea::vertical()
                    .id_salt("inspector_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| match self.right_tab {
                        RightTab::Keyer => self.keyer_controls(ui),
                        RightTab::Scenes => self.scene_controls(ui),
                        RightTab::Jobs => self.queue_controls(ui),
                        RightTab::Info => {
                            let clip = self.current().map(|session| {
                                let media = session.media();
                                (
                                    media
                                        .path
                                        .file_name()
                                        .unwrap_or_default()
                                        .to_string_lossy()
                                        .into_owned(),
                                    format!("{} × {}", media.width, media.height),
                                    format!("{:.3} fps", fps(media)),
                                    format!("{}", media.frame_count),
                                    timecode(media.frame_count, fps(media)),
                                )
                            });
                            theme::card(ui, "Clip", None, |ui| match &clip {
                                Some((name, size, rate, frames, duration)) => {
                                    ui.add(
                                        egui::Label::new(RichText::new(name).color(theme::TEXT))
                                            .truncate(),
                                    );
                                    theme::field_row(ui, "Resolution", |ui| ui.label(size));
                                    theme::field_row(ui, "Frame rate", |ui| ui.label(rate));
                                    theme::field_row(ui, "Frames", |ui| ui.label(frames));
                                    theme::field_row(ui, "Duration", |ui| ui.label(duration));
                                }
                                None => {
                                    theme::hint(ui, "No media loaded.");
                                }
                            });
                            let mut recheck = false;
                            let mut settings = false;
                            theme::card(ui, "Application", None, |ui| {
                                let width = ui.available_width();
                                recheck = ui
                                    .add(theme::secondary_button("Check required files", width))
                                    .clicked();
                                settings = ui
                                    .add(theme::secondary_button("Advanced settings…", width))
                                    .clicked();
                            });
                            if recheck {
                                self.assets.recheck();
                            }
                            if settings {
                                self.open_settings = true;
                            }
                        }
                    });
                self.inspector_rect = ui.min_rect();
            });
        if let (Some(before), Some((ck_settings, despill_settings))) =
            (history_before, settings_before)
        {
            let changed = self.current().is_some_and(|session| {
                session.project.ck_settings != ck_settings
                    || session.project.despill_settings != despill_settings
            });
            if changed {
                self.app_history.record(
                    crate::app_history::AppContext::Keyer,
                    "inspector settings",
                    before,
                );
            }
        }
    }

    fn keyer_controls(&mut self, ui: &mut egui::Ui) {
        let scene = self.current().and_then(|session| {
            timeline::scene_index_at_frame(&session.project.scenes, session.playhead).map(|index| {
                let scene = &session.project.scenes[index];
                (index, scene.enabled, scene.start, scene.end_exclusive)
            })
        });
        let reconnect = self
            .current()
            .is_some_and(|session| session.project.despill_settings.is_enabled())
            && self
                .loaded
                .as_ref()
                .is_some_and(|frame| frame.despill_unavailable);
        match scene {
            Some((index, enabled, start, end)) => {
                let range = format!("frames {}–{}", start + 1, end);
                theme::card(ui, &format!("Scene {}", index + 1), Some(&range), |ui| {
                    theme::hint(
                        ui,
                        if enabled {
                            "Follows the playhead. Run current scene keys it."
                        } else {
                            "Keying is off for this scene. Turn it on from its timeline checkbox."
                        },
                    );
                    if reconnect {
                        ui.label(
                            RichText::new(
                                "Reconnect the original footage to apply Result despill.",
                            )
                            .size(11.5)
                            .color(ORANGE),
                        );
                    }
                });
            }
            None => {
                theme::card(ui, "No clip", None, |ui| {
                    theme::hint(ui, "Open media to set up keying.");
                });
            }
        }
        let hann_advice_input = self.hann_advice_input();
        let checkbox_on = self.icons.checkbox_texture(true);
        let checkbox_off = self.icons.checkbox_texture(false);
        let mut hints = self.hints;
        theme::card(ui, "Keying", None, |ui| {
            theme::field_row(ui, "Mask hint", |ui| {
                egui::ComboBox::from_id_salt("mask_hint")
                    .width(ui.available_width().min(190.0))
                    .selected_text(match hints {
                        HintMode::None => "None",
                        HintMode::First => "First-frame SAM mask",
                        HintMode::Video => "Propagated SAM masks",
                        HintMode::External => "External hint footage",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut hints, HintMode::None, "None");
                        ui.selectable_value(&mut hints, HintMode::First, "First-frame SAM mask");
                        ui.selectable_value(&mut hints, HintMode::Video, "Propagated SAM masks");
                        ui.selectable_value(
                            &mut hints,
                            HintMode::External,
                            "External hint footage",
                        );
                    });
            });
            if hints != HintMode::None
                && let Some(session) = self.current_mut()
            {
                use ck_desktop_core::HintQuality;
                let mut quality = session.project.hint_quality;
                theme::field_row(ui, "Hint quality", |ui| {
                    theme::segmented(
                        ui,
                        &mut quality,
                        &[HintQuality::High, HintQuality::Rough, HintQuality::Loose]
                            .map(|q| (q, q.label(), q.note())),
                    );
                });
                theme::hint(ui, quality.note());
                session.project.hint_quality = quality;
            }
            if let Some(session) = self.current_mut() {
                ck_settings_controls(
                    ui,
                    &mut session.project.ck_settings,
                    hann_advice_input,
                    checkbox_on,
                    checkbox_off,
                );
            }
        });
        self.hints = hints;
        let idle = self.active.is_none() && self.jobs.is_empty() && !self.project_io_busy;
        let valid = self
            .current()
            .is_some_and(|s| s.project.ck_settings.validate().is_ok());
        let range = self.current().map(|session| {
            format!(
                "frames {}–{}",
                session.range.start + 1,
                session.range.end_exclusive
            )
        });
        theme::card(ui, "In/out range", range.as_deref(), |ui| {
            theme::hint(ui, "Key the playback range set by the timeline handles.");
            if ui
                .add_enabled(
                    idle && valid && self.current().is_some(),
                    theme::secondary_button("Run in/out range", ui.available_width()),
                )
                .clicked()
            {
                self.enqueue(Action::RenderRange);
            }
        });
        if let Some(active) = &self.active {
            let policy = active.policy_message.clone();
            let retry = active.memory_retry_message.clone();
            if policy.is_some() || retry.is_some() {
                theme::card(ui, "Current job", None, |ui| {
                    if let Some(policy) = &policy {
                        theme::hint(ui, policy);
                    }
                    if let Some(retry) = &retry {
                        ui.label(RichText::new(retry).size(11.5).color(ORANGE));
                    }
                });
            }
        }
    }

    fn keyer_actions(&mut self, ui: &mut egui::Ui) {
        let idle = self.active.is_none() && self.jobs.is_empty() && !self.project_io_busy;
        let valid = self
            .current()
            .is_some_and(|session| session.project.ck_settings.validate().is_ok());
        let current_enabled = self
            .current()
            .is_some_and(|session| session.current_scene_range().is_ok());
        let count = self.current().map_or(0, |session| {
            session
                .project
                .scenes
                .iter()
                .filter(|scene| scene.enabled)
                .count()
        });
        let width = ui.available_width();
        if ui
            .add_enabled(
                idle && valid && current_enabled,
                theme::primary_button("Run current scene", width),
            )
            .on_hover_text("Render the enabled scene under the playhead (Ctrl+Shift+R)")
            .clicked()
        {
            self.enqueue(Action::RenderScene);
        }
        if ui
            .add_enabled(
                idle && valid && count > 0,
                theme::secondary_button(format!("Run enabled scenes ({count})"), width),
            )
            .on_hover_text("Render enabled timeline scenes (Ctrl+R)")
            .clicked()
        {
            self.enqueue(Action::RenderAll);
        }
        ui.vertical_centered(|ui| {
            theme::hint(
                ui,
                if self.project_path.is_some() {
                    "Calculated footage is saved inside the .chroma project."
                } else {
                    "Unsaved · calculations are temporary until you save."
                },
            );
        });
    }

    fn despill_controls(&mut self, ui: &mut egui::Ui) {
        theme::card(ui, "Result", None, |ui| self.despill_card_body(ui));
    }

    fn despill_card_body(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Despill").color(theme::TEXT_DIM));
        let status = if self
            .loaded
            .as_ref()
            .is_some_and(|frame| frame.despill_unavailable)
        {
            "Original footage is unavailable; showing the rendered result. Reconnect the source to apply despill."
        } else if self
            .loaded
            .as_ref()
            .is_some_and(|frame| frame.despill_applied)
        {
            "Colour unmix is applied to the current Result frame and to exports."
        } else {
            "Live colour unmix. 0% is as rendered; 100% recovers foreground colour from the original footage and matte. The alpha is unchanged."
        };
        let mut changed = false;
        if let Some(session) = self.current_mut() {
            let mut percent = session.project.despill_settings.amount * 100.0;
            changed = despill_slider_row(ui, Some(&mut percent), status);
            if changed {
                session.project.despill_settings.amount = percent / 100.0;
                // Recompose the existing result; keep calculated frames and
                // the expensive inference results available at every value.
                session.revision += 1;
            }
        } else {
            despill_slider_row(ui, None, status);
        }
        if changed {
            self.requested_key = None;
            ui.ctx().request_repaint();
        }
    }
    fn scene_controls(&mut self, ui: &mut egui::Ui) {
        let idle = self.active.is_none() && self.jobs.is_empty();
        let has_clip = self.current().is_some();
        let mut detect = false;
        let mut split = false;
        let mut merge = false;
        let mut threshold = self.scene_threshold;
        let mut min_frames = self.min_scene_frames;
        theme::card(ui, "Scene detection", None, |ui| {
            let width = ui.available_width();
            detect = ui
                .add_enabled(
                    has_clip && idle,
                    theme::secondary_button("Detect scenes", width),
                )
                .on_hover_text("Find cuts with TransNetV2 (Ctrl+D)")
                .clicked();
            ui.label(RichText::new("Cut threshold").color(theme::TEXT_DIM));
            ui.add(egui::Slider::new(&mut threshold, 0.05..=0.95).fixed_decimals(2))
                .on_hover_text("Lower finds more cuts");
            theme::field_row(ui, "Minimum frames", |ui| {
                ui.add(egui::DragValue::new(&mut min_frames).range(1..=600));
            });
            ui.columns(2, |columns| {
                split = columns[0]
                    .add_enabled(
                        has_clip && idle,
                        theme::secondary_button("Split here", columns[0].available_width()),
                    )
                    .on_hover_text("Split the scene at the playhead")
                    .clicked();
                merge = columns[1]
                    .add_enabled(
                        has_clip && idle,
                        theme::secondary_button("Merge next", columns[1].available_width()),
                    )
                    .on_hover_text("Merge the current scene with the next one")
                    .clicked();
            });
        });
        self.scene_threshold = threshold;
        self.min_scene_frames = min_frames;
        if detect {
            self.enqueue(Action::Detect);
        }
        if split {
            self.split_scene();
        }
        if merge {
            self.merge_scene();
        }
        let mut select = None;
        let rows = self.current().map(|session| {
            let current = timeline::scene_index_at_frame(&session.project.scenes, session.playhead);
            session
                .project
                .scenes
                .iter()
                .enumerate()
                .map(|(index, scene)| {
                    (
                        index,
                        current == Some(index),
                        scene.enabled,
                        format!("{}–{}", scene.start + 1, scene.end_exclusive),
                    )
                })
                .collect::<Vec<_>>()
        });
        let count = rows.as_ref().map_or(0, Vec::len);
        let count_text = format!("{count}");
        theme::card(ui, "Scenes", Some(&count_text), |ui| match &rows {
            Some(rows) if !rows.is_empty() => {
                theme::hint(
                    ui,
                    "Turn keying on or off with each scene's timeline checkbox.",
                );
                for (index, current, enabled, range) in rows {
                    let response = scene_list_row(ui, *index, *current, *enabled, range)
                        .on_hover_text("Go to this scene and set its in/out range");
                    if response.clicked() {
                        select = Some(*index);
                    }
                }
            }
            _ => {
                theme::hint(ui, "Open media to see its scenes.");
            }
        });
        if let Some(index) = select
            && let Some(session) = self.current_mut()
        {
            session.select_scene(index);
            self.playing = false;
        }
        let checkbox_on = self.icons.checkbox_texture(true);
        let checkbox_off = self.icons.checkbox_texture(false);
        let Some(session) = self.current_mut() else {
            return;
        };
        let Some(index) = timeline::scene_index_at_frame(&session.project.scenes, session.playhead)
        else {
            theme::card(ui, "Overrides", None, |ui| {
                theme::hint(ui, "No scene under the playhead.");
            });
            return;
        };
        let global = session.project.despill_settings.amount;
        let current = scene_despill_override(&session.project, index);
        let mut active = current.is_some();
        // The slider edits percent; the project stores 0..=1.
        let mut percent = current.unwrap_or(global) * 100.0;
        let scene_badge = format!("scene {}", index + 1);
        let mut changed = false;
        theme::card(ui, "Overrides", Some(&scene_badge), |ui| {
            theme::field_row(ui, "Override despill", |ui| {
                changed |= checkbox_widget(ui, &mut active, "", checkbox_on, checkbox_off)
                    .on_hover_text(
                        "Use a despill amount for this scene only; off means the global slider applies",
                    )
                    .clicked();
            });
            if active {
                changed |= despill_slider_row(
                    ui,
                    Some(&mut percent),
                    "Result despill for this scene only. The global slider is ignored here.",
                );
            } else {
                theme::hint(
                    ui,
                    format!("Using the global value ({:.0} %).", global * 100.0),
                );
            }
            theme::hint(
                ui,
                "Right-click a scene on the timeline for the same options.",
            );
        });
        if changed
            && set_scene_despill(
                &mut session.project,
                index,
                active.then_some(percent / 100.0),
            )
        {
            session.revision = session.revision.wrapping_add(1);
            self.requested_key = None;
            ui.ctx().request_repaint();
        }
    }

    /// Garbage/despill mask layer list for the scene under the playhead.
    fn mask_layers_controls(&mut self, ui: &mut egui::Ui, kind: MaskKind) {
        let scene = self.current().map(|session| {
            let index = timeline::scene_index_at_frame(&session.project.scenes, session.playhead)
                .unwrap_or(session.selected_scene);
            format!("scene {}", index + 1)
        });
        let title = format!("{} masks", kind.noun());
        theme::card(ui, &title, scene.as_deref(), |ui| {
            self.mask_layers_body(ui, kind);
        });
        theme::hint(
            ui,
            "Click to add a corner point, drag to pull tangents, click the first point to close. \
             Double-click a point to transform the whole shape.",
        );
    }

    fn mask_layers_body(&mut self, ui: &mut egui::Ui, kind: MaskKind) {
        let checkbox_on = self.icons.checkbox_texture(true);
        let checkbox_off = self.icons.checkbox_texture(false);
        let Some(session) = self.current_mut() else {
            ui.label(
                RichText::new("Open media to add garbage/despill masks.")
                    .small()
                    .color(MUTED),
            );
            return;
        };
        session.mask_ui.set_active_kind(kind);
        let playhead = session.playhead;
        let scene_index = timeline::scene_index_at_frame(&session.project.scenes, playhead)
            .unwrap_or(session.selected_scene);
        let scene = scene_index as u32;
        let mut editing = session.mask_ui.editing;
        ui.horizontal(|ui| {
            checkbox_widget(
                ui,
                &mut editing,
                "Edit in viewer",
                checkbox_on,
                checkbox_off,
            )
            .on_hover_text("Draw and edit the selected mask's bezier shape in the viewer");
            if editing != session.mask_ui.editing {
                session.mask_ui.editing = editing;
            }
        });
        ui.horizontal(|ui| {
            let can_undo = session.mask_history.can_undo();
            let can_redo = session.mask_history.can_redo();
            if ui
                .add_enabled(can_undo, egui::Button::new("Undo"))
                .on_hover_text("Undo the last mask edit (Ctrl+Z)")
                .clicked()
            {
                session.undo_mask_edit();
            }
            if ui
                .add_enabled(can_redo, egui::Button::new("Redo"))
                .on_hover_text("Redo the last undone mask edit (Ctrl+Shift+Z)")
                .clicked()
            {
                session.redo_mask_edit();
            }
        });
        let remembered = session.mask_ui.remembered(kind, scene);
        let valid = remembered.filter(|id| {
            session
                .mask_stacks
                .stack(kind)
                .layers
                .iter()
                .any(|layer| layer.layer_id == *id && layer.scene_index == scene)
        });
        let first = session
            .mask_stacks
            .stack(kind)
            .layers
            .iter()
            .find(|layer| layer.scene_index == scene)
            .map(|layer| layer.layer_id);
        let selected = valid.or(first);
        session.mask_stacks.stack_mut(kind).select(selected);
        session.mask_ui.remember(kind, scene, selected);

        struct MaskRow {
            id: u32,
            name: String,
            visible: bool,
            mode: MaskMode,
            color: String,
            selected: bool,
            keys: usize,
            softness: u32,
        }
        let rows: Vec<MaskRow> = session
            .mask_stacks
            .stack(kind)
            .layers
            .iter()
            .filter(|layer| layer.scene_index == scene)
            .map(|layer| MaskRow {
                id: layer.layer_id,
                name: layer.name.clone(),
                visible: layer.visible,
                mode: layer.mode,
                color: layer.color.clone(),
                selected: Some(layer.layer_id) == selected,
                keys: layer.keyframes().len(),
                softness: layer.softness,
            })
            .collect();
        enum MaskAction {
            Select(u32),
            Visible(u32, bool),
            Mode(u32, MaskMode),
            Softness(u32, u32, bool),
            Add,
            Remove,
        }
        let mut actions: Vec<MaskAction> = Vec::new();
        if rows.is_empty() {
            ui.label(
                RichText::new(format!(
                    "No {} masks in this scene.",
                    kind.noun().to_lowercase()
                ))
                .small()
                .color(MUTED),
            );
        }
        for row in &rows {
            ui.horizontal(|ui| {
                let mut visible = row.visible;
                checkbox_widget(ui, &mut visible, "", checkbox_on, checkbox_off);
                if visible != row.visible {
                    actions.push(MaskAction::Visible(row.id, visible));
                }
                let (rect, _) = ui.allocate_exact_size(Vec2::splat(12.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 2.0, mask_color(&row.color));
                if ui.selectable_label(row.selected, &row.name).clicked() {
                    actions.push(MaskAction::Select(row.id));
                }
                let mut mode = row.mode;
                egui::ComboBox::from_id_salt(("mask_mode", row.id))
                    .width(88.0)
                    .selected_text(mode.label())
                    .show_ui(ui, |ui| {
                        for candidate in kind.modes() {
                            ui.selectable_value(&mut mode, *candidate, candidate.label());
                        }
                    });
                if mode != row.mode {
                    actions.push(MaskAction::Mode(row.id, mode));
                }
                if row.keys > 0 {
                    ui.label(
                        RichText::new(if row.keys == 1 {
                            "1 key".to_owned()
                        } else {
                            format!("{} keys", row.keys)
                        })
                        .small()
                        .color(MUTED),
                    );
                }
            });
        }
        if let Some(row) = rows.iter().find(|row| row.selected) {
            ui.label(RichText::new("Softness").small().color(MUTED));
            let mut softness = row.softness as f32;
            let response = value_slider_row(
                ui,
                ("mask_softness", row.id),
                Some(&mut softness),
                0.0..=MASK_SOFTNESS_MAX,
                " px",
                "Feather the selected mask's edge by this many source pixels (0 = sharp edge)",
            );
            // One undo step per slider drag; clicks and typed values record each change.
            let recorded = egui::Id::new(("mask_softness_recorded", row.id));
            if !response.dragging {
                ui.data_mut(|data| data.remove::<bool>(recorded));
            }
            if response.changed {
                let record = !(response.dragging
                    && ui
                        .data(|data| data.get_temp::<bool>(recorded))
                        .unwrap_or(false));
                if response.dragging {
                    ui.data_mut(|data| data.insert_temp(recorded, true));
                }
                actions.push(MaskAction::Softness(
                    row.id,
                    softness.round().max(0.0) as u32,
                    record,
                ));
            }
        }
        ui.horizontal(|ui| {
            if ui.button("+ Add layer").clicked() {
                actions.push(MaskAction::Add);
            }
            if ui
                .add_enabled(selected.is_some(), egui::Button::new("Remove"))
                .clicked()
            {
                actions.push(MaskAction::Remove);
            }
        });
        for action in actions {
            match action {
                MaskAction::Select(id) => session.select_mask_layer(kind, id),
                MaskAction::Visible(id, visible) => {
                    session.set_mask_layer_visible(kind, id, visible);
                }
                MaskAction::Mode(id, mode) => {
                    session.set_mask_layer_mode(kind, id, mode);
                }
                MaskAction::Softness(id, softness, record) => {
                    session.set_mask_layer_softness(kind, id, softness, record);
                }
                MaskAction::Add => {
                    session.add_mask_layer(kind);
                }
                MaskAction::Remove => {
                    if let Some(id) = session.mask_stacks.stack(kind).selected {
                        session.remove_mask_layer(kind, id);
                    }
                }
            }
        }
    }

    fn queue_controls(&mut self, ui: &mut egui::Ui) {
        let mut cancel = false;
        match &self.active {
            Some(active) => {
                let frames = format!(
                    "frames {}–{}",
                    active.queued.range.start + 1,
                    active.queued.range.end_exclusive
                );
                theme::card(ui, &active.queued.label, Some(&frames), |ui| {
                    ui.add(
                        egui::ProgressBar::new(active.progress.fraction())
                            .fill(ACCENT)
                            .corner_radius(4)
                            .show_percentage(),
                    );
                    ui.label(RichText::new(active.progress.stage_text()).color(theme::TEXT));
                    if !active.message.is_empty() {
                        theme::hint(ui, &active.message);
                    }
                    cancel = ui
                        .add(theme::secondary_button("Cancel job", ui.available_width()))
                        .clicked();
                });
            }
            None => {
                theme::card(ui, "Jobs", None, |ui| {
                    theme::hint(ui, "No job running. Run a scene from the Keyer tab.");
                });
            }
        }
        if cancel {
            self.cancel();
        }
        if !self.jobs.is_empty() {
            let waiting = format!("{}", self.jobs.len());
            let mut clear = false;
            theme::card(ui, "Waiting", Some(&waiting), |ui| {
                for (index, job) in self.jobs.iter().enumerate() {
                    theme::field_row(ui, &format!("{}. {}", index + 1, job.label), |ui| {
                        ui.label(
                            RichText::new(format!(
                                "{}–{}",
                                job.range.start + 1,
                                job.range.end_exclusive
                            ))
                            .color(theme::TEXT_DIM),
                        );
                    });
                }
                clear = ui
                    .add(theme::secondary_button(
                        "Clear waiting jobs",
                        ui.available_width(),
                    ))
                    .clicked();
            });
            if clear {
                self.jobs.clear();
            }
        }
        if ui
            .add(theme::secondary_button(
                "Open job log",
                ui.available_width(),
            ))
            .clicked()
        {
            self.log_open = true;
        }
    }

    fn toggle_playback(&mut self) {
        self.playing = !self.playing;
        if self.playing
            && let Some(session) = self.current_mut()
            && !session.range.contains(session.playhead)
        {
            session.seek(session.range.start);
        }
        self.playback_tick = Instant::now();
        self.playback_fraction = 0.0;
    }

    fn timeline(&mut self, ctx: &egui::Context) {
        const TRANSPORT_HEIGHT: f32 = 30.0;
        let margin = egui::Margin::same(6);
        let spacing = ctx.style().spacing.item_spacing.y;
        // A normal button is at least interact_size.y, even if its explicit
        // min_size is smaller. Reserve the same height that the footer uses:
        // undercounting it grows the remembered panel on every repaint.
        let status_height = ctx.style().spacing.interact_size.y.max(24.0);
        let minimum_height = margin.sum().y
            + TRANSPORT_HEIGHT
            + spacing * 2.0
            + 2.0
            + timeline::MIN_TRACK_HEIGHT
            + status_height;
        egui::TopBottomPanel::bottom(TIMELINE_PANEL_ID)
            .resizable(true)
            .default_height(minimum_height.max(224.0))
            .min_height(minimum_height)
            .max_height(ctx.content_rect().height() * 0.7)
            .show_separator_line(true)
            .frame(egui::Frame::new().fill(theme::BG_APP).inner_margin(margin))
            .show(ctx, |ui| {
                let Some(session) = self.current() else {
                    ui.label("Open media to start your timeline");
                    self.timeline_rect = ui.min_rect();
                    return;
                };
                let frame_count = session.media().frame_count;
                let current_frame = session.playhead;
                let rate = fps(session.media());
                let range = session.range;
                let scenes = session.project.scenes.clone();
                let scene_despill = (0..scenes.len())
                    .map(|index| scene_despill_override(&session.project, index))
                    .collect::<Vec<_>>();
                let global_despill = session.project.despill_settings.amount;
                let lanes = session
                    .frames
                    .iter()
                    .map(|(&frame, files)| {
                        (frame, session.has_subject_mask(files), files.rgba.is_some())
                    })
                    .collect::<Vec<_>>();
                let ram_media = self.preview_cache.resident_frames();
                let result_signature = self.current().map(|session| {
                    preview_composition_signature(
                        session,
                        ViewMode::Composite,
                        self.overlay,
                        self.background,
                    )
                });
                let ram_results = result_signature
                    .map(|signature| self.result_cache.resident_frames_with_prefix(&signature))
                    .unwrap_or_default();
                let mask_show = session.mask_ui.show_timeline;
                let mask_scene =
                    timeline::scene_index_at_frame(&session.project.scenes, session.playhead);
                let mask_keyframes: Vec<i64> = if mask_show {
                    session
                        .active_mask_layer(session.mask_ui.active_kind)
                        .map(|layer| layer.keyframes())
                        .unwrap_or_default()
                } else {
                    Vec::new()
                };
                let idle = self.active.is_none() && self.jobs.is_empty() && !self.project_io_busy;
                let mut seek = None;
                let mut frame_display = current_frame + 1;
                let mut in_display = range.start + 1;
                let mut out_display = range.end_exclusive;
                let mut range_changed = false;
                let center_width = 368.0;
                let outer =
                    ((ui.available_width() - center_width - ui.spacing().item_spacing.x * 2.0)
                        * 0.5)
                        .max(0.0);
                ui.horizontal(|ui| {
                    ui.allocate_ui_with_layout(
                        Vec2::new(outer, TRANSPORT_HEIGHT),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            ui.set_min_width(outer);
                            ui.label("Frame");
                            if ui
                                .add(
                                    egui::DragValue::new(&mut frame_display)
                                        .range(1..=frame_count)
                                        .speed(1.0),
                                )
                                .changed()
                            {
                                seek = Some(frame_display - 1);
                            }
                            ui.label(
                                RichText::new(format!("/ {frame_count}"))
                                    .size(13.0)
                                    .color(MUTED),
                            );
                            if outer > 260.0 {
                                ui.label(
                                    RichText::new(timecode(current_frame, rate))
                                        .font(FontId::monospace(13.0))
                                        .color(MUTED),
                                );
                            }
                        },
                    );
                    ui.allocate_ui_with_layout(
                        Vec2::new(center_width, TRANSPORT_HEIGHT),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            ui.set_min_width(center_width);
                            ui.spacing_mut().item_spacing.x = 6.0;
                            ui.spacing_mut().button_padding = Vec2::new(8.0, 4.0);
                            ui.label("In");
                            range_changed |= ui
                                .add(egui::DragValue::new(&mut in_display).range(1..=frame_count))
                                .on_hover_text("Playback range In (I)")
                                .changed();
                            if ui
                                .add(
                                    self.icons
                                        .button(Icon::RangeStart, None)
                                        .min_size(Vec2::new(32.0, 28.0)),
                                )
                                .on_hover_text("Go to In (Home)")
                                .clicked()
                            {
                                seek = Some(range.start);
                            }
                            if ui
                                .add(
                                    self.icons
                                        .button(Icon::StepBack, None)
                                        .min_size(Vec2::new(32.0, 28.0)),
                                )
                                .on_hover_text("Previous frame (Left)")
                                .clicked()
                            {
                                seek = Some(current_frame.saturating_sub(1));
                            }
                            if ui
                                .add(
                                    self.icons
                                        .button(
                                            if self.playing {
                                                Icon::Pause
                                            } else {
                                                Icon::Play
                                            },
                                            None,
                                        )
                                        .min_size(Vec2::new(40.0, 28.0)),
                                )
                                .on_hover_text("Play / pause (Space)")
                                .clicked()
                            {
                                self.toggle_playback();
                            }
                            if ui
                                .add(
                                    self.icons
                                        .button(Icon::StepForward, None)
                                        .min_size(Vec2::new(32.0, 28.0)),
                                )
                                .on_hover_text("Next frame (Right)")
                                .clicked()
                            {
                                seek = Some((current_frame + 1).min(frame_count - 1));
                            }
                            if ui
                                .add(
                                    self.icons
                                        .button(Icon::RangeEnd, None)
                                        .min_size(Vec2::new(32.0, 28.0)),
                                )
                                .on_hover_text("Go to Out (End)")
                                .clicked()
                            {
                                seek = Some(range.end_exclusive - 1);
                            }
                            // Value first, label to its right: "184 Out".
                            range_changed |= ui
                                .add(egui::DragValue::new(&mut out_display).range(1..=frame_count))
                                .on_hover_text("Playback range Out (O)")
                                .changed();
                            ui.label("Out");
                        },
                    );
                    ui.allocate_ui_with_layout(
                        Vec2::new(outer, TRANSPORT_HEIGHT),
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            ui.set_min_width(outer);
                            egui::ComboBox::from_id_salt("test_background")
                                .width(if outer < 250.0 { 72.0 } else { 110.0 })
                                .selected_text(self.background.label())
                                .show_ui(ui, |ui| {
                                    for bg in Background::ALL {
                                        ui.selectable_value(&mut self.background, bg, bg.label());
                                    }
                                });
                            if self.zoom.dropdown(ui, self.viewer_fit) {
                                self.pan = Vec2::ZERO;
                            }
                            let (cache_used, cache_budget) = self.ram_cache_usage();
                            let media_residents = self.preview_cache.resident_frames().len();
                            let result_signature = self.current().map(|session| {
                                preview_composition_signature(
                                    session,
                                    ViewMode::Composite,
                                    self.overlay,
                                    self.background,
                                )
                            });
                            let result_residents = result_signature
                                .map(|signature| {
                                    self.result_cache
                                        .resident_frames_with_prefix(&signature)
                                        .len()
                                })
                                .unwrap_or(0);
                            ui.label(
                                RichText::new(format!(
                                    "RAM {:.1}/{:.1} GB",
                                    crate::memory_budget::gib(cache_used as u64),
                                    crate::memory_budget::gib(cache_budget as u64),
                                ))
                                .small()
                                .color(MUTED),
                            )
                            .on_hover_text(format!(
                                "RAM cache: {}% of {:.1} GB physical memory\nMedia frames: {} ({:.0} MB)\nResult frames: {} ({:.0} MB)\nMask rasters: {:.0} MB",
                                self.cache_settings.percent,
                                crate::memory_budget::gib(self.ram_total),
                                media_residents,
                                self.preview_cache.bytes() as f64 / (1024.0 * 1024.0),
                                result_residents,
                                self.result_cache.bytes() as f64 / (1024.0 * 1024.0),
                                self.mask_raster_bytes() as f64 / (1024.0 * 1024.0),
                            ));
                            let mut mask_visible = mask_show;
                            if self
                                .icons
                                .checkbox(ui, &mut mask_visible, "Masks")
                                .on_hover_text("Show the selected mask keyframes on the timeline")
                                .changed()
                                && let Some(session) = self.current_mut()
                            {
                                session.mask_ui.show_timeline = mask_visible;
                            }
                        },
                    );
                });
                ui.add_space(2.0);
                let mut track_view = std::mem::take(&mut self.current_mut().unwrap().timeline_view);
                let toggle_on = self.icons.checkbox_texture(true);
                let toggle_off = self.icons.checkbox_texture(false);
                let action = {
                    let mut mask_lane = mask_show.then(|| timeline::MaskLaneInput {
                        keyframes: &mask_keyframes,
                        scene: mask_scene,
                        selected: &mut self.mask_edit.timeline_selected,
                        drag: &mut self.mask_edit.timeline_drag,
                    });
                    timeline::show_track(
                        ui,
                        timeline::TrackInput {
                            frame_count,
                            playhead: current_frame,
                            range,
                            scenes: &scenes,
                            lanes: &lanes,
                            scene_controls_enabled: idle,
                            toggle_on: Some(toggle_on),
                            toggle_off: Some(toggle_off),
                            mask_lane: mask_lane.take(),
                            scene_despill: &scene_despill,
                            global_despill,
                            ram_media: &ram_media,
                            ram_results: &ram_results,
                            height: (ui.available_height()
                                - status_height
                                - ui.spacing().item_spacing.y)
                                .max(timeline::MIN_TRACK_HEIGHT),
                        },
                        timeline::TrackColors {
                            accent: ACCENT,
                            mask: GREEN,
                            muted: MUTED,
                        },
                        &mut track_view,
                    )
                };
                self.current_mut().unwrap().timeline_view = track_view;
                if let Some(
                    timeline::TrackAction::SetPlaybackRange(_)
                    | timeline::TrackAction::SetSceneEnabled { .. }
                    | timeline::TrackAction::SplitScene
                    | timeline::TrackAction::MergeNext
                    | timeline::TrackAction::MergePrevious,
                ) = &action
                {
                    self.record_app_edit(crate::app_history::AppContext::Timeline, "timeline edit");
                }
                match action {
                    Some(timeline::TrackAction::Seek(frame)) => {
                        self.current_mut().unwrap().seek(frame);
                        self.playing = false;
                    }
                    Some(timeline::TrackAction::SetSceneDespill { scene, amount }) => {
                        self.record_app_edit(
                            crate::app_history::AppContext::Timeline,
                            "scene despill override",
                        );
                        if let Some(session) = self.current_mut()
                            && set_scene_despill(&mut session.project, scene, amount)
                        {
                            session.revision = session.revision.wrapping_add(1);
                            self.requested_key = None;
                            ui.ctx().request_repaint();
                        }
                    }
                    Some(timeline::TrackAction::SetSceneEnabled { scene, enabled }) => {
                        if let Some(item) =
                            self.current_mut().unwrap().project.scenes.get_mut(scene)
                        {
                            item.enabled = enabled;
                        }
                    }
                    Some(timeline::TrackAction::MoveMaskKeys { delta }) => {
                        let selection = self.mask_edit.timeline_selected.clone();
                        let mut moved: Option<std::collections::BTreeSet<i64>> = None;
                        if delta != 0
                            && !selection.is_empty()
                            && let Some(session) = self.current_mut()
                        {
                            let kind = session.mask_ui.active_kind;
                            let selected = session.mask_stacks.stack(kind).selected;
                            session.record_mask_edit();
                            if let Some(id) = selected
                                && let Some(layer) = session
                                    .mask_stacks
                                    .stack_mut(kind)
                                    .layers
                                    .iter_mut()
                                    .find(|layer| layer.layer_id == id)
                            {
                                moved = Some(crate::mask_editor::move_mask_keyframes(
                                    layer, &selection, delta,
                                ));
                                session.mask_revision = session.mask_revision.wrapping_add(1);
                            }
                        }
                        if let Some(moved) = moved {
                            self.mask_edit.timeline_selected = moved;
                            self.requested_key = None;
                        }
                    }
                    Some(timeline::TrackAction::SelectScene(scene)) => {
                        self.current_mut().unwrap().select_scene(scene);
                        self.playing = false;
                    }
                    Some(timeline::TrackAction::SetPlaybackRange(range)) => {
                        self.current_mut().unwrap().range = range;
                    }
                    Some(timeline::TrackAction::SplitScene) => self.split_scene(),
                    Some(timeline::TrackAction::MergeNext) => self.merge_scene(),
                    Some(timeline::TrackAction::MergePrevious) => self.merge_previous_scene(),
                    None => {}
                }
                if range_changed {
                    self.record_app_edit(
                        crate::app_history::AppContext::Timeline,
                        "playback range",
                    );
                    self.current_mut().unwrap().range = FrameRange {
                        start: in_display.min(out_display).saturating_sub(1),
                        end_exclusive: out_display.max(in_display),
                    };
                }
                if let Some(frame) = seek {
                    self.current_mut().unwrap().seek(frame);
                    self.playing = false;
                }
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), status_height),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.spacing_mut().button_padding = Vec2::new(7.0, 3.0);
                        let progress = self.active.as_ref().map(|active| StatusProgress {
                            label: &active.queued.label,
                            stage: active.progress.stage_text(),
                            fraction: active.progress.fraction(),
                            message: &active.message,
                        });
                        let cancel = status_strip(ui, &self.toast, progress);
                        if cancel.clicked() {
                            self.cancel();
                        }
                    },
                );
                self.timeline_rect = ui.min_rect();
            });
    }
    /// Write an edited bezier shape back into the selected mask layer.
    fn commit_mask_shape(&mut self, kind: MaskKind, layer_id: u32, frame: i64, shape: BezierShape) {
        let Some(session) = self.current_mut() else {
            return;
        };
        if let Some(layer) = session
            .mask_stacks
            .stack_mut(kind)
            .layers
            .iter_mut()
            .find(|layer| layer.layer_id == layer_id)
        {
            layer.set_shape(frame, shape);
            if !layer.visible {
                layer.visible = true;
            }
            session.mask_revision = session.mask_revision.wrapping_add(1);
        }
        self.requested_key = None;
    }

    fn active_mask_tab(&self) -> Option<MaskKind> {
        match self.left_tab {
            LeftTab::Garbage => Some(MaskKind::Garbage),
            LeftTab::Despill => Some(MaskKind::Despill),
            LeftTab::Media | LeftTab::Subject => None,
        }
    }

    fn app_snapshot(&self) -> Option<crate::app_history::AppSnapshot> {
        let session = self.current()?;
        Some(crate::app_history::AppSnapshot {
            session_key: session.key.clone(),
            project: session.project.clone(),
            range: session.range,
            playhead: session.playhead,
            embed_sources: self.embed_sources,
        })
    }

    fn record_app_edit(&mut self, context: crate::app_history::AppContext, label: &'static str) {
        if let Some(snapshot) = self.app_snapshot() {
            self.app_history.record(context, label, snapshot);
        }
    }

    fn restore_app_snapshot(&mut self, snapshot: crate::app_history::AppSnapshot) -> bool {
        let Some(index) = self
            .sessions
            .iter()
            .position(|session| session.key == snapshot.session_key)
        else {
            return false;
        };
        self.selected = index;
        let session = &mut self.sessions[index];
        // Media imports and the runtime mask stacks are not part of this
        // snapshot; keep them while restoring the editable document state.
        let media = session.project.media.clone();
        let mask_layers = session.project.mask_layers.clone();
        let mut project = snapshot.project;
        project.media = media;
        project.mask_layers = mask_layers;
        session.project = project;
        session.range = snapshot.range;
        let last_frame = session.media().frame_count.saturating_sub(1);
        session.playhead = snapshot.playhead.min(last_frame);
        session.selected_scene =
            timeline::scene_index_at_frame(&session.project.scenes, session.playhead).unwrap_or(0);
        session.revision = session.revision.wrapping_add(1);
        session.sync_mask_selection();
        self.embed_sources = snapshot.embed_sources;
        self.loaded = None;
        self.requested_key = None;
        self.playing = false;
        true
    }

    fn pointer_undo_context(&self, pointer: Option<Pos2>) -> crate::app_history::AppContext {
        use crate::app_history::AppContext;
        let Some(pointer) = pointer else {
            return AppContext::Global;
        };
        if self.viewer_canvas.contains(pointer) {
            return AppContext::Mask;
        }
        if self.timeline_rect.contains(pointer) {
            return AppContext::Timeline;
        }
        if self.inspector_rect.contains(pointer) {
            return AppContext::Keyer;
        }
        if self.left_panel_rect.contains(pointer) {
            return if self.active_mask_tab().is_some() {
                AppContext::Mask
            } else {
                AppContext::Media
            };
        }
        AppContext::Global
    }

    fn undo_at_pointer(&mut self, ctx: &egui::Context) {
        use crate::app_history::AppContext;
        let context = self.pointer_undo_context(ctx.pointer_hover_pos());
        if context == AppContext::Mask
            && let Some(session) = self.current_mut()
            && session.undo_mask_edit()
        {
            return;
        }
        if let Some(current) = self.app_snapshot()
            && let Some(snapshot) = self.app_history.undo_for(context, current.clone())
        {
            self.restore_app_snapshot(snapshot);
            return;
        }
        if let Some(snapshot) = self
            .app_snapshot()
            .and_then(|current| self.app_history.undo_for(AppContext::Global, current))
        {
            self.restore_app_snapshot(snapshot);
            return;
        }
        if let Some(session) = self.current_mut() {
            session.undo_mask_edit();
        }
    }

    fn redo_at_pointer(&mut self, ctx: &egui::Context) {
        use crate::app_history::AppContext;
        let context = self.pointer_undo_context(ctx.pointer_hover_pos());
        if context == AppContext::Mask
            && let Some(session) = self.current_mut()
            && session.redo_mask_edit()
        {
            return;
        }
        if let Some(current) = self.app_snapshot()
            && let Some(snapshot) = self.app_history.redo_for(context, current.clone())
        {
            self.restore_app_snapshot(snapshot);
            return;
        }
        if let Some(snapshot) = self
            .app_snapshot()
            .and_then(|current| self.app_history.redo_for(AppContext::Global, current))
        {
            self.restore_app_snapshot(snapshot);
            return;
        }
        if let Some(session) = self.current_mut() {
            session.redo_mask_edit();
        }
    }

    fn viewer(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(8)).show(ctx, |ui| {
            let modes = [ViewMode::Source, ViewMode::Composite, ViewMode::Alpha, ViewMode::Sam];
            let selected = modes.iter().position(|mode| *mode == self.mode).unwrap_or(0);
            let labels = modes.map(|mode| if mode == ViewMode::Composite { "Result" } else { mode.label() });
            self.mode = modes[theme::tab_bar(ui, "viewer_tabs", &labels, selected)];
            let (canvas, response) = ui.allocate_exact_size(ui.available_size().max(Vec2::splat(1.0)), Sense::click_and_drag());
            let painter = ui.painter_at(canvas);
            self.viewer_canvas = canvas;
            painter.rect_filled(canvas, 0, Color32::from_gray(12));
            if self.sessions.is_empty() {
                painter.text(canvas.center() - Vec2::new(0.0, 22.0), egui::Align2::CENTER_CENTER, "BRP CHROMA", FontId::proportional(25.0), theme::TEXT);
                painter.text(canvas.center() + Vec2::new(0.0, 15.0), egui::Align2::CENTER_CENTER, "Open media or drop a clip here", FontId::proportional(14.0), MUTED);
                return;
            }
            // Subject prompts (points, point text, boxes) are only shown and edited on the Subject tab.
            let subject_tab = self.left_tab == LeftTab::Subject;
            if !subject_tab {
                self.box_drag = None;
                self.point_text = None;
                self.point_drag = None;
            }
            let session = self.current().expect("nonempty session selection");
            let source_size = Vec2::new(session.media().width as f32, session.media().height as f32);
            let playhead = session.playhead;
            let prompts = if subject_tab { session.project.prompts.clone() } else { Vec::new() };
            let boxes = if subject_tab { session.project.sam_boxes.clone() } else { Vec::new() };
            let sam_layer = session.sam_layer;
            let density = ctx.pixels_per_point();
            let fit = ViewerZoom::fit_scale(canvas.size(), source_size, density);
            self.viewer_fit = fit;
            if response.hovered() {
                let scroll = ctx.input(|input| input.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    let old = self.zoom.physical_scale(fit);
                    let new = (old * (scroll * 0.0025).exp()).clamp(0.005, 64.0);
                    self.zoom = ViewerZoom::Custom(new);
                    if let Some(pointer) = ctx.pointer_hover_pos() {
                        let offset = pointer - canvas.center();
                        self.pan = offset - (offset - self.pan) * (new / old);
                    }
                }
            }
            if response.dragged_by(egui::PointerButton::Middle) { self.pan += ctx.input(|input| input.pointer.delta()); }
            self.viewer_scale = self.zoom.physical_scale(fit);
            let image_rect = crate::viewport::image_rect(canvas.center() + self.pan, source_size, self.viewer_scale, density);
            let size = image_rect.size();
            self.viewer_image = image_rect;
            if let Some(loaded) = &self.loaded {
                painter.image(loaded.texture.id(), image_rect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
                let frame = loaded.frame;
                // Other layers' points are drawn faded; only the selected layer's are editable.
                let mut ordered: Vec<&SamPrompt> = prompts.iter().filter(|point| point.frame == frame).collect();
                ordered.sort_by_key(|point| point.layer == sam_layer);
                for point in ordered {
                    let position = image_rect.min + Vec2::new(point.x / source_size.x * size.x, point.y / source_size.y * size.y);
                    let base = if point.positive { GREEN } else { theme::BAD };
                    let color = if point.layer == sam_layer { base } else { base.gamma_multiply(0.35) };
                    painter.circle_filled(position, 6.0, Color32::from_black_alpha(if point.layer == sam_layer { 150 } else { 60 }));
                    painter.circle_stroke(position, 5.0, Stroke::new(2.0_f32, color));
                    painter.line_segment([position - Vec2::new(3.0, 0.0), position + Vec2::new(3.0, 0.0)], Stroke::new(1.5_f32, color));
                    if point.positive { painter.line_segment([position - Vec2::new(0.0, 3.0), position + Vec2::new(0.0, 3.0)], Stroke::new(1.5_f32, color)); }
                    if let Some(text) = point.text.as_deref().filter(|t| !t.trim().is_empty()) {
                        let galley = painter.layout_no_wrap(text.to_owned(), FontId::proportional(12.0), color);
                        let at = position + Vec2::new(10.0, -galley.size().y / 2.0);
                        painter.rect_filled(Rect::from_min_size(at - Vec2::new(4.0, 2.0), galley.size() + Vec2::new(8.0, 4.0)), 4.0, Color32::from_black_alpha(170));
                        painter.galley(at, galley, color);
                    }
                }
                // Boxes (selected layer solid, others faded) and the box being dragged.
                let to_screen = |x: f32, y: f32| image_rect.min + Vec2::new(x / source_size.x * size.x, y / source_size.y * size.y);
                for b in boxes.iter().filter(|b| b.frame == frame) {
                    let color = if b.layer == sam_layer { GREEN } else { GREEN.gamma_multiply(0.35) };
                    painter.rect_stroke(Rect::from_two_pos(to_screen(b.x0, b.y0), to_screen(b.x1, b.y1)), 2.0, Stroke::new(2.0_f32, color), egui::StrokeKind::Middle);
                }
                if let Some((box_frame, _, a, b)) = self.box_drag
                    && box_frame == frame
                {
                    painter.rect_stroke(Rect::from_two_pos(to_screen(a.x, a.y), to_screen(b.x, b.y)), 2.0, Stroke::new(1.5_f32, theme::ACCENT), egui::StrokeKind::Middle);
                }
            } else {
                painter.text(canvas.center(), egui::Align2::CENTER_CENTER, "Loading source frame…", FontId::proportional(17.0), MUTED);
                ctx.request_repaint_after(Duration::from_millis(30));
            }
            // Latest pointer during a mask gesture (may lie outside the image,
            // so knots can be placed and dragged beyond the video area).
            let pointer_position = response
                .interact_pointer_pos()
                .or_else(|| response.dragged().then(|| ctx.input(|i| i.pointer.latest_pos())).flatten());
            // On a mask tab the first click/drag should just work: create the
            // scene's layer when none is selected yet.
            if let Some(kind) = self.active_mask_tab()
                && (response.drag_started() || response.clicked())
                && pointer_position.is_some()
                && let Some(session) = self.current_mut()
                && session.mask_ui.editing
                && session.active_mask_layer(kind).is_none()
            {
                session.add_mask_layer(kind);
            }
            let mask_overlay_info = self.current().and_then(|session| {
                if !session.mask_ui.editing {
                    return None;
                }
                let kind = session.mask_ui.active_kind;
                session.active_mask_layer(kind).map(|layer| {
                    let edit_frame = layer.shape_edit_frame(playhead as i64);
                    let shape = layer
                        .shape_at(playhead as i64)
                        .unwrap_or_else(BezierShape::new);
                    (kind, layer.layer_id, edit_frame, shape)
                })
            });
            if let Some((mask_kind, layer_id, edit_frame, shape)) = mask_overlay_info.clone() {
                // Masks may extend past the video area: edit anywhere in the
                // canvas, not only inside the image rectangle.
                let pointer_source = pointer_position.map(|position| {
                    crate::mask_overlay::screen_to_source(image_rect, source_size, position)
                });
                let tolerance = crate::mask_overlay::HIT_RADIUS
                    * crate::mask_overlay::source_per_screen(image_rect, source_size);
                if response.drag_started()
                    && let Some(current) = pointer_source
                {
                    // egui reports a drag only after the pointer has moved a few
                    // pixels, so a quick press-and-drag may already be off the knot.
                    // Hit-test and measure the gesture from where it was pressed.
                    let point = ctx
                        .input(|input| input.pointer.press_origin())
                        .map(|position| {
                            crate::mask_overlay::screen_to_source(image_rect, source_size, position)
                        })
                        .unwrap_or(current);
                    let modifiers = ctx.input(|input| input.modifiers);
                    let gizmo_hit = self
                        .mask_edit
                        .gizmo_open
                        .then(|| {
                            crate::mask_overlay::gizmo_hit(
                                crate::mask_overlay::shape_bounds(&shape),
                                self.mask_edit.gizmo_angle,
                                point,
                                tolerance,
                            )
                        })
                        .flatten();
                    if let Some(hit) = gizmo_hit {
                        if let Some(session) = self.current_mut() {
                            session.record_mask_edit();
                        }
                        let gizmo = match hit {
                            crate::mask_overlay::GizmoHit::Move => {
                                crate::mask_overlay::GizmoDrag::Move
                            }
                            crate::mask_overlay::GizmoHit::Scale(_) => {
                                crate::mask_overlay::GizmoDrag::Scale
                            }
                            crate::mask_overlay::GizmoHit::Rotate => {
                                crate::mask_overlay::GizmoDrag::Rotate
                            }
                        };
                        self.mask_edit.drag = Some(crate::mask_overlay::MaskDrag {
                            kind: crate::mask_overlay::MaskDragKind::Gizmo(gizmo),
                            layer_id,
                            frame: edit_frame,
                            start_shape: shape.clone(),
                            start_pointer: point,
                            start_angle: self.mask_edit.gizmo_angle,
                        });
                    } else if let Some(control) =
                        crate::mask_overlay::control_at(&shape, point, tolerance)
                    {
                        if modifiers.alt {
                            let mut edited = shape.clone();
                            match control {
                                crate::mask_overlay::MaskControl::Anchor(index)
                                | crate::mask_overlay::MaskControl::In(index)
                                | crate::mask_overlay::MaskControl::Out(index) => {
                                    edited.reset_handles(index);
                                }
                            }
                            if let Some(session) = self.current_mut() {
                                session.record_mask_edit();
                            }
                            self.commit_mask_shape(mask_kind, layer_id, edit_frame, edited);
                        } else {
                            let smooth = modifiers.shift
                                && matches!(control, crate::mask_overlay::MaskControl::Anchor(_));
                            if let Some(session) = self.current_mut() {
                                session.record_mask_edit();
                            }
                            self.mask_edit.selected_knot = Some(match control {
                                crate::mask_overlay::MaskControl::Anchor(index)
                                | crate::mask_overlay::MaskControl::In(index)
                                | crate::mask_overlay::MaskControl::Out(index) => index,
                            });
                            self.mask_edit.drag = Some(crate::mask_overlay::MaskDrag {
                                kind: crate::mask_overlay::MaskDragKind::Control(control, smooth),
                                layer_id,
                                frame: edit_frame,
                                start_shape: shape.clone(),
                                start_pointer: point,
                                start_angle: 0.0,
                            });
                        }
                    } else {
                        // Press-and-drag on empty canvas seeds a smooth node: the
                        // press point is the anchor and the drag pulls its tangents.
                        let origin = point;
                        if let Some(session) = self.current_mut() {
                            session.record_mask_edit();
                        }
                        self.mask_edit.selected_knot = Some(shape.nodes.len());
                        self.mask_edit.drag = Some(crate::mask_overlay::MaskDrag {
                            kind: crate::mask_overlay::MaskDragKind::CreateNode,
                            layer_id,
                            frame: edit_frame,
                            start_shape: shape.clone(),
                            start_pointer: origin,
                            start_angle: 0.0,
                        });
                    }
                }
                if response.dragged()
                    && let Some(drag) = self.mask_edit.drag.clone()
                    && let Some(point) = pointer_source
                {
                    let edited = crate::mask_overlay::drag_shape(&drag, point);
                    if let crate::mask_overlay::MaskDragKind::Gizmo(
                        crate::mask_overlay::GizmoDrag::Rotate,
                    ) = drag.kind
                    {
                        let centre = crate::mask_overlay::shape_bounds(&drag.start_shape).center();
                        let start = (drag.start_pointer.y - centre.y)
                            .atan2(drag.start_pointer.x - centre.x);
                        let current = (point.y - centre.y).atan2(point.x - centre.x);
                        self.mask_edit.gizmo_angle = drag.start_angle + current - start;
                    }
                    self.commit_mask_shape(mask_kind, drag.layer_id, drag.frame, edited);
                }
                if response.drag_stopped() {
                    if let Some(drag) = self.mask_edit.drag.as_ref()
                        && matches!(
                            drag.kind,
                            crate::mask_overlay::MaskDragKind::CreateNode
                        )
                    {
                        self.mask_edit.selected_knot = Some(drag.start_shape.nodes.len());
                        self.mask_edit.gizmo_open = false;
                    }
                    self.mask_edit.drag = None;
                    self.mask_edit.gizmo_angle = 0.0;
                }
                if response.double_clicked()
                    && let Some(point) = pointer_source
                {
                    if self.mask_edit.gizmo_open {
                        self.mask_edit.gizmo_open = false;
                        self.mask_edit.selected_knot = None;
                    } else if let Some(crate::mask_overlay::MaskControl::Anchor(index)) =
                        crate::mask_overlay::control_at(&shape, point, tolerance)
                    {
                        self.mask_edit.selected_knot = Some(index);
                        self.mask_edit.gizmo_open = true;
                        self.mask_edit.gizmo_angle = 0.0;
                    }
                } else if response.clicked()
                    && self.mask_edit.drag.is_none()
                    && let Some(point) = pointer_source
                {
                    let mut edited = shape.clone();
                    if self.mask_edit.gizmo_open {
                        self.mask_edit.gizmo_open = false;
                    } else if let Some(control) =
                        crate::mask_overlay::control_at(&edited, point, tolerance)
                    {
                        let index = match control {
                            crate::mask_overlay::MaskControl::Anchor(index)
                            | crate::mask_overlay::MaskControl::In(index)
                            | crate::mask_overlay::MaskControl::Out(index) => index,
                        };
                        let is_anchor = matches!(
                            control,
                            crate::mask_overlay::MaskControl::Anchor(_)
                        );
                        let close_gesture = is_anchor
                            && !edited.closed
                            && edited.nodes.len() >= 2
                            && (index == 0 || index + 1 == edited.nodes.len());
                        let mut changed = false;
                        if ctx.input(|input| input.modifiers.shift) {
                            edited.remove_node(index);
                            self.mask_edit.selected_knot = None;
                            changed = true;
                        } else if ctx.input(|input| input.modifiers.alt) {
                            edited.reset_handles(index);
                            changed = true;
                        } else if close_gesture {
                            edited.closed = true;
                            self.mask_edit.selected_knot = Some(index);
                            changed = true;
                        } else {
                            self.mask_edit.selected_knot = Some(index);
                        }
                        if changed && let Some(session) = self.current_mut() {
                            session.record_mask_edit();
                        }
                        self.commit_mask_shape(mask_kind, layer_id, edit_frame, edited);
                    } else if let Some((segment, t)) =
                        crate::mask_overlay::segment_at(&edited, point, tolerance)
                    {
                        if let Some(index) = edited.insert_point_exact(segment, t) {
                            if let Some(session) = self.current_mut() {
                                session.record_mask_edit();
                            }
                            self.mask_edit.selected_knot = Some(index);
                            self.commit_mask_shape(mask_kind, layer_id, edit_frame, edited);
                        }
                    } else if edited.nodes.len() >= 2
                        && let Some(first) = edited.nodes.first()
                        && (Pos2::new(first.x, first.y) - point).length() <= tolerance
                    {
                        edited.closed = true;
                        if let Some(session) = self.current_mut() {
                            session.record_mask_edit();
                        }
                        self.commit_mask_shape(mask_kind, layer_id, edit_frame, edited);
                    } else {
                        if let Some(session) = self.current_mut() {
                            session.record_mask_edit();
                        }
                        let index = edited.add_node(point.x, point.y, 0.0, 0.0, true);
                        self.mask_edit.selected_knot = Some(index);
                        self.commit_mask_shape(mask_kind, layer_id, edit_frame, edited);
                    }
                }
                if response.hovered() || self.mask_edit.drag.is_some() {
                    ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
                }
            }
            // Redraw after edits so creating/dragging a knot and its tangents is
            // visible immediately instead of one frame late.
            let mask_overlay_draw = self.current().and_then(|session| {
                if !session.mask_ui.editing {
                    return None;
                }
                let kind = session.mask_ui.active_kind;
                session.active_mask_layer(kind).map(|layer| {
                    layer
                        .shape_at(playhead as i64)
                        .unwrap_or_else(BezierShape::new)
                })
            });
            if let Some(shape) = &mask_overlay_draw {
                let gizmo = self.mask_edit.gizmo_open.then(|| {
                    (
                        crate::mask_overlay::shape_bounds(shape),
                        self.mask_edit.gizmo_angle,
                    )
                });
                crate::mask_overlay::draw(
                    &painter,
                    image_rect,
                    source_size,
                    shape,
                    self.mask_edit.selected_knot,
                    gizmo,
                );
            }
            let current_ready = self.loaded.as_ref().is_some_and(|frame| frame.frame == playhead && self.requested_key.as_ref() == Some(&frame.key));
            // Points can be placed on any view; the frame is segmented again automatically
            // once nothing else is running.
            let editing_points = current_ready && mask_overlay_info.is_none() && !self.playing && self.left_tab == LeftTab::Subject;
            // Selected-layer point under the pointer (screen-space hit radius).
            let point_at = |position: Pos2| -> Option<usize> {
                prompts
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.frame == playhead && p.layer == sam_layer)
                    .map(|(i, p)| {
                        let at = image_rect.min + Vec2::new(p.x / source_size.x * size.x, p.y / source_size.y * size.y);
                        (i, (at - position).length())
                    })
                    .filter(|(_, distance)| *distance <= 9.0)
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(i, _)| i)
            };
            let sam31 = self.effective_sam_model() == SamModel::Sam31;
            let box_tool = sam31 && self.point_tool == PointTool::Box;
            let clamp_source = |position: Pos2| -> Pos2 {
                let clamped = position.clamp(image_rect.min, image_rect.max - Vec2::splat(0.01));
                source_coordinate(image_rect, clamped, source_size).map_or(Pos2::ZERO, |(x, y)| Pos2::new(x, y))
            };
            if editing_points && box_tool {
                // Box tool: drag a rectangle; it replaces the layer's box on this frame.
                let pointer = response.interact_pointer_pos();
                if response.drag_started_by(egui::PointerButton::Primary)
                    && let Some(position) = pointer
                {
                    let start = clamp_source(position);
                    self.box_drag = Some((playhead, sam_layer, start, start));
                }
                if let Some((frame, layer, start, _)) = self.box_drag {
                    if let Some(position) = ctx.input(|i| i.pointer.latest_pos()) {
                        self.box_drag = Some((frame, layer, start, clamp_source(position)));
                        ctx.request_repaint();
                    }
                    if response.drag_stopped() {
                        let (_, _, a, b) = self.box_drag.take().expect("dragging a box");
                        if (a.x - b.x).abs() >= 4.0 && (a.y - b.y).abs() >= 4.0
                            && let Some(session) = self.current_mut()
                        {
                            let layer = if layer == NO_LAYER { session.layer_for_edit() } else { layer };
                            session.project.sam_boxes.retain(|x| !(x.frame == frame && x.layer == layer));
                            session.project.sam_boxes.push(ck_desktop_core::SamBox {
                                frame,
                                layer,
                                x0: a.x.min(b.x),
                                y0: a.y.min(b.y),
                                x1: a.x.max(b.x),
                                y1: a.y.max(b.y),
                            });
                            session.invalidate_prompt_scene(frame, layer);
                            self.segment_pending = Some((frame, layer));
                        }
                    }
                }
                if response.hovered() {
                    ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
                }
            }
            // Right-click inside a box of the selected layer removes it (points take precedence).
            if editing_points
                && sam31
                && response.secondary_clicked()
                && let Some(position) = response.interact_pointer_pos()
                && point_at(position).is_none()
            {
                let at = clamp_source(position);
                if let Some(index) = boxes.iter().position(|b| {
                    b.frame == playhead && b.layer == sam_layer && at.x >= b.x0 && at.x <= b.x1 && at.y >= b.y0 && at.y <= b.y1
                }) && let Some(session) = self.current_mut()
                {
                    let removed = boxes[index].clone();
                    session.project.sam_boxes.retain(|b| *b != removed);
                    session.invalidate_prompt_scene(playhead, sam_layer);
                    self.segment_pending = Some((playhead, sam_layer));
                }
            }
            if editing_points && !box_tool {
                let pointer = response.interact_pointer_pos();
                if response.drag_started_by(egui::PointerButton::Primary)
                    && let Some(index) = pointer.and_then(point_at)
                {
                    self.point_drag = Some(index);
                }
                if let Some(index) = self.point_drag {
                    if response.dragged_by(egui::PointerButton::Primary)
                        && let Some(position) = ctx.input(|i| i.pointer.latest_pos())
                    {
                        // Clamp to the frame so a point can never leave the image.
                        let clamped = position.clamp(image_rect.min, image_rect.max - Vec2::splat(0.01));
                        if let Some((x, y)) = source_coordinate(image_rect, clamped, source_size)
                            && let Some(session) = self.current_mut()
                            && let Some(point) = session.project.prompts.get_mut(index)
                        {
                            point.x = x;
                            point.y = y;
                        }
                        ctx.set_cursor_icon(egui::CursorIcon::Grabbing);
                        ctx.request_repaint();
                    }
                    if response.drag_stopped() {
                        self.point_drag = None;
                        if let Some(session) = self.current_mut() {
                            session.invalidate_prompt_scene(playhead, sam_layer);
                        }
                        self.segment_pending = Some((playhead, sam_layer));
                    }
                } else if response.hovered()
                    && ctx.pointer_hover_pos().and_then(point_at).is_some()
                {
                    ctx.set_cursor_icon(egui::CursorIcon::Grab);
                }
                if response.secondary_clicked()
                    && let Some(index) = pointer.and_then(point_at)
                {
                    if let Some(session) = self.current_mut() {
                        session.project.prompts.remove(index);
                        session.invalidate_prompt_scene(playhead, sam_layer);
                    }
                    self.segment_pending = Some((playhead, sam_layer));
                }
                // SAM 3.1: clicking a point opens its text field.
                if sam31
                    && response.clicked()
                    && let Some(index) = pointer.and_then(point_at)
                {
                    let text = prompts[index].text.clone().unwrap_or_default();
                    self.point_text = Some((index, text));
                }
                // Points can be placed on any view; the frame is segmented again automatically
                // once nothing else is running.
                if response.clicked()
                    && let Some(position) = pointer
                    && point_at(position).is_none()
                    && let Some((x, y)) = source_coordinate(image_rect, position, source_size)
                {
                    let positive = self.point_tool == PointTool::Subject && !ctx.input(|input| input.modifiers.shift);
                    if let Some(session) = self.current_mut() {
                        // The first point of a scene creates its layer.
                        let layer = session.layer_for_edit();
                        session.invalidate_prompt_scene(playhead, layer);
                        session.project.prompts.push(SamPrompt { frame: playhead, x, y, positive, layer, text: None });
                        self.segment_pending = Some((playhead, layer));
                    }
                }
            }
            // Text field beside the point being edited: Enter (or clicking away) applies it, an
            // empty text makes it a plain point again, Escape cancels.
            if let Some((index, mut text)) = self.point_text.take() {
                let valid = editing_points
                    && prompts.get(index).is_some_and(|p| p.frame == playhead && p.layer == sam_layer);
                if valid {
                    let p = &prompts[index];
                    let anchor = image_rect.min + Vec2::new(p.x / source_size.x * size.x, p.y / source_size.y * size.y) + Vec2::new(12.0, -14.0);
                    // Read-only translation of the point's saved text (hidden while it is edited).
                    let committed = p.text.clone().unwrap_or_default();
                    self.translator.request(&committed);
                    let point_caption = (text.trim() == committed.trim())
                        .then(|| crate::translation::caption(&self.translator, &committed))
                        .flatten();
                    let mut done = None;
                    egui::Area::new(egui::Id::new("sam-point-text"))
                        .fixed_pos(anchor)
                        .order(egui::Order::Foreground)
                        .show(ctx, |ui| {
                            egui::Frame::popup(ui.style()).show(ui, |ui| {
                                let field = ui.add(
                                    egui::TextEdit::singleline(&mut text)
                                        .hint_text("text, e.g. hat")
                                        .desired_width(160.0),
                                );
                                if !field.has_focus() && !field.lost_focus() {
                                    field.request_focus();
                                }
                                if let Some(caption) = &point_caption {
                                    ui.label(egui::RichText::new(caption).small().color(MUTED));
                                }
                                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                    done = Some(false);
                                } else if field.lost_focus() {
                                    done = Some(true);
                                }
                            });
                        });
                    match done {
                        None => self.point_text = Some((index, text)),
                        Some(false) => {}
                        Some(true) => {
                            let value = Some(text.trim().to_owned()).filter(|t| !t.is_empty());
                            if let Some(value) = &value {
                                self.translator.request(value);
                            }
                            if let Some(session) = self.current_mut()
                                && let Some(point) = session.project.prompts.get_mut(index)
                                && point.text != value
                            {
                                point.text = value;
                                session.invalidate_prompt_scene(playhead, sam_layer);
                                self.segment_pending = Some((playhead, sam_layer));
                            }
                        }
                    }
                }
            }
            response.on_hover_text("Wheel: zoom around cursor · Middle drag: pan · Drag panel dividers to resize the viewer");
        });
    }

    fn split_scene(&mut self) {
        let Some(session) = self.current_mut() else {
            return;
        };
        let frame = session.playhead;
        let Some(index) = session
            .project
            .scenes
            .iter()
            .position(|scene| frame > scene.start && frame < scene.end_exclusive)
        else {
            return;
        };
        let end = session.project.scenes[index].end_exclusive;
        let enabled = session.project.scenes[index].enabled;
        session.revision += 1;
        session.project.scenes[index].end_exclusive = frame;
        session.project.scenes.insert(
            index + 1,
            SceneRange {
                index: 0,
                start: frame,
                end_exclusive: end,
                enabled,
            },
        );
        for (index, scene) in session.project.scenes.iter_mut().enumerate() {
            scene.index = index as u32;
        }
        session.project.shift_sam_layers_after_split(index as u32);
        session.rehome_sam_layers();
        session.selected_scene = index + 1;
    }

    fn merge_scene_at(&mut self, index: usize) {
        let Some(session) = self.current_mut() else {
            return;
        };
        if index + 1 >= session.project.scenes.len() {
            return;
        }
        session.project.scenes[index].end_exclusive =
            session.project.scenes[index + 1].end_exclusive;
        session.project.scenes.remove(index + 1);
        for (index, scene) in session.project.scenes.iter_mut().enumerate() {
            scene.index = index as u32;
        }
        session.project.shift_sam_layers_after_merge(index as u32);
        session.rehome_sam_layers();
        session.revision += 1;
        session.selected_scene =
            timeline::scene_index_at_frame(&session.project.scenes, session.playhead).unwrap_or(0);
    }

    fn merge_scene(&mut self) {
        if let Some(index) = self.current().and_then(|session| {
            timeline::scene_index_at_frame(&session.project.scenes, session.playhead)
        }) {
            self.merge_scene_at(index);
        }
    }

    fn merge_previous_scene(&mut self) {
        if let Some(index) = self.current().and_then(|session| {
            timeline::scene_index_at_frame(&session.project.scenes, session.playhead)
        }) && index > 0
        {
            self.merge_scene_at(index - 1);
        }
    }
    fn settings(&mut self, ctx: &egui::Context) {
        let mut open = self.open_settings;
        let mut apply = false;
        egui::Window::new("Advanced runtime settings").open(&mut open).default_width(740.0).resizable(true).show(ctx, |ui| {
            if self.active.is_some() || !self.jobs.is_empty() { ui.disable(); }
            ui.label("Required files are managed automatically beside the application. These optional overrides apply to this session only.");
            if ui.button("Restore automatic locations").clicked() { self.runtime = RuntimePaths::installed(&self.app_directory); }
            if ui.button("Choose model-pack root…").clicked()
                && let Some(path) = rfd::FileDialog::new().set_title("Choose model-pack root").pick_folder() { self.runtime.set_model_root(&path); }
            ui.separator();
            egui::Grid::new("runtime_paths").num_columns(2).spacing([14.0, 10.0]).show(ui, |ui| {
                path_field(ui, "Native worker", &mut self.runtime.worker);
                path_field(ui, "FFmpeg", &mut self.runtime.ffmpeg);
                path_field(ui, "FFprobe", &mut self.runtime.ffprobe);
                path_field(ui, "ONNX Runtime", &mut self.runtime.ort_library);
                path_field(ui, "CorridorKey", &mut self.runtime.ck_1080_models);
                path_field(ui, "SAM video", &mut self.runtime.sam_models);
                path_field(ui, "TransNetV2", &mut self.runtime.transnet_model);
                let mut cuda = self.runtime.cuda_library_directory.clone().unwrap_or_default();
                path_field(ui, "CUDA library folder", &mut cuda);
                self.runtime.cuda_library_directory = if cuda.as_os_str().is_empty() { None } else { Some(cuda) };
            });
            ui.add_space(8.0);
            let missing = self.runtime.missing_components();
            if missing.is_empty() { ui.label(RichText::new("Native runtime and all model paths are available").color(GREEN)); }
            else { ui.label(RichText::new(format!("Missing: {}", missing.join(", "))).color(ORANGE)); }
            ui.add_space(8.0);
            if ui.button("Apply settings").clicked() { apply = true; }
            ui.add_space(8.0);
            egui::ComboBox::from_id_salt("advanced_provider").selected_text(match self.provider { Provider::DirectMl => "Windows GPU", Provider::Cpu => "CPU", Provider::Cuda => "CUDA (optional)" }).show_ui(ui, |ui| {
                if cfg!(windows) { ui.selectable_value(&mut self.provider, Provider::DirectMl, "Windows GPU"); }
                ui.selectable_value(&mut self.provider, Provider::Cpu, "CPU");
                ui.selectable_value(&mut self.provider, Provider::Cuda, "CUDA (optional)");
            });
            ui.label(RichText::new("Windows GPU uses the included DirectML runtime. Optional CUDA requires a separately installed compatible runtime and libraries.").small().color(MUTED));
            ui.separator();
            ui.label(RichText::new("RAM CACHE").strong());
            let mut percent = self.cache_settings.percent;
            let slider = ui.add(
                egui::Slider::new(
                    &mut percent,
                    crate::memory_budget::MIN_PERCENT..=crate::memory_budget::MAX_PERCENT,
                )
                .text("Share of physical RAM"),
            );
            if slider.changed() {
                self.cache_settings = crate::memory_budget::CacheSettings::clamped(percent);
                self.apply_cache_settings();
            }
            let limits = self.cache_limits();
            let (used, budget) = self.ram_cache_usage();
            ui.label(format!(
                "{}% of {:.1} GB physical memory → {:.1} GB cache budget (currently {:.1} GB)",
                self.cache_settings.percent,
                crate::memory_budget::gib(self.ram_total),
                crate::memory_budget::gib(budget as u64),
                crate::memory_budget::gib(used as u64),
            ));
            ui.label(
                RichText::new(format!(
                    "Media frames {:.0} MB · result frames {:.0} MB · mask rasters {:.0} MB",
                    limits.source_bytes as f64 / (1024.0 * 1024.0),
                    limits.result_bytes as f64 / (1024.0 * 1024.0),
                    limits.mask_bytes as f64 / (1024.0 * 1024.0),
                ))
                .small()
                .color(MUTED),
            );
            if ui.button("Clear RAM caches").clicked() {
                self.clear_ram_caches();
            }
        });
        self.open_settings = open;
        if apply {
            if self.active.is_some() || !self.jobs.is_empty() {
                self.fail("Finish or cancel the current job before changing runtime files.");
                return;
            }
            self.runtime.resolve_relative_to(&self.app_directory);
            self.preview = PreviewService::new(
                self.runtime.clone(),
                self.work_root.join("source-preview"),
                ctx.clone(),
            );
            self.requested_key = None;
            self.toast = "Advanced overrides applied for this session".into();
            self.open_settings = false;
        }
    }

    fn errors(&mut self, ctx: &egui::Context) {
        if let Some(error) = self.error.clone() {
            egui::Window::new("Unable to complete this action")
                .collapsible(false)
                .default_width(550.0)
                .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        .max_height(240.0)
                        .show(ui, |ui| {
                            ui.label(error);
                        });
                    ui.add_space(10.0);
                    if self.can_select_cpu {
                        ui.label("You can select CPU and run the action again. CPU processing may take longer.");
                        if ui.button("Select CPU").clicked() {
                            self.provider = Provider::Cpu;
                            self.can_select_cpu = false;
                            self.error = None;
                            self.toast = "CPU selected. Run the action again to retry.".into();
                        }
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Dismiss").clicked() {
                            self.error = None;
                        }
                        if ui.button("Required files").clicked() {
                            self.error = None;
                            self.assets.recheck();
                        }
                    });
                });
        }
    }

    fn job_log(&mut self, ctx: &egui::Context) {
        let mut open = self.log_open;
        egui::Window::new("Native job log")
            .open(&mut open)
            .default_width(760.0)
            .default_height(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in &self.log {
                            ui.monospace(line);
                        }
                        if let Some(active) = &self.active {
                            ui.separator();
                            ui.label(&active.message);
                            ui.monospace(format!(
                                "Worker log: {}",
                                active.worker.stdout_log.display()
                            ));
                            ui.monospace(format!(
                                "Error log: {}",
                                active.worker.stderr_log.display()
                            ));
                        }
                    });
            });
        self.log_open = open;
    }

    fn input(&mut self, ctx: &egui::Context) {
        let dropped = ctx.input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| file.path.clone())
                .collect::<Vec<_>>()
        });
        for path in dropped {
            self.import(path);
        }
        if ctx.wants_keyboard_input() {
            return;
        }
        let (open, save_as, save, export_range, detect, run_scene, run_all) =
            ctx.input_mut(|input| {
                (
                    input.consume_key(egui::Modifiers::COMMAND, egui::Key::O),
                    input.consume_key(
                        egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
                        egui::Key::S,
                    ),
                    input.consume_key(egui::Modifiers::COMMAND, egui::Key::S),
                    input.consume_key(egui::Modifiers::COMMAND, egui::Key::E),
                    input.consume_key(egui::Modifiers::COMMAND, egui::Key::D),
                    input.consume_key(
                        egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
                        egui::Key::R,
                    ),
                    input.consume_key(egui::Modifiers::COMMAND, egui::Key::R),
                )
            });
        if open {
            self.open_footage();
        }
        if save_as {
            self.save_project_as();
        } else if save {
            self.save_project();
        }
        if export_range && self.current().is_some() && self.export.is_none() {
            self.export_dialog = true;
        }
        let idle = self.active.is_none() && self.jobs.is_empty();
        if idle && detect {
            self.enqueue(Action::Detect);
        }
        if idle && run_scene {
            self.enqueue(Action::RenderScene);
        } else if idle && run_all {
            self.enqueue(Action::RenderAll);
        }
        let (undo, redo) = ctx.input_mut(|input| {
            (
                input.consume_key(egui::Modifiers::COMMAND, egui::Key::Z),
                input.consume_key(
                    egui::Modifiers::COMMAND | egui::Modifiers::SHIFT,
                    egui::Key::Z,
                ) || input.consume_key(egui::Modifiers::COMMAND, egui::Key::Y),
            )
        });
        if undo {
            self.undo_at_pointer(ctx);
        }
        if redo {
            self.redo_at_pointer(ctx);
        }
        let (copy_mask, paste_mask) = ctx.input_mut(|input| {
            (
                input.consume_key(egui::Modifiers::COMMAND, egui::Key::C),
                input.consume_key(egui::Modifiers::COMMAND, egui::Key::V),
            )
        });
        if self.active_mask_tab().is_some() {
            if copy_mask {
                self.copy_selected_mask_layer();
            }
            if paste_mask {
                self.paste_mask_layer();
            }
        }
        let (space, left, right, home, end, mark_in, mark_out, shift) = ctx.input_mut(|input| {
            let modifiers = input.modifiers;
            (
                input.consume_key(modifiers, egui::Key::Space),
                input.consume_key(modifiers, egui::Key::ArrowLeft),
                input.consume_key(modifiers, egui::Key::ArrowRight),
                input.consume_key(modifiers, egui::Key::Home),
                input.consume_key(modifiers, egui::Key::End),
                input.consume_key(modifiers, egui::Key::I),
                input.consume_key(modifiers, egui::Key::O),
                input.modifiers.shift,
            )
        });
        if space {
            self.toggle_playback();
        }
        if mark_in || mark_out {
            self.record_app_edit(crate::app_history::AppContext::Timeline, "playback range");
        }
        if let Some(session) = self.current_mut() {
            if left {
                session.seek(session.playhead.saturating_sub(if shift { 10 } else { 1 }));
            }
            if right {
                session.seek(session.playhead.saturating_add(if shift { 10 } else { 1 }));
            }
            if home {
                session.seek(session.range.start);
            }
            if end {
                session.seek(session.range.end_exclusive - 1);
            }
            if mark_in {
                session.range.start = session.playhead.min(session.range.end_exclusive - 1);
            }
            if mark_out {
                session.range.end_exclusive = (session.playhead + 1).max(session.range.start + 1);
            }
        }
        if left || right || home || end {
            self.playing = false;
        }
    }

    fn advance_playback(&mut self, ctx: &egui::Context) {
        if !self.playing {
            return;
        }
        if self.loaded.is_none() {
            self.playback_tick = Instant::now();
            ctx.request_repaint_after(Duration::from_millis(12));
            return;
        }
        let now = Instant::now();
        let elapsed = now.duration_since(self.playback_tick).as_secs_f64();
        self.playback_tick = now;
        let Some(session) = self.current() else {
            self.playing = false;
            return;
        };
        self.playback_fraction += elapsed * fps(session.media());
        let frames = self.playback_fraction.floor() as u64;
        self.playback_fraction -= frames as f64;
        if frames > 0
            && let Some(session) = self.current_mut()
        {
            let next = session.playhead.saturating_add(frames);
            if next >= session.range.end_exclusive {
                session.seek(session.range.start);
            } else {
                session.seek(next);
            }
        }
        ctx.request_repaint_after(Duration::from_millis(12));
    }

    fn write_ui_report(&mut self, ctx: &egui::Context, force: bool) {
        if self.ui_report.is_none()
            || (!force && self.last_ui_report.elapsed() < Duration::from_millis(250))
        {
            return;
        }
        self.last_ui_report = Instant::now();
        if let Some(report_path) = &self.ui_report {
            let (maximized, monitor_size, inner, outer) = ctx.input(|input| {
                let viewport = input.viewport();
                (
                    viewport.maximized,
                    viewport.monitor_size,
                    viewport.inner_rect,
                    viewport.outer_rect,
                )
            });
            let report = serde_json::json!({
                "maximized": maximized,
                "monitor_logical_size": monitor_size.map(|size| [size.x,size.y]),
                "inner_logical_rect": inner.map(|rect| [rect.left(),rect.top(),rect.width(),rect.height()]),
                "outer_logical_rect": outer.map(|rect| [rect.left(),rect.top(),rect.width(),rect.height()]),
                "pixels_per_point": ctx.pixels_per_point(),
                "source_size": self.current().map(|session| [session.media().width,session.media().height]),
                "viewer_zoom_percent": self.viewer_scale * 100.0,
                "viewer_zoom_label": self.zoom.label(self.viewer_fit),
                "viewer_fit_percent": self.viewer_fit * 100.0,
                "inference_disabled": self.ui_only,
                "elapsed_seconds": self.started.elapsed().as_secs_f64(),
                "presented_frames": self.presented_frames,
                "playing": self.playing,
                "playhead": self.current().map(|session| session.playhead),
                "displayed_frame": self.loaded.as_ref().map(|loaded| loaded.frame),
                        "preview_kind": self.loaded.as_ref().map(|loaded| format!("{:?}", loaded.mode)),
                        "preview_artifact": self.loaded.as_ref().and_then(|loaded| loaded.path.as_ref()).map(|path| display_path(path)),
                        "preview_background_applied": self.loaded.as_ref().map(|loaded| loaded.background_applied),
                        "source_snapshot_bytes": self.loaded.as_ref().and_then(|loaded| loaded.source_rgba.as_ref()).map(|rgba| rgba.len()),
                "viewer_canvas_logical_size": [self.viewer_canvas.width(), self.viewer_canvas.height()],
                "viewer_image_physical_size": [self.viewer_image.width() * ctx.pixels_per_point(), self.viewer_image.height() * ctx.pixels_per_point()],
                "timeline_zoom": self.current().map(|session| session.timeline_view.zoom_factor()),
                "timeline_visible_range": self.current().map(|session| { let (start,end) = session.timeline_view.visible_range(); [start,end] }),
            });
            let _ = std::fs::write(
                report_path,
                serde_json::to_vec_pretty(&report).unwrap_or_default(),
            );
        }
    }

    fn screenshots(&mut self, ctx: &egui::Context) {
        let Some(path) = self.screenshot.clone() else {
            return;
        };
        let events = ctx.input(|input| input.events.clone());
        for event in events {
            if let egui::Event::Screenshot { image, .. } = event {
                self.write_ui_report(ctx, true);
                let bytes = image
                    .pixels
                    .iter()
                    .flat_map(|pixel| pixel.to_array())
                    .collect::<Vec<_>>();
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if let Err(error) = image::save_buffer(
                    &path,
                    &bytes,
                    image.width() as u32,
                    image.height() as u32,
                    image::ColorType::Rgba8,
                ) {
                    self.fail(error.to_string());
                }
                self.screenshot = None;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                return;
            }
        }
        if !self.screenshot_started
            && self.started.elapsed() > Duration::from_secs(2)
            && !self.assets.open
            && (self.sessions.is_empty() || self.loaded.is_some())
        {
            self.screenshot_started = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }
        ctx.request_repaint_after(Duration::from_millis(50));
    }
}

impl eframe::App for Studio {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(session) = self.current_mut() {
            session.sync_sam_layer();
        }
        self.pump_auto_result_cache();
        if self.startup_maximize {
            // Send this once the native window exists, after creation geometry
            // and DPI negotiation, so the client bounds follow the OS work area.
            if let Some(monitor) = ctx.input(|input| input.viewport().monitor_size) {
                ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(Vec2::new(
                    (monitor.x * 0.8).min(800.0),
                    (monitor.y * 0.7).min(480.0),
                )));
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
            self.startup_maximize = false;
        }
        self.icons.sync_density(ctx);
        self.assets.poll();
        if self.assets.take_changed() {
            let automatic = RuntimePaths::installed(&self.app_directory);
            let rebuild_preview = self.runtime.ffmpeg != automatic.ffmpeg
                || self.runtime.ffprobe != automatic.ffprobe;
            self.runtime = automatic;
            if rebuild_preview {
                self.preview = PreviewService::new(
                    self.runtime.clone(),
                    self.work_root.join("source-preview"),
                    ctx.clone(),
                );
            }
            self.requested_key = None;
            if self.assets.ready_for(&[AssetGroup::Core]) {
                for path in std::mem::take(&mut self.pending_imports) {
                    self.import(path);
                }
            }
        }
        self.poll_io();
        self.poll_export();
        self.accept_messages(ctx);
        self.poll_jobs();
        self.pump_auto_segment();
        self.input(ctx);
        self.advance_playback(ctx);
        self.request_preview();
        self.toolbar(ctx);
        self.timeline(ctx);
        self.sidebar(ctx);
        self.tools(ctx);
        self.viewer(ctx);
        self.export_window(ctx);
        self.settings(ctx);
        let idle = self.active.is_none() && self.jobs.is_empty();
        self.updater.window(ctx, idle);
        self.assets.show(ctx);
        self.errors(ctx);
        self.job_log(ctx);
        self.window_edge_resize(ctx);
        self.screenshots(ctx);
        if self.calculated_dirty
            && !self.project_io_busy
            && self.active.is_none()
            && self.jobs.is_empty()
            && self.export.is_none()
            && let Some(path) = self.project_path.clone()
        {
            self.save_project_to(path);
        }
        if self.ui_report.is_some() {
            self.write_ui_report(ctx, false);
            ctx.request_repaint_after(Duration::from_millis(250));
        }
        if self.project_io_busy
            || self.export.is_some()
            || self.active.is_some()
            || self.loaded.is_none() && !self.sessions.is_empty()
        {
            ctx.request_repaint_after(Duration::from_millis(30));
        }
    }
}

impl Drop for Studio {
    fn drop(&mut self) {
        self.export.take();
        if let Some(mut active) = self.active.take() {
            let _ = active.worker.cancel();
        }
        self.preview.shutdown();
    }
}

fn refresh_embedded_identity(
    media: &mut MediaInfo,
    archive: Option<&project_store::EmbeddedArchive>,
) -> anyhow::Result<()> {
    if let Some(archive) = archive
        && archive.is_embedded(&media.path)
        && media.path.is_file()
    {
        media.source_identity = archive.materialized_identity(&media.path)?;
    }
    Ok(())
}

fn reveal_path(path: &std::path::Path) {
    let mut command;
    #[cfg(windows)]
    {
        command = std::process::Command::new("explorer.exe");
        command.arg(format!("/select,{}", display_path(path)));
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    #[cfg(target_os = "macos")]
    {
        command = std::process::Command::new("open");
        command.arg("-R").arg(path);
    }
    #[cfg(all(not(windows), not(target_os = "macos")))]
    {
        command = std::process::Command::new("xdg-open");
        command.arg(path.parent().unwrap_or(path));
    }
    let _ = command.spawn();
}

fn ck_settings_controls(
    ui: &mut egui::Ui,
    settings: &mut CkSettings,
    hann_advice_input: Option<(usize, usize, usize, f64)>,
    checkbox_on: egui::TextureId,
    checkbox_off: egui::TextureId,
) {
    theme::field_row(ui, "Window", |ui| {
        theme::segmented(
            ui,
            &mut settings.window_mode,
            &[
                (
                    CkWindowMode::Balanced,
                    "Auto",
                    "Keep Hann windows small for speed, adjusted for resolution and available GPU memory.",
                ),
                (
                    CkWindowMode::Wide,
                    "Wide",
                    "Choose longer Hann windows from available GPU memory. Long windows can be slower.",
                ),
                (
                    CkWindowMode::Custom,
                    "Custom",
                    "Use the Hann chunk and stride below. Auto chunk uses the wide-window policy; Auto stride uses half the chosen chunk.",
                ),
            ],
        );
    });
    if settings.window_mode == CkWindowMode::Custom {
        hann_frame_control(ui, "Hann chunk", &mut settings.hann_chunk);
        hann_frame_control(ui, "Hann stride", &mut settings.hann_stride);
    }
    theme::field_row(ui, "Low VRAM", |ui| {
        checkbox_widget(ui, &mut settings.low_vram, "", checkbox_on, checkbox_off).on_hover_text(
            "Use slower staged execution with no retained GPU buffers to minimize memory use.",
        );
    });
    if let Err(error) = settings.validate() {
        ui.label(RichText::new(error.to_string()).size(11.5).color(ORANGE));
    }
    if let Some((width, height, frames, budget_gib)) = hann_advice_input
        && let Some(advice) = custom_hann_advice(settings, width, height, frames, Some(budget_gib))
    {
        let color = match advice.kind {
            HannAdviceKind::OverBudget | HannAdviceKind::LargerThanAuto => ORANGE,
            HannAdviceKind::StrideDiffers | HannAdviceKind::SmallerThanAuto => MUTED,
        };
        ui.label(RichText::new(advice.message).size(11.5).color(color));
    }
    theme::hint(ui, "Saved with this clip. Queued jobs keep their settings.");
}

fn hann_frame_control(ui: &mut egui::Ui, label: &str, value: &mut usize) {
    theme::field_row(ui, label, |ui| {
        let response = ui.add(
            egui::DragValue::new(value)
                .range(0..=256)
                .speed(1.0)
                .custom_formatter(|value, _| {
                    if value == 0.0 {
                        "Auto".into()
                    } else {
                        format!("{value:.0}")
                    }
                })
                .custom_parser(|text| {
                    if text.trim().eq_ignore_ascii_case("auto") {
                        Some(0.0)
                    } else {
                        text.parse().ok()
                    }
                }),
        );
        response.on_hover_text(
            "Auto picks the window that fits this GPU. Larger windows use more VRAM and measured whole-job time can be 2-4x slower.",
        );
    });
}

/// Draw the Result despill slider inside a fixed-size row.
///
/// The slider and its value suffix are laid out inside a pre-allocated
/// rectangle whose width does not depend on the displayed value. Without this
/// the wider "100%" suffix grows the row's minimum width, the resizable
/// inspector follows it, and the wider panel then widens the row again on the
/// next frame (a per-frame layout feedback loop).
fn despill_slider_row(ui: &mut egui::Ui, amount: Option<&mut f32>, hover: &str) -> bool {
    value_slider_row(ui, "despill_slider_row", amount, 0.0..=100.0, "%", hover).changed
}

/// Largest mask feather offered in the UI, in source pixels (v1 slider range).
const MASK_SOFTNESS_MAX: f32 = 120.0;

#[derive(Default)]
struct SliderRowResponse {
    changed: bool,
    dragging: bool,
}

/// Slider plus a fixed-width value box pinned to the right edge, in a rectangle whose width
/// does not depend on the displayed value (see `despill_slider_row` for why).
fn value_slider_row(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash,
    value: Option<&mut f32>,
    range: std::ops::RangeInclusive<f32>,
    suffix: &str,
    hover: &str,
) -> SliderRowResponse {
    let width = ui.available_width().max(180.0);
    let height = ui.spacing().interact_size.y.max(20.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, height), egui::Sense::hover());
    let mut child = ui.new_child(
        egui::UiBuilder::new()
            .id_salt(id_salt)
            .max_rect(rect)
            .layout(egui::Layout::left_to_right(egui::Align::Center)),
    );
    let value_width = 64.0;
    let gap = child.spacing().item_spacing.x;
    child.spacing_mut().slider_width = (width - value_width - gap).max(96.0);
    let mut current = value
        .as_ref()
        .map(|value| **value)
        .unwrap_or(*range.start());
    let enabled = value.is_some();
    let slider = child.add_enabled(
        enabled,
        egui::Slider::new(&mut current, range.clone())
            .step_by(1.0)
            .show_value(false),
    );
    let typed = child
        .add_enabled_ui(enabled, |ui| {
            ui.add_sized(
                [value_width, height],
                egui::DragValue::new(&mut current)
                    .range(range)
                    .speed(1.0)
                    .suffix(suffix)
                    .fixed_decimals(0),
            )
        })
        .inner
        .on_hover_text(hover);
    if enabled && let Some(target) = value {
        *target = current;
    }
    let response = SliderRowResponse {
        changed: slider.changed() || typed.changed(),
        dragging: slider.dragged() || typed.dragged(),
    };
    slider.on_hover_text(hover);
    response
}

/// Active per-scene despill override (0..=1) for the scene at position `index`.
fn scene_despill_override(project: &ck_desktop_core::Project, index: usize) -> Option<f32> {
    project
        .scene_overrides
        .get(&(index as u32))
        .and_then(|entry| entry.despill)
        .filter(|value| value.enabled)
        .map(|value| value.amount)
}

/// Sets (`Some`, 0..=1) or clears (`None`) a scene's despill override. A cleared override keeps
/// its last amount so turning it back on restores it. Returns whether anything changed.
fn set_scene_despill(
    project: &mut ck_desktop_core::Project,
    index: usize,
    amount: Option<f32>,
) -> bool {
    let key = index as u32;
    let mut entry = project
        .scene_overrides
        .get(&key)
        .cloned()
        .unwrap_or_default();
    let previous = entry.despill;
    let last = previous.map_or(project.despill_settings.amount, |value| value.amount);
    entry.despill = Some(ck_desktop_core::SceneOverrideValue {
        enabled: amount.is_some(),
        amount: amount.unwrap_or(last).clamp(0.0, 1.0),
    });
    if entry.despill == previous {
        return false;
    }
    project.scene_overrides.insert(key, entry);
    true
}

/// Parse a `#rrggbb` layer colour, falling back to the muted UI colour.
fn mask_color(value: &str) -> Color32 {
    let hex = value.trim().trim_start_matches('#');
    if hex.len() == 6 {
        let parse = |range: std::ops::Range<usize>| u8::from_str_radix(&hex[range], 16).ok();
        if let (Some(r), Some(g), Some(b)) = (parse(0..2), parse(2..4), parse(4..6)) {
            return Color32::from_rgb(r, g, b);
        }
    }
    MUTED
}

/// One clip in the media list: name and a muted metadata line; the open clip is highlighted.
fn media_list_row(ui: &mut egui::Ui, name: &str, meta: &str, selected: bool) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 46.0), egui::Sense::click());
    let painter = ui.painter_at(rect);
    let fill = if selected {
        theme::SEGMENT_ON
    } else if response.hovered() {
        theme::BG_PANEL_ALT
    } else {
        theme::CARD_FILL
    };
    painter.rect(
        rect,
        6.0,
        fill,
        Stroke::new(
            1.0_f32,
            if selected {
                theme::PRIMARY_HOVER
            } else {
                theme::CARD_BORDER
            },
        ),
        egui::StrokeKind::Inside,
    );
    let inner = rect.shrink2(Vec2::new(10.0, 6.0));
    let clip = painter.with_clip_rect(inner);
    clip.text(
        inner.left_top(),
        egui::Align2::LEFT_TOP,
        name,
        egui::FontId::proportional(13.0),
        theme::TEXT,
    );
    clip.text(
        inner.left_bottom(),
        egui::Align2::LEFT_BOTTOM,
        meta,
        egui::FontId::proportional(11.0),
        MUTED,
    );
    response
}

/// One selectable scene row: name, range badge, on/off state; the current scene is
/// highlighted.
fn scene_list_row(
    ui: &mut egui::Ui,
    index: usize,
    current: bool,
    enabled: bool,
    range: &str,
) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), egui::Sense::click());
    let painter = ui.painter_at(rect);
    let fill = if current {
        theme::SEGMENT_ON
    } else if response.hovered() {
        theme::BG_PANEL_ALT
    } else {
        Color32::TRANSPARENT
    };
    painter.rect_filled(rect, 5.0, fill);
    let text = if enabled { theme::TEXT } else { MUTED };
    painter.text(
        rect.left_center() + Vec2::new(8.0, 0.0),
        egui::Align2::LEFT_CENTER,
        format!("Scene {:02}", index + 1),
        egui::FontId::proportional(13.0),
        text,
    );
    painter.text(
        rect.right_center() - Vec2::new(8.0, 0.0),
        egui::Align2::RIGHT_CENTER,
        if enabled {
            range.to_owned()
        } else {
            format!("{range} · off")
        },
        egui::FontId::proportional(12.0),
        MUTED,
    );
    response
}

/// Height of the custom title bar row (window buttons fill it).
const TITLE_BAR_HEIGHT: f32 = 40.0;
/// Width of the invisible resize band at the window edges, in points.
const RESIZE_BAND: f32 = 5.0;

/// Edge or corner under `pointer` for frameless resizing (corners get a wider band).
fn edge_direction(window: Rect, pointer: Pos2) -> Option<egui::viewport::ResizeDirection> {
    use egui::viewport::ResizeDirection as Dir;
    if !window.expand(1.0).contains(pointer) {
        return None;
    }
    let corner = RESIZE_BAND * 3.0;
    let left = pointer.x - window.left() <= RESIZE_BAND;
    let right = window.right() - pointer.x <= RESIZE_BAND;
    let top = pointer.y - window.top() <= RESIZE_BAND;
    let bottom = window.bottom() - pointer.y <= RESIZE_BAND;
    let near_left = pointer.x - window.left() <= corner;
    let near_right = window.right() - pointer.x <= corner;
    let near_top = pointer.y - window.top() <= corner;
    let near_bottom = window.bottom() - pointer.y <= corner;
    if (top && near_left) || (left && near_top) {
        Some(Dir::NorthWest)
    } else if (top && near_right) || (right && near_top) {
        Some(Dir::NorthEast)
    } else if (bottom && near_left) || (left && near_bottom) {
        Some(Dir::SouthWest)
    } else if (bottom && near_right) || (right && near_bottom) {
        Some(Dir::SouthEast)
    } else if top {
        Some(Dir::North)
    } else if bottom {
        Some(Dir::South)
    } else if left {
        Some(Dir::West)
    } else if right {
        Some(Dir::East)
    } else {
        None
    }
}

#[derive(Clone, Copy)]
enum WindowGlyph {
    Minimize,
    Maximize,
    Restore,
    Close,
}

/// One flat window button the full title-bar height; close turns red on hover.
fn window_button(ui: &mut egui::Ui, glyph: WindowGlyph, hover: &str) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(46.0, TITLE_BAR_HEIGHT), egui::Sense::click());
    let close = matches!(glyph, WindowGlyph::Close);
    if response.hovered() {
        let fill = if close {
            Color32::from_rgb(196, 43, 28)
        } else {
            Color32::from_gray(52)
        };
        ui.painter().rect_filled(rect, 0.0, fill);
    }
    let color = if close && response.hovered() {
        Color32::WHITE
    } else {
        theme::TEXT
    };
    let stroke = Stroke::new(1.0_f32, color);
    let c = rect.center();
    let painter = ui.painter();
    match glyph {
        WindowGlyph::Minimize => {
            painter.hline(c.x - 5.0..=c.x + 5.0, c.y, stroke);
        }
        WindowGlyph::Maximize => {
            painter.rect_stroke(
                Rect::from_center_size(c, Vec2::splat(10.0)),
                0.0,
                stroke,
                egui::StrokeKind::Inside,
            );
        }
        WindowGlyph::Restore => {
            painter.rect_stroke(
                Rect::from_center_size(c + Vec2::new(-1.5, 1.5), Vec2::splat(8.0)),
                0.0,
                stroke,
                egui::StrokeKind::Inside,
            );
            painter.line_segment(
                [c + Vec2::new(-2.5, -4.5), c + Vec2::new(5.5, -4.5)],
                stroke,
            );
            painter.line_segment([c + Vec2::new(5.5, -4.5), c + Vec2::new(5.5, 3.5)], stroke);
        }
        WindowGlyph::Close => {
            painter.line_segment([c + Vec2::new(-5.0, -5.0), c + Vec2::new(5.0, 5.0)], stroke);
            painter.line_segment([c + Vec2::new(-5.0, 5.0), c + Vec2::new(5.0, -5.0)], stroke);
        }
    }
    response.on_hover_text(hover)
}

/// Minimise / maximise-restore / close, laid out right to left.
fn window_buttons(ui: &mut egui::Ui, maximized: bool) {
    let ctx = ui.ctx().clone();
    ui.spacing_mut().item_spacing.x = 0.0;
    if window_button(ui, WindowGlyph::Close, "Close").clicked() {
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }
    let (glyph, hover) = if maximized {
        (WindowGlyph::Restore, "Restore down")
    } else {
        (WindowGlyph::Maximize, "Maximize")
    };
    if window_button(ui, glyph, hover).clicked() {
        ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized));
    }
    if window_button(ui, WindowGlyph::Minimize, "Minimize").clicked() {
        ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
    }
    ui.spacing_mut().item_spacing.x = 8.0;
}

struct StatusProgress<'a> {
    label: &'a str,
    /// Steady stage text such as "Stage 2 frames 5/16" (diagnostic events never change it).
    stage: String,
    /// Overall job fraction; never decreases within an attempt.
    fraction: f32,
    message: &'a str,
}

fn status_strip(
    ui: &mut egui::Ui,
    toast: &str,
    progress: Option<StatusProgress<'_>>,
) -> egui::Response {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
        // Allocate controls before variable text, anchored to the right edge.
        let cancel = ui.add_enabled(
            progress.is_some(),
            egui::Button::new("Cancel").min_size(Vec2::new(64.0, 24.0)),
        );
        if let Some(progress) = progress {
            ui.add(
                egui::ProgressBar::new(progress.fraction.clamp(0.0, 1.0))
                    .fill(ACCENT)
                    .desired_width(96.0)
                    .show_percentage(),
            );
            let text = format!("{} · {}", progress.label, progress.stage);
            ui.add(egui::Label::new(RichText::new(&text).small().color(MUTED)).truncate())
                .on_hover_text(format!("{text}\n{}", progress.message));
        } else {
            ui.add(egui::Label::new(RichText::new(toast).small().color(MUTED)).truncate());
        }
        cancel
    })
    .inner
}

/// Frame number may change during playback while the clip/view/revision must not.
fn same_preview_context(a: &str, b: &str) -> bool {
    let mut a = a.splitn(3, ':');
    let mut b = b.splitn(3, ':');
    let same_clip = a.next() == b.next();
    let frames_present = a.next().is_some() && b.next().is_some();
    let (a, b) = (a.next(), b.next());
    same_clip && frames_present && a.is_some() && a == b
}

fn playback_frame_order(frame: u64, playhead: u64, displayed: Option<u64>) -> bool {
    frame <= playhead && displayed.is_none_or(|previous| frame >= previous || previous > playhead)
}
fn source_coordinate(image: Rect, position: Pos2, source: Vec2) -> Option<(f32, f32)> {
    if !image.contains(position)
        || image.width() <= 0.0
        || image.height() <= 0.0
        || source.x < 1.0
        || source.y < 1.0
    {
        return None;
    }
    Some((
        ((position.x - image.left()) / image.width() * source.x).clamp(0.0, source.x - 1.0),
        ((position.y - image.top()) / image.height() * source.y).clamp(0.0, source.y - 1.0),
    ))
}
fn fps(media: &MediaInfo) -> f64 {
    media.fps.num as f64 / media.fps.den.max(1) as f64
}
fn timecode(frame: u64, fps: f64) -> String {
    let total = frame as f64 / fps;
    format!(
        "{:02}:{:02}:{:06.3}",
        (total / 3600.0) as u64,
        (total / 60.0) as u64 % 60,
        total % 60.0
    )
}
fn path_field(ui: &mut egui::Ui, label: &str, path: &mut PathBuf) {
    ui.label(label);
    let mut value = path.to_string_lossy().to_string();
    if ui
        .add(egui::TextEdit::singleline(&mut value).desired_width(530.0))
        .changed()
    {
        *path = PathBuf::from(value);
    }
    ui.end_row();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_studio_timeline_height_is_stable_during_pointer_motion() {
        for (width, height, density) in [
            (1060.0, 720.0, 1.0),
            (1520.0, 980.0, 1.25),
            (1920.0, 1080.0, 2.0),
        ] {
            let ctx = egui::Context::default();
            ctx.set_pixels_per_point(density);
            let creation = eframe::CreationContext::_new_kittest(ctx.clone());
            let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
            studio.sessions.push(session());
            let mut first_height = None;
            for pass in 0..160 {
                let mut panel = Rect::NOTHING;
                let _ = ctx.run(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(
                            Pos2::ZERO,
                            Vec2::new(width, height),
                        )),
                        time: Some(1.0 + pass as f64 / 60.0),
                        events: vec![egui::Event::PointerMoved(Pos2::new(
                            20.0 + (pass * 37 % (width as usize - 40)) as f32,
                            120.0 + (pass * 23 % (height as usize - 180)) as f32,
                        ))],
                        ..Default::default()
                    },
                    |ctx| {
                        studio.toolbar(ctx);
                        studio.timeline(ctx);
                        panel = egui::containers::panel::PanelState::load(
                            ctx,
                            egui::Id::new(TIMELINE_PANEL_ID),
                        )
                        .expect("timeline panel was rendered")
                        .rect;
                        egui::CentralPanel::default().show(ctx, |_ui| {});
                    },
                );
                let initial = *first_height.get_or_insert(panel.height());
                assert!(
                    (panel.height() - initial).abs() < 0.1,
                    "{width}x{height} at {density}x: pointer repaint {pass} grew timeline from {initial} to {}",
                    panel.height()
                );
            }
        }
    }

    #[test]
    fn actual_studio_timeline_divider_can_resize_then_stays_fixed() {
        let ctx = egui::Context::default();
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        studio.sessions.push(session());
        let mut time = 1.0;
        let mut frame = |events| {
            time += 0.02;
            let mut panel = Rect::NOTHING;
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1060.0, 720.0))),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    studio.toolbar(ctx);
                    studio.timeline(ctx);
                    panel = egui::containers::panel::PanelState::load(
                        ctx,
                        egui::Id::new(TIMELINE_PANEL_ID),
                    )
                    .expect("timeline panel was rendered")
                    .rect;
                    egui::CentralPanel::default().show(ctx, |_ui| {});
                },
            );
            panel
        };
        let initial = frame(vec![]);
        assert!((initial.height() - 224.0).abs() < 0.1);
        let press = initial.center_top();
        frame(vec![egui::Event::PointerMoved(press)]);
        frame(vec![egui::Event::PointerButton {
            pos: press,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        }]);
        let dragged = press - Vec2::new(0.0, 100.0);
        for _ in 0..3 {
            frame(vec![egui::Event::PointerMoved(dragged)]);
        }
        let resized = frame(vec![egui::Event::PointerButton {
            pos: dragged,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]);
        assert!((resized.height() - initial.height() - 100.0).abs() < 0.1);
        for pass in 0..80 {
            let panel = frame(vec![egui::Event::PointerMoved(Pos2::new(
                20.0 + pass as f32 * 7.0,
                400.0,
            ))]);
            assert!((panel.height() - resized.height()).abs() < 0.1);
        }
        assert_eq!(
            studio.current().unwrap().range,
            FrameRange {
                start: 0,
                end_exclusive: 100
            }
        );
    }

    #[test]
    fn playback_accepts_late_frames_only_for_the_same_clip_view_and_revision() {
        assert!(same_preview_context(
            "5:102:Source:4:true:identity:Checker",
            "5:100:Source:4:true:identity:Checker"
        ));
        for stale in [
            "6:100:Source:4:true:identity:Checker",
            "5:100:Alpha:4:true:identity:Checker",
            "5:100:Source:3:true:identity:Checker",
            "5:100:Source:4:false:identity:Checker",
            "invalid",
        ] {
            assert!(!same_preview_context(
                "5:102:Source:4:true:identity:Checker",
                stale
            ));
        }
    }

    #[test]
    fn playback_loop_can_present_its_first_frames_without_accepting_old_cycle_frames() {
        assert!(playback_frame_order(0, 1, Some(49)));
        assert!(!playback_frame_order(49, 1, Some(49)));
        assert!(playback_frame_order(3, 4, Some(2)));
        assert!(!playback_frame_order(2, 4, Some(3)));
    }

    #[test]
    fn cancel_control_does_not_move_with_phase_text_or_progress_digits() {
        let ctx = egui::Context::default();
        let mut positions = Vec::new();
        for (stage, fraction) in [
            ("Starting", 0.0),
            ("Stage 2 frames 9/50", 0.4),
            ("A much longer progress phase 10000/100000", 0.1),
        ] {
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 100.0))),
                    ..Default::default()
                },
                |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        positions.push(
                            status_strip(
                                ui,
                                "",
                                Some(StatusProgress {
                                    label: "CorridorKey",
                                    stage: stage.into(),
                                    fraction,
                                    message: "Variable worker output",
                                }),
                            )
                            .rect,
                        );
                    });
                },
            );
        }
        assert!(positions.windows(2).all(|pair| pair[0] == pair[1]));
    }
    use ck_desktop_core::{Rational, SourceIdentity};

    #[test]
    fn frameless_resize_bands_cover_edges_and_wider_corners() {
        use egui::viewport::ResizeDirection as Dir;
        let window = Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 700.0));
        let at = |x: f32, y: f32| edge_direction(window, Pos2::new(x, y));
        assert_eq!(at(500.0, 350.0), None);
        assert_eq!(at(2.0, 350.0), Some(Dir::West));
        assert_eq!(at(998.0, 350.0), Some(Dir::East));
        assert_eq!(at(500.0, 1.0), Some(Dir::North));
        assert_eq!(at(500.0, 699.0), Some(Dir::South));
        // Corners accept a longer stretch along each edge.
        assert_eq!(at(10.0, 2.0), Some(Dir::NorthWest));
        assert_eq!(at(998.0, 12.0), Some(Dir::NorthEast));
        assert_eq!(at(2.0, 690.0), Some(Dir::SouthWest));
        assert_eq!(at(990.0, 698.0), Some(Dir::SouthEast));
        assert_eq!(at(1200.0, 350.0), None);
    }

    fn override_project() -> Project {
        Project {
            schema_version: 1,
            media: Some(media()),
            scenes: vec![],
            prompts: vec![],
            sam_layers: vec![ck_desktop_core::SamLayer::subject()],
            hint_quality: Default::default(),
            sam_boxes: Vec::new(),
            output_dir: PathBuf::from("out"),
            ck_settings: CkSettings::default(),
            despill_settings: ck_desktop_core::DespillSettings { amount: 0.25 },
            mask_layers: Default::default(),
            scene_overrides: Default::default(),
        }
    }

    #[test]
    fn scene_despill_override_is_stored_as_a_fraction_and_keeps_its_amount_when_off() {
        let mut project = override_project();
        assert_eq!(scene_despill_override(&project, 2), None);
        // The menu and inspector pass percent / 100; 60 % must stay 0.6, not clamp to 1.
        assert!(set_scene_despill(&mut project, 2, Some(0.6)));
        assert_eq!(scene_despill_override(&project, 2), Some(0.6));
        assert!(
            !set_scene_despill(&mut project, 2, Some(0.6)),
            "unchanged is not an edit"
        );
        assert!(set_scene_despill(&mut project, 2, None));
        assert_eq!(scene_despill_override(&project, 2), None);
        let kept = project.scene_overrides[&2].despill.unwrap();
        assert!(!kept.enabled);
        assert_eq!(kept.amount, 0.6);
        assert_eq!(scene_despill_override(&project, 1), None);
    }

    fn media() -> MediaInfo {
        MediaInfo {
            path: PathBuf::from("source.mp4"),
            width: 1920,
            height: 1080,
            frame_count: 100,
            fps: Rational {
                num: 30000,
                den: 1001,
            },
            time_base: Rational { num: 1, den: 30000 },
            first_pts: 0,
            has_b_frames: true,
            source_identity: Some(SourceIdentity {
                bytes: 1000,
                modified_unix_ns: 1,
            }),
        }
    }

    /// A one-scene session whose first layer was created (as a first point would).
    fn session() -> Session {
        let mut session = Session::new(Project::new(media(), PathBuf::from("renders")));
        session.layer_for_edit();
        session
    }

    #[test]
    fn current_scene_job_follows_needle_even_with_stale_selection_and_in_out() {
        let mut session = session();
        session.project.scenes = vec![
            SceneRange {
                index: 0,
                start: 0,
                end_exclusive: 50,
                enabled: true,
            },
            SceneRange {
                index: 1,
                start: 50,
                end_exclusive: 100,
                enabled: true,
            },
        ];
        session.selected_scene = 0;
        session.range = FrameRange {
            start: 3,
            end_exclusive: 9,
        };
        session.playhead = 72;
        assert_eq!(
            session.current_scene_range().unwrap(),
            FrameRange {
                start: 50,
                end_exclusive: 100
            }
        );
        session.playhead = 49;
        assert_eq!(
            session.current_scene_range().unwrap(),
            FrameRange {
                start: 0,
                end_exclusive: 50
            }
        );
        session.seek(50);
        assert_eq!(session.selected_scene, 1);
        assert_eq!(
            session.range,
            FrameRange {
                start: 3,
                end_exclusive: 9
            }
        );
        assert_eq!(
            session.current_scene_range().unwrap(),
            FrameRange {
                start: 50,
                end_exclusive: 100
            }
        );
        session.project.scenes[1].enabled = false;
        assert_eq!(
            session.current_scene_range(),
            Err(timeline::SceneRunError::Disabled(1))
        );
    }

    #[test]
    fn queued_scene_requests_keep_their_policy_after_clip_controls_change() {
        let mut session = session();
        let requested = CkSettings {
            window_mode: CkWindowMode::Custom,
            hann_chunk: 12,
            hann_stride: 6,
            low_vram: true,
        };
        session.project.ck_settings = requested.clone();
        let queued = [(0, 50), (50, 100)].map(|(start, end_exclusive)| {
            session.ck_job(
                FrameRange {
                    start,
                    end_exclusive,
                },
                Provider::DirectMl,
                HintInput::None,
            )
        });
        session.project.ck_settings = CkSettings::default();
        session.refresh_source(media());
        for job in queued {
            let restored: JobKind =
                serde_json::from_str(&serde_json::to_string(&job).unwrap()).unwrap();
            let JobKind::CorridorKey { settings, .. } = restored else {
                panic!("expected CorridorKey job")
            };
            assert_eq!(settings, requested);
        }
        assert_eq!(session.project.ck_settings, CkSettings::default());
    }

    #[test]
    fn memory_retry_discards_attempt_outputs_but_preserves_masks_and_other_results() {
        let mut session = session();
        session.frames.insert(
            4,
            FrameFiles {
                rgba: Some(PathBuf::from("failed/frames/000004.png")),
                alpha: Some(PathBuf::from("failed/alpha/000004.png")),
                checker: Some(PathBuf::from("failed/checker/000004.png")),
                mask: Some(PathBuf::from("sam/000004.png")),
                sam_layers: Default::default(),
            },
        );
        session.frames.insert(
            5,
            FrameFiles {
                rgba: Some(PathBuf::from("failed/frames/000005.png")),
                ..FrameFiles::default()
            },
        );
        session.frames.insert(
            8,
            FrameFiles {
                rgba: Some(PathBuf::from("prior/frames/000008.png")),
                ..FrameFiles::default()
            },
        );
        let mut attempt_frames = BTreeSet::from([4, 5]);
        session.discard_render_attempt(&std::mem::take(&mut attempt_frames));
        let retained = &session.frames[&4];
        assert!(retained.rgba.is_none() && retained.alpha.is_none() && retained.checker.is_none());
        assert_eq!(
            retained.mask.as_deref(),
            Some(std::path::Path::new("sam/000004.png"))
        );
        assert!(!session.frames.contains_key(&5));
        assert_eq!(
            session.frames[&8].rgba.as_deref(),
            Some(std::path::Path::new("prior/frames/000008.png"))
        );
        assert_eq!(session.revision, 1);
        assert!(attempt_frames.is_empty());
    }

    #[test]
    fn prompt_coordinates_follow_displayed_image_with_letterbox_pan_and_zoom() {
        let source = Vec2::new(1920.0, 1080.0);
        let letterboxed = Rect::from_min_size(Pos2::new(100.0, 50.0), source * 0.5);
        assert_eq!(
            source_coordinate(letterboxed, Pos2::new(580.0, 320.0), source),
            Some((960.0, 540.0))
        );
        assert_eq!(
            source_coordinate(letterboxed, Pos2::new(99.0, 320.0), source),
            None
        );
        let zoomed_panned = Rect::from_min_size(Pos2::new(-600.0, -100.0), source * 1.5);
        assert_eq!(
            source_coordinate(zoomed_panned, Pos2::new(840.0, 710.0), source),
            Some((960.0, 540.0))
        );
        assert_eq!(
            source_coordinate(letterboxed, letterboxed.right_bottom(), source),
            Some((1919.0, 1079.0))
        );
    }

    #[test]
    fn explicit_reimport_replaces_legacy_identity_and_evicts_stale_results() {
        let mut session = session();
        session.project.media.as_mut().unwrap().source_identity = None;
        session.playhead = 99;
        session.frames.insert(99, FrameFiles::default());
        session.masks.push(MaskBatch {
            range: session.range,
            directory: PathBuf::from("old-masks"),
        });
        session.project.prompts = vec![
            SamPrompt {
                frame: 4,
                x: 300.0,
                y: 100.0,
                positive: true,
                layer: 0,
                text: None,
            },
            SamPrompt {
                frame: 55,
                x: 300.0,
                y: 100.0,
                positive: true,
                layer: 0,
                text: None,
            },
            SamPrompt {
                frame: 4,
                x: 1800.0,
                y: 100.0,
                positive: false,
                layer: 0,
                text: None,
            },
            SamPrompt {
                frame: 4,
                x: f32::NAN,
                y: 100.0,
                positive: false,
                layer: 0,
                text: None,
            },
        ];
        let mut replacement = media();
        replacement.frame_count = 50;
        replacement.width = 1280;
        session.refresh_source(replacement.clone());
        assert_eq!(session.media().source_identity, replacement.source_identity);
        assert_eq!(session.playhead, 49);
        assert_eq!(
            session.range,
            FrameRange {
                start: 0,
                end_exclusive: 50
            }
        );
        assert_eq!(session.project.scenes[0].end_exclusive, 50);
        assert_eq!(session.project.prompts.len(), 1);
        assert!(session.frames.is_empty() && session.masks.is_empty());
        assert_eq!(session.revision, 1);
    }

    #[test]
    fn prompt_correction_invalidates_affected_scene_with_exact_source_boundaries() {
        let mut session = session();
        session.project.scenes = vec![
            SceneRange {
                index: 0,
                start: 0,
                end_exclusive: 50,
                enabled: true,
            },
            SceneRange {
                index: 1,
                start: 50,
                end_exclusive: 100,
                enabled: true,
            },
        ];
        for frame in [0, 49, 50, 99] {
            session.frames.insert(frame, FrameFiles::default());
        }
        session.masks = vec![
            MaskBatch {
                range: FrameRange {
                    start: 0,
                    end_exclusive: 50,
                },
                directory: PathBuf::from("first"),
            },
            MaskBatch {
                range: FrameRange {
                    start: 50,
                    end_exclusive: 100,
                },
                directory: PathBuf::from("second"),
            },
        ];
        session.select_scene(1);
        assert_eq!(session.playhead, 50);
        assert_eq!(
            session.range,
            FrameRange {
                start: 50,
                end_exclusive: 100
            }
        );
        session.frames.insert(
            60,
            FrameFiles {
                rgba: Some(PathBuf::from("frames/000060.png")),
                mask: Some(PathBuf::from("sam/000060.png")),
                ..Default::default()
            },
        );
        session.invalidate_prompt_scene(50, 0);
        assert_eq!(
            session.frames.keys().copied().collect::<Vec<_>>(),
            vec![0, 49, 60]
        );
        assert!(
            session.frames[&60].rgba.is_some(),
            "keyed results survive point edits"
        );
        assert!(
            session.frames[&60].mask.is_none(),
            "the scene's SAM masks are dropped"
        );
        assert_eq!(session.masks.len(), 1);
    }

    #[test]
    fn point_edits_drop_only_their_layer_and_layer_changes_drop_combined_masks() {
        let mut session = session();
        let second = session.project.add_sam_layer(0);
        assert_eq!(second, 1);
        session.frames.insert(
            3,
            FrameFiles {
                mask: Some(PathBuf::from("combined.png")),
                sam_layers: BTreeMap::from([
                    (0, PathBuf::from("subject.png")),
                    (second, PathBuf::from("hat.png")),
                ]),
                ..Default::default()
            },
        );
        session.invalidate_prompt_scene(3, second);
        let files = &session.frames[&3];
        assert_eq!(
            files.sam_layers.keys().copied().collect::<Vec<_>>(),
            vec![0]
        );
        assert!(
            files.mask.is_none(),
            "the combined mask is rebuilt from the layers"
        );
        assert!(session.has_subject_mask(&session.frames[&3]));
        session.project.sam_layers[0].enabled = false;
        assert!(!session.has_subject_mask(&session.frames[&3]));
        // A single enabled Keep layer is used as-is (no combined file is written).
        session.project.sam_layers[0].enabled = true;
        assert_eq!(
            session.subject_mask(3, None),
            Some(PathBuf::from("subject.png"))
        );
        // Removing a layer drops its points and results; the last layer cannot be removed.
        session.project.prompts.push(SamPrompt {
            frame: 3,
            x: 1.0,
            y: 1.0,
            positive: true,
            layer: second,
            text: None,
        });
        session.project.sam_boxes.push(ck_desktop_core::SamBox {
            frame: 3,
            layer: second,
            x0: 1.0,
            y0: 1.0,
            x1: 9.0,
            y1: 9.0,
        });
        session.sam_layer = second;
        session.remove_sam_layer(second);
        assert!(session.project.prompts.is_empty());
        assert!(session.project.sam_boxes.is_empty());
        assert_eq!(session.sam_layer, 0);
        // The last layer can be deleted too.
        session.remove_sam_layer(0);
        assert!(session.project.sam_layers.is_empty());
        assert_eq!(session.sam_layer, NO_LAYER);
    }

    #[test]
    fn subject_layers_are_bound_to_their_scene() {
        let mut session = session();
        session.project.scenes = vec![
            SceneRange {
                index: 0,
                start: 0,
                end_exclusive: 50,
                enabled: true,
            },
            SceneRange {
                index: 1,
                start: 50,
                end_exclusive: 100,
                enabled: true,
            },
        ];
        let first = session.sam_layer;
        assert_eq!(session.sam_layer(first).unwrap().scene, Some(0));
        // Scene 2 has no layer until the user adds a point, box or text prompt there.
        session.playhead = 60;
        session.sync_sam_layer();
        assert_eq!(session.sam_layer, NO_LAYER);
        assert!(session.project.layers_in_scene(1).is_empty());
        let second = session.layer_for_edit();
        assert_ne!(second, first);
        assert_eq!(session.sam_layer(second).unwrap().scene, Some(1));
        assert_eq!(session.project.layers_in_scene(0).len(), 1);
        assert_eq!(session.project.layers_in_scene(1).len(), 1);
        // A frame only composites its own scene's layers.
        assert!(
            session
                .project
                .layers_for_frame(10)
                .iter()
                .all(|l| l.id == first)
        );
        // Every layer can be deleted, including a scene's last one.
        session.remove_sam_layer(second);
        assert!(session.sam_layer(second).is_none());
        assert_eq!(session.sam_layer, NO_LAYER);
        // Layers from older projects (no scene) apply to every scene.
        let mut legacy = ck_desktop_core::SamLayer::subject();
        legacy.id = 99;
        session.project.sam_layers.push(legacy);
        assert_eq!(session.project.layers_in_scene(0).len(), 2);
        assert_eq!(session.project.layers_in_scene(1).len(), 1);
    }

    #[test]
    fn actual_sidebar_layer_checkbox_disables_the_layer() {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let session = session();
        let layer = session.sam_layer;
        studio.sessions.push(session);
        studio.left_tab = LeftTab::Subject;
        let mut time = 1.0_f64;
        let mut run = |studio: &mut Studio, events: Vec<egui::Event>| {
            time += 0.05;
            ctx.run(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 900.0))),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    if let Some(session) = studio.current_mut() {
                        session.sync_sam_layer();
                    }
                    studio.sidebar(ctx);
                },
            )
        };
        run(&mut studio, Vec::new());
        let at = studio
            .layer_checkbox_rects
            .first()
            .copied()
            .expect("layer row checkbox")
            .center();
        run(
            &mut studio,
            vec![
                egui::Event::PointerMoved(at),
                egui::Event::PointerButton {
                    pos: at,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
        run(
            &mut studio,
            vec![egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        run(&mut studio, Vec::new());
        let enabled = studio.sessions[0].sam_layer(layer).expect("layer").enabled;
        assert!(
            !enabled,
            "clicking the layer checkbox must disable the layer"
        );
    }

    #[test]
    fn legacy_global_layers_are_bound_to_their_scenes_on_load() {
        let mut session = session();
        session.project.scenes = vec![
            SceneRange {
                index: 0,
                start: 0,
                end_exclusive: 50,
                enabled: true,
            },
            SceneRange {
                index: 1,
                start: 50,
                end_exclusive: 100,
                enabled: true,
            },
        ];
        let layer = session.sam_layer;
        session.project.sam_layers[0].scene = None; // saved before scene-bound layers
        session.project.prompts.push(SamPrompt {
            frame: 60,
            x: 1.0,
            y: 1.0,
            positive: true,
            layer,
            text: None,
        });
        session.migrate_sam_layers();
        assert_eq!(session.sam_layer(layer).unwrap().scene, Some(1));
        assert!(session.project.sam_layers.iter().all(|l| l.scene.is_some()));
        // Scene 1 no longer shows it (and gets no automatic layer).
        session.playhead = 10;
        session.sync_sam_layer();
        assert_eq!(session.sam_layer, NO_LAYER);
        assert!(session.project.layers_in_scene(0).is_empty());
    }

    #[test]
    fn splitting_a_scene_splits_a_layer_with_points_on_both_halves() {
        let ctx = egui::Context::default();
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        let layer = session.sam_layer;
        for frame in [10, 70] {
            session.project.prompts.push(SamPrompt {
                frame,
                x: 1.0,
                y: 1.0,
                positive: true,
                layer,
                text: None,
            });
        }
        session.frames.insert(
            70,
            FrameFiles {
                sam_layers: BTreeMap::from([(layer, PathBuf::from("m70.png"))]),
                ..Default::default()
            },
        );
        session.playhead = 50;
        studio.sessions.push(session);
        studio.split_scene();
        let session = studio.current().unwrap();
        assert_eq!(session.project.scenes.len(), 2);
        let low = session.project.layers_in_scene(0);
        let high = session.project.layers_in_scene(1);
        assert_eq!((low.len(), high.len()), (1, 1));
        assert_eq!(low[0].id, layer);
        let moved = high[0].id;
        assert_ne!(moved, layer);
        assert!(
            session
                .project
                .prompts
                .iter()
                .any(|p| p.frame == 70 && p.layer == moved)
        );
        assert!(
            session
                .project
                .prompts
                .iter()
                .any(|p| p.frame == 10 && p.layer == layer)
        );
        assert!(session.frames[&70].sam_layers.contains_key(&moved));
        // Merging puts both layers back into one scene.
        studio.merge_scene_at(0);
        let session = studio.current().unwrap();
        assert_eq!(session.project.layers_in_scene(0).len(), 2);
    }

    #[test]
    fn removal_overlay_is_the_union_of_enabled_remove_layers() {
        let dir = std::env::temp_dir().join(format!("brp-removal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, values: [u8; 3]| {
            let path = dir.join(name);
            image::GrayImage::from_raw(3, 1, values.to_vec())
                .unwrap()
                .save(&path)
                .unwrap();
            path
        };
        let mut session = session();
        session.project.output_dir = dir.join("out");
        let hat = session.project.add_sam_layer(0);
        let cane = session.project.add_sam_layer(0);
        for id in [hat, cane] {
            session
                .project
                .sam_layers
                .iter_mut()
                .find(|l| l.id == id)
                .unwrap()
                .mode = ck_desktop_core::SamLayerMode::Remove;
        }
        let hat_mask = write("hat.png", [255, 0, 0]);
        let cane_mask = write("cane.png", [0, 0, 255]);
        session.frames.insert(
            1,
            FrameFiles {
                sam_layers: BTreeMap::from([
                    (0, write("subject.png", [255, 255, 255])),
                    (hat, hat_mask.clone()),
                    (cane, cane_mask),
                ]),
                ..Default::default()
            },
        );
        let union = session.removal_mask(1, None).unwrap();
        assert_eq!(
            image::open(&union).unwrap().into_luma8().into_raw(),
            [255, 0, 255]
        );
        // Cached while the inputs are unchanged.
        assert_eq!(session.removal_mask(1, None), Some(union));
        // Disabling one Remove layer leaves the other's own file; Keep layers never show red.
        session
            .project
            .sam_layers
            .iter_mut()
            .find(|l| l.id == cane)
            .unwrap()
            .enabled = false;
        assert_eq!(session.removal_mask(1, None), Some(hat_mask));
        // And the combined subject mask subtracts the hat.
        session.invalidate_sam_composites();
        let combined = session.subject_mask(1, None).unwrap();
        assert_eq!(
            image::open(&combined).unwrap().into_luma8().into_raw(),
            [0, 255, 255]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn old_projects_get_a_subject_layer_holding_their_masks() {
        let mut project = Project::new(media(), PathBuf::from("out"));
        project.sam_layers.clear();
        project.prompts.push(SamPrompt {
            frame: 0,
            x: 1.0,
            y: 1.0,
            positive: true,
            layer: 9,
            text: None,
        });
        let mut session = Session::new(project);
        assert_eq!(session.project.sam_layers.len(), 1);
        assert_eq!(session.project.prompts[0].layer, 0);
        session.frames.insert(
            2,
            FrameFiles {
                mask: Some(PathBuf::from("legacy.png")),
                ..Default::default()
            },
        );
        session.migrate_sam_layers();
        assert_eq!(
            session.frames[&2].sam_layers.get(&0),
            Some(&PathBuf::from("legacy.png"))
        );
    }

    #[test]
    fn hint_quality_presets_match_v1() {
        use ck_desktop_core::HintQuality;
        assert_eq!(
            [HintQuality::High, HintQuality::Rough, HintQuality::Loose].map(HintQuality::value),
            [0.95, 0.75, 0.50]
        );
        assert_eq!(HintQuality::default(), HintQuality::Rough);
    }

    #[test]
    fn scene_detection_artifacts_keep_playback_range_and_playhead() {
        let mut session = session();
        session.range = FrameRange {
            start: 31,
            end_exclusive: 84,
        };
        session.playhead = 70;
        session.project.scenes = vec![SceneRange {
            index: 0,
            start: 0,
            end_exclusive: 100,
            enabled: true,
        }];
        let detected = vec![
            SceneRange {
                index: 0,
                start: 0,
                end_exclusive: 60,
                enabled: true,
            },
            SceneRange {
                index: 1,
                start: 60,
                end_exclusive: 100,
                enabled: true,
            },
        ];
        session.apply_detected_scenes(detected.clone());
        assert_eq!(
            session.range,
            FrameRange {
                start: 31,
                end_exclusive: 84
            }
        );
        assert_eq!(session.playhead, 70);
        assert_eq!(session.project.scenes, detected);
        assert_eq!(session.selected_scene, 1);
    }

    #[test]
    fn actual_studio_inspector_width_is_stable_while_despill_value_changes() {
        let ctx = egui::Context::default();
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        studio.sessions.push(session());
        studio.right_tab = RightTab::Keyer;
        let mut time = 1.0;
        let mut widths = Vec::new();
        for value in [0.0_f32, 1.0, 0.5, 0.0, 1.0] {
            if let Some(current) = studio.current_mut() {
                current.project.despill_settings.amount = value;
            }
            for _ in 0..3 {
                time += 0.02;
                let mut width = None;
                let _ = ctx.run(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(
                            Pos2::ZERO,
                            Vec2::new(1520.0, 980.0),
                        )),
                        time: Some(time),
                        ..Default::default()
                    },
                    |ctx| {
                        studio.tools(ctx);
                        width = egui::containers::panel::PanelState::load(
                            ctx,
                            egui::Id::new("studio_inspector"),
                        )
                        .map(|state| state.rect.width());
                        egui::CentralPanel::default().show(ctx, |_ui| {});
                    },
                );
                widths.push(width.expect("inspector panel was rendered"));
            }
        }
        let first = widths[0];
        assert!(
            widths.iter().all(|width| (width - first).abs() < 0.1),
            "inspector width changed while only the despill value changed: {widths:?}"
        );
    }

    #[test]
    fn despill_row_fills_every_inspector_width_without_overflow() {
        let ctx = egui::Context::default();
        crate::theme::configure(&ctx);
        for width in [265.0_f32, 310.0, 430.0] {
            for value in [0.0_f32, 1.0] {
                let mut amount = value;
                let mut used = f32::NAN;
                let _ = ctx.run(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(600.0, 200.0))),
                        ..Default::default()
                    },
                    |ctx| {
                        egui::CentralPanel::default().show(ctx, |ui| {
                            let mut child = ui.new_child(
                                egui::UiBuilder::new()
                                    .max_rect(Rect::from_min_size(
                                        ui.min_rect().min,
                                        Vec2::new(width, 30.0),
                                    ))
                                    .layout(egui::Layout::top_down(egui::Align::Min)),
                            );
                            despill_slider_row(&mut child, Some(&mut amount), "test");
                            used = child.min_rect().width();
                        });
                    },
                );
                assert!(
                    (used - width).abs() < 0.2,
                    "despill row width {used} did not fit the {width} px panel at {value}"
                );
            }
        }
    }

    #[test]
    fn mask_layers_are_per_scene_and_copy_to_other_scenes() {
        let mut session = session();
        session.project.scenes = vec![
            SceneRange {
                index: 0,
                start: 0,
                end_exclusive: 50,
                enabled: true,
            },
            SceneRange {
                index: 1,
                start: 50,
                end_exclusive: 100,
                enabled: true,
            },
        ];
        session.playhead = 10;
        session.range = FrameRange {
            start: 0,
            end_exclusive: 50,
        };
        let id = session
            .add_mask_layer(MaskKind::Garbage)
            .expect("garbage layer");
        let layer = session
            .mask_stacks
            .stack(MaskKind::Garbage)
            .layers
            .iter()
            .find(|layer| layer.layer_id == id)
            .expect("added layer");
        assert_eq!(layer.scene_index, 0);
        assert_eq!((layer.scene_start, layer.scene_end), (0, 49));
        assert!(session.set_mask_layer_mode(MaskKind::Garbage, id, MaskMode::Add));
        session.set_mask_layer_visible(MaskKind::Garbage, id, false);
        assert!(!session.mask_stacks.stack(MaskKind::Garbage).layers[0].visible);

        // The next scene has no garbage layer and no active selection.
        session.playhead = 60;
        session.select_scene(1);
        assert!(session.active_mask_layer(MaskKind::Garbage).is_none());

        // Copy the first scene's garbage mask onto the second scene.
        session.playhead = 10;
        session.select_scene(0);
        assert_eq!(session.copy_mask_layers_to_scene(MaskKind::Garbage, 1), 1);
        assert_eq!(
            session
                .mask_stacks
                .stack(MaskKind::Garbage)
                .layers
                .iter()
                .filter(|layer| layer.scene_index == 1)
                .count(),
            1
        );

        // Removing the selected layer clears that selection.
        assert!(session.remove_mask_layer(MaskKind::Garbage, id));
        assert!(
            session
                .mask_stacks
                .stack(MaskKind::Garbage)
                .layers
                .iter()
                .all(|layer| layer.layer_id != id)
        );
        // The despill bank is independent.
        assert!(
            session
                .mask_stacks
                .stack(MaskKind::Despill)
                .layers
                .is_empty()
        );
    }

    #[test]
    fn actual_studio_garbage_tab_renders_mask_panel_with_layers() {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        session.playhead = 10;
        session.range = FrameRange {
            start: 0,
            end_exclusive: 50,
        };
        session.project.scenes = vec![SceneRange {
            index: 0,
            start: 0,
            end_exclusive: 100,
            enabled: true,
        }];
        session.add_mask_layer(MaskKind::Garbage);
        session.mask_ui.editing = true;
        studio.sessions.push(session);
        studio.left_tab = LeftTab::Garbage;
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1520.0, 980.0))),
                ..Default::default()
            },
            |ctx| {
                studio.tools(ctx);
                egui::CentralPanel::default().show(ctx, |_ui| {});
            },
        );
        assert_eq!(studio.sessions[0].mask_ui.active_kind, MaskKind::Garbage);
    }

    #[test]
    fn actual_studio_mask_click_on_last_knot_closes_the_shape() {
        use ck_core::mask_layers::BezierNode;
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        session.playhead = 0;
        session.range = FrameRange {
            start: 0,
            end_exclusive: 100,
        };
        let id = session
            .add_mask_layer(MaskKind::Garbage)
            .expect("garbage layer");
        let shape = ck_core::mask_layers::BezierShape::with_nodes(
            vec![
                BezierNode::new(300.0, 300.0),
                BezierNode::new(900.0, 300.0),
                BezierNode::new(900.0, 700.0),
            ],
            false,
        );
        session
            .mask_stacks
            .stack_mut(MaskKind::Garbage)
            .layers
            .iter_mut()
            .find(|layer| layer.layer_id == id)
            .expect("layer")
            .set_shape(0, shape);
        session
            .mask_stacks
            .stack_mut(MaskKind::Garbage)
            .select(Some(id));
        session.mask_ui.editing = true;
        studio.sessions.push(session);

        let mut time = 1.0_f64;
        let mut run = |studio: &mut Studio, events: Vec<egui::Event>| {
            time += 0.05;
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    studio.viewer(ctx);
                },
            );
        };
        run(&mut studio, Vec::new());
        let image = studio.viewer_image;
        let screen = image.min
            + Vec2::new(
                900.0 / 1920.0 * image.width(),
                700.0 / 1080.0 * image.height(),
            );
        run(
            &mut studio,
            vec![
                egui::Event::PointerMoved(screen),
                egui::Event::PointerButton {
                    pos: screen,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
        run(
            &mut studio,
            vec![egui::Event::PointerButton {
                pos: screen,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        run(&mut studio, Vec::new());
        let layer = studio.sessions[0]
            .mask_stacks
            .stack(MaskKind::Garbage)
            .layers
            .iter()
            .find(|layer| layer.layer_id == id)
            .expect("mask layer");
        let shape = layer.shape_at(0).expect("shape key");
        assert!(shape.closed, "clicking the last knot must close the shape");
    }

    #[test]
    fn actual_studio_quick_drag_from_a_knot_moves_it_instead_of_creating_one() {
        use ck_core::mask_layers::BezierNode;
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        session.playhead = 0;
        session.range = FrameRange {
            start: 0,
            end_exclusive: 100,
        };
        let id = session
            .add_mask_layer(MaskKind::Garbage)
            .expect("garbage layer");
        let shape = ck_core::mask_layers::BezierShape::with_nodes(
            vec![
                BezierNode::new(300.0, 300.0),
                BezierNode::new(900.0, 300.0),
                BezierNode::new(900.0, 700.0),
            ],
            true,
        );
        let layers = &mut session.mask_stacks.stack_mut(MaskKind::Garbage).layers;
        layers
            .iter_mut()
            .find(|layer| layer.layer_id == id)
            .expect("layer")
            .set_shape(0, shape);
        session
            .mask_stacks
            .stack_mut(MaskKind::Garbage)
            .select(Some(id));
        session.mask_ui.editing = true;
        studio.sessions.push(session);

        let mut time = 1.0_f64;
        let mut run = |studio: &mut Studio, events: Vec<egui::Event>| {
            time += 0.05;
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    studio.viewer(ctx);
                },
            );
        };
        run(&mut studio, Vec::new());
        let image = studio.viewer_image;
        let knot = image.min
            + Vec2::new(
                900.0 / 1920.0 * image.width(),
                300.0 / 1080.0 * image.height(),
            );
        // Press on the knot and move 20 px in the same frame: the drag is only
        // recognised once the pointer is already well off the knot.
        let end = knot + Vec2::new(20.0, 0.0);
        run(
            &mut studio,
            vec![
                egui::Event::PointerMoved(knot),
                egui::Event::PointerButton {
                    pos: knot,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerMoved(end),
            ],
        );
        run(&mut studio, vec![egui::Event::PointerMoved(end)]);
        run(
            &mut studio,
            vec![egui::Event::PointerButton {
                pos: end,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        run(&mut studio, Vec::new());
        let shape = studio.sessions[0]
            .mask_stacks
            .stack(MaskKind::Garbage)
            .layers
            .iter()
            .find(|layer| layer.layer_id == id)
            .expect("mask layer")
            .shape_at(0)
            .expect("shape key");
        assert_eq!(shape.nodes.len(), 3, "dragging a knot must not add one");
        assert!(
            shape.nodes[1].x > 905.0,
            "the dragged knot moves right, got x = {}",
            shape.nodes[1].x
        );
        assert_eq!(shape.nodes[0].x, 300.0);
    }

    #[test]
    fn actual_studio_mask_drag_on_empty_canvas_creates_a_smooth_node() {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        session.playhead = 0;
        session.range = FrameRange {
            start: 0,
            end_exclusive: 100,
        };
        let id = session
            .add_mask_layer(MaskKind::Garbage)
            .expect("garbage layer");
        session
            .mask_stacks
            .stack_mut(MaskKind::Garbage)
            .select(Some(id));
        session.mask_ui.editing = true;
        studio.sessions.push(session);

        let mut time = 1.0_f64;
        let mut run = |studio: &mut Studio, events: Vec<egui::Event>| {
            time += 0.05;
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    studio.viewer(ctx);
                },
            );
        };
        run(&mut studio, Vec::new());
        let start = studio.viewer_canvas.center();
        let end = start + Vec2::new(80.0, 40.0);
        run(
            &mut studio,
            vec![
                egui::Event::PointerMoved(start),
                egui::Event::PointerButton {
                    pos: start,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
        run(&mut studio, vec![egui::Event::PointerMoved(end)]);
        // While still dragging, the created knot must already be selected and
        // carry tangents so the overlay can draw its handles live.
        assert!(studio.mask_edit.drag.is_some());
        assert_eq!(studio.mask_edit.selected_knot, Some(0));
        let live = studio.sessions[0]
            .mask_stacks
            .stack(MaskKind::Garbage)
            .layers
            .iter()
            .find(|layer| layer.layer_id == id)
            .expect("mask layer")
            .shape_at(0)
            .expect("live shape while dragging");
        assert!(
            live.nodes[0].out_dx.abs() > 0.5 || live.nodes[0].out_dy.abs() > 0.5,
            "tangents must be visible while dragging"
        );
        run(
            &mut studio,
            vec![egui::Event::PointerButton {
                pos: end,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        run(&mut studio, Vec::new());
        let layer = studio.sessions[0]
            .mask_stacks
            .stack(MaskKind::Garbage)
            .layers
            .iter()
            .find(|layer| layer.layer_id == id)
            .expect("mask layer");
        let shape = layer.shape_at(0).expect("a shape key was created");
        assert_eq!(shape.nodes.len(), 1);
        let node = &shape.nodes[0];
        assert!(
            node.out_dx.abs() > 0.5 || node.out_dy.abs() > 0.5,
            "press-and-drag must pull a tangent"
        );
        assert!(
            node.in_dx.abs() > 0.5 || node.in_dy.abs() > 0.5,
            "the created node must be smooth"
        );
    }

    #[test]
    fn auto_segment_waits_for_jobs_and_needs_a_subject_point() {
        let ctx = egui::Context::default();
        let creation = eframe::CreationContext::_new_kittest(ctx);
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        session.playhead = 7;
        session.project.prompts.push(SamPrompt {
            frame: 7,
            x: 10.0,
            y: 10.0,
            positive: false,
            layer: 0,
            text: None,
        });
        studio.sessions.push(session);
        // Only a background point: nothing to segment, and the request is dropped.
        studio.segment_pending = Some((7, 0));
        studio.pump_auto_segment();
        assert_eq!(studio.segment_pending, None);
        assert!(studio.jobs.is_empty() && studio.active.is_none());
        // While other work is queued the request waits instead of being lost.
        studio.project_io_busy = true;
        studio.segment_pending = Some((7, 0));
        studio.pump_auto_segment();
        assert_eq!(studio.segment_pending, Some((7, 0)));
    }

    #[test]
    fn actual_studio_mask_tab_click_creates_a_layer_when_none_selected() {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        session.playhead = 0;
        session.range = FrameRange {
            start: 0,
            end_exclusive: 100,
        };
        session.mask_ui.editing = true;
        studio.sessions.push(session);
        studio.left_tab = LeftTab::Garbage;
        let mut time = 1.0_f64;
        let mut run = |studio: &mut Studio, events: Vec<egui::Event>| {
            time += 0.05;
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    studio.viewer(ctx);
                },
            );
        };
        run(&mut studio, Vec::new());
        let pos = studio.viewer_canvas.center();
        run(
            &mut studio,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
        run(
            &mut studio,
            vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        run(&mut studio, Vec::new());
        let stack = &studio.sessions[0].mask_stacks.stack(MaskKind::Garbage);
        assert_eq!(stack.layers.len(), 1, "a layer is auto-created");
        let layer = &stack.layers[0];
        let shape = layer.shape_at(0).expect("the click also created a knot");
        assert_eq!(shape.nodes.len(), 1);
        assert_eq!(studio.sessions[0].mask_ui.active_kind, MaskKind::Garbage);
    }

    #[test]
    fn actual_studio_mask_editing_adds_knots_from_viewer_clicks() {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        session.playhead = 0;
        session.range = FrameRange {
            start: 0,
            end_exclusive: 100,
        };
        let id = session
            .add_mask_layer(MaskKind::Garbage)
            .expect("garbage layer");
        session
            .mask_stacks
            .stack_mut(MaskKind::Garbage)
            .select(Some(id));
        session.mask_ui.editing = true;
        studio.sessions.push(session);

        let mut time = 1.0_f64;
        let mut run = |studio: &mut Studio, events: Vec<egui::Event>| {
            time += 0.05;
            let _ = ctx.run(
                egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0))),
                    time: Some(time),
                    events,
                    ..Default::default()
                },
                |ctx| {
                    studio.viewer(ctx);
                },
            );
        };
        run(&mut studio, Vec::new());
        let pos = studio.viewer_canvas.center();
        run(
            &mut studio,
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
        run(
            &mut studio,
            vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        run(&mut studio, Vec::new());
        let layer = studio.sessions[0]
            .mask_stacks
            .stack(MaskKind::Garbage)
            .layers
            .iter()
            .find(|layer| layer.layer_id == id)
            .expect("mask layer");
        let shape = layer.shape_at(0).expect("a shape key was created");
        assert_eq!(shape.nodes.len(), 1);
    }

    #[test]
    fn mask_layer_edits_are_undoable_and_redoable() {
        let mut session = session();
        session.playhead = 10;
        session.range = FrameRange {
            start: 0,
            end_exclusive: 50,
        };
        session.project.scenes = vec![SceneRange {
            index: 0,
            start: 0,
            end_exclusive: 100,
            enabled: true,
        }];
        let id = session
            .add_mask_layer(MaskKind::Garbage)
            .expect("garbage layer");
        assert_eq!(session.mask_stacks.stack(MaskKind::Garbage).layers.len(), 1);
        assert!(session.undo_mask_edit());
        assert!(
            session
                .mask_stacks
                .stack(MaskKind::Garbage)
                .layers
                .is_empty()
        );
        assert!(session.redo_mask_edit());
        assert_eq!(session.mask_stacks.stack(MaskKind::Garbage).layers.len(), 1);
        let id = session
            .mask_stacks
            .stack(MaskKind::Garbage)
            .layers
            .iter()
            .find(|layer| layer.layer_id == id)
            .map(|layer| layer.layer_id)
            .expect("restored layer");
        session.set_mask_layer_visible(MaskKind::Garbage, id, false);
        assert!(!session.mask_stacks.stack(MaskKind::Garbage).layers[0].visible);
        assert!(session.undo_mask_edit());
        assert!(session.mask_stacks.stack(MaskKind::Garbage).layers[0].visible);
    }

    #[test]
    fn new_mask_edit_clears_the_redo_branch() {
        let mut session = session();
        session.playhead = 10;
        session.project.scenes = vec![SceneRange {
            index: 0,
            start: 0,
            end_exclusive: 100,
            enabled: true,
        }];
        session.add_mask_layer(MaskKind::Garbage);
        assert!(session.undo_mask_edit());
        assert!(session.redo_mask_edit());
        assert!(!session.mask_history.can_redo());
        session.add_mask_layer(MaskKind::Garbage);
        assert!(!session.mask_history.can_redo());
    }

    #[test]
    fn ctrl_z_undoes_a_mask_layer_edit_through_the_shortcut_path() {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        session.playhead = 10;
        session.project.scenes = vec![SceneRange {
            index: 0,
            start: 0,
            end_exclusive: 100,
            enabled: true,
        }];
        session.add_mask_layer(MaskKind::Garbage);
        studio.sessions.push(session);
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(960.0, 640.0))),
                events: vec![egui::Event::Key {
                    key: egui::Key::Z,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::COMMAND,
                }],
                ..Default::default()
            },
            |ctx| studio.input(ctx),
        );
        assert!(
            studio.sessions[0]
                .mask_stacks
                .stack(MaskKind::Garbage)
                .layers
                .is_empty()
        );
    }

    #[test]
    fn pointer_context_follows_the_hovered_panel() {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        studio.viewer_canvas = Rect::from_min_size(Pos2::ZERO, Vec2::new(100.0, 100.0));
        studio.timeline_rect = Rect::from_min_size(Pos2::new(200.0, 0.0), Vec2::new(100.0, 100.0));
        studio.inspector_rect = Rect::from_min_size(Pos2::new(400.0, 0.0), Vec2::new(100.0, 100.0));
        studio.left_panel_rect =
            Rect::from_min_size(Pos2::new(-100.0, 0.0), Vec2::new(100.0, 100.0));
        use crate::app_history::AppContext;
        assert_eq!(
            studio.pointer_undo_context(Some(Pos2::new(50.0, 50.0))),
            AppContext::Mask
        );
        assert_eq!(
            studio.pointer_undo_context(Some(Pos2::new(250.0, 50.0))),
            AppContext::Timeline
        );
        assert_eq!(
            studio.pointer_undo_context(Some(Pos2::new(450.0, 50.0))),
            AppContext::Keyer
        );
        assert_eq!(
            studio.pointer_undo_context(Some(Pos2::new(-50.0, 50.0))),
            AppContext::Media
        );
        assert_eq!(
            studio.pointer_undo_context(Some(Pos2::new(900.0, 900.0))),
            AppContext::Global
        );
    }

    #[test]
    fn ctrl_z_over_the_timeline_restores_the_previous_playback_range() {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        session.playhead = 10;
        session.range = FrameRange {
            start: 0,
            end_exclusive: 50,
        };
        session.project.scenes = vec![SceneRange {
            index: 0,
            start: 0,
            end_exclusive: 100,
            enabled: true,
        }];
        let original = session.range;
        studio.sessions.push(session);
        studio.timeline_rect = Rect::from_min_size(Pos2::new(500.0, 500.0), Vec2::new(200.0, 80.0));
        studio.record_app_edit(crate::app_history::AppContext::Timeline, "test range");
        studio.sessions[0].range = FrameRange {
            start: 5,
            end_exclusive: 40,
        };
        let pointer = Pos2::new(550.0, 550.0);
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 800.0))),
                events: vec![
                    egui::Event::PointerMoved(pointer),
                    egui::Event::Key {
                        key: egui::Key::Z,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::COMMAND,
                    },
                ],
                ..Default::default()
            },
            |ctx| studio.input(ctx),
        );
        assert_eq!(studio.sessions[0].range, original);
    }

    #[test]
    fn ctrl_z_over_the_inspector_restores_the_previous_despill_amount() {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        session.playhead = 10;
        session.project.scenes = vec![SceneRange {
            index: 0,
            start: 0,
            end_exclusive: 100,
            enabled: true,
        }];
        let original = session.project.despill_settings;
        studio.sessions.push(session);
        studio.inspector_rect =
            Rect::from_min_size(Pos2::new(500.0, 100.0), Vec2::new(200.0, 80.0));
        studio.record_app_edit(crate::app_history::AppContext::Keyer, "despill");
        studio.sessions[0].project.despill_settings.amount = 0.75;
        let pointer = Pos2::new(550.0, 150.0);
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1200.0, 800.0))),
                events: vec![
                    egui::Event::PointerMoved(pointer),
                    egui::Event::Key {
                        key: egui::Key::Z,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::COMMAND,
                    },
                ],
                ..Default::default()
            },
            |ctx| studio.input(ctx),
        );
        assert_eq!(studio.sessions[0].project.despill_settings, original);
    }

    #[test]
    fn preview_cache_key_includes_mask_revision_and_despill_amount() {
        let mut session = session();
        let base = preview_composition_key(
            &session,
            10,
            crate::preview::ViewMode::Composite,
            true,
            crate::preview::Background::Checker,
        );
        session.project.despill_settings.amount = 0.5;
        let despilled = preview_composition_key(
            &session,
            10,
            crate::preview::ViewMode::Composite,
            true,
            crate::preview::Background::Checker,
        );
        assert_ne!(base, despilled, "despill amount must invalidate the cache");
        session.revision = session.revision.wrapping_add(1);
        let masked = preview_composition_key(
            &session,
            10,
            crate::preview::ViewMode::Composite,
            true,
            crate::preview::Background::Checker,
        );
        assert_ne!(
            despilled, masked,
            "mask revisions must invalidate the cache"
        );
    }

    #[test]
    fn mask_edits_change_only_the_cache_keys_of_frames_they_cover() {
        use ck_core::mask_layers::{BezierNode, BezierShape};
        let mut session = session();
        let key = |session: &Session, frame: u64| {
            preview_composition_key(
                session,
                frame,
                crate::preview::ViewMode::Composite,
                true,
                crate::preview::Background::Checker,
            )
        };
        let id = session.add_mask_layer(MaskKind::Garbage).expect("layer");
        let shape = |x: f32| {
            BezierShape::with_nodes(
                vec![
                    BezierNode::new(x, 100.0),
                    BezierNode::new(x + 200.0, 100.0),
                    BezierNode::new(x + 200.0, 300.0),
                ],
                true,
            )
        };
        fn layer(session: &mut Session, id: u32) -> &mut MaskLayer {
            session
                .mask_stacks
                .stack_mut(MaskKind::Garbage)
                .layers
                .iter_mut()
                .find(|layer| layer.layer_id == id)
                .expect("layer")
        }
        layer(&mut session, id).set_scene_range(0, 9);
        layer(&mut session, id).set_shape(0, shape(100.0));
        let (inside, outside) = (key(&session, 4), key(&session, 40));
        layer(&mut session, id).set_shape(0, shape(150.0));
        session.mask_revision += 1;
        assert_ne!(key(&session, 4), inside, "an edited frame gets a new key");
        assert_eq!(
            key(&session, 40),
            outside,
            "frames the mask does not cover keep theirs"
        );
        layer(&mut session, id).softness = 25;
        assert_ne!(key(&session, 4), inside);
        let edited = key(&session, 4);
        layer(&mut session, id).softness = 0;
        layer(&mut session, id).set_shape(0, shape(100.0));
        assert_eq!(
            key(&session, 4),
            inside,
            "undoing an edit finds the old cached frame again"
        );
        assert_ne!(key(&session, 4), edited);
    }

    #[test]
    fn auto_cache_frames_are_ordered_nearest_to_the_playhead() {
        let ordered = auto_cache_frame_order([10, 4, 7, 3].into_iter(), 6);
        assert_eq!(ordered.into_iter().collect::<Vec<_>>(), vec![7, 4, 3, 10]);
    }

    #[test]
    fn scene_despill_override_changes_the_preview_composition_key() {
        let mut session = session();
        session.project.despill_settings.amount = 0.2;
        let before =
            preview_composition_signature(&session, ViewMode::Composite, true, Background::Checker);
        session.project.scene_overrides.insert(
            0,
            ck_desktop_core::SceneOverrides {
                despill: Some(ck_desktop_core::SceneOverrideValue {
                    enabled: true,
                    amount: 0.8,
                }),
            },
        );
        let after =
            preview_composition_signature(&session, ViewMode::Composite, true, Background::Checker);
        assert_ne!(before, after);
    }

    #[test]
    fn mask_copy_and_paste_moves_the_selected_layer_to_another_scene() {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        session.project.scenes = vec![
            SceneRange {
                index: 0,
                start: 0,
                end_exclusive: 10,
                enabled: true,
            },
            SceneRange {
                index: 1,
                start: 10,
                end_exclusive: 20,
                enabled: true,
            },
        ];
        session.playhead = 2;
        studio.sessions.push(session);
        studio.selected = 0;
        studio.left_tab = LeftTab::Garbage;
        let layer_id = studio.sessions[0]
            .add_mask_layer(MaskKind::Garbage)
            .expect("layer");
        studio.sessions[0]
            .mask_stacks
            .stack_mut(MaskKind::Garbage)
            .select(Some(layer_id));
        assert!(studio.copy_selected_mask_layer());
        studio.sessions[0].playhead = 12;
        assert!(studio.paste_mask_layer());
        let stack = studio.sessions[0].mask_stacks.stack(MaskKind::Garbage);
        assert_eq!(stack.layers.len(), 2);
        let pasted = stack.layers.last().unwrap();
        assert_eq!(pasted.scene_index, 1);
        assert_ne!(pasted.layer_id, layer_id);
        assert_eq!(stack.selected, Some(pasted.layer_id));
    }

    #[test]
    fn ctrl_c_and_ctrl_v_copy_the_selected_mask_layer_to_another_scene() {
        let ctx = egui::Context::default();
        ctx.set_pixels_per_point(1.0);
        let creation = eframe::CreationContext::_new_kittest(ctx.clone());
        let mut studio = Studio::new(&creation, PathBuf::new(), None, None, true, false, None);
        let mut session = session();
        session.project.scenes = vec![
            SceneRange {
                index: 0,
                start: 0,
                end_exclusive: 10,
                enabled: true,
            },
            SceneRange {
                index: 1,
                start: 10,
                end_exclusive: 20,
                enabled: true,
            },
        ];
        session.playhead = 2;
        session.add_mask_layer(MaskKind::Garbage);
        studio.sessions.push(session);
        studio.left_tab = LeftTab::Garbage;
        let shortcut = |key| egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(960.0, 640.0))),
            events: vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::COMMAND,
            }],
            ..Default::default()
        };
        let _ = ctx.run(shortcut(egui::Key::C), |ctx| studio.input(ctx));
        studio.sessions[0].playhead = 12;
        let _ = ctx.run(shortcut(egui::Key::V), |ctx| studio.input(ctx));
        let layers = &studio.sessions[0]
            .mask_stacks
            .stack(MaskKind::Garbage)
            .layers;
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[1].scene_index, 1);
        assert_eq!(
            studio.sessions[0]
                .mask_stacks
                .stack(MaskKind::Garbage)
                .selected,
            Some(layers[1].layer_id)
        );
    }
}
