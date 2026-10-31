use crate::{CkSettings, MediaInfo, PROJECT_SCHEMA_VERSION, SamPrompt, SceneRange};
use anyhow::{Context, Result, ensure};
use ck_core::mask_layers::{MaskKind, MaskLayer, MaskStack};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

/// Interactive result-only adjustment. Zero preserves calculated pixels exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DespillSettings {
    pub amount: f32,
}
impl DespillSettings {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.amount.is_finite() && (0.0..=1.0).contains(&self.amount),
            "Result despill must be between 0 and 100 percent"
        );
        Ok(())
    }
    pub fn is_enabled(self) -> bool {
        self.amount.is_finite() && self.amount > 0.0
    }
}

/// One v1 scene override entry (`{"enabled": true, "amount": 0.4}`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SceneOverrideValue {
    pub enabled: bool,
    pub amount: f32,
}

/// Per-scene render overrides, v1 `job.overrides`.
///
/// Only Result despill exists today (v1 `OVERRIDE_SPECS`); an absent or
/// disabled entry means the global [`DespillSettings`] applies.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SceneOverrides {
    pub despill: Option<SceneOverrideValue>,
}

impl SceneOverrides {
    /// Effective 0..1 despill amount for this scene (global fallback).
    pub fn despill_amount(&self, fallback: f32) -> f32 {
        match self.despill {
            Some(entry) if entry.enabled && entry.amount.is_finite() => {
                entry.amount.clamp(0.0, 1.0)
            }
            _ => fallback,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.despill.is_none()
    }
}

/// Convert the app's half-open frame range to the inclusive scene bounds that
/// v1-compatible mask layers store (`scene_start..=scene_end`).
///
/// ck-core mask layers use v1's inclusive-at-both-ends rule, while every
/// [`SceneRange`] in the project is half-open, so the last active frame is
/// `end_exclusive - 1`. An empty range maps to `(start, start - 1)`, which
/// ck-core treats as "unrestricted".
pub fn scene_bounds_inclusive(start: u64, end_exclusive: u64) -> (i64, i64) {
    (start as i64, end_exclusive.saturating_sub(1) as i64)
}

/// On-disk garbage/despill mask layer banks (v1 `mask_layers`).
///
/// Serialised as the v1 project shape `{ "garbage": [...], "despill": [...] }`
/// where each bank is an ordered list of [`MaskLayer`] values in compositing
/// order. The layer serde unit is the ck-core one, which keeps v1's field names
/// (`id`, `name`, `mode`, `visible`, `color`, `softness`, `scene`,
/// `scene_start`, `scene_end`, `interp`, `auto_key`, `tangents`, `shapes`) and
/// adds the app extensions `kind`, `opacity`, `width` and `height`. The v1
/// `frames` PNG-name list is not part of this schema and is ignored on load.
///
/// Painted raster keyframes are runtime state in each layer's
/// [`MaskLayer::doc`] (`#[serde(skip)]`): v1 stored them as PNGs next to the
/// JSON, and this port stores them as `.chroma` blobs (see
/// `ck_desktop::project_store`). Layer metadata therefore round-trips through
/// project JSON while the painted masks round-trip through the container blobs.
///
/// The layer selected in the viewer is UI state per `(kind, scene)`; it is not
/// persisted here. Use [`MaskStack::selected`] on the runtime stack instead.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaskLayers {
    /// Garbage mask layers in compositing order.
    pub garbage: Vec<MaskLayer>,
    /// Despill mask layers in compositing order.
    pub despill: Vec<MaskLayer>,
}

impl MaskLayers {
    /// Whether both banks hold no layers.
    pub fn is_empty(&self) -> bool {
        self.garbage.is_empty() && self.despill.is_empty()
    }

    /// Layers of one bank in compositing order.
    pub fn layers(&self, kind: MaskKind) -> &[MaskLayer] {
        match kind {
            MaskKind::Garbage => &self.garbage,
            MaskKind::Despill => &self.despill,
        }
    }

    /// Mutable layer list of one bank, in compositing order.
    pub fn layers_mut(&mut self, kind: MaskKind) -> &mut Vec<MaskLayer> {
        match kind {
            MaskKind::Garbage => &mut self.garbage,
            MaskKind::Despill => &mut self.despill,
        }
    }

    /// Layer with `layer_id` in one bank.
    pub fn layer(&self, kind: MaskKind, layer_id: u32) -> Option<&MaskLayer> {
        self.layers(kind)
            .iter()
            .find(|layer| layer.layer_id == layer_id)
    }

    /// Mutable layer with `layer_id` in one bank.
    pub fn layer_mut(&mut self, kind: MaskKind, layer_id: u32) -> Option<&mut MaskLayer> {
        self.layers_mut(kind)
            .iter_mut()
            .find(|layer| layer.layer_id == layer_id)
    }

    /// Build the runtime stack of one bank.
    ///
    /// Thin wrapper over [`MaskStack::from_layers`]: the stack kind is
    /// authoritative, invalid modes fall back to the kind's v1 default and every
    /// layer's raster size adopts `width`/`height` (`set_raster_size` resamples
    /// painted keys when the stored size differs).
    pub fn stack(&self, width: u32, height: u32, kind: MaskKind) -> MaskStack {
        MaskStack::from_layers(width, height, kind, self.layers(kind).to_vec())
    }

    /// Store a runtime stack back into its bank (painted keys included).
    pub fn set_stack(&mut self, stack: &MaskStack) {
        *self.layers_mut(stack.kind) = stack.layers.clone();
    }

    /// Convert both banks into runtime stacks, consuming the on-disk layers.
    pub fn into_stacks(self, width: u32, height: u32) -> MaskStacks {
        MaskStacks {
            garbage: MaskStack::from_layers(width, height, MaskKind::Garbage, self.garbage),
            despill: MaskStack::from_layers(width, height, MaskKind::Despill, self.despill),
        }
    }

    /// Normalise the schema invariants that v1 data does not carry.
    ///
    /// The bank determines each layer's `kind`; a mode that is not valid for the
    /// kind becomes the kind's default, and layers without the app-extension
    /// `width`/`height` (v1 layers) adopt the source resolution when one is
    /// known.
    pub fn normalize(&mut self, width: u32, height: u32) {
        for kind in [MaskKind::Garbage, MaskKind::Despill] {
            for layer in self.layers_mut(kind) {
                layer.kind = kind;
                if !layer.mode.is_valid_for(kind) {
                    layer.mode = kind.default_mode();
                }
                if (layer.width == 0 || layer.height == 0) && width > 0 && height > 0 {
                    layer.set_raster_size(width, height);
                }
            }
        }
    }

    /// Validate the project-level mask invariants.
    pub fn validate(&self) -> Result<()> {
        for kind in [MaskKind::Garbage, MaskKind::Despill] {
            let mut ids = BTreeSet::new();
            for layer in self.layers(kind) {
                ensure!(
                    layer.opacity.is_finite() && (0.0..=1.0).contains(&layer.opacity),
                    "Mask layer opacity must be between 0 and 100 percent"
                );
                ensure!(
                    (layer.width == 0) == (layer.height == 0),
                    "Mask layer raster size must define both width and height"
                );
                ensure!(
                    ids.insert(layer.layer_id),
                    "Mask layer ids must be unique within a stack"
                );
                if !layer.doc.frames.is_empty() {
                    ensure!(
                        layer.width > 0 && layer.height > 0,
                        "Painted mask layer has no raster resolution"
                    );
                    for (frame, mask) in &layer.doc.frames {
                        ensure!(
                            mask.width() == layer.width && mask.height() == layer.height,
                            "Painted mask raster at frame {frame} does not match its layer resolution"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

/// Runtime garbage/despill mask stacks of one project document.
///
/// Built from the serialised [`MaskLayers`] with [`Project::mask_stacks`] and
/// written back with [`Project::set_mask_stacks`] / [`Project::replace_mask_stacks`]
/// before saving. `MaskStack::selected` is per-window UI state and is not
/// persisted.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskStacks {
    /// Garbage mask layers in compositing order.
    pub garbage: MaskStack,
    /// Despill mask layers in compositing order.
    pub despill: MaskStack,
}

impl MaskStacks {
    /// Empty stacks sharing a raster size.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            garbage: MaskStack::new(width, height, MaskKind::Garbage),
            despill: MaskStack::new(width, height, MaskKind::Despill),
        }
    }

    /// Empty zero-sized stacks (a project without a source).
    pub fn empty() -> Self {
        Self::new(0, 0)
    }

    /// Stack of one bank.
    pub fn stack(&self, kind: MaskKind) -> &MaskStack {
        match kind {
            MaskKind::Garbage => &self.garbage,
            MaskKind::Despill => &self.despill,
        }
    }

    /// Mutable stack of one bank.
    pub fn stack_mut(&mut self, kind: MaskKind) -> &mut MaskStack {
        match kind {
            MaskKind::Garbage => &mut self.garbage,
            MaskKind::Despill => &mut self.despill,
        }
    }

    /// Whether both stacks hold no layers.
    pub fn is_empty(&self) -> bool {
        self.garbage.layers.is_empty() && self.despill.layers.is_empty()
    }

    /// Shared raster size of both stacks.
    pub fn raster_size(&self) -> (u32, u32) {
        (self.garbage.width, self.garbage.height)
    }

    /// Consume the stacks into the serialisable layer banks (painted keys
    /// included; they are written to `.chroma` blobs on save).
    pub fn into_mask_layers(self) -> MaskLayers {
        MaskLayers {
            garbage: self.garbage.layers,
            despill: self.despill.layers,
        }
    }

    /// Copy every layer of `source_scene` in one bank onto `target_scene`.
    ///
    /// Wrapper over [`MaskStack::copy_layers_from_scene`] for the UI "copy mask
    /// to other scene" action; `target_start..=target_end` are inclusive source
    /// frames (use [`scene_bounds_inclusive`] for a saved [`SceneRange`]). Keys
    /// keep their scene-local offset and painted rasters are deep-copied.
    pub fn copy_layers_from_scene(
        &mut self,
        kind: MaskKind,
        source_scene: u32,
        target_scene: u32,
        target_start: i64,
        target_end: i64,
    ) -> Vec<u32> {
        self.stack_mut(kind).copy_layers_from_scene(
            source_scene,
            target_scene,
            target_start,
            target_end,
        )
    }
}

/// How a SAM layer combines with the layers below it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamLayerMode {
    /// Adds its mask (union).
    #[default]
    Keep,
    /// Subtracts its mask from the layers below it.
    Remove,
}

/// One subject-mask layer: its own points (`SamPrompt::layer`), its own SAM jobs and results.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamLayer {
    pub id: u64,
    pub name: String,
    #[serde(default)]
    pub mode: SamLayerMode,
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
    /// SAM 3.1 whole-frame text prompt ("person"): every match on the range's first frame.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prompt: String,
    /// Scene index the layer belongs to (like garbage/despill layers). `None` = every scene
    /// (layers from projects saved before scene-bound layers).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene: Option<u32>,
}

fn enabled_by_default() -> bool {
    true
}

impl SamLayer {
    pub fn subject() -> Self {
        Self {
            id: 0,
            name: "Subject".into(),
            mode: SamLayerMode::Keep,
            enabled: true,
            prompt: String::new(),
            scene: None,
        }
    }

    /// True when the layer is shown and applied in `scene`.
    pub fn in_scene(&self, scene: u32) -> bool {
        self.scene.is_none_or(|own| own == scene)
    }
}

/// v1 mask hint quality presets passed to CorridorKey with a hint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HintQuality {
    High,
    #[default]
    Rough,
    Loose,
}

impl HintQuality {
    pub fn value(self) -> f32 {
        match self {
            Self::High => 0.95,
            Self::Rough => 0.75,
            Self::Loose => 0.50,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::High => "High",
            Self::Rough => "Rough",
            Self::Loose => "Loose",
        }
    }
    pub fn note(self) -> &'static str {
        match self {
            Self::High => "Clean artist matte or precise mask.",
            Self::Rough => "Rough but useful automated hint.",
            Self::Loose => "Scribble-like, weak or unreliable mask.",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Project {
    pub schema_version: u32,
    pub media: Option<MediaInfo>,
    pub scenes: Vec<SceneRange>,
    pub prompts: Vec<SamPrompt>,
    /// Subject-mask layers in application order (index 0 is applied first).
    #[serde(default)]
    pub sam_layers: Vec<SamLayer>,
    /// Mask hint quality given to CorridorKey.
    #[serde(default)]
    pub hint_quality: HintQuality,
    /// SAM 3.1 box prompts (one per layer and frame).
    #[serde(default)]
    pub sam_boxes: Vec<crate::SamBox>,
    pub output_dir: PathBuf,
    #[serde(default)]
    pub ck_settings: CkSettings,
    #[serde(default)]
    pub despill_settings: DespillSettings,
    /// Per-scene garbage/despill mask layers (v1 `mask_layers` banks).
    #[serde(default)]
    pub mask_layers: MaskLayers,
    /// Per-scene render overrides keyed by scene index (v1 `job.overrides`).
    #[serde(default)]
    pub scene_overrides: BTreeMap<u32, SceneOverrides>,
}
impl Default for Project {
    fn default() -> Self {
        Self {
            schema_version: PROJECT_SCHEMA_VERSION,
            media: None,
            scenes: Vec::new(),
            prompts: Vec::new(),
            // Layers are created when the user first adds a point, box or text prompt.
            sam_layers: Vec::new(),
            hint_quality: HintQuality::Rough,
            sam_boxes: Vec::new(),
            output_dir: PathBuf::new(),
            ck_settings: CkSettings::default(),
            despill_settings: DespillSettings::default(),
            mask_layers: MaskLayers::default(),
            scene_overrides: BTreeMap::new(),
        }
    }
}
impl Project {
    pub fn new(media: MediaInfo, output_dir: PathBuf) -> Self {
        let scenes = vec![SceneRange {
            index: 0,
            start: 0,
            end_exclusive: media.frame_count,
            enabled: true,
        }];
        Self {
            media: Some(media),
            scenes,
            output_dir,
            ..Self::default()
        }
    }
    /// Projects saved before SAM layers have none: give them the default layer and attach any
    /// points that name a missing layer to the first layer.
    pub fn ensure_sam_layers(&mut self) {
        let ids: std::collections::BTreeSet<u64> = self.sam_layers.iter().map(|l| l.id).collect();
        let orphans: Vec<usize> = (0..self.prompts.len())
            .filter(|&i| !ids.contains(&self.prompts[i].layer))
            .collect();
        for i in orphans {
            let scene = self.scene_of_frame(self.prompts[i].frame);
            self.prompts[i].layer = self.ensure_scene_layer(scene);
        }
    }

    /// Scene index of `frame` (0 when no scene contains it).
    pub fn scene_of_frame(&self, frame: u64) -> u32 {
        self.scene_index_containing(frame).unwrap_or(0) as u32
    }

    /// Layers shown and applied in `scene`, in application order.
    pub fn layers_in_scene(&self, scene: u32) -> Vec<&SamLayer> {
        self.sam_layers
            .iter()
            .filter(|layer| layer.in_scene(scene))
            .collect()
    }

    /// Layers that apply to `frame` (its scene's layers), in application order.
    pub fn layers_for_frame(&self, frame: u64) -> Vec<SamLayer> {
        let scene = self.scene_of_frame(frame);
        self.sam_layers
            .iter()
            .filter(|layer| layer.in_scene(scene))
            .cloned()
            .collect()
    }

    /// First layer of `scene`, creating its "Subject" layer when the scene has none.
    pub fn ensure_scene_layer(&mut self, scene: u32) -> u64 {
        if let Some(layer) = self.sam_layers.iter().find(|layer| layer.in_scene(scene)) {
            return layer.id;
        }
        let id = self.next_sam_layer_id();
        self.sam_layers.push(SamLayer {
            id,
            scene: Some(scene),
            ..SamLayer::subject()
        });
        id
    }

    fn next_sam_layer_id(&self) -> u64 {
        self.sam_layers.iter().map(|l| l.id + 1).max().unwrap_or(0)
    }

    /// A new Keep layer of `scene` with a fresh id, appended on top.
    pub fn add_sam_layer(&mut self, scene: u32) -> u64 {
        let id = self.next_sam_layer_id();
        let count = self.layers_in_scene(scene).len();
        self.sam_layers.push(SamLayer {
            id,
            name: format!("Layer {}", count + 1),
            mode: SamLayerMode::Keep,
            enabled: true,
            prompt: String::new(),
            scene: Some(scene),
        });
        id
    }

    /// Scene `index` was split in two: later scenes moved up by one.
    pub fn shift_sam_layers_after_split(&mut self, index: u32) {
        for layer in &mut self.sam_layers {
            if let Some(scene) = layer.scene.as_mut()
                && *scene > index
            {
                *scene += 1;
            }
        }
    }

    /// Scenes `index` and `index + 1` were merged: the second one's layers join the first,
    /// later scenes moved down by one.
    pub fn shift_sam_layers_after_merge(&mut self, index: u32) {
        for layer in &mut self.sam_layers {
            if let Some(scene) = layer.scene.as_mut()
                && *scene > index
            {
                *scene -= 1;
            }
        }
    }

    /// Index of the scene whose half-open range contains `frame`.
    pub fn scene_index_containing(&self, frame: u64) -> Option<usize> {
        self.scenes
            .iter()
            .position(|scene| scene.start <= frame && frame < scene.end_exclusive)
    }

    /// Drop overrides whose scene index no longer exists (scene detection,
    /// split/merge and project migration can leave stale entries).
    pub fn sanitize_scene_overrides(&mut self) {
        let scenes = self.scenes.len();
        self.scene_overrides
            .retain(|index, _| (*index as usize) < scenes);
    }

    /// Result despill for one frame, applying a v1 scene override when active.
    pub fn despill_settings_for_frame(&self, frame: u64) -> DespillSettings {
        let mut settings = self.despill_settings;
        if let Some(index) = self.scene_index_containing(frame)
            && let Some(overrides) = self.scene_overrides.get(&(index as u32))
        {
            settings.amount = overrides.despill_amount(settings.amount);
        }
        settings
    }

    pub fn validate(&self) -> Result<()> {
        self.ck_settings.validate()?;
        self.despill_settings.validate()?;
        for (index, overrides) in &self.scene_overrides {
            ensure!(
                (*index as usize) < self.scenes.len(),
                "Scene override references a missing scene"
            );
            if let Some(entry) = overrides.despill {
                ensure!(
                    entry.amount.is_finite() && (0.0..=1.0).contains(&entry.amount),
                    "Scene despill override must be between 0 and 100 percent"
                );
            }
        }
        ensure!(
            (1..=PROJECT_SCHEMA_VERSION).contains(&self.schema_version),
            "Unsupported native project schema"
        );
        self.mask_layers.validate()?;
        if let Some(media) = &self.media {
            media.validate()?;
            let mut end = 0;
            for scene in &self.scenes {
                scene.range().validate(media.frame_count)?;
                ensure!(
                    scene.start >= end,
                    "Project scene ranges overlap or are out of order"
                );
                end = scene.end_exclusive;
            }
            for point in &self.prompts {
                ensure!(
                    point.frame < media.frame_count
                        && point.x.is_finite()
                        && point.y.is_finite()
                        && point.x >= 0.0
                        && point.y >= 0.0
                        && point.x < media.width as f32
                        && point.y < media.height as f32,
                    "Project prompt is outside the source frame"
                );
            }
        } else {
            ensure!(
                self.scenes.is_empty() && self.prompts.is_empty(),
                "A project with no source cannot contain scenes or prompts"
            );
        }
        Ok(())
    }
    /// Upgrade an in-memory project document to the current project schema.
    ///
    /// * Version 1 (pre-mask-layers) is accepted and bumped to
    ///   [`PROJECT_SCHEMA_VERSION`]: a missing `mask_layers` key deserialises to
    ///   empty banks, so older `.chroma`/native JSON projects open with empty
    ///   stacks and are rewritten in the current schema on the next save.
    /// * Mask bank kinds/modes are normalised for their stack and zero-sized
    ///   v1 layers adopt the source resolution.
    /// * Future versions are rejected so a newer file is never silently
    ///   downgraded.
    pub fn migrate(&mut self) -> Result<()> {
        ensure!(
            (1..=PROJECT_SCHEMA_VERSION).contains(&self.schema_version),
            "Unsupported native project schema"
        );
        self.schema_version = PROJECT_SCHEMA_VERSION;
        let (width, height) = self.mask_raster_size();
        self.mask_layers.normalize(width, height);
        Ok(())
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut project: Self = serde_json::from_slice(&fs::read(path).context("Reading native project")?).context("This is not a supported native JSON project; legacy .brp ZIP import is not implemented yet")?;
        project.migrate()?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        if let Some(media) = &mut project.media
            && media.path.is_relative()
        {
            media.path = base.join(&media.path);
        }
        if project.output_dir.is_relative() {
            project.output_dir = base.join(&project.output_dir);
        }
        project.validate()?;
        Ok(project)
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(base)?;
        let mut document = self.clone();
        document.migrate()?;
        if let Some(media) = &mut document.media
            && let Ok(relative) = media.path.strip_prefix(base)
        {
            media.path = relative.to_path_buf();
        }
        if let Ok(relative) = document.output_dir.strip_prefix(base) {
            document.output_dir = relative.to_path_buf();
        }
        write_json_atomic(path, &document)
    }

    /// Raster size mask layers use for this project.
    ///
    /// v1 masks live at the source resolution; a project without a source has
    /// zero-sized mask rasters.
    pub fn mask_raster_size(&self) -> (u32, u32) {
        self.media
            .as_ref()
            .map_or((0, 0), |media| (media.width, media.height))
    }

    /// Build the runtime garbage/despill mask stacks of this document.
    ///
    /// The returned stacks own deep copies of the painted masks; keep them in
    /// the editor's runtime state and write them back with
    /// [`Project::set_mask_stacks`] or [`Project::replace_mask_stacks`] before
    /// saving (the `.chroma` writer persists them as blobs).
    pub fn mask_stacks(&self) -> MaskStacks {
        let (width, height) = self.mask_raster_size();
        self.mask_layers.clone().into_stacks(width, height)
    }

    /// Build the runtime stack of one garbage/despill bank.
    pub fn mask_stack(&self, kind: MaskKind) -> MaskStack {
        let (width, height) = self.mask_raster_size();
        self.mask_layers.stack(width, height, kind)
    }

    /// Store one runtime stack back into its bank (the stack kind selects it).
    pub fn set_mask_stack(&mut self, stack: &MaskStack) {
        self.mask_layers.set_stack(stack);
    }

    /// Store both runtime stacks back with minimal copying.
    pub fn replace_mask_stacks(&mut self, stacks: MaskStacks) {
        self.mask_layers = stacks.into_mask_layers();
    }

    /// Store copies of both runtime stacks.
    pub fn set_mask_stacks(&mut self, stacks: &MaskStacks) {
        self.mask_layers = stacks.clone().into_mask_layers();
    }

    /// Inclusive source-frame bounds (`scene_start..=scene_end`) of a saved
    /// scene, for creating scene-scoped mask layers.
    ///
    /// The app stores half-open scene ranges, ck-core mask layers inclusive
    /// ones (v1 rule); `end_exclusive - 1` is the last active frame.
    /// `None` when the project has no scene with that index.
    pub fn scene_inclusive_range(&self, scene_index: u32) -> Option<(i64, i64)> {
        self.scenes
            .iter()
            .find(|scene| scene.index == scene_index)
            .map(|scene| scene_bounds_inclusive(scene.start, scene.end_exclusive))
    }

    /// Copy every mask layer of `source_scene` onto `target_scene`.
    ///
    /// Convenience wrapper over [`MaskStack::copy_layers_from_scene`] for the
    /// UI "copy mask layers to another scene" action: the target scene's saved
    /// half-open range is converted to the inclusive range mask layers store
    /// (v1 rule, `end_exclusive - 1`) and animation keys keep their scene-local
    /// offset. Returns the new layer ids (empty when the source scene has no
    /// layers).
    pub fn copy_mask_layers_from_scene(
        &mut self,
        kind: MaskKind,
        source_scene: u32,
        target_scene: u32,
    ) -> Result<Vec<u32>> {
        let bounds = self
            .scene_inclusive_range(target_scene)
            .or_else(|| {
                self.media
                    .as_ref()
                    .map(|media| (0, media.frame_count.saturating_sub(1) as i64))
            })
            .context("Mask layers cannot be copied to a scene the project does not contain")?;
        let mut stack = self.mask_stack(kind);
        let layers = stack.copy_layers_from_scene(source_scene, target_scene, bounds.0, bounds.1);
        if !layers.is_empty() {
            self.set_mask_stack(&stack);
        }
        Ok(layers)
    }
}

pub(crate) fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let temporary = parent.join(format!(".brp-write-{}-{nonce}.tmp", std::process::id()));
    let result = (|| -> Result<()> {
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path).context("Publishing saved project")?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rational;
    use ck_core::mask_layers::{BezierNode, BezierShape, InterpMode, Mask, MaskMode, Tangent};

    fn temp_folder(prefix: &str) -> PathBuf {
        let folder = std::env::temp_dir().join(format!(
            "{prefix}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&folder).unwrap();
        folder
    }

    fn media_fixture(folder: &Path) -> MediaInfo {
        MediaInfo {
            path: folder.join("source.mp4"),
            width: 64,
            height: 48,
            frame_count: 120,
            fps: Rational { num: 24, den: 1 },
            time_base: Rational { num: 1, den: 24 },
            first_pts: 0,
            has_b_frames: false,
            source_identity: None,
        }
    }

    fn two_scene_project(folder: &Path) -> Project {
        let mut project = Project::new(media_fixture(folder), folder.join("renders"));
        project.scenes = vec![
            SceneRange {
                index: 0,
                start: 0,
                end_exclusive: 50,
                enabled: true,
            },
            SceneRange {
                index: 1,
                start: 50,
                end_exclusive: 120,
                enabled: false,
            },
        ];
        project
    }

    fn shape_fixture() -> BezierShape {
        BezierShape::with_nodes(
            vec![
                BezierNode::new(4.0, 6.0),
                BezierNode::smooth(30.0, 40.0, 3.0, 0.0),
                BezierNode::new(50.0, 10.0),
            ],
            true,
        )
    }

    fn painted_mask(width: u32, height: u32, seed: u8) -> Mask {
        let mut mask = Mask::new(width, height);
        for (index, value) in mask.as_mut_slice().iter_mut().enumerate() {
            *value = if (index as u8).wrapping_add(seed).is_multiple_of(5) {
                255
            } else {
                seed
            };
        }
        mask
    }

    #[test]
    fn project_round_trip_preserves_half_open_scenes_and_relative_media() {
        let folder = temp_folder("brp-project-test");
        let media = media_fixture(&folder);
        let mut project = Project::new(media, folder.join("renders"));
        project.ck_settings = CkSettings {
            window_mode: crate::CkWindowMode::Custom,
            hann_chunk: 12,
            hann_stride: 5,
            low_vram: true,
        };
        project.despill_settings = DespillSettings { amount: 0.37 };
        project.scenes = vec![
            SceneRange {
                index: 0,
                start: 0,
                end_exclusive: 50,
                enabled: true,
            },
            SceneRange {
                index: 1,
                start: 50,
                end_exclusive: 119,
                enabled: false,
            },
        ];
        let path = folder.join("test.brp.json");
        project.save(&path).unwrap();
        project.save(&path).unwrap();
        let loaded = Project::load(&path).unwrap();
        assert_eq!(loaded.scenes, project.scenes);
        assert_eq!(loaded.ck_settings, project.ck_settings);
        assert_eq!(loaded.despill_settings, project.despill_settings);
        assert_eq!(loaded.schema_version, PROJECT_SCHEMA_VERSION);
        assert!(loaded.mask_layers.is_empty());
        assert_eq!(loaded.media.unwrap().path, folder.join("source.mp4"));
        fs::remove_file(path).unwrap();
        fs::remove_dir(folder).unwrap();
    }

    #[test]
    fn old_project_without_render_settings_remains_loadable() {
        let project: Project = serde_json::from_str(
            r#"{"schema_version":1,"media":null,"scenes":[],"prompts":[],"output_dir":"renders"}"#,
        )
        .unwrap();
        project.validate().unwrap();
        assert_eq!(project.ck_settings, CkSettings::default());
        assert_eq!(project.despill_settings, DespillSettings::default());
        assert!(project.mask_layers.is_empty());
    }

    #[test]
    fn result_despill_validates_serialized_range_without_changing_render_settings() {
        for amount in [f32::NAN, f32::INFINITY, -0.01, 1.01] {
            let project = Project {
                despill_settings: DespillSettings { amount },
                ..Project::default()
            };
            assert!(project.validate().is_err());
        }
        for amount in [0.0, 0.37, 1.0] {
            let original = Project {
                despill_settings: DespillSettings { amount },
                ..Project::default()
            };
            let restored: Project =
                serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
            restored.validate().unwrap();
            assert_eq!(restored.despill_settings, original.despill_settings);
            assert_eq!(restored.ck_settings, CkSettings::default());
        }
    }

    #[test]
    fn version_one_projects_without_mask_layers_load_with_empty_stacks() {
        let folder = temp_folder("brp-mask-v1-project");
        let path = folder.join("legacy.brp.json");
        fs::write(
            &path,
            r#"{
                "schema_version": 1,
                "media": {
                    "path": "source.mp4", "width": 64, "height": 48, "frame_count": 120,
                    "fps": {"num": 24, "den": 1}, "time_base": {"num": 1, "den": 24},
                    "first_pts": 0, "has_b_frames": false
                },
                "scenes": [{"index": 0, "start": 0, "end_exclusive": 120, "enabled": true}],
                "prompts": [],
                "output_dir": "renders"
            }"#,
        )
        .unwrap();
        let project = Project::load(&path).unwrap();
        assert_eq!(project.schema_version, PROJECT_SCHEMA_VERSION);
        assert!(project.mask_layers.is_empty());
        assert!(project.mask_stacks().is_empty());
        assert_eq!(
            project.media.as_ref().unwrap().path,
            folder.join("source.mp4")
        );
        let rewritten = folder.join("upgraded.brp.json");
        project.save(&rewritten).unwrap();
        let document: serde_json::Value =
            serde_json::from_slice(&fs::read(&rewritten).unwrap()).unwrap();
        assert_eq!(document["schema_version"], PROJECT_SCHEMA_VERSION);
        assert_eq!(
            document["mask_layers"]["garbage"].as_array().unwrap().len(),
            0
        );
        assert_eq!(
            document["mask_layers"]["despill"].as_array().unwrap().len(),
            0
        );
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn v1_shaped_mask_layers_load_with_inclusive_scenes_and_ignore_png_frames() {
        let folder = temp_folder("brp-mask-v1-layers");
        let path = folder.join("v1-layers.chroma.json");
        fs::write(
            &path,
            r##"{
                "schema_version": 1,
                "media": {
                    "path": "source.mp4", "width": 1920, "height": 1080, "frame_count": 120,
                    "fps": {"num": 24, "den": 1}, "time_base": {"num": 1, "den": 24},
                    "first_pts": 0, "has_b_frames": false
                },
                "scenes": [
                    {"index": 0, "start": 0, "end_exclusive": 50, "enabled": true},
                    {"index": 1, "start": 50, "end_exclusive": 120, "enabled": false}
                ],
                "prompts": [],
                "output_dir": "renders",
                "mask_layers": {
                    "garbage": [{
                        "id": 3, "name": "Garbage 03", "mode": "intersect",
                        "visible": true, "color": "#4a7dff", "softness": 4,
                        "scene": 1, "scene_start": 50, "scene_end": 119,
                        "interp": "linear", "auto_key": false,
                        "tangents": {"50": [0.25, 0.75]},
                        "shapes": {"50": {"closed": true, "nodes": [
                            {"x": 1.0, "y": 2.0, "out_dx": 3.0, "out_dy": 4.0,
                             "in_dx": -3.0, "in_dy": -4.0, "linked": true},
                            {"x": 5.0, "y": 6.0},
                            {"x": 7.0, "y": 8.0}
                        ]}},
                        "frames": ["masks/garbage_03_000050.png"]
                    }],
                    "despill": [{
                        "id": 1, "name": "Despill 01", "mode": "confine",
                        "visible": false, "color": "#ff0000", "softness": 0,
                        "scene": 0, "scene_start": 0, "scene_end": 49,
                        "interp": "hold", "auto_key": true,
                        "tangents": {}, "shapes": {}, "frames": []
                    }]
                }
            }"##,
        )
        .unwrap();
        let project = Project::load(&path).unwrap();
        project.validate().unwrap();
        assert_eq!(project.schema_version, PROJECT_SCHEMA_VERSION);
        let MaskStacks { garbage, despill } = project.mask_stacks();
        assert_eq!(garbage.layers.len(), 1);
        let layer = &garbage.layers[0];
        assert_eq!(layer.layer_id, 3);
        assert_eq!(layer.kind, MaskKind::Garbage);
        assert_eq!(layer.mode, MaskMode::Intersect);
        assert_eq!(layer.interp, InterpMode::Linear);
        assert!(!layer.auto_key);
        assert_eq!(layer.scene_index, 1);
        assert_eq!((layer.scene_start, layer.scene_end), (50, 119));
        assert!(layer.in_scene(50) && layer.in_scene(119) && !layer.in_scene(120));
        assert!(!layer.in_scene(49));
        assert_eq!(layer.tangent(50), Tangent::new(0.25, 0.75));
        let shape = layer.shapes.get(&50).expect("v1 vector key");
        assert_eq!(shape.nodes.len(), 3);
        assert!(shape.closed);
        assert_eq!(
            (
                shape.nodes[0].x,
                shape.nodes[0].out_dx,
                shape.nodes[0].in_dy
            ),
            (1.0, 3.0, -4.0)
        );
        assert!(
            layer.doc.frames.is_empty(),
            "v1 PNG frame lists are ignored"
        );
        assert_eq!(
            (layer.width, layer.height),
            (1920, 1080),
            "v1 layers adopt the source resolution"
        );
        assert_eq!(despill.layers.len(), 1);
        assert_eq!(despill.layers[0].kind, MaskKind::Despill);
        assert_eq!(despill.layers[0].mode, MaskMode::Confine);
        assert!(!despill.layers[0].visible);
        assert_eq!(despill.layers[0].scene_end, 49);
        // Re-saving the loaded project drops the v1 `frames` PNG-name list.
        let serialized = serde_json::to_value(&project).unwrap();
        assert!(
            serialized["mask_layers"]["garbage"][0]
                .get("frames")
                .is_none()
        );
        assert_eq!(serialized["mask_layers"]["garbage"][0]["id"], 3);
        assert_eq!(serialized["mask_layers"]["garbage"][0]["width"], 1920);
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn project_json_round_trip_keeps_mask_metadata_but_not_painted_pixels() {
        let folder = temp_folder("brp-mask-roundtrip");
        let mut project = two_scene_project(&folder);
        let (width, height) = project.mask_raster_size();
        let mut stacks = project.mask_stacks();
        let garbage = stacks.garbage.add(1, Some(MaskMode::Intersect));
        garbage.name = "Garbage 01".into();
        garbage.color = "#123456".into();
        garbage.softness = 3;
        garbage.set_scene_range(0, 49);
        garbage.set_shape(50, shape_fixture());
        garbage.set_tangent(50, 0.5);
        garbage.set_mask(50, &painted_mask(width, height, 7));
        stacks.despill.add(0, None).set_scene_range(0, 119);
        project.replace_mask_stacks(stacks);
        let path = folder.join("masked.brp.json");
        project.save(&path).unwrap();
        let document: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let layer = &document["mask_layers"]["garbage"][0];
        assert_eq!(document["schema_version"], PROJECT_SCHEMA_VERSION);
        assert_eq!(layer["id"], 1);
        assert_eq!(layer["kind"], "garbage");
        assert_eq!(layer["mode"], "intersect");
        assert_eq!(layer["softness"], 3);
        assert_eq!(layer["shapes"]["50"]["nodes"].as_array().unwrap().len(), 3);
        assert!(
            layer.get("frames").is_none(),
            "painted rasters live in .chroma blobs, not layer JSON"
        );
        let loaded = Project::load(&path).unwrap();
        let stacks = loaded.mask_stacks();
        assert_eq!(stacks.garbage.layers.len(), 1);
        let layer = &stacks.garbage.layers[0];
        assert_eq!(layer.layer_id, 1);
        assert_eq!(layer.color, "#123456");
        assert_eq!(layer.softness, 3);
        assert_eq!((layer.scene_start, layer.scene_end), (0, 49));
        assert_eq!(layer.tangent(50), Tangent::new(0.5, 0.5));
        assert_eq!(layer.shapes.get(&50), Some(&shape_fixture()));
        assert_eq!(
            layer.keyframes(),
            vec![50],
            "only the vector key is in JSON; the painted key is a blob"
        );
        assert!(layer.doc.frames.is_empty());
        assert_eq!((layer.width, layer.height), (width, height));
        assert_eq!(stacks.despill.layers.len(), 1);
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn mask_layers_copy_to_another_scene_shift_keys_and_deep_copy_rasters() {
        let folder = temp_folder("brp-mask-copy");
        let mut project = two_scene_project(&folder);
        let (width, height) = project.mask_raster_size();
        let painted = painted_mask(width, height, 11);
        let mut stacks = project.mask_stacks();
        let source = stacks.garbage.add(0, None);
        source.set_scene_range(0, 49);
        source.set_shape(10, shape_fixture());
        source.set_mask(10, &painted);
        source.set_tangent(10, 0.5);
        project.replace_mask_stacks(stacks);
        let ids = project
            .copy_mask_layers_from_scene(MaskKind::Garbage, 0, 1)
            .unwrap();
        assert_eq!(ids.len(), 1);
        let stacks = project.mask_stacks();
        assert_eq!(stacks.garbage.layers.len(), 2);
        let copied = stacks
            .garbage
            .layers
            .iter()
            .find(|layer| layer.layer_id == ids[0])
            .expect("copied layer");
        assert_eq!(copied.scene_index, 1);
        assert_eq!((copied.scene_start, copied.scene_end), (50, 119));
        assert!(copied.in_scene(50) && copied.in_scene(119) && !copied.in_scene(120));
        assert!(copied.has_shape(60));
        assert_eq!(
            copied.doc.get(60).map(Mask::as_slice),
            Some(painted.as_slice())
        );
        assert_eq!(copied.tangent(60), Tangent::new(0.5, 0.5));
        assert!(
            stacks
                .garbage
                .layers
                .iter()
                .any(|layer| layer.scene_start == 0 && layer.doc.has(10)),
            "the original scene keeps its layer and raster"
        );
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn half_open_scene_ranges_convert_to_inclusive_mask_layer_bounds() {
        assert_eq!(scene_bounds_inclusive(0, 1), (0, 0));
        assert_eq!(scene_bounds_inclusive(50, 120), (50, 119));
        let folder = temp_folder("brp-mask-scene-bounds");
        let project = two_scene_project(&folder);
        assert_eq!(project.scene_inclusive_range(1), Some((50, 119)));
        assert_eq!(project.scene_inclusive_range(7), None);
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn project_validation_rejects_invalid_mask_layer_state() {
        let folder = temp_folder("brp-mask-validate");
        let base = two_scene_project(&folder);

        let mut invalid_opacity = base.clone();
        let mut stacks = invalid_opacity.mask_stacks();
        stacks.garbage.add(0, None).opacity = 1.5;
        invalid_opacity.replace_mask_stacks(stacks);
        assert!(invalid_opacity.validate().is_err());

        let mut duplicate_ids = base.clone();
        let mut stacks = duplicate_ids.mask_stacks();
        let id = stacks.despill.add(0, None).layer_id;
        stacks.despill.add(0, None).layer_id = id;
        duplicate_ids.replace_mask_stacks(stacks);
        assert!(duplicate_ids.validate().is_err());

        let mut mismatched_raster = base.clone();
        let mut stacks = mismatched_raster.mask_stacks();
        let layer = stacks.garbage.add(0, None);
        layer.doc.frames.insert(0, Mask::new(3, 3));
        mismatched_raster.replace_mask_stacks(stacks);
        assert!(mismatched_raster.validate().is_err());

        // Files written by a newer app are refused instead of downgraded.
        let mut future = base.clone();
        future.schema_version = PROJECT_SCHEMA_VERSION + 1;
        assert!(future.validate().is_err());
        assert!(future.migrate().is_err());

        assert!(base.validate().is_ok());
        fs::remove_dir_all(folder).unwrap();
    }

    #[test]
    fn scene_despill_override_applies_only_inside_its_scene() {
        let mut project = two_scene_project(Path::new("C:/tmp/chroma-override"));
        project.despill_settings.amount = 0.2;
        project.scene_overrides.insert(
            1,
            SceneOverrides {
                despill: Some(SceneOverrideValue {
                    enabled: true,
                    amount: 0.7,
                }),
            },
        );
        assert_eq!(project.despill_settings_for_frame(5).amount, 0.2);
        assert_eq!(project.despill_settings_for_frame(60).amount, 0.7);
        project.validate().unwrap();
        project.scene_overrides.get_mut(&1).unwrap().despill = Some(SceneOverrideValue {
            enabled: false,
            amount: 0.7,
        });
        assert_eq!(project.despill_settings_for_frame(60).amount, 0.2);
        project.scene_overrides.get_mut(&1).unwrap().despill = Some(SceneOverrideValue {
            enabled: true,
            amount: 1.5,
        });
        assert!(project.validate().is_err());
    }

    #[test]
    fn scene_overrides_round_trip_the_v1_json_shape() {
        let json = r#"{"despill":{"enabled":true,"amount":0.4}}"#;
        let overrides: SceneOverrides = serde_json::from_str(json).unwrap();
        assert_eq!(overrides.despill_amount(0.2), 0.4);
        let encoded = serde_json::to_string(&overrides).unwrap();
        assert!(encoded.contains("\"despill\""), "{encoded}");
        assert!(encoded.contains("\"enabled\":true"), "{encoded}");
        let disabled: SceneOverrides =
            serde_json::from_str(r#"{"despill":{"enabled":false,"amount":1.0}}"#).unwrap();
        assert_eq!(disabled.despill_amount(0.2), 0.2);
    }

    #[test]
    fn sanitize_scene_overrides_drops_stale_scene_indexes() {
        let mut project = two_scene_project(Path::new("C:/tmp/chroma-stale"));
        project.scene_overrides.insert(1, SceneOverrides::default());
        project.scene_overrides.insert(9, SceneOverrides::default());
        project.sanitize_scene_overrides();
        assert!(project.scene_overrides.contains_key(&1));
        assert!(!project.scene_overrides.contains_key(&9));
    }
}
