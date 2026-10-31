//! Animated auxiliary mask layers (garbage / despill) and their stacks,
//! matching v1 `masklayers.py`.
//!
//! A [`MaskLayer`] animates a matte with integer source-frame keys. Each key can
//! hold a raster mask (painted) and/or a [`BezierShape`] (vector). Between keys
//! the layer interpolates, and the timing curve is one of [`InterpMode`]:
//!
//! * `Bezier` — cubic Hermite time remap with per-key auto tangents (the v1
//!   default: ease in/out at both ends, see [`bezier_ease`]),
//! * `Linear` — constant rate,
//! * `Hold` — step: keep the previous key until the next one.
//!
//! Frame and timing conventions (v1 parity):
//!
//! * Keyframe keys are **absolute source frames** (the same value as the
//!   playhead / `Project` frame index), not scene-relative offsets.
//! * A layer may be scoped to a scene: [`MaskLayer::in_scene`] returns true for
//!   `scene_start <= frame <= scene_end`, inclusive at both ends (v1 uses
//!   `<=`; the `.chroma` scene ranges are half-open, so the UI must pass
//!   `end - 1`). A layer with `scene_start < 0` or
//!   `scene_end < scene_start` is unrestricted.
//! * Tangent timing is stored per key as a [`Tangent`] pair `[out, in]` of
//!   **normalised slopes in 0..1**: `0` = flat/auto (ease at that end), `1` =
//!   linear/constant speed. They are dimensionless (fraction of the linear rate
//!   over that key interval), so they do not depend on the frame distance
//!   between keys. The time remap formula is in [`bezier_ease`]; v1's UI slider
//!   `0..100` maps to `0.0..1.0`.
//! * Raster dimensions live on the layer ([`MaskLayer::width`] /
//!   [`MaskLayer::height`]; normally the project/source resolution) and are
//!   explicit in the API. All masks returned are 8-bit row-major (see
//!   [`crate::mask_layers::raster`]).
//!
//! `opacity` is an **app-schema extension**, not present in v1: `1.0` (the
//! default) reproduces v1 byte-for-byte; lower values scale the layer's mask
//! before compositing. `kind` is also not stored inside v1 layer objects (v1
//! groups layer lists under `garbage`/`despill` keys); this port keeps it on the
//! layer for a flat, strongly typed schema. Raster key masks are runtime state
//! (`MaskLayer::doc`) and are not part of JSON — the project layer stores them
//! as blobs, like v1 stored PNGs.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::mask_layers::raster::{self, Mask};
use crate::mask_layers::spline::BezierShape;

/// Tangent slope meaning "auto / ease" (v1 `AUTO_TANGENT = 0.0`).
pub const AUTO_TANGENT: f32 = 0.0;

/// Garbage mask composition modes, in v1 panel order.
pub const GARBAGE_MODES: [MaskMode; 3] = [MaskMode::Add, MaskMode::Subtract, MaskMode::Intersect];
/// Despill mask restriction modes, in v1 panel order.
pub const DESPILL_MODES: [MaskMode; 2] = [MaskMode::Protect, MaskMode::Confine];
/// Keyframe interpolation modes, in v1 panel order.
pub const INTERP_MODES: [InterpMode; 3] =
    [InterpMode::Bezier, InterpMode::Linear, InterpMode::Hold];

/// Which auxiliary matte a layer belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaskKind {
    /// Matte cleanup applied to the subject alpha.
    Garbage,
    /// Despill weighting / protection region.
    Despill,
}

impl MaskKind {
    /// v1 display noun used for default layer names (`"Garbage 01"`).
    pub fn noun(self) -> &'static str {
        match self {
            Self::Garbage => "Garbage",
            Self::Despill => "Despill",
        }
    }

    /// v1 default mode for a new layer of this kind.
    pub fn default_mode(self) -> MaskMode {
        match self {
            Self::Garbage => MaskMode::Subtract,
            Self::Despill => MaskMode::Protect,
        }
    }

    /// Modes that are meaningful for this kind (v1 mode lists).
    pub fn modes(self) -> &'static [MaskMode] {
        match self {
            Self::Garbage => &GARBAGE_MODES,
            Self::Despill => &DESPILL_MODES,
        }
    }
}

/// Composition mode of one layer (v1 `GARBAGE_MODES` / `DESPILL_MODES`).
///
/// The on-disk strings are exactly v1's: `"add"`, `"subtract"`, `"intersect"`,
/// `"protect"`, `"confine"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MaskMode {
    /// `out = max(out, mask)`.
    Add,
    /// `out = min(out, 255 - mask)`.
    Subtract,
    /// `out = min(out, mask)`.
    Intersect,
    /// Despill only: no despill where the mask is set.
    Protect,
    /// Despill only: despill only where the mask is set.
    Confine,
}

impl MaskMode {
    /// v1 panel label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Subtract => "Subtract",
            Self::Intersect => "Intersect",
            Self::Protect => "Protect (no despill)",
            Self::Confine => "Only here",
        }
    }

    /// Whether this mode belongs to the given kind (v1 mode lists).
    pub fn is_valid_for(self, kind: MaskKind) -> bool {
        kind.modes().contains(&self)
    }

    /// Default mode for a kind (v1 `"subtract"` / `"protect"`).
    pub fn default_for(kind: MaskKind) -> Self {
        kind.default_mode()
    }
}

/// Keyframe interpolation (time-remap) mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InterpMode {
    /// Cubic Hermite with per-key tangents (v1 default, auto = ease in/out).
    Bezier,
    /// Constant rate between keys.
    Linear,
    /// Step: hold the previous key.
    Hold,
}

impl InterpMode {
    /// v1 panel label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Bezier => "Bezier (auto tangents)",
            Self::Linear => "Linear",
            Self::Hold => "Hold (step)",
        }
    }
}

/// Two normalised Hermite tangent slopes for one keyframe, as v1 stores them
/// in the project JSON: `[out_slope, in_slope]`.
///
/// Both values are in `0..=1`: `0` is auto/flat (ease at that end), `1` is
/// linear/constant speed. They are dimensionless — the "in" slope of key `b`
/// applies to the interval that ends at `b`, the "out" slope of key `a` to the
/// interval that starts at `a`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Tangent(pub f32, pub f32);

impl Tangent {
    /// Auto tangents (flat ease in/out at both ends) — the v1 default.
    pub const AUTO: Tangent = Tangent(0.0, 0.0);
    /// Linear tangents (constant speed over the adjacent intervals).
    pub const LINEAR: Tangent = Tangent(1.0, 1.0);

    /// Build a tangent pair from normalised slopes.
    pub fn new(out_slope: f32, in_slope: f32) -> Self {
        Self(out_slope, in_slope)
    }

    /// Normalised slope at the start of the interval leaving this key.
    pub fn out_slope(self) -> f32 {
        self.0
    }

    /// Normalised slope at the end of the interval entering this key.
    pub fn in_slope(self) -> f32 {
        self.1
    }
}

/// Cubic Hermite (== cubic bezier) time remap between two keys.
///
/// `t` is the normalised position in the interval (`(frame - a) / (b - a)`,
/// clamped to `0..1`). `out_slope` is the start tangent and `in_slope` the end
/// tangent, both normalised to `0..1` (clamped): `0` is flat (ease in/out at
/// that end), `1` is constant speed (linear end). The result is a normalised
/// interpolation amount, monotonic for slopes in range:
///
/// ```text
/// t2 = t*t;  t3 = t2*t
/// h10 = t3 - 2*t2 + t         // start tangent basis
/// h01 = -2*t3 + 3*t2          // smoothstep basis
/// h11 = t3 - t2               // end tangent basis
/// ease = h10*out_slope + h01 + h11*in_slope
/// ```
///
/// With both slopes auto (0) this is the smoothstep curve; with both at 1 it is
/// exactly `t` (`h10 + h01 + h11 == t`). A `hold` interpolation is handled by
/// [`MaskLayer::ease`] returning `0.0` before this function is called.
pub fn bezier_ease(t: f32, out_slope: f32, in_slope: f32) -> f32 {
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let m0 = if out_slope.is_finite() {
        out_slope.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let m1 = if in_slope.is_finite() {
        in_slope.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let t2 = t * t;
    let t3 = t2 * t;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h10 * m0 + h01 + h11 * m1
}

/// A key interval around a frame: the keys before/after and the normalised
/// position `position` in `0..1` (v1 returned the same triple as a tuple).
///
/// At the ends of the animation both keys are the same and `position` is `0.0`,
/// which makes [`MaskLayer::ease`] return `0.0` (hold the first/last key).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeySegment {
    /// Key at or before the frame.
    pub before: i64,
    /// Key after the frame (equal to `before` at the ends).
    pub after: i64,
    /// Normalised position between the keys, `0..1`.
    pub position: f32,
}

/// Per-frame raster mask store used for painted keys (v1 `MaskDocument`).
///
/// Masks are full-resolution 8-bit rasters (row-major). This type only owns the
/// pixels and their frame keys; undo/redo and brush painting are UI concerns in
/// the native application.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaskDocument {
    /// Raster width in pixels.
    pub width: u32,
    /// Raster height in pixels.
    pub height: u32,
    /// Painted key masks by absolute source frame.
    pub frames: BTreeMap<i64, Mask>,
}

impl MaskDocument {
    /// Create an empty document for a raster size.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            frames: BTreeMap::new(),
        }
    }

    /// An all-zero mask at the document size.
    pub fn blank(&self) -> Mask {
        Mask::new(self.width, self.height)
    }

    /// Whether the frame holds a painted key.
    pub fn has(&self, frame: i64) -> bool {
        self.frames.contains_key(&frame)
    }

    /// Painted key mask, if any.
    pub fn get(&self, frame: i64) -> Option<&Mask> {
        self.frames.get(&frame)
    }

    /// Mutable painted key mask, if any.
    pub fn get_mut(&mut self, frame: i64) -> Option<&mut Mask> {
        self.frames.get_mut(&frame)
    }

    /// Sorted painted key frames (v1 `keyframes`).
    pub fn keyframes(&self) -> Vec<i64> {
        self.frames.keys().copied().collect()
    }

    /// Store a mask at a frame, resampling to the document size and optionally
    /// taking the maximum with an existing mask (v1 `set_mask(..., add)`).
    pub fn set_mask(&mut self, frame: i64, mask: &Mask, add: bool) {
        let mut resized = mask.resampled(self.width, self.height);
        if add {
            if let Some(existing) = self.frames.get(&frame) {
                for (slot, value) in resized.as_mut_slice().iter_mut().zip(existing.as_slice()) {
                    *slot = (*slot).max(*value);
                }
            }
        }
        self.frames.insert(frame, resized);
    }

    /// Remove a painted key.
    pub fn remove(&mut self, frame: i64) -> Option<Mask> {
        self.frames.remove(&frame)
    }

    /// Remove every painted key.
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    /// Fraction of opaque pixels (`> 127`) for a frame, 0 when absent (v1
    /// `coverage`).
    pub fn coverage(&self, frame: i64) -> f32 {
        self.frames.get(&frame).map_or(0.0, Mask::coverage)
    }

    /// Copy the mask at `index` to every frame in `start..=end`, excluding the
    /// source frame (v1 `copy_to_range`; used by "Copy current mask across
    /// scene").
    pub fn copy_to_range(&mut self, index: i64, start: i64, end: i64) -> bool {
        let Some(source) = self.frames.get(&index).cloned() else {
            return false;
        };
        let mut copied = false;
        for frame in start..=end {
            if frame != index {
                self.frames.insert(frame, source.clone());
                copied = true;
            }
        }
        copied
    }
}

/// One animated garbage or despill mask layer.
///
/// JSON shape (v1-compatible layer object; `id` and `scene` keep v1's field
/// names):
///
/// ```json
/// {
///   "id": 1, "name": "Garbage 01", "kind": "garbage", "mode": "subtract",
///   "visible": true, "opacity": 1.0, "color": "#4a7dff", "softness": 0,
///   "scene": 0, "scene_start": 0, "scene_end": 23,
///   "interp": "bezier", "auto_key": true,
///   "tangents": { "0": [0.0, 0.0] },
///   "shapes": { "0": { "closed": true, "nodes": [ {"x":..,"y":..,"out_dx":..,
///       "out_dy":..,"in_dx":..,"in_dy":..,"linked":true} ] } },
///   "width": 1920, "height": 1080
/// }
/// ```
///
/// `kind`, `opacity`, `width` and `height` are app-schema extensions; every
/// field is optional on load (`serde` fills defaults), so v1 layer JSON parses
/// directly. v1's `"frames": ["masks/...png"]` array is ignored here: raster
/// keys are runtime state in [`MaskLayer::doc`] and the project layer owns blob
/// persistence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaskLayer {
    /// Stable layer id within its stack (v1 `id`).
    #[serde(rename = "id")]
    pub layer_id: u32,
    /// Display name (v1 default `"Garbage NN"` / `"Despill NN"`).
    pub name: String,
    /// Garbage or despill layer.
    pub kind: MaskKind,
    /// Composition mode; must be valid for `kind` for v1 parity.
    pub mode: MaskMode,
    /// Whether the layer is applied/rendered.
    pub visible: bool,
    /// Layer mask opacity multiplier, `0..=1` (app extension; v1 has no
    /// opacity and always used `1.0`).
    pub opacity: f32,
    /// Overlay colour (`#rrggbb`) for the viewer; not a render input.
    pub color: String,
    /// Feather radius in raster pixels (v1 `softness`).
    pub softness: u32,
    /// Scene this layer belongs to (`scene_index` in v1 code, `scene` in JSON).
    #[serde(rename = "scene")]
    pub scene_index: u32,
    /// First frame the layer is active on; `-1` = unrestricted (v1 sentinel).
    pub scene_start: i64,
    /// Last frame the layer is active on (inclusive); `< scene_start` =
    /// unrestricted (v1 sentinel).
    pub scene_end: i64,
    /// Keyframe time-remap mode.
    pub interp: InterpMode,
    /// Whether painting on a non-key frame seeds a new key.
    pub auto_key: bool,
    /// Per-key tangent slopes (normalised, see [`Tangent`]).
    pub tangents: BTreeMap<i64, Tangent>,
    /// Per-key vector shapes.
    pub shapes: BTreeMap<i64, BezierShape>,
    /// Raster width in pixels for this layer's masks (app extension).
    pub width: u32,
    /// Raster height in pixels for this layer's masks (app extension).
    pub height: u32,
    /// Painted raster keys; runtime state, not serialised (project blobs own it).
    #[serde(skip)]
    pub doc: MaskDocument,
}

impl Default for MaskLayer {
    fn default() -> Self {
        Self {
            layer_id: 0,
            name: String::new(),
            kind: MaskKind::Garbage,
            mode: MaskMode::Subtract,
            visible: true,
            opacity: 1.0,
            color: String::new(),
            softness: 0,
            scene_index: 0,
            scene_start: -1,
            scene_end: -1,
            interp: InterpMode::Bezier,
            auto_key: true,
            tangents: BTreeMap::new(),
            shapes: BTreeMap::new(),
            width: 0,
            height: 0,
            doc: MaskDocument::default(),
        }
    }
}

impl MaskLayer {
    /// Create an empty layer with the default mode for its kind.
    pub fn new(
        layer_id: u32,
        name: impl Into<String>,
        kind: MaskKind,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            layer_id,
            name: name.into(),
            kind,
            mode: kind.default_mode(),
            color: suggest_color(layer_id),
            width,
            height,
            doc: MaskDocument::new(width, height),
            ..Self::default()
        }
    }

    /// Set the raster size, resampling any painted keys (see
    /// [`raster::Mask::resampled`]).
    pub fn set_raster_size(&mut self, width: u32, height: u32) {
        if self.width == width
            && self.height == height
            && self.doc.width == width
            && self.doc.height == height
        {
            return;
        }
        for mask in self.doc.frames.values_mut() {
            *mask = mask.resampled(width, height);
        }
        self.width = width;
        self.height = height;
        self.doc.width = width;
        self.doc.height = height;
    }

    // -- keyframes -----------------------------------------------------------

    /// All key frames (painted and vector), sorted (v1 `keyframes`).
    pub fn keyframes(&self) -> Vec<i64> {
        let mut frames: BTreeSet<i64> = self.shapes.keys().copied().collect();
        frames.extend(self.doc.frames.keys().copied());
        frames.into_iter().collect()
    }

    /// Whether a painted or vector key exists on `frame`.
    pub fn key_at(&self, frame: i64) -> bool {
        self.doc.has(frame) || self.shapes.contains_key(&frame)
    }

    // -- scene scope ---------------------------------------------------------

    /// Restrict the layer to the inclusive source-frame range `start..=end`.
    pub fn set_scene_range(&mut self, start: i64, end: i64) {
        self.scene_start = start;
        self.scene_end = end;
    }

    /// Remove the scene restriction (`scene_start = -1`).
    pub fn clear_scene_range(&mut self) {
        self.scene_start = -1;
        self.scene_end = -1;
    }

    /// Whether the layer applies on `frame`.
    ///
    /// v1 rule: an unrestricted layer (`scene_start < 0` or
    /// `scene_end < scene_start`) applies everywhere; otherwise the range is
    /// **inclusive** at both ends.
    pub fn in_scene(&self, frame: i64) -> bool {
        if self.scene_start < 0 || self.scene_end < self.scene_start {
            return true;
        }
        self.scene_start <= frame && frame <= self.scene_end
    }

    // -- vector (bezier) keys ------------------------------------------------

    /// Sorted vector key frames.
    pub fn shape_keys(&self) -> Vec<i64> {
        self.shapes.keys().copied().collect()
    }

    /// Whether a vector key exists on `frame`.
    pub fn has_shape(&self, frame: i64) -> bool {
        self.shapes.contains_key(&frame)
    }

    /// Key interval around `frame` using only vector keys (v1
    /// `shape_segment`).
    pub fn shape_segment(&self, frame: i64) -> Option<KeySegment> {
        segment_for_keys(&self.shape_keys(), frame)
    }

    /// Vector shape at `frame`, interpolated between vector keys with the
    /// layer's time-remap mode (v1 `shape_at`).
    pub fn shape_at(&self, frame: i64) -> Option<BezierShape> {
        if self.shapes.is_empty() || !self.in_scene(frame) {
            return None;
        }
        if let Some(exact) = self.shapes.get(&frame) {
            return Some(exact.clone());
        }
        let segment = self.shape_segment(frame)?;
        if segment.before == segment.after {
            return self.shapes.get(&segment.before).cloned();
        }
        let first = self.shapes.get(&segment.before);
        let second = self.shapes.get(&segment.after);
        match (first, second) {
            (Some(first), Some(second)) => Some(first.interpolate(
                second,
                self.ease(segment.before, segment.after, segment.position),
            )),
            (Some(first), None) => Some(first.clone()),
            (None, Some(second)) => Some(second.clone()),
            (None, None) => None,
        }
    }

    /// Rasterised vector shape at `frame` (no softness applied), or `None` when
    /// there is no usable vector shape (v1 `raster_shape`).
    pub fn raster_shape(&self, frame: i64) -> Option<Mask> {
        if self.shapes.is_empty() {
            return None;
        }
        let shape = self.shape_at(frame)?;
        if shape.len() < 2 {
            return None;
        }
        Some(shape.rasterize(self.width, self.height, 0))
    }

    /// Store the vector key at `frame`.
    pub fn set_shape(&mut self, frame: i64, shape: BezierShape) {
        self.shapes.insert(frame, shape);
    }

    /// Remove the vector key at `frame`.
    pub fn remove_shape(&mut self, frame: i64) -> Option<BezierShape> {
        self.shapes.remove(&frame)
    }

    /// Vector key frame that an edit on `frame` should land on (v1
    /// `shape_edit_frame`): the exact key, the playhead when `auto_key` is on,
    /// otherwise the previous vector key.
    pub fn shape_edit_frame(&self, frame: i64) -> i64 {
        if self.shapes.contains_key(&frame) {
            return frame;
        }
        if self.auto_key {
            return frame;
        }
        self.shape_keys()
            .into_iter()
            .rfind(|key| *key <= frame)
            .unwrap_or(frame)
    }

    // -- keyframes and timing ------------------------------------------------

    /// Key interval around `frame` using all keys (v1 `segment`).
    pub fn segment(&self, frame: i64) -> Option<KeySegment> {
        segment_for_keys(&self.keyframes(), frame)
    }

    /// Tangent pair of the key at `frame`; missing keys use auto tangents.
    pub fn tangent(&self, frame: i64) -> Tangent {
        self.tangents.get(&frame).copied().unwrap_or(Tangent::AUTO)
    }

    /// v1 panel amount for the key at `frame`: `0` = auto/smooth, `100` =
    /// linear.
    pub fn tangent_amount(&self, frame: i64) -> u32 {
        let out = self.tangent(frame).out_slope().clamp(0.0, 1.0);
        (out * 100.0).round() as u32
    }

    /// Set both tangents of the key at `frame` from a v1 panel amount
    /// (0..1 clamped; `1.0` = linear).
    pub fn set_tangent(&mut self, frame: i64, amount: f32) {
        let amount = if amount.is_finite() {
            amount.clamp(0.0, 1.0)
        } else {
            AUTO_TANGENT
        };
        self.tangents.insert(frame, Tangent::new(amount, amount));
    }

    /// Set the tangent pair of the key at `frame` directly.
    pub fn set_tangent_pair(&mut self, frame: i64, tangent: Tangent) {
        self.tangents.insert(frame, tangent);
    }

    /// Remove the explicit tangent of the key at `frame` (back to auto).
    pub fn remove_tangent(&mut self, frame: i64) -> Option<Tangent> {
        self.tangents.remove(&frame)
    }

    /// Reset every key to auto tangents (v1 `auto_tangents`, the "Auto" button).
    pub fn auto_tangents(&mut self) {
        self.tangents.clear();
    }

    /// Normalised interpolation amount between keys `a` and `b` at position
    /// `t` (v1 `ease`).
    ///
    /// `Hold` returns `0.0` (previous key), `Linear` returns `t`, `Bezier` uses
    /// [`bezier_ease`] with `a`'s out slope and `b`'s in slope. Identical keys
    /// return `t` unchanged.
    pub fn ease(&self, a: i64, b: i64, t: f32) -> f32 {
        if a == b {
            return t;
        }
        match self.interp {
            InterpMode::Hold => 0.0,
            InterpMode::Linear => t,
            InterpMode::Bezier => {
                bezier_ease(t, self.tangent(a).out_slope(), self.tangent(b).in_slope())
            }
        }
    }

    // -- mask evaluation -----------------------------------------------------

    /// Painted-mask animation at `frame` (v1 `_painted_raw`): the exact painted
    /// key, a blend of the surrounding keys eased by [`Self::ease`], or the
    /// clamped first/last key. `None` outside the scene or without raster keys.
    pub fn painted_mask(&self, frame: i64) -> Option<Mask> {
        if !self.in_scene(frame) {
            return None;
        }
        if self.keyframes().is_empty() {
            return None;
        }
        if let Some(exact) = self.doc.get(frame) {
            return Some(exact.resampled(self.width, self.height));
        }
        let segment = self.segment(frame)?;
        let first = self.doc.get(segment.before);
        if segment.before == segment.after {
            return first.map(|mask| mask.resampled(self.width, self.height));
        }
        let first = first?;
        let Some(second) = self.doc.get(segment.after) else {
            return Some(first.resampled(self.width, self.height));
        };
        let eased = self.ease(segment.before, segment.after, segment.position);
        if eased <= 0.0 {
            return Some(first.resampled(self.width, self.height));
        }
        if eased >= 1.0 {
            return Some(second.resampled(self.width, self.height));
        }
        Some(raster::blend_masks(first, second, eased))
    }

    /// Unfeathered mask at `frame`: the union (max) of the animated vector shape
    /// and the painted keys, or either one when only one exists (v1 `raw`).
    pub fn raw_mask(&self, frame: i64) -> Option<Mask> {
        if self.shapes.is_empty() {
            return self.painted_mask(frame);
        }
        let vector = self.raster_shape(frame);
        let painted = self.painted_mask(frame);
        match (vector, painted) {
            (None, painted) => painted,
            (vector, None) => vector,
            (Some(vector), Some(painted)) => {
                let painted = painted.resampled(vector.width(), vector.height());
                let mut out = vector;
                for (slot, value) in out.as_mut_slice().iter_mut().zip(painted.as_slice()) {
                    *slot = (*slot).max(*value);
                }
                Some(out)
            }
        }
    }

    /// Feathered mask at `frame` (v1 `mask`): [`Self::raw_mask`] softened by
    /// `softness` pixels. `None` means the layer does not affect this frame.
    pub fn mask(&self, frame: i64) -> Option<Mask> {
        self.raw_mask(frame)
            .map(|mask| raster::soften(&mask, self.softness))
    }

    /// Store a painted raster key, resampled to the layer size (v1
    /// `set_mask`).
    pub fn set_mask(&mut self, frame: i64, mask: &Mask) {
        let resized = mask.resampled(self.width, self.height);
        self.doc.width = self.width;
        self.doc.height = self.height;
        self.doc.frames.insert(frame, resized);
    }

    /// Paint a key on `frame`, seeded from the current animation, or remove the
    /// key when it already exists (v1 `toggle_key`). Returns whether a key is
    /// present afterwards.
    pub fn toggle_key(&mut self, frame: i64) -> bool {
        if !self.shapes.is_empty() {
            if self.shapes.remove(&frame).is_some() {
                return false;
            }
            let shape = self.shape_at(frame).unwrap_or_default();
            self.shapes.insert(frame, shape);
            return true;
        }
        if self.doc.has(frame) {
            self.doc.remove(frame);
            return false;
        }
        let seed = self
            .raw_mask(frame)
            .unwrap_or_else(|| Mask::new(self.width, self.height));
        self.doc.frames.insert(frame, seed);
        true
    }

    /// Frame whose key a paint stroke should land on (v1 `edit_frame`).
    ///
    /// * an existing painted key is reused,
    /// * with `auto_key` the animation is evaluated and seeded as a new key on
    ///   `frame`,
    /// * without `auto_key` the previous key is edited; when none exists a blank
    ///   key is created on `frame`.
    pub fn edit_frame(&mut self, frame: i64) -> i64 {
        if self.doc.has(frame) {
            return frame;
        }
        if self.auto_key {
            let seed = self
                .raw_mask(frame)
                .unwrap_or_else(|| Mask::new(self.width, self.height));
            self.doc.width = self.width;
            self.doc.height = self.height;
            self.doc.frames.insert(frame, seed);
            return frame;
        }
        let previous = self.keyframes().into_iter().rfind(|key| *key <= frame);
        if let Some(previous) = previous {
            return previous;
        }
        self.doc.width = self.width;
        self.doc.height = self.height;
        self.doc
            .frames
            .insert(frame, Mask::new(self.width, self.height));
        frame
    }

    /// Remove every key (painted, vector and tangent) on `frame` (v1
    /// `clear_frame`).
    pub fn clear_frame(&mut self, frame: i64) {
        self.doc.remove(frame);
        self.shapes.remove(&frame);
        self.tangents.remove(&frame);
    }

    // -- copying -------------------------------------------------------------

    /// Deep-copy this layer for another scene under a new id.
    ///
    /// * layer id/kind/name/colour/softness/interp/auto_key/opacity and the
    ///   raster size are preserved,
    /// * the scene index and range become `target_scene`,
    ///   `target_start..=target_end`,
    /// * painted, vector and tangent keys are shifted by
    ///   `target_start - scene_start` when both the source range and the target
    ///   range are valid, so animation stays at the same offset inside the
    ///   scene. Keys are not clipped: keys that land outside the target range
    ///   are ignored by [`Self::in_scene`] but kept (lossless copy).
    pub fn copy_for_scene(
        &self,
        new_layer_id: u32,
        target_scene: u32,
        target_start: i64,
        target_end: i64,
    ) -> MaskLayer {
        let shift = if self.scene_start >= 0 && self.scene_end >= self.scene_start {
            target_start - self.scene_start
        } else {
            0
        };
        let remap = |frame: i64| frame + shift;
        let mut copy = self.clone();
        copy.layer_id = new_layer_id;
        copy.scene_index = target_scene;
        copy.scene_start = target_start;
        copy.scene_end = target_end;
        copy.doc.frames = self
            .doc
            .frames
            .iter()
            .map(|(frame, mask)| (remap(*frame), mask.clone()))
            .collect();
        copy.shapes = self
            .shapes
            .iter()
            .map(|(frame, shape)| (remap(*frame), shape.clone()))
            .collect();
        copy.tangents = self
            .tangents
            .iter()
            .map(|(frame, tangent)| (remap(*frame), *tangent))
            .collect();
        copy
    }
}

/// Ordered list of layers of one kind (v1 `MaskStack`).
///
/// Layer order is the compositing order (`layers[0]` first). The stack owns the
/// raster dimensions used by its layers. `selected` is UI state for the native
/// viewer: v1 remembered the active layer per `(kind, scene)` in the window, so
/// the app is expected to save/restore its own per-scene choice and call
/// [`MaskStack::select`] on scene switches; loading a stack leaves no selection.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskStack {
    /// Raster width in pixels shared by the layers.
    pub width: u32,
    /// Raster height in pixels shared by the layers.
    pub height: u32,
    /// Layer kind of this stack.
    pub kind: MaskKind,
    /// Layers in compositing order.
    pub layers: Vec<MaskLayer>,
    /// Currently selected layer id (UI state, not layer data).
    pub selected: Option<u32>,
    next_id: u32,
}

impl MaskStack {
    /// Create an empty stack for a raster size and kind.
    pub fn new(width: u32, height: u32, kind: MaskKind) -> Self {
        Self {
            width,
            height,
            kind,
            layers: Vec::new(),
            selected: None,
            next_id: 1,
        }
    }

    /// Rebuild a stack from loaded layers.
    ///
    /// The stack kind is authoritative (v1 stores layers under `garbage` /
    /// `despill` keys, not inside the layer), so every layer is assigned
    /// `kind`; a mode that is not valid for the kind falls back to the kind's
    /// default. Each layer's raster size is synchronised to the stack size.
    pub fn from_layers(
        width: u32,
        height: u32,
        kind: MaskKind,
        mut layers: Vec<MaskLayer>,
    ) -> Self {
        for layer in &mut layers {
            layer.kind = kind;
            if !layer.mode.is_valid_for(kind) {
                layer.mode = kind.default_mode();
            }
            layer.set_raster_size(width, height);
        }
        let mut stack = Self::new(width, height, kind);
        stack.layers = layers;
        stack.sync_next_id();
        stack
    }

    /// Next layer id this stack will hand out.
    pub fn next_id(&self) -> u32 {
        self.next_id
    }

    /// Recompute the id counter from the current layers (call after loading).
    pub fn sync_next_id(&mut self) {
        let max = self
            .layers
            .iter()
            .map(|layer| layer.layer_id)
            .max()
            .unwrap_or(0);
        self.next_id = max.saturating_add(1).max(1);
    }

    // -- list management -----------------------------------------------------

    /// Append a new layer for `scene_index` and select it.
    ///
    /// The name follows v1 (`"Garbage 01"`), the colour is deterministic per id
    /// and `mode` defaults to the kind's v1 default.
    pub fn add(&mut self, scene_index: u32, mode: Option<MaskMode>) -> &mut MaskLayer {
        let layer_id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let name = format!("{} {:02}", self.kind.noun(), layer_id);
        let mut layer = MaskLayer::new(layer_id, name, self.kind, self.width, self.height);
        layer.mode = mode.unwrap_or(self.kind.default_mode());
        layer.scene_index = scene_index;
        self.layers.push(layer);
        self.selected = Some(layer_id);
        self.layers
            .last_mut()
            .expect("a layer was just pushed onto the stack")
    }

    /// Paste a deep copy of `source` into this stack for another scene.
    ///
    /// The copy gets a fresh layer id, is appended in compositing order, becomes
    /// the selected layer and is returned so the caller can remember it.
    pub fn add_copy(
        &mut self,
        source: &MaskLayer,
        scene_index: u32,
        scene_start: i64,
        scene_end: i64,
    ) -> u32 {
        let layer_id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let mut copy = source.copy_for_scene(layer_id, scene_index, scene_start, scene_end);
        copy.name = format!("{} copy", source.name);
        self.layers.push(copy);
        self.selected = Some(layer_id);
        layer_id
    }

    /// Remove a layer by id. Removing the selected layer clears the selection.
    pub fn remove(&mut self, layer_id: u32) -> bool {
        let before = self.layers.len();
        self.layers.retain(|layer| layer.layer_id != layer_id);
        let removed = self.layers.len() != before;
        if removed && self.selected == Some(layer_id) {
            self.selected = None;
        }
        removed
    }

    /// Layer by id.
    pub fn by_id(&self, layer_id: u32) -> Option<&MaskLayer> {
        self.layers.iter().find(|layer| layer.layer_id == layer_id)
    }

    /// Mutable layer by id.
    pub fn by_id_mut(&mut self, layer_id: u32) -> Option<&mut MaskLayer> {
        self.layers
            .iter_mut()
            .find(|layer| layer.layer_id == layer_id)
    }

    /// Stack index of a layer id.
    pub fn index_of(&self, layer_id: u32) -> Option<usize> {
        self.layers
            .iter()
            .position(|layer| layer.layer_id == layer_id)
    }

    /// Whether a layer id exists in this stack.
    pub fn contains(&self, layer_id: u32) -> bool {
        self.by_id(layer_id).is_some()
    }

    /// Move a layer from stack index `source` to stack index `target`
    /// (v1 `move`). Returns false for out-of-range indices.
    pub fn move_layer(&mut self, source: usize, target: usize) -> bool {
        if source >= self.layers.len() || target >= self.layers.len() {
            return false;
        }
        let layer = self.layers.remove(source);
        self.layers.insert(target, layer);
        true
    }

    /// Reorder layers by id (v1 `reorder`).
    ///
    /// Unknown duplicates are ignored (the first occurrence wins); layers whose
    /// ids are not listed keep their relative order at the end.
    pub fn reorder(&mut self, ids: &[u32]) {
        let mut remaining = std::mem::take(&mut self.layers);
        let mut ordered = Vec::with_capacity(remaining.len());
        for id in ids {
            if let Some(index) = remaining.iter().position(|layer| layer.layer_id == *id) {
                ordered.push(remaining.remove(index));
            }
        }
        ordered.extend(remaining);
        self.layers = ordered;
    }

    /// Layers belonging to a scene, in stack order (v1 `for_scene`).
    pub fn for_scene(&self, scene_index: u32) -> Vec<&MaskLayer> {
        self.layers
            .iter()
            .filter(|layer| layer.scene_index == scene_index)
            .collect()
    }

    /// Whether any visible layer has keys (v1 `any_visible`).
    pub fn any_visible(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| layer.visible && !layer.keyframes().is_empty())
    }

    /// Select a layer id, or clear the selection with `None`. Returns false when
    /// a non-existent id was requested.
    pub fn select(&mut self, layer_id: Option<u32>) -> bool {
        match layer_id {
            Some(id) if self.contains(id) => {
                self.selected = Some(id);
                true
            }
            Some(_) => false,
            None => {
                self.selected = None;
                true
            }
        }
    }

    /// Selected layer, if any.
    pub fn selected_layer(&self) -> Option<&MaskLayer> {
        self.selected.and_then(|id| self.by_id(id))
    }

    /// Selected layer mutably, if any.
    pub fn selected_layer_mut(&mut self) -> Option<&mut MaskLayer> {
        self.selected.and_then(|id| self.by_id_mut(id))
    }

    // -- application ---------------------------------------------------------

    /// Composite the visible garbage layers onto a subject mask (8-bit).
    ///
    /// * `base = None` starts from an all-zero matte.
    /// * `add` → `max`, `intersect` → `min`, every other mode (including the
    ///   despill-only modes) → `min(out, 255 - mask)`, matching v1's mode
    ///   dispatch.
    /// * Layers outside their scene range or without keys are skipped.
    /// * `opacity < 1.0` scales the layer mask first (app extension).
    ///
    /// Returns `None` only when the base is `None` and no layer contributed.
    pub fn apply(&self, frame: i64, base: Option<&Mask>) -> Option<Mask> {
        let mut out = base.map(|mask| mask.resampled(self.width, self.height));
        for layer in &self.layers {
            if !layer.visible {
                continue;
            }
            let Some(mask) = layer.mask(frame) else {
                continue;
            };
            let mask = mask.resampled(self.width, self.height);
            let effective = if layer.opacity >= 1.0 {
                mask
            } else {
                mask.scaled(layer.opacity)
            };
            if out.is_none() {
                out = Some(Mask::new(self.width, self.height));
            }
            let Some(target) = out.as_mut() else {
                continue;
            };
            match layer.mode {
                MaskMode::Add => {
                    for (slot, value) in target.as_mut_slice().iter_mut().zip(effective.as_slice())
                    {
                        *slot = (*slot).max(*value);
                    }
                }
                MaskMode::Intersect => {
                    for (slot, value) in target.as_mut_slice().iter_mut().zip(effective.as_slice())
                    {
                        *slot = (*slot).min(*value);
                    }
                }
                MaskMode::Subtract | MaskMode::Protect | MaskMode::Confine => {
                    for (slot, value) in target.as_mut_slice().iter_mut().zip(effective.as_slice())
                    {
                        *slot = (*slot).min(255 - *value);
                    }
                }
            }
        }
        out
    }

    /// Per-pixel despill amount in `0..=1` for `frame`, row-major at
    /// `width * height`; `None` means "no restriction" (no visible despill key).
    ///
    /// v1 `despill_weight` semantics: `confine` layers take the maximum of their
    /// masks, `protect` layers take the minimum of their inverted masks, in
    /// stack order. The returned elements are length `width * height`.
    pub fn despill_weight(&self, frame: i64) -> Option<Vec<f32>> {
        let mut weight: Option<Vec<u8>> = None;
        for layer in &self.layers {
            if !layer.visible {
                continue;
            }
            let Some(mask) = layer.mask(frame) else {
                continue;
            };
            let mask = mask.resampled(self.width, self.height);
            let effective = if layer.opacity >= 1.0 {
                mask
            } else {
                mask.scaled(layer.opacity)
            };
            match layer.mode {
                MaskMode::Confine => {
                    weight = Some(match weight {
                        None => effective.as_slice().to_vec(),
                        Some(current) => current
                            .iter()
                            .zip(effective.as_slice())
                            .map(|(left, right)| (*left).max(*right))
                            .collect(),
                    });
                }
                MaskMode::Add | MaskMode::Subtract | MaskMode::Intersect | MaskMode::Protect => {
                    let inverted = |value: u8| 255 - value;
                    match weight.as_mut() {
                        None => {
                            weight =
                                Some(effective.as_slice().iter().copied().map(inverted).collect());
                        }
                        Some(current) => {
                            for (slot, value) in current.iter_mut().zip(effective.as_slice()) {
                                *slot = (*slot).min(inverted(*value));
                            }
                        }
                    }
                }
            }
        }
        weight.map(|values| {
            values
                .into_iter()
                .map(|value| f32::from(value) / 255.0)
                .collect()
        })
    }

    // -- copying -------------------------------------------------------------

    /// Copy every layer of `source_scene` onto `target_scene` and append the
    /// copies to the stack.
    ///
    /// Each copy gets a fresh id from the stack counter and the target scene
    /// range (`target_start..=target_end`); timing is preserved as a scene-local
    /// offset (see [`MaskLayer::copy_for_scene`]). Returns the new layer ids in
    /// copy order. The selection is unchanged.
    pub fn copy_layers_from_scene(
        &mut self,
        source_scene: u32,
        target_scene: u32,
        target_start: i64,
        target_end: i64,
    ) -> Vec<u32> {
        let source_ids: Vec<u32> = self
            .layers
            .iter()
            .filter(|layer| layer.scene_index == source_scene)
            .map(|layer| layer.layer_id)
            .collect();
        let mut copies = Vec::with_capacity(source_ids.len());
        for source_id in source_ids {
            let new_id = self.next_id;
            let Some(copy) = self.by_id(source_id).map(|source| {
                source.copy_for_scene(new_id, target_scene, target_start, target_end)
            }) else {
                continue;
            };
            self.next_id = self.next_id.saturating_add(1);
            self.layers.push(copy);
            copies.push(new_id);
        }
        copies
    }
}

/// Key interval around `frame` for a sorted key list (v1 segment maths).
fn segment_for_keys(keys: &[i64], frame: i64) -> Option<KeySegment> {
    let first = *keys.first()?;
    let last = *keys.last()?;
    if frame <= first {
        return Some(KeySegment {
            before: first,
            after: first,
            position: 0.0,
        });
    }
    if frame >= last {
        return Some(KeySegment {
            before: last,
            after: last,
            position: 0.0,
        });
    }
    let before = keys.iter().copied().rfind(|key| *key <= frame)?;
    let after = keys.iter().copied().find(|key| *key > frame)?;
    let position = (frame - before) as f32 / (after - before).max(1) as f32;
    Some(KeySegment {
        before,
        after,
        position,
    })
}

/// Deterministic overlay colour for a layer id (`#rrggbb`).
///
/// v1 picked a random HSV colour per new layer (`random_mask_color`), which
/// makes saved projects depend on the random state. This port uses the same
/// palette (golden-ratio hue steps, saturation 0.62, value 1.0) but derived
/// deterministically from the layer id, so a project always renders its
/// overlays with the same colours.
pub fn suggest_color(layer_id: u32) -> String {
    let hue = (f64::from(layer_id) * 0.618_033_988_7) % 1.0;
    hsv_to_hex(hue, 0.62, 1.0)
}

/// HSV (all components 0..1) to `#rrggbb`, truncating like Python's `int()`.
fn hsv_to_hex(hue: f64, saturation: f64, value: f64) -> String {
    let scaled = (hue % 1.0) * 6.0;
    let sector = scaled.floor();
    let fraction = scaled - sector;
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * fraction);
    let t = value * (1.0 - saturation * (1.0 - fraction));
    let (red, green, blue) = match sector as i32 % 6 {
        0 => (value, t, p),
        1 => (q, value, p),
        2 => (p, value, t),
        3 => (p, q, value),
        4 => (t, p, value),
        _ => (value, p, q),
    };
    let channel = |component: f64| (component * 255.0) as u8;
    format!(
        "#{:02x}{:02x}{:02x}",
        channel(red),
        channel(green),
        channel(blue)
    )
}
