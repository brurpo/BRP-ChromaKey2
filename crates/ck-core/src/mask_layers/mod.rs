//! Bezier roto shapes, garbage/despill mask layers and CPU rasterisation.
//!
//! This module is the native, dependency-light port of the v1 Python mask core:
//!
//! * [`spline`] — [`BezierShape`] / [`BezierNode`]: anchors with in/out tangent
//!   offsets, tangent linking, transforms, sampling, bounds, hit-testing,
//!   de Casteljau point insertion and keyframe interpolation.
//! * [`raster`] — [`Mask`] (8-bit row-major raster), even-odd scanline
//!   rasterisation with anti-aliasing, separable Gaussian softness and mask
//!   blend/resample helpers. No OpenCV, no image codec, no GPU.
//! * [`layers`] — [`MaskKind`], [`MaskMode`], [`InterpMode`], [`Tangent`],
//!   [`MaskDocument`], [`MaskLayer`] and [`MaskStack`]: animated garbage/despill
//!   layers keyed on absolute source frames, cubic-bezier time remapping with
//!   auto tangents, scene scoping, stacking/compositing, despill weighting and
//!   layer/scene copy helpers.
//!
//! Everything here is CPU-only, deterministic and serde-serialisable at the
//! layer level in the v1 `.brp` JSON shape (plus the app extensions documented
//! on [`MaskLayer`]). Raster key masks are runtime state: persistence stores
//! them as compressed blobs, exactly like v1 stored PNGs next to layer metadata.
//!
//! # Conventions in one place
//!
//! * Coordinates: `f32` source/display pixels, origin top-left, pixel
//!   `(px, py)` covering `[px, px+1) x [py, py+1)`.
//! * Masks: 8-bit alpha, `0` = outside, `255` = inside, row-major.
//! * Raster size: explicit `width`/`height` parameters in pixels (project/source
//!   resolution for full-quality masks).
//! * Frames: `i64` absolute source frames (same index as the playhead).
//! * Scene ranges: inclusive `start..=end` (v1 rule); unrestricted = `-1`.
//! * Interpolation: `t = (frame - before) / (after - before)` in `0..1`, then
//!   eased by [`InterpMode`]; [`Tangent`] slopes are normalised `0..1` where
//!   `0` = auto/flat and `1` = linear.
//! * Transforms: [`BezierShape::transform`] takes radians, not degrees.
//!
//! The types are not wired into the application here; they are the runtime
//! model the native mask UI and the `.chroma` project schema are expected to
//! build on.

pub mod layers;
pub mod raster;
pub mod spline;

#[cfg(test)]
mod tests;

pub use layers::{
    bezier_ease, suggest_color, InterpMode, KeySegment, MaskDocument, MaskKind, MaskLayer,
    MaskMode, MaskStack, Tangent, AUTO_TANGENT, DESPILL_MODES, GARBAGE_MODES, INTERP_MODES,
};
pub use raster::{
    blend_masks, flatten, rasterize, soften, Mask, RASTER_FLATTEN_STEPS, SUPERSAMPLE_ROWS,
};
pub use spline::{
    BezierNode, BezierShape, Bounds, ControlKind, CubicSegment, HandleSide, NearestControl,
    NearestSegment, HANDLE_EPSILON, SAMPLE_STEPS,
};
