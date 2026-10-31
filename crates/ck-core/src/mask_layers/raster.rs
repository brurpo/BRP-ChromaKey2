//! Pure-Rust CPU rasterisation for bezier mask shapes and 8-bit mask buffers.
//!
//! The v1 application rasterised shapes with OpenCV (`cv2.fillPoly` with
//! `LINE_AA`, followed by a 3x3 Gaussian blur). This module reproduces the same
//! workflow without any image dependency:
//!
//! * the outline is flattened into a polyline and filled with an **even-odd
//!   scanline** rule (the same rule `fillPoly` uses for self-intersecting
//!   contours),
//! * vertical anti-aliasing uses [`SUPERSAMPLE_ROWS`] sub-scanlines per raster
//!   row and horizontal anti-aliasing is computed analytically (exact horizontal
//!   coverage per pixel, so the effective horizontal resolution is continuous),
//! * an explicit non-zero `softness` feathers the result with a separable
//!   Gaussian blur whose kernel mirrors v1's `cv2.GaussianBlur` parameters.
//!
//! Conventions:
//!
//! * `width`/`height` are the raster dimensions in pixels, usually the source
//!   (media) resolution: a 4K project uses `3840 x 2160`, an 8-bit mask is
//!   `width * height` bytes.
//! * Shape coordinates are in the same pixel space, as `f32`; row-major, origin
//!   top-left. Pixel `(px, py)` covers the half-open square
//!   `[px, px+1) x [py, py+1)` in that coordinate space (its centre is
//!   `(px + 0.5, py + 0.5)`). A rectangle whose corners are on integer
//!   coordinates therefore fills whole pixels only.
//! * An *open* bezier shape is still filled as a closed outline; v1's
//!   `fillPoly` implicitly closed contours too.
//! * Masks are 8-bit alpha (`0` = fully outside, `255` = fully inside) because
//!   that is what v1 stores, composites and spills; callers that want 0..1
//!   weights divide by 255.
//!
//! v1 parity notes:
//!
//! * v1 additionally blurred every raster with a 3x3, sigma 0.6 kernel even at
//!   `softness == 0`. This port relies on the supersampled edge instead, so the
//!   edge profile of an untouched shape is *sharper* than OpenCV's output (and
//!   identical to v1's own pure-Python fallback, which performed no blur).
//! * Softening uses `sigma = softness / 2.5` and kernel size
//!   `max(3, softness | 1)`, exactly the v1 formula; the kernel is applied with
//!   reflect-101 border handling to approximate `cv2.GaussianBlur`.

use crate::mask_layers::spline::BezierShape;

/// Number of sub-scanline samples per raster row used for vertical coverage.
///
/// The value is the task-mandated minimum of 4; horizontal coverage is computed
/// analytically and is therefore not limited by this constant.
pub const SUPERSAMPLE_ROWS: u32 = 4;

/// Number of cubic subdivisions per segment when flattening a shape for the
/// rasteriser. v1 sampled 14 points per segment for display/hit-testing; the
/// rasteriser uses a slightly finer fixed step so that long curved edges stay
/// within a fraction of a pixel.
pub const RASTER_FLATTEN_STEPS: u32 = 16;

/// An 8-bit single-channel mask raster, row-major, origin top-left.
///
/// This is the in-memory representation used by the mask module (and the same
/// layout a decoded grayscale mask PNG has). It intentionally carries no image
/// encoder or decoder: project persistence stores masks as compressed blobs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Mask {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl Mask {
    /// Create a fully transparent (`0`) mask of `width x height` pixels.
    ///
    /// # Panics
    ///
    /// Panics if `width * height` does not fit in memory (`usize`).
    pub fn new(width: u32, height: u32) -> Self {
        let len = Self::len_for(width, height);
        Self {
            width,
            height,
            data: vec![0; len],
        }
    }

    /// Wrap existing row-major 8-bit pixels, checking the length.
    pub fn from_pixels(width: u32, height: u32, data: Vec<u8>) -> Option<Self> {
        if data.len() == Self::len_for(width, height) {
            Some(Self {
                width,
                height,
                data,
            })
        } else {
            None
        }
    }

    /// Raster width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Raster height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Number of pixels (`width * height`).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the raster has no pixels (a zero-sized mask).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Row-major pixel bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Mutable row-major pixel bytes (for brush painting done by the caller).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Consume the mask and return its row-major pixels.
    pub fn into_pixels(self) -> Vec<u8> {
        self.data
    }

    /// Pixel value at `(x, y)`; panics when out of bounds.
    pub fn get(&self, x: u32, y: u32) -> u8 {
        self.data[(y as usize) * (self.width as usize) + (x as usize)]
    }

    /// Set the pixel at `(x, y)`; panics when out of bounds.
    pub fn set(&mut self, x: u32, y: u32, value: u8) {
        let index = (y as usize) * (self.width as usize) + (x as usize);
        self.data[index] = value;
    }

    /// Fraction of pixels above the v1 opaque test threshold (`> 127`), 0..1.
    pub fn coverage(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let set = self.data.iter().filter(|value| **value > 127).count();
        set as f32 / self.data.len() as f32
    }

    /// Multiply every pixel by `factor` (clamped to 0..1) and round.
    ///
    /// Used to apply a layer `opacity`, an app-schema extension absent from v1;
    /// `factor == 1.0` returns the mask unchanged.
    pub fn scaled(&self, factor: f32) -> Mask {
        if factor >= 1.0 {
            return self.clone();
        }
        let factor = if factor.is_finite() {
            factor.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut out = self.clone();
        if factor <= 0.0 {
            out.data.fill(0);
            return out;
        }
        for value in &mut out.data {
            *value = (f32::from(*value) * factor).round().clamp(0.0, 255.0) as u8;
        }
        out
    }

    /// Bilinear resample to `width x height`.
    ///
    /// This mirrors v1's `Image.resize(..., Image.BILINEAR)` / `resize_mask`
    /// fallback for imported or painting masks: it is a resampling convenience,
    /// not exact PIL parity. Same-size requests return a copy.
    pub fn resampled(&self, width: u32, height: u32) -> Mask {
        if width == self.width && height == self.height {
            return self.clone();
        }
        let mut out = Mask::new(width, height);
        if self.width == 0 || self.height == 0 || width == 0 || height == 0 || self.data.is_empty()
        {
            return out;
        }
        let scale_x = (self.width as f32) / (width as f32);
        let scale_y = (self.height as f32) / (height as f32);
        let max_x = (self.width - 1) as f32;
        let max_y = (self.height - 1) as f32;
        for y in 0..height {
            let source_y = (((y as f32) + 0.5) * scale_y - 0.5).clamp(0.0, max_y);
            let y0 = source_y.floor() as u32;
            let y1 = (y0 + 1).min(self.height - 1);
            let fy = source_y - y0 as f32;
            for x in 0..width {
                let source_x = (((x as f32) + 0.5) * scale_x - 0.5).clamp(0.0, max_x);
                let x0 = source_x.floor() as u32;
                let x1 = (x0 + 1).min(self.width - 1);
                let fx = source_x - x0 as f32;
                let top =
                    f32::from(self.get(x0, y0)) * (1.0 - fx) + f32::from(self.get(x1, y0)) * fx;
                let bottom =
                    f32::from(self.get(x0, y1)) * (1.0 - fx) + f32::from(self.get(x1, y1)) * fx;
                let value = top * (1.0 - fy) + bottom * fy;
                out.set(x, y, value.round().clamp(0.0, 255.0) as u8);
            }
        }
        out
    }

    fn len_for(width: u32, height: u32) -> usize {
        let len = u64::from(width) * u64::from(height);
        usize::try_from(len).expect("mask dimensions overflow addressable memory")
    }
}

/// Blend two masks: `first * (1 - amount) + second * amount`, rounded to 8 bits.
///
/// Mirrors v1's `blend_masks` (which used `cv2.addWeighted`). `amount` is
/// clamped to 0..1. If the sizes differ, `second` is bilinearly resampled to
/// the size of `first`.
pub fn blend_masks(first: &Mask, second: &Mask, amount: f32) -> Mask {
    let amount = if amount.is_finite() {
        amount.clamp(0.0, 1.0)
    } else {
        0.0
    };
    if first.width() != second.width() || first.height() != second.height() {
        let second = second.resampled(first.width(), first.height());
        return blend_masks(first, &second, amount);
    }
    let mut out = first.clone();
    if amount <= 0.0 {
        return out;
    }
    if amount >= 1.0 {
        return second.clone();
    }
    for (index, value) in out.data.iter_mut().enumerate() {
        let a = f32::from(first.data[index]);
        let b = f32::from(second.data[index]);
        *value = (a * (1.0 - amount) + b * amount).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Rasterise a bezier shape into a `width x height` 8-bit mask.
///
/// * Fewer than two nodes produces an empty mask (v1 behaviour).
/// * The outline is filled with the even-odd rule and closed implicitly.
/// * `softness` is in pixels at the target raster resolution (0 = only edge AA).
pub fn rasterize(shape: &BezierShape, width: u32, height: u32, softness: u32) -> Mask {
    let mut mask = Mask::new(width, height);
    if width == 0 || height == 0 || shape.nodes.len() < 2 {
        return mask;
    }
    let polygon = flatten(shape);
    if polygon.len() < 2 {
        return mask;
    }
    fill_even_odd(&mut mask, &polygon);
    if softness > 0 {
        mask = soften(&mask, softness);
    }
    mask
}

/// Flatten the shape into a polygon suitable for scanline filling.
///
/// The returned point list is not explicitly closed; the filler always wraps
/// the last point back to the first, matching `cv2.fillPoly`.
pub fn flatten(shape: &BezierShape) -> Vec<(f32, f32)> {
    let mut points = shape.sample(RASTER_FLATTEN_STEPS);
    if points.len() > 1 {
        let first = points[0];
        let last = points[points.len() - 1];
        if (first.0 - last.0).abs() <= 1e-4 && (first.1 - last.1).abs() <= 1e-4 {
            points.pop();
        }
    }
    points.retain(|point| point.0.is_finite() && point.1.is_finite());
    points
}

/// Even-odd scanline fill with analytic horizontal and supersampled vertical
/// coverage.
///
/// Pixels outside the polygon stay 0, pixels fully covered become 255 and
/// boundary pixels get intermediate coverage, so a static integer-aligned
/// rectangle fills its interior with exactly `255` (verified by the tests).
fn fill_even_odd(mask: &mut Mask, polygon: &[(f32, f32)]) {
    let width = mask.width as usize;
    let height = mask.height as usize;
    if width == 0 || height == 0 || polygon.len() < 2 {
        return;
    }
    let rows = SUPERSAMPLE_ROWS.max(1);
    let row_weight = 1.0 / rows as f32;
    let mut coverage = vec![0.0_f32; width];
    let mut crossings: Vec<f32> = Vec::with_capacity(16);
    let count = polygon.len();
    for row in 0..height {
        coverage.fill(0.0);
        for sub in 0..rows {
            let sample_y = (row as f32) + ((sub as f32) + 0.5) * row_weight;
            crossings.clear();
            for index in 0..count {
                let (x0, y0) = polygon[index];
                let (x1, y1) = polygon[(index + 1) % count];
                // Half-open edge test: a vertex shared by two edges is counted
                // exactly once, so the crossing parity stays consistent.
                let upward = y0 <= sample_y && y1 > sample_y;
                let downward = y1 <= sample_y && y0 > sample_y;
                if !upward && !downward {
                    continue;
                }
                let span = y1 - y0;
                if span == 0.0 {
                    continue;
                }
                let t = (sample_y - y0) / span;
                let x = x0 + t * (x1 - x0);
                if x.is_finite() {
                    crossings.push(x);
                }
            }
            if crossings.len() < 2 {
                continue;
            }
            crossings.sort_by(f32::total_cmp);
            for pair in crossings.as_chunks::<2>().0 {
                accumulate_span(&mut coverage, pair[0], pair[1], row_weight);
            }
        }
        let row_start = row * width;
        let out_row = &mut mask.data[row_start..row_start + width];
        for (pixel, value) in out_row.iter_mut().zip(coverage.iter()) {
            *pixel = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
}

/// Add the horizontal coverage of the half-open span `[x0, x1)` to a row
/// accumulator, weighted by `weight`. Fully covered interior pixels are bumped
/// in bulk; the two fractional pixels use their exact overlap length.
fn accumulate_span(coverage: &mut [f32], x0: f32, x1: f32, weight: f32) {
    let width = coverage.len();
    if !x0.is_finite() || !x1.is_finite() || x0 >= x1 {
        return;
    }
    let first = x0.floor();
    let last = (x1.floor()).min(width as f32);
    if last < 0.0 {
        return;
    }
    if first == last {
        let index = first as isize;
        if index >= 0 && (index as usize) < width {
            coverage[index as usize] += (x1 - x0) * weight;
        }
        return;
    }
    // Fractional pixel at the start of the span.
    let first_index = first as isize;
    if first_index >= 0 && (first_index as usize) < width {
        let partial = (first + 1.0 - x0).clamp(0.0, 1.0);
        coverage[first_index as usize] += partial * weight;
    }
    // Fully covered interior pixels.
    let start = (first + 1.0).max(0.0);
    let end = last.min(width as f32);
    if end > start {
        let start_index = start as usize;
        let end_index = end as usize;
        for value in &mut coverage[start_index..end_index] {
            *value += weight;
        }
    }
    // Fractional pixel at the end of the span.
    let last_index = last as isize;
    if last_index >= 0 && (last_index as usize) < width && x1 > last {
        coverage[last_index as usize] += (x1 - last).clamp(0.0, 1.0) * weight;
    }
}

/// Separable Gaussian soften of an 8-bit mask, mirroring v1's `soften`.
///
/// * `softness == 0` returns a copy of the input.
/// * Kernel size is `max(3, softness | 1)` and sigma is `max(0.1, softness / 2.5)`
///   (the v1 `cv2.GaussianBlur` parameters).
/// * Borders use reflect-101, approximating OpenCV's default border.
pub fn soften(mask: &Mask, softness: u32) -> Mask {
    if softness == 0 || mask.is_empty() {
        return mask.clone();
    }
    let kernel_size = softness.max(3) | 1;
    let radius = (kernel_size / 2) as i32;
    let sigma = f32::max(0.1, softness as f32 / 2.5);
    let mut kernel = Vec::with_capacity((radius as usize) * 2 + 1);
    for offset in -radius..=radius {
        let value = (-0.5 * (offset as f32 / sigma).powi(2)).exp();
        kernel.push(value);
    }
    let sum: f32 = kernel.iter().sum();
    for weight in &mut kernel {
        *weight /= sum;
    }
    let width = mask.width as i32;
    let height = mask.height as i32;
    let w = width as usize;
    let mut out = Mask::new(mask.width, mask.height);
    // Only the non-zero bounding box grown by `radius` can become non-zero: every
    // window outside it sees only zeros (reflect-101 included), and adding 0.0 never
    // changes a float sum, so skipping that area is exact.
    let mut bounds: Option<(i32, i32, i32, i32)> = None;
    for y in 0..height {
        let row = &mask.data[y as usize * w..(y as usize + 1) * w];
        if let (Some(first), Some(last)) = (
            row.iter().position(|&v| v != 0),
            row.iter().rposition(|&v| v != 0),
        ) {
            let (first, last) = (first as i32, last as i32);
            bounds = Some(match bounds {
                None => (first, y, last, y),
                Some((x0, y0, x1, _)) => (x0.min(first), y0, x1.max(last), y),
            });
        }
    }
    let Some((min_x, min_y, max_x, max_y)) = bounds else {
        return out;
    };
    let x0 = (min_x - radius).max(0) as usize;
    let x1 = (max_x + radius).min(width - 1) as usize;
    let y0 = (min_y - radius).max(0);
    let y1 = (max_y + radius).min(height - 1);
    // Horizontal pass: u8 -> f32 for the source rows that hold mask pixels. Interior
    // pixels index directly; only the `radius` pixels at each end need reflect-101.
    let mut horizontal = vec![0.0_f32; mask.data.len()];
    for y in min_y as usize..=max_y as usize {
        let src = &mask.data[y * w..(y + 1) * w];
        let dst = &mut horizontal[y * w..(y + 1) * w];
        let edge = |x: i32| -> f32 {
            kernel
                .iter()
                .enumerate()
                .map(|(k, weight)| {
                    f32::from(src[reflect101(x + k as i32 - radius, width) as usize]) * weight
                })
                .sum()
        };
        let interior = radius..(width - radius).max(radius);
        for x in x0 as i32..=x1 as i32 {
            if interior.contains(&x) {
                let window = &src[(x - radius) as usize..=(x + radius) as usize];
                dst[x as usize] = window
                    .iter()
                    .zip(&kernel)
                    .map(|(value, weight)| f32::from(*value) * weight)
                    .sum();
            } else {
                dst[x as usize] = edge(x);
            }
        }
    }
    // Vertical pass: f32 -> u8, accumulating whole row spans (row-major, vectorisable).
    // Source rows outside the mask rows are all zero and are skipped.
    let span = x1 - x0 + 1;
    let mut row = vec![0.0_f32; span];
    for y in y0..=y1 {
        row.iter_mut().for_each(|value| *value = 0.0);
        for (k, weight) in kernel.iter().enumerate() {
            let sample = reflect101(y + k as i32 - radius, height);
            if sample < min_y || sample > max_y {
                continue;
            }
            let base = sample as usize * w;
            let source = &horizontal[base + x0..=base + x1];
            for (acc, value) in row.iter_mut().zip(source) {
                *acc += value * weight;
            }
        }
        let base = y as usize * w;
        let target = &mut out.data[base + x0..=base + x1];
        for (dst, value) in target.iter_mut().zip(&row) {
            *dst = value.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Reflect-101 index (mirror without repeating the edge pixel), like OpenCV.
fn reflect101(mut index: i32, len: i32) -> i32 {
    if len <= 1 {
        return 0;
    }
    while index < 0 || index >= len {
        if index < 0 {
            index = -index;
        } else {
            index = 2 * (len - 1) - index;
        }
    }
    index
}
