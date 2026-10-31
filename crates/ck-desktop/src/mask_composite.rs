//! Rasterise garbage/despill mask stacks for the preview and export
//! compositor (v1 `MaskStack.apply` / `MaskStack.despill_weight` semantics).
//!
//! The GUI builds one [`MaskApplication`] per requested frame; the preview
//! worker applies the garbage matte to the rendered alpha and the despill
//! weight to the colour unmix without touching the GUI thread.

use ck_core::mask_layers::{MaskMode, MaskStack};
use ck_desktop_core::MaskStacks;
use image::{GrayImage, RgbaImage, imageops::FilterType};
use std::sync::Arc;

/// v1 combines despill masks at this width and upsamples the feather.
pub const DESPILL_WEIGHT_LIMIT: u32 = 1024;

/// Frame mask data consumed by the compositor.
#[derive(Debug)]
pub struct MaskApplication {
    /// Garbage layers in stack order: mode and full-size 8-bit matte.
    pub garbage: Vec<(MaskMode, Arc<GrayImage>)>,
    /// Combined despill weight at [`DESPILL_WEIGHT_LIMIT`] (255 = unmix).
    pub despill_weight: Option<Arc<GrayImage>>,
}

impl MaskApplication {
    /// Rasterise both stacks at `frame` for a `width x height` source.
    pub fn build(stacks: &MaskStacks, frame: i64, width: u32, height: u32) -> Option<Self> {
        let garbage = garbage_ops(&stacks.garbage, frame, width, height);
        let despill_weight = despill_weight(&stacks.despill, frame);
        (!garbage.is_empty() || despill_weight.is_some()).then_some(Self {
            garbage,
            despill_weight,
        })
    }

    /// Fold the garbage layers into an 8-bit matte (v1 `_apply_garbage`):
    /// add = max, intersect = min, subtract = min(255 - mask).
    pub fn apply_garbage_alpha(&self, alpha: &mut [u8], width: u32, height: u32) {
        for (mode, mask) in &self.garbage {
            if mask.width() != width || mask.height() != height {
                continue;
            }
            for (value, pixel) in alpha.iter_mut().zip(mask.pixels()) {
                let matte = pixel[0];
                *value = match mode {
                    MaskMode::Add => (*value).max(matte),
                    MaskMode::Intersect => (*value).min(matte),
                    MaskMode::Subtract => (*value).min(255 - matte),
                    // A garbage stack never contains despill modes.
                    MaskMode::Protect | MaskMode::Confine => *value,
                };
            }
        }
    }
}

/// Content identity of everything that shapes the masks applied at `frame`: every visible
/// layer active there, with its mode, softness, raster size, interpolated shape and painted
/// keys. Equal fingerprints give identical [`MaskApplication`]s, so cached results stay valid
/// across edits that do not touch this frame. `None` when no layer applies at `frame`.
pub fn mask_fingerprint(stacks: &MaskStacks, frame: i64) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut any = false;
    for (tag, stack) in [(0u8, &stacks.garbage), (1u8, &stacks.despill)] {
        for layer in &stack.layers {
            if !layer.visible || !layer.in_scene(frame) {
                continue;
            }
            any = true;
            tag.hash(&mut hasher);
            layer.layer_id.hash(&mut hasher);
            format!("{:?}", layer.mode).hash(&mut hasher);
            layer.softness.hash(&mut hasher);
            (layer.width, layer.height).hash(&mut hasher);
            format!("{:?}", layer.shape_at(frame)).hash(&mut hasher);
            for key in layer.doc.frames.keys() {
                key.hash(&mut hasher);
            }
        }
    }
    any.then(|| hasher.finish())
}

fn gray(mask: ck_core::mask_layers::Mask, width: u32, height: u32) -> Option<GrayImage> {
    let mask = if mask.width() != width || mask.height() != height {
        mask.resampled(width, height)
    } else {
        mask
    };
    GrayImage::from_raw(mask.width(), mask.height(), mask.into_pixels())
}

fn garbage_ops(
    stack: &MaskStack,
    frame: i64,
    width: u32,
    height: u32,
) -> Vec<(MaskMode, Arc<GrayImage>)> {
    let mut ops = Vec::new();
    for layer in &stack.layers {
        if !layer.visible {
            continue;
        }
        let Some(raster) = layer.mask(frame) else {
            continue;
        };
        let Some(matte) = gray(raster, width, height) else {
            continue;
        };
        ops.push((layer.mode, Arc::new(matte)));
    }
    ops
}

fn limited_size(width: u32, height: u32) -> (u32, u32) {
    let largest = width.max(height);
    if largest <= DESPILL_WEIGHT_LIMIT {
        return (width, height);
    }
    let scale = DESPILL_WEIGHT_LIMIT as f32 / largest as f32;
    (
        ((width as f32 * scale).round() as u32).max(1),
        ((height as f32 * scale).round() as u32).max(1),
    )
}

fn combine(current: GrayImage, mode: MaskMode, mask: GrayImage) -> GrayImage {
    let mut out = current;
    for (value, pixel) in out.pixels_mut().zip(mask.pixels()) {
        let matte = pixel[0];
        value[0] = match mode {
            MaskMode::Confine => value[0].max(matte),
            MaskMode::Protect => value[0].min(255 - matte),
            _ => value[0],
        };
    }
    out
}

fn despill_weight(stack: &MaskStack, frame: i64) -> Option<Arc<GrayImage>> {
    let mut weight: Option<GrayImage> = None;
    for layer in &stack.layers {
        if !layer.visible {
            continue;
        }
        let Some(raster) = layer.mask(frame) else {
            continue;
        };
        let (width, height) = limited_size(raster.width(), raster.height());
        let mut matte = gray(raster, width, height)?;
        if matte.width() != width || matte.height() != height {
            matte = image::imageops::resize(&matte, width, height, FilterType::Triangle);
        }
        weight = Some(match weight {
            None => match layer.mode {
                MaskMode::Protect => {
                    let mut inverse = matte;
                    for pixel in inverse.pixels_mut() {
                        pixel[0] = 255 - pixel[0];
                    }
                    inverse
                }
                _ => matte,
            },
            Some(current) => combine(current, layer.mode, matte),
        });
    }
    weight.map(Arc::new)
}

/// Copy the garbage matte into an RGBA image's alpha channel.
pub fn apply_garbage_masks(image: &mut RgbaImage, masks: &MaskApplication) {
    if masks.garbage.is_empty() {
        return;
    }
    let (width, height) = image.dimensions();
    let mut alpha: Vec<u8> = image.pixels().map(|pixel| pixel[3]).collect();
    masks.apply_garbage_alpha(&mut alpha, width, height);
    for (pixel, value) in image.pixels_mut().zip(alpha) {
        pixel[3] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ck_core::mask_layers::{BezierNode, BezierShape, MaskKind};

    fn rectangle(x0: f32, y0: f32, x1: f32, y1: f32) -> BezierShape {
        BezierShape::with_nodes(
            vec![
                BezierNode::new(x0, y0),
                BezierNode::new(x1, y0),
                BezierNode::new(x1, y1),
                BezierNode::new(x0, y1),
            ],
            true,
        )
    }

    fn new_stacks(width: u32, height: u32) -> MaskStacks {
        MaskStacks::new(width, height)
    }

    #[test]
    fn garbage_subtract_cuts_the_alpha_inside_the_shape() {
        let mut stacks = new_stacks(8, 8);
        let layer = stacks.garbage.add(0, Some(MaskMode::Subtract));
        layer.visible = true;
        layer.set_shape(0, rectangle(0.0, 0.0, 4.0, 8.0));
        let application = MaskApplication::build(&stacks, 0, 8, 8).expect("garbage mask");
        let mut alpha = vec![255_u8; 8 * 8];
        application.apply_garbage_alpha(&mut alpha, 8, 8);
        assert_eq!(alpha[8], 0, "left half is subtracted");
        assert_eq!(alpha[6], 255, "right half is untouched");
    }

    #[test]
    fn garbage_add_and_intersect_combine_in_stack_order() {
        let mut stacks = new_stacks(8, 8);
        let add = stacks.garbage.add(0, Some(MaskMode::Add));
        add.visible = true;
        add.set_shape(0, rectangle(0.0, 0.0, 4.0, 8.0));
        let intersect = stacks.garbage.add(0, Some(MaskMode::Intersect));
        intersect.visible = true;
        intersect.set_shape(0, rectangle(0.0, 0.0, 8.0, 4.0));
        let application = MaskApplication::build(&stacks, 0, 8, 8).expect("garbage masks");
        let mut alpha = vec![0_u8; 8 * 8];
        application.apply_garbage_alpha(&mut alpha, 8, 8);
        assert_eq!(alpha[0], 255, "add fills the top-left");
        assert_eq!(alpha[40], 0, "intersect clears the bottom-left");
        assert_eq!(alpha[6], 0, "add did not cover the right half");
    }

    #[test]
    fn invisible_layers_are_not_applied() {
        let mut stacks = new_stacks(8, 8);
        let layer = stacks.garbage.add(0, Some(MaskMode::Add));
        layer.visible = false;
        layer.set_shape(0, rectangle(0.0, 0.0, 8.0, 8.0));
        assert!(MaskApplication::build(&stacks, 0, 8, 8).is_none());
    }

    #[test]
    fn despill_protect_and_confine_produce_v1_weights() {
        let mut stacks = new_stacks(8, 8);
        let protect = stacks.despill.add(0, Some(MaskMode::Protect));
        protect.visible = true;
        protect.set_shape(0, rectangle(0.0, 0.0, 4.0, 8.0));
        let application = MaskApplication::build(&stacks, 0, 8, 8).expect("despill weight");
        let weight = application.despill_weight.as_ref().expect("weight");
        assert_eq!(
            weight.get_pixel(1, 1)[0],
            0,
            "protected area is not despilled"
        );
        assert_eq!(weight.get_pixel(6, 1)[0], 255);
        assert!(application.garbage.is_empty());

        let mut stacks = new_stacks(8, 8);
        let confine = stacks.despill.add(0, Some(MaskMode::Confine));
        confine.visible = true;
        confine.set_shape(0, rectangle(0.0, 0.0, 4.0, 8.0));
        let application = MaskApplication::build(&stacks, 0, 8, 8).expect("despill weight");
        let weight = application.despill_weight.as_ref().expect("weight");
        assert_eq!(weight.get_pixel(1, 1)[0], 255, "confined area is despilled");
        assert_eq!(weight.get_pixel(6, 1)[0], 0);
    }

    #[test]
    fn frame_outside_the_layer_scene_range_contributes_nothing() {
        let mut stacks = new_stacks(8, 8);
        let layer = stacks.garbage.add(0, Some(MaskMode::Add));
        layer.set_shape(10, rectangle(0.0, 0.0, 8.0, 8.0));
        layer.set_scene_range(10, 20);
        assert!(MaskApplication::build(&stacks, 0, 8, 8).is_none());
        assert!(MaskApplication::build(&stacks, 12, 8, 8).is_some());
    }

    #[test]
    fn garbage_masks_update_only_the_alpha_channel() {
        let mut stacks = new_stacks(2, 1);
        let layer = stacks.garbage.add(0, Some(MaskMode::Subtract));
        layer.visible = true;
        layer.set_shape(0, rectangle(0.0, 0.0, 1.0, 1.0));
        let application = MaskApplication::build(&stacks, 0, 2, 1).expect("garbage mask");
        let mut image = RgbaImage::from_pixel(2, 1, image::Rgba([10, 20, 30, 255]));
        apply_garbage_masks(&mut image, &application);
        assert_eq!(image.get_pixel(0, 0).0, [10, 20, 30, 0]);
        assert_eq!(image.get_pixel(1, 0).0, [10, 20, 30, 255]);
    }

    #[test]
    fn mask_kind_is_ignored_but_visibility_and_modes_respected() {
        let mut stacks = MaskStacks::new(4, 4);
        let layer = stacks.garbage.add(0, None);
        assert_eq!(layer.mode, MaskMode::Subtract);
        layer.set_shape(0, rectangle(0.0, 0.0, 4.0, 4.0));
        let application = MaskApplication::build(&stacks, 0, 4, 4).expect("mask");
        assert!(matches!(application.garbage[0].0, MaskMode::Subtract));
        let _ = MaskKind::Garbage;
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    use ck_core::mask_layers::{BezierNode, BezierShape, MaskKind};

    /// Timing probe (run with `--release --ignored --nocapture`).
    #[test]
    #[ignore]
    fn mask_build_timing_probe_1080p() {
        for softness in [0u32, 30, 120] {
            let mut stacks = MaskStacks::new(1920, 1080);
            let stack = stacks.stack_mut(MaskKind::Garbage);
            let layer = stack.add(0, None);
            layer.softness = softness;
            layer.set_shape(
                0,
                BezierShape::with_nodes(
                    vec![
                        BezierNode::new(1350.0, 120.0),
                        BezierNode::new(1850.0, 160.0),
                        BezierNode::new(1800.0, 620.0),
                        BezierNode::new(1420.0, 560.0),
                    ],
                    true,
                ),
            );
            let started = std::time::Instant::now();
            let built = MaskApplication::build(&stacks, 0, 1920, 1080).is_some();
            println!(
                "build softness {softness}: {:.3} s ({built})",
                started.elapsed().as_secs_f64()
            );
        }
    }
}
