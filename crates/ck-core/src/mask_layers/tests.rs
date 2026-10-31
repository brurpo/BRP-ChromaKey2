//! Behavioural tests ported from v1 `scripts/mask_layers_test.py`,
//! `scripts/garbage_test.py` (core maths only) and the documented
//! `spline.py`/`masklayers.py` semantics, plus JSON round-trip checks for the
//! v1 `.brp` layer shape.

use std::f32::consts::FRAC_PI_2;

use serde_json::json;

use super::*;

// -- helpers ---------------------------------------------------------------

fn rect_shape(x0: f32, y0: f32, x1: f32, y1: f32) -> BezierShape {
    let mut shape = BezierShape::new();
    shape.add_corner(x0, y0);
    shape.add_corner(x1, y0);
    shape.add_corner(x1, y1);
    shape.add_corner(x0, y1);
    shape.closed = true;
    shape
}

fn box_mask(width: u32, height: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> Mask {
    let mut mask = Mask::new(width, height);
    for y in y0..y1 {
        for x in x0..x1 {
            mask.set(x, y, 255);
        }
    }
    mask
}

fn filled(width: u32, height: u32, value: u8) -> Mask {
    Mask::from_pixels(width, height, vec![value; (width * height) as usize])
        .expect("test mask dimensions match")
}

fn lerp(a: (f32, f32), b: (f32, f32), t: f32) -> (f32, f32) {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

fn assert_approx(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected} ± {tolerance}, got {actual}"
    );
}

fn assert_mask_bounds(mask: &Mask, min_x: u32, min_y: u32, max_x: u32, max_y: u32) {
    let mut found = (u32::MAX, u32::MAX, 0_u32, 0_u32);
    for y in 0..mask.height() {
        for x in 0..mask.width() {
            if mask.get(x, y) > 0 {
                found.0 = found.0.min(x);
                found.1 = found.1.min(y);
                found.2 = found.2.max(x);
                found.3 = found.3.max(y);
            }
        }
    }
    assert_eq!(found, (min_x, min_y, max_x, max_y), "non-zero mask bounds");
}

/// v1 test shape: corner, smooth point with mirrored handles, corner, corner.
fn v1_shape() -> BezierShape {
    let mut shape = BezierShape::new();
    shape.add_corner(20.0, 20.0);
    shape.add_node(80.0, 30.0, 18.0, 8.0, true);
    shape.add_corner(70.0, 56.0);
    shape.add_corner(18.0, 50.0);
    shape.closed = true;
    shape
}

fn animated_layer() -> MaskLayer {
    let mut layer = MaskLayer::new(1, "Garbage 01", MaskKind::Garbage, 32, 32);
    let first = box_mask(32, 32, 2, 2, 12, 12);
    let second = box_mask(32, 32, 20, 20, 30, 30);
    layer.set_mask(0, &first);
    layer.set_mask(20, &second);
    layer
}

// -- spline: node/tangent basics -------------------------------------------

#[test]
fn node_tangent_linking_and_mirroring() {
    let mut shape = BezierShape::new();
    shape.add_corner(10.0, 10.0);
    let index = shape.add_node(40.0, 20.0, 18.0, 8.0, true);
    assert_eq!(index, 1);
    assert!(!shape.nodes[0].has_out());
    assert!(shape.nodes[1].has_out());
    assert!(shape.nodes[1].has_in());
    assert!(shape.nodes[1].linked);
    assert_approx(shape.nodes[1].in_dx, -18.0, 1e-6);
    assert_approx(shape.nodes[1].in_dy, -8.0, 1e-6);

    // Placing an absolute out handle mirrors the in handle on a linked node.
    assert!(shape.set_handle(1, HandleSide::Out, 50.0, 25.0, None));
    assert_approx(shape.nodes[1].out_dx, 10.0, 1e-6);
    assert_approx(shape.nodes[1].out_dy, 5.0, 1e-6);
    assert_approx(shape.nodes[1].in_dx, -10.0, 1e-6);
    assert_approx(shape.nodes[1].in_dy, -5.0, 1e-6);

    // Unlinking lets the in handle move without touching the out handle.
    assert!(!shape.set_linked(1, false), "returns the new linked flag");
    assert!(!shape.nodes[1].linked);
    assert!(shape.set_handle(1, HandleSide::In, 30.0, 0.0, None));
    assert_approx(shape.nodes[1].in_dx, -10.0, 1e-6);
    assert_approx(shape.nodes[1].in_dy, -20.0, 1e-6);
    assert_approx(shape.nodes[1].out_dx, 10.0, 1e-6);
    assert_approx(shape.nodes[1].out_dy, 5.0, 1e-6);

    // Relinking re-mirrors in from out (v1 set_linked).
    assert!(shape.set_linked(1, true));
    assert_approx(shape.nodes[1].in_dx, -10.0, 1e-6);
    assert_approx(shape.nodes[1].in_dy, -5.0, 1e-6);

    // reset_handles flattens the node into a corner.
    assert!(shape.reset_handles(1));
    assert!(!shape.nodes[1].has_out());
    assert!(!shape.nodes[1].has_in());
    assert!(shape.nodes[1].linked);

    // Out-of-range edits are reported, not panics.
    assert!(!shape.reset_handles(99));
    assert!(!shape.set_handle(99, HandleSide::In, 0.0, 0.0, None));
    assert!(!shape.set_anchor(99, 0.0, 0.0));
    assert!(!shape.move_anchor(99, 1.0, 1.0));
}

#[test]
fn add_node_mirror_flag_keeps_linked_true() {
    // v1 quirk: add_node(mirror=False) leaves linked == True but the in handle
    // at zero, so a later out drag starts mirroring.
    let mut shape = BezierShape::new();
    shape.add_node(5.0, 5.0, 2.0, 3.0, false);
    let node = shape.nodes[0];
    assert_approx(node.in_dx, 0.0, 1e-6);
    assert_approx(node.in_dy, 0.0, 1e-6);
    assert!(node.linked);
    assert!(node.has_out());

    // v1 remove_node forces the shape open only below three nodes; removing one
    // node of a quad keeps it closed.
    let mut shape = rect_shape(0.0, 0.0, 10.0, 10.0);
    assert!(shape.closed);
    assert!(shape.remove_node(0));
    assert!(shape.closed);
    assert_eq!(shape.len(), 3);
    assert!(shape.remove_node(0));
    assert!(!shape.closed);
    assert!(!shape.remove_node(97));
}

#[test]
fn transform_moves_scales_rotates_anchors_and_handles() {
    let mut shape = rect_shape(10.0, 10.0, 30.0, 20.0);
    shape.nodes[0].out_dx = 5.0;
    shape.transform(0.0, 0.0, 1.0, FRAC_PI_2, None);
    let bounds = shape.bounds(false);
    assert_approx(bounds.min_x, 15.0, 1e-4);
    assert_approx(bounds.min_y, 5.0, 1e-4);
    assert_approx(bounds.max_x, 25.0, 1e-4);
    assert_approx(bounds.max_y, 25.0, 1e-4);
    // A +x handle rotated by +90 degrees points down (+y) in image space.
    assert_approx(shape.nodes[0].out_dx, 0.0, 1e-4);
    assert_approx(shape.nodes[0].out_dy, 5.0, 1e-4);

    // Uniform scale around an explicit centre.
    let mut scaled = rect_shape(10.0, 10.0, 30.0, 20.0);
    scaled.transform(0.0, 0.0, 2.0, 0.0, Some((20.0, 15.0)));
    let bounds = scaled.bounds(false);
    assert_approx(bounds.min_x, 0.0, 1e-4);
    assert_approx(bounds.min_y, 5.0, 1e-4);
    assert_approx(bounds.max_x, 40.0, 1e-4);
    assert_approx(bounds.max_y, 25.0, 1e-4);

    // Translation is applied after scaling/rotation.
    let mut moved = rect_shape(10.0, 10.0, 30.0, 20.0);
    moved.transform(100.0, -50.0, 1.0, 0.0, Some((20.0, 15.0)));
    let bounds = moved.bounds(false);
    assert_approx(bounds.min_x, 110.0, 1e-4);
    assert_approx(bounds.max_x, 130.0, 1e-4);
    assert_approx(bounds.min_y, -40.0, 1e-4);
    assert_approx(bounds.max_y, -30.0, 1e-4);
}

#[test]
fn segments_and_sample_follow_cubic_geometry() {
    let mut open = BezierShape::new();
    open.add_node(0.0, 0.0, 0.0, 50.0, true);
    open.add_corner(100.0, 0.0);
    let segments = open.segments();
    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].start, (0.0, 0.0));
    assert_eq!(segments[0].control1, (0.0, 50.0));
    // No in-handle on the end node: its anchor doubles as control2.
    assert_eq!(segments[0].control2, (100.0, 0.0));
    assert_eq!(segments[0].end, (100.0, 0.0));
    let mid = segments[0].eval(0.5);
    assert_approx(mid.0, 50.0, 1e-4);
    assert_approx(mid.1, 18.75, 1e-4);
    let points = open.sample(SAMPLE_STEPS);
    assert_eq!(points.len(), 1 + SAMPLE_STEPS as usize);
    assert_eq!(points[points.len() - 1], (100.0, 0.0));

    // A closed two-node shape wraps, producing two segments.
    let mut closed = open.clone();
    closed.closed = true;
    assert_eq!(closed.segments().len(), 2);
    assert_eq!(closed.sample(4).len(), 1 + 2 * 4);

    // Degenerate shapes.
    assert!(BezierShape::new().sample(4).is_empty());
    assert!(BezierShape::new().segments().is_empty());
    let mut single = BezierShape::new();
    single.add_corner(3.0, 4.0);
    assert_eq!(single.sample(4), vec![(3.0, 4.0)]);
}

#[test]
fn bounds_include_handles_on_request() {
    let mut shape = rect_shape(10.0, 10.0, 30.0, 20.0);
    let anchors = shape.bounds(false);
    assert_eq!(
        anchors,
        Bounds {
            min_x: 10.0,
            min_y: 10.0,
            max_x: 30.0,
            max_y: 20.0,
        }
    );
    shape.nodes[0].out_dx = -5.0;
    shape.nodes[0].out_dy = -6.0;
    let with_handles = shape.bounds(true);
    assert_approx(with_handles.min_x, 5.0, 1e-6);
    assert_approx(with_handles.min_y, 4.0, 1e-6);
    assert_approx(with_handles.width(), 25.0, 1e-6);
    assert_approx(with_handles.height(), 16.0, 1e-6);
    let empty = BezierShape::new().bounds(false);
    assert_eq!(
        empty,
        Bounds {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 0.0,
            max_y: 0.0,
        }
    );
}

// -- spline: editing and animation -----------------------------------------

#[test]
fn insert_point_parity_splits_segment_and_preserves_area() {
    let original = v1_shape();
    let before_raster = original.rasterize(96, 64, 0);
    let mut shape = original.clone();
    let inserted = shape.insert_point(0, 0.5).expect("valid segment");
    assert_eq!(inserted, 1);
    assert_eq!(shape.len(), 5);

    // The new anchor is the de Casteljau point of the original segment.
    let segment = original.segments()[0];
    let point = segment.eval(0.5);
    assert_approx(shape.nodes[1].x, point.0, 1e-4);
    assert_approx(shape.nodes[1].y, point.1, 1e-4);

    // v1 parity: out <- d - point, in <- e - point (the swapped assignment).
    let a = lerp(segment.start, segment.control1, 0.5);
    let b = lerp(segment.control1, segment.control2, 0.5);
    let c = lerp(segment.control2, segment.end, 0.5);
    let d = lerp(a, b, 0.5);
    let e = lerp(b, c, 0.5);
    assert_approx(shape.nodes[1].out_dx, d.0 - point.0, 1e-4);
    assert_approx(shape.nodes[1].in_dx, e.0 - point.0, 1e-4);

    // Nearby handles still follow v1: start out gets a, end in gets c.
    assert_approx(shape.nodes[0].out_dx, a.0 - shape.nodes[0].x, 1e-4);
    assert_approx(shape.nodes[2].in_dx, c.0 - shape.nodes[2].x, 1e-4);

    // The v1 area-preservation check (coverage within 0.02 of the original).
    let after_raster = shape.rasterize(96, 64, 0);
    let delta = (after_raster.coverage() - before_raster.coverage()).abs();
    assert!(delta < 0.02, "area changed by {delta}");

    assert!(shape.insert_point(97, 0.5).is_none());
    let mut clamped = original.clone();
    assert!(clamped.insert_point(0, -4.0).is_some());
}

#[test]
fn insert_point_exact_preserves_the_outline() {
    let original = v1_shape();
    let segment = original.segments()[0];
    let mut shape = original.clone();
    let inserted = shape.insert_point_exact(0, 0.5).expect("valid segment");
    assert_eq!(inserted, 1);
    // Exact de Casteljau split: segment0 = (start, a, d, point) and
    // segment1 = (point, e, c, end) — the same geometric curve as the original.
    let a = lerp(segment.start, segment.control1, 0.5);
    let b = lerp(segment.control1, segment.control2, 0.5);
    let c = lerp(segment.control2, segment.end, 0.5);
    let d = lerp(a, b, 0.5);
    let e = lerp(b, c, 0.5);
    let point = lerp(d, e, 0.5);
    let segments = shape.segments();
    assert_eq!(segments[0].start, segment.start);
    assert_approx(segments[0].control1.0, a.0, 1e-4);
    assert_approx(segments[0].control2.0, d.0, 1e-4);
    assert_approx(segments[0].end.0, point.0, 1e-4);
    assert_approx(segments[1].control1.0, e.0, 1e-4);
    assert_approx(segments[1].control2.0, c.0, 1e-4);
    assert_eq!(segments[1].end, segment.end);
    // The new node starts auto-linked and smooth.
    assert!(shape.nodes[1].linked);
}

#[test]
fn interpolate_blends_equal_node_counts_and_switches_flags() {
    let mut first = rect_shape(10.0, 10.0, 20.0, 20.0);
    first.closed = false;
    let mut second = rect_shape(30.0, 10.0, 40.0, 20.0);
    second.nodes[0].linked = false;
    second.closed = true;

    // v1 switches the flags over at t >= 0.5 (the second shape owns them from
    // the exact midpoint on).
    let early = first.interpolate(&second, 0.4);
    assert!(!early.closed);
    assert!(early.nodes[0].linked);
    let mid = first.interpolate(&second, 0.5);
    assert_approx(mid.nodes[0].x, 20.0, 1e-6);
    assert_approx(mid.nodes[0].y, 10.0, 1e-6);
    assert!(mid.closed, "closed flag switches over at t >= 0.5");
    assert!(!mid.nodes[0].linked);

    let late = first.interpolate(&second, 0.75);
    assert!(late.closed);
    assert!(!late.nodes[0].linked);

    assert_eq!(first.interpolate(&second, 0.0), first);
    assert_eq!(first.interpolate(&second, 1.0), second);

    // Node-count mismatch returns a copy of self (v1 behaviour).
    let mut triangle = BezierShape::new();
    triangle.add_corner(0.0, 0.0);
    triangle.add_corner(1.0, 0.0);
    triangle.add_corner(0.0, 1.0);
    assert_eq!(first.interpolate(&triangle, 0.5), first);
}

// -- raster ----------------------------------------------------------------

#[test]
fn rasterize_axis_aligned_rectangle_is_exact_and_bounded() {
    let shape = rect_shape(10.0, 10.0, 30.0, 30.0);
    let mask = shape.rasterize(64, 64, 0);
    assert_eq!(mask.width(), 64);
    assert_eq!(mask.height(), 64);
    assert_eq!(mask.len(), 64 * 64);
    assert_approx(mask.coverage(), 400.0 / 4096.0, 1e-6);
    assert_eq!(mask.get(10, 10), 255);
    assert_eq!(mask.get(29, 29), 255);
    assert_eq!(mask.get(9, 10), 0);
    assert_eq!(mask.get(30, 10), 0);
    assert_eq!(mask.get(10, 9), 0);
    assert_eq!(mask.get(10, 30), 0);
    assert_mask_bounds(&mask, 10, 10, 29, 29);

    // An open outline is filled as if closed, like cv2.fillPoly.
    let mut open = shape.clone();
    open.closed = false;
    assert_eq!(open.rasterize(64, 64, 0), mask);

    // The raster size is explicit: the same shape at half size.
    let small = rect_shape(10.0, 10.0, 20.0, 20.0).rasterize(32, 32, 0);
    assert_approx(small.coverage(), 100.0 / 1024.0, 1e-6);

    // Degenerate shapes rasterise to empty masks (v1 returns zeros).
    assert_eq!(BezierShape::new().rasterize(8, 8, 0).coverage(), 0.0);
    let mut single = BezierShape::new();
    single.add_corner(2.0, 2.0);
    assert_eq!(single.rasterize(8, 8, 0).coverage(), 0.0);
    assert_eq!(Mask::new(0, 0).coverage(), 0.0);
}

#[test]
fn rasterize_antialiases_fractional_edges() {
    // Integer horizontal edges, half-pixel vertical start/end.
    let vertical = rect_shape(10.0, 10.5, 30.0, 20.5).rasterize(64, 64, 0);
    assert_eq!(vertical.get(20, 10), 128);
    assert_eq!(vertical.get(20, 11), 255);
    assert_eq!(vertical.get(20, 19), 255);
    assert_eq!(vertical.get(20, 20), 128);
    assert_eq!(vertical.get(20, 21), 0);

    // Half-pixel horizontal start/end.
    let horizontal = rect_shape(10.5, 10.0, 30.5, 30.0).rasterize(64, 64, 0);
    assert_eq!(horizontal.get(10, 15), 128);
    assert_eq!(horizontal.get(11, 15), 255);
    assert_eq!(horizontal.get(29, 15), 255);
    assert_eq!(horizontal.get(30, 15), 128);
    assert_eq!(horizontal.get(31, 15), 0);
}

#[test]
fn soften_feathers_edges_and_keeps_interior() {
    let hard = rect_shape(10.0, 10.0, 30.0, 30.0).rasterize(64, 64, 0);
    let soft = soften(&hard, 4);
    assert_eq!(soft.get(20, 20), 255, "interior stays opaque");
    let edge = soft.get(10, 20);
    assert!(
        (100..=230).contains(&edge),
        "edge value {edge} is feathered"
    );
    assert_eq!(soft.get(7, 20), 0, "feather does not reach 3px away");
    assert_eq!(soften(&hard, 0), hard, "softness 0 is a no-op");
    assert_eq!(soften(&Mask::new(0, 0), 6), Mask::new(0, 0));
    // A feathered layer mask has more than two values across an edge.
    let steps: std::collections::BTreeSet<u8> = (0..64).map(|x| soft.get(x, 20)).collect();
    assert!(
        steps.len() >= 5,
        "only {} distinct edge values",
        steps.len()
    );
}

#[test]
fn mask_resample_scale_blend_and_document() {
    let mask = Mask::from_pixels(2, 2, vec![0, 255, 255, 0]).expect("2x2 mask");
    let upscaled = mask.resampled(4, 4);
    assert_eq!((upscaled.width(), upscaled.height()), (4, 4));
    assert_eq!(upscaled.get(0, 0), 0);
    assert_eq!(upscaled.get(3, 0), 255);
    assert_eq!(mask.resampled(2, 2), mask);

    let scaled = filled(4, 4, 255).scaled(0.5);
    assert_eq!(scaled.get(0, 0), 128);
    assert_eq!(filled(4, 4, 255).scaled(0.0).get(0, 0), 0);
    assert_eq!(filled(4, 4, 255).scaled(3.0).get(0, 0), 255);

    let black = filled(4, 4, 0);
    let white = filled(4, 4, 255);
    assert_eq!(blend_masks(&black, &white, 0.5).get(0, 0), 128);
    assert_eq!(blend_masks(&black, &white, 2.0).get(0, 0), 255);
    assert_eq!(blend_masks(&black, &white, -1.0).get(0, 0), 0);

    let mut doc = MaskDocument::new(8, 8);
    doc.set_mask(5, &filled(4, 4, 255), false);
    assert!(doc.has(5));
    assert_eq!(doc.get(5).map(Mask::coverage), Some(1.0));
    assert_eq!(doc.keyframes(), vec![5]);
    let partial = box_mask(4, 4, 0, 0, 2, 2);
    doc.set_mask(5, &partial, true);
    assert_eq!(doc.get(5).map(Mask::coverage), Some(1.0), "add takes max");
    assert!(doc.copy_to_range(5, 0, 3));
    assert_eq!(doc.keyframes(), vec![0, 1, 2, 3, 5]);
    assert_eq!(doc.coverage(1), 1.0);
    assert!(doc.remove(1).is_some());
    assert_eq!(doc.coverage(1), 0.0);
    doc.clear();
    assert!(doc.keyframes().is_empty());
}

#[test]
fn nearest_control_distance_and_nearest_segment() {
    let shape = v1_shape();
    let anchor = shape.nearest_control(20.0, 20.0, 6.0).expect("anchor hit");
    assert_eq!(anchor.kind, ControlKind::Anchor);
    assert_eq!(anchor.index, 0);
    let (out_x, out_y) = shape.nodes[1].out_point();
    let handle = shape
        .nearest_control(out_x, out_y, 6.0)
        .expect("out handle hit");
    assert_eq!(handle.kind, ControlKind::Out);
    assert_eq!(handle.index, 1);
    assert!(shape.nearest_control(150.0, 5.0, 4.0).is_none());

    let rect = rect_shape(10.0, 10.0, 30.0, 30.0);
    assert_approx(rect.distance(20.0, 10.0), 0.0, 0.5);
    assert_approx(rect.distance(20.0, 20.0), 10.0, 0.3);
    assert!(BezierShape::new().distance(0.0, 0.0).is_infinite());
    let mut single = BezierShape::new();
    single.add_corner(1.0, 1.0);
    assert_approx(single.distance(4.0, 5.0), 5.0, 1e-6);

    let point = rect.segments()[0].eval(0.5);
    let nearest = rect.nearest_segment(point.0, point.1).expect("segment hit");
    assert_eq!(nearest.segment, 0);
    assert!(nearest.distance < 0.05);
    assert!(BezierShape::new().nearest_segment(0.0, 0.0).is_none());
}

// -- timing ----------------------------------------------------------------

#[test]
fn bezier_ease_matches_smoothstep_linear_and_clamps() {
    assert_approx(bezier_ease(0.0, 0.0, 0.0), 0.0, 1e-6);
    assert_approx(bezier_ease(1.0, 0.0, 0.0), 1.0, 1e-6);
    // Auto tangents reduce to smoothstep; midpoint is exactly 0.5.
    assert_approx(bezier_ease(0.5, 0.0, 0.0), 0.5, 1e-6);
    // Ease-in: slower than linear early in the interval.
    assert!(bezier_ease(0.25, 0.0, 0.0) < 0.25);
    assert_approx(bezier_ease(0.25, 0.0, 0.0), 0.15625, 1e-6);
    // Linear tangents reproduce the identity curve at every t.
    for t in [0.1_f32, 0.25, 0.5, 0.75, 0.9] {
        assert_approx(bezier_ease(t, 1.0, 1.0), t, 1e-6);
    }
    // Slopes and t are clamped into their documented range.
    assert_approx(bezier_ease(2.0, 1.0, 1.0), 1.0, 1e-6);
    assert_approx(bezier_ease(-1.0, 1.0, 1.0), 0.0, 1e-6);
    assert_approx(bezier_ease(0.5, 5.0, 5.0), 0.5, 1e-6);
    assert_eq!(bezier_ease(f32::NAN, 0.0, 0.0), 0.0);
    assert_approx(bezier_ease(0.5, 0.0, f32::NAN), 0.5, 1e-6);
}

#[test]
fn layer_bezier_keyframe_interpolation_timing() {
    let mut layer = MaskLayer::new(1, "Garbage 01", MaskKind::Garbage, 64, 64);
    let base = rect_shape(10.0, 10.0, 20.0, 20.0);
    layer.set_shape(0, base.clone());
    let mut moved = base.clone();
    moved.transform(20.0, 0.0, 1.0, 0.0, None);
    layer.set_shape(20, moved.clone());

    // Auto bezier timing is symmetric: the midpoint is the linear midpoint.
    let mid = layer.shape_at(10).expect("animated shape");
    assert_approx(mid.nodes[0].x, 20.0, 1e-4);
    // Off-centre the ease-in keeps the first key's position longer.
    layer.interp = InterpMode::Linear;
    let linear = layer.shape_at(4).expect("linear shape");
    assert_approx(linear.nodes[0].x, 14.0, 1e-4);
    layer.interp = InterpMode::Bezier;
    let eased = layer.shape_at(4).expect("eased shape");
    assert!(eased.nodes[0].x < linear.nodes[0].x);
    assert_approx(eased.nodes[0].x, 10.0 + 20.0 * 0.104, 1e-3);

    // Hold returns the previous key.
    layer.interp = InterpMode::Hold;
    let held = layer.shape_at(10).expect("held shape");
    assert_approx(held.nodes[0].x, 10.0, 1e-6);
    assert_eq!(layer.ease(0, 20, 0.9), 0.0);

    // Frames before the first / after the last key clamp to that key.
    layer.interp = InterpMode::Bezier;
    assert_eq!(layer.shape_at(-5).expect("clamped first"), base);
    assert_eq!(layer.shape_at(200).expect("clamped last"), moved);

    // Linear tangents on both keys reproduce the linear ramp.
    layer.set_tangent(0, 1.0);
    layer.set_tangent(20, 1.0);
    let tangent_linear = layer.shape_at(4).expect("tangent-linear shape");
    assert_approx(tangent_linear.nodes[0].x, linear.nodes[0].x, 1e-4);
    assert_eq!(layer.tangent_amount(0), 100);
    layer.auto_tangents();
    assert_eq!(layer.tangent_amount(0), 0);
    assert_eq!(layer.tangent(20), Tangent::AUTO);

    // Direct tangent pairs are preserved independently.
    layer.set_tangent_pair(0, Tangent::new(0.25, 0.75));
    assert_approx(layer.tangent(0).out_slope(), 0.25, 1e-6);
    assert_approx(layer.tangent(0).in_slope(), 0.75, 1e-6);
    assert!(layer.remove_tangent(0).is_some());
    assert_eq!(layer.tangent(0), Tangent::AUTO);
    assert_eq!(layer.ease(0, 20, 0.25), bezier_ease(0.25, 0.0, 0.0));
}

#[test]
fn layer_painted_keys_blend_hold_and_union_with_shapes() {
    let mut layer = animated_layer();
    layer.interp = InterpMode::Linear;
    let mid = layer.mask(10).expect("animated mask");
    assert_eq!(mid.get(5, 5), 128, "half-way blend of 255 and 0");
    assert_eq!(mid.get(25, 25), 128);
    assert_eq!(mid.get(15, 25), 0);

    // Hold keeps the previous key.
    layer.interp = InterpMode::Hold;
    let held = layer.mask(10).expect("held mask");
    assert_eq!(held.get(5, 5), 255);
    assert_eq!(held.get(25, 25), 0);

    // Before the first / after the last key the nearest key is held.
    layer.interp = InterpMode::Bezier;
    let before = layer.raw_mask(-5).expect("before first key");
    assert_eq!(before.get(5, 5), 255);
    let after = layer.raw_mask(200).expect("after last key");
    assert_eq!(after.get(25, 25), 255);

    // Painted and vector keys combine with max (v1 raw()).
    let mut union = MaskLayer::new(2, "Garbage 02", MaskKind::Garbage, 32, 32);
    let mut painted = Mask::new(32, 32);
    painted.set(2, 2, 255);
    union.set_mask(0, &painted);
    union.set_shape(0, rect_shape(20.0, 20.0, 30.0, 30.0));
    let raw = union.raw_mask(0).expect("union mask");
    assert_eq!(raw.get(2, 2), 255);
    assert_eq!(raw.get(25, 25), 255);
    assert_eq!(raw.get(0, 25), 0);

    // Softness is applied only by mask(), not by raw_mask().
    union.softness = 2;
    let hard = union.raw_mask(0).expect("hard mask");
    let soft = union.mask(0).expect("soft mask");
    assert_eq!(hard.get(20, 25), 255);
    let feathered = soft.get(20, 25);
    assert!(
        (1..255).contains(&feathered),
        "edge must be feathered, got {feathered}"
    );
}

#[test]
fn layer_edit_frame_and_toggle_key_follow_v1() {
    let mut layer = animated_layer();
    assert_eq!(layer.keyframes(), vec![0, 20]);
    assert!(layer.key_at(0));
    assert!(!layer.key_at(10));

    // Auto-key seeds a new key on the playhead from the interpolation.
    assert!(layer.auto_key);
    assert_eq!(layer.edit_frame(10), 10);
    assert!(layer.doc.has(10));
    let seeded = layer.doc.get(10).expect("seeded key").clone();
    assert_eq!(seeded.get(5, 5), 128, "seeded from the linear blend");

    // Without auto-key the previous key is edited.
    layer.auto_key = false;
    assert_eq!(layer.edit_frame(15), 10);
    // With no previous key a blank key is created on the playhead.
    let mut fresh = MaskLayer::new(1, "Garbage 01", MaskKind::Garbage, 8, 8);
    fresh.auto_key = false;
    assert_eq!(fresh.edit_frame(4), 4);
    assert!(fresh.doc.has(4));

    layer.clear_frame(10);
    assert!(!layer.key_at(10));
    assert!(layer.toggle_key(12));
    assert!(layer.key_at(12));
    assert!(!layer.toggle_key(12));
    assert!(!layer.key_at(12));

    // Vector keys toggle by interpolating the current shape.
    let mut vector = MaskLayer::new(1, "Garbage 01", MaskKind::Garbage, 16, 16);
    vector.set_shape(0, rect_shape(1.0, 1.0, 5.0, 5.0));
    vector.set_shape(8, rect_shape(3.0, 3.0, 7.0, 7.0));
    assert!(vector.toggle_key(4));
    assert!(vector.has_shape(4));
    let shape = vector.shapes.get(&4).expect("toggled shape");
    assert_approx(shape.nodes[0].x, 2.0, 1e-4);
    assert!(!vector.toggle_key(4));
    assert!(!vector.has_shape(4));
}

#[test]
fn layer_scene_scope_is_inclusive() {
    let mut layer = animated_layer();
    layer.set_scene_range(5, 15);
    assert!(layer.in_scene(5));
    assert!(layer.in_scene(15));
    assert!(!layer.in_scene(4));
    assert!(!layer.in_scene(16));
    assert!(layer.mask(4).is_none(), "outside the scene range");
    assert!(layer.mask(10).is_some());
    layer.clear_scene_range();
    assert!(layer.in_scene(-100));
    assert!(layer.in_scene(1_000_000));
}

// -- stacks ----------------------------------------------------------------

#[test]
fn mask_stack_composite_modes_and_opacity() {
    let base = filled(32, 32, 128);
    let patch = box_mask(32, 32, 10, 10, 20, 20);

    let mut stack = MaskStack::new(32, 32, MaskKind::Garbage);
    stack.add(0, Some(MaskMode::Add)).set_mask(0, &patch);
    let out = stack.apply(0, Some(&base)).expect("add composite");
    assert_eq!(out.get(15, 15), 255, "add raises the matte");
    assert_eq!(out.get(2, 2), 128);

    let mut stack = MaskStack::new(32, 32, MaskKind::Garbage);
    stack.add(0, Some(MaskMode::Subtract)).set_mask(0, &patch);
    let out = stack.apply(0, Some(&base)).expect("subtract composite");
    assert_eq!(out.get(15, 15), 0, "subtract clears inside");
    assert_eq!(out.get(2, 2), 128);

    let mut stack = MaskStack::new(32, 32, MaskKind::Garbage);
    stack.add(0, Some(MaskMode::Intersect)).set_mask(0, &patch);
    let out = stack.apply(0, Some(&base)).expect("intersect composite");
    assert_eq!(out.get(15, 15), 128, "intersect keeps inside");
    assert_eq!(out.get(2, 2), 0, "intersect clears outside");

    // Hidden layers are skipped.
    let mut stack = MaskStack::new(32, 32, MaskKind::Garbage);
    stack.add(0, Some(MaskMode::Subtract)).set_mask(0, &patch);
    stack.layers[0].visible = false;
    assert_eq!(stack.apply(0, Some(&base)), Some(base.clone()));

    // No base: an add layer starts from an empty matte.
    let mut stack = MaskStack::new(32, 32, MaskKind::Garbage);
    stack.add(0, Some(MaskMode::Add)).set_mask(0, &patch);
    let out = stack.apply(0, None).expect("empty base");
    assert_eq!(out.get(15, 15), 255);
    assert_eq!(out.get(2, 2), 0);
    assert_eq!(
        MaskStack::new(32, 32, MaskKind::Garbage).apply(0, None),
        None
    );

    // Despill-only modes fall back to subtract in the garbage dispatch (v1
    // `else` branch).
    let mut stack = MaskStack::new(32, 32, MaskKind::Garbage);
    stack.add(0, Some(MaskMode::Protect)).set_mask(0, &patch);
    assert_eq!(
        stack.apply(0, Some(&base)).expect("fallback").get(15, 15),
        0
    );

    // opacity is an app extension: 0.5 halves the mask before compositing.
    let mut stack = MaskStack::new(32, 32, MaskKind::Garbage);
    let layer = stack.add(0, Some(MaskMode::Subtract));
    layer.opacity = 0.5;
    layer.set_mask(0, &patch);
    let out = stack.apply(0, Some(&base)).expect("opacity composite");
    assert_eq!(out.get(15, 15), 127, "255*0.5=128, 255-128=127");
    let layer = stack.by_id_mut(1).expect("layer");
    layer.opacity = 0.0;
    let out = stack.apply(0, Some(&base)).expect("zero opacity");
    assert_eq!(out.get(15, 15), 128, "opacity 0 is a no-op");
}

#[test]
fn mask_stack_despill_weight_modes() {
    let patch = box_mask(32, 32, 10, 10, 20, 20);
    let mut stack = MaskStack::new(32, 32, MaskKind::Despill);
    stack.add(0, None).set_mask(0, &patch);
    let weight = stack.despill_weight(0).expect("protect weight");
    assert_eq!(weight.len(), 32 * 32);
    assert_approx(weight[15 * 32 + 15], 0.0, 1e-6);
    assert_approx(weight[2 * 32 + 2], 1.0, 1e-6);

    let mut confine = MaskStack::new(32, 32, MaskKind::Despill);
    let layer = confine.add(0, Some(MaskMode::Confine));
    layer.set_mask(0, &patch);
    let weight = confine.despill_weight(0).expect("confine weight");
    assert_approx(weight[15 * 32 + 15], 1.0, 1e-6);
    assert_approx(weight[2 * 32 + 2], 0.0, 1e-6);

    // confine ∪ then protect ∖: the hole inside the confine area is protected.
    let big = box_mask(32, 32, 5, 5, 25, 25);
    let hole = box_mask(32, 32, 12, 12, 18, 18);
    let mut combined = MaskStack::new(32, 32, MaskKind::Despill);
    combined.add(0, Some(MaskMode::Confine)).set_mask(0, &big);
    combined.add(0, None).set_mask(0, &hole);
    let weight = combined.despill_weight(0).expect("combined weight");
    assert_approx(weight[15 * 32 + 15], 0.0, 1e-6);
    assert_approx(weight[8 * 32 + 8], 1.0, 1e-6);
    assert_approx(weight[32 + 1], 0.0, 1e-6);

    assert_eq!(
        MaskStack::new(4, 4, MaskKind::Despill).despill_weight(0),
        None
    );

    // Animated linear protect: the weight moves strictly between 0 and 1.
    let mut animated = MaskStack::new(32, 32, MaskKind::Despill);
    let layer = animated.add(0, None);
    layer.set_mask(0, &box_mask(32, 32, 10, 10, 20, 20));
    layer.set_mask(20, &box_mask(32, 32, 0, 0, 5, 5));
    layer.interp = InterpMode::Linear;
    let weight = animated.despill_weight(10).expect("animated weight");
    let value = weight[15 * 32 + 15];
    assert!(0.0 < value && value < 1.0, "animated weight {value}");
}

#[test]
fn mask_stack_management_selection_and_ids() {
    let mut stack = MaskStack::new(16, 16, MaskKind::Garbage);
    let first_id = stack.add(0, None).layer_id;
    assert_eq!(first_id, 1);
    assert_eq!(
        stack.by_id(first_id).map(|layer| layer.name.clone()),
        Some("Garbage 01".to_string())
    );
    assert_eq!(stack.selected, Some(1), "new layers become selected");
    let second = stack.add(1, Some(MaskMode::Add));
    assert_eq!(second.layer_id, 2);
    assert_eq!(second.mode, MaskMode::Add);
    assert_eq!(second.name, "Garbage 02");
    assert_eq!(stack.selected, Some(2));
    assert_eq!(stack.for_scene(0).len(), 1);
    assert_eq!(stack.for_scene(1).len(), 1);
    assert!(stack.contains(1));
    assert_eq!(stack.index_of(2), Some(1));

    stack.reorder(&[2, 1]);
    assert_eq!(
        stack
            .layers
            .iter()
            .map(|layer| layer.layer_id)
            .collect::<Vec<_>>(),
        vec![2, 1]
    );
    // Unlisted layers keep their order at the end.
    stack.reorder(&[1]);
    assert_eq!(
        stack
            .layers
            .iter()
            .map(|layer| layer.layer_id)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    // Duplicate ids are ignored.
    stack.reorder(&[1, 1, 2]);
    assert_eq!(
        stack
            .layers
            .iter()
            .map(|layer| layer.layer_id)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );

    assert!(stack.move_layer(0, 1));
    assert_eq!(
        stack
            .layers
            .iter()
            .map(|layer| layer.layer_id)
            .collect::<Vec<_>>(),
        vec![2, 1]
    );
    assert!(!stack.move_layer(0, 9));

    assert!(!stack.any_visible());
    stack
        .by_id_mut(1)
        .expect("layer")
        .set_mask(0, &box_mask(16, 16, 1, 1, 4, 4));
    assert!(stack.any_visible());

    assert!(stack.remove(2));
    assert!(!stack.contains(2));
    assert_eq!(
        stack.selected, None,
        "removing the selected layer clears it"
    );
    assert!(stack.select(Some(1)));
    assert_eq!(stack.selected_layer().map(|layer| layer.layer_id), Some(1));
    assert!(!stack.select(Some(99)));
    assert!(stack.select(None));
    assert!(stack.selected_layer().is_none());

    // from_layers: stack kind and raster size win over loaded layer metadata.
    let mut bad_layer = MaskLayer::new(7, "loaded", MaskKind::Garbage, 4, 4);
    bad_layer.mode = MaskMode::Add;
    let mut despill_layer = MaskLayer::new(9, "confine", MaskKind::Despill, 4, 4);
    despill_layer.mode = MaskMode::Confine;
    let loaded = MaskStack::from_layers(8, 8, MaskKind::Despill, vec![bad_layer, despill_layer]);
    assert_eq!(loaded.kind, MaskKind::Despill);
    assert_eq!(loaded.selected, None);
    assert_eq!(loaded.next_id(), 10);
    assert_eq!(loaded.by_id(7).expect("loaded").kind, MaskKind::Despill);
    assert_eq!(
        loaded.by_id(7).expect("loaded").mode,
        MaskMode::Protect,
        "garbage mode falls back to the despill default"
    );
    assert_eq!(
        loaded.by_id(9).expect("loaded").mode,
        MaskMode::Confine,
        "valid mode is preserved"
    );
    assert_eq!(loaded.by_id(7).expect("loaded").width, 8);
    assert_eq!(loaded.by_id(7).expect("loaded").height, 8);

    // Deterministic overlay palette.
    assert_eq!(suggest_color(1), suggest_color(1));
    assert_ne!(suggest_color(1), suggest_color(2));
    assert_eq!(suggest_color(1).len(), 7);
    assert!(suggest_color(1).starts_with('#'));
}

#[test]
fn copy_layer_for_scene_preserves_animation_offset() {
    let mut layer = MaskLayer::new(3, "Garbage 03", MaskKind::Garbage, 16, 16);
    layer.set_scene_range(0, 23);
    layer.softness = 7;
    layer.opacity = 0.5;
    layer.set_shape(0, rect_shape(1.0, 1.0, 5.0, 5.0));
    layer.set_shape(4, rect_shape(2.0, 2.0, 6.0, 6.0));
    layer.set_shape(10, rect_shape(3.0, 3.0, 7.0, 7.0));
    let mut painted = Mask::new(16, 16);
    painted.set(2, 2, 255);
    layer.set_mask(4, &painted);
    layer.set_tangent_pair(4, Tangent::new(0.3, 0.7));

    let copy = layer.copy_for_scene(9, 1, 100, 119);
    assert_eq!(copy.layer_id, 9);
    assert_eq!(copy.scene_index, 1);
    assert_eq!(copy.scene_start, 100);
    assert_eq!(copy.scene_end, 119);
    assert_eq!(copy.name, "Garbage 03");
    assert_eq!(copy.kind, MaskKind::Garbage);
    assert_eq!(copy.softness, 7);
    assert_eq!(copy.opacity, 0.5);
    assert_eq!(copy.shape_keys(), vec![100, 104, 110]);
    assert_eq!(copy.tangent(104), Tangent::new(0.3, 0.7));
    assert!(copy.doc.has(104));
    assert!(!copy.doc.has(4));
    assert_eq!(copy.shapes.get(&100), Some(&rect_shape(1.0, 1.0, 5.0, 5.0)));
    assert!(copy.in_scene(100));
    assert!(copy.in_scene(119));
    assert!(!copy.in_scene(99));

    // The source layer is untouched.
    assert_eq!(layer.shape_keys(), vec![0, 4, 10]);
    assert!(layer.doc.has(4));

    // An unscoped source keeps absolute keys (offset 0).
    let mut absolute = MaskLayer::new(1, "Garbage 01", MaskKind::Garbage, 8, 8);
    absolute.set_shape(7, rect_shape(1.0, 1.0, 4.0, 4.0));
    let copy = absolute.copy_for_scene(2, 5, 50, 60);
    assert_eq!(copy.shape_keys(), vec![7]);
    assert_eq!(copy.scene_start, 50);
}

#[test]
fn copy_layers_from_scene_appends_and_assigns_ids() {
    let mut stack = MaskStack::new(16, 16, MaskKind::Garbage);
    let first = stack.add(0, None);
    first.set_scene_range(0, 23);
    first.set_shape(0, rect_shape(1.0, 1.0, 5.0, 5.0));
    let second = stack.add(0, Some(MaskMode::Add));
    second.set_scene_range(0, 23);
    second.set_mask(2, &box_mask(16, 16, 1, 1, 4, 4));
    stack
        .add(1, Some(MaskMode::Confine))
        .set_scene_range(24, 47);
    assert_eq!(stack.next_id(), 4);

    let copies = stack.copy_layers_from_scene(0, 1, 24, 47);
    assert_eq!(copies, vec![4, 5]);
    assert_eq!(stack.layers.len(), 5);
    assert_eq!(stack.next_id(), 6);
    assert_eq!(
        stack.selected,
        Some(3),
        "selection is unchanged by the copy"
    );

    let copy = stack.by_id(4).expect("first copy");
    assert_eq!(copy.scene_index, 1);
    assert_eq!(copy.scene_start, 24);
    assert_eq!(copy.scene_end, 47);
    assert_eq!(copy.shape_keys(), vec![24]);
    assert_eq!(copy.mode, MaskMode::Subtract);
    let copy = stack.by_id(5).expect("second copy");
    assert_eq!(copy.mode, MaskMode::Add);
    assert!(copy.doc.has(26));
    assert_eq!(copy.layer_id, 5);
}

// -- serialisation ---------------------------------------------------------

#[test]
fn layer_json_round_trip_matches_v1_fields() {
    let mut layer = MaskLayer::new(1, "Garbage 01", MaskKind::Garbage, 96, 64);
    layer.mode = MaskMode::Subtract;
    layer.softness = 7;
    layer.opacity = 0.5;
    layer.auto_key = false;
    layer.set_scene_range(0, 23);
    layer.set_tangent_pair(0, Tangent::new(0.25, 0.75));
    let mut shape = rect_shape(10.0, 10.0, 20.0, 20.0);
    shape.nodes[1].out_dx = 4.0;
    shape.nodes[1].out_dy = -3.0;
    shape.nodes[1].in_dx = -4.0;
    shape.nodes[1].in_dy = 3.0;
    layer.set_shape(0, shape.clone());

    let json = serde_json::to_string(&layer).expect("serialise layer");
    let value: serde_json::Value = serde_json::from_str(&json).expect("parse layer JSON");
    assert_eq!(value["id"], json!(1));
    assert_eq!(value["name"], json!("Garbage 01"));
    assert_eq!(value["kind"], json!("garbage"));
    assert_eq!(value["mode"], json!("subtract"));
    assert_eq!(value["visible"], json!(true));
    assert_eq!(value["opacity"], json!(0.5));
    assert_eq!(value["softness"], json!(7));
    assert_eq!(value["scene"], json!(0));
    assert_eq!(value["interp"], json!("bezier"));
    assert_eq!(value["auto_key"], json!(false));
    assert_eq!(value["width"], json!(96));
    assert_eq!(value["height"], json!(64));
    // v1 stores tangents as a frame -> [out, in] array.
    assert_eq!(value["tangents"]["0"], json!([0.25, 0.75]));
    let node = &value["shapes"]["0"]["nodes"][1];
    assert_eq!(node["x"], json!(20.0));
    assert_eq!(node["y"], json!(10.0));
    assert_eq!(node["out_dx"], json!(4.0));
    assert_eq!(node["in_dy"], json!(3.0));
    assert_eq!(node["linked"], json!(true));
    // Raster keys are runtime state, never in layer JSON.
    assert!(!json.contains("\"doc\""));

    let mut restored: MaskLayer = serde_json::from_str(&json).expect("deserialise layer");
    restored.set_raster_size(restored.width, restored.height);
    assert_eq!(restored, layer);
    assert_eq!(restored.shapes.get(&0), Some(&shape));

    // Missing fields fall back to the v1 defaults.
    let defaults: MaskLayer = serde_json::from_str("{}").expect("empty layer");
    assert_eq!(defaults.kind, MaskKind::Garbage);
    assert_eq!(defaults.mode, MaskMode::Subtract);
    assert!(defaults.visible);
    assert_eq!(defaults.opacity, 1.0);
    assert!(defaults.auto_key);
    assert_eq!(defaults.interp, InterpMode::Bezier);
    assert_eq!(defaults.scene_start, -1);
    assert_eq!(defaults.scene_end, -1);
}

#[test]
fn v1_on_disk_layer_json_loads_with_defaults() {
    // Shape of a v1 `project.json` mask layer (v1 groups layers by kind, so the
    // layer object has no `kind`; it also has a `frames` PNG-name list and no
    // `opacity`/`width`/`height`). Unknown fields are ignored.
    let v1 = r##"{
        "id": 3,
        "name": "Despill 03",
        "mode": "confine",
        "visible": true,
        "color": "#123456",
        "softness": 4,
        "scene": 2,
        "scene_start": 48,
        "scene_end": 95,
        "interp": "linear",
        "auto_key": false,
        "tangents": {"48": [0.5, 0.5]},
        "shapes": {"48": {"closed": true, "nodes": [
            {"x": 10.0, "y": 20.0, "out_dx": 1.5, "out_dy": 0.0,
             "in_dx": -1.5, "in_dy": 0.0, "linked": true},
            {"x": 30.0, "y": 20.0}
        ]}},
        "frames": ["masks/despill_03_000048.png"]
    }"##;
    let layer: MaskLayer = serde_json::from_str(v1).expect("v1 layer JSON");
    assert_eq!(layer.layer_id, 3);
    assert_eq!(layer.name, "Despill 03");
    assert_eq!(layer.mode, MaskMode::Confine);
    assert_eq!(layer.softness, 4);
    assert_eq!(layer.scene_index, 2);
    assert_eq!(layer.scene_start, 48);
    assert_eq!(layer.scene_end, 95);
    assert_eq!(layer.interp, InterpMode::Linear);
    assert!(!layer.auto_key);
    assert_eq!(layer.tangent(48), Tangent::new(0.5, 0.5));
    let shape = layer.shapes.get(&48).expect("v1 vector key");
    assert!(shape.closed);
    assert_eq!(shape.len(), 2);
    assert_approx(shape.nodes[1].out_dx, 0.0, 1e-6);
    assert!(
        shape.nodes[1].linked,
        "missing node fields use node defaults"
    );
    // Fields absent from v1 layer objects use the app-schema defaults.
    assert_eq!(layer.kind, MaskKind::Garbage);
    assert_eq!(layer.opacity, 1.0);
    assert_eq!(layer.width, 0);
    assert_eq!(layer.height, 0);
    assert!(layer.doc.frames.is_empty());

    // The kind is restored by the stack, which is authoritative on disk.
    let stack = MaskStack::from_layers(96, 64, MaskKind::Despill, vec![layer]);
    let restored = stack.by_id(3).expect("slot restored");
    assert_eq!(restored.kind, MaskKind::Despill);
    assert_eq!(restored.mode, MaskMode::Confine);
    assert_eq!(restored.width, 96);
    assert_eq!(restored.mode.label(), "Only here");
}

#[test]
fn enum_wire_names_and_mode_lists_match_v1() {
    assert_eq!(
        serde_json::to_string(&MaskKind::Garbage).expect("kind"),
        "\"garbage\""
    );
    assert_eq!(
        serde_json::to_string(&MaskKind::Despill).expect("kind"),
        "\"despill\""
    );
    assert_eq!(
        serde_json::to_string(&MaskMode::Intersect).expect("mode"),
        "\"intersect\""
    );
    assert_eq!(
        serde_json::to_string(&MaskMode::Protect).expect("mode"),
        "\"protect\""
    );
    assert_eq!(
        serde_json::to_string(&InterpMode::Hold).expect("interp"),
        "\"hold\""
    );
    assert_eq!(
        GARBAGE_MODES,
        [MaskMode::Add, MaskMode::Subtract, MaskMode::Intersect]
    );
    assert_eq!(DESPILL_MODES, [MaskMode::Protect, MaskMode::Confine]);
    assert_eq!(
        INTERP_MODES,
        [InterpMode::Bezier, InterpMode::Linear, InterpMode::Hold]
    );
    assert_eq!(MaskKind::Garbage.default_mode(), MaskMode::Subtract);
    assert_eq!(MaskKind::Despill.default_mode(), MaskMode::Protect);
    assert!(MaskMode::Add.is_valid_for(MaskKind::Garbage));
    assert!(!MaskMode::Add.is_valid_for(MaskKind::Despill));
    assert_eq!(MaskMode::Add.label(), "Add");
    assert_eq!(InterpMode::Bezier.label(), "Bezier (auto tangents)");
    assert!(!ControlKind::Anchor.is_handle());
    assert!(ControlKind::Out.is_handle());
}

#[test]
fn add_copy_pastes_a_layer_into_another_scene_and_selects_it() {
    let mut stack = MaskStack::new(16, 16, MaskKind::Garbage);
    let source_id = stack.add(0, None).layer_id;
    {
        let source = stack.by_id_mut(source_id).unwrap();
        source.set_scene_range(0, 9);
        source.set_shape(0, rect_shape(1.0, 1.0, 5.0, 5.0));
    }
    let source_snapshot = stack.by_id(source_id).unwrap().clone();
    let id = stack.add_copy(&source_snapshot, 1, 20, 29);
    assert_eq!(id, 2);
    assert_eq!(stack.selected, Some(id));
    let copy = stack.by_id(id).unwrap();
    assert_eq!(copy.scene_index, 1);
    assert_eq!(copy.scene_start, 20);
    assert_eq!(copy.scene_end, 29);
    assert!(
        copy.shapes.keys().all(|frame| (20..=29).contains(frame)),
        "pasted keys shift into the target scene: {:?}",
        copy.shapes.keys().collect::<Vec<_>>()
    );
}

/// Timing probe for the feather blur (run with `--release --ignored --nocapture`).
#[test]
#[ignore]
fn soften_timing_probe_1080p() {
    let mut mask = Mask::new(1920, 1080);
    for y in 300..800 {
        for x in 600..1400 {
            mask.as_mut_slice()[y * 1920 + x] = 255;
        }
    }
    for softness in [10u32, 40, 120] {
        let started = std::time::Instant::now();
        let out = soften(&mask, softness);
        println!(
            "soften {softness} px: {:.3} s ({})",
            started.elapsed().as_secs_f64(),
            out.as_slice()[540 * 1920 + 600]
        );
    }
}

/// The optimised `soften` must match the straightforward per-pixel v1 port exactly.
#[test]
fn soften_matches_the_reference_blur_bit_for_bit() {
    fn reference(mask: &Mask, softness: u32) -> Vec<u8> {
        let kernel_size = softness.max(3) | 1;
        let radius = (kernel_size / 2) as i32;
        let sigma = f32::max(0.1, softness as f32 / 2.5);
        let mut kernel: Vec<f32> = (-radius..=radius)
            .map(|o| (-0.5 * (o as f32 / sigma).powi(2)).exp())
            .collect();
        let sum: f32 = kernel.iter().sum();
        kernel.iter_mut().for_each(|w| *w /= sum);
        let reflect = |mut i: i32, len: i32| -> i32 {
            if len <= 1 {
                return 0;
            }
            while i < 0 || i >= len {
                i = if i < 0 { -i } else { 2 * (len - 1) - i };
            }
            i
        };
        let (w, h) = (mask.width() as i32, mask.height() as i32);
        let data = mask.as_slice();
        let mut horizontal = vec![0.0f32; data.len()];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0f32;
                for (k, weight) in kernel.iter().enumerate() {
                    s += f32::from(data[(y * w + reflect(x + k as i32 - radius, w)) as usize])
                        * weight;
                }
                horizontal[(y * w + x) as usize] = s;
            }
        }
        let mut out = vec![0u8; data.len()];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0f32;
                for (k, weight) in kernel.iter().enumerate() {
                    s += horizontal[(reflect(y + k as i32 - radius, h) * w + x) as usize] * weight;
                }
                out[(y * w + x) as usize] = s.round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for (width, height) in [(37u32, 23u32), (5, 9), (64, 3)] {
        let mut mask = Mask::new(width, height);
        for value in mask.as_mut_slice() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *value = if state.is_multiple_of(3) {
                255
            } else {
                (state >> 56) as u8
            };
        }
        for softness in [1u32, 4, 7, 12, 40] {
            assert_eq!(
                soften(&mask, softness).as_slice(),
                reference(&mask, softness).as_slice(),
                "{width}x{height} softness {softness}"
            );
        }
    }
    // Sparse masks (the bounding-box path) touching each border and the empty mask.
    for (x, y) in [(0u32, 10u32), (47, 20), (20, 0), (25, 31), (24, 15)] {
        let mut mask = Mask::new(48, 32);
        for dy in 0..3 {
            for dx in 0..4 {
                let (px, py) = ((x + dx).min(47), (y + dy).min(31));
                mask.as_mut_slice()[(py * 48 + px) as usize] = 200 + dx as u8;
            }
        }
        for softness in [3u32, 9, 40] {
            assert_eq!(
                soften(&mask, softness).as_slice(),
                reference(&mask, softness).as_slice(),
                "blob at {x},{y} softness {softness}"
            );
        }
    }
    let empty = Mask::new(16, 8);
    assert_eq!(
        soften(&empty, 9).as_slice(),
        reference(&empty, 9).as_slice()
    );
}
