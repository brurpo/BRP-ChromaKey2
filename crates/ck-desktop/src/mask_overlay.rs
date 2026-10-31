//! Viewer overlay geometry and interaction state for bezier garbage/despill
//! masks. Pure helpers; the Studio owns the rendering and event dispatch.

use ck_core::mask_layers::{BezierShape, ControlKind, HandleSide};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};

/// Overlay accent for mask outlines and knots.
pub const MASK_ACCENT: Color32 = Color32::from_rgb(88, 182, 242);
/// Tangent handle colour.
pub const HANDLE_COLOR: Color32 = Color32::from_rgb(196, 206, 220);
/// Selected knot colour.
pub const SELECTED_COLOR: Color32 = Color32::from_rgb(251, 191, 36);

pub const KNOT_RADIUS: f32 = 3.5;
pub const SELECTED_KNOT_RADIUS: f32 = 5.0;
pub const HANDLE_RADIUS: f32 = 3.0;
pub const HIT_RADIUS: f32 = 9.0;
pub const ROTATE_HANDLE_OFFSET: f32 = 34.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskControl {
    Anchor(usize),
    In(usize),
    Out(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoDrag {
    Move,
    Scale,
    Rotate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoHit {
    Move,
    Scale(usize),
    Rotate,
}

#[derive(Clone, Debug)]
pub enum MaskDragKind {
    Control(MaskControl, bool),
    Gizmo(GizmoDrag),
    /// Press-and-drag on empty canvas: add a node whose tangents follow the drag.
    CreateNode,
}

/// One in-flight mask edit drag, in source-pixel coordinates.
#[derive(Clone, Debug)]
pub struct MaskDrag {
    pub kind: MaskDragKind,
    pub layer_id: u32,
    pub frame: i64,
    pub start_shape: BezierShape,
    pub start_pointer: Pos2,
    pub start_angle: f32,
}

/// Window-only mask overlay interaction state.
#[derive(Clone, Debug, Default)]
pub struct MaskEditState {
    pub drag: Option<MaskDrag>,
    pub gizmo_open: bool,
    pub gizmo_angle: f32,
    pub selected_knot: Option<usize>,
    /// Timeline mask keyframes selected for moving.
    pub timeline_selected: std::collections::BTreeSet<i64>,
    /// In-flight timeline keyframe gesture.
    pub timeline_drag: Option<crate::timeline::MaskTimelineDrag>,
}

/// Source pixels per logical viewer point (uniform; the viewer keeps aspect).
pub fn source_per_screen(image_rect: Rect, source_size: Vec2) -> f32 {
    source_size.x.max(1.0) / image_rect.width().max(1.0)
}

pub fn source_to_screen(image_rect: Rect, source_size: Vec2, point: Pos2) -> Pos2 {
    image_rect.min
        + Vec2::new(
            point.x / source_size.x.max(1.0) * image_rect.width(),
            point.y / source_size.y.max(1.0) * image_rect.height(),
        )
}

pub fn screen_to_source(image_rect: Rect, source_size: Vec2, point: Pos2) -> Pos2 {
    let offset = point - image_rect.min;
    Pos2::new(
        offset.x / image_rect.width().max(1.0) * source_size.x,
        offset.y / image_rect.height().max(1.0) * source_size.y,
    )
}

/// Axis-aligned source-space bounds of a shape as an egui rectangle.
pub fn shape_bounds(shape: &BezierShape) -> Rect {
    let bounds = shape.bounds(false);
    Rect::from_min_max(
        Pos2::new(bounds.min_x, bounds.min_y),
        Pos2::new(bounds.max_x, bounds.max_y),
    )
}

/// Control under the cursor (anchors and tangent handles), in source space.
pub fn control_at(shape: &BezierShape, point: Pos2, tolerance: f32) -> Option<MaskControl> {
    let hit = shape.nearest_control(point.x, point.y, tolerance.max(0.5))?;
    Some(match hit.kind {
        ControlKind::Anchor => MaskControl::Anchor(hit.index),
        ControlKind::In => MaskControl::In(hit.index),
        ControlKind::Out => MaskControl::Out(hit.index),
    })
}

/// Outline segment under the cursor as `(segment, t)`, in source space.
pub fn segment_at(shape: &BezierShape, point: Pos2, tolerance: f32) -> Option<(usize, f32)> {
    let hit = shape.nearest_segment(point.x, point.y)?;
    (hit.distance <= tolerance).then_some((hit.segment, hit.t))
}

fn rotate_point(point: Pos2, centre: Pos2, angle: f32) -> Pos2 {
    let (sin, cos) = angle.sin_cos();
    let offset = point - centre;
    centre
        + Vec2::new(
            offset.x * cos - offset.y * sin,
            offset.x * sin + offset.y * cos,
        )
}

/// Four rotated corner handles of an axis-aligned bounds rectangle.
pub fn gizmo_corners(bounds: Rect, angle: f32) -> [Pos2; 4] {
    let centre = bounds.center();
    [
        rotate_point(bounds.left_top(), centre, angle),
        rotate_point(bounds.right_top(), centre, angle),
        rotate_point(bounds.right_bottom(), centre, angle),
        rotate_point(bounds.left_bottom(), centre, angle),
    ]
}

/// Rotation handle position above the rotated bounds.
pub fn rotate_handle(bounds: Rect, angle: f32) -> Pos2 {
    let centre = bounds.center();
    rotate_point(
        Pos2::new(centre.x, bounds.top() - ROTATE_HANDLE_OFFSET),
        centre,
        angle,
    )
}

fn point_in_rotated_rect(point: Pos2, bounds: Rect, angle: f32) -> bool {
    let centre = bounds.center();
    let local = rotate_point(point, centre, -angle);
    bounds.contains(local)
}

/// Which gizmo part the cursor is over, in source space.
pub fn gizmo_hit(bounds: Rect, angle: f32, point: Pos2, tolerance: f32) -> Option<GizmoHit> {
    if bounds.width() <= 0.0 || bounds.height() <= 0.0 {
        return None;
    }
    if (rotate_handle(bounds, angle) - point).length() <= tolerance.max(0.5) + HANDLE_RADIUS {
        return Some(GizmoHit::Rotate);
    }
    for (index, corner) in gizmo_corners(bounds, angle).into_iter().enumerate() {
        if (corner - point).length() <= tolerance.max(0.5) + HANDLE_RADIUS {
            return Some(GizmoHit::Scale(index));
        }
    }
    point_in_rotated_rect(point, bounds, angle).then_some(GizmoHit::Move)
}

/// Shape after one drag update at `pointer` (source space).
pub fn drag_shape(drag: &MaskDrag, pointer: Pos2) -> BezierShape {
    let mut shape = drag.start_shape.clone();
    match drag.kind {
        MaskDragKind::Control(MaskControl::Anchor(index), smooth) => {
            if smooth {
                if let Some(node) = shape.nodes.get_mut(index) {
                    let dx = pointer.x - node.x;
                    let dy = pointer.y - node.y;
                    node.out_dx = dx;
                    node.out_dy = dy;
                    node.in_dx = -dx;
                    node.in_dy = -dy;
                }
            } else {
                shape.set_anchor(index, pointer.x, pointer.y);
            }
        }
        MaskDragKind::Control(MaskControl::In(index), _) => {
            shape.set_handle(index, HandleSide::In, pointer.x, pointer.y, None);
        }
        MaskDragKind::Control(MaskControl::Out(index), _) => {
            shape.set_handle(index, HandleSide::Out, pointer.x, pointer.y, None);
        }
        MaskDragKind::CreateNode => {
            let out_dx = pointer.x - drag.start_pointer.x;
            let out_dy = pointer.y - drag.start_pointer.y;
            shape.add_node(
                drag.start_pointer.x,
                drag.start_pointer.y,
                out_dx,
                out_dy,
                true,
            );
        }
        MaskDragKind::Gizmo(GizmoDrag::Move) => {
            let delta = pointer - drag.start_pointer;
            shape.transform(delta.x, delta.y, 1.0, 0.0, None);
        }
        MaskDragKind::Gizmo(GizmoDrag::Scale) => {
            let (cx, cy) = drag.start_shape.bounds(false).centre();
            let centre = Pos2::new(cx, cy);
            let base = (drag.start_pointer - centre).length().max(0.001);
            let factor = ((pointer - centre).length() / base).clamp(0.05, 20.0);
            shape.transform(0.0, 0.0, factor, 0.0, Some((cx, cy)));
        }
        MaskDragKind::Gizmo(GizmoDrag::Rotate) => {
            let (cx, cy) = drag.start_shape.bounds(false).centre();
            let centre = Pos2::new(cx, cy);
            let start = (drag.start_pointer.y - centre.y).atan2(drag.start_pointer.x - centre.x);
            let current = (pointer.y - centre.y).atan2(pointer.x - centre.x);
            shape.transform(0.0, 0.0, 1.0, current - start, Some((cx, cy)));
        }
    }
    shape
}

/// Draw the mask outline, knots/tangents and an optional transform gizmo.
pub fn draw(
    painter: &egui::Painter,
    image_rect: Rect,
    source_size: Vec2,
    shape: &BezierShape,
    selected_knot: Option<usize>,
    gizmo: Option<(Rect, f32)>,
) {
    if !shape.nodes.is_empty() {
        let samples = shape.sample(18);
        if samples.len() >= 2 {
            let points: Vec<Pos2> = samples
                .iter()
                .map(|&(x, y)| source_to_screen(image_rect, source_size, Pos2::new(x, y)))
                .collect();
            painter.add(egui::Shape::line(
                points.clone(),
                Stroke::new(1.5_f32, MASK_ACCENT),
            ));
            if shape.closed && points.len() >= 3 {
                painter.add(egui::Shape::convex_polygon(
                    points,
                    Color32::from_rgba_unmultiplied(88, 182, 242, 24),
                    Stroke::NONE,
                ));
            }
        }
    }
    for (index, node) in shape.nodes.iter().enumerate() {
        let position = source_to_screen(image_rect, source_size, Pos2::new(node.x, node.y));
        let selected = selected_knot == Some(index);
        let radius = if selected {
            SELECTED_KNOT_RADIUS
        } else {
            KNOT_RADIUS
        };
        painter.circle_filled(position, radius, Color32::from_black_alpha(190));
        painter.circle_stroke(
            position,
            radius,
            Stroke::new(
                1.4_f32,
                if selected {
                    SELECTED_COLOR
                } else {
                    MASK_ACCENT
                },
            ),
        );
        if selected {
            for (handle, side) in [
                (node.in_point(), HandleSide::In),
                (node.out_point(), HandleSide::Out),
            ] {
                let present = match side {
                    HandleSide::In => node.has_in(),
                    HandleSide::Out => node.has_out(),
                };
                if !present {
                    continue;
                }
                let handle_position =
                    source_to_screen(image_rect, source_size, Pos2::new(handle.0, handle.1));
                painter.line_segment(
                    [position, handle_position],
                    Stroke::new(1.0_f32, Color32::from_white_alpha(120)),
                );
                painter.circle_filled(handle_position, HANDLE_RADIUS, HANDLE_COLOR);
                painter.circle_stroke(
                    handle_position,
                    HANDLE_RADIUS,
                    Stroke::new(1.0_f32, Color32::from_black_alpha(180)),
                );
            }
        }
    }
    if let Some((bounds, angle)) = gizmo {
        let corners = gizmo_corners(bounds, angle);
        let screen: Vec<Pos2> = corners
            .iter()
            .map(|&point| source_to_screen(image_rect, source_size, point))
            .collect();
        for index in 0..4 {
            painter.line_segment(
                [screen[index], screen[(index + 1) % 4]],
                Stroke::new(1.2_f32, SELECTED_COLOR),
            );
        }
        for corner in &screen {
            painter.rect_filled(
                Rect::from_center_size(*corner, Vec2::splat(7.0)),
                1.0,
                SELECTED_COLOR,
            );
        }
        let rotate_top = source_to_screen(
            image_rect,
            source_size,
            Pos2::new(bounds.center().x, bounds.top()),
        );
        let rotate_handle = source_to_screen(image_rect, source_size, rotate_handle(bounds, angle));
        painter.line_segment(
            [rotate_top, rotate_handle],
            Stroke::new(1.2_f32, SELECTED_COLOR),
        );
        painter.circle_filled(rotate_handle, 5.0, SELECTED_COLOR);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ck_core::mask_layers::BezierNode;

    fn square() -> BezierShape {
        BezierShape {
            nodes: vec![
                BezierNode::new(10.0, 10.0),
                BezierNode::new(110.0, 10.0),
                BezierNode::new(110.0, 110.0),
                BezierNode::new(10.0, 110.0),
            ],
            closed: true,
        }
    }

    fn drag(kind: MaskDragKind, start: Pos2, shape: BezierShape) -> MaskDrag {
        MaskDrag {
            kind,
            layer_id: 1,
            frame: 0,
            start_shape: shape,
            start_pointer: start,
            start_angle: 0.0,
        }
    }

    #[test]
    fn coordinate_mapping_round_trips() {
        let image = Rect::from_min_size(Pos2::new(20.0, 30.0), Vec2::new(960.0, 540.0));
        let source = Vec2::new(1920.0, 1080.0);
        let point = Pos2::new(640.0, 400.0);
        let screen = source_to_screen(image, source, point);
        let back = screen_to_source(image, source, screen);
        assert!((back - point).length() < 0.001);
        assert!((source_per_screen(image, source) - 2.0).abs() < 0.001);
    }

    #[test]
    fn anchor_and_tangent_hit_testing_uses_source_units() {
        let shape = square();
        assert_eq!(
            control_at(&shape, Pos2::new(11.0, 9.0), 4.0),
            Some(MaskControl::Anchor(0))
        );
        assert_eq!(control_at(&shape, Pos2::new(60.0, 60.0), 4.0), None);
    }

    #[test]
    fn segment_hit_testing_returns_insert_position() {
        let shape = square();
        let (segment, t) = segment_at(&shape, Pos2::new(60.0, 10.0), 4.0).expect("segment");
        assert_eq!(segment, 0);
        assert!((0.3..0.7).contains(&t));
    }

    #[test]
    fn gizmo_hit_testing_finds_corners_rotate_handle_and_inside() {
        let bounds = Rect::from_min_max(Pos2::new(10.0, 10.0), Pos2::new(110.0, 110.0));
        assert_eq!(
            gizmo_hit(bounds, 0.0, Pos2::new(10.0, 10.0), 6.0),
            Some(GizmoHit::Scale(0))
        );
        assert_eq!(
            gizmo_hit(bounds, 0.0, Pos2::new(60.0, 60.0), 6.0),
            Some(GizmoHit::Move)
        );
        let handle = rotate_handle(bounds, 0.0);
        assert_eq!(gizmo_hit(bounds, 0.0, handle, 6.0), Some(GizmoHit::Rotate));
    }

    #[test]
    fn gizmo_corners_and_rotate_handle_follow_the_rotation() {
        let bounds = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(100.0, 50.0));
        let centred = Rect::from_min_max(Pos2::new(-50.0, -25.0), Pos2::new(50.0, 25.0));
        let corners = gizmo_corners(centred, std::f32::consts::FRAC_PI_2);
        assert!((corners[0] - Pos2::new(25.0, -50.0)).length() < 0.01);
        let handle = rotate_handle(bounds, 0.0);
        assert!((handle - Pos2::new(50.0, -34.0)).length() < 0.01);
    }

    #[test]
    fn drag_move_translates_every_node() {
        let shape = square();
        let drag = drag(
            MaskDragKind::Gizmo(GizmoDrag::Move),
            Pos2::new(60.0, 60.0),
            shape.clone(),
        );
        let moved = drag_shape(&drag, Pos2::new(80.0, 50.0));
        for (before, after) in shape.nodes.iter().zip(&moved.nodes) {
            assert!((after.x - (before.x + 20.0)).abs() < 0.001);
            assert!((after.y - (before.y - 10.0)).abs() < 0.001);
        }
    }

    #[test]
    fn drag_scale_scales_around_the_bounds_centre() {
        let shape = square();
        let (cx, cy) = shape.bounds(false).centre();
        let centre = Pos2::new(cx, cy);
        let drag = drag(
            MaskDragKind::Gizmo(GizmoDrag::Scale),
            Pos2::new(centre.x + 50.0, centre.y),
            shape.clone(),
        );
        let scaled = drag_shape(&drag, Pos2::new(centre.x + 100.0, centre.y));
        let after = scaled.bounds(false);
        assert!((after.width() - 200.0).abs() < 0.5, "{after:?}");
        assert!((Pos2::new(after.centre().0, after.centre().1) - centre).length() < 0.5);
    }

    #[test]
    fn drag_rotate_rotates_around_the_bounds_centre() {
        let shape = square();
        let (cx, cy) = shape.bounds(false).centre();
        let centre = Pos2::new(cx, cy);
        let drag = drag(
            MaskDragKind::Gizmo(GizmoDrag::Rotate),
            Pos2::new(centre.x + 50.0, centre.y),
            shape.clone(),
        );
        let rotated = drag_shape(&drag, Pos2::new(centre.x, centre.y + 50.0));
        let after = rotated.bounds(false);
        assert!((after.width() - 100.0).abs() < 0.5);
        assert!((Pos2::new(after.centre().0, after.centre().1) - centre).length() < 0.5);
    }

    #[test]
    fn control_drag_moves_only_the_grabbed_anchor() {
        let shape = square();
        let drag = drag(
            MaskDragKind::Control(MaskControl::Anchor(2), false),
            Pos2::new(110.0, 110.0),
            shape.clone(),
        );
        let moved = drag_shape(&drag, Pos2::new(150.0, 160.0));
        assert!((moved.nodes[2].x - 150.0).abs() < 0.001);
        assert!((moved.nodes[2].y - 160.0).abs() < 0.001);
        assert!((moved.nodes[1].x - 110.0).abs() < 0.001);
    }

    #[test]
    fn create_node_drag_adds_a_smooth_node_with_follow_tangents() {
        let mut shape = square();
        shape.closed = false;
        let start = Pos2::new(160.0, 60.0);
        let drag = MaskDrag {
            kind: MaskDragKind::CreateNode,
            layer_id: 1,
            frame: 0,
            start_shape: shape.clone(),
            start_pointer: start,
            start_angle: 0.0,
        };
        let created = drag_shape(&drag, Pos2::new(190.0, 80.0));
        assert_eq!(created.nodes.len(), shape.nodes.len() + 1);
        let node = created.nodes.last().expect("created node");
        assert!((node.x - start.x).abs() < 0.001);
        assert!((node.y - start.y).abs() < 0.001);
        assert!((node.out_dx - 30.0).abs() < 0.001);
        assert!((node.out_dy - 20.0).abs() < 0.001);
        assert!((node.in_dx + 30.0).abs() < 0.001);
        assert!((node.in_dy + 20.0).abs() < 0.001);
    }

    #[test]
    fn source_mapping_allows_points_outside_the_video_area() {
        let image = Rect::from_min_size(Pos2::new(10.0, 20.0), Vec2::new(100.0, 50.0));
        let size = Vec2::new(200.0, 100.0);
        let outside_left_top = screen_to_source(image, size, Pos2::new(0.0, 0.0));
        assert!(outside_left_top.x < 0.0);
        assert!(outside_left_top.y < 0.0);
        let outside_right_bottom = screen_to_source(image, size, Pos2::new(210.0, 90.0));
        assert!(outside_right_bottom.x > size.x);
        assert!(outside_right_bottom.y > size.y);
        let round_trip = source_to_screen(image, size, outside_left_top);
        assert!((round_trip - Pos2::new(0.0, 0.0)).length() < 0.01);
    }
}
