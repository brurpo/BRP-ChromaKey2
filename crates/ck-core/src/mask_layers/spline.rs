//! Bezier roto shapes: anchors with tangent handles, matching v1 `spline.py`.
//!
//! A shape is a list of [`BezierNode`] values: each node stores an anchor plus
//! its `in`/`out` tangents as *offsets* from the anchor (the same representation
//! the v1 `.brp` project JSON uses), and a `linked` flag that mirrors the two
//! handles while they are dragged together.
//!
//! Conventions:
//!
//! * All coordinates are `f32` in **source/display pixel space** (origin
//!   top-left). The v1 implementation used Python floats (double precision);
//!   `f32` matches the on-disk JSON and is enough for interactive masks, at the
//!   cost of a small precision difference for very large accumulated transforms.
//! * A closed shape produces one cubic segment per node; an open shape produces
//!   `nodes.len() - 1` segments. An open outline is still filled as a closed
//!   polygon when rasterised (v1 `cv2.fillPoly` behaviour), and hit-testing a
//!   closed shape also tests the closing edge.
//! * Angles are **radians**, matching v1 `math.cos`/`math.sin`, unless a method
//!   name says otherwise. UI code that works in degrees must convert
//!   (`to_radians`) before calling [`BezierShape::transform`].
//! * Curves are cubic Bezier segments with the v1 mapping: a segment's first
//!   control point is the start node's `out` handle, its second control point is
//!   the end node's `in` handle. A missing (zero) handle makes the segment
//!   edge tangent.
//!
//! The shape is serialised exactly like the v1 JSON:
//! `{ "closed": bool, "nodes": [ { "x", "y", "out_dx", "out_dy", "in_dx",
//! "in_dy", "linked" } ] }`, with every entry optional on load.

use serde::{Deserialize, Serialize};

use crate::mask_layers::raster::{self, Mask};

/// Default number of subdivisions per cubic segment used by
/// [`BezierShape::sample`] (matching v1's `SAMPLE_STEPS`).
pub const SAMPLE_STEPS: u32 = 14;

/// Magnitude below which a tangent offset counts as "no handle" (v1 used the
/// same `1e-6` sum-of-absolute-components test).
pub const HANDLE_EPSILON: f32 = 1e-6;

/// Which side of a node a tangent handle belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandleSide {
    /// Handle towards the previous segment.
    In,
    /// Handle towards the next segment.
    Out,
}

/// A kind of interactive control point that can be picked in the viewer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ControlKind {
    /// The anchor itself.
    Anchor,
    /// The `in` tangent handle.
    In,
    /// The `out` tangent handle.
    Out,
}

impl ControlKind {
    /// Whether this is a tangent handle rather than the anchor.
    pub fn is_handle(self) -> bool {
        !matches!(self, Self::Anchor)
    }
}

/// Result of picking the nearest control point to a cursor position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NearestControl {
    /// Which control was closest.
    pub kind: ControlKind,
    /// Node index that owns the control.
    pub index: usize,
    /// Distance from the queried point, in pixels.
    pub distance: f32,
}

/// Result of finding the nearest point on the sampled outline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NearestSegment {
    /// Cubic segment index (into [`BezierShape::segments`]).
    pub segment: usize,
    /// Normalised position along that segment, 0..1.
    pub t: f32,
    /// Distance from the queried point, in pixels.
    pub distance: f32,
}

/// Axis-aligned bounds in source pixel space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    /// Left edge.
    pub min_x: f32,
    /// Top edge.
    pub min_y: f32,
    /// Right edge.
    pub max_x: f32,
    /// Bottom edge.
    pub max_y: f32,
}

impl Bounds {
    /// Width of the box in pixels (`max_x - min_x`).
    pub fn width(self) -> f32 {
        self.max_x - self.min_x
    }

    /// Height of the box in pixels (`max_y - min_y`).
    pub fn height(self) -> f32 {
        self.max_y - self.min_y
    }

    /// Centre of the box.
    pub fn centre(self) -> (f32, f32) {
        (
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
        )
    }

    /// Whether the box has zero or negative extent (e.g. an empty shape).
    pub fn is_empty(self) -> bool {
        self.max_x <= self.min_x || self.max_y <= self.min_y
    }
}

/// One cubic Bezier segment in source pixel space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CubicSegment {
    /// Segment start point (anchor of the start node).
    pub start: (f32, f32),
    /// First control point (start node `out` handle, or the anchor when absent).
    pub control1: (f32, f32),
    /// Second control point (end node `in` handle, or the anchor when absent).
    pub control2: (f32, f32),
    /// Segment end point (anchor of the end node).
    pub end: (f32, f32),
}

impl CubicSegment {
    /// Evaluate the cubic at normalised position `t` (clamped to 0..1).
    pub fn eval(&self, t: f32) -> (f32, f32) {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let u = 1.0 - t;
        let (x0, y0) = self.start;
        let (x1, y1) = self.control1;
        let (x2, y2) = self.control2;
        let (x3, y3) = self.end;
        let x = u * u * u * x0 + 3.0 * u * u * t * x1 + 3.0 * u * t * t * x2 + t * t * t * x3;
        let y = u * u * u * y0 + 3.0 * u * u * t * y1 + 3.0 * u * t * t * y2 + t * t * t * y3;
        (x, y)
    }
}

/// One anchor with its two tangent handles, stored as offsets.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BezierNode {
    /// Anchor x in source pixels.
    pub x: f32,
    /// Anchor y in source pixels.
    pub y: f32,
    /// Outgoing handle offset from the anchor.
    pub out_dx: f32,
    /// Outgoing handle offset from the anchor.
    pub out_dy: f32,
    /// Incoming handle offset from the anchor.
    pub in_dx: f32,
    /// Incoming handle offset from the anchor.
    pub in_dy: f32,
    /// When true, dragging one handle mirrors the other.
    pub linked: bool,
}

impl Default for BezierNode {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            out_dx: 0.0,
            out_dy: 0.0,
            in_dx: 0.0,
            in_dy: 0.0,
            linked: true,
        }
    }
}

impl BezierNode {
    /// Create a corner node (no handles, linked tangents enabled).
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            ..Self::default()
        }
    }

    /// Create a smooth node with an outgoing handle; `in` mirrors it, like v1
    /// `BezierShape.add_node(..., mirror: true)`.
    pub fn smooth(x: f32, y: f32, out_dx: f32, out_dy: f32) -> Self {
        Self {
            x,
            y,
            out_dx,
            out_dy,
            in_dx: -out_dx,
            in_dy: -out_dy,
            linked: true,
        }
    }

    /// Absolute position of the outgoing handle.
    pub fn out_point(&self) -> (f32, f32) {
        (self.x + self.out_dx, self.y + self.out_dy)
    }

    /// Absolute position of the incoming handle.
    pub fn in_point(&self) -> (f32, f32) {
        (self.x + self.in_dx, self.y + self.in_dy)
    }

    /// Whether an outgoing handle is present (magnitude above
    /// [`HANDLE_EPSILON`]).
    pub fn has_out(&self) -> bool {
        self.out_dx.abs() + self.out_dy.abs() > HANDLE_EPSILON
    }

    /// Whether an incoming handle is present.
    pub fn has_in(&self) -> bool {
        self.in_dx.abs() + self.in_dy.abs() > HANDLE_EPSILON
    }
}

/// A bezier roto shape: ordered nodes plus a closed flag.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BezierShape {
    /// Nodes in drawing order.
    pub nodes: Vec<BezierNode>,
    /// Whether the last node connects back to the first.
    pub closed: bool,
}

impl BezierShape {
    /// Create an empty open shape.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a shape from nodes.
    pub fn with_nodes(nodes: Vec<BezierNode>, closed: bool) -> Self {
        Self { nodes, closed }
    }

    /// Number of nodes (v1 `len(shape)`).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the shape has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Append a node and return its index.
    ///
    /// With `mirror` (the v1 default) a non-zero outgoing handle is mirrored
    /// into the incoming handle, producing a smooth point. Note that v1 keeps
    /// `linked == true` even when `mirror` is false; this port does the same.
    pub fn add_node(&mut self, x: f32, y: f32, out_dx: f32, out_dy: f32, mirror: bool) -> usize {
        let mut node = BezierNode::new(x, y);
        node.out_dx = out_dx;
        node.out_dy = out_dy;
        if mirror && (out_dx != 0.0 || out_dy != 0.0) {
            node.in_dx = -out_dx;
            node.in_dy = -out_dy;
        }
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    /// Append a corner node (no handles) and return its index.
    pub fn add_corner(&mut self, x: f32, y: f32) -> usize {
        self.add_node(x, y, 0.0, 0.0, true)
    }

    /// Remove a node; shapes below three nodes become open again (v1 rule).
    pub fn remove_node(&mut self, index: usize) -> bool {
        if index >= self.nodes.len() {
            return false;
        }
        self.nodes.remove(index);
        if self.nodes.len() < 3 {
            self.closed = false;
        }
        true
    }

    /// Move an anchor without moving its handles (v1 `set_anchor`).
    pub fn set_anchor(&mut self, index: usize, x: f32, y: f32) -> bool {
        match self.nodes.get_mut(index) {
            Some(node) => {
                node.x = x;
                node.y = y;
                true
            }
            None => false,
        }
    }

    /// Translate an anchor without moving its handles (v1 `move_anchor`).
    pub fn move_anchor(&mut self, index: usize, dx: f32, dy: f32) -> bool {
        match self.nodes.get_mut(index) {
            Some(node) => {
                node.x += dx;
                node.y += dy;
                true
            }
            None => false,
        }
    }

    /// Place one handle at an absolute pixel position (v1 `set_handle`).
    ///
    /// `mirror = None` follows the node's `linked` flag, so an unlinked tangent
    /// stays where the user put it; `Some(false)` moves only the requested side
    /// and keeps the node linked for the next drag.
    pub fn set_handle(
        &mut self,
        index: usize,
        side: HandleSide,
        x: f32,
        y: f32,
        mirror: Option<bool>,
    ) -> bool {
        let Some(node) = self.nodes.get_mut(index) else {
            return false;
        };
        let mirror = mirror.unwrap_or(node.linked);
        let dx = x - node.x;
        let dy = y - node.y;
        match side {
            HandleSide::In => {
                node.in_dx = dx;
                node.in_dy = dy;
                if mirror {
                    node.out_dx = -dx;
                    node.out_dy = -dy;
                }
            }
            HandleSide::Out => {
                node.out_dx = dx;
                node.out_dy = dy;
                if mirror {
                    node.in_dx = -dx;
                    node.in_dy = -dy;
                }
            }
        }
        true
    }

    /// Flatten both handles of a node back to a corner (v1 `reset_handles`).
    pub fn reset_handles(&mut self, index: usize) -> bool {
        match self.nodes.get_mut(index) {
            Some(node) => {
                node.in_dx = 0.0;
                node.in_dy = 0.0;
                node.out_dx = 0.0;
                node.out_dy = 0.0;
                node.linked = true;
                true
            }
            None => false,
        }
    }

    /// Break or restore mirrored tangents (v1 `set_linked`).
    ///
    /// Restoring the link re-mirrors the `in` handle from the `out` handle when
    /// one exists. Returns the node's resulting `linked` flag, or `false` for an
    /// invalid index.
    pub fn set_linked(&mut self, index: usize, linked: bool) -> bool {
        let Some(node) = self.nodes.get_mut(index) else {
            return false;
        };
        node.linked = linked;
        if linked && node.has_out() {
            node.in_dx = -node.out_dx;
            node.in_dy = -node.out_dy;
        }
        node.linked
    }

    /// Move, scale and rotate every anchor and tangent around `centre`.
    ///
    /// * `dx`/`dy` are a translation in pixels applied after scaling/rotation.
    /// * `scale` is a uniform multiplier (1.0 = unchanged).
    /// * `angle` is **radians**, positive is clockwise in image coordinates
    ///   (y grows downwards), matching v1's `math.cos/math.sin` maths.
    /// * `centre` defaults to the centre of the anchor bounds; handles rotate
    ///   and scale around the same centre without being translated by it.
    pub fn transform(
        &mut self,
        dx: f32,
        dy: f32,
        scale: f32,
        angle: f32,
        centre: Option<(f32, f32)>,
    ) {
        if self.nodes.is_empty() {
            return;
        }
        let (cx, cy) = centre.unwrap_or_else(|| self.bounds(false).centre());
        let (sin_a, cos_a) = angle.sin_cos();
        for node in &mut self.nodes {
            let px = (node.x - cx) * scale;
            let py = (node.y - cy) * scale;
            node.x = cx + px * cos_a - py * sin_a + dx;
            node.y = cy + px * sin_a + py * cos_a + dy;
            let ox = node.out_dx * scale;
            let oy = node.out_dy * scale;
            node.out_dx = ox * cos_a - oy * sin_a;
            node.out_dy = ox * sin_a + oy * cos_a;
            let ix = node.in_dx * scale;
            let iy = node.in_dy * scale;
            node.in_dx = ix * cos_a - iy * sin_a;
            node.in_dy = ix * sin_a + iy * cos_a;
        }
    }

    /// Cubic segments in drawing order (v1 `segments`).
    pub fn segments(&self) -> Vec<CubicSegment> {
        let count = self.nodes.len();
        if count < 2 {
            return Vec::new();
        }
        let last = if self.closed { count } else { count - 1 };
        let mut out = Vec::with_capacity(last);
        for index in 0..last {
            let a = self.nodes[index];
            let b = self.nodes[(index + 1) % count];
            let control1 = if a.has_out() {
                a.out_point()
            } else {
                (a.x, a.y)
            };
            let control2 = if b.has_in() { b.in_point() } else { (b.x, b.y) };
            out.push(CubicSegment {
                start: (a.x, a.y),
                control1,
                control2,
                end: (b.x, b.y),
            });
        }
        out
    }

    /// Polyline approximation of the outline (v1 `sample`).
    ///
    /// Returns the first anchor followed by `steps` points per segment (at least
    /// 3 per segment); a single-node shape returns one point and an empty shape
    /// returns no points. A closed shape's polyline ends back at its first
    /// anchor.
    pub fn sample(&self, steps: u32) -> Vec<(f32, f32)> {
        let Some(first) = self.nodes.first() else {
            return Vec::new();
        };
        let steps = steps.max(3);
        let mut points = vec![(first.x, first.y)];
        for segment in self.segments() {
            for step in 1..=steps {
                let t = step as f32 / steps as f32;
                points.push(segment.eval(t));
            }
        }
        points
    }

    /// Anchors bounds in pixels, optionally including tangent handles
    /// (v1 `bounds`). An empty shape returns zeros.
    pub fn bounds(&self, with_handles: bool) -> Bounds {
        let Some(first) = self.nodes.first() else {
            return Bounds {
                min_x: 0.0,
                min_y: 0.0,
                max_x: 0.0,
                max_y: 0.0,
            };
        };
        let mut min_x = first.x;
        let mut min_y = first.y;
        let mut max_x = first.x;
        let mut max_y = first.y;
        for node in &self.nodes {
            let mut extend = |x: f32, y: f32| {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            };
            extend(node.x, node.y);
            if with_handles {
                let (out_x, out_y) = node.out_point();
                extend(out_x, out_y);
                let (in_x, in_y) = node.in_point();
                extend(in_x, in_y);
            }
        }
        Bounds {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Rasterise the shape to an 8-bit mask of `width x height` pixels, with an
    /// optional feather of `softness` pixels (see [`raster::rasterize`]).
    pub fn rasterize(&self, width: u32, height: u32, softness: u32) -> Mask {
        raster::rasterize(self, width, height, softness)
    }

    /// Closest control point within `tolerance` pixels, for viewer picking.
    ///
    /// The tolerance applies to anchors and to existing handles only; ties keep
    /// the earlier control (anchor before `in` before `out`), like v1.
    pub fn nearest_control(&self, x: f32, y: f32, tolerance: f32) -> Option<NearestControl> {
        let mut best: Option<NearestControl> = None;
        for (index, node) in self.nodes.iter().enumerate() {
            let controls = [
                (ControlKind::Anchor, (node.x, node.y), true),
                (ControlKind::In, node.in_point(), node.has_in()),
                (ControlKind::Out, node.out_point(), node.has_out()),
            ];
            for (kind, (px, py), present) in controls {
                if !present {
                    continue;
                }
                let distance = ((px - x).powi(2) + (py - y).powi(2)).sqrt();
                if distance <= tolerance && best.is_none_or(|current| distance < current.distance) {
                    best = Some(NearestControl {
                        kind,
                        index,
                        distance,
                    });
                }
            }
        }
        best
    }

    /// Distance from a point to the sampled outline in pixels (v1 `distance`).
    ///
    /// Returns `f32::INFINITY` for an empty shape, the anchor distance for a
    /// single-node shape, and otherwise the closest point on the polyline
    /// sampled with 8 subdivisions per segment. For closed shapes the closing
    /// edge is tested too.
    pub fn distance(&self, x: f32, y: f32) -> f32 {
        let points = self.sample(8);
        if points.is_empty() {
            return f32::INFINITY;
        }
        if points.len() == 1 {
            return ((points[0].0 - x).powi(2) + (points[0].1 - y).powi(2)).sqrt();
        }
        let mut best = f32::INFINITY;
        for window in points.windows(2) {
            best = best.min(point_segment_distance(x, y, window[0], window[1]));
        }
        if self.closed && points.len() > 1 {
            best = best.min(point_segment_distance(
                x,
                y,
                points[points.len() - 1],
                points[0],
            ));
        }
        best
    }

    /// Closest point on the cubic outline: `(segment index, t, distance)` in v1.
    ///
    /// Each segment is sampled with 24 steps; `None` means the shape has fewer
    /// than two nodes.
    pub fn nearest_segment(&self, x: f32, y: f32) -> Option<NearestSegment> {
        let segments = self.segments();
        if segments.is_empty() {
            return None;
        }
        let steps = 24_u32;
        let mut best: Option<NearestSegment> = None;
        for (index, segment) in segments.iter().enumerate() {
            for step in 0..=steps {
                let t = step as f32 / steps as f32;
                let (px, py) = segment.eval(t);
                let distance = ((px - x).powi(2) + (py - y).powi(2)).sqrt();
                if best.is_none_or(|current| distance < current.distance) {
                    best = Some(NearestSegment {
                        segment: index,
                        t,
                        distance,
                    });
                }
            }
        }
        best
    }

    /// Split `segment` at `t` and return the index of the new node.
    ///
    /// This is a **literal port of v1 `insert_point`**, including its handle
    /// assignment: the new node's `out` gets the de Casteljau point `d` and its
    /// `in` gets `e`. Mathematically the `in` handle should be `d` and the `out`
    /// handle `e`, so for a curved segment the v1 split slightly changes the
    /// curve; straight segments are unaffected. Use
    /// [`Self::insert_point_exact`] for a geometrically exact split. `None` is
    /// returned for an invalid segment index (v1 returned `-1`).
    pub fn insert_point(&mut self, segment: usize, t: f32) -> Option<usize> {
        let mut split = self.de_casteljau_split(segment, t)?;
        // v1 parity: out <- d, in <- e (swapped relative to the exact split).
        split.node.out_dx = split.d.0 - split.point.0;
        split.node.out_dy = split.d.1 - split.point.1;
        split.node.in_dx = split.e.0 - split.point.0;
        split.node.in_dy = split.e.1 - split.point.1;
        Some(split.into_inserted(self))
    }

    /// Split `segment` at `t` with the mathematically exact handle assignment.
    ///
    /// The resulting outline is the same cubic curve as before the split (only
    /// the segment parameterisation changes), so inserting a point never pops
    /// the shape. Returns the new node index.
    pub fn insert_point_exact(&mut self, segment: usize, t: f32) -> Option<usize> {
        let split = self.de_casteljau_split(segment, t)?;
        Some(split.into_inserted(self))
    }

    fn de_casteljau_split(&self, segment: usize, t: f32) -> Option<Split> {
        let segments = self.segments();
        let source = *segments.get(segment)?;
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let lerp = |a: (f32, f32), b: (f32, f32)| (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
        let a = lerp(source.start, source.control1);
        let b = lerp(source.control1, source.control2);
        let c = lerp(source.control2, source.end);
        let d = lerp(a, b);
        let e = lerp(b, c);
        let point = lerp(d, e);
        let node = BezierNode {
            x: point.0,
            y: point.1,
            out_dx: e.0 - point.0,
            out_dy: e.1 - point.1,
            in_dx: d.0 - point.0,
            in_dy: d.1 - point.1,
            linked: true,
        };
        Some(Split {
            segment,
            a,
            c,
            d,
            e,
            point,
            node,
        })
    }

    /// Blend towards `other` by `t` (0..1), matching v1 `interpolate`.
    ///
    /// Shapes must have the same node count; a mismatch returns a copy of
    /// `self`. Handles and anchors lerp component-wise, `linked` and `closed`
    /// switch over at `t >= 0.5`.
    pub fn interpolate(&self, other: &BezierShape, t: f32) -> BezierShape {
        let t = if t.is_finite() {
            t.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if other.nodes.len() != self.nodes.len() {
            return self.clone();
        }
        let mut nodes = Vec::with_capacity(self.nodes.len());
        for (a, b) in self.nodes.iter().zip(other.nodes.iter()) {
            nodes.push(BezierNode {
                x: a.x + (b.x - a.x) * t,
                y: a.y + (b.y - a.y) * t,
                out_dx: a.out_dx + (b.out_dx - a.out_dx) * t,
                out_dy: a.out_dy + (b.out_dy - a.out_dy) * t,
                in_dx: a.in_dx + (b.in_dx - a.in_dx) * t,
                in_dy: a.in_dy + (b.in_dy - a.in_dy) * t,
                linked: if t < 0.5 { a.linked } else { b.linked },
            });
        }
        BezierShape {
            nodes,
            closed: if t < 0.5 { self.closed } else { other.closed },
        }
    }
}

/// Distance from `(x, y)` to the segment `a`-`b` in pixels.
fn point_segment_distance(x: f32, y: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
    let vx = b.0 - a.0;
    let vy = b.1 - a.1;
    let length_squared = vx * vx + vy * vy;
    let t = if length_squared <= 1e-9 {
        0.0
    } else {
        (((x - a.0) * vx + (y - a.1) * vy) / length_squared).clamp(0.0, 1.0)
    };
    let dx = a.0 + vx * t - x;
    let dy = a.1 + vy * t - y;
    (dx * dx + dy * dy).sqrt()
}

/// Intermediate de Casteljau state shared by the two split flavours.
struct Split {
    segment: usize,
    a: (f32, f32),
    c: (f32, f32),
    d: (f32, f32),
    e: (f32, f32),
    point: (f32, f32),
    node: BezierNode,
}

impl Split {
    /// Rewrite the neighbouring handles and insert the node.
    fn into_inserted(self, shape: &mut BezierShape) -> usize {
        let Split {
            segment,
            a,
            c,
            node,
            ..
        } = self;
        let count = shape.nodes.len();
        {
            let first = &mut shape.nodes[segment];
            first.out_dx = a.0 - first.x;
            first.out_dy = a.1 - first.y;
        }
        let second_index = (segment + 1) % count;
        {
            let second = &mut shape.nodes[second_index];
            second.in_dx = c.0 - second.x;
            second.in_dy = c.1 - second.y;
        }
        let inserted = segment + 1;
        shape.nodes.insert(inserted, node);
        inserted
    }
}
