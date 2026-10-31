//! Timeline scene controls and exact-frame scene targeting.
//!
//! Scene selection and the playhead are separate UI state. A current-scene job
//! must always resolve the playhead against the source's half-open ranges.

use ck_desktop_core::{FrameRange, SceneRange};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SceneRunError {
    NoScene,
    Disabled(usize),
}

/// Return the vector position, rather than assuming `SceneRange::index` is one.
pub fn scene_index_at_frame(scenes: &[SceneRange], frame: u64) -> Option<usize> {
    scenes
        .iter()
        .position(|scene| scene.start <= frame && frame < scene.end_exclusive)
}

/// Disabled scenes stay under the needle but cannot be accidentally queued.
pub fn enabled_scene_range_at_frame(
    scenes: &[SceneRange],
    frame: u64,
) -> Result<FrameRange, SceneRunError> {
    let index = scene_index_at_frame(scenes, frame).ok_or(SceneRunError::NoScene)?;
    let scene = &scenes[index];
    if scene.enabled {
        Ok(scene.range())
    } else {
        Err(SceneRunError::Disabled(index))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrackAction {
    Seek(u64),
    SetSceneEnabled {
        scene: usize,
        enabled: bool,
    },
    SelectScene(usize),
    SetPlaybackRange(FrameRange),
    SplitScene,
    MergeNext,
    MergePrevious,
    /// Move the selected mask keyframes by `delta` source frames.
    MoveMaskKeys {
        delta: i64,
    },
    /// Per-scene Result despill override (0..=1); `None` returns the scene to the global value.
    SetSceneDespill {
        scene: usize,
        amount: Option<f32>,
    },
}

/// Mask keyframe lane state shared with the surrounding Studio.
#[derive(Debug)]
pub struct MaskLaneInput<'a> {
    /// Sorted keyframes of the selected mask layer.
    pub keyframes: &'a [i64],
    /// Scene whose bar hosts the lane, when a layer is selected.
    pub scene: Option<usize>,
    /// Selected keyframes (mutated by click/marquee selection).
    pub selected: &'a mut std::collections::BTreeSet<i64>,
    /// In-flight drag (marquee or move) owned by the Studio.
    pub drag: &'a mut Option<MaskTimelineDrag>,
}

/// In-flight timeline mask keyframe gesture.
#[derive(Clone, Debug)]
pub enum MaskTimelineDrag {
    /// Rubber-band selection from `start_frame` to the current pointer.
    Marquee { start_frame: f64, shift: bool },
    /// Move the selected keyframes by the pointer delta since `start_frame`.
    Move { start_frame: f64 },
}

/// Select the mask keyframes inside a marquee range (`shift` keeps the
/// existing selection). Frames are compared with a half-frame tolerance so the
/// pointer does not have to land exactly on a diamond.
pub fn mask_marquee_selection(
    keyframes: &[i64],
    start_frame: f64,
    end_frame: f64,
    shift: bool,
    selected: &mut std::collections::BTreeSet<i64>,
) {
    let (low, high) = if start_frame <= end_frame {
        (start_frame, end_frame)
    } else {
        (end_frame, start_frame)
    };
    if !shift {
        selected.clear();
    }
    for &key in keyframes {
        let frame = key as f64;
        if frame >= low - 0.5 && frame <= high + 0.5 {
            selected.insert(key);
        }
    }
}

/// Merge a set of cached frames into contiguous inclusive runs.
pub(crate) fn cache_runs(frames: &std::collections::BTreeSet<u64>) -> Vec<(u64, u64)> {
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for &frame in frames {
        match runs.last_mut() {
            Some((_, end)) if end.saturating_add(1) == frame => *end = frame,
            _ => runs.push((frame, frame)),
        }
    }
    runs
}

pub const MIN_TRACK_HEIGHT: f32 = 140.0;
const PAN_BAR_SPACE: f32 = 22.0;
const RULER_HEIGHT: f32 = 20.0;
const SCENE_TOP: f32 = 44.0;
const CACHE_LANES_HEIGHT: f32 = 12.0;
const SCENE_TOGGLE_SIZE: f32 = 18.0;
/// Minimum height of the mask keyframe band under a scene bar.
const MASK_LANE_HEIGHT: f32 = 24.0;

pub struct TrackInput<'a> {
    pub frame_count: u64,
    pub playhead: u64,
    pub range: FrameRange,
    pub scenes: &'a [SceneRange],
    /// Source-frame index, SAM-mask availability, render availability.
    pub lanes: &'a [(u64, bool, bool)],
    /// Source frames currently retained in the RAM media cache.
    pub ram_media: &'a std::collections::BTreeSet<u64>,
    /// Result/preview frames currently retained in the RAM result cache.
    pub ram_results: &'a std::collections::BTreeSet<u64>,
    pub scene_controls_enabled: bool,
    /// Optional supplied checkbox artwork for the scene enable toggles.
    pub toggle_on: Option<egui::TextureId>,
    pub toggle_off: Option<egui::TextureId>,
    /// Mask keyframe lane when the selected layer's keyframes are shown.
    pub mask_lane: Option<MaskLaneInput<'a>>,
    /// Active per-scene despill override (0..=1) by scene position; `None` uses the global value.
    pub scene_despill: &'a [Option<f32>],
    /// Global Result despill amount (0..=1) shown for scenes without an override.
    pub global_despill: f32,
    /// Total canvas plus pan-scrollbar height, in logical UI points.
    pub height: f32,
}

#[derive(Clone, Copy)]
pub struct TrackColors {
    pub accent: Color32,
    pub mask: Color32,
    pub muted: Color32,
}

impl Default for TrackColors {
    fn default() -> Self {
        Self {
            accent: Color32::from_rgb(88, 182, 242),
            mask: Color32::from_rgb(88, 214, 158),
            muted: Color32::from_gray(148),
        }
    }
}

/// Per-media timeline navigation. These are UI coordinates, never job ranges.
#[derive(Clone, Debug)]
pub struct TimelineView {
    frames: u64,
    zoom: f64,
    offset: f64,
    last_playhead: Option<u64>,
}

impl Default for TimelineView {
    fn default() -> Self {
        Self {
            frames: 0,
            zoom: 1.0,
            offset: 0.0,
            last_playhead: None,
        }
    }
}

impl TimelineView {
    pub fn fit_all(&mut self) {
        self.zoom = 1.0;
        self.offset = 0.0;
    }

    /// Zoom so `start..end_exclusive` fills about 70 % of the track, centred.
    pub fn frame_range(&mut self, frames: u64, start: u64, end_exclusive: u64) {
        self.synchronize_frames(frames);
        let span = end_exclusive.saturating_sub(start).max(1) as f64 / 0.7;
        self.zoom = (frames as f64 / span).clamp(1.0, frames.max(1) as f64);
        let centre = (start + end_exclusive) as f64 / 2.0;
        self.offset = centre - self.visible_frames() / 2.0;
        self.clamp_offset();
    }

    pub fn zoom_factor(&self) -> f64 {
        self.zoom
    }

    pub fn visible_range(&self) -> (f64, f64) {
        (self.offset, self.offset + self.visible_frames())
    }

    fn visible_frames(&self) -> f64 {
        self.frames as f64 / self.zoom
    }

    fn clamp_offset(&mut self) {
        if !self.offset.is_finite() {
            self.offset = 0.0;
        }
        self.offset = self
            .offset
            .clamp(0.0, (self.frames as f64 - self.visible_frames()).max(0.0));
    }

    fn synchronize_frames(&mut self, frames: u64) {
        if self.frames != frames {
            self.frames = frames;
            self.fit_all();
            self.last_playhead = None;
        }
        self.zoom = self.zoom.clamp(1.0, frames.max(1) as f64);
        self.clamp_offset();
    }

    /// Follow a newly stepped/played frame, without undoing a manual pan when
    /// the needle stays still. This mirrors v1's ensure-visible behavior.
    fn prepare(&mut self, frames: u64, playhead: u64) {
        self.synchronize_frames(frames);
        if self.last_playhead != Some(playhead) {
            let frame = playhead.min(frames.saturating_sub(1)) as f64;
            let span = self.visible_frames();
            if frame < self.offset {
                self.offset = frame - span * 0.15;
            } else if frame >= self.offset + span {
                self.offset = frame - span * 0.85;
            }
            self.clamp_offset();
            self.last_playhead = Some(playhead);
        }
    }

    /// Keep the source frame under `fraction` stationary while changing zoom.
    pub fn zoom_at_fraction(&mut self, frames: u64, fraction: f64, factor: f64) {
        self.synchronize_frames(frames);
        if !factor.is_finite() || factor <= 0.0 || frames == 0 {
            return;
        }
        let fraction = fraction.clamp(0.0, 1.0);
        let anchor = self.offset + fraction * self.visible_frames();
        self.zoom = (self.zoom * factor).clamp(1.0, frames.max(1) as f64);
        self.offset = anchor - fraction * self.visible_frames();
        self.clamp_offset();
    }

    /// Positive delta moves the visible content right, like a middle drag.
    pub fn pan_pixels(&mut self, frames: u64, delta: f32, width: f32) {
        self.synchronize_frames(frames);
        if delta.is_finite() && width.is_finite() && width > 0.0 {
            self.offset -= delta as f64 / width as f64 * self.visible_frames();
            self.clamp_offset();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RangeDragMode {
    In,
    Out,
    Move,
}

#[derive(Clone, Copy)]
struct PlaybackDrag {
    mode: RangeDragMode,
    original: FrameRange,
    pointer_frame: f64,
}

fn moved_playback_range(
    original: FrameRange,
    mode: RangeDragMode,
    delta: f64,
    frames: u64,
) -> FrameRange {
    if frames == 0 || original.is_empty() || original.end_exclusive > frames || !delta.is_finite() {
        return original;
    }
    let delta = delta.round();
    match mode {
        RangeDragMode::In => FrameRange {
            start: (original.start as f64 + delta).clamp(0.0, (original.end_exclusive - 1) as f64)
                as u64,
            ..original
        },
        RangeDragMode::Out => FrameRange {
            end_exclusive: (original.end_exclusive as f64 + delta)
                .clamp((original.start + 1) as f64, frames as f64)
                as u64,
            ..original
        },
        RangeDragMode::Move => {
            let start =
                (original.start as f64 + delta).clamp(0.0, (frames - original.len()) as f64) as u64;
            FrameRange {
                start,
                end_exclusive: start + original.len(),
            }
        }
    }
}

#[derive(Clone, Copy)]
struct TrackGeometry {
    inner: Rect,
    frames: u64,
    offset: f64,
    span: f64,
}

impl TrackGeometry {
    fn frame_x(self, frame: u64) -> f32 {
        if self.frames == 0 {
            return self.inner.left();
        }
        (self.inner.left() as f64
            + (frame.min(self.frames) as f64 - self.offset) / self.span * self.inner.width() as f64)
            as f32
    }

    fn frame_at_x(self, x: f32) -> Option<u64> {
        if self.frames == 0 || self.inner.width() <= 0.0 || !x.is_finite() {
            return None;
        }
        let fraction =
            ((x as f64 - self.inner.left() as f64) / self.inner.width() as f64).clamp(0.0, 1.0);
        let position = self.offset + fraction * self.span;
        let nearest_boundary = position.round() as u64;
        // A painted boundary is a rounded UI coordinate. Recover that exact
        // boundary instead of turning e.g. frame13 into12 through f32 pixel
        // rounding. Ordinary positions keep the half-open frame-cell mapping.
        let frame = if self.frame_x(nearest_boundary) == x {
            nearest_boundary
        } else {
            position as u64
        };
        Some(frame.min(self.frames - 1))
    }

    fn scene_rect(self, scene: &SceneRange) -> Rect {
        Rect::from_min_max(
            Pos2::new(
                self.frame_x(scene.start).max(self.inner.left()),
                self.inner.top() + SCENE_TOP,
            ),
            Pos2::new(
                self.frame_x(scene.end_exclusive).min(self.inner.right()),
                self.inner.bottom() - CACHE_LANES_HEIGHT,
            ),
        )
    }

    fn pointer_frame(self, x: f32) -> f64 {
        self.offset + (x as f64 - self.inner.left() as f64) / self.inner.width() as f64 * self.span
    }

    fn playback_lane(self) -> Rect {
        Rect::from_min_max(
            Pos2::new(self.inner.left(), self.inner.top() + 22.0),
            Pos2::new(self.inner.right(), self.inner.top() + 37.0),
        )
    }

    fn playback_handle(self, boundary: u64) -> Option<Rect> {
        let x = self.frame_x(boundary);
        (x >= self.inner.left() && x <= self.inner.right()).then(|| {
            Rect::from_min_max(
                Pos2::new(x - 6.0, self.inner.top() + 20.0),
                Pos2::new(x + 6.0, self.inner.top() + 40.0),
            )
            .intersect(self.inner)
        })
    }

    fn playback_body(self, range: FrameRange) -> Rect {
        Rect::from_min_max(
            Pos2::new(
                self.frame_x(range.start).max(self.inner.left()),
                self.inner.top() + 22.0,
            ),
            Pos2::new(
                self.frame_x(range.end_exclusive).min(self.inner.right()),
                self.inner.top() + 37.0,
            ),
        )
    }

    fn playback_hit(self, range: FrameRange, pos: Pos2) -> Option<RangeDragMode> {
        let start = self
            .playback_handle(range.start)
            .filter(|rect| rect.contains(pos));
        let end = self
            .playback_handle(range.end_exclusive)
            .filter(|rect| rect.contains(pos));
        match (start, end) {
            (Some(_), Some(_)) => {
                if (pos.x - self.frame_x(range.start)).abs()
                    <= (pos.x - self.frame_x(range.end_exclusive)).abs()
                {
                    Some(RangeDragMode::In)
                } else {
                    Some(RangeDragMode::Out)
                }
            }
            (Some(_), None) => Some(RangeDragMode::In),
            (None, Some(_)) => Some(RangeDragMode::Out),
            (None, None) => self
                .playback_body(range)
                .contains(pos)
                .then_some(RangeDragMode::Move),
        }
    }

    #[cfg(test)]
    fn toggle_rect(self, scene: &SceneRange) -> Option<Rect> {
        self.toggle_rect_laned(scene, false)
    }

    /// Scene enable checkbox. With a mask keyframe band active the toggle is
    /// centred in the remaining upper part of the scene bar instead of the
    /// whole bar, so it never overlaps the band.
    fn toggle_rect_laned(self, scene: &SceneRange, lane_active: bool) -> Option<Rect> {
        let region = self.scene_rect(scene).shrink2(Vec2::new(1.0, 0.0));
        if region.width() < 30.0 {
            return None;
        }
        let centre_y = if lane_active {
            (region.top() + self.mask_lane_rect(scene).top()) * 0.5
        } else {
            region.center().y
        };
        Some(Rect::from_min_size(
            Pos2::new(region.right() - 24.0, centre_y - SCENE_TOGGLE_SIZE * 0.5),
            Vec2::splat(SCENE_TOGGLE_SIZE),
        ))
    }

    fn ruler_rect(self) -> Rect {
        Rect::from_min_max(
            self.inner.min,
            Pos2::new(self.inner.right(), self.inner.top() + RULER_HEIGHT),
        )
    }

    /// Bottom band of a scene bar reserved for mask keyframes.
    fn mask_lane_rect(self, scene: &SceneRange) -> Rect {
        let region = self.scene_rect(scene).shrink2(Vec2::new(1.0, 0.0));
        Rect::from_min_max(
            Pos2::new(
                region.left(),
                (region.bottom() - MASK_LANE_HEIGHT).max(region.top()),
            ),
            region.max,
        )
    }

    fn playhead_cap(self, frame: u64) -> [Pos2; 5] {
        let x = self.frame_x(frame);
        let top = self.inner.top();
        [
            Pos2::new(x - 6.0, top),
            Pos2::new(x + 6.0, top),
            Pos2::new(x + 6.0, top + 7.0),
            Pos2::new(x, top + 13.0),
            Pos2::new(x - 6.0, top + 7.0),
        ]
    }

    fn toggle_at(self, input: &TrackInput<'_>, pos: Pos2) -> Option<usize> {
        input.scenes.iter().enumerate().find_map(|(index, scene)| {
            let lane_active = input
                .mask_lane
                .as_ref()
                .is_some_and(|lane| lane.scene == Some(index));
            self.toggle_rect_laned(scene, lane_active)
                .filter(|rect| rect.contains(pos))
                .map(|_| index)
        })
    }

    fn scene_at(self, input: &TrackInput<'_>, pos: Pos2) -> Option<usize> {
        if !self.inner.contains(pos) {
            return None;
        }
        self.frame_at_x(pos.x)
            .and_then(|frame| scene_index_at_frame(input.scenes, frame))
    }

    fn primary_click(
        self,
        input: &TrackInput<'_>,
        pos: Pos2,
        alt: bool,
        double: bool,
    ) -> Option<TrackAction> {
        let toggle = self
            .toggle_at(input, pos)
            .or_else(|| alt.then(|| self.scene_at(input, pos)).flatten());
        if let Some(scene) = toggle {
            // Clicking a locked checkbox must not unexpectedly seek either.
            return input
                .scene_controls_enabled
                .then(|| TrackAction::SetSceneEnabled {
                    scene,
                    enabled: !input.scenes[scene].enabled,
                });
        }
        if double && let Some(scene) = self.scene_at(input, pos) {
            return Some(TrackAction::SelectScene(scene));
        }
        self.frame_at_x(pos.x).map(TrackAction::Seek)
    }
}

/// Paint the native scene/ruler/cache track and return a single unambiguous edit.
/// Transport and the In/Out controls remain owned by the surrounding interface.
/// Despill override controls for one scene inside the scene context menu. The menu stays
/// open while the value is edited; every change is returned as an action.
fn scene_despill_menu(
    ui: &mut egui::Ui,
    input: &TrackInput<'_>,
    index: usize,
) -> Option<TrackAction> {
    let current = input.scene_despill.get(index).copied().flatten();
    let mut enabled = current.is_some();
    let mut percent = current.unwrap_or(input.global_despill) * 100.0;
    let mut edit = None;
    let toggle = match (input.toggle_on, input.toggle_off) {
        (Some(on), Some(off)) => {
            crate::icons::checkbox_widget(ui, &mut enabled, "Override Result despill", on, off)
        }
        _ => ui.checkbox(&mut enabled, "Override Result despill"),
    };
    let toggle = toggle.on_hover_text(
        "Use a despill amount for this scene only; off means the global slider applies",
    );
    if toggle.clicked() {
        edit = Some(TrackAction::SetSceneDespill {
            scene: index,
            amount: enabled.then_some(percent / 100.0),
        });
    }
    if enabled {
        ui.horizontal(|ui| {
            let slider = ui.add(
                egui::Slider::new(&mut percent, 0.0..=100.0)
                    .step_by(1.0)
                    .suffix("%")
                    .fixed_decimals(0),
            );
            if slider.changed() {
                edit = Some(TrackAction::SetSceneDespill {
                    scene: index,
                    amount: Some(percent / 100.0),
                });
            }
        });
    } else {
        ui.label(
            egui::RichText::new(format!(
                "Using the global value ({:.0} %)",
                input.global_despill * 100.0
            ))
            .small(),
        );
    }
    edit
}

pub fn show_track(
    ui: &mut egui::Ui,
    mut input: TrackInput<'_>,
    colors: TrackColors,
    view: &mut TimelineView,
) -> Option<TrackAction> {
    view.prepare(input.frame_count, input.playhead);
    // Reserve the entire component exactly once. The canvas and scrollbar use
    // explicit hit rectangles inside it, so global widget height/spacing cannot
    // inflate the parent panel on each repaint.
    let (component, component_response) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), input.height.max(MIN_TRACK_HEIGHT)),
        Sense::hover(),
    );
    let rect = Rect::from_min_max(component.min, component.max - Vec2::new(0.0, PAN_BAR_SPACE));
    let response = ui.interact(
        rect,
        component_response.id.with("scene_canvas"),
        Sense::click_and_drag(),
    );
    let inner = Rect::from_min_max(
        rect.min + Vec2::new(18.0, 0.0),
        rect.max - Vec2::new(18.0, 0.0),
    );
    if input.frame_count == 0 || inner.width() <= 0.0 {
        return None;
    }
    let before = (view.zoom, view.offset);
    if response.hovered() {
        let (scroll, shift, pointer) = ui.input_mut(|state| {
            let scroll = state.raw_scroll_delta;
            if scroll != Vec2::ZERO {
                // Navigation belongs to this canvas, not an ancestor panel.
                state.raw_scroll_delta = Vec2::ZERO;
                state.smooth_scroll_delta = Vec2::ZERO;
            }
            (scroll, state.modifiers.shift, state.pointer.hover_pos())
        });
        if scroll.x != 0.0 || shift {
            view.pan_pixels(input.frame_count, scroll.x + scroll.y, inner.width());
        } else if scroll.y != 0.0 {
            let anchor = pointer.map_or(0.5, |pos| {
                ((pos.x - inner.left()) / inner.width()).clamp(0.0, 1.0) as f64
            });
            view.zoom_at_fraction(
                input.frame_count,
                anchor,
                1.3_f64.powf(scroll.y as f64 / 50.0),
            );
        }
    }
    if response.dragged_by(egui::PointerButton::Middle) {
        let frame_delta = ui.input(|state| state.pointer.delta().x);
        view.pan_pixels(input.frame_count, frame_delta, inner.width());
    }
    if before != (view.zoom, view.offset) {
        ui.ctx().request_repaint();
    }
    let (visible_start, visible_end) = view.visible_range();
    let geometry = TrackGeometry {
        inner,
        frames: input.frame_count,
        offset: visible_start,
        span: visible_end - visible_start,
    };
    let painter = ui.painter().with_clip_rect(ui.clip_rect().intersect(inner));
    painter.rect_filled(inner, 4.0, Color32::from_gray(23));
    painter.rect_stroke(
        inner,
        4.0,
        Stroke::new(1.0_f32, Color32::from_gray(70)),
        StrokeKind::Inside,
    );
    painter.rect_filled(geometry.ruler_rect(), 3.0, Color32::from_gray(36));
    painter.hline(
        inner.x_range(),
        inner.top() + RULER_HEIGHT,
        Stroke::new(1.0_f32, Color32::from_gray(73)),
    );
    let current_scene = scene_index_at_frame(input.scenes, input.playhead);
    for (index, scene) in input.scenes.iter().enumerate() {
        let region = geometry.scene_rect(scene).shrink2(Vec2::new(1.0, 0.0));
        if region.width() <= 0.0 {
            continue;
        }
        let color = if scene.enabled {
            Color32::from_gray(64)
        } else {
            Color32::from_gray(35)
        };
        painter.rect_filled(region, 3.0, color);
        painter.rect_stroke(
            region,
            3.0,
            Stroke::new(1.0_f32, Color32::from_gray(88)),
            StrokeKind::Inside,
        );
        if !scene.enabled {
            let clipped = painter.with_clip_rect(region);
            let mut x = region.left() - region.height();
            while x < region.right() {
                clipped.line_segment(
                    [
                        Pos2::new(x, region.bottom()),
                        Pos2::new(x + region.height(), region.top()),
                    ],
                    Stroke::new(1.0_f32, Color32::from_white_alpha(15)),
                );
                x += 10.0;
            }
        }
        if current_scene == Some(index) {
            painter.rect_stroke(
                region,
                3.0,
                Stroke::new(1.5_f32, colors.accent),
                StrokeKind::Inside,
            );
            painter.hline(
                region.x_range(),
                region.top() + 1.0,
                Stroke::new(2.0_f32, colors.accent),
            );
        }
        let lane_active = input
            .mask_lane
            .as_ref()
            .is_some_and(|lane| lane.scene == Some(index));
        if region.width() > 66.0 {
            // Keep the scene label clear of the mask-keyframe band so it is
            // never covered or clipped when the timeline is short.
            let label_bottom = if lane_active {
                (geometry.mask_lane_rect(scene).top() - 2.0).max(region.top() + 18.0)
            } else {
                region.bottom()
            };
            let label_clip =
                Rect::from_min_max(region.min, Pos2::new(region.right() - 48.0, label_bottom));
            let label_center = label_clip.center().y;
            let details = label_clip.height() >= 46.0 && label_clip.width() > 104.0;
            painter.with_clip_rect(label_clip).text(
                Pos2::new(
                    region.left() + 9.0,
                    label_center - if details { 9.0 } else { 0.0 },
                ),
                Align2::LEFT_CENTER,
                format!(
                    "Scene {}{}",
                    scene.index + 1,
                    if scene.enabled { "" } else { " · off" }
                ),
                FontId::proportional(13.0),
                if scene.enabled {
                    Color32::from_gray(230)
                } else {
                    colors.muted
                },
            );
            if details {
                painter.with_clip_rect(label_clip).text(
                    Pos2::new(region.left() + 9.0, label_center + 10.0),
                    Align2::LEFT_CENTER,
                    format!("{}–{}", scene.start + 1, scene.end_exclusive),
                    FontId::monospace(13.0),
                    Color32::from_gray(180),
                );
            }
        }
        if let Some(toggle) = geometry.toggle_rect_laned(scene, lane_active) {
            // v1-style on/off text immediately left of the scene checkbox.
            if region.width() > 58.0 {
                let label = if scene.enabled { "on" } else { "off" };
                painter.with_clip_rect(region).text(
                    Pos2::new(toggle.left() - 4.0, toggle.center().y),
                    Align2::RIGHT_CENTER,
                    label,
                    FontId::proportional(11.0),
                    if scene.enabled {
                        colors.accent
                    } else {
                        colors.muted
                    },
                );
            }
            match (input.toggle_on, input.toggle_off) {
                (Some(on), Some(off)) => {
                    let texture = if scene.enabled { on } else { off };
                    painter.image(
                        texture,
                        toggle,
                        egui::Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                        Color32::WHITE,
                    );
                }
                _ => {
                    painter.rect_filled(
                        toggle,
                        3.0,
                        if scene.enabled {
                            Color32::from_gray(202)
                        } else {
                            Color32::from_gray(18)
                        },
                    );
                    painter.rect_stroke(
                        toggle,
                        3.0,
                        Stroke::new(1.0_f32, Color32::from_white_alpha(70)),
                        StrokeKind::Inside,
                    );
                    if scene.enabled {
                        painter.line_segment(
                            [
                                toggle.left_center() + Vec2::new(3.0, 0.0),
                                toggle.center() + Vec2::new(-1.0, 3.0),
                            ],
                            Stroke::new(1.6_f32, Color32::from_gray(13)),
                        );
                        painter.line_segment(
                            [
                                toggle.center() + Vec2::new(-1.0, 3.0),
                                toggle.right_top() + Vec2::new(-3.0, 4.0),
                            ],
                            Stroke::new(1.6_f32, Color32::from_gray(13)),
                        );
                    }
                }
            }
        }
        if let Some(lane) = input.mask_lane.as_ref()
            && lane.scene == Some(index)
        {
            let band = geometry.mask_lane_rect(scene);
            painter.rect_filled(band, 2.0, Color32::from_gray(28));
            painter.rect_stroke(
                band,
                2.0,
                Stroke::new(1.0_f32, Color32::from_gray(64)),
                StrokeKind::Inside,
            );
            let y = band.center().y;
            for &key in lane.keyframes {
                if key < 0 || key as u64 >= input.frame_count {
                    continue;
                }
                let x = geometry.frame_x(key as u64);
                if x < region.left() - 4.0 || x > region.right() + 4.0 {
                    continue;
                }
                let selected = lane.selected.contains(&key);
                let radius = if selected { 5.0 } else { 4.0 };
                let color = if selected { colors.accent } else { colors.mask };
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        Pos2::new(x, y - radius),
                        Pos2::new(x + radius, y),
                        Pos2::new(x, y + radius),
                        Pos2::new(x - radius, y),
                    ],
                    color,
                    Stroke::new(1.0_f32, Color32::from_black_alpha(140)),
                ));
            }
            if let Some(MaskTimelineDrag::Marquee { start_frame, .. }) = lane.drag.as_ref()
                && let Some(pos) = response.interact_pointer_pos()
            {
                let x0 = geometry.frame_x((*start_frame).max(0.0) as u64);
                let x1 = pos.x;
                let marquee = Rect::from_min_max(
                    Pos2::new(x0.min(x1), band.top()),
                    Pos2::new(x0.max(x1), band.bottom()),
                );
                painter.rect_filled(
                    marquee,
                    1.0,
                    Color32::from_rgba_unmultiplied(88, 182, 242, 36),
                );
                painter.rect_stroke(
                    marquee,
                    1.0,
                    Stroke::new(1.0_f32, colors.accent),
                    StrokeKind::Inside,
                );
            }
        }
    }
    let step = ruler_step(geometry.span, inner.width());
    let mut frame = (geometry.offset / step as f64).ceil() as u64 * step;
    while frame < input.frame_count && frame as f64 <= geometry.offset + geometry.span {
        painter.text(
            Pos2::new(geometry.frame_x(frame) + 3.0, inner.top() + 1.0),
            Align2::LEFT_TOP,
            format!("{}", frame + 1),
            FontId::monospace(13.0),
            Color32::from_gray(195),
        );
        painter.line_segment(
            [
                Pos2::new(geometry.frame_x(frame), inner.top() + 15.0),
                Pos2::new(geometry.frame_x(frame), inner.top() + RULER_HEIGHT),
            ],
            Stroke::new(1.0_f32, Color32::from_gray(105)),
        );
        let Some(next) = frame.checked_add(step) else {
            break;
        };
        frame = next;
    }
    // v1-style RAM cache lanes: media frames (blue) above result frames (green).
    let cache_top = inner.bottom() - CACHE_LANES_HEIGHT;
    painter.rect_filled(
        Rect::from_min_max(
            Pos2::new(inner.left(), cache_top),
            Pos2::new(inner.right(), inner.bottom()),
        ),
        2.0,
        Color32::from_gray(21),
    );
    for (frames, y, color) in [
        (input.ram_media, cache_top + 2.0, colors.accent),
        (input.ram_results, cache_top + 7.0, colors.mask),
    ] {
        for (start, end) in cache_runs(frames) {
            let left = geometry.frame_x(start).max(inner.left());
            let right = geometry.frame_x(end.saturating_add(1)).min(inner.right());
            if right <= left {
                continue;
            }
            painter.rect_filled(
                Rect::from_min_max(Pos2::new(left, y), Pos2::new(right, y + 3.0)),
                1.0,
                color,
            );
        }
    }
    let _ = input.lanes;
    painter.rect_filled(geometry.playback_lane(), 2.0, Color32::from_gray(15));
    let playback_body = geometry.playback_body(input.range);
    if playback_body.width() > 0.0 {
        painter.rect_filled(playback_body, 2.0, Color32::from_gray(78));
        painter.rect_stroke(
            playback_body,
            2.0,
            Stroke::new(1.0_f32, Color32::from_gray(120)),
            StrokeKind::Inside,
        );
    }
    for boundary in [input.range.start, input.range.end_exclusive] {
        if let Some(handle) = geometry.playback_handle(boundary) {
            painter.rect_filled(handle, 2.0, Color32::from_gray(206));
            painter.rect_stroke(
                handle,
                2.0,
                Stroke::new(1.0_f32, colors.accent),
                StrokeKind::Inside,
            );
            let x = geometry
                .frame_x(boundary)
                .clamp(handle.left() + 1.0, handle.right() - 1.0);
            painter.line_segment(
                [
                    Pos2::new(x, handle.top() + 3.0),
                    Pos2::new(x, handle.bottom() - 3.0),
                ],
                Stroke::new(1.0_f32, Color32::from_gray(48)),
            );
        }
    }
    let play_x = geometry.frame_x(input.playhead);
    if play_x >= inner.left() && play_x <= inner.right() {
        let line = [
            Pos2::new(play_x, inner.top()),
            Pos2::new(play_x, inner.bottom()),
        ];
        painter.line_segment(line, Stroke::new(5.0_f32, Color32::from_gray(10)));
        painter.line_segment(line, Stroke::new(2.5_f32, colors.accent));
        painter.add(egui::Shape::convex_polygon(
            geometry.playhead_cap(input.playhead).to_vec(),
            colors.accent,
            Stroke::new(1.0_f32, Color32::from_gray(10)),
        ));
    }

    // Mask keyframe lane: click select, drag empty space for a marquee, drag a
    // selected diamond to move all selected keyframes.
    let mut mask_action: Option<TrackAction> = None;
    let mask_gesture_id = response.id.with("mask_keyframe_gesture");
    let mut mask_gesture_active = ui
        .data_mut(|data| data.get_temp::<bool>(mask_gesture_id))
        .unwrap_or(false);
    let mut mask_gesture_lane = false;
    // Keep the lane attached to the input: hit-testing the scene toggle after
    // this block must use the same lane-aware toggle rectangle as painting.
    if let Some(lane) = input.mask_lane.as_mut() {
        let lane_rect = lane
            .scene
            .and_then(|index| input.scenes.get(index))
            .map(|scene| geometry.mask_lane_rect(scene));
        let pointer = response
            .interact_pointer_pos()
            .or_else(|| response.hover_pos());
        if let (Some(rect), Some(pos)) = (lane_rect, pointer)
            && rect.contains(pos)
        {
            mask_gesture_lane = true;
        }
        let hit_key = |pos: Pos2| -> Option<i64> {
            lane.keyframes.iter().copied().find(|key| {
                let x = geometry.frame_x((*key).max(0) as u64);
                (x - pos.x).abs() <= 6.0
            })
        };
        if !mask_gesture_active
            && response.drag_started_by(egui::PointerButton::Primary)
            && mask_gesture_lane
            && let Some(pos) = response.interact_pointer_pos()
        {
            // The gesture belongs to the press origin, not the first position
            // past egui's drag threshold.
            let origin = ui
                .input(|state| state.pointer.press_origin())
                .unwrap_or(pos);
            let start_frame = geometry.pointer_frame(origin.x);
            let shift = ui.input(|state| state.modifiers.shift);
            if let Some(key) = hit_key(origin) {
                if !lane.selected.contains(&key) {
                    lane.selected.clear();
                    lane.selected.insert(key);
                }
                *lane.drag = Some(MaskTimelineDrag::Move { start_frame });
            } else {
                *lane.drag = Some(MaskTimelineDrag::Marquee { start_frame, shift });
            }
            mask_gesture_active = true;
            ui.data_mut(|data| data.insert_temp(mask_gesture_id, true));
            ui.ctx().request_repaint();
        }
        if mask_gesture_active {
            let released =
                ui.input(|state| state.pointer.button_released(egui::PointerButton::Primary));
            let down = ui.input(|state| state.pointer.primary_down());
            if let Some(pos) = pointer
                && let Some(drag_state) = lane.drag.as_mut()
            {
                match drag_state {
                    MaskTimelineDrag::Marquee { start_frame, shift } => {
                        mask_marquee_selection(
                            lane.keyframes,
                            *start_frame,
                            geometry.pointer_frame(pos.x),
                            *shift,
                            lane.selected,
                        );
                    }
                    MaskTimelineDrag::Move { start_frame } => {
                        let delta = (geometry.pointer_frame(pos.x) - *start_frame).round() as i64;
                        if released && delta != 0 {
                            mask_action = Some(TrackAction::MoveMaskKeys { delta });
                        }
                    }
                }
            }
            if released || !down {
                *lane.drag = None;
                mask_gesture_active = false;
                ui.data_mut(|data| data.insert_temp(mask_gesture_id, false));
            } else {
                ui.ctx().request_repaint();
            }
        } else if response.clicked_by(egui::PointerButton::Primary)
            && mask_gesture_lane
            && let Some(pos) = response.interact_pointer_pos()
        {
            let shift = ui.input(|state| state.modifiers.shift);
            match hit_key(pos) {
                Some(key) => {
                    if shift {
                        if !lane.selected.remove(&key) {
                            lane.selected.insert(key);
                        }
                    } else {
                        lane.selected.clear();
                        lane.selected.insert(key);
                    }
                }
                None if !shift => {
                    lane.selected.clear();
                }
                None => {}
            }
        }
        if mask_gesture_lane || mask_gesture_active {
            response.clone().on_hover_cursor(if mask_gesture_active {
                egui::CursorIcon::Grabbing
            } else {
                egui::CursorIcon::PointingHand
            });
        }
    }
    let playback_drag_id = response.id.with("playback_drag");
    let mut playback_drag = ui.data_mut(|data| data.get_temp::<PlaybackDrag>(playback_drag_id));
    if !mask_gesture_active
        && response.is_pointer_button_down_on()
        && ui.input(|state| state.pointer.button_pressed(egui::PointerButton::Primary))
        && let Some(pos) = response.interact_pointer_pos()
        && let Some(mode) = geometry.playback_hit(input.range, pos)
    {
        let drag = PlaybackDrag {
            mode,
            original: input.range,
            pointer_frame: geometry.pointer_frame(pos.x),
        };
        playback_drag = Some(drag);
        ui.data_mut(|data| data.insert_temp(playback_drag_id, drag));
    }
    let owns_playback_gesture = playback_drag.is_some() || mask_gesture_active;
    let mut action = mask_action;
    if let Some(drag) = playback_drag {
        let (down, released, pointer) = ui.input(|state| {
            (
                state.pointer.primary_down(),
                state.pointer.button_released(egui::PointerButton::Primary),
                state.pointer.interact_pos(),
            )
        });
        if (down || released)
            && let Some(pos) = pointer
        {
            let range = moved_playback_range(
                drag.original,
                drag.mode,
                geometry.pointer_frame(pos.x) - drag.pointer_frame,
                input.frame_count,
            );
            if range != input.range {
                action = Some(TrackAction::SetPlaybackRange(range));
            }
        }
        if released || !down {
            ui.data_mut(|data| data.remove::<PlaybackDrag>(playback_drag_id));
        }
    }
    if let Some(pos) = response.hover_pos()
        && let Some(mode) = geometry.playback_hit(input.range, pos)
    {
        response.clone().on_hover_cursor(match mode {
            RangeDragMode::In | RangeDragMode::Out => egui::CursorIcon::ResizeHorizontal,
            RangeDragMode::Move => egui::CursorIcon::Grab,
        });
    }

    let gesture_id = response.id.with("scene_toggle_gesture");
    let mut toggle_gesture = ui
        .data_mut(|data| data.get_temp::<bool>(gesture_id))
        .unwrap_or(false);
    if response.drag_started_by(egui::PointerButton::Primary) {
        toggle_gesture = ui.input(|state| {
            state.pointer.press_origin().is_some_and(|pos| {
                geometry.toggle_at(&input, pos).is_some()
                    || state.modifiers.alt && geometry.scene_at(&input, pos).is_some()
            })
        });
    }
    if !owns_playback_gesture
        && !mask_gesture_active
        && !mask_gesture_lane
        && response.clicked_by(egui::PointerButton::Primary)
        && let Some(pos) = response.interact_pointer_pos()
    {
        action = geometry.primary_click(
            &input,
            pos,
            ui.input(|state| state.modifiers.alt),
            response.double_clicked_by(egui::PointerButton::Primary),
        );
    } else if !owns_playback_gesture
        && !mask_gesture_active
        && !mask_gesture_lane
        && response.dragged_by(egui::PointerButton::Primary)
        && !toggle_gesture
        && let Some(pos) = response.interact_pointer_pos()
    {
        action = geometry.frame_at_x(pos.x).map(TrackAction::Seek);
    }
    if response.drag_stopped_by(egui::PointerButton::Primary) {
        toggle_gesture = false;
    }
    ui.data_mut(|data| data.insert_temp(gesture_id, toggle_gesture));

    let hovered = response
        .interact_pointer_pos()
        .or_else(|| response.hover_pos())
        .and_then(|pos| geometry.scene_at(&input, pos));
    let menu_scene_id = response.id.with("context_scene");
    if response.secondary_clicked() {
        ui.data_mut(|data| {
            data.remove::<usize>(menu_scene_id);
            if let Some(index) = hovered {
                data.insert_temp(menu_scene_id, index);
            }
        });
    }
    let menu_scene = ui.data_mut(|data| data.get_temp::<usize>(menu_scene_id));
    response.context_menu(|ui| {
        if let Some(index) = menu_scene.or(hovered)
            && let Some(scene) = input.scenes.get(index)
        {
            ui.label(format!(
                "Scene {} · frames {}–{}",
                scene.index + 1,
                scene.start + 1,
                scene.end_exclusive
            ));
            if ui
                .add_enabled(
                    input.scene_controls_enabled,
                    egui::Button::new(if scene.enabled {
                        "Disable scene keying"
                    } else {
                        "Enable scene keying"
                    }),
                )
                .clicked()
            {
                action = Some(TrackAction::SetSceneEnabled {
                    scene: index,
                    enabled: !scene.enabled,
                });
                ui.close();
            }
            if ui.button("Set range to this scene").clicked() {
                action = Some(TrackAction::SelectScene(index));
                ui.close();
            }
            ui.separator();
            ui.label("Overrides");
            if let Some(edit) = scene_despill_menu(ui, &input, index) {
                action = Some(edit);
            }
        } else {
            ui.label("No scene at this frame");
        }
        ui.separator();
        ui.label("Scene at playhead");
        let current = scene_index_at_frame(input.scenes, input.playhead);
        let can_split = current.is_some_and(|index| input.playhead > input.scenes[index].start);
        for (label, available, edit) in [
            ("Split at playhead", can_split, TrackAction::SplitScene),
            (
                "Merge current with next",
                current.is_some_and(|index| index + 1 < input.scenes.len()),
                TrackAction::MergeNext,
            ),
            (
                "Merge current with previous",
                current.is_some_and(|index| index > 0),
                TrackAction::MergePrevious,
            ),
        ] {
            if ui
                .add_enabled(
                    input.scene_controls_enabled && available,
                    egui::Button::new(label),
                )
                .clicked()
            {
                action = Some(edit);
                ui.close();
            }
        }
    });
    if let Some(index) = hovered {
        let scene = &input.scenes[index];
        response.on_hover_text(format!(
            "Scene {} · frames {}–{} · keying {}\nClick its checkbox or Alt-click to toggle.\nClick/drag to scrub · double-click to set the scene range.\nDrag work-area handles to resize; drag its bar to move the playback range.\nWheel: zoom ({:.1}×) · middle-drag or Shift-wheel: pan.",
            scene.index + 1,
            scene.start + 1,
            scene.end_exclusive,
            if scene.enabled { "on" } else { "off" },
            view.zoom_factor(),
        ));
    }
    show_pan_bar(ui, component, component_response.id.with("pan_bar"), view);
    action
}

fn pan_bar_rect(component: Rect) -> Rect {
    Rect::from_min_max(
        Pos2::new(component.left() + 18.0, component.bottom() - 16.0),
        Pos2::new(component.right() - 18.0, component.bottom()),
    )
}

fn pan_thumb_rect(rail: Rect, view: &TimelineView) -> Rect {
    let fraction = if view.frames == 0 {
        1.0
    } else {
        view.visible_frames() / view.frames as f64
    };
    let width = (rail.width() * fraction as f32).clamp(20.0_f32.min(rail.width()), rail.width());
    let max_offset = (view.frames as f64 - view.visible_frames()).max(0.0);
    let position = if max_offset == 0.0 {
        0.0
    } else {
        view.offset / max_offset
    };
    let left = rail.left() + (rail.width() - width) * position as f32;
    Rect::from_min_size(Pos2::new(left, rail.top()), Vec2::new(width, rail.height()))
}

fn show_pan_bar(ui: &mut egui::Ui, component: Rect, id: egui::Id, view: &mut TimelineView) {
    let rail = pan_bar_rect(component);
    if rail.width() <= 0.0 {
        return;
    }
    let response = ui.interact(rail, id, Sense::click_and_drag());
    let thumb = pan_thumb_rect(rail, view);
    let max_offset = (view.frames as f64 - view.visible_frames()).max(0.0);
    let travel = rail.width() - thumb.width();
    let grip_id = id.with("drag_origin");
    let before = view.offset;
    if max_offset > 0.0 && travel > 0.0 {
        if response.is_pointer_button_down_on()
            && ui.input(|state| state.pointer.button_pressed(egui::PointerButton::Primary))
            && let Some(pos) = response.interact_pointer_pos()
        {
            if !thumb.contains(pos) {
                view.offset =
                    ((pos.x - rail.left() - thumb.width() * 0.5) / travel) as f64 * max_offset;
                view.clamp_offset();
            }
            ui.data_mut(|data| data.insert_temp(grip_id, (view.offset, pos.x)));
        }
        if response.dragged_by(egui::PointerButton::Primary)
            && let Some(pos) = response.interact_pointer_pos()
            && let Some((offset, origin_x)) =
                ui.data_mut(|data| data.get_temp::<(f64, f32)>(grip_id))
        {
            view.offset = offset + (pos.x - origin_x) as f64 / travel as f64 * max_offset;
            view.clamp_offset();
        }
    }
    if response.drag_stopped_by(egui::PointerButton::Primary) {
        ui.data_mut(|data| data.remove::<(f64, f32)>(grip_id));
    }
    if before != view.offset {
        ui.ctx().request_repaint();
    }
    let painter = ui
        .painter()
        .with_clip_rect(ui.clip_rect().intersect(component));
    painter.rect_filled(rail, 3.0, Color32::from_gray(18));
    painter.rect_stroke(
        rail,
        3.0,
        Stroke::new(1.0_f32, Color32::from_gray(48)),
        StrokeKind::Inside,
    );
    let thumb = pan_thumb_rect(rail, view).shrink(2.0);
    painter.rect_filled(
        thumb,
        2.0,
        Color32::from_gray(if max_offset == 0.0 {
            35
        } else if response.dragged() || response.hovered() {
            92
        } else {
            70
        }),
    );
    response
        .on_hover_text("Pan the visible timeline. Wheel zooms; middle-drag or Shift-wheel pans.");
}

fn ruler_step(span: f64, width: f32) -> u64 {
    let target_ticks = (width as f64 / 80.0).clamp(2.0, 20.0);
    let raw = (span / target_ticks).max(1.0);
    let magnitude = 10.0_f64.powf(raw.log10().floor());
    let scaled = raw / magnitude;
    let factor = [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .find(|&value| value >= scaled)
        .unwrap_or(10.0);
    (factor * magnitude).max(1.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CpuTimeline {
        context: egui::Context,
        time: f64,
        view: TimelineView,
        range: FrameRange,
        texts: Vec<(String, Rect)>,
    }

    impl CpuTimeline {
        fn new(scenes: &[SceneRange]) -> Self {
            let mut timeline = Self {
                context: egui::Context::default(),
                time: 1.0,
                view: TimelineView::default(),
                range: FrameRange {
                    start: 0,
                    end_exclusive: 120,
                },
                texts: Vec::new(),
            };
            // Warm the CPU layout/font pass before sending pointer input. No
            // native window, WGPU adapter or inference worker is constructed.
            timeline.frame(scenes, Vec::new(), false);
            timeline
        }

        fn frame(
            &mut self,
            scenes: &[SceneRange],
            events: Vec<egui::Event>,
            alt: bool,
        ) -> Option<TrackAction> {
            let mut action = None;
            self.time += 0.02;
            let raw = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 360.0))),
                time: Some(self.time),
                events,
                modifiers: egui::Modifiers {
                    alt,
                    ..Default::default()
                },
                ..Default::default()
            };
            let range = self.range;
            let output = self.context.run(raw, |context| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(context, |ui| {
                        let mut request = input(scenes);
                        request.range = range;
                        action = show_track(ui, request, TrackColors::default(), &mut self.view);
                    });
            });
            self.texts.clear();
            for shape in &output.shapes {
                collect_text(&shape.shape, &mut self.texts);
            }
            if let Some(TrackAction::SetPlaybackRange(range)) = action {
                self.range = range;
            }
            action
        }

        fn button(
            &mut self,
            scenes: &[SceneRange],
            pos: Pos2,
            pressed: bool,
            alt: bool,
        ) -> Option<TrackAction> {
            self.frame(
                scenes,
                vec![
                    egui::Event::PointerMoved(pos),
                    egui::Event::PointerButton {
                        pos,
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: egui::Modifiers {
                            alt,
                            ..Default::default()
                        },
                    },
                ],
                alt,
            )
        }
    }

    fn collect_text(shape: &egui::epaint::Shape, output: &mut Vec<(String, Rect)>) {
        match shape {
            egui::epaint::Shape::Text(text) => {
                output.push((text.galley.job.text.clone(), text.visual_bounding_rect()))
            }
            egui::epaint::Shape::Vec(shapes) => {
                for shape in shapes {
                    collect_text(shape, output);
                }
            }
            _ => {}
        }
    }

    fn scenes() -> Vec<SceneRange> {
        vec![
            SceneRange {
                index: 17,
                start: 0,
                end_exclusive: 50,
                enabled: true,
            },
            SceneRange {
                index: 28,
                start: 50,
                end_exclusive: 120,
                enabled: true,
            },
        ]
    }

    static EMPTY_RAM_FRAMES: std::sync::LazyLock<std::collections::BTreeSet<u64>> =
        std::sync::LazyLock::new(std::collections::BTreeSet::new);

    fn input(scenes: &[SceneRange]) -> TrackInput<'_> {
        TrackInput {
            frame_count: 120,
            playhead: 74,
            range: FrameRange {
                start: 0,
                end_exclusive: 120,
            },
            scenes,
            lanes: &[],
            scene_controls_enabled: true,
            toggle_on: None,
            toggle_off: None,
            mask_lane: None,
            scene_despill: &[],
            global_despill: 0.0,
            ram_media: &EMPTY_RAM_FRAMES,
            ram_results: &EMPTY_RAM_FRAMES,
            height: MIN_TRACK_HEIGHT,
        }
    }

    fn geometry() -> TrackGeometry {
        TrackGeometry {
            inner: Rect::from_min_max(
                Pos2::new(20.0, 0.0),
                Pos2::new(620.0, MIN_TRACK_HEIGHT - PAN_BAR_SPACE),
            ),
            frames: 120,
            offset: 0.0,
            span: 120.0,
        }
    }

    fn ui_geometry() -> TrackGeometry {
        TrackGeometry {
            inner: Rect::from_min_max(
                Pos2::new(18.0, 0.0),
                Pos2::new(622.0, MIN_TRACK_HEIGHT - PAN_BAR_SPACE),
            ),
            ..geometry()
        }
    }

    #[test]
    fn taller_scene_body_and_controls_remain_separate_and_inside_component_bounds() {
        let scenes = scenes();
        for height in [MIN_TRACK_HEIGHT, 140.0, 220.0] {
            let geometry = TrackGeometry {
                inner: Rect::from_min_max(
                    Pos2::new(18.0, 0.0),
                    Pos2::new(622.0, height - PAN_BAR_SPACE),
                ),
                ..geometry()
            };
            assert_eq!(geometry.ruler_rect().height(), 20.0);
            assert!(geometry.playback_lane().top() >= geometry.ruler_rect().bottom() + 2.0);
            for scene in &scenes {
                let body = geometry.scene_rect(scene);
                let checkbox = geometry.toggle_rect(scene).unwrap();
                assert!(
                    body.height() >= 34.0,
                    "scene body must hold a readable label and18px control"
                );
                assert_eq!(checkbox.size(), Vec2::splat(18.0));
                assert!(body.shrink(4.0).contains_rect(checkbox));
                assert_eq!(checkbox.center().y, body.center().y);
                assert!(
                    body.top() >= geometry.playback_handle(scene.start).unwrap().bottom() + 4.0
                );
                assert!(body.bottom() <= geometry.inner.bottom() - CACHE_LANES_HEIGHT);
                assert!(geometry.inner.contains_rect(body));
            }
            let cap = geometry.playhead_cap(74);
            assert_eq!(cap[1].x - cap[0].x, 12.0);
            assert_eq!(cap[3].x, geometry.frame_x(74));
            assert_eq!(cap[3].y - cap[0].y, 13.0);
            assert!(
                cap.iter()
                    .all(|point| geometry.ruler_rect().contains(*point))
            );
        }
    }

    #[test]
    fn cpu_expanded_scene_draws_source_frame_span_inside_readable_body() {
        let context = egui::Context::default();
        crate::theme::configure(&context);
        let scenes = scenes();
        let mut view = TimelineView::default();
        let output = context.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 300.0))),
                ..Default::default()
            },
            |context| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(context, |ui| {
                        // CentralPanel deliberately fills the entire screen;
                        // measure the component in its own ordinary child UI.
                        let mut track_ui = ui.new_child(
                            egui::UiBuilder::new()
                                .id_salt("expanded_scene_readability")
                                .max_rect(ui.max_rect())
                                .layout(egui::Layout::top_down(egui::Align::Min)),
                        );
                        let mut request = input(&scenes);
                        request.height = 140.0;
                        show_track(&mut track_ui, request, TrackColors::default(), &mut view);
                        assert_eq!(track_ui.min_rect().height(), 140.0);
                    });
            },
        );
        let mut texts = Vec::new();
        for shape in output.shapes {
            collect_text(&shape.shape, &mut texts);
        }
        let scene_rect = TrackGeometry {
            inner: Rect::from_min_max(
                Pos2::new(18.0, 0.0),
                Pos2::new(622.0, 140.0 - PAN_BAR_SPACE),
            ),
            ..geometry()
        }
        .scene_rect(&scenes[0]);
        for label in ["Scene 18", "1–50"] {
            let (_, rect) = texts
                .iter()
                .find(|(text, _)| text == label)
                .unwrap_or_else(|| panic!("missing readable scene text: {label}"));
            assert!(
                scene_rect.contains_rect(*rect),
                "scene text must fit the allocated scene body"
            );
        }
    }

    #[test]
    fn cpu_pointer_motion_across_larger_controls_keeps_navigation_and_range_unchanged() {
        let scenes = scenes();
        let mut timeline = CpuTimeline::new(&scenes);
        timeline.view.zoom_at_fraction(120, 0.5, 4.0);
        let before = timeline.view.visible_range();
        let range = timeline.range;
        for pass in 0..120 {
            let pointer = Pos2::new(
                18.0 + (pass * 37 % 604) as f32,
                (pass * 17 % MIN_TRACK_HEIGHT as usize) as f32,
            );
            assert_eq!(
                timeline.frame(&scenes, vec![egui::Event::PointerMoved(pointer)], false),
                None
            );
            assert_eq!(timeline.view.visible_range(), before);
            assert_eq!(timeline.range, range);
            assert_eq!(timeline.view.last_playhead, Some(74));
        }
    }

    #[test]
    fn current_scene_uses_half_open_source_frame_boundaries() {
        let scenes = scenes();
        assert_eq!(scene_index_at_frame(&scenes, 0), Some(0));
        assert_eq!(scene_index_at_frame(&scenes, 49), Some(0));
        assert_eq!(scene_index_at_frame(&scenes, 50), Some(1));
        assert_eq!(scene_index_at_frame(&scenes, 119), Some(1));
        assert_eq!(scene_index_at_frame(&scenes, 120), None);
        assert_eq!(
            enabled_scene_range_at_frame(&scenes, 74),
            Ok(FrameRange {
                start: 50,
                end_exclusive: 120
            })
        );
    }

    #[test]
    fn disabled_scene_never_falls_back_to_another_enabled_scene() {
        let mut scenes = scenes();
        scenes[1].enabled = false;
        assert_eq!(scene_index_at_frame(&scenes, 74), Some(1));
        assert_eq!(
            enabled_scene_range_at_frame(&scenes, 74),
            Err(SceneRunError::Disabled(1))
        );
        assert_eq!(
            enabled_scene_range_at_frame(&scenes, 49),
            Ok(scenes[0].range())
        );
        assert_eq!(
            enabled_scene_range_at_frame(&scenes, 120),
            Err(SceneRunError::NoScene)
        );
    }

    #[test]
    fn gaps_and_invalid_empty_scenes_are_not_a_current_scene() {
        let mut scenes = scenes();
        scenes[0].end_exclusive = 40;
        scenes.push(SceneRange {
            index: 99,
            start: 120,
            end_exclusive: 120,
            enabled: true,
        });
        assert_eq!(scene_index_at_frame(&scenes, 40), None);
        assert_eq!(
            enabled_scene_range_at_frame(&scenes, 45),
            Err(SceneRunError::NoScene)
        );
        assert_eq!(scene_index_at_frame(&scenes, 120), None);
    }

    #[test]
    fn checkbox_changes_enable_state_without_seeking() {
        let scenes = scenes();
        let input = input(&scenes);
        let geometry = geometry();
        let click = geometry.toggle_rect(&scenes[1]).unwrap().center();
        assert_eq!(
            geometry.primary_click(&input, click, false, false),
            Some(TrackAction::SetSceneEnabled {
                scene: 1,
                enabled: false
            })
        );
        assert_eq!(input.playhead, 74);
    }

    #[test]
    fn scrubbing_scene_body_does_not_toggle_it() {
        let scenes = scenes();
        let input = input(&scenes);
        assert_eq!(
            geometry().primary_click(
                &input,
                Pos2::new(geometry().frame_x(75), 45.0),
                false,
                false
            ),
            Some(TrackAction::Seek(75))
        );
        assert_eq!(
            geometry().primary_click(&input, Pos2::new(geometry().frame_x(75), 45.0), false, true),
            Some(TrackAction::SelectScene(1))
        );
    }

    #[test]
    fn narrow_scene_has_alt_toggle_when_checkbox_does_not_fit() {
        let scenes = vec![SceneRange {
            index: 9,
            start: 50,
            end_exclusive: 51,
            enabled: false,
        }];
        let input = input(&scenes);
        let geometry = geometry();
        assert!(geometry.toggle_rect(&scenes[0]).is_none());
        let pos = Pos2::new(
            geometry.frame_x(50) + 2.0,
            geometry.scene_rect(&scenes[0]).center().y,
        );
        assert_eq!(
            geometry.primary_click(&input, pos, true, false),
            Some(TrackAction::SetSceneEnabled {
                scene: 0,
                enabled: true
            })
        );
        assert_eq!(
            geometry.primary_click(&input, pos, false, false),
            Some(TrackAction::Seek(50))
        );
    }

    #[test]
    fn locked_toggle_does_not_become_a_seek() {
        let scenes = scenes();
        let mut input = input(&scenes);
        input.scene_controls_enabled = false;
        assert_eq!(
            geometry().primary_click(
                &input,
                geometry().toggle_rect(&scenes[1]).unwrap().center(),
                false,
                false
            ),
            None
        );
    }

    #[test]
    fn scrub_clamps_to_real_frames_and_empty_media_is_safe() {
        let geometry = geometry();
        for frame in 0..120 {
            assert_eq!(geometry.frame_at_x(geometry.frame_x(frame)), Some(frame));
        }
        assert_eq!(geometry.frame_at_x(-100.0), Some(0));
        assert_eq!(geometry.frame_at_x(620.0), Some(119));
        assert_eq!(geometry.frame_at_x(10_000.0), Some(119));
        assert_eq!(geometry.frame_at_x(f32::NAN), None);
        assert_eq!(
            TrackGeometry {
                frames: 0,
                ..geometry
            }
            .frame_at_x(100.0),
            None
        );
    }

    #[test]
    fn cpu_pointer_click_toggles_only_checkbox_and_uses_updated_scene_state() {
        let mut scenes = scenes();
        let mut timeline = CpuTimeline::new(&scenes);
        // Use the painted hit geometry: the 18-point control stays centered
        // inside the taller scene body at both minimum and expanded heights.
        let checkbox = ui_geometry().toggle_rect(&scenes[1]).unwrap().center();
        assert_eq!(timeline.button(&scenes, checkbox, true, false), None);
        let action = timeline.button(&scenes, checkbox, false, false);
        assert_eq!(
            action,
            Some(TrackAction::SetSceneEnabled {
                scene: 1,
                enabled: false,
            })
        );
        if let Some(TrackAction::SetSceneEnabled { scene, enabled }) = action {
            scenes[scene].enabled = enabled;
        }
        assert!(!scenes[1].enabled);
        // The next UI frame reads the model's new state instead of keeping a
        // private checkbox boolean. It still returns a toggle, never a seek.
        timeline.frame(&scenes, Vec::new(), false);
        assert_eq!(timeline.button(&scenes, checkbox, true, false), None);
        assert_eq!(
            timeline.button(&scenes, checkbox, false, false),
            Some(TrackAction::SetSceneEnabled {
                scene: 1,
                enabled: true,
            })
        );
    }

    #[test]
    fn cpu_pointer_drag_starting_in_checkbox_never_scrubs() {
        let scenes = scenes();
        let mut timeline = CpuTimeline::new(&scenes);
        let checkbox = ui_geometry().toggle_rect(&scenes[1]).unwrap().center();
        assert_eq!(timeline.button(&scenes, checkbox, true, false), None);
        for pos in [Pos2::new(450.0, 45.0), Pos2::new(100.0, 45.0)] {
            assert_eq!(
                timeline.frame(&scenes, vec![egui::Event::PointerMoved(pos)], false),
                None
            );
        }
        assert_eq!(
            timeline.button(&scenes, Pos2::new(100.0, 45.0), false, false),
            None
        );
        // After release, an ordinary body click can scrub again.
        let body = Pos2::new(395.5, 45.0);
        assert_eq!(timeline.button(&scenes, body, true, false), None);
        assert_eq!(
            timeline.button(&scenes, body, false, false),
            Some(TrackAction::Seek(75))
        );
    }

    #[test]
    fn cpu_alt_click_toggles_narrow_scene_without_moving_playhead() {
        let scenes = vec![SceneRange {
            index: 9,
            start: 50,
            end_exclusive: 51,
            enabled: false,
        }];
        let mut timeline = CpuTimeline::new(&scenes);
        let narrow_body = Pos2::new(271.5, ui_geometry().scene_rect(&scenes[0]).center().y);
        assert_eq!(timeline.button(&scenes, narrow_body, true, true), None);
        assert_eq!(
            timeline.button(&scenes, narrow_body, false, true),
            Some(TrackAction::SetSceneEnabled {
                scene: 0,
                enabled: true,
            })
        );
    }

    #[test]
    fn zoom_keeps_the_cursor_source_frame_stationary_and_fit_restores_clip() {
        let mut view = TimelineView::default();
        view.prepare(120, 74);
        view.zoom_at_fraction(120, 0.3, 4.0);
        let (start, end) = view.visible_range();
        assert!((start - 27.0).abs() < 1e-9);
        assert!((end - 57.0).abs() < 1e-9);
        assert!((start + (end - start) * 0.3 - 36.0).abs() < 1e-9);
        view.fit_all();
        assert_eq!(view.visible_range(), (0.0, 120.0));
        assert_eq!(view.zoom_factor(), 1.0);
    }

    #[test]
    fn pan_clamps_to_source_and_zoom_can_reach_individual_frames() {
        let mut view = TimelineView::default();
        view.prepare(1_000_000, 500_000);
        view.zoom_at_fraction(1_000_000, 0.5, 1_000_000.0);
        let (start, end) = view.visible_range();
        assert!((end - start - 1.0).abs() < 1e-9);
        view.pan_pixels(1_000_000, 1e9, 600.0);
        assert_eq!(view.visible_range(), (0.0, 1.0));
        view.pan_pixels(1_000_000, -1e9, 600.0);
        assert_eq!(view.visible_range(), (999_999.0, 1_000_000.0));
        view.prepare(0, 0);
        assert_eq!(view.visible_range(), (0.0, 0.0));
        assert_eq!(view.zoom_factor(), 1.0);
    }

    #[test]
    fn stepping_follows_the_needle_but_a_stationary_needle_does_not_undo_pan() {
        let mut view = TimelineView::default();
        view.prepare(120, 74);
        view.zoom_at_fraction(120, 0.5, 4.0);
        view.pan_pixels(120, -10_000.0, 600.0);
        assert_eq!(view.visible_range(), (90.0, 120.0));
        view.prepare(120, 74);
        assert_eq!(view.visible_range(), (90.0, 120.0));
        view.prepare(120, 75);
        let (start, end) = view.visible_range();
        assert!(start <= 75.0 && end > 75.0);
        view.prepare(120, 0);
        assert_eq!(view.visible_range().0, 0.0);
        view.prepare(60, 20);
        assert_eq!(view.visible_range(), (0.0, 60.0));
    }

    #[test]
    fn zoomed_mapping_preserves_source_cut_boundaries_and_clips_scene_toggles() {
        let scenes = scenes();
        let input = input(&scenes);
        let geometry = TrackGeometry {
            offset: 40.0,
            span: 40.0,
            ..geometry()
        };
        assert_eq!(geometry.frame_at_x(geometry.frame_x(50)), Some(50));
        assert_eq!(
            geometry.primary_click(&input, Pos2::new(geometry.frame_x(50), 45.0), false, false),
            Some(TrackAction::Seek(50))
        );
        // Scene1 continues beyond the visible viewport, but its checkbox is
        // still available at the visible edge of the scene, inside the track.
        let toggle = geometry.toggle_rect(&scenes[1]).unwrap();
        assert!(geometry.inner.contains_rect(toggle));
        assert_eq!(
            geometry.primary_click(&input, toggle.center(), false, false),
            Some(TrackAction::SetSceneEnabled {
                scene: 1,
                enabled: false
            })
        );
        let fractional = TrackGeometry {
            offset: 13.25,
            span: 17.5,
            ..geometry
        };
        for frame in 14..=30 {
            assert_eq!(
                fractional.frame_at_x(fractional.frame_x(frame)),
                Some(frame)
            );
        }
        assert!(ruler_step(8.0, 600.0) < ruler_step(120.0, 600.0));
    }

    #[test]
    fn cpu_wheel_zooms_at_pointer_without_seeking_and_zoomed_checkbox_still_toggles() {
        let scenes = scenes();
        let mut timeline = CpuTimeline::new(&scenes);
        let pointer = Pos2::new(471.0, 45.0); // 75% of the visible canvas.
        assert_eq!(
            timeline.frame(
                &scenes,
                vec![
                    egui::Event::PointerMoved(pointer),
                    egui::Event::MouseWheel {
                        unit: egui::MouseWheelUnit::Line,
                        delta: Vec2::new(0.0, 1.0),
                        modifiers: egui::Modifiers::default(),
                    },
                ],
                false
            ),
            None
        );
        assert!(timeline.view.zoom_factor() > 1.0);
        let (start, end) = timeline.view.visible_range();
        assert!((start + (end - start) * 0.75 - 90.0).abs() < 1e-6);
        let checkbox = ui_geometry().toggle_rect(&scenes[1]).unwrap().center();
        assert_eq!(timeline.button(&scenes, checkbox, true, false), None);
        assert_eq!(
            timeline.button(&scenes, checkbox, false, false),
            Some(TrackAction::SetSceneEnabled {
                scene: 1,
                enabled: false
            })
        );
    }

    #[test]
    fn frame_range_centres_a_short_work_range_and_stays_in_bounds() {
        let mut view = TimelineView::default();
        view.frame_range(558, 0, 16);
        let (start, end) = view.visible_range();
        assert!(start == 0.0, "clamped at the clip start, got {start}");
        assert!((end - start - 16.0 / 0.7).abs() < 1e-6);
        view.frame_range(558, 300, 340);
        let (start, end) = view.visible_range();
        assert!(start < 300.0 && end > 340.0);
        assert!(((start + end) / 2.0 - 320.0).abs() < 1e-6);
        view.frame_range(558, 0, 558);
        assert_eq!(view.zoom_factor(), 1.0);
    }

    #[test]
    fn cpu_scene_context_menu_turns_on_a_despill_override() {
        let scenes = scenes();
        let mut timeline = CpuTimeline::new(&scenes);
        let scene = ui_geometry().scene_rect(&scenes[1]);
        let target = Pos2::new(scene.center().x, scene.bottom() - 6.0);
        for pressed in [true, false] {
            timeline.frame(
                &scenes,
                vec![
                    egui::Event::PointerMoved(target),
                    egui::Event::PointerButton {
                        pos: target,
                        button: egui::PointerButton::Secondary,
                        pressed,
                        modifiers: egui::Modifiers::default(),
                    },
                ],
                false,
            );
        }
        timeline.frame(&scenes, Vec::new(), false);
        let (_, item) = timeline
            .texts
            .iter()
            .find(|(text, _)| text == "Override Result despill")
            .cloned()
            .expect("scene menu lists the despill override");
        assert!(timeline.texts.iter().any(|(text, _)| text == "Overrides"));
        assert_eq!(timeline.button(&scenes, item.center(), true, false), None);
        assert_eq!(
            timeline.button(&scenes, item.center(), false, false),
            Some(TrackAction::SetSceneDespill {
                scene: 1,
                amount: Some(0.0)
            })
        );
    }

    #[test]
    fn cpu_middle_drag_pans_without_scrubbing_or_toggling_scene() {
        let scenes = scenes();
        let mut timeline = CpuTimeline::new(&scenes);
        timeline.view.zoom_at_fraction(120, 0.5, 4.0);
        let before = timeline.view.visible_range();
        let press = Pos2::new(400.0, 45.0);
        assert_eq!(
            timeline.frame(
                &scenes,
                vec![
                    egui::Event::PointerMoved(press),
                    egui::Event::PointerButton {
                        pos: press,
                        button: egui::PointerButton::Middle,
                        pressed: true,
                        modifiers: egui::Modifiers::default(),
                    },
                ],
                false
            ),
            None
        );
        let release = Pos2::new(300.0, 45.0);
        assert_eq!(
            timeline.frame(&scenes, vec![egui::Event::PointerMoved(release)], false),
            None
        );
        assert_eq!(
            timeline.frame(
                &scenes,
                vec![egui::Event::PointerButton {
                    pos: release,
                    button: egui::PointerButton::Middle,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                }],
                false
            ),
            None
        );
        assert!(timeline.view.visible_range().0 > before.0);
        assert_eq!(timeline.view.last_playhead, Some(74));
    }

    #[test]
    fn cpu_multiframe_middle_pan_uses_each_pointer_delta_once() {
        let scenes = scenes();
        let mut timeline = CpuTimeline::new(&scenes);
        timeline.view.zoom_at_fraction(120, 0.5, 4.0);
        let before = timeline.view.visible_range().0;
        let press = Pos2::new(400.0, 45.0);
        timeline.frame(
            &scenes,
            vec![
                egui::Event::PointerMoved(press),
                egui::Event::PointerButton {
                    pos: press,
                    button: egui::PointerButton::Middle,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
            false,
        );
        for x in [390.0, 380.0, 370.0] {
            assert_eq!(
                timeline.frame(
                    &scenes,
                    vec![egui::Event::PointerMoved(Pos2::new(x, 45.0))],
                    false
                ),
                None
            );
        }
        let expected = before + 30.0 / 604.0 * 30.0;
        assert!((timeline.view.visible_range().0 - expected).abs() < 1e-9);
    }

    #[test]
    fn cpu_component_height_is_stable_with_real_theme_widget_spacing() {
        let context = egui::Context::default();
        crate::theme::configure(&context);
        let scenes = scenes();
        let mut view = TimelineView::default();
        let mut time = 1.0;
        for initial_height in [MIN_TRACK_HEIGHT, MIN_TRACK_HEIGHT + 24.0, 180.0] {
            let mut requested_height = initial_height;
            for pass in 0..10 {
                if pass == 3 {
                    view.zoom_at_fraction(120, 0.5, 4.0);
                }
                if pass == 6 {
                    view.fit_all();
                }
                time += 0.02;
                let mut measured = 0.0;
                let _ = context.run(
                    egui::RawInput {
                        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 300.0))),
                        time: Some(time),
                        events: vec![egui::Event::PointerMoved(Pos2::new(
                            40.0 + pass as f32 * 31.0,
                            50.0,
                        ))],
                        ..Default::default()
                    },
                    |context| {
                        egui::CentralPanel::default().frame(egui::Frame::NONE).show(
                            context,
                            |parent| {
                                let mut ui = parent.new_child(
                                    egui::UiBuilder::new()
                                        .id_salt("height_contract")
                                        .max_rect(Rect::from_min_size(
                                            Pos2::ZERO,
                                            Vec2::new(640.0, 300.0),
                                        ))
                                        .layout(egui::Layout::top_down(egui::Align::Min)),
                                );
                                // These actual studio metrics caused the former Slider
                                // row to exceed its claimed22-point height reservation.
                                assert_eq!(ui.spacing().item_spacing.y, 6.0);
                                assert_eq!(ui.spacing().interact_size.y, 28.0);
                                let mut request = input(&scenes);
                                request.height = requested_height;
                                show_track(&mut ui, request, TrackColors::default(), &mut view);
                                measured = ui.min_rect().height();
                            },
                        );
                    },
                );
                assert!(
                    (measured - requested_height).abs() < 0.01,
                    "component requested{requested_height}, allocated{measured}"
                );
                // Simulate a parent feeding the last content height back into
                // the next frame. It must not accumulate hidden layout pixels.
                requested_height = requested_height.max(measured);
                assert!((requested_height - initial_height).abs() < 0.01);
            }
        }
    }

    #[test]
    fn cpu_full_width_pan_thumb_drag_changes_only_visible_range() {
        let scenes = scenes();
        let mut timeline = CpuTimeline::new(&scenes);
        timeline.view.zoom_at_fraction(120, 0.5, 4.0);
        let component = Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, MIN_TRACK_HEIGHT));
        let rail = pan_bar_rect(component);
        assert_eq!(rail.width(), 604.0);
        let thumb = pan_thumb_rect(rail, &timeline.view);
        assert_eq!(thumb.width(), 151.0);
        let before = timeline.view.visible_range().0;
        let press = thumb.center();
        assert_eq!(timeline.button(&scenes, press, true, false), None);
        let drag = press + Vec2::new(100.0, 0.0);
        assert_eq!(
            timeline.frame(&scenes, vec![egui::Event::PointerMoved(drag)], false),
            None
        );
        assert_eq!(timeline.button(&scenes, drag, false, false), None);
        let expected = before + 100.0 / (604.0 - 151.0) * 90.0;
        assert!((timeline.view.visible_range().0 - expected).abs() < 1e-6);
        assert_eq!(timeline.view.last_playhead, Some(74));
    }

    #[test]
    fn playback_range_resize_and_move_keep_half_open_source_bounds() {
        let range = FrameRange {
            start: 20,
            end_exclusive: 80,
        };
        assert_eq!(
            moved_playback_range(range, RangeDragMode::In, 100.0, 120),
            FrameRange {
                start: 79,
                end_exclusive: 80
            }
        );
        assert_eq!(
            moved_playback_range(range, RangeDragMode::Out, -100.0, 120),
            FrameRange {
                start: 20,
                end_exclusive: 21
            }
        );
        assert_eq!(
            moved_playback_range(range, RangeDragMode::Move, -100.0, 120),
            FrameRange {
                start: 0,
                end_exclusive: 60
            }
        );
        assert_eq!(
            moved_playback_range(range, RangeDragMode::Move, 100.0, 120),
            FrameRange {
                start: 60,
                end_exclusive: 120
            }
        );
        assert_eq!(
            moved_playback_range(range, RangeDragMode::In, 0.49, 120),
            range
        );
        assert_eq!(
            moved_playback_range(range, RangeDragMode::In, 0.51, 120),
            FrameRange {
                start: 21,
                end_exclusive: 80
            }
        );
    }

    fn ui_frame_position(frame: u64, y: f32) -> Pos2 {
        Pos2::new((18.0 + frame as f64 / 120.0 * 604.0) as f32, y)
    }

    #[test]
    fn cpu_playback_handles_resize_without_seeking_or_collapsing_range() {
        let scenes = scenes();
        let mut timeline = CpuTimeline::new(&scenes);
        timeline.range = FrameRange {
            start: 20,
            end_exclusive: 80,
        };
        let start = ui_frame_position(20, 25.0);
        let moved = ui_frame_position(30, 25.0);
        assert_eq!(timeline.button(&scenes, start, true, false), None);
        assert_eq!(
            timeline.frame(&scenes, vec![egui::Event::PointerMoved(moved)], false),
            Some(TrackAction::SetPlaybackRange(FrameRange {
                start: 30,
                end_exclusive: 80
            }))
        );
        assert_eq!(timeline.button(&scenes, moved, false, false), None);
        assert_eq!(
            timeline.range,
            FrameRange {
                start: 30,
                end_exclusive: 80
            }
        );

        let end = ui_frame_position(80, 25.0);
        assert_eq!(timeline.button(&scenes, end, true, false), None);
        assert_eq!(
            timeline.frame(
                &scenes,
                vec![egui::Event::PointerMoved(Pos2::new(900.0, 25.0))],
                false
            ),
            Some(TrackAction::SetPlaybackRange(FrameRange {
                start: 30,
                end_exclusive: 120
            }))
        );
        timeline.button(&scenes, Pos2::new(900.0, 25.0), false, false);
        assert_eq!(
            timeline.range,
            FrameRange {
                start: 30,
                end_exclusive: 120
            }
        );
        assert_eq!(timeline.view.last_playhead, Some(74));
        assert!(scenes.iter().all(|scene| scene.enabled));
    }

    #[test]
    fn cpu_playback_bar_moves_entire_range_with_fixed_length_even_with_alt() {
        let scenes = scenes();
        let mut timeline = CpuTimeline::new(&scenes);
        timeline.range = FrameRange {
            start: 20,
            end_exclusive: 80,
        };
        assert_eq!(
            timeline.button(&scenes, ui_frame_position(50, 25.0), true, true),
            None
        );
        assert_eq!(
            timeline.frame(
                &scenes,
                vec![egui::Event::PointerMoved(ui_frame_position(10, 25.0))],
                true
            ),
            Some(TrackAction::SetPlaybackRange(FrameRange {
                start: 0,
                end_exclusive: 60
            }))
        );
        assert_eq!(
            timeline.frame(
                &scenes,
                vec![egui::Event::PointerMoved(ui_frame_position(100, 25.0))],
                true
            ),
            Some(TrackAction::SetPlaybackRange(FrameRange {
                start: 60,
                end_exclusive: 120
            }))
        );
        timeline.button(&scenes, ui_frame_position(100, 25.0), false, true);
        assert_eq!(timeline.range.len(), 60);
        assert_eq!(timeline.view.last_playhead, Some(74));
        assert!(scenes.iter().all(|scene| scene.enabled));
    }

    #[test]
    fn cpu_context_scene_edits_target_playhead_independently_of_clicked_scene() {
        let scenes = vec![
            SceneRange {
                index: 0,
                start: 0,
                end_exclusive: 40,
                enabled: true,
            },
            SceneRange {
                index: 1,
                start: 40,
                end_exclusive: 80,
                enabled: true,
            },
            SceneRange {
                index: 2,
                start: 80,
                end_exclusive: 120,
                enabled: true,
            },
        ];
        for (label, expected) in [
            ("Split at playhead", TrackAction::SplitScene),
            ("Merge current with next", TrackAction::MergeNext),
            ("Merge current with previous", TrackAction::MergePrevious),
        ] {
            let mut timeline = CpuTimeline::new(&scenes);
            // Right-click scene0, while the needle remains at74 in scene1.
            let click = ui_frame_position(10, 45.0);
            for pressed in [true, false] {
                assert_eq!(
                    timeline.frame(
                        &scenes,
                        vec![
                            egui::Event::PointerMoved(click),
                            egui::Event::PointerButton {
                                pos: click,
                                button: egui::PointerButton::Secondary,
                                pressed,
                                modifiers: egui::Modifiers::default(),
                            },
                        ],
                        false
                    ),
                    None
                );
            }
            timeline.frame(&scenes, Vec::new(), false);
            let center = timeline
                .texts
                .iter()
                .find(|(text, _)| text == label)
                .unwrap_or_else(|| {
                    panic!(
                        "Missing context menu item{label}; painted{:?}",
                        timeline.texts
                    )
                })
                .1
                .center();
            assert_eq!(timeline.button(&scenes, center, true, false), None);
            assert_eq!(
                timeline.button(&scenes, center, false, false),
                Some(expected)
            );
            assert_eq!(timeline.view.last_playhead, Some(74));
        }
    }

    #[test]
    fn mask_marquee_selection_respects_shift_and_half_frame_tolerance() {
        use std::collections::BTreeSet;
        let keys = [10, 20, 30, 40];
        let mut selected = BTreeSet::from([40]);
        mask_marquee_selection(&keys, 12.4, 31.0, false, &mut selected);
        assert_eq!(selected, BTreeSet::from([20, 30]));
        mask_marquee_selection(&keys, 12.4, 31.0, true, &mut selected);
        assert_eq!(selected, BTreeSet::from([20, 30]));
        mask_marquee_selection(&keys, 8.0, 12.0, true, &mut selected);
        assert_eq!(selected, BTreeSet::from([10, 20, 30]));
        mask_marquee_selection(&keys, 50.0, 60.0, false, &mut selected);
        assert!(selected.is_empty());
    }

    #[test]
    fn mask_lane_click_selects_marquee_extents_and_drag_reports_move() {
        use std::collections::BTreeSet;
        let scenes = scenes();
        let keys = vec![10_i64, 20, 40];
        let context = egui::Context::default();
        let mut time = 1.0_f64;
        let mut view = TimelineView::default();
        let mut selected: BTreeSet<i64> = BTreeSet::new();
        let mut drag: Option<MaskTimelineDrag> = None;
        let run = |context: &egui::Context,
                   time: &mut f64,
                   selected: &mut BTreeSet<i64>,
                   drag: &mut Option<MaskTimelineDrag>,
                   view: &mut TimelineView,
                   events: Vec<egui::Event>|
         -> Option<TrackAction> {
            *time += 0.02;
            let mut action = None;
            let raw = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 360.0))),
                time: Some(*time),
                events,
                ..Default::default()
            };
            let _ = context.run(raw, |context| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(context, |ui| {
                        let mut request = input(&scenes);
                        request.mask_lane = Some(MaskLaneInput {
                            keyframes: &keys,
                            scene: Some(0),
                            selected,
                            drag,
                        });
                        action = show_track(ui, request, TrackColors::default(), view);
                    });
            });
            action
        };
        // Warm the layout pass, then marquee from frame 2 to frame 45.
        run(
            &context,
            &mut time,
            &mut selected,
            &mut drag,
            &mut view,
            Vec::new(),
        );
        let start = Pos2::new(28.0, 94.0);
        let end = Pos2::new(244.0, 94.0);
        run(
            &context,
            &mut time,
            &mut selected,
            &mut drag,
            &mut view,
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
        run(
            &context,
            &mut time,
            &mut selected,
            &mut drag,
            &mut view,
            vec![egui::Event::PointerMoved(end)],
        );
        let release = run(
            &context,
            &mut time,
            &mut selected,
            &mut drag,
            &mut view,
            vec![egui::Event::PointerButton {
                pos: end,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        assert!(release.is_none());
        assert_eq!(selected, BTreeSet::from([10, 20, 40]));

        // A plain click on the frame-20 diamond selects only that key and does
        // not scrub the playhead.
        let key = Pos2::new(18.0 + 20.0 / 120.0 * 604.0, 94.0);
        run(
            &context,
            &mut time,
            &mut selected,
            &mut drag,
            &mut view,
            vec![
                egui::Event::PointerMoved(key),
                egui::Event::PointerButton {
                    pos: key,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
        let click = run(
            &context,
            &mut time,
            &mut selected,
            &mut drag,
            &mut view,
            vec![egui::Event::PointerButton {
                pos: key,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        assert!(click.is_none());
        assert_eq!(selected, BTreeSet::from([20]));

        // Dragging the selected diamond by ten frames reports a key move.
        let target = Pos2::new(key.x + 10.0 / 120.0 * 604.0, 94.0);
        run(
            &context,
            &mut time,
            &mut selected,
            &mut drag,
            &mut view,
            vec![
                egui::Event::PointerMoved(key),
                egui::Event::PointerButton {
                    pos: key,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ],
        );
        run(
            &context,
            &mut time,
            &mut selected,
            &mut drag,
            &mut view,
            vec![egui::Event::PointerMoved(target)],
        );
        let moved = run(
            &context,
            &mut time,
            &mut selected,
            &mut drag,
            &mut view,
            vec![egui::Event::PointerButton {
                pos: target,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        assert_eq!(moved, Some(TrackAction::MoveMaskKeys { delta: 10 }));
    }

    #[test]
    fn mask_keyframe_band_has_a_minimum_height_and_recentres_the_toggle() {
        let scenes = scenes();
        let geometry = ui_geometry();
        let band = geometry.mask_lane_rect(&scenes[1]);
        assert!(
            (band.height() - MASK_LANE_HEIGHT).abs() < 0.01,
            "band must keep its minimum height, got {}",
            band.height()
        );
        let region = geometry.scene_rect(&scenes[1]).shrink2(Vec2::new(1.0, 0.0));
        let toggle = geometry
            .toggle_rect_laned(&scenes[1], true)
            .expect("toggle");
        let expected = (region.top() + band.top()) * 0.5;
        assert!(
            (toggle.center().y - expected).abs() < 0.01,
            "lane-aware toggle must centre between scene top and keyframe band"
        );
        assert!(
            toggle.bottom() <= band.top(),
            "toggle must not cover the band"
        );
        let plain = geometry
            .toggle_rect_laned(&scenes[1], false)
            .expect("toggle");
        assert!((plain.center().y - region.center().y).abs() < 0.01);
    }

    #[test]
    fn cache_runs_merge_contiguous_ram_cached_frames() {
        use std::collections::BTreeSet;
        let frames = BTreeSet::from([2, 3, 4, 9, 11, 12]);
        assert_eq!(cache_runs(&frames), vec![(2, 4), (9, 9), (11, 12)]);
        assert!(cache_runs(&BTreeSet::new()).is_empty());
    }

    #[test]
    fn lane_aware_scene_toggle_click_matches_the_drawn_checkbox() {
        use std::collections::BTreeSet;
        let scenes = scenes();
        let keys = vec![10_i64];
        let context = egui::Context::default();
        let mut time = 1.0_f64;
        let mut view = TimelineView::default();
        let mut selected: BTreeSet<i64> = BTreeSet::new();
        let mut drag: Option<MaskTimelineDrag> = None;
        let run = |context: &egui::Context,
                   time: &mut f64,
                   selected: &mut BTreeSet<i64>,
                   drag: &mut Option<MaskTimelineDrag>,
                   view: &mut TimelineView,
                   events: Vec<egui::Event>|
         -> Option<TrackAction> {
            *time += 0.02;
            let mut action = None;
            let raw = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(640.0, 360.0))),
                time: Some(*time),
                events,
                ..Default::default()
            };
            let _ = context.run(raw, |context| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(context, |ui| {
                        let mut request = input(&scenes);
                        request.mask_lane = Some(MaskLaneInput {
                            keyframes: &keys,
                            scene: Some(0),
                            selected,
                            drag,
                        });
                        action = show_track(ui, request, TrackColors::default(), view);
                    });
            });
            action
        };
        run(
            &context,
            &mut time,
            &mut selected,
            &mut drag,
            &mut view,
            Vec::new(),
        );
        // The checkbox visual is lane-aware (centred above the keyframe band);
        // clicking exactly there must toggle the scene, not seek.
        let toggle = ui_geometry()
            .toggle_rect_laned(&scenes[0], true)
            .expect("toggle");
        let pos = toggle.center();
        run(
            &context,
            &mut time,
            &mut selected,
            &mut drag,
            &mut view,
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
        let release = run(
            &context,
            &mut time,
            &mut selected,
            &mut drag,
            &mut view,
            vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::default(),
            }],
        );
        assert_eq!(
            release,
            Some(TrackAction::SetSceneEnabled {
                scene: 0,
                enabled: false,
            })
        );
    }
}
