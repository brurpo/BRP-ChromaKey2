//! UI-only state for the garbage/despill mask editor.
//!
//! Layer data lives in `MaskStacks`; this keeps v1's window state (per-scene
//! selected layer memory, timeline mask visibility and keyframe selection).
//! It is never persisted.

use ck_desktop_core::{MaskKind, MaskLayer};
use std::collections::{BTreeMap, BTreeSet};

/// Move every key (vector shape, painted raster, tangent) of `frames` by
/// `delta` source frames and return the new key frames.
///
/// Colliding target keys are replaced, matching an in-place key edit; frames
/// shifted below zero are dropped. Painted keys move as whole rasters.
pub fn move_mask_keyframes(
    layer: &mut MaskLayer,
    frames: &BTreeSet<i64>,
    delta: i64,
) -> BTreeSet<i64> {
    let mut moved = BTreeSet::new();
    if delta == 0 {
        return frames.clone();
    }
    let ordered: Vec<i64> = if delta > 0 {
        frames.iter().rev().copied().collect()
    } else {
        frames.iter().copied().collect()
    };
    for frame in ordered {
        let target = frame + delta;
        if target < 0 {
            continue;
        }
        let shape = layer.shapes.remove(&frame);
        let tangent = layer.tangents.remove(&frame);
        let painted = layer.doc.remove(frame);
        if shape.is_none() && tangent.is_none() && painted.is_none() {
            continue;
        }
        layer.shapes.remove(&target);
        layer.tangents.remove(&target);
        layer.doc.remove(target);
        if let Some(shape) = shape {
            layer.shapes.insert(target, shape);
        }
        if let Some(tangent) = tangent {
            layer.tangents.insert(target, tangent);
        }
        if let Some(painted) = painted {
            layer.doc.set_mask(target, &painted, false);
        }
        moved.insert(target);
    }
    moved
}

#[derive(Clone, Debug)]
pub struct MaskUi {
    /// Garbage or despill bank shown in the mask panel.
    pub active_kind: MaskKind,
    /// Whether the timeline shows mask keyframes.
    pub show_timeline: bool,
    /// Whether the viewer draws/edits the selected mask shape.
    pub editing: bool,
    remembered: BTreeMap<(u8, u32), u32>,
}

impl Default for MaskUi {
    fn default() -> Self {
        Self {
            active_kind: MaskKind::Garbage,
            show_timeline: true,
            editing: false,
            remembered: BTreeMap::new(),
        }
    }
}

impl MaskUi {
    fn kind_index(kind: MaskKind) -> u8 {
        match kind {
            MaskKind::Garbage => 0,
            MaskKind::Despill => 1,
        }
    }

    /// Remember the selected layer for a `(kind, scene)` pair (v1 layer memory).
    pub fn remember(&mut self, kind: MaskKind, scene: u32, layer: Option<u32>) {
        let key = (Self::kind_index(kind), scene);
        match layer {
            Some(id) => {
                self.remembered.insert(key, id);
            }
            None => {
                self.remembered.remove(&key);
            }
        }
    }

    /// Selected layer remembered for a `(kind, scene)` pair.
    pub fn remembered(&self, kind: MaskKind, scene: u32) -> Option<u32> {
        self.remembered
            .get(&(Self::kind_index(kind), scene))
            .copied()
    }

    /// Switch the visible bank.
    pub fn set_active_kind(&mut self, kind: MaskKind) {
        self.active_kind = kind;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_memory_is_per_kind_and_scene() {
        let mut ui = MaskUi::default();
        ui.remember(MaskKind::Garbage, 0, Some(4));
        ui.remember(MaskKind::Garbage, 1, Some(7));
        ui.remember(MaskKind::Despill, 0, Some(2));
        assert_eq!(ui.remembered(MaskKind::Garbage, 0), Some(4));
        assert_eq!(ui.remembered(MaskKind::Garbage, 1), Some(7));
        assert_eq!(ui.remembered(MaskKind::Despill, 0), Some(2));
        assert_eq!(ui.remembered(MaskKind::Despill, 1), None);
        ui.remember(MaskKind::Garbage, 0, None);
        assert_eq!(ui.remembered(MaskKind::Garbage, 0), None);
    }

    #[test]
    fn switching_kind_updates_the_active_bank() {
        let mut ui = MaskUi::default();
        ui.set_active_kind(MaskKind::Despill);
        assert_eq!(ui.active_kind, MaskKind::Despill);
    }

    #[test]
    fn moving_keyframes_shifts_shapes_tangents_and_keeps_other_keys() {
        use ck_core::mask_layers::{BezierNode, BezierShape, Tangent};
        let mut layer = MaskLayer::new(1, "Garbage 01", MaskKind::Garbage, 32, 32);
        layer.set_shape(
            10,
            BezierShape::with_nodes(vec![BezierNode::new(4.0, 4.0)], false),
        );
        layer.set_shape(
            20,
            BezierShape::with_nodes(vec![BezierNode::new(8.0, 8.0)], false),
        );
        layer.set_shape(
            30,
            BezierShape::with_nodes(vec![BezierNode::new(12.0, 12.0)], false),
        );
        layer.tangents.insert(10, Tangent::new(0.25, 0.75));
        let moved = move_mask_keyframes(&mut layer, &BTreeSet::from([10, 20]), 5);
        assert_eq!(moved, BTreeSet::from([15, 25]));
        assert!(layer.shapes.contains_key(&15));
        assert!(layer.shapes.contains_key(&25));
        assert!(layer.shapes.contains_key(&30));
        assert!(!layer.shapes.contains_key(&10));
        assert!(!layer.shapes.contains_key(&20));
        assert_eq!(layer.tangents.get(&15), Some(&Tangent::new(0.25, 0.75)));
    }

    #[test]
    fn moving_keyframes_forward_does_not_chain_selected_keys() {
        use ck_core::mask_layers::{BezierNode, BezierShape};
        let mut layer = MaskLayer::new(1, "Garbage 01", MaskKind::Garbage, 32, 32);
        layer.set_shape(
            1,
            BezierShape::with_nodes(vec![BezierNode::new(1.0, 1.0)], false),
        );
        layer.set_shape(
            2,
            BezierShape::with_nodes(vec![BezierNode::new(2.0, 2.0)], false),
        );
        let moved = move_mask_keyframes(&mut layer, &BTreeSet::from([1, 2]), 1);
        assert_eq!(moved, BTreeSet::from([2, 3]));
        assert_eq!(layer.shapes.len(), 2);
        assert!(layer.shapes.contains_key(&2));
        assert!(layer.shapes.contains_key(&3));
    }
}
