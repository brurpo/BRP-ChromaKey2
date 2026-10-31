//! Single-object forward-time memory selection from official SAM2.
//! Neural memory encoding/attention remain exported trained ONNX graphs.

use std::collections::BTreeSet;

use crate::{ContractError, Result};

#[derive(Debug, Clone)]
pub struct MemoryPolicy {
    pub num_maskmem: usize,
    pub temporal_stride: usize,
    /// None is the official max_cond_frames_in_attn=-1 setting.
    pub max_conditioning_frames: Option<usize>,
    pub max_object_pointers: usize,
    pub only_past_object_pointers: bool,
    pub signed_pointer_distance: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpatialSource {
    pub frame: usize,
    pub temporal_position: usize,
    pub temporal_embedding_index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointerSource {
    pub frame: usize,
    pub distance: isize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySelection {
    pub conditioning: Vec<usize>,
    pub spatial: Vec<SpatialSource>,
    pub pointers: Vec<PointerSource>,
    pub pointer_distance_denominator: usize,
}

/// Safe state horizon only when the remaining events are consecutive-forward.
/// Backward correction/replay keeps historical states until consumed.
pub struct HistoryRetention {
    oldest: usize,
    current: usize,
    feedback_frames: BTreeSet<usize>,
}

impl HistoryRetention {
    pub fn keeps(&self, frame: usize, conditioning: bool) -> bool {
        conditioning
            || self.feedback_frames.contains(&frame)
            || (frame >= self.oldest && frame <= self.current)
    }
}

pub fn select_closest_conditioning(
    current: usize,
    conditioning_order: &[usize],
    maximum: Option<usize>,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let unique: BTreeSet<_> = conditioning_order.iter().copied().collect();
    if unique.len() != conditioning_order.len() {
        return Err(ContractError("duplicate SAM conditioning frame".into()));
    }
    let Some(maximum) = maximum else {
        return Ok((conditioning_order.to_vec(), Vec::new()));
    };
    if maximum < 2 {
        return Err(ContractError(
            "SAM conditioning limit must be at least two or unlimited".into(),
        ));
    }
    if conditioning_order.len() <= maximum {
        return Ok((conditioning_order.to_vec(), Vec::new()));
    }
    let mut selected = Vec::new();
    if let Some(before) = conditioning_order
        .iter()
        .copied()
        .filter(|&frame| frame < current)
        .max()
    {
        selected.push(before);
    }
    if let Some(after) = conditioning_order
        .iter()
        .copied()
        .filter(|&frame| frame >= current)
        .min()
    {
        selected.push(after);
    }
    let mut remaining: Vec<_> = conditioning_order
        .iter()
        .copied()
        .filter(|frame| !selected.contains(frame))
        .collect();
    // Stable sort preserves official Python insertion order when distances tie.
    remaining.sort_by_key(|&frame| frame.abs_diff(current));
    selected.extend(remaining.into_iter().take(maximum - selected.len()));
    let unselected = conditioning_order
        .iter()
        .copied()
        .filter(|frame| !selected.contains(frame))
        .collect();
    Ok((selected, unselected))
}

impl MemoryPolicy {
    /// Future entries are (frame index, needs old prediction as mask feedback).
    pub fn retention_after(
        &self,
        current: usize,
        future: &[(usize, bool)],
    ) -> Option<HistoryRetention> {
        let mut previous = current;
        for &(index, _) in future {
            if previous.checked_add(1) != Some(index) {
                return None;
            }
            previous = index;
        }
        let horizon = self.max_object_pointers.max(
            self.num_maskmem
                .saturating_sub(2)
                .saturating_mul(self.temporal_stride)
                .saturating_add(2),
        );
        Some(HistoryRetention {
            oldest: current.saturating_sub(horizon),
            current,
            feedback_frames: future
                .iter()
                .filter_map(|&(index, feedback)| feedback.then_some(index))
                .collect(),
        })
    }

    /// Memory sources for a non-initial frame. Initialization bypasses memory
    /// attention with no_mem_embed and therefore never calls this function.
    pub fn select(
        &self,
        current: usize,
        total_frames: usize,
        conditioning_order: &[usize],
        non_conditioning: &BTreeSet<usize>,
    ) -> Result<MemorySelection> {
        if self.num_maskmem == 0 || self.temporal_stride == 0 || self.max_object_pointers == 0 {
            return Err(ContractError(
                "positive SAM memory, stride and pointer limits required".into(),
            ));
        }
        if current >= total_frames
            || conditioning_order.is_empty()
            || conditioning_order
                .iter()
                .chain(non_conditioning.iter())
                .any(|&index| index >= total_frames)
        {
            return Err(ContractError(
                "invalid SAM frame index or missing conditioning memory".into(),
            ));
        }
        let (conditioning, unselected) =
            select_closest_conditioning(current, conditioning_order, self.max_conditioning_frames)?;
        let mut spatial: Vec<_> = conditioning
            .iter()
            .map(|&frame| SpatialSource {
                frame,
                temporal_position: 0,
                temporal_embedding_index: self.num_maskmem - 1,
            })
            .collect();
        let current_signed = isize::try_from(current)
            .map_err(|_| ContractError("SAM frame index overflow".into()))?;
        let stride = isize::try_from(self.temporal_stride)
            .map_err(|_| ContractError("SAM stride overflow".into()))?;
        for position in 1..self.num_maskmem {
            let relative = self.num_maskmem - position;
            let prior = if relative == 1 {
                current_signed - 1
            } else {
                (current_signed - 2).div_euclid(stride) * stride - (relative as isize - 2) * stride
            };
            if prior < 0 {
                continue;
            }
            let prior = prior as usize;
            if non_conditioning.contains(&prior) || unselected.contains(&prior) {
                spatial.push(SpatialSource {
                    frame: prior,
                    temporal_position: position,
                    temporal_embedding_index: self.num_maskmem - position - 1,
                });
            }
        }
        let maximum = total_frames.min(self.max_object_pointers);
        let mut pointers = Vec::new();
        for &frame in &conditioning {
            if self.only_past_object_pointers && frame > current {
                continue;
            }
            let delta = current_signed - frame as isize;
            pointers.push(PointerSource {
                frame,
                distance: if self.signed_pointer_distance {
                    delta
                } else {
                    delta.abs()
                },
            });
        }
        for difference in 1..maximum {
            let Some(prior) = current.checked_sub(difference) else {
                break;
            };
            if non_conditioning.contains(&prior) || unselected.contains(&prior) {
                pointers.push(PointerSource {
                    frame: prior,
                    distance: difference as isize,
                });
            }
        }
        if !pointers.is_empty() && maximum <= 1 {
            return Err(ContractError(
                "SAM temporal pointer encoding needs at least two clip frames".into(),
            ));
        }
        Ok(MemorySelection {
            conditioning,
            spatial,
            pointers,
            pointer_distance_denominator: maximum.saturating_sub(1),
        })
    }
}

/// Official get_1d_sine_pe. Its paired frequency exponents and lack of 2*pi
/// are intentional; this is not SAM's 2D image positional encoding.
pub fn sine_position_encoding(positions: &[f32], dimensions: usize) -> Result<Vec<f32>> {
    if dimensions == 0
        || !dimensions.is_multiple_of(2)
        || positions.iter().any(|value| !value.is_finite())
    {
        return Err(ContractError(
            "finite SAM positions and positive even PE dimensions required".into(),
        ));
    }
    let half = dimensions / 2;
    let denominators: Vec<_> = (0..half)
        .map(|index| 10_000.0_f32.powf((2 * (index / 2)) as f32 / half as f32))
        .collect();
    let mut output = Vec::with_capacity(positions.len() * dimensions);
    for &position in positions {
        output.extend(
            denominators
                .iter()
                .map(|&denominator| (position / denominator).sin()),
        );
        output.extend(
            denominators
                .iter()
                .map(|&denominator| (position / denominator).cos()),
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> MemoryPolicy {
        MemoryPolicy {
            num_maskmem: 7,
            temporal_stride: 1,
            max_conditioning_frames: None,
            max_object_pointers: 16,
            only_past_object_pointers: true,
            signed_pointer_distance: true,
        }
    }

    #[test]
    fn limited_conditioning_keeps_nearest_before_after_then_closest_remaining() {
        let (selected, unselected) =
            select_closest_conditioning(10, &[0, 3, 9, 11, 19], Some(3)).unwrap();
        assert_eq!(selected, [9, 11, 3]);
        assert_eq!(unselected, [0, 19]);
        assert!(select_closest_conditioning(10, &[0, 3], Some(1)).is_err());
    }

    #[test]
    fn strided_memory_keeps_immediate_previous_frame_independently_of_stride() {
        let mut policy = policy();
        policy.temporal_stride = 3;
        let non_conditioning = [3, 6, 9, 11].into_iter().collect();
        let selected = policy.select(12, 20, &[0], &non_conditioning).unwrap();
        assert_eq!(
            selected
                .spatial
                .iter()
                .map(|item| (item.frame, item.temporal_embedding_index))
                .collect::<Vec<_>>(),
            [(0, 6), (3, 3), (6, 2), (9, 1), (11, 0)]
        );
    }

    #[test]
    fn future_conditioning_can_supply_spatial_memory_but_not_past_only_pointer() {
        let mut policy = policy();
        policy.max_conditioning_frames = Some(2);
        let non_conditioning = [1, 3, 4, 6].into_iter().collect();
        let selection = policy
            .select(5, 12, &[0, 2, 10], &non_conditioning)
            .unwrap();
        assert_eq!(selection.conditioning, [2, 10]);
        assert!(selection.spatial.iter().any(|item| item.frame == 10));
        assert!(selection.spatial.iter().any(|item| item.frame == 0));
        assert!(!selection
            .pointers
            .iter()
            .any(|item| item.frame == 10 || item.frame == 6));
        assert_eq!(
            selection
                .pointers
                .iter()
                .map(|item| item.frame)
                .collect::<Vec<_>>(),
            [2, 4, 3, 1, 0]
        );
    }

    #[test]
    fn correction_does_not_consume_future_nonconditioning_predictions() {
        let bank = [1, 2, 3, 4].into_iter().collect();
        let selection = policy().select(2, 5, &[0], &bank).unwrap();
        assert!(selection.spatial.iter().all(|item| item.frame < 2));
        assert_eq!(
            selection
                .pointers
                .iter()
                .map(|item| item.frame)
                .collect::<Vec<_>>(),
            [0, 1]
        );
    }

    #[test]
    fn pointer_sine_encoding_uses_source_frequency_pairing() {
        let zero = sine_position_encoding(&[0.0], 8).unwrap();
        assert_eq!(zero, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        let actual = sine_position_encoding(&[1.0], 8).unwrap();
        let expected = [
            1.0_f32.sin(),
            1.0_f32.sin(),
            0.01_f32.sin(),
            0.01_f32.sin(),
            1.0_f32.cos(),
            1.0_f32.cos(),
            0.01_f32.cos(),
            0.01_f32.cos(),
        ];
        for (&value, expected) in actual.iter().zip(expected) {
            assert!((value - expected).abs() < 1e-7);
        }
    }

    #[test]
    fn state_eviction_keeps_conditioning_stride_horizon_and_future_feedback() {
        let mut policy = policy();
        policy.temporal_stride = 3;
        let retention = policy
            .retention_after(50, &[(51, false), (52, true), (53, false)])
            .unwrap();
        assert!(retention.keeps(0, true));
        assert!(!retention.keeps(32, false));
        assert!(retention.keeps(33, false));
        assert!(retention.keeps(50, false));
        assert!(!retention.keeps(51, false));
        assert!(retention.keeps(52, false));
    }

    #[test]
    fn state_eviction_waits_until_backward_correction_history_is_consumed() {
        assert!(policy()
            .retention_after(40, &[(41, false), (10, true), (11, false)])
            .is_none());
        let after_correction = policy()
            .retention_after(10, &[(11, false), (12, false)])
            .unwrap();
        assert!(after_correction.keeps(9, false));
        assert!(!after_correction.keeps(40, false));
        // A later correction after a gap could need old memories from skipped
        // frames. Keep history until those dependencies have been consumed.
        assert!(policy()
            .retention_after(10, &[(11, false), (40, true), (41, false)])
            .is_none());
    }
}
