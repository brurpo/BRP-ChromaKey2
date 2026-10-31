//! CK `window_starts`, taper assignment order and feature-baton accumulation.

use std::collections::BTreeMap;
use std::f32::consts::PI;

use crate::{ContractError, Result};

#[derive(Debug, Clone)]
pub struct Window {
    pub index: usize,
    pub start: usize,
    pub end: usize,
    pub weights: Vec<f32>,
    /// Frames below this absolute index cannot be touched by a later window.
    pub finalize_before: usize,
}

impl Window {
    pub fn len(&self) -> usize {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

#[derive(Debug, Clone)]
pub struct WindowPlan {
    pub total_frames: usize,
    pub chunk: usize,
    pub stride: usize,
    pub windows: Vec<Window>,
}

impl WindowPlan {
    pub fn new(total_frames: usize, chunk: usize, stride: usize) -> Result<Self> {
        if total_frames == 0 || chunk == 0 || stride == 0 || stride > chunk {
            return Err(ContractError(
                "require total_frames > 0 and 0 < stride <= chunk".into(),
            ));
        }
        let starts = if total_frames <= chunk {
            vec![0]
        } else {
            let last = total_frames - chunk;
            let mut values: Vec<_> = (0..=last).step_by(stride).collect();
            if values.last().copied() != Some(last) {
                values.push(last);
            }
            values
        };
        let windows = starts
            .iter()
            .enumerate()
            .map(|(index, &start)| {
                let end = (start + chunk).min(total_frames);
                Window {
                    index,
                    start,
                    end,
                    weights: taper_weights(
                        end - start,
                        chunk - stride,
                        index == 0,
                        index + 1 == starts.len(),
                    ),
                    finalize_before: starts.get(index + 1).copied().unwrap_or(total_frames),
                }
            })
            .collect();
        Ok(Self {
            total_frames,
            chunk,
            stride,
            windows,
        })
    }
}

/// Mirrors CK assignment, including tail overwriting head if ramps intersect.
pub fn taper_weights(frames: usize, overlap: usize, is_first: bool, is_last: bool) -> Vec<f32> {
    let mut weights = vec![1.0; frames];
    if overlap == 0 || frames <= 1 {
        return weights;
    }
    let count = overlap.min(frames);
    let ramp: Vec<_> = (0..count)
        .map(|i| 0.5 * (1.0 - (PI * (i + 1) as f32 / (overlap + 1) as f32).cos()))
        .collect();
    if !is_first {
        weights[..count].copy_from_slice(&ramp);
    }
    if !is_last {
        for i in 0..count {
            weights[frames - count + i] = ramp[count - 1 - i];
        }
    }
    weights
}

#[derive(Debug)]
struct PendingFrame {
    weighted: Vec<f32>,
    weight: f32,
}

#[derive(Debug)]
pub struct BlendedFrame {
    pub index: usize,
    /// Float32 normalized features. Match the reference's final FP16 storage
    /// boundary in the host before feeding stage 2 for runtime parity tests.
    pub values: Vec<f32>,
}

pub struct BatonAccumulator {
    frame_elements: usize,
    next_window: usize,
    pending: BTreeMap<usize, PendingFrame>,
}

impl BatonAccumulator {
    pub fn new(frame_elements: usize) -> Result<Self> {
        if frame_elements == 0 {
            return Err(ContractError("baton frame cannot be empty".into()));
        }
        Ok(Self {
            frame_elements,
            next_window: 0,
            pending: BTreeMap::new(),
        })
    }

    /// Input is contiguous [T,C,H/16,W/16]. Only finalizable frames are returned.
    pub fn push_window(&mut self, window: &Window, baton: &[f32]) -> Result<Vec<BlendedFrame>> {
        if window.index != self.next_window {
            return Err(ContractError(format!(
                "expected window {}, got {}",
                self.next_window, window.index
            )));
        }
        let expected = window
            .len()
            .checked_mul(self.frame_elements)
            .ok_or_else(|| ContractError("baton size overflow".into()))?;
        if baton.len() != expected || window.weights.len() != window.len() {
            return Err(ContractError("baton shape or taper length mismatch".into()));
        }
        if baton.iter().any(|value| !value.is_finite()) {
            return Err(ContractError("nonfinite stage-1 baton".into()));
        }
        for (relative, values) in baton.chunks_exact(self.frame_elements).enumerate() {
            let weight = window.weights[relative];
            let pending = self
                .pending
                .entry(window.start + relative)
                .or_insert_with(|| PendingFrame {
                    weighted: vec![0.0; self.frame_elements],
                    weight: 0.0,
                });
            for (sum, value) in pending.weighted.iter_mut().zip(values) {
                *sum += value * weight;
            }
            pending.weight += weight;
        }
        self.next_window += 1;
        let ready_indices: Vec<_> = self
            .pending
            .range(..window.finalize_before)
            .map(|(&index, _)| index)
            .collect();
        Ok(ready_indices
            .into_iter()
            .map(|index| {
                let mut frame = self.pending.remove(&index).expect("key collected from map");
                for value in &mut frame.weighted {
                    *value /= frame.weight.max(1e-6);
                }
                BlendedFrame {
                    index,
                    values: frame.weighted,
                }
            })
            .collect())
    }

    pub fn pending_frames(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_window_is_anchored_at_clip_end_not_stride_padded() {
        let plan = WindowPlan::new(55, 24, 12).unwrap();
        assert_eq!(
            plan.windows.iter().map(|w| w.start).collect::<Vec<_>>(),
            [0, 12, 24, 31]
        );
        assert_eq!(
            plan.windows
                .iter()
                .map(|w| w.finalize_before)
                .collect::<Vec<_>>(),
            [12, 24, 31, 55]
        );
        assert!(plan.windows.iter().all(|w| w.len() == 24));
        let short = WindowPlan::new(3, 24, 12).unwrap();
        assert_eq!(short.windows[0].weights, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn taper_does_not_zero_endpoints_and_has_ck_assignment_order() {
        let weights = taper_weights(4, 2, false, false);
        for (actual, expected) in weights.into_iter().zip([0.25, 0.75, 0.75, 0.25]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        let short = taper_weights(3, 4, false, false);
        // Tail assignment overwrites the first assignment in this small case.
        let expected: Vec<_> = (1..=3)
            .rev()
            .map(|i| 0.5 * (1.0 - (PI * i as f32 / 5.0).cos()))
            .collect();
        assert_eq!(short, expected);
    }

    #[test]
    fn normalized_batons_cover_irregular_final_overlap_once() {
        let plan = WindowPlan::new(55, 24, 12).unwrap();
        let mut accumulator = BatonAccumulator::new(2).unwrap();
        let mut result = Vec::new();
        for window in &plan.windows {
            let values: Vec<_> = (window.start..window.end)
                .flat_map(|frame| [frame as f32, -(frame as f32)])
                .collect();
            result.extend(accumulator.push_window(window, &values).unwrap());
            assert!(accumulator.pending_frames() <= plan.chunk);
        }
        assert_eq!(accumulator.pending_frames(), 0);
        assert_eq!(result.len(), 55);
        for (index, frame) in result.iter().enumerate() {
            assert_eq!(frame.index, index);
            assert!((frame.values[0] - index as f32).abs() < 1e-5);
            assert!((frame.values[1] + index as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn accumulator_requires_execution_order_and_valid_shapes() {
        let plan = WindowPlan::new(7, 4, 2).unwrap();
        let mut accumulator = BatonAccumulator::new(1).unwrap();
        assert!(accumulator
            .push_window(&plan.windows[1], &[0.0; 4])
            .is_err());
        assert!(accumulator
            .push_window(&plan.windows[0], &[0.0; 3])
            .is_err());
        assert!(accumulator
            .push_window(&plan.windows[0], &[f32::NAN; 4])
            .is_err());
        assert!(WindowPlan::new(7, 2, 3).is_err());
    }
}
