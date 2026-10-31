//! Hint assembly and auxiliary-alpha carry matching `hann_stage1_stream`.

use std::collections::BTreeMap;

use crate::windows::Window;
use crate::{ContractError, Result};

#[derive(Debug, Clone, Copy)]
pub struct HintQualities {
    pub initial: f32,
    pub full_video: f32,
    pub carry: f32,
}

impl Default for HintQualities {
    fn default() -> Self {
        Self {
            initial: 0.95,
            full_video: 0.95,
            carry: 0.95,
        }
    }
}

#[derive(Debug)]
pub struct HintBatch {
    /// Contiguous [T,1,H,W], corresponding to [1,T,1,H,W] graph input.
    pub masks: Vec<f32>,
    /// [T], corresponding to [1,T] graph input; all -1 only if no hint exists.
    pub quality: Vec<f32>,
    pub present: Vec<bool>,
}

pub struct HintCarry {
    frame_elements: usize,
    masks: BTreeMap<usize, Vec<f32>>,
}

impl HintCarry {
    pub fn new(frame_elements: usize) -> Result<Self> {
        if frame_elements == 0 {
            return Err(ContractError("hint frame cannot be empty".into()));
        }
        Ok(Self {
            frame_elements,
            masks: BTreeMap::new(),
        })
    }

    /// CK precedence is full-video hint, then cached auxiliary carry, then the
    /// initial hint on absolute frame zero. Carry can replace video hints.
    pub fn build_window(
        &self,
        window: &Window,
        full_video: Option<&[f32]>,
        initial: Option<&[f32]>,
        qualities: HintQualities,
    ) -> Result<HintBatch> {
        let total = window
            .len()
            .checked_mul(self.frame_elements)
            .ok_or_else(|| ContractError("hint tensor size overflow".into()))?;
        if full_video.is_some_and(|values| values.len() != total)
            || initial.is_some_and(|values| values.len() != self.frame_elements)
        {
            return Err(ContractError("hint shape mismatch".into()));
        }
        let mut batch = HintBatch {
            masks: vec![0.0; total],
            quality: vec![0.0; window.len()],
            present: vec![false; window.len()],
        };
        if let Some(full) = full_video {
            batch.masks.copy_from_slice(full);
            batch.quality.fill(qualities.full_video);
            batch.present.fill(true);
        }
        for (relative, absolute) in (window.start..window.end).enumerate() {
            if let Some(mask) = self.masks.get(&absolute) {
                batch.masks[relative * self.frame_elements..(relative + 1) * self.frame_elements]
                    .copy_from_slice(mask);
                batch.quality[relative] = qualities.carry;
                batch.present[relative] = true;
            }
        }
        if window.start == 0 {
            if let Some(initial) = initial {
                batch.masks[..self.frame_elements].copy_from_slice(initial);
                batch.quality[0] = qualities.initial;
                batch.present[0] = true;
            }
        }
        if !batch.present.iter().any(|present| *present) {
            batch.quality.fill(-1.0);
        }
        Ok(batch)
    }

    /// Aux is already bilinear-upsampled to full padded resolution and clamped
    /// to [0,1]. The host owns the reference BF16 storage boundary if required.
    /// Call only when carry_hint is enabled, after executing this window.
    pub fn commit_auxiliary(&mut self, window: &Window, auxiliary_full: &[f32]) -> Result<()> {
        if auxiliary_full.len()
            != window
                .len()
                .checked_mul(self.frame_elements)
                .ok_or_else(|| ContractError("auxiliary hint size overflow".into()))?
        {
            return Err(ContractError("auxiliary hint shape mismatch".into()));
        }
        for (relative, values) in auxiliary_full.chunks_exact(self.frame_elements).enumerate() {
            self.masks.insert(window.start + relative, values.to_vec());
        }
        self.masks
            .retain(|&frame, _| frame >= window.finalize_before);
        Ok(())
    }

    /// Accept only frames a later window can consume. The native host can
    /// upsample that overlap directly, avoiding a full-T padded alpha buffer.
    /// Indices are absolute within the scene. Validate the complete replacement
    /// before changing carry state, so a malformed stream cannot partially
    /// replace the previous valid hints.
    pub fn commit_auxiliary_overlap(
        &mut self,
        window: &Window,
        frames: impl IntoIterator<Item = (usize, Vec<f32>)>,
    ) -> Result<()> {
        if window.finalize_before < window.start || window.finalize_before > window.end {
            return Err(ContractError("invalid auxiliary overlap boundary".into()));
        }
        let mut replacement = BTreeMap::new();
        for (index, values) in frames {
            if index < window.finalize_before
                || index >= window.end
                || values.len() != self.frame_elements
                || values.iter().any(|value| !value.is_finite())
                || replacement.insert(index, values).is_some()
            {
                return Err(ContractError(
                    "invalid or duplicate auxiliary overlap frame".into(),
                ));
            }
        }
        if replacement.len() != window.end - window.finalize_before {
            return Err(ContractError("missing auxiliary overlap frame".into()));
        }
        self.masks
            .retain(|&index, _| index >= window.finalize_before);
        self.masks.extend(replacement);
        Ok(())
    }

    pub fn pending_frames(&self) -> usize {
        self.masks.len()
    }
    pub fn reset(&mut self) {
        self.masks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::windows::WindowPlan;

    #[test]
    fn absent_hints_are_minus_one_but_partial_hint_keeps_other_frames_zero() {
        let plan = WindowPlan::new(7, 4, 2).unwrap();
        let carry = HintCarry::new(1).unwrap();
        let absent = carry
            .build_window(&plan.windows[0], None, None, HintQualities::default())
            .unwrap();
        assert_eq!(absent.quality, [-1.0; 4]);
        let first = carry
            .build_window(
                &plan.windows[0],
                None,
                Some(&[0.8]),
                HintQualities::default(),
            )
            .unwrap();
        assert_eq!(first.masks, [0.8, 0.0, 0.0, 0.0]);
        assert_eq!(first.quality, [0.95, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn auxiliary_carry_overwrites_full_video_in_overlap_and_evicts_old_frames() {
        let plan = WindowPlan::new(7, 4, 2).unwrap();
        let mut carry = HintCarry::new(1).unwrap();
        carry
            .commit_auxiliary(&plan.windows[0], &[0.1, 0.2, 0.3, 0.4])
            .unwrap();
        assert_eq!(carry.pending_frames(), 2);
        let qualities = HintQualities {
            initial: 0.9,
            full_video: 0.8,
            carry: 0.7,
        };
        let next = carry
            .build_window(&plan.windows[1], Some(&[0.99; 4]), Some(&[0.5]), qualities)
            .unwrap();
        assert_eq!(next.masks, [0.3, 0.4, 0.99, 0.99]);
        assert_eq!(next.quality, [0.7, 0.7, 0.8, 0.8]);
        assert!(next.present.iter().all(|present| *present));
        carry
            .commit_auxiliary(plan.windows.last().unwrap(), &[0.6; 4])
            .unwrap();
        assert_eq!(carry.pending_frames(), 0);
    }

    #[test]
    fn overlap_only_carry_matches_full_buffers_at_anchored_tail() {
        let plan = WindowPlan::new(7, 4, 2).unwrap();
        let mut original = HintCarry::new(3).unwrap();
        let mut bounded = HintCarry::new(3).unwrap();
        for window in &plan.windows {
            let video = vec![0.99; window.len() * 3];
            let before = original
                .build_window(
                    window,
                    Some(&video),
                    Some(&[0.5; 3]),
                    HintQualities::default(),
                )
                .unwrap();
            let after = bounded
                .build_window(
                    window,
                    Some(&video),
                    Some(&[0.5; 3]),
                    HintQualities::default(),
                )
                .unwrap();
            assert_eq!(before.masks, after.masks);
            assert_eq!(before.quality, after.quality);
            assert_eq!(before.present, after.present);
            let full: Vec<f32> = (window.start..window.end)
                .flat_map(|index| [index as f32 / 10.0; 3])
                .collect();
            original.commit_auxiliary(window, &full).unwrap();
            bounded
                .commit_auxiliary_overlap(
                    window,
                    (window.finalize_before..window.end)
                        .map(|index| (index, vec![index as f32 / 10.0; 3])),
                )
                .unwrap();
            assert_eq!(original.pending_frames(), bounded.pending_frames());
        }
        assert_eq!(bounded.pending_frames(), 0);
    }

    #[test]
    fn invalid_overlap_cannot_partially_replace_valid_carry() {
        let plan = WindowPlan::new(8, 4, 2).unwrap();
        let window = &plan.windows[0];
        let mut carry = HintCarry::new(1).unwrap();
        carry
            .commit_auxiliary(window, &[0.1, 0.2, 0.3, 0.4])
            .unwrap();
        for malformed in [
            vec![(2, vec![0.8])],
            vec![(2, vec![0.8]), (2, vec![0.9])],
            vec![(2, vec![0.8]), (4, vec![0.9])],
            vec![(2, vec![0.8]), (3, vec![f32::NAN])],
        ] {
            assert!(carry.commit_auxiliary_overlap(window, malformed).is_err());
            let next = carry
                .build_window(&plan.windows[1], None, None, HintQualities::default())
                .unwrap();
            assert_eq!(&next.masks[..2], &[0.3, 0.4]);
        }
    }
}
