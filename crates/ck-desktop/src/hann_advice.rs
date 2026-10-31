//! Guard rails for the Custom Hann window controls.
//!
//! The worker resolves the real window against the live WDDM budget; the UI can
//! only approximate that budget from the selected adapter's dedicated memory.
//! These messages are advisory and never feed job validation.

use ck_core::window_policy::{
    MemoryEstimate, WindowChoice, WindowMode, WindowRequest, resolve_window,
};
use ck_desktop_core::{CkSettings, CkWindowMode};

/// Multipliers the DirectML mixed-precision worker applies to the upstream
/// BF16 peak tables when it chooses and validates windows.
const ESTIMATE: MemoryEstimate = MemoryEstimate {
    normal_scale: 1.75,
    normal_extra_gib: 0.0,
    low_vram_scale: 2.0,
    low_vram_extra_gib: 0.0,
};

const TEMPORAL_MAX: usize = 256;
/// WDDM local-budget fraction of dedicated memory observed on the reference
/// adapter; the worker additionally keeps 512 MiB of headroom.
const BUDGET_SCALE: f64 = 0.952;
const BUDGET_HEADROOM_GIB: f64 = 0.5;
const BYTES_PER_GIB: f64 = 1_073_741_824.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HannAdviceKind {
    /// Estimated peak exceeds the approximated usable GPU budget.
    OverBudget,
    /// Window is larger than the automatic choice.
    LargerThanAuto,
    /// Window overlaps differently from the automatic choice.
    StrideDiffers,
    /// Window is smaller than the automatic choice.
    SmallerThanAuto,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HannAdvice {
    pub kind: HannAdviceKind,
    pub message: String,
}

/// Approximate the usable DirectML budget in GiB from dedicated VRAM.
///
/// This mirrors the linear relation between dedicated memory and the WDDM
/// local budget observed at the reference adapter, so the UI can warn before a
/// job starts. The worker still resolves the authoritative live budget.
pub fn estimated_budget_gib(dedicated_bytes: u64) -> Option<f64> {
    (dedicated_bytes > 0).then(|| {
        let dedicated_gib = dedicated_bytes as f64 / BYTES_PER_GIB;
        (dedicated_gib * BUDGET_SCALE - BUDGET_HEADROOM_GIB).max(0.5)
    })
}

fn resolve(
    settings: &CkSettings,
    mode: WindowMode,
    width: usize,
    height: usize,
    frames: usize,
    budget_gib: f64,
) -> Option<WindowChoice> {
    resolve_window(
        WindowRequest {
            mode,
            chunk: settings.hann_chunk,
            stride: settings.hann_stride,
            low_vram: settings.low_vram,
        },
        width,
        height,
        frames,
        Some(budget_gib),
        TEMPORAL_MAX,
        ESTIMATE,
    )
    .ok()
}

/// Warning or hint for the current Custom window.
///
/// Returns `None` when the window is exactly the automatic choice, when the
/// window mode is not Custom, or when no usable GPU budget is available (for
/// example CPU jobs).
pub fn custom_hann_advice(
    settings: &CkSettings,
    width: usize,
    height: usize,
    frames: usize,
    budget_gib: Option<f64>,
) -> Option<HannAdvice> {
    if settings.window_mode != CkWindowMode::Custom {
        return None;
    }
    let budget_gib = budget_gib?;
    if width == 0 || height == 0 || frames == 0 || !budget_gib.is_finite() || budget_gib <= 0.0 {
        return None;
    }
    let balanced = resolve(
        settings,
        WindowMode::Balanced,
        width,
        height,
        frames,
        budget_gib,
    )?;
    let custom = resolve(
        settings,
        WindowMode::Custom,
        width,
        height,
        frames,
        budget_gib,
    )?;
    if !custom.estimated_peak_gib.is_finite() || !balanced.estimated_peak_gib.is_finite() {
        return None;
    }
    if custom.estimated_peak_gib > budget_gib {
        return Some(HannAdvice {
            kind: HannAdviceKind::OverBudget,
            message: format!(
                "Custom Hann {}/{} needs about {:.1} GiB; the estimated usable GPU budget is {:.1} GiB, so it may be much slower or fail to fit. Auto would use {}/{}.",
                custom.chunk,
                custom.stride,
                custom.estimated_peak_gib,
                budget_gib,
                balanced.chunk,
                balanced.stride
            ),
        });
    }
    if custom.chunk > balanced.chunk {
        return Some(HannAdvice {
            kind: HannAdviceKind::LargerThanAuto,
            message: format!(
                "Custom Hann {}/{} is larger than Auto's {}/{}: it uses more GPU memory and measured whole-job time can be 2-4x slower. Auto is recommended unless you need the longer window.",
                custom.chunk, custom.stride, balanced.chunk, balanced.stride
            ),
        });
    }
    if custom.chunk < balanced.chunk {
        return Some(HannAdvice {
            kind: HannAdviceKind::SmallerThanAuto,
            message: format!(
                "Custom Hann {}/{} is smaller than Auto's {}/{}: less GPU memory but less temporal context for stage 1.",
                custom.chunk, custom.stride, balanced.chunk, balanced.stride
            ),
        });
    }
    if custom.stride != balanced.stride {
        return Some(HannAdvice {
            kind: HannAdviceKind::StrideDiffers,
            message: format!(
                "Custom stride {} differs from Auto's {}: smaller strides overlap windows more (more work, smoother blends); larger strides are faster but blend less.",
                custom.stride, balanced.stride
            ),
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const VRAM_16_GIB: u64 = 16_771_973_120;
    const VRAM_24_GIB: u64 = 25_769_803_776;

    fn custom(chunk: usize, stride: usize) -> CkSettings {
        CkSettings {
            window_mode: CkWindowMode::Custom,
            hann_chunk: chunk,
            hann_stride: stride,
            low_vram: false,
        }
    }

    fn advice(
        settings: &CkSettings,
        width: usize,
        height: usize,
        frames: usize,
    ) -> Option<HannAdvice> {
        custom_hann_advice(
            settings,
            width,
            height,
            frames,
            estimated_budget_gib(VRAM_16_GIB),
        )
    }

    #[test]
    fn budget_estimate_matches_reference_fraction() {
        let budget = estimated_budget_gib(VRAM_16_GIB).expect("budget");
        assert!((14.0..15.0).contains(&budget), "unexpected budget {budget}");
    }

    #[test]
    fn automatic_and_wide_modes_never_warn() {
        let mut settings = CkSettings::default();
        assert_eq!(advice(&settings, 1920, 1080, 16), None);
        settings.window_mode = CkWindowMode::Wide;
        assert_eq!(advice(&settings, 1920, 1080, 16), None);
    }

    #[test]
    fn over_budget_custom_window_suggests_auto() {
        let advice = advice(&custom(16, 8), 1920, 1080, 16).expect("warning");
        assert_eq!(advice.kind, HannAdviceKind::OverBudget);
        assert!(
            advice.message.contains("Auto would use 8/4"),
            "{}",
            advice.message
        );
    }

    #[test]
    fn equal_to_auto_has_no_advice() {
        assert_eq!(advice(&custom(8, 4), 1920, 1080, 16), None);
    }

    #[test]
    fn different_stride_is_a_hint() {
        let advice = advice(&custom(8, 2), 1920, 1080, 16).expect("hint");
        assert_eq!(advice.kind, HannAdviceKind::StrideDiffers);
        assert!(advice.message.contains("Auto"), "{}", advice.message);
    }

    #[test]
    fn smaller_custom_window_is_a_hint() {
        let advice = advice(&custom(4, 2), 1920, 1080, 16).expect("hint");
        assert_eq!(advice.kind, HannAdviceKind::SmallerThanAuto);
        assert!(
            advice.message.contains("temporal context"),
            "{}",
            advice.message
        );
    }

    #[test]
    fn larger_but_fitting_custom_window_warns_on_a_larger_card() {
        // 720p on a 24 GiB card resolves Auto to the speed cap (32); a custom
        // 40/20 window still fits the estimate but is larger than Auto.
        let budget = estimated_budget_gib(VRAM_24_GIB);
        let advice = custom_hann_advice(&custom(40, 20), 1280, 720, 16, budget).expect("warning");
        assert_eq!(advice.kind, HannAdviceKind::LargerThanAuto);
    }

    #[test]
    fn four_k_custom_window_exceeds_budget() {
        let advice = advice(&custom(8, 4), 3840, 2160, 16).expect("warning");
        assert_eq!(advice.kind, HannAdviceKind::OverBudget);
    }

    #[test]
    fn no_budget_disables_advice() {
        assert_eq!(
            custom_hann_advice(&custom(16, 8), 1920, 1080, 16, None),
            None
        );
        assert_eq!(estimated_budget_gib(0), None);
    }
}
