//! Resolution-aware window selection from the original BRP/CK runtime.
//!
//! The tables describe upstream PyTorch measurements, not native-provider
//! guarantees. Hosts supply a live usable budget and a conservative calibrated
//! adjustment, and retain allocation-failure recovery as a separate concern.

use crate::{ContractError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowMode {
    #[default]
    Balanced,
    Wide,
    Custom,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WindowRequest {
    pub mode: WindowMode,
    /// Zero means automatic, matching the original Custom controls.
    pub chunk: usize,
    /// Zero means half the resolved chunk, rounded down with a minimum of one.
    pub stride: usize,
    pub low_vram: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryEstimate {
    pub normal_scale: f64,
    pub normal_extra_gib: f64,
    pub low_vram_scale: f64,
    pub low_vram_extra_gib: f64,
}

impl Default for MemoryEstimate {
    fn default() -> Self {
        Self {
            normal_scale: 1.0,
            normal_extra_gib: 0.0,
            low_vram_scale: 1.0,
            low_vram_extra_gib: 0.0,
        }
    }
}

impl MemoryEstimate {
    pub fn peak_gib(self, megapixels: f64, frames: usize, low_vram: bool) -> f64 {
        let (scale, extra) = if low_vram {
            (self.low_vram_scale, self.low_vram_extra_gib)
        } else {
            (self.normal_scale, self.normal_extra_gib)
        };
        estimate_peak_gib(megapixels, frames, low_vram) * scale + extra
    }

    fn validate(self) -> Result<()> {
        if [self.normal_scale, self.low_vram_scale]
            .into_iter()
            .any(|v| !v.is_finite() || v <= 0.0)
            || [self.normal_extra_gib, self.low_vram_extra_gib]
                .into_iter()
                .any(|v| !v.is_finite() || v < 0.0)
        {
            return Err(ContractError("invalid backend memory calibration".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WindowChoice {
    pub chunk: usize,
    pub stride: usize,
    pub low_vram: bool,
    pub estimated_peak_gib: f64,
    pub budget_gib: Option<f64>,
    pub source: &'static str,
}

const NORMAL_CHUNKS: [usize; 4] = [8, 24, 48, 80];
const NORMAL_ROWS: [(f64, [f64; 4]); 4] = [
    (0.98, [6.19, 7.52, 9.77, 12.89]),
    (2.21, [7.50, 11.33, 18.66, 24.89]),
    (3.93, [9.06, 14.78, 23.92, 33.19]),
    (8.36, [13.94, 25.96, 45.15, 70.45]),
];
const LOW_CHUNKS: [usize; 7] = [1, 2, 4, 8, 24, 48, 80];
const LOW_ROWS: [(f64, [f64; 7]); 4] = [
    (0.98, [2.71, 2.78, 2.94, 3.25, 4.54, 6.57, 9.50]),
    (2.21, [2.80, 2.97, 3.35, 4.12, 7.69, 13.20, 20.56]),
    (3.93, [2.92, 3.23, 3.85, 5.09, 10.09, 17.57, 27.75]),
    (8.36, [3.25, 3.92, 5.28, 8.38, 20.78, 39.66, 64.65]),
];
const BALANCED_CANDIDATES: [usize; 10] = [4, 8, 12, 16, 24, 32, 40, 48, 64, 80];

fn interpolate(x: f64, points: &[(f64, f64)]) -> f64 {
    for (index, pair) in points.windows(2).enumerate() {
        let [(x0, y0), (x1, y1)] = [pair[0], pair[1]];
        if x <= x1 || index + 2 == points.len() {
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    points[0].1
}

fn table_peak<const N: usize>(
    megapixels: f64,
    frames: usize,
    chunks: &[usize; N],
    rows: &[(f64, [f64; N]); 4],
) -> f64 {
    let points: Vec<_> = (0..N)
        .map(|index| {
            let sizes: Vec<_> = rows.iter().map(|(mp, peak)| (*mp, peak[index])).collect();
            (
                chunks[index] as f64,
                interpolate(megapixels, &sizes).max(rows[0].1[index]),
            )
        })
        .collect();
    interpolate(frames.max(chunks[0]) as f64, &points)
}

/// Original CK interpolation, with no claim about a particular native backend.
pub fn estimate_peak_gib(megapixels: f64, frames: usize, low_vram: bool) -> f64 {
    if low_vram {
        table_peak(megapixels, frames, &LOW_CHUNKS, &LOW_ROWS)
    } else {
        table_peak(megapixels, frames, &NORMAL_CHUNKS, &NORMAL_ROWS)
    }
}

/// Original BRP balanced throughput knee: about fifty frame-megapixels.
pub fn balanced_speed_cap(width: usize, height: usize) -> usize {
    let megapixels = ((width as f64 * height as f64) / 1e6).max(0.1);
    (50.0 / megapixels).round_ties_even().clamp(4.0, 32.0) as usize
}

/// Exact v1 reference formula. Native hosts should use their actual adapter
/// budget rather than treating the three-GiB PyTorch residency as universal.
pub fn original_window_budget_gib(free_gib: f64, total_gib: f64) -> Option<f64> {
    (free_gib.is_finite() && total_gib.is_finite() && free_gib >= 0.0 && total_gib > 0.0)
        .then(|| ((free_gib + 3.0).min(total_gib * 0.9) - 2.0).max(4.0))
}

/// Select a real window for the current graph and memory model. Custom values
/// affect Custom mode only. A short clip keeps the chosen nominal chunk, while
/// its actual graph T is `min(chunk, total_frames)`, just as in the original.
#[allow(clippy::too_many_arguments)]
pub fn resolve_window(
    request: WindowRequest,
    width: usize,
    height: usize,
    total_frames: usize,
    budget_gib: Option<f64>,
    temporal_max: usize,
    estimate: MemoryEstimate,
) -> Result<WindowChoice> {
    if width == 0 || height == 0 || total_frames == 0 || temporal_max == 0 || temporal_max > 256 {
        return Err(ContractError(
            "invalid resolution, clip length or temporal model limit".into(),
        ));
    }
    if budget_gib.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(ContractError("invalid usable memory budget".into()));
    }
    estimate.validate()?;
    let megapixels = ((width as f64 * height as f64) / 1e6).max(0.1);
    let choice = |chunk: usize, stride: usize, low_vram: bool, source| {
        let chunk = chunk.clamp(1, temporal_max);
        WindowChoice {
            chunk,
            stride: stride.clamp(1, chunk),
            low_vram,
            estimated_peak_gib: estimate.peak_gib(megapixels, chunk.min(total_frames), low_vram),
            budget_gib,
            source,
        }
    };
    if request.mode == WindowMode::Custom && request.chunk > 0 {
        if request.chunk > temporal_max || request.stride > request.chunk {
            return Err(ContractError(
                "custom window exceeds the model limit or stride exceeds window".into(),
            ));
        }
        return Ok(choice(
            request.chunk,
            if request.stride == 0 {
                (request.chunk / 2).max(1)
            } else {
                request.stride
            },
            request.low_vram,
            "custom",
        ));
    }
    if request.mode == WindowMode::Balanced {
        let speed_cap = balanced_speed_cap(width, height).min(temporal_max);
        let mut chunk = speed_cap;
        if let Some(budget) = budget_gib {
            let mut allowed = 1;
            for candidate in BALANCED_CANDIDATES {
                let candidate = candidate.min(temporal_max);
                if estimate.peak_gib(megapixels, candidate.min(total_frames), request.low_vram)
                    <= budget
                {
                    allowed = allowed.max(candidate);
                }
            }
            chunk = chunk.min(allowed);
            // The reference normal table bottoms out at eight frames. When it
            // cannot fit even a small normal window, switch to truly staged
            // inference rather than claiming a fictitious four-GiB minimum.
            if !request.low_vram
                && estimate.peak_gib(megapixels, chunk.min(total_frames), false) > budget
            {
                for candidate in (1..=speed_cap).rev() {
                    if estimate.peak_gib(megapixels, candidate.min(total_frames), true) <= budget {
                        return Ok(choice(
                            candidate,
                            (candidate / 2).max(1),
                            true,
                            "balanced_staged_budget",
                        ));
                    }
                }
                return Ok(choice(1, 1, true, "balanced_minimum_memory"));
            }
        }
        return Ok(choice(
            chunk,
            (chunk / 2).max(1),
            request.low_vram,
            "balanced",
        ));
    }
    // Wide mirrors CK's largest-fitting profile order, including preference
    // for a larger staged window over a smaller normal window.
    let wide_choice = |chunk: usize, low_vram: bool, source| {
        let chunk = chunk.min(temporal_max);
        let stride = if request.mode == WindowMode::Custom && request.stride > 0 {
            if request.stride > chunk {
                return Err(ContractError(format!(
                    "custom stride {} exceeds the automatically selected window {chunk}",
                    request.stride
                )));
            }
            request.stride
        } else {
            (chunk / 2).max(1)
        };
        Ok(choice(chunk, stride, low_vram, source))
    };
    for candidate in LOW_CHUNKS.into_iter().rev() {
        let candidate = candidate.min(temporal_max);
        for low_vram in [false, true] {
            if request.low_vram && !low_vram {
                continue;
            }
            if !low_vram && !NORMAL_CHUNKS.contains(&candidate) && candidate != temporal_max {
                continue;
            }
            let peak = estimate.peak_gib(megapixels, candidate.min(total_frames), low_vram);
            if budget_gib.is_none_or(|budget| peak * 1.1 + 0.3 <= budget) {
                return wide_choice(candidate, low_vram, "wide");
            }
        }
    }
    wide_choice(1, true, "wide_minimum_memory")
}

/// Allocation-OOM-only recovery. The host must restart the scene in a fresh
/// attempt directory; changing Hann context midway through output is invalid.
pub fn retry_choices(
    initial: WindowChoice,
    temporal_max: usize,
    estimate: MemoryEstimate,
    width: usize,
    height: usize,
    total_frames: usize,
) -> Vec<WindowChoice> {
    let mut choices = vec![initial];
    let megapixels = ((width as f64 * height as f64) / 1e6).max(0.1);
    let mut push = |chunk: usize, low_vram: bool| {
        let chunk = chunk.clamp(1, temporal_max.max(1));
        let stride = (chunk / 2).max(1);
        if choices.iter().any(|old| {
            old.low_vram == low_vram
                && ((old.chunk >= total_frames && chunk >= total_frames)
                    || (old.chunk == chunk && old.stride == stride))
        }) {
            return;
        }
        choices.push(WindowChoice {
            chunk,
            stride,
            low_vram,
            estimated_peak_gib: estimate.peak_gib(megapixels, chunk.min(total_frames), low_vram),
            budget_gib: initial.budget_gib,
            source: "allocation_oom_retry",
        });
    };
    if !initial.low_vram && initial.chunk > 1 {
        push((initial.chunk / 2).clamp(1, 32), false);
    }
    push(
        if initial.source == "wide" {
            initial.chunk.min(64)
        } else {
            initial.chunk
        },
        true,
    );
    let mut chunk = initial.chunk.clamp(1, 24);
    loop {
        push(chunk, true);
        if chunk == 1 {
            break;
        }
        chunk = (chunk / 2).max(1);
    }
    choices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_balanced_resolution_knee_and_memory_reduction() {
        assert_eq!(balanced_speed_cap(1280, 720), 32);
        assert_eq!(balanced_speed_cap(1920, 1080), 24);
        assert_eq!(balanced_speed_cap(3840, 2160), 6);
        let unlimited = resolve_window(
            WindowRequest::default(),
            1920,
            1080,
            69,
            None,
            256,
            MemoryEstimate::default(),
        )
        .unwrap();
        assert_eq!((unlimited.chunk, unlimited.stride), (24, 12));
        let limited = resolve_window(
            WindowRequest::default(),
            1920,
            1080,
            69,
            Some(9.0),
            256,
            MemoryEstimate::default(),
        )
        .unwrap();
        assert_eq!((limited.chunk, limited.stride), (12, 6));
        assert!(!limited.low_vram);
    }

    #[test]
    fn upstream_peak_table_values_and_wide_order_are_preserved() {
        assert_eq!(estimate_peak_gib(2.21, 24, false), 11.33);
        assert_eq!(estimate_peak_gib(2.21, 24, true), 7.69);
        assert_eq!(estimate_peak_gib(0.1, 1, false), 6.19);
        let wide = resolve_window(
            WindowRequest {
                mode: WindowMode::Wide,
                ..Default::default()
            },
            1920,
            1080,
            100,
            Some(10.0),
            256,
            MemoryEstimate::default(),
        )
        .unwrap();
        assert_eq!((wide.chunk, wide.stride, wide.low_vram), (24, 12, true));
        assert_eq!(original_window_budget_gib(12.0, 16.0), Some(12.4));
    }

    #[test]
    fn inactive_custom_values_never_override_balanced_or_wide() {
        for mode in [WindowMode::Balanced, WindowMode::Wide] {
            let clean = resolve_window(
                WindowRequest {
                    mode,
                    ..Default::default()
                },
                1920,
                1080,
                100,
                Some(16.0),
                256,
                MemoryEstimate::default(),
            )
            .unwrap();
            let stale = resolve_window(
                WindowRequest {
                    mode,
                    chunk: 256,
                    stride: 255,
                    low_vram: false,
                },
                1920,
                1080,
                100,
                Some(16.0),
                256,
                MemoryEstimate::default(),
            )
            .unwrap();
            assert_eq!(
                (clean.chunk, clean.stride, clean.low_vram),
                (stale.chunk, stale.stride, stale.low_vram)
            );
        }
    }

    #[test]
    fn custom_zero_is_auto_and_custom_stride_zero_is_half() {
        let auto = resolve_window(
            WindowRequest {
                mode: WindowMode::Custom,
                ..Default::default()
            },
            1920,
            1080,
            50,
            None,
            256,
            MemoryEstimate::default(),
        )
        .unwrap();
        assert_eq!((auto.chunk, auto.stride), (80, 40));
        let explicit_stride = resolve_window(
            WindowRequest {
                mode: WindowMode::Custom,
                stride: 3,
                ..Default::default()
            },
            1920,
            1080,
            50,
            None,
            256,
            MemoryEstimate::default(),
        )
        .unwrap();
        assert_eq!((explicit_stride.chunk, explicit_stride.stride), (80, 3));
        assert!(resolve_window(
            WindowRequest {
                mode: WindowMode::Custom,
                stride: 3,
                ..Default::default()
            },
            1920,
            1080,
            50,
            Some(0.1),
            256,
            MemoryEstimate::default(),
        )
        .is_err());
        let custom = resolve_window(
            WindowRequest {
                mode: WindowMode::Custom,
                chunk: 9,
                stride: 0,
                low_vram: true,
            },
            1920,
            1080,
            5,
            None,
            256,
            MemoryEstimate::default(),
        )
        .unwrap();
        assert_eq!((custom.chunk, custom.stride, custom.low_vram), (9, 4, true));
        assert_eq!(
            crate::windows::WindowPlan::new(5, custom.chunk, custom.stride)
                .unwrap()
                .windows[0]
                .len(),
            5
        );
    }

    #[test]
    fn short_clip_recovery_never_repeats_the_same_effective_window() {
        let initial = resolve_window(
            WindowRequest {
                mode: WindowMode::Wide,
                ..Default::default()
            },
            1920,
            1080,
            2,
            None,
            256,
            MemoryEstimate::default(),
        )
        .unwrap();
        let retries = retry_choices(initial, 256, MemoryEstimate::default(), 1920, 1080, 2);
        assert_eq!(retries.len(), 3);
        assert_eq!(
            retries
                .iter()
                .map(|choice| (choice.chunk.min(2), choice.low_vram))
                .collect::<Vec<_>>(),
            [(2, false), (2, true), (1, true)]
        );
    }

    #[test]
    fn backend_calibration_and_real_small_budgets_change_selection() {
        let profile = MemoryEstimate {
            normal_extra_gib: 2.5,
            ..Default::default()
        };
        let calibrated = resolve_window(
            WindowRequest::default(),
            1920,
            1080,
            69,
            Some(12.0),
            256,
            profile,
        )
        .unwrap();
        assert_eq!(calibrated.chunk, 16);
        let tiny = resolve_window(
            WindowRequest::default(),
            1920,
            1080,
            69,
            Some(3.1),
            256,
            MemoryEstimate::default(),
        )
        .unwrap();
        assert!(tiny.low_vram);
        assert!(tiny.chunk <= 4);
        assert!(tiny.estimated_peak_gib <= 3.1);
    }

    #[test]
    fn graph_limits_validation_and_oom_recovery_reach_one() {
        let request = WindowRequest {
            mode: WindowMode::Custom,
            chunk: 24,
            stride: 12,
            low_vram: false,
        };
        assert!(
            resolve_window(request, 1920, 1080, 69, None, 8, MemoryEstimate::default()).is_err()
        );
        assert!(resolve_window(
            WindowRequest {
                chunk: 4,
                stride: 5,
                ..request
            },
            1920,
            1080,
            69,
            None,
            256,
            MemoryEstimate::default()
        )
        .is_err());
        let initial = resolve_window(
            request,
            1920,
            1080,
            69,
            Some(12.0),
            256,
            MemoryEstimate::default(),
        )
        .unwrap();
        let retries = retry_choices(initial, 256, MemoryEstimate::default(), 1920, 1080, 69);
        assert_eq!((retries[1].chunk, retries[1].low_vram), (12, false));
        assert_eq!(
            (
                retries.last().unwrap().chunk,
                retries.last().unwrap().low_vram
            ),
            (1, true)
        );
        assert_eq!(initial.chunk, 24);
    }
}
