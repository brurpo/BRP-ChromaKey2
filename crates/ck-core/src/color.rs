//! Signed color transforms from CK's `corridorkey/color_utils.py`.

pub const HDR_ASINH_KNEE: f32 = 0.008;
pub const HDR_ASINH_REF: f32 = 16.0;

/// Sign-preserving IEC 61966-2-1 display RGB to scene-linear RGB.
pub fn srgb_to_linear(value: f32) -> f32 {
    let magnitude = value.abs();
    let linear = if magnitude <= 0.04045 {
        magnitude / 12.92
    } else {
        ((magnitude + 0.055) / 1.055).powf(2.4)
    };
    linear.copysign(value)
}

/// Sign-preserving IEC 61966-2-1 scene-linear RGB to display RGB.
pub fn linear_to_srgb(value: f32) -> f32 {
    let magnitude = value.abs();
    let display = if magnitude <= 0.003_130_8 {
        magnitude * 12.92
    } else {
        1.055 * magnitude.powf(1.0 / 2.4) - 0.055
    };
    display.copysign(value)
}

/// CK model RGB. Values above the HDR reference remain above one; no clamp.
pub fn linear_to_asinh(value: f32) -> f32 {
    (value / HDR_ASINH_KNEE).asinh() / (HDR_ASINH_REF / HDR_ASINH_KNEE).asinh()
}

pub fn asinh_to_linear(value: f32) -> f32 {
    (value * (HDR_ASINH_REF / HDR_ASINH_KNEE).asinh()).sinh() * HDR_ASINH_KNEE
}

/// Display input for the frozen foundations after optional linear adjustment.
pub fn linear_to_foundation_sdr(value: f32) -> f32 {
    linear_to_srgb(value.max(0.0)).clamp(0.0, 1.0)
}

pub fn apply_linear_adjust(value: f32, brightness: f32, contrast: f32, pivot: f32) -> f32 {
    ((value - pivot) * contrast + pivot) * brightness
}

pub fn inverse_linear_adjust(value: f32, brightness: f32, contrast: f32, pivot: f32) -> f32 {
    (value / brightness - pivot) / contrast + pivot
}

/// Native stage-2 proof output is straight linear foreground and unit alpha.
/// This conversion is for display only and must not be applied to linear EXR.
pub fn model_foreground_to_display(value: f32) -> f32 {
    linear_to_srgb(asinh_to_linear(value).max(0.0)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_srgb_matches_reference_points_and_round_trips() {
        for (display, linear) in [
            (0.0, 0.0),
            (0.04045, 0.003130805),
            (0.5, 0.21404114),
            (1.0, 1.0),
        ] {
            assert!((srgb_to_linear(display) - linear).abs() < 2e-7);
            assert!((srgb_to_linear(-display) + linear).abs() < 2e-7);
        }
        for value in [-3.0, -0.5, -0.001, 0.0, 0.001, 0.18, 1.0, 4.0] {
            let recovered = srgb_to_linear(linear_to_srgb(value));
            assert!((recovered - value).abs() < 2e-6, "{value} -> {recovered}");
        }
    }

    #[test]
    fn model_encoding_preserves_signed_hdr() {
        assert!((linear_to_asinh(16.0) - 1.0).abs() < 1e-7);
        assert!(linear_to_asinh(32.0) > 1.0);
        for value in [-32.0, -1.0, -0.008, 0.0, 0.008, 0.18, 1.0, 16.0, 32.0] {
            let recovered = asinh_to_linear(linear_to_asinh(value));
            assert!((recovered - value).abs() <= value.abs().max(1.0) * 2e-6);
        }
    }

    #[test]
    fn linear_adjust_inverse_and_foundation_clamp_are_explicit() {
        let value = -0.03;
        assert!(
            (inverse_linear_adjust(apply_linear_adjust(value, 1.2, 0.8, 0.18), 1.2, 0.8, 0.18)
                - value)
                .abs()
                < 1e-7
        );
        assert_eq!(linear_to_foundation_sdr(-1.0), 0.0);
        assert_eq!(linear_to_foundation_sdr(4.0), 1.0);
    }
}
