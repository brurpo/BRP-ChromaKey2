//! Exact bulk expansion of half outputs after native GPU readback.
//!
//! `half` selects its vector conversion with runtime CPU feature detection. No
//! application-wide CPU target feature is required, and unsupported processors
//! retain the portable conversion. Neural precision and output values do not
//! change: every finite binary16 value is exactly representable in binary32.
use half::{f16, slice::HalfFloatSliceExt};
use ndarray::{ArrayD, ArrayViewD};

pub(crate) fn to_f32(array: ArrayViewD<'_, f16>) -> ArrayD<f32> {
    let Some(source) = array.as_slice() else {
        // Preserve the existing semantics for strided and Fortran-order arrays.
        return array.mapv(f16::to_f32);
    };
    let mut output = ArrayD::<f32>::zeros(array.raw_dim());
    source.convert_to_f32_slice(
        output
            .as_slice_mut()
            .expect("new half readback array has standard layout"),
    );
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3, IxDyn, s};
    use std::{hint::black_box, time::Instant};

    fn reference(bits: u16) -> f32 {
        let exponent = (bits >> 10) & 31;
        let fraction = bits & 1023;
        let magnitude = match exponent {
            0 => f32::from(fraction) * 2_f32.powi(-24),
            31 if fraction == 0 => f32::INFINITY,
            31 => f32::NAN,
            _ => (1.0 + f32::from(fraction) / 1024.0) * 2_f32.powi(i32::from(exponent) - 15),
        };
        if bits & 0x8000 == 0 {
            magnitude
        } else {
            -magnitude
        }
    }

    #[test]
    fn every_finite_half_pattern_is_bit_exact_and_nonfinite_stays_nonfinite() {
        let source = ArrayD::from_shape_vec(
            IxDyn(&[256, 256]),
            (0..=u16::MAX).map(f16::from_bits).collect(),
        )
        .unwrap();
        let output = to_f32(source.view());
        assert_eq!(output.shape(), &[256, 256]);
        for (bits, actual) in (0..=u16::MAX).zip(output.iter()) {
            let expected = reference(bits);
            if expected.is_finite() || expected.is_infinite() {
                assert_eq!(actual.to_bits(), expected.to_bits(), "half bits {bits:04x}");
            } else {
                assert!(actual.is_nan(), "half NaN bits {bits:04x}");
            }
        }
        assert_eq!(output[[0, 0]].to_bits(), 0.0_f32.to_bits());
        assert_eq!(output[[128, 0]].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn strided_and_fortran_inputs_keep_logical_shape_and_values() {
        let values: Vec<_> = (0..24)
            .map(|value| f16::from_f32(value as f32 - 12.0))
            .collect();
        let source = Array3::from_shape_vec((2, 3, 4), values).unwrap();
        let strided = source.slice(s![.., ..;-1, ..;2]).into_dyn();
        assert!(strided.as_slice().is_none());
        let output = to_f32(strided.view());
        assert_eq!(output.shape(), strided.shape());
        assert!(
            output
                .iter()
                .zip(strided.iter())
                .all(|(actual, before)| actual.to_bits() == before.to_f32().to_bits())
        );

        let source = Array2::from_shape_vec(
            (2, 3),
            (0..6).map(|value| f16::from_f32(value as f32)).collect(),
        )
        .unwrap();
        let fortran = source.reversed_axes();
        assert!(fortran.as_slice().is_none());
        let output = to_f32(fortran.view().into_dyn());
        assert_eq!(output.shape(), fortran.shape());
        assert!(
            output
                .iter()
                .zip(fortran.iter())
                .all(|(actual, before)| actual.to_bits() == before.to_f32().to_bits())
        );
    }

    #[test]
    fn scalar_and_empty_outputs_are_supported() {
        let scalar = ArrayD::from_elem(IxDyn(&[]), f16::from_bits(0x8001));
        assert_eq!(
            to_f32(scalar.view())[IxDyn(&[])].to_bits(),
            reference(0x8001).to_bits()
        );
        let empty = ArrayD::from_elem(IxDyn(&[0, 3]), f16::ZERO);
        let output = to_f32(empty.view());
        assert_eq!(output.shape(), &[0, 3]);
        assert!(output.is_empty());
    }

    fn acceleration() -> &'static str {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::arch::is_x86_feature_detected!("f16c") {
            return "runtime_f16c_bulk";
        }
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("fp16") {
            return "runtime_fp16_bulk";
        }
        "portable_fallback"
    }

    #[test]
    #[ignore = "Explicit CPU-only 1080p T8 readback benchmark; about 354MB peak, no models/GPU"]
    fn benchmark_real_stage1_baton_geometry_against_elementwise_map() {
        let shape = [1, 8, 512, 72, 120];
        let elements: usize = shape.iter().product();
        let source = ArrayD::from_shape_vec(
            IxDyn(&shape),
            (0..elements)
                .map(|index| {
                    let bits = ((index % 0x7c00) as u16) | if index & 1 == 0 { 0 } else { 0x8000 };
                    f16::from_bits(bits)
                })
                .collect(),
        )
        .unwrap();
        let mut scalar_seconds = Vec::new();
        let mut bulk_seconds = Vec::new();
        for run in 0..4_usize {
            let measure_scalar = || {
                let start = Instant::now();
                let output = black_box(source.view()).mapv(f16::to_f32);
                (output, start.elapsed().as_secs_f64())
            };
            let measure_bulk = || {
                let start = Instant::now();
                let output = to_f32(black_box(source.view()));
                (output, start.elapsed().as_secs_f64())
            };
            // One warmup, then alternate execution order to limit cache bias.
            let ((scalar, scalar_time), (bulk, bulk_time)) = if run.is_multiple_of(2) {
                (measure_scalar(), measure_bulk())
            } else {
                let bulk = measure_bulk();
                (measure_scalar(), bulk)
            };
            assert!(
                scalar
                    .iter()
                    .zip(bulk.iter())
                    .all(|(left, right)| left.to_bits() == right.to_bits())
            );
            if run > 0 {
                scalar_seconds.push(scalar_time);
                bulk_seconds.push(bulk_time);
            }
            black_box((scalar, bulk));
        }
        let scalar_mean = scalar_seconds.iter().sum::<f64>() / scalar_seconds.len() as f64;
        let bulk_mean = bulk_seconds.iter().sum::<f64>() / bulk_seconds.len() as f64;
        println!(
            "CPU_HALF_READBACK {}",
            serde_json::json!({
                "shape":shape,"elements":elements,"acceleration":acceleration(),
                "scalar_seconds":scalar_seconds,"bulk_seconds":bulk_seconds,
                "scalar_mean_seconds":scalar_mean,"bulk_mean_seconds":bulk_mean,
                "scalar_over_bulk":scalar_mean/bulk_mean,
                "all_finite_output_bits_equal":true,"gpu_executed":false,
                "baseline":"Original ndarray elementwise map in the same std-enabled build; scalar may also use F16C, so this does not time the frozen software-only build",
                "scope":"CPU conversion only; excludes GPU readback, model execution and finite scan"
            })
        );
    }
}
