//! CHW padding/cropping and resize contracts used around exported CK graphs.

use crate::{ContractError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Padding {
    pub native_height: usize,
    pub native_width: usize,
    pub height: usize,
    pub width: usize,
    pub left: usize,
    pub right: usize,
    pub top: usize,
    pub bottom: usize,
}

impl Padding {
    pub fn new(native_height: usize, native_width: usize, quantum: usize) -> Result<Self> {
        if native_height == 0 || native_width == 0 || quantum == 0 {
            return Err(ContractError(
                "positive native dimensions and shape quantum required".into(),
            ));
        }
        let ceil = |value: usize| {
            value
                .checked_add(quantum - 1)
                .map(|n| n / quantum * quantum)
                .ok_or_else(|| ContractError("padded dimension overflow".into()))
        };
        let height = ceil(native_height)?;
        let width = ceil(native_width)?;
        let left = (width - native_width) / 2;
        let top = (height - native_height) / 2;
        Ok(Self {
            native_height,
            native_width,
            height,
            width,
            left,
            right: width - native_width - left,
            top,
            bottom: height - native_height - top,
        })
    }

    /// OpenCV BORDER_REFLECT_101, including pads wider than the source image.
    pub fn pad_chw(&self, input: &[f32], channels: usize) -> Result<Vec<f32>> {
        check_shape(input, channels, self.native_height, self.native_width)?;
        let width = self
            .left
            .checked_add(self.native_width)
            .and_then(|width| width.checked_add(self.right));
        let height = self
            .top
            .checked_add(self.native_height)
            .and_then(|height| height.checked_add(self.bottom));
        if width != Some(self.width) || height != Some(self.height) {
            return Err(ContractError("inconsistent padded image geometry".into()));
        }
        let elements = image_elements(channels, self.height, self.width)?;
        if elements
            .checked_mul(std::mem::size_of::<f32>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(ContractError(
                "padded image exceeds addressable allocation size".into(),
            ));
        }
        if self.native_height == self.height && self.native_width == self.width {
            return Ok(input.to_vec());
        }
        // Most source pixels need no reflection at all. Copy whole source rows
        // instead of recalculating a signed remainder for every interior pixel.
        // Border maps are tiny, shared across rows/channels, and also cover
        // padding wider than the original image and one-pixel dimensions.
        let left: Vec<_> = (0..self.left)
            .map(|x| reflect101_index(x as isize - self.left as isize, self.native_width))
            .collect();
        let right: Vec<_> = (0..self.right)
            .map(|x| reflect101_index((self.native_width + x) as isize, self.native_width))
            .collect();
        let rows: Vec<_> = (0..self.height)
            .map(|y| reflect101_index(y as isize - self.top as isize, self.native_height))
            .collect();
        let mut output = Vec::with_capacity(elements);
        for channel in 0..channels {
            for &source_y in &rows {
                let start = (channel * self.native_height + source_y) * self.native_width;
                let source = &input[start..start + self.native_width];
                output.extend(left.iter().map(|&x| source[x]));
                output.extend_from_slice(source);
                output.extend(right.iter().map(|&x| source[x]));
            }
        }
        Ok(output)
    }

    pub fn crop_chw(&self, input: &[f32], channels: usize) -> Result<Vec<f32>> {
        check_shape(input, channels, self.height, self.width)?;
        let mut output = Vec::with_capacity(image_elements(
            channels,
            self.native_height,
            self.native_width,
        )?);
        for channel in 0..channels {
            for y in self.top..self.top + self.native_height {
                let offset = (channel * self.height + y) * self.width + self.left;
                output.extend_from_slice(&input[offset..offset + self.native_width]);
            }
        }
        Ok(output)
    }
}

pub fn reflect101_index(index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * (length as isize - 1);
    let reflected = index.rem_euclid(period);
    if reflected < length as isize {
        reflected as usize
    } else {
        (period - reflected) as usize
    }
}

/// Separable triangle resampling with half-pixel centers and antialiasing.
/// This is the float-input resize contract used by SAM's official transform.
pub fn resize_bilinear_antialias_chw(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    output_height: usize,
    output_width: usize,
) -> Result<Vec<f32>> {
    check_shape(input, channels, height, width)?;
    image_elements(channels, output_height, output_width)?;
    let weights = |source: usize, target: usize| -> Vec<(usize, Vec<f32>)> {
        let scale = source as f32 / target as f32;
        let support = scale.max(1.0);
        let inverse = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        (0..target)
            .map(|index| {
                let center = scale * (index as f32 + 0.5);
                let first = ((center - support + 0.5) as isize).max(0) as usize;
                let end = ((center + support + 0.5) as usize).min(source);
                let mut coefficients: Vec<f32> = (first..end)
                    .map(|position| {
                        (1.0 - ((position as f32 - center + 0.5) * inverse).abs()).max(0.0)
                    })
                    .collect();
                let sum: f32 = coefficients.iter().sum();
                for value in &mut coefficients {
                    *value /= sum;
                }
                (first, coefficients)
            })
            .collect()
    };
    let horizontal_weights = weights(width, output_width);
    let vertical_weights = weights(height, output_height);
    let mut horizontal = vec![0.0; image_elements(channels, height, output_width)?];
    for channel in 0..channels {
        for row in 0..height {
            for (column, (first, coefficients)) in horizontal_weights.iter().enumerate() {
                let base = (channel * height + row) * width + first;
                let mut value = input[base] * coefficients[0];
                for (offset, weight) in coefficients.iter().enumerate().skip(1) {
                    value += input[base + offset] * weight;
                }
                horizontal[(channel * height + row) * output_width + column] = value;
            }
        }
    }
    let mut output = vec![0.0; image_elements(channels, output_height, output_width)?];
    for channel in 0..channels {
        for (row, (first, coefficients)) in vertical_weights.iter().enumerate() {
            for column in 0..output_width {
                let base = (channel * height + first) * output_width + column;
                let mut value = horizontal[base] * coefficients[0];
                for (offset, weight) in coefficients.iter().enumerate().skip(1) {
                    value += horizontal[base + offset * output_width] * weight;
                }
                output[(channel * output_height + row) * output_width + column] = value;
            }
        }
    }
    Ok(output)
}

fn image_elements(channels: usize, height: usize, width: usize) -> Result<usize> {
    if channels == 0 || height == 0 || width == 0 {
        return Err(ContractError("positive CHW dimensions required".into()));
    }
    channels
        .checked_mul(height)
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| ContractError("CHW tensor size overflow".into()))
}

fn check_shape(input: &[f32], channels: usize, height: usize, width: usize) -> Result<()> {
    if input.len() != image_elements(channels, height, width)? {
        return Err(ContractError("CHW tensor length mismatch".into()));
    }
    Ok(())
}

/// PyTorch interpolate(mode="area"): adaptive average-pooling bins, not a
/// fractional pixel-area integral. Foundation preprocessing depends on this.
pub fn resize_area_chw(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    output_height: usize,
    output_width: usize,
) -> Result<Vec<f32>> {
    resize_area_chw_order(
        input,
        channels,
        height,
        width,
        output_height,
        output_width,
        false,
    )
}

/// Original PyTorch CUDA contiguous adaptive-average-pool arithmetic:
/// accumulate row-major, then divide by bin height and bin width separately.
/// This preserves the CPU routine's existing combined-denominator contract.
pub fn resize_area_chw_cuda_order(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    output_height: usize,
    output_width: usize,
) -> Result<Vec<f32>> {
    resize_area_chw_order(
        input,
        channels,
        height,
        width,
        output_height,
        output_width,
        true,
    )
}

fn resize_area_chw_order(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    output_height: usize,
    output_width: usize,
    cuda_order: bool,
) -> Result<Vec<f32>> {
    check_shape(input, channels, height, width)?;
    let mut output = vec![0.0; image_elements(channels, output_height, output_width)?];
    for channel in 0..channels {
        for oy in 0..output_height {
            let y0 = oy * height / output_height;
            let y1 = ((oy + 1) * height).div_ceil(output_height);
            for ox in 0..output_width {
                let x0 = ox * width / output_width;
                let x1 = ((ox + 1) * width).div_ceil(output_width);
                let mut sum = 0.0f32;
                for y in y0..y1 {
                    for x in x0..x1 {
                        sum += input[(channel * height + y) * width + x];
                    }
                }
                output[(channel * output_height + oy) * output_width + ox] = if cuda_order {
                    (sum / (y1 - y0) as f32) / (x1 - x0) as f32
                } else {
                    sum / ((y1 - y0) * (x1 - x0)) as f32
                };
            }
        }
    }
    Ok(output)
}

/// PyTorch interpolate(mode="bilinear", align_corners=False), border clamped.
pub fn resize_bilinear_chw(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    output_height: usize,
    output_width: usize,
) -> Result<Vec<f32>> {
    resize_bilinear_chw_impl::<false, false>(
        input,
        channels,
        height,
        width,
        output_height,
        output_width,
    )
}

/// The CK CUDA hint-carry bilinear contract, with the first weighted term fused.
///
/// Recorded Torch CUDA outputs distinguish this arithmetic order from the CPU
/// contract above. Keeping it separate preserves SAM/CPU interpolation behavior.
pub fn resize_bilinear_chw_cuda_order(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    output_height: usize,
    output_width: usize,
) -> Result<Vec<f32>> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("fma") {
        // SAFETY: runtime detection guards the entire FMA-enabled kernel.
        return unsafe {
            resize_bilinear_chw_hardware(
                input,
                channels,
                height,
                width,
                output_height,
                output_width,
            )
        };
    }
    resize_bilinear_chw_impl::<true, false>(
        input,
        channels,
        height,
        width,
        output_height,
        output_width,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
unsafe fn resize_bilinear_chw_hardware(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    output_height: usize,
    output_width: usize,
) -> Result<Vec<f32>> {
    resize_bilinear_chw_impl::<true, true>(
        input,
        channels,
        height,
        width,
        output_height,
        output_width,
    )
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[target_feature(enable = "fma")]
unsafe fn hardware_fma(first: f32, second: f32, addend: f32) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::{_mm_cvtss_f32, _mm_fmadd_ss, _mm_set_ss};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::{_mm_cvtss_f32, _mm_fmadd_ss, _mm_set_ss};
    _mm_cvtss_f32(_mm_fmadd_ss(
        _mm_set_ss(first),
        _mm_set_ss(second),
        _mm_set_ss(addend),
    ))
}

#[inline(always)]
fn fused_multiply_add<const HARDWARE: bool>(first: f32, second: f32, addend: f32) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if HARDWARE {
        // SAFETY: this specialization is only called by the detected FMA kernel.
        return unsafe { hardware_fma(first, second, addend) };
    }
    // Generic Windows GNU f32::mul_add can lower to a CRT fmaf which double
    // rounds this recorded oracle. libm supplies the IEEE software fallback.
    libm::fmaf(first, second, addend)
}

#[inline(always)]
fn resize_bilinear_chw_impl<const FUSED: bool, const HARDWARE: bool>(
    input: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    output_height: usize,
    output_width: usize,
) -> Result<Vec<f32>> {
    check_shape(input, channels, height, width)?;
    let mut output = vec![0.0; image_elements(channels, output_height, output_width)?];
    let coordinate = |out: usize, source_length: usize, output_length: usize| {
        let position =
            ((out as f32 + 0.5) * (source_length as f32 / output_length as f32) - 0.5).max(0.0);
        let first = (position.floor() as usize).min(source_length - 1);
        (
            first,
            (first + 1).min(source_length - 1),
            position - first as f32,
        )
    };
    for channel in 0..channels {
        for oy in 0..output_height {
            let (y0, y1, wy) = coordinate(oy, height, output_height);
            for ox in 0..output_width {
                let (x0, x1, wx) = coordinate(ox, width, output_width);
                let at = |y, x| input[(channel * height + y) * width + x];
                output[(channel * output_height + oy) * output_width + ox] = if FUSED {
                    let upper =
                        fused_multiply_add::<HARDWARE>(1.0 - wx, at(y0, x0), wx * at(y0, x1));
                    let lower =
                        fused_multiply_add::<HARDWARE>(1.0 - wx, at(y1, x0), wx * at(y1, x1));
                    fused_multiply_add::<HARDWARE>(1.0 - wy, upper, wy * lower)
                } else {
                    let upper = at(y0, x0) * (1.0 - wx) + at(y0, x1) * wx;
                    let lower = at(y1, x0) * (1.0 - wx) + at(y1, x1) * wx;
                    upper * (1.0 - wy) + lower * wy
                };
            }
        }
    }
    Ok(output)
}

/// PIL `Image.resize(size, Image.BILINEAR)` on interleaved 8-bit pixels (whole-image box, no
/// reducing gap), as torchvision's `resize` applies it to PIL images: a separable triangle filter
/// with 22-bit fixed-point coefficients, horizontal pass first, each pass rounded to u8.
pub fn resize_bilinear_pil_u8(
    input: &[u8],
    channels: usize,
    height: usize,
    width: usize,
    output_height: usize,
    output_width: usize,
) -> Result<Vec<u8>> {
    const PRECISION_BITS: u32 = 32 - 8 - 2;
    let elements = image_elements(channels, height, width)?;
    image_elements(channels, output_height, output_width)?;
    if input.len() != elements {
        return Err(ContractError("HWC u8 tensor length mismatch".into()));
    }
    // PIL precompute_coeffs + normalize_coeffs_8bpc: (first source index, fixed-point weights).
    let coefficients = |source: usize, target: usize| -> Vec<(usize, Vec<i64>)> {
        let scale = source as f64 / target as f64;
        let filter_scale = scale.max(1.0);
        let support = filter_scale;
        (0..target)
            .map(|index| {
                let center = (index as f64 + 0.5) * scale;
                let first = ((center - support + 0.5) as i64).max(0) as usize;
                let end = ((center + support + 0.5) as i64).min(source as i64) as usize;
                let weights: Vec<f64> = (first..end)
                    .map(|position| {
                        let x = ((position as f64 - center + 0.5) / filter_scale).abs();
                        if x < 1.0 {
                            1.0 - x
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let total: f64 = weights.iter().sum();
                let fixed = weights
                    .iter()
                    .map(|&weight| {
                        let weight = if total != 0.0 { weight / total } else { weight };
                        let scaled = weight * f64::from(1u32 << PRECISION_BITS);
                        (if weight < 0.0 {
                            scaled - 0.5
                        } else {
                            scaled + 0.5
                        }) as i64
                    })
                    .collect();
                (first, fixed)
            })
            .collect()
    };
    let clip8 = |sum: i64| -> u8 { (sum >> PRECISION_BITS).clamp(0, 255) as u8 };
    let half = 1i64 << (PRECISION_BITS - 1);
    let horizontal = if output_width == width {
        input.to_vec()
    } else {
        let weights = coefficients(width, output_width);
        let mut out = vec![0u8; height * output_width * channels];
        parallel_rows(&mut out, output_width * channels, |row, line| {
            for (column, (first, fixed)) in weights.iter().enumerate() {
                for channel in 0..channels {
                    let mut sum = half;
                    for (offset, weight) in fixed.iter().enumerate() {
                        let source = (row * width + first + offset) * channels + channel;
                        sum += i64::from(input[source]) * weight;
                    }
                    line[column * channels + channel] = clip8(sum);
                }
            }
        });
        out
    };
    if output_height == height {
        return Ok(horizontal);
    }
    let weights = coefficients(height, output_height);
    let mut out = vec![0u8; output_height * output_width * channels];
    parallel_rows(&mut out, output_width * channels, |row, line| {
        let (first, fixed) = &weights[row];
        for column in 0..output_width {
            for channel in 0..channels {
                let mut sum = half;
                for (offset, weight) in fixed.iter().enumerate() {
                    let source = ((first + offset) * output_width + column) * channels + channel;
                    sum += i64::from(horizontal[source]) * weight;
                }
                line[column * channels + channel] = clip8(sum);
            }
        }
    });
    Ok(out)
}

/// Runs `f(row, row_slice)` for every `row_len`-sized row of `out`, split across up to 8 threads.
/// Each output element is computed exactly as in a sequential loop.
pub fn parallel_rows<T: Send>(out: &mut [T], row_len: usize, f: impl Fn(usize, &mut [T]) + Sync) {
    let rows = out.len().checked_div(row_len).unwrap_or(0);
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, 8)
        .min(rows.max(1));
    if workers <= 1 {
        for (row, line) in out.chunks_mut(row_len.max(1)).enumerate() {
            f(row, line);
        }
        return;
    }
    let per = rows.div_ceil(workers);
    std::thread::scope(|scope| {
        for (index, block) in out.chunks_mut(per * row_len).enumerate() {
            let f = &f;
            scope.spawn(move || {
                for (offset, line) in block.chunks_mut(row_len).enumerate() {
                    f(index * per + offset, line);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pil_bilinear_u8_matches_pillow_bytes() {
        // Pillow 11 `Image.fromarray(src, "RGB").resize((w, h), Image.BILINEAR)` known answers for a
        // seeded 7x5 RGB image: downscale, mixed down/up and upscale.
        let src: [u8; 105] = [
            139, 74, 229, 241, 169, 65, 6, 160, 149, 106, 38, 175, 188, 205, 175, 229, 98, 249, 10,
            148, 95, 86, 147, 198, 66, 39, 106, 213, 171, 45, 167, 57, 69, 81, 55, 14, 153, 178,
            215, 76, 52, 39, 250, 72, 215, 50, 161, 223, 112, 172, 161, 233, 38, 17, 89, 1, 221, 6,
            242, 127, 143, 6, 60, 210, 94, 25, 166, 33, 249, 189, 12, 204, 33, 57, 124, 30, 199,
            149, 202, 119, 72, 220, 3, 209, 122, 136, 147, 77, 133, 82, 115, 87, 162, 230, 70, 71,
            240, 62, 47, 184, 31, 34, 63, 65, 146,
        ];
        let down = resize_bilinear_pil_u8(&src, 3, 5, 7, 3, 4).unwrap();
        assert_eq!(
            down,
            [
                136, 112, 153, 120, 122, 107, 150, 113, 139, 111, 119, 143, 148, 101, 182, 143,
                105, 131, 104, 83, 124, 115, 97, 111, 140, 103, 124, 140, 64, 151, 157, 94, 84,
                157, 55, 116
            ]
        );
        let mixed = resize_bilinear_pil_u8(&src, 3, 5, 7, 5, 3).unwrap();
        assert_eq!(
            mixed,
            [
                154, 133, 142, 117, 117, 167, 142, 139, 178, 105, 106, 126, 151, 89, 61, 110, 107,
                109, 135, 132, 207, 147, 80, 117, 73, 106, 123, 187, 50, 159, 87, 82, 153, 172, 95,
                137, 101, 124, 122, 194, 74, 85, 153, 50, 77
            ]
        );
        let up = resize_bilinear_pil_u8(&src, 3, 5, 7, 9, 11).unwrap();
        assert_eq!(
            &up[..24],
            [
                139, 74, 229, 185, 117, 154, 220, 168, 73, 70, 162, 126, 42, 116, 158, 106, 38,
                175, 158, 144, 175, 199, 176, 195
            ]
        );
        assert_eq!(
            &up[up.len() - 12..],
            [225, 54, 43, 189, 34, 35, 118, 50, 95, 63, 65, 146]
        );
        assert_eq!(up.len(), 9 * 11 * 3);
        assert!(resize_bilinear_pil_u8(&src[1..], 3, 5, 7, 3, 4).is_err());
    }

    #[test]
    fn cuda_area_matches_original_gpu_bits_without_changing_cpu_division() {
        // Independent original Torch 2.9.1 CUDA FP32 known answer. The output
        // is 2x1, avoiding PyTorch's separate 1x1 global-mean fast path.
        // Each 5x6 bin holds one nonzero value, so this isolates division order.
        let mut input = vec![0.0; 3 * 10 * 6];
        for (channel, values) in [[0.1, 1.0], [0.18, 0.2], [-0.1, 4.0]].iter().enumerate() {
            input[channel * 60] = values[0];
            input[channel * 60 + 30] = values[1];
        }
        let actual = resize_area_chw_cuda_order(&input, 3, 10, 6, 2, 1).unwrap();
        let expected_bits = [
            0x3b5a740d, 0x3d088889, 0x3bc49ba7, 0x3bda740d, 0xbb5a740d, 0x3e088889,
        ];
        assert_eq!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_bits
        );
        let cpu = resize_area_chw(&input, 3, 10, 6, 2, 1).unwrap();
        assert_eq!(cpu[0].to_bits(), 0x3b5a740e);
        assert_eq!(cpu[2].to_bits(), 0x3bc49ba6);
        assert_ne!(actual[0].to_bits(), cpu[0].to_bits());
    }

    #[test]
    fn ck_padding_is_centered_quantum128_and_crop_reverses_it() {
        let padding = Padding::new(1080, 1920, 128).unwrap();
        assert_eq!(
            (padding.height, padding.width, padding.top, padding.bottom),
            (1152, 1920, 36, 36)
        );
        let tiny = Padding::new(3, 4, 128).unwrap();
        let source: Vec<_> = (0..24).map(|value| value as f32).collect();
        let padded = tiny.pad_chw(&source, 2).unwrap();
        assert_eq!(tiny.crop_chw(&padded, 2).unwrap(), source);
    }

    #[test]
    fn reflect101_omits_edge_duplication_and_handles_singletons() {
        let actual: Vec<_> = (-5..=7).map(|index| reflect101_index(index, 4)).collect();
        assert_eq!(actual, [1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1]);
        assert_eq!(reflect101_index(-100, 1), 0);
    }

    fn folding_reference(padding: Padding, input: &[f32], channels: usize) -> Vec<f32> {
        // Independent BORDER_REFLECT_101 oracle: repeatedly fold coordinates
        // over the image edges rather than using the production modulo map.
        let fold = |mut position: isize, length: usize| {
            if length == 1 {
                return 0;
            }
            let last = length as isize - 1;
            while position < 0 || position > last {
                position = if position < 0 {
                    -position
                } else {
                    last * 2 - position
                };
            }
            position as usize
        };
        let mut expected = Vec::with_capacity(channels * padding.height * padding.width);
        for channel in 0..channels {
            for y in 0..padding.height {
                for x in 0..padding.width {
                    let source_y = fold(y as isize - padding.top as isize, padding.native_height);
                    let source_x = fold(x as isize - padding.left as isize, padding.native_width);
                    expected.push(
                        input[(channel * padding.native_height + source_y) * padding.native_width
                            + source_x],
                    );
                }
            }
        }
        expected
    }

    #[test]
    fn row_copy_padding_matches_independent_folding_reference_bit_for_bit() {
        let mut cases = Vec::new();
        for height in [1, 2, 3, 7, 8, 9, 17] {
            for width in [1, 2, 3, 7, 8, 9, 17] {
                for quantum in [1, 2, 8, 32] {
                    for channels in [1, 3] {
                        cases.push((height, width, quantum, channels));
                    }
                }
            }
        }
        cases.extend([
            (1, 7, 256, 4),
            (7, 1, 256, 4),
            (2, 3, 128, 2),
            (3, 4, 512, 1),
            (479, 853, 128, 3),
        ]);
        for (height, width, quantum, channels) in cases {
            let padding = Padding::new(height, width, quantum).unwrap();
            let source: Vec<_> = (0..height * width * channels)
                .map(|index| {
                    // Include signed zero, infinities and NaN payloads: padding is
                    // a byte-preserving copy, not floating-point arithmetic.
                    let bits = match index % 13 {
                        0 => 0x8000_0000,
                        1 => 0x7f80_0000,
                        2 => 0xff80_0000,
                        3 => 0x7fc0_1234,
                        4 => 0x7fa0_4321,
                        _ => (index as u32).wrapping_mul(0x9e37_79b9).rotate_left(7),
                    };
                    f32::from_bits(bits)
                })
                .collect();
            let expected = folding_reference(padding, &source, channels);
            let actual = padding.pad_chw(&source, channels).unwrap();
            assert_eq!(actual.len(), expected.len());
            let mismatch = actual
                .iter()
                .zip(&expected)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert!(mismatch.is_none(), "{height}x{width}, quantum {quantum}, channels {channels}: mismatch at {mismatch:?}");
            let cropped = padding.crop_chw(&actual, channels).unwrap();
            assert!(cropped
                .iter()
                .zip(&source)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn padding_rejects_inconsistent_public_geometry_and_capacity_before_allocation() {
        let valid = Padding::new(2, 3, 8).unwrap();
        let source = [0.0; 6];
        for malformed in [
            Padding {
                right: valid.right + 1,
                ..valid
            },
            Padding {
                bottom: valid.bottom + 1,
                ..valid
            },
            Padding {
                left: usize::MAX,
                ..valid
            },
            Padding {
                top: usize::MAX,
                ..valid
            },
        ] {
            assert!(malformed.pad_chw(&source, 1).is_err());
        }
        // Even the no-padding shortcut must reject inconsistent public fields.
        let invalid_shortcut = Padding {
            left: 1,
            ..Padding::new(1, 1, 1).unwrap()
        };
        assert!(invalid_shortcut.pad_chw(&[1.0], 1).is_err());

        // Valid axis sums but an impossible Vec<f32> allocation must fail before
        // constructing gigantic border maps or invoking the allocator.
        let width = isize::MAX as usize / std::mem::size_of::<f32>() + 1;
        let impossible = Padding {
            native_height: 1,
            native_width: 1,
            height: 1,
            width,
            left: 0,
            right: width - 1,
            top: 0,
            bottom: 0,
        };
        assert!(impossible.pad_chw(&[1.0], 1).is_err());
        let product_overflow = Padding {
            height: usize::MAX,
            bottom: usize::MAX - 1,
            width: usize::MAX,
            right: usize::MAX - 1,
            ..Padding::new(1, 1, 1).unwrap()
        };
        assert!(product_overflow.pad_chw(&[1.0], 1).is_err());
    }

    fn legacy_pixel_padding(padding: Padding, input: &[f32], channels: usize) -> Vec<f32> {
        // Original implementation retained only as the CPU benchmark baseline.
        let mut output = vec![0.0; channels * padding.height * padding.width];
        for channel in 0..channels {
            for y in 0..padding.height {
                let source_y =
                    reflect101_index(y as isize - padding.top as isize, padding.native_height);
                for x in 0..padding.width {
                    let source_x =
                        reflect101_index(x as isize - padding.left as isize, padding.native_width);
                    output[(channel * padding.height + y) * padding.width + x] =
                        input[(channel * padding.native_height + source_y) * padding.native_width
                            + source_x];
                }
            }
        }
        output
    }

    #[test]
    #[ignore = "Explicit CPU-only release microbenchmark; about300MB peak RAM, no models/GPU"]
    fn benchmark_reflect101_padding_row_copy_at_1080p_and_4k() {
        use std::{hint::black_box, time::Instant};
        if cfg!(debug_assertions) {
            panic!("Run this CPU benchmark with --release");
        }
        for (height, width) in [(1080, 1920), (2160, 3840), (479, 853)] {
            let padding = Padding::new(height, width, 128).unwrap();
            let input: Vec<_> = (0..height * width * 3)
                .map(|index| (index % 251) as f32 / 255.0)
                .collect();
            let legacy = legacy_pixel_padding(black_box(padding), black_box(&input), 3);
            let actual = padding.pad_chw(black_box(&input), 3).unwrap();
            assert!(actual
                .iter()
                .zip(&legacy)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            drop(actual);
            drop(legacy);
            let mut legacy_seconds = Vec::new();
            let mut row_seconds = Vec::new();
            for _ in 0..5 {
                let started = Instant::now();
                let legacy = legacy_pixel_padding(black_box(padding), black_box(&input), 3);
                legacy_seconds.push(started.elapsed().as_secs_f64());
                black_box(&legacy);
                drop(legacy);
                let started = Instant::now();
                let actual = black_box(padding).pad_chw(black_box(&input), 3).unwrap();
                row_seconds.push(started.elapsed().as_secs_f64());
                black_box(&actual);
                drop(actual);
            }
            legacy_seconds.sort_by(f64::total_cmp);
            row_seconds.sort_by(f64::total_cmp);
            println!(
                "CPU_REFLECT101_PADDING {{\"source_hw\":[{height},{width}],\"channels\":3,\"padded_hw\":[{},{}],\"repeats\":5,\"legacy_median_ms\":{:.6},\"row_copy_median_ms\":{:.6},\"median_speedup\":{:.6},\"bit_exact\":true,\"gpu_used\":false}}",
                padding.height, padding.width, legacy_seconds[2] * 1000.0, row_seconds[2] * 1000.0, legacy_seconds[2] / row_seconds[2]
            );
        }
    }

    #[test]
    fn area_is_adaptive_pooling_for_nonintegral_ratio() {
        assert_eq!(
            resize_area_chw(&[0.0, 1.0, 2.0, 3.0, 4.0], 1, 1, 5, 1, 3).unwrap(),
            [0.5, 2.0, 3.5]
        );
        assert_eq!(
            resize_area_chw(&[1.0, 2.0, 3.0, 4.0], 1, 2, 2, 1, 1).unwrap(),
            [2.5]
        );
    }

    #[test]
    fn bilinear_uses_half_pixel_coordinates_not_align_corners() {
        assert_eq!(
            resize_bilinear_chw(&[0.0, 1.0], 1, 1, 2, 1, 4).unwrap(),
            [0.0, 0.25, 0.75, 1.0]
        );
    }

    #[test]
    fn ck_cuda_bilinear_matches_recorded_actual_aux_oracle_bits() {
        // Actual window8 auxiliary alpha at y47,x932. These four INTERIOR
        // output bits were recorded from full-size original Torch CUDA resize,
        // independent of this implementation. Values are inside (0,1).
        let input = [994428560, 987983184, 991355640, 983499712].map(f32::from_bits);
        let expected = [992335005, 989512966, 990902443, 987063434];
        let actual = resize_bilinear_chw_cuda_order(&input, 1, 2, 2, 4, 4).unwrap();
        assert_eq!([5, 6, 9, 10].map(|index| actual[index].to_bits()), expected);
        let cpu = resize_bilinear_chw(&input, 1, 2, 2, 4, 4).unwrap();
        assert_ne!([5, 6, 9, 10].map(|index| cpu[index].to_bits()), expected);
        let portable = resize_bilinear_chw_impl::<true, false>(&input, 1, 2, 2, 4, 4).unwrap();
        assert_eq!(
            [5, 6, 9, 10].map(|index| portable[index].to_bits()),
            expected
        );
    }
}
