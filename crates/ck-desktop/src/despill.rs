//! CPU port of v1's display-space colour unmix. The original plate and rendered
//! matte estimate a local screen colour; calculated alpha is never modified.
use anyhow::{Context, Result, ensure};
use ck_desktop_core::DespillSettings;
use std::sync::{Arc, OnceLock};

type Rgb = [f32; 3];
type Weighted = [f32; 4];

/// Per-pixel despill amount from a garbage/despill mask weight image
/// (v1 `masklayers.despill_weight`, 0 = protected, 255 = full unmix).
#[derive(Clone, Copy)]
pub struct MaskWeight<'a> {
    width: usize,
    height: usize,
    data: &'a [u8],
}

impl<'a> MaskWeight<'a> {
    /// Wrap an 8-bit weight image (usually the low-res combined despill mask).
    pub fn from_gray(gray: &'a image::GrayImage) -> Self {
        Self {
            width: gray.width() as usize,
            height: gray.height() as usize,
            data: gray.as_raw(),
        }
    }

    /// Bilinear sample of the (possibly upsampled) weight, 0..1.
    fn sample(&self, x: usize, y: usize, dest_width: usize, dest_height: usize) -> f32 {
        if self.width == 0 || self.height == 0 || dest_width == 0 || dest_height == 0 {
            return 1.0;
        }
        let fx = ((x as f32 + 0.5) * self.width as f32 / dest_width as f32 - 0.5)
            .clamp(0.0, (self.width - 1) as f32);
        let fy = ((y as f32 + 0.5) * self.height as f32 / dest_height as f32 - 0.5)
            .clamp(0.0, (self.height - 1) as f32);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let at = |x: usize, y: usize| f32::from(self.data[y * self.width + x]) * (1.0 / 255.0);
        let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
        let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
        top + (bottom - top) * ty
    }
}

fn dimensions(width: u32, height: u32, rgba: &[u8]) -> Result<(usize, usize)> {
    let width = usize::try_from(width)?;
    let height = usize::try_from(height)?;
    let bytes = width
        .checked_mul(height)
        .and_then(|count| count.checked_mul(4))
        .context("Despill dimensions overflow")?;
    ensure!(
        width > 0 && height > 0 && bytes <= 512 * 1024 * 1024,
        "Invalid despill dimensions"
    );
    ensure!(rgba.len() == bytes, "Despill image dimensions do not match");
    Ok((width, height))
}

/// OpenCV INTER_AREA overlap weights, including non-multiple-of-four dimensions.
fn area_axis(source: usize, target: usize) -> Vec<Vec<(usize, f64)>> {
    let scale = source as f64 / target as f64;
    (0..target)
        .map(|index| {
            let lo = index as f64 * scale;
            let hi = (index + 1) as f64 * scale;
            (lo.floor() as usize..(hi.ceil() as usize).min(source))
                .filter_map(|sample| {
                    let overlap = (hi.min((sample + 1) as f64) - lo.max(sample as f64)).max(0.0);
                    (overlap > 0.0).then_some((sample, overlap / scale))
                })
                .collect()
        })
        .collect()
}

#[derive(Clone, Copy)]
struct LinearSample {
    lo: usize,
    hi: usize,
    fraction: f32,
}
fn linear_axis(source: usize, target: usize) -> Vec<LinearSample> {
    (0..target)
        .map(|index| {
            // OpenCV INTER_LINEAR / align_corners=false pixel-centre convention.
            let position = ((index as f64 + 0.5) * source as f64 / target as f64 - 0.5)
                .clamp(0.0, (source - 1) as f64);
            let lo = position.floor() as usize;
            LinearSample {
                lo,
                hi: (lo + 1).min(source - 1),
                fraction: (position - lo as f64) as f32,
            }
        })
        .collect()
}

fn box_sums(input: &[Weighted], width: usize, height: usize, kernel: usize) -> Vec<Weighted> {
    let radius = kernel / 2;
    let clamp = |index: isize, limit: usize| index.clamp(0, limit as isize - 1) as usize;
    let mut horizontal = vec![[0.0_f64; 4]; input.len()];
    for y in 0..height {
        let mut sum = [0.0_f64; 4];
        for offset in -(radius as isize)..=radius as isize {
            let value = input[y * width + clamp(offset, width)];
            for channel in 0..4 {
                sum[channel] += f64::from(value[channel]);
            }
        }
        for x in 0..width {
            horizontal[y * width + x] = sum;
            let remove = input[y * width + clamp(x as isize - radius as isize, width)];
            let add = input[y * width + clamp(x as isize + radius as isize + 1, width)];
            for channel in 0..4 {
                sum[channel] += f64::from(add[channel]) - f64::from(remove[channel]);
            }
        }
    }
    let mut output = vec![[0.0; 4]; input.len()];
    for x in 0..width {
        let mut sum = [0.0_f64; 4];
        for offset in -(radius as isize)..=radius as isize {
            let value = horizontal[clamp(offset, height) * width + x];
            for channel in 0..4 {
                sum[channel] += value[channel];
            }
        }
        for y in 0..height {
            output[y * width + x] = sum.map(|value| value as f32);
            let remove = horizontal[clamp(y as isize - radius as isize, height) * width + x];
            let add = horizontal[clamp(y as isize + radius as isize + 1, height) * width + x];
            for channel in 0..4 {
                sum[channel] += add[channel] - remove[channel];
            }
        }
    }
    output
}

fn spill_channel(mean: Rgb) -> Option<usize> {
    let excess = std::array::from_fn::<_, 3, _>(|channel| {
        mean[channel] - 0.5 * (mean[(channel + 1) % 3] + mean[(channel + 2) % 3])
    });
    let mut best = 0;
    for channel in 1..3 {
        if excess[channel] > excess[best] {
            best = channel;
        }
    }
    (excess[best] > 0.05).then_some(best)
}

struct Prepared {
    width: usize,
    height: usize,
    alpha: Vec<u8>,
    background: Vec<Rgb>,
    small_width: usize,
    x: Vec<LinearSample>,
    y: Vec<LinearSample>,
    channel: Option<usize>,
}
impl Prepared {
    fn new(plate: &[u8], result: &[u8], width: u32, height: u32) -> Result<Self> {
        let (width, height) = dimensions(width, height, result)?;
        ensure!(
            plate.len() == result.len(),
            "Original plate dimensions differ from the rendered result"
        );
        let scale = if width.max(height) > 1600 { 4 } else { 1 };
        let small_width = (width / scale).max(1);
        let small_height = (height / scale).max(1);
        let xs = area_axis(width, small_width);
        let ys = area_axis(height, small_height);
        let mut small_rgb = Vec::with_capacity(small_width * small_height);
        let mut weighted = Vec::with_capacity(small_width * small_height);
        for yw in &ys {
            for xw in &xs {
                let mut sample = [0.0_f64; 4];
                for &(y, wy) in yw {
                    for &(x, wx) in xw {
                        let offset = (y * width + x) * 4;
                        let weight = wx * wy;
                        for channel in 0..3 {
                            sample[channel] +=
                                f64::from(f32::from(plate[offset + channel]) / 255.0) * weight;
                        }
                        sample[3] +=
                            f64::from(f32::from(result[offset + 3]) * (1.0 / 255.0)) * weight;
                    }
                }
                let rgb = [sample[0] as f32, sample[1] as f32, sample[2] as f32];
                let weight = 1.0 - sample[3] as f32;
                small_rgb.push(rgb);
                weighted.push([rgb[0] * weight, rgb[1] * weight, rgb[2] * weight, weight]);
            }
        }
        let radius = ((width.min(height) / 40) | 1).max(9);
        let kernel = ((radius / scale) | 1).max(3);
        let sums = box_sums(&weighted, small_width, small_height, kernel);
        let background = sums
            .iter()
            .zip(small_rgb)
            .map(|(sum, fallback)| {
                if sum[3] < 1e-6 {
                    fallback
                } else {
                    let denominator = sum[3].max(1e-6);
                    [
                        sum[0] / denominator,
                        sum[1] / denominator,
                        sum[2] / denominator,
                    ]
                }
            })
            .collect();
        let mut prepared = Self {
            width,
            height,
            alpha: result
                .as_chunks::<4>()
                .0
                .iter()
                .map(|pixel| pixel[3])
                .collect(),
            background,
            small_width,
            x: linear_axis(small_width, width),
            y: linear_axis(small_height, height),
            channel: None,
        };
        let mut mean = [0.0_f64; 3];
        for y in 0..height {
            for x in 0..width {
                let bg = prepared.sample(x, y);
                for channel in 0..3 {
                    mean[channel] += f64::from(bg[channel]);
                }
            }
        }
        prepared.channel =
            spill_channel(mean.map(|value| (value / (width * height) as f64) as f32));
        Ok(prepared)
    }
    fn sample(&self, x: usize, y: usize) -> Rgb {
        let x = self.x[x];
        let y = self.y[y];
        let a = self.background[y.lo * self.small_width + x.lo];
        let b = self.background[y.lo * self.small_width + x.hi];
        let c = self.background[y.hi * self.small_width + x.lo];
        let d = self.background[y.hi * self.small_width + x.hi];
        std::array::from_fn(|channel| {
            let top = a[channel] + (b[channel] - a[channel]) * x.fraction;
            let bottom = c[channel] + (d[channel] - c[channel]) * x.fraction;
            top + (bottom - top) * y.fraction
        })
    }
    fn apply(&self, plate: &[u8], rgba: &mut [u8], amount: f32, weight: Option<&MaskWeight<'_>>) {
        for (index, (original, output)) in plate
            .as_chunks::<4>()
            .0
            .iter()
            .zip(rgba.as_chunks_mut::<4>().0.iter_mut())
            .enumerate()
        {
            let bg = self.sample(index % self.width, index / self.width);
            let alpha = f32::from(output[3]) * (1.0 / 255.0);
            let rgb: Rgb = std::array::from_fn(|channel| f32::from(original[channel]) / 255.0);
            let mut clean: Rgb = std::array::from_fn(|channel| {
                ((rgb[channel] - (1.0 - alpha) * bg[channel]) / alpha.max(1e-3)).clamp(0.0, 1.0)
            });
            if let Some(channel) = self.channel {
                let limit = 0.5 * (clean[(channel + 1) % 3] + clean[(channel + 2) % 3]);
                clean[channel] = clean[channel].min(limit);
            }
            let factor = match weight {
                Some(weight) => {
                    amount
                        * weight.sample(
                            index % self.width,
                            index / self.width,
                            self.width,
                            self.height,
                        )
                }
                None => amount,
            };
            for channel in 0..3 {
                output[channel] = ((rgb[channel] + (clean[channel] - rgb[channel]) * factor)
                    .clamp(0.0, 1.0)
                    * 255.0
                    + 0.5) as u8;
            }
        }
    }
}

/// One prepared frame only. The source Arc, dimensions and exact matte bytes
/// identify the recipe; amount changes reuse local background estimation.
#[derive(Default)]
pub struct DespillProcessor {
    cached: Option<(Arc<[u8]>, Prepared)>,
}
impl DespillProcessor {
    #[cfg(test)]
    pub fn apply(
        &mut self,
        plate: Arc<[u8]>,
        result: &mut image::RgbaImage,
        settings: DespillSettings,
    ) -> Result<bool> {
        self.apply_weighted(plate, result, settings, None)
    }

    /// Apply the colour unmix with an optional per-pixel mask weight.
    pub fn apply_weighted(
        &mut self,
        plate: Arc<[u8]>,
        result: &mut image::RgbaImage,
        settings: DespillSettings,
        weight: Option<&MaskWeight<'_>>,
    ) -> Result<bool> {
        settings.validate()?;
        if !settings.is_enabled() {
            return Ok(false);
        }
        let reusable = self.cached.as_ref().is_some_and(|(previous, prepared)| {
            Arc::ptr_eq(previous, &plate)
                && prepared.width == result.width() as usize
                && prepared.height == result.height() as usize
                && prepared
                    .alpha
                    .iter()
                    .copied()
                    .eq(result.pixels().map(|pixel| pixel[3]))
        });
        if !reusable {
            self.cached = None;
            let prepared = Prepared::new(&plate, result.as_raw(), result.width(), result.height())?;
            self.cached = Some((plate, prepared));
        }
        let (plate, prepared) = self.cached.as_ref().expect("prepared colour unmix");
        prepared.apply(plate, result.as_mut(), settings.amount, weight);
        Ok(true)
    }
}

#[cfg(test)]
pub fn apply_rgba(
    plate: &[u8],
    result: &mut [u8],
    width: u32,
    height: u32,
    settings: DespillSettings,
) -> Result<()> {
    apply_rgba_weighted(plate, result, width, height, settings, None)
}

/// v1 colour unmix with an optional per-pixel despill mask weight.
pub fn apply_rgba_weighted(
    plate: &[u8],
    result: &mut [u8],
    width: u32,
    height: u32,
    settings: DespillSettings,
    weight: Option<&MaskWeight<'_>>,
) -> Result<()> {
    settings.validate()?;
    if !settings.is_enabled() {
        return Ok(());
    }
    let prepared = Prepared::new(plate, result, width, height)?;
    prepared.apply(plate, result, settings.amount, weight);
    Ok(())
}

struct TransferTables {
    to_linear: [f32; 256],
    to_srgb: Vec<f32>,
    checker: [f32; 2],
}
fn tables() -> &'static TransferTables {
    static TABLES: OnceLock<TransferTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let linear = |value: f64| {
            if value <= 0.04045 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        };
        TransferTables {
            to_linear: std::array::from_fn(|index| linear(index as f64 / 255.0) as f32),
            to_srgb: (0..65536)
                .map(|index| {
                    let value = index as f64 / 65535.0;
                    (if value <= 0.0031308 {
                        value * 12.92
                    } else {
                        1.055 * value.powf(1.0 / 2.4) - 0.055
                    }) as f32
                })
                .collect(),
            checker: [linear(0.75) as f32, linear(0.5) as f32],
        }
    })
}

/// v1 RESULT checker, distinct from its dark 14px viewport transparency tile:
/// 64 source pixels, 75%/50% display gray, composited in linear light.
pub fn composite_checker(image: &mut image::RgbaImage) {
    let table = tables();
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if pixel[3] == 255 {
            continue;
        }
        let alpha = f32::from(pixel[3]) * (1.0 / 255.0);
        let background = table.checker[((x / 64 + y / 64) % 2) as usize];
        for channel in 0..3 {
            let linear =
                table.to_linear[pixel[channel] as usize] * alpha + background * (1.0 - alpha);
            let index = (linear.clamp(0.0, 1.0) * 65535.0 + 0.5) as usize;
            pixel[channel] = (table.to_srgb[index] * 255.0 + 0.5) as u8;
        }
        pixel[3] = 255;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(width: u32, height: u32) -> (Arc<[u8]>, image::RgbaImage) {
        let mut plate = Vec::new();
        let mut result = image::RgbaImage::new(width, height);
        for (x, y, pixel) in result.enumerate_pixels_mut() {
            plate.extend([
                ((x * 13 + y * 11) % 256) as u8,
                ((90 + x * 17 + y * 23) % 256) as u8,
                ((x * 7 + y * 29) % 256) as u8,
                255,
            ]);
            *pixel = image::Rgba([19, 37, 53, ((x * 19 + y * 31) % 256) as u8]);
        }
        (plate.into(), result)
    }

    #[test]
    fn disabled_adjustment_is_byte_identical_without_an_original_plate() {
        let (_, mut image) = fixture(9, 7);
        let original = image.clone();
        apply_rgba(&[], image.as_mut(), 9, 7, DespillSettings::default()).unwrap();
        assert_eq!(image, original);
        assert!(
            !DespillProcessor::default()
                .apply(Arc::from([]), &mut image, DespillSettings::default())
                .unwrap()
        );
        assert_eq!(image, original);
    }

    #[test]
    fn screen_channel_is_automatic_for_green_blue_red_and_neutral() {
        assert_eq!(spill_channel([0.2, 0.8, 0.3]), Some(1));
        assert_eq!(spill_channel([0.2, 0.3, 0.8]), Some(2));
        assert_eq!(spill_channel([0.8, 0.2, 0.3]), Some(0));
        assert_eq!(spill_channel([0.5, 0.5, 0.5]), None);
        assert_eq!(spill_channel([0.5, 0.54, 0.5]), None);
    }

    #[test]
    fn replicated_box_unmix_matches_unchanged_v1_numpy_opencv_oracle() {
        // Generated by the unchanged v1 color_unmix/_rgba_numpy CPU functions,
        // with OpenCV boxFilter(BORDER_REPLICATE); no application/torch loaded.
        let expected: Vec<[u8; 4]> = serde_json::from_str(r#"[[0,57,0,0],[8,67,4,19],[16,78,9,38],[36,95,13,57],[52,109,18,76],[65,123,23,95],[78,139,33,114],[92,155,42,133],[105,170,51,152],[7,71,18,31],[17,87,30,50],[38,107,41,69],[52,122,48,88],[65,136,55,107],[77,150,61,126],[90,164,68,145],[103,179,75,164],[116,193,83,183],[14,101,66,62],[35,119,73,81],[50,133,78,100],[63,146,82,119],[75,159,87,138],[88,173,93,157],[100,187,99,176],[113,201,106,195],[126,10,113,214],[32,128,100,93],[47,143,104,112],[61,155,108,131],[73,168,112,150],[86,181,117,169],[98,195,122,188],[111,3,129,207],[124,17,136,226],[137,38,143,245],[44,150,128,124],[58,163,131,143],[71,176,135,162],[84,189,140,181],[96,203,145,200],[109,7,151,219],[122,26,158,238],[179,28,142,1],[188,39,166,20],[55,170,153,155],[69,183,157,174],[81,197,162,193],[94,0,167,212],[107,13,174,231],[120,33,180,250],[153,32,212,13],[166,43,213,32],[181,69,214,51],[66,191,179,186],[79,204,184,205],[92,4,190,224],[105,21,195,243],[147,25,222,6],[139,36,226,25],[155,47,230,44],[172,95,235,63],[187,125,239,82]]"#).unwrap();
        let (plate, mut image) = fixture(9, 7);
        let alpha: Vec<_> = image.pixels().map(|pixel| pixel[3]).collect();
        apply_rgba(
            &plate,
            image.as_mut(),
            9,
            7,
            DespillSettings { amount: 0.37 },
        )
        .unwrap();
        for (actual, expected) in image.pixels().zip(expected) {
            assert_eq!(actual.0, expected);
        }
        assert!(image.pixels().map(|pixel| pixel[3]).eq(alpha));
    }

    #[test]
    fn quarter_size_noninteger_area_and_bilinear_upscale_match_v1_oracle() {
        let expected: Vec<[u32; 6]> = serde_json::from_str(r#"[[0,0,0,57,0,0],[1,0,8,67,4,19],[399,0,58,231,241,157],[400,0,73,242,246,176],[800,0,168,121,235,96],[1200,0,246,6,225,16],[1601,0,76,173,204,211],[1602,0,90,190,209,230],[0,4,31,201,116,124],[1,4,51,218,125,143],[399,4,103,33,59,25],[400,4,123,44,63,44],[800,4,207,219,81,220],[1200,4,18,96,49,140],[1601,4,138,4,37,79],[1602,4,154,15,42,98],[0,8,88,17,233,248],[1,8,158,22,245,11],[399,8,165,152,226,149],[400,8,175,170,230,168],[800,8,251,32,220,88],[1200,8,45,217,210,8],[1601,8,171,96,180,203],[1602,8,182,115,186,222]]"#).unwrap();
        let (plate, mut image) = fixture(1603, 9);
        apply_rgba(
            &plate,
            image.as_mut(),
            1603,
            9,
            DespillSettings { amount: 0.37 },
        )
        .unwrap();
        for [x, y, r, g, b, alpha] in expected {
            assert_eq!(
                image.get_pixel(x, y).0,
                [r as u8, g as u8, b as u8, alpha as u8],
                "v1 area/linear oracle at {x},{y}"
            );
        }
    }

    #[test]
    fn strength_changes_reuse_prepared_background_and_alpha_changes_invalidate_it() {
        let (plate, image) = fixture(9, 7);
        let mut processor = DespillProcessor::default();
        let mut first = image.clone();
        processor
            .apply(plate.clone(), &mut first, DespillSettings { amount: 0.25 })
            .unwrap();
        let background = processor.cached.as_ref().unwrap().1.background.as_ptr();
        let mut changed = image.clone();
        processor
            .apply(
                plate.clone(),
                &mut changed,
                DespillSettings { amount: 0.75 },
            )
            .unwrap();
        assert_eq!(
            processor.cached.as_ref().unwrap().1.background.as_ptr(),
            background
        );
        let mut expected = image.clone();
        apply_rgba(
            &plate,
            expected.as_mut(),
            9,
            7,
            DespillSettings { amount: 0.75 },
        )
        .unwrap();
        assert_eq!(changed, expected);
        changed = image;
        changed.get_pixel_mut(0, 0)[3] = 99;
        processor
            .apply(plate, &mut changed, DespillSettings { amount: 1.0 })
            .unwrap();
        assert_eq!(processor.cached.as_ref().unwrap().1.alpha[0], 99);
        assert_eq!(changed.get_pixel(0, 0)[3], 99);
    }

    #[test]
    fn result_checker_half_alpha_matches_independent_v1_linear_lut_oracle() {
        let values = [
            [200, 100, 50, 128],
            [255, 0, 0, 128],
            [10, 80, 240, 0],
            [255, 255, 255, 0],
            [0, 0, 0, 255],
            [90, 140, 201, 85],
        ];
        let expected = [
            [
                [196, 154, 143],
                [226, 140, 140],
                [191, 191, 191],
                [191, 191, 191],
                [0, 0, 0],
                [166, 176, 195],
            ],
            [
                [169, 115, 98],
                [205, 92, 92],
                [127, 127, 127],
                [127, 127, 127],
                [0, 0, 0],
                [117, 132, 157],
            ],
        ];
        let mut image = image::RgbaImage::new(70, 1);
        for start in [0, 64] {
            for (index, value) in values.iter().enumerate() {
                *image.get_pixel_mut(start + index as u32, 0) = image::Rgba(*value);
            }
        }
        composite_checker(&mut image);
        for (tile, start) in [0, 64].into_iter().enumerate() {
            for (index, rgb) in expected[tile].iter().enumerate() {
                assert_eq!(
                    image.get_pixel(start + index as u32, 0).0,
                    [rgb[0], rgb[1], rgb[2], 255]
                );
            }
        }
    }

    #[test]
    fn mask_weight_scales_the_unmix_per_pixel() {
        let (plate, image) = fixture(9, 7);
        let mut weight_data = vec![255_u8; 9 * 7];
        for y in 0..7 {
            for x in 0..4 {
                weight_data[y * 9 + x] = 0;
            }
        }
        let gray = image::GrayImage::from_raw(9, 7, weight_data).unwrap();
        let weight = MaskWeight::from_gray(&gray);
        let mut weighted = image.clone();
        DespillProcessor::default()
            .apply_weighted(
                plate.clone(),
                &mut weighted,
                DespillSettings { amount: 1.0 },
                Some(&weight),
            )
            .unwrap();
        let mut full = image.clone();
        apply_rgba(&plate, full.as_mut(), 9, 7, DespillSettings { amount: 1.0 }).unwrap();
        for y in 0..7 {
            for x in 0..4 {
                // v1 color_unmix bases the blend on the plate RGB, so a zero
                // weight keeps the original plate colour (alpha is untouched).
                let offset = ((y * 9 + x) * 4) as usize;
                assert_eq!(
                    weighted.get_pixel(x, y).0[..3],
                    plate[offset..offset + 3],
                    "protected pixel {x},{y} must keep the plate colour"
                );
                assert_eq!(weighted.get_pixel(x, y)[3], image.get_pixel(x, y)[3]);
            }
            for x in 5..9 {
                assert_eq!(weighted.get_pixel(x, y), full.get_pixel(x, y));
            }
        }
    }
}
