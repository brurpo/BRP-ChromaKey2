//! Developer-only scalar/tensor contract probe; never part of app packaging.
use ck_core::color;
use ck_core::spatial::{
    resize_area_chw, resize_area_chw_cuda_order, resize_bilinear_chw,
    resize_bilinear_chw_cuda_order, Padding,
};
use ck_core::windows::{taper_weights, BatonAccumulator, WindowPlan};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.first().is_some_and(|value| {
        matches!(
            value.as_str(),
            "ck-bilinear-cuda" | "ck-area-cuda" | "ck-area-cpu"
        )
    }) {
        if arguments.len() != 8 {
            return Err(
                "usage: ck-contract-probe ck-bilinear-cuda|ck-area-cuda INPUT_F32 C H W OUT_H OUT_W OUTPUT_F32"
                    .into(),
            );
        }
        let packed = std::fs::read(&arguments[1])?;
        let (values, remainder) = packed.as_chunks::<4>();
        if !remainder.is_empty() {
            return Err("partial float32 input".into());
        }
        let input: Vec<_> = values
            .iter()
            .map(|value| f32::from_le_bytes(*value))
            .collect();
        let resize = match arguments[0].as_str() {
            "ck-area-cuda" => resize_area_chw_cuda_order,
            "ck-area-cpu" => resize_area_chw,
            _ => resize_bilinear_chw_cuda_order,
        };
        let output = resize(
            &input,
            arguments[2].parse()?,
            arguments[3].parse()?,
            arguments[4].parse()?,
            arguments[5].parse()?,
            arguments[6].parse()?,
        )?;
        use std::io::Write;
        let mut file = std::fs::File::create(&arguments[7])?;
        for values in output.chunks(65536) {
            let bytes: Vec<_> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            file.write_all(&bytes)?;
        }
        return Ok(());
    }
    if arguments
        .first()
        .is_some_and(|value| value == "ck-frame" || value == "ck-foundation")
    {
        let foundation = arguments[0] == "ck-foundation";
        if arguments.len() != if foundation { 8 } else { 5 } {
            return Err("usage: ck-contract-probe ck-frame INPUT_RGB24 H W OUTPUT_F32; or ck-foundation INPUT_RGB24 H W area|bilinear OUT_H OUT_W OUTPUT_F32".into());
        }
        let height: usize = arguments[2].parse()?;
        let width: usize = arguments[3].parse()?;
        let packed = std::fs::read(&arguments[1])?;
        if packed.len() != height * width * 3 {
            return Err("RGB24 frame byte count mismatch".into());
        }
        let mut planar = vec![0.0; packed.len()];
        for (index, rgb) in packed.as_chunks::<3>().0.iter().enumerate() {
            for channel in 0..3 {
                planar[channel * height * width + index] = f32::from(rgb[channel]) / 255.0;
            }
        }
        let padding = Padding::new(height, width, 128)?;
        let padded = padding.pad_chw(&planar, 3)?;
        let output = if foundation {
            let out_height: usize = arguments[5].parse()?;
            let out_width: usize = arguments[6].parse()?;
            match arguments[4].as_str() {
                "area" => resize_area_chw(
                    &padded,
                    3,
                    padding.height,
                    padding.width,
                    out_height,
                    out_width,
                )?,
                "bilinear" => resize_bilinear_chw(
                    &padded,
                    3,
                    padding.height,
                    padding.width,
                    out_height,
                    out_width,
                )?,
                _ => return Err("foundation resize mode must be area or bilinear".into()),
            }
        } else {
            padded
                .into_iter()
                .map(|value| color::linear_to_asinh(color::srgb_to_linear(value)))
                .collect()
        };
        use std::io::Write;
        let mut file = std::fs::File::create(if foundation {
            &arguments[7]
        } else {
            &arguments[4]
        })?;
        for values in output.chunks(65536) {
            let bytes: Vec<_> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            file.write_all(&bytes)?;
        }
        return Ok(());
    }
    if arguments.first().is_some_and(|value| value == "aa") {
        if arguments.len() != 7 {
            return Err(
                "usage: ck-contract-probe aa INPUT_RGB24 H W OUT_H OUT_W OUTPUT_F32".into(),
            );
        }
        let height: usize = arguments[2].parse()?;
        let width: usize = arguments[3].parse()?;
        let out_height: usize = arguments[4].parse()?;
        let out_width: usize = arguments[5].parse()?;
        let packed = std::fs::read(&arguments[1])?;
        if packed.len() != height * width * 3 {
            return Err("RGB24 frame byte count mismatch".into());
        }
        let mut planar = vec![0.0; packed.len()];
        for (index, rgb) in packed.as_chunks::<3>().0.iter().enumerate() {
            for channel in 0..3 {
                planar[channel * height * width + index] = f32::from(rgb[channel]) / 255.0;
            }
        }
        let output = ck_core::spatial::resize_bilinear_antialias_chw(
            &planar, 3, height, width, out_height, out_width,
        )?;
        use std::io::Write;
        let mut file = std::fs::File::create(&arguments[6])?;
        let bytes: Vec<_> = output.into_iter().flat_map(f32::to_le_bytes).collect();
        file.write_all(&bytes)?;
        return Ok(());
    }
    let colors: [f32; 12] = [
        -32.0, -1.0, -0.04045, -0.0031308, 0.0, 0.0031308, 0.04045, 0.18, 0.5, 1.0, 16.0, 32.0,
    ];
    let plan = WindowPlan::new(55, 24, 12)?;
    let starts: Vec<_> = plan.windows.iter().map(|w| w.start).collect();
    let mut accumulator = BatonAccumulator::new(2)?;
    let mut blended = Vec::new();
    for window in &plan.windows {
        let values: Vec<_> = (window.start..window.end)
            .flat_map(|frame| {
                [
                    frame as f32 + window.index as f32 * 0.1,
                    -(frame as f32) + window.index as f32 * 0.2,
                ]
            })
            .collect();
        for frame in accumulator.push_window(window, &values)? {
            blended.extend(frame.values);
        }
    }
    let area_input: Vec<_> = (0..35).map(|x| x as f32 * 0.1).collect();
    let area = resize_area_chw(&area_input, 1, 5, 7, 3, 4)?;
    let bilinear = resize_bilinear_chw(&area_input, 1, 5, 7, 8, 11)?;
    let padded = Padding::new(2, 3, 8)?.pad_chw(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 1)?;
    println!("{{\"colors\":{:?},\"srgb_to_linear\":{:?},\"linear_to_srgb\":{:?},\"linear_to_asinh\":{:?},\"window_starts\":{:?},\"taper_first\":{:?},\"taper_middle\":{:?},\"taper_last\":{:?},\"short_taper\":{:?},\"blended\":{:?},\"area\":{:?},\"bilinear\":{:?},\"padded\":{:?}}}",
        colors, colors.map(color::srgb_to_linear), colors.map(color::linear_to_srgb), colors.map(color::linear_to_asinh), starts,
        taper_weights(24, 12, true, false), taper_weights(24, 12, false, false), taper_weights(24, 12, false, true), taper_weights(3, 4, false, false), blended, area, bilinear, padded);
    Ok(())
}
