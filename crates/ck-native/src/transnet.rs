//! Native TransNetV2 window inference and BRP scene conventions.
use super::npy::write_npy;
use super::{
    Args, Metrics, TensorSpec, Tolerance, compare, create_session, initialize_runtime, load_tensor,
    pack_path, profile_counts,
};
use anyhow::{Context, Result, ensure};
use ck_core::scenes::{filter_cuts, merge_ranges, transition_cuts, transnet_window_indices};
use ndarray::{ArrayD, IxDyn};
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::Path,
    process::Command,
    time::Instant,
};

const FRAME_BYTES: usize = 27 * 48 * 3;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    kind: String,
    model: String,
    model_sha256: String,
    source_video: Source,
    source_rgb: String,
    source_rgb_sha256: String,
    frame_count: usize,
    fps: [u64; 2],
    source_timing: Timing,
    threshold: f64,
    min_length: usize,
    window: usize,
    stride: usize,
    context: usize,
    outputs: Vec<TensorSpec>,
    cuts: Vec<usize>,
    scenes: Vec<ExpectedScene>,
    tolerance: Tolerance,
}
#[derive(Deserialize)]
struct Source {
    file: String,
    sha256: String,
}
#[derive(Deserialize, Serialize)]
struct Timing {
    mode: String,
    first_pts: i64,
    time_base: [u64; 2],
    has_b_frames: u64,
}
#[derive(Deserialize, Serialize, PartialEq, Eq)]
struct ExpectedScene {
    index: usize,
    start: usize,
    end: usize,
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn validate_cfr_pts(pts: &[i64], fps: [u64; 2], time_base: [u64; 2]) -> Result<()> {
    ensure!(
        !pts.is_empty()
            && fps
                .iter()
                .chain(time_base.iter())
                .all(|&value| value > 0 && value <= u64::from(u32::MAX)),
        "invalid or unsupported video timing scale"
    );
    for pair in pts.windows(2) {
        let delta = pair[1]
            .checked_sub(pair[0])
            .context("PTS difference overflow")?;
        ensure!(
            delta > 0
                && delta as u128 * time_base[0] as u128 * fps[0] as u128
                    == fps[1] as u128 * time_base[1] as u128,
            "VFR or irregular PTS is unsupported; decode CFR explicitly before scene detection"
        );
    }
    Ok(())
}

fn probe_timing(ffmpeg: &Path, video: &Path, fixture: &Fixture) -> Result<()> {
    probe_source_timing(
        ffmpeg,
        video,
        fixture.frame_count,
        fixture.fps,
        &fixture.source_timing,
    )
}

fn probe_source_timing(
    ffmpeg: &Path,
    video: &Path,
    frame_count: usize,
    fps: [u64; 2],
    timing: &Timing,
) -> Result<()> {
    let ffprobe = ffmpeg.with_file_name(if cfg!(windows) {
        "ffprobe.exe"
    } else {
        "ffprobe"
    });
    let mut command = Command::new(ffprobe);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let output = command
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_streams",
            "-show_entries",
            "stream=time_base,has_b_frames:packet=pts",
            "-of",
            "json",
        ])
        .arg(video)
        .output()?;
    ensure!(output.status.success(), "native ffprobe timing failed");
    let data: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let time_base = data["streams"][0]["time_base"]
        .as_str()
        .context("missing video time base")?;
    ensure!(
        time_base == format!("{}/{}", timing.time_base[0], timing.time_base[1]),
        "video time base mismatch"
    );
    let mut pts: Vec<i64> = data["packets"]
        .as_array()
        .context("missing packet PTS")?
        .iter()
        .map(|packet| {
            packet["pts"]
                .as_i64()
                .context("missing packet PTS; unsupported timing")
        })
        .collect::<Result<_>>()?;
    // Encoded packet order may be DTS order; PTS sorting recovers display order.
    pts.sort_unstable();
    ensure!(
        pts.len() == frame_count && pts[0] == timing.first_pts,
        "source timing/frame-count mismatch"
    );
    validate_cfr_pts(&pts, fps, timing.time_base)
}

pub(crate) fn desktop_detect(
    job: &ck_desktop_core::JobRequest,
    threshold: f64,
    min_scene_frames: u64,
) -> Result<ck_desktop_core::JobArtifacts> {
    use ck_desktop_core::{JobArtifacts, Provider, SceneRange};
    let count = usize::try_from(job.media.frame_count)?;
    ensure!(count > 0, "source has no video frames");
    let model = &job.runtime.transnet_model;
    ensure!(
        model.is_file(),
        "TransNet model is missing: {}",
        model.display()
    );
    let timing = Timing {
        mode: "cfr".into(),
        first_pts: job.media.first_pts,
        time_base: [job.media.time_base.num, job.media.time_base.den],
        has_b_frames: u64::from(job.media.has_b_frames),
    };
    let fps = [job.media.fps.num, job.media.fps.den];
    probe_source_timing(&job.runtime.ffmpeg, &job.media.path, count, fps, &timing)?;
    let rgb = job.output_dir.join("transnet.rgb24");
    super::desktop::progress("decode_scenes", 0, 1)?;
    decode_small(&job.runtime.ffmpeg, &job.media.path, &rgb, count)?;
    ensure!(
        fs::metadata(&rgb)?.len() == (count * FRAME_BYTES) as u64,
        "TransNet decoded frame count mismatch"
    );
    super::desktop::progress("decode_scenes", 1, 1)?;
    let args = super::desktop::worker_args(
        job,
        job.output_dir.join("inference-report.json"),
        model.parent().context("model parent missing")?.to_owned(),
        Provider::Cpu,
    );
    initialize_runtime(&args)?;
    let mut probabilities = Vec::with_capacity(count);
    if count > 1 {
        let mut source = File::open(&rgb)?;
        let mut session = create_session(&args, model, None)?;
        for (window, start) in (0..count).step_by(50).enumerate() {
            let input = Tensor::from_array((
                [1usize, 100, 27, 48, 3],
                window_input(&mut source, count, start)?,
            ))?;
            let output = session.run(ort::inputs!["rgb"=>input])?;
            let (shape, values) = output["single_probabilities"].try_extract_tensor::<f32>()?;
            ensure!(
                shape.as_ref() == [1, 100, 1]
                    && values
                        .iter()
                        .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                "invalid TransNet probabilities"
            );
            probabilities.extend_from_slice(&values[25..25 + (count - start).min(50)]);
            super::desktop::progress("transnet", window + 1, count.div_ceil(50))?;
        }
    }
    let raw = transition_cuts(&probabilities, threshold)?;
    let cuts = filter_cuts(&raw, usize::try_from(min_scene_frames)?, count - 1);
    let scenes: Vec<_> = merge_ranges(&cuts, count - 1)
        .into_iter()
        .map(|s| SceneRange {
            index: s.index as u32,
            start: s.start as u64,
            end_exclusive: s.end as u64 + 1,
            enabled: true,
        })
        .collect();
    let report = job.output_dir.join("inference-report.json");
    fs::write(
        &report,
        serde_json::to_vec_pretty(
            &serde_json::json!({"schema_version":1,"component":"transnet_desktop_inference","execution_success":true,"validation_performed":false,"numerical_passed":null,"source":job.media,"model_sha256":sha256_file(model)?,"cuts":cuts,"scenes":scenes,"threshold":threshold,"minimum_scene_frames":min_scene_frames,"window":100,"stride":50,"context":25,"frame_index_base":0,"scene_bounds":"half_open"}),
        )?,
    )?;
    Ok(JobArtifacts {
        scenes,
        report: Some(report),
        ..Default::default()
    })
}

fn decode_small(ffmpeg: &Path, source: &Path, destination: &Path, count: usize) -> Result<()> {
    let mut command = Command::new(ffmpeg);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let status = command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i"])
        .arg(source)
        .args([
            "-vf",
            "scale=48:27:flags=bilinear,setsar=1,format=rgb24",
            "-an",
            "-sn",
            "-dn",
            "-frames:v",
        ])
        .arg(count.to_string())
        .args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
        .arg(destination)
        .status()?;
    ensure!(status.success(), "TransNet native FFmpeg decode failed");
    Ok(())
}

fn window_input(file: &mut File, count: usize, output_start: usize) -> Result<Vec<f32>> {
    let indices = transnet_window_indices(count, output_start)?;
    let mut input = Vec::with_capacity(100 * FRAME_BYTES);
    let mut frame = vec![0u8; FRAME_BYTES];
    for index in indices {
        file.seek(SeekFrom::Start((index as u64) * FRAME_BYTES as u64))?;
        file.read_exact(&mut frame)?;
        input.extend(frame.iter().map(|&value| f32::from(value)));
    }
    Ok(input)
}

pub(crate) fn run(args: &Args) -> Result<()> {
    let started = Instant::now();
    let root = args.manifest.parent().unwrap_or(Path::new("."));
    let fixture: Fixture = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    ensure!(
        fixture.schema_version == 1 && fixture.kind == "transnet_video_fixture",
        "unsupported TransNet fixture"
    );
    ensure!(
        fixture.window == 100 && fixture.stride == 50 && fixture.context == 25,
        "unsupported TransNet window profile"
    );
    ensure!(
        fixture.frame_count > 0 && fixture.frame_count.checked_mul(FRAME_BYTES).is_some(),
        "invalid TransNet frame count"
    );
    ensure!(
        fixture.fps[0] > 0 && fixture.fps[1] > 0 && fixture.threshold.is_finite(),
        "invalid fps/threshold"
    );
    ensure!(
        fixture.source_timing.mode == "cfr"
            || (fixture.source_timing.mode == "single_frame" && fixture.frame_count == 1),
        "only validated CFR source timing is supported"
    );
    let tolerance = Tolerance {
        atol: args.atol.unwrap_or(fixture.tolerance.atol),
        rtol: args.rtol.unwrap_or(fixture.tolerance.rtol),
    };
    ensure!(
        tolerance.atol.is_finite()
            && tolerance.atol >= 0.0
            && tolerance.rtol.is_finite()
            && tolerance.rtol >= 0.0,
        "invalid tolerance"
    );
    let output = args.output.clone().unwrap_or_else(|| root.join("native"));
    fs::create_dir_all(&output)?;
    let model = pack_path(root, &fixture.model)?;
    ensure!(
        sha256_file(&model)? == fixture.model_sha256,
        "TransNet model SHA256 mismatch"
    );
    let (rgb_path, input_mode) = if let Some(video) = &args.video {
        ensure!(
            sha256_file(video)? == fixture.source_video.sha256,
            "supplied video does not match TransNet fixture source"
        );
        let rgb_path = output.join("decoded-small.rgb24");
        probe_timing(
            args.ffmpeg
                .as_ref()
                .context("--ffmpeg required for --video")?,
            video,
            &fixture,
        )?;
        decode_small(
            args.ffmpeg
                .as_ref()
                .context("--ffmpeg required for --video")?,
            video,
            &rgb_path,
            fixture.frame_count,
        )?;
        (rgb_path, "native_ffmpeg_bilinear_rgb24")
    } else {
        (
            pack_path(root, &fixture.source_rgb)?,
            "prepared_rgb24_fixture",
        )
    };
    let rgb_hash = sha256_file(&rgb_path)?;
    ensure!(
        rgb_hash == fixture.source_rgb_sha256,
        "TransNet decoded RGB SHA256 differs from original app reference"
    );
    ensure!(
        fs::metadata(&rgb_path)?.len() == (fixture.frame_count * FRAME_BYTES) as u64,
        "TransNet RGB frame-count mismatch"
    );
    let mut file = File::open(&rgb_path)?;
    initialize_runtime(args)?;
    let load_start = Instant::now();
    let mut session = create_session(args, &model, args.profile.as_deref())?;
    let model_load_seconds = load_start.elapsed().as_secs_f64();
    let mut single = Vec::with_capacity(fixture.frame_count);
    let mut many = Vec::with_capacity(fixture.frame_count);
    let mut inference_seconds = 0.0;
    let mut calls = 0;
    if fixture.frame_count > 1 {
        for output_start in (0..fixture.frame_count).step_by(50) {
            let input = window_input(&mut file, fixture.frame_count, output_start)?;
            let input = Tensor::from_array(([1usize, 100, 27, 48, 3], input))?;
            let inference_start = Instant::now();
            let values = session.run(ort::inputs!["rgb"=>input])?;
            inference_seconds += inference_start.elapsed().as_secs_f64();
            let count = (fixture.frame_count - output_start).min(50);
            for (name, target) in [
                ("single_probabilities", &mut single),
                ("many_probabilities", &mut many),
            ] {
                let (shape, data) = values[name].try_extract_tensor::<f32>()?;
                ensure!(
                    shape.as_ref() == [1, 100, 1],
                    "invalid TransNet output geometry"
                );
                ensure!(
                    data.iter()
                        .all(|&value| value.is_finite() && (0.0..=1.0).contains(&value)),
                    "invalid TransNet probability"
                );
                target.extend_from_slice(&data[25..25 + count]);
            }
            calls += 1;
        }
    }
    let raw_cuts = transition_cuts(&single, fixture.threshold)?;
    let cuts = filter_cuts(&raw_cuts, fixture.min_length, fixture.frame_count - 1);
    let scenes: Vec<ExpectedScene> = merge_ranges(&cuts, fixture.frame_count - 1)
        .into_iter()
        .map(|scene| ExpectedScene {
            index: scene.index,
            start: scene.start,
            end: scene.end,
        })
        .collect();
    let expected_length = if fixture.frame_count == 1 {
        0
    } else {
        fixture.frame_count
    };
    ensure!(
        single.len() == expected_length && many.len() == expected_length,
        "TransNet stitch coverage mismatch"
    );
    let mut metrics = Vec::new();
    ensure!(
        fixture.outputs.len() == 2,
        "TransNet fixture must cover both output heads"
    );
    for (name, values) in [
        ("single_probabilities", &single),
        ("many_probabilities", &many),
    ] {
        let spec = fixture
            .outputs
            .iter()
            .find(|spec| spec.name == name)
            .context("missing probability reference")?;
        let actual = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values.clone())?;
        let reference = load_tensor(root, spec)?;
        if actual.is_empty() {
            ensure!(
                reference.shape() == [0],
                "one-frame bypass must have empty original probabilities"
            );
            metrics.push(Metrics {
                name: name.into(),
                shape: vec![0],
                elements: 0,
                max_abs_error: 0.0,
                mean_abs_error: 0.0,
                root_mean_square_error: 0.0,
                failed_elements: 0,
                passed: true,
            });
        } else {
            metrics.push(compare(name, &actual, &reference, tolerance)?);
        }
        write_npy(output.join(format!("{name}.npy")), &actual)?;
    }
    let cuts_match = cuts == fixture.cuts;
    let scenes_match = scenes == fixture.scenes;
    let passed = metrics.iter().all(|metric| metric.passed) && cuts_match && scenes_match;
    let fps = fixture.fps[0] as f64 / fixture.fps[1] as f64;
    let first_pts_seconds = fixture.source_timing.first_pts as f64
        * fixture.source_timing.time_base[0] as f64
        / fixture.source_timing.time_base[1] as f64;
    let mut scene_times = Vec::new();
    for scene in &scenes {
        let start_num = (scene.start as u64)
            .checked_mul(fixture.fps[1])
            .context("scene timestamp overflow")?;
        let end_num = ((scene.end + 1) as u64)
            .checked_mul(fixture.fps[1])
            .context("scene timestamp overflow")?;
        scene_times.push(serde_json::json!({"index":scene.index,"start":scene.start,"end":scene.end,
            "start_seconds":scene.start as f64/fps,"end_frame_seconds":scene.end as f64/fps,
            "end_exclusive_seconds":(scene.end+1) as f64/fps,"half_open_export_frames":[scene.start,scene.end+1],
            "export_start_time_rational":[start_num,fixture.fps[0]],"export_end_time_rational":[end_num,fixture.fps[0]],
            "source_start_seconds":first_pts_seconds+scene.start as f64/fps,
            "source_end_exclusive_seconds":first_pts_seconds+(scene.end+1) as f64/fps}));
    }
    let profile = if args.profile.is_some() {
        Some(session.end_profiling()?)
    } else {
        None
    };
    let providers = if let Some(path) = &profile {
        profile_counts(path)?
    } else {
        BTreeMap::new()
    };
    let report = serde_json::json!({"schema_version":1,"component":"transnetv2_native_scene_detection","source":fixture.source_video.file,
        "source_sha256":fixture.source_video.sha256,"decoded_rgb_sha256":rgb_hash,"input_mode":input_mode,"model_sha256":fixture.model_sha256,
        "frame_count":fixture.frame_count,"fps":fixture.fps,"source_timing":fixture.source_timing,"timing_validated_from_native_packets":args.video.is_some(),"threshold":fixture.threshold,"min_length":fixture.min_length,
        "window":100,"stride":50,"context":25,"retained_window_range":[25,75],"model_load_seconds":model_load_seconds,
        "inference_seconds":inference_seconds,"calls":calls,"elapsed_seconds":started.elapsed().as_secs_f64(),"requested_provider":args.provider,
        "ort_optimization":args.ort_optimization,"outputs":metrics,"raw_cuts":raw_cuts,"cuts":cuts,"scenes":scene_times,
        "cuts_match":cuts_match,"scenes_match":scenes_match,"atol":tolerance.atol,"rtol":tolerance.rtol,"profile":profile,
        "provider_node_counts":providers,"passed":passed,"scope":"Native video decode, original BRP TransNet windowing and inclusive scene postprocessing. No adaptive/ffmpeg detector fallback or UI integration."});
    let path = args
        .report
        .clone()
        .unwrap_or_else(|| output.join("report.json"));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "TransNet frames={}, windows={calls}, cuts={:?}, pass={passed}, report={}",
        fixture.frame_count,
        cuts,
        path.display()
    );
    ensure!(passed, "TransNet native/reference parity failed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cfr_pts_preserves_offset_and_rejects_vfr_or_duplicate_frames() {
        assert!(validate_cfr_pts(&[0, 1001, 2002], [30000, 1001], [1, 30000]).is_ok());
        assert!(validate_cfr_pts(&[30030, 31031, 32032], [30000, 1001], [1, 30000]).is_ok());
        assert!(validate_cfr_pts(&[0, 1001, 2502], [30000, 1001], [1, 30000]).is_err());
        assert!(validate_cfr_pts(&[0, 1001, 1001], [30000, 1001], [1, 30000]).is_err());
    }
}
