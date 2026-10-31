//! Native desktop worker. Jobs contain footage/model paths and user prompts,
//! never golden tensors. The existing validation commands remain independent.
use super::{Args, npy::write_npy, pack_path};
use anyhow::{Context, Result, bail, ensure};
use ck_desktop_core::{HintInput, JobArtifacts, JobEvent, JobKind, JobRequest, Provider, SamModel};
use image::{ColorType, ImageFormat};
use ndarray::{ArrayD, IxDyn};
use serde_json::json;
use std::{
    collections::BTreeMap,
    env,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::{Mutex, OnceLock},
};

struct WorkerContext {
    job_id: String,
    events: File,
    output: PathBuf,
    start: usize,
    hints: HintInput,
}
static WORKER: OnceLock<Mutex<WorkerContext>> = OnceLock::new();

fn emit(event: JobEvent) -> Result<()> {
    let Some(context) = WORKER.get() else {
        return Ok(());
    };
    let mut context = context
        .lock()
        .map_err(|_| anyhow::anyhow!("worker event lock poisoned"))?;
    serde_json::to_writer(&mut context.events, &event)?;
    context.events.write_all(b"\n")?;
    context.events.flush()?;
    Ok(())
}

pub(crate) fn progress(phase: &str, current: usize, total: usize) -> Result<()> {
    progress_message(phase, current, total, format!("{phase}: {current}/{total}"))
}

pub(crate) fn progress_message(
    phase: &str,
    current: usize,
    total: usize,
    message: String,
) -> Result<()> {
    let Some(context) = WORKER.get() else {
        return Ok(());
    };
    let job_id = context
        .lock()
        .map_err(|_| anyhow::anyhow!("worker lock poisoned"))?
        .job_id
        .clone();
    emit(JobEvent::Progress {
        job_id,
        phase: phase.into(),
        current: current as u64,
        total: total as u64,
        message,
    })
}

pub(crate) fn set_output(output: &Path) -> Result<()> {
    let context = WORKER.get().context("desktop worker context missing")?;
    context
        .lock()
        .map_err(|_| anyhow::anyhow!("worker lock poisoned"))?
        .output = output.to_path_buf();
    Ok(())
}

pub(crate) fn write_frame(
    frame: usize,
    width: usize,
    height: usize,
    rgba: Option<&[u8]>,
    alpha: Option<&[u8]>,
    checker: Option<&[u8]>,
    mask: Option<&[u8]>,
) -> Result<()> {
    let Some(context) = WORKER.get() else {
        return Ok(());
    };
    let (job_id, output, start) = {
        let context = context
            .lock()
            .map_err(|_| anyhow::anyhow!("worker lock poisoned"))?;
        (
            context.job_id.clone(),
            context.output.clone(),
            context.start,
        )
    };
    let absolute = start
        .checked_add(frame)
        .context("absolute frame overflow")?;
    let write =
        |folder: &str, pixels: Option<&[u8]>, color: ColorType| -> Result<Option<PathBuf>> {
            let Some(pixels) = pixels else {
                return Ok(None);
            };
            let directory = output.join(folder);
            fs::create_dir_all(&directory)?;
            let path = directory.join(format!("{absolute:06}.png"));
            image::save_buffer_with_format(
                &path,
                pixels,
                width as u32,
                height as u32,
                color,
                ImageFormat::Png,
            )?;
            Ok(Some(path))
        };
    let rgba_path = write("frames", rgba, ColorType::Rgba8)?;
    let alpha_path = write("alpha", alpha, ColorType::L8)?;
    let checker_path = write("checker", checker, ColorType::Rgb8)?;
    let mask_path = write("masks", mask, ColorType::L8)?;
    emit(JobEvent::FrameReady {
        job_id,
        frame: absolute as u64,
        rgba_path,
        alpha_path,
        checker_path,
        mask_path,
    })
}

pub(crate) fn sam_hint(frame: usize, width: usize, height: usize) -> Result<Option<Vec<f32>>> {
    let Some(context) = WORKER.get() else {
        return Ok(None);
    };
    let context = context
        .lock()
        .map_err(|_| anyhow::anyhow!("worker lock poisoned"))?;
    let absolute = context.start + frame;
    let path = match &context.hints {
        HintInput::None => return Ok(None),
        HintInput::FirstFrame {
            mask_file,
            frame: source_frame,
            ..
        } => {
            if absolute as u64 != *source_frame {
                return Ok(None);
            }
            mask_file.clone()
        }
        HintInput::Video {
            mask_directory,
            frame_offset,
            ..
        } => {
            let index = i64::try_from(absolute)?
                .checked_sub(*frame_offset)
                .context("SAM mask frame offset overflow")?;
            ensure!(index >= 0, "SAM mask frame offset precedes zero");
            mask_directory.join(format!("{index:06}.png"))
        }
    };
    ensure!(
        path.is_file(),
        "SAM hint missing for source frame {}: {}",
        context.start + frame,
        path.display()
    );
    let mask = image::open(&path)?.into_luma8();
    ensure!(
        mask.width() == width as u32 && mask.height() == height as u32,
        "SAM hint geometry differs from source"
    );
    Ok(Some(
        mask.into_raw()
            .into_iter()
            .map(|x| f32::from(x) / 255.0)
            .collect(),
    ))
}

pub(crate) fn hint_is_video() -> bool {
    WORKER
        .get()
        .and_then(|c| c.lock().ok())
        .is_some_and(|c| matches!(c.hints, HintInput::Video { .. }))
}

pub(crate) fn hint_quality() -> f32 {
    WORKER
        .get()
        .and_then(|c| c.lock().ok())
        .map_or(0.95, |c| match c.hints {
            HintInput::None => 0.95,
            HintInput::FirstFrame { quality, .. } | HintInput::Video { quality, .. } => quality,
        })
}

fn decode_range(job: &JobRequest, start: u64, end: u64, destination: &Path) -> Result<()> {
    ensure!(
        start < end && end <= job.media.frame_count,
        "invalid half-open source frame range"
    );
    fs::create_dir_all(destination.parent().context("missing RGB parent")?)?;
    let mut command = Command::new(&job.runtime.ffmpeg);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let output = command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i"])
        .arg(&job.media.path)
        .args(["-map", "0:v:0", "-vf"])
        .arg(format!(
            "trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,format=rgb24"
        ))
        .args([
            "-an",
            "-sn",
            "-dn",
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
        ])
        .arg(destination)
        .output()?;
    ensure!(
        output.status.success(),
        "native range decode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected = (end - start)
        .checked_mul(job.media.width as u64)
        .and_then(|n| n.checked_mul(job.media.height as u64))
        .and_then(|n| n.checked_mul(3))
        .context("RGB length overflow")?;
    ensure!(
        fs::metadata(destination)?.len() == expected,
        "decoded source range frame count differs"
    );
    job.media.verify_source_unchanged()?;
    Ok(())
}

/// ORT graph optimization per job. NONE at 4K comes from the strict-FP32 CUDA qualification.
/// On DirectML it leaves the ViT feature models unfused: cradio at 4K then needs 12-28 GB and
/// runs 6-14 s/frame (ORT 1.23) instead of 3.7 GiB / 0.17 s, so DirectML always uses ALL.
fn ort_optimization_for(large_frame: bool, provider: Provider) -> &'static str {
    if large_frame && provider == Provider::Cuda {
        "none"
    } else {
        "all"
    }
}

pub(crate) fn worker_args(
    job: &JobRequest,
    manifest: PathBuf,
    model_root: PathBuf,
    provider: Provider,
) -> Args {
    let large_frame = u64::from(job.media.width) * u64::from(job.media.height) >= 3840 * 2160;
    Args {
        command: "desktop".into(),
        manifest,
        models_root: Some(model_root),
        video: None,
        ffmpeg: Some(job.runtime.ffmpeg.clone()),
        runtime: Some(job.runtime.ort_library.clone()),
        output: Some(job.output_dir.clone()),
        report: Some(job.output_dir.join("inference-report.json")),
        provider: match provider {
            Provider::Cpu => "cpu",
            Provider::DirectMl => "directml",
            Provider::Cuda => "cuda",
        }
        .into(),
        threads: 4,
        cuda_memory_mib: if large_frame { 12288 } else { 8192 },
        ort_optimization: ort_optimization_for(large_frame, provider).into(),
        tensor_dumps: "none".into(),
        profile: env::var("BRP_CHROMA_DIAGNOSTICS")
            .is_ok_and(|value| value == "1")
            .then(|| job.output_dir.join("provider-profile")),
        ..Default::default()
    }
}

pub fn run_cli() -> Result<()> {
    let mut raw = env::args().skip(2);
    let (mut job_file, mut event_file) = (None, None);
    let mut start_gate = None;
    while let Some(flag) = raw.next() {
        if matches!(flag.as_str(), "--help" | "-h") {
            println!(
                "ck-native desktop --job FILE --events FILE [--start-gate FILE]\n\nRuns an imported-media job using native FFmpeg and ONNX Runtime.\nFrame indices are zero-based; selected ranges are half-open [start,end)."
            );
            return Ok(());
        }
        let value = raw
            .next()
            .with_context(|| format!("missing {flag} value"))?;
        match flag.as_str() {
            "--job" | "--manifest" => job_file = Some(PathBuf::from(value)),
            "--events" => event_file = Some(PathBuf::from(value)),
            "--start-gate" => start_gate = Some(PathBuf::from(value)),
            _ => bail!("unknown desktop option {flag}"),
        }
    }
    let job: JobRequest =
        serde_json::from_slice(&fs::read(job_file.context("desktop --job is required")?)?)?;
    job.validate()?;
    // The desktop parent first attaches this process to its kill-on-close job
    // object, then publishes this gate. No FFmpeg child can escape assignment.
    if let Some(gate) = start_gate {
        let started = std::time::Instant::now();
        while !gate.is_file() {
            ensure!(
                started.elapsed() < std::time::Duration::from_secs(30),
                "Desktop start gate was not published"
            );
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
    }
    let observed = ck_desktop_core::MediaInfo::probe(&job.media.path, &job.runtime)?;
    ensure!(
        observed.width == job.media.width
            && observed.height == job.media.height
            && observed.frame_count == job.media.frame_count
            && observed.fps == job.media.fps
            && observed.time_base == job.media.time_base
            && observed.first_pts == job.media.first_pts,
        "Source metadata changed since import; reopen the footage before rendering"
    );
    ensure!(job.schema_version == 1, "unsupported desktop job schema");
    ensure!(
        job.media.path.is_file()
            && job.media.frame_count > 0
            && job.media.width > 0
            && job.media.height > 0,
        "invalid source media"
    );
    fs::create_dir_all(&job.output_dir)?;
    let event_file = event_file.context("desktop --events is required")?;
    if let Some(parent) = event_file.parent() {
        fs::create_dir_all(parent)?;
    }
    let events = OpenOptions::new()
        .create(true)
        .append(true)
        .open(event_file)?;
    let (start, hints) = match &job.kind {
        JobKind::CorridorKey { range, hints, .. } => (range.start as usize, hints.clone()),
        JobKind::SamVideo { range, .. } => (range.start as usize, HintInput::None),
        _ => (0, HintInput::None),
    };
    WORKER
        .set(Mutex::new(WorkerContext {
            job_id: job.job_id.clone(),
            events,
            output: job.output_dir.clone(),
            start,
            hints,
        }))
        .map_err(|_| anyhow::anyhow!("desktop worker already initialized"))?;
    emit(JobEvent::Started {
        job_id: job.job_id.clone(),
    })?;
    let result = (|| -> Result<JobArtifacts> {
        fs::write(
            job.output_dir.join("job-request.json"),
            serde_json::to_vec_pretty(&job)?,
        )?;
        let artifacts = execute(&job)?;
        job.media.verify_source_unchanged()?;
        Ok(artifacts)
    })();
    match result {
        Ok(artifacts) => emit(JobEvent::Completed {
            job_id: job.job_id,
            artifacts,
        }),
        Err(error) => {
            let _ = emit(JobEvent::Failed {
                job_id: job.job_id,
                message: format!("{error:#}"),
            });
            Err(error)
        }
    }
}

fn execute(job: &JobRequest) -> Result<JobArtifacts> {
    let work = job.output_dir.join("worker");
    fs::create_dir_all(&work)?;
    match &job.kind {
        JobKind::DetectScenes {
            threshold,
            min_scene_frames,
        } => super::transnet::desktop_detect(job, *threshold, *min_scene_frames),
        JobKind::CorridorKey {
            range,
            provider,
            hints,
            settings,
        } => {
            // Validate complete temporal coverage before loading large models.
            if let HintInput::Video {
                mask_directory,
                frame_offset,
                ..
            } = hints
            {
                for frame in range.start..range.end_exclusive {
                    let mask_frame = i64::try_from(frame)?
                        .checked_sub(*frame_offset)
                        .context("hint offset overflow")?;
                    ensure!(
                        mask_frame >= 0,
                        "Hint offset produces a negative mask frame"
                    );
                    let file = mask_directory.join(format!("{mask_frame:06}.png"));
                    ensure!(
                        file.is_file(),
                        "SAM mask coverage does not include source frame {frame}: {}",
                        file.display()
                    );
                    let (width, height) = image::image_dimensions(&file)?;
                    ensure!(
                        (width, height) == (job.media.width, job.media.height),
                        "SAM mask geometry differs at source frame {frame}"
                    );
                }
            }
            let models = job
                .runtime
                .ck_models_for(job.media.width, job.media.height)?
                .to_path_buf();
            super::video::run_desktop(job, range, settings, models, *provider)
        }
        JobKind::SamVideo {
            range,
            provider,
            prompts,
            model: SamModel::Sam31,
            boxes,
            text,
        } => {
            let count = usize::try_from(range.end_exclusive - range.start)?;
            ensure!(count > 0, "SAM range cannot be empty");
            let models = job.runtime.sam31_models();
            ensure!(
                job.runtime.sam31_available(),
                "SAM 3.1 pack not found at {}",
                models.display()
            );
            // One layer: points (optionally with text) and boxes on any frame of the range.
            let mut layer = super::sam31::LayerPrompts {
                text: text.clone().filter(|t| !t.trim().is_empty()),
                frames: BTreeMap::new(),
            };
            for p in prompts {
                ensure!(
                    range.contains(p.frame),
                    "SAM point outside the selected range"
                );
                layer
                    .frames
                    .entry((p.frame - range.start) as usize)
                    .or_default()
                    .points
                    .push(super::sam31::PointPrompt {
                        xy: [p.x, p.y],
                        positive: p.positive,
                        text: p.text.clone(),
                    });
            }
            for b in boxes {
                ensure!(
                    range.contains(b.frame),
                    "SAM box outside the selected range"
                );
                layer
                    .frames
                    .entry((b.frame - range.start) as usize)
                    .or_default()
                    .boxes
                    .push([b.x0, b.y0, b.x1, b.y1]);
            }
            progress("decode", 0, 1)?;
            let source = work.join("source.rgb24");
            decode_range(job, range.start, range.end_exclusive, &source)?;
            progress("decode", 1, 1)?;
            let args = worker_args(job, work.join("sam31-job.json"), models.clone(), *provider);
            super::sam31::run_desktop(
                &args,
                &models,
                &source,
                count,
                job.media.height as usize,
                job.media.width as usize,
                &layer,
            )?;
            Ok(JobArtifacts {
                masks_dir: Some(job.output_dir.join("masks")),
                report: Some(job.output_dir.join("inference-report.json")),
                ..Default::default()
            })
        }
        JobKind::SamVideo {
            range,
            provider,
            prompts,
            model: SamModel::Sam21,
            ..
        } => {
            let count = usize::try_from(range.end_exclusive - range.start)?;
            ensure!(count > 0, "SAM range cannot be empty");
            let config: serde_json::Value = serde_json::from_slice(&fs::read(
                job.runtime.sam_models.join("video_config.json"),
            )?)?;
            let constants = config["constants"]
                .as_object()
                .context("SAM model config constants missing")?;
            for value in constants.values() {
                let relative = value["file"]
                    .as_str()
                    .context("SAM constant filename missing")?;
                let target = pack_path(&work, relative)?;
                fs::create_dir_all(target.parent().context("constant parent missing")?)?;
                fs::copy(pack_path(&job.runtime.sam_models, relative)?, target)?;
            }
            let mut grouped: BTreeMap<usize, Vec<[f32; 3]>> = BTreeMap::new();
            for point in prompts {
                ensure!(
                    point.frame >= range.start
                        && point.frame < range.end_exclusive
                        && point.x.is_finite()
                        && point.y.is_finite()
                        && point.x >= 0.0
                        && point.x < job.media.width as f32
                        && point.y >= 0.0
                        && point.y < job.media.height as f32,
                    "SAM point outside selected source range/image"
                );
                grouped
                    .entry((point.frame - range.start) as usize)
                    .or_default()
                    .push([point.x, point.y, if point.positive { 1.0 } else { 0.0 }]);
            }
            ensure!(
                grouped
                    .get(&0)
                    .is_some_and(|p| p.iter().any(|v| v[2] == 1.0)),
                "Place a positive SAM point on the first selected frame"
            );
            let mut point_specs = BTreeMap::new();
            for (&frame, points) in &grouped {
                let coords = points.iter().flat_map(|v| [v[0], v[1]]).collect();
                let labels = points.iter().map(|v| v[2]).collect();
                let cf = format!("point-{frame:06}.npy");
                let lf = format!("label-{frame:06}.npy");
                write_npy(
                    work.join(&cf),
                    &ArrayD::from_shape_vec(IxDyn(&[1, points.len(), 2]), coords)?,
                )?;
                write_npy(
                    work.join(&lf),
                    &ArrayD::from_shape_vec(IxDyn(&[1, points.len()]), labels)?,
                )?;
                point_specs.insert(frame,(json!({"name":"point_coords","file":cf,"dtype":"float32","shape":[1,points.len(),2]}),json!({"name":"point_labels","file":lf,"dtype":"float32","shape":[1,points.len()]})));
            }
            let mut events = Vec::new();
            for frame in 0..count {
                let first = frame == 0;
                let points = if first { point_specs.get(&0) } else { None };
                events.push(json!({"name":format!("track-{frame:06}"),"frame_idx":frame,"is_init_cond_frame":first,"store_as_conditioning":first,"point_coords":points.map(|p|&p.0),"point_labels":points.map(|p|&p.1),"outputs":[]}));
            }
            for (&frame, points) in point_specs.iter().filter(|(frame, _)| **frame > 0) {
                events.push(json!({"name":format!("correct-{frame:06}"),"frame_idx":frame,"is_init_cond_frame":false,"store_as_conditioning":true,"point_coords":points.0,"point_labels":points.1,"outputs":[]}));
                for next in frame + 1..count {
                    events.push(json!({"name":format!("replay-{frame:06}-{next:06}"),"frame_idx":next,"is_init_cond_frame":false,"store_as_conditioning":false,"outputs":[]}));
                }
            }
            progress("decode", 0, 1)?;
            decode_range(
                job,
                range.start,
                range.end_exclusive,
                &work.join("source.rgb24"),
            )?;
            progress("decode", 1, 1)?;
            let fixture = json!({"schema_version":1,"kind":"sam2_video_inference_job","image_size":1024,"original_hw":[job.media.height,job.media.width],"total_frames":count,"point_coordinate_space":"original_pixels","frames":[],"events":events,"constants":config["constants"],"config":config["config"],"models":{"image_encoder":"image_encoder/model.onnx","memory_encoder":"memory_encoder/model.onnx","memory_attention":"memory_attention/model.onnx","video_heads":"video_heads/model.onnx"},"tolerance":{"atol":0.0005,"rtol":0.0005},"source":{"desktop_rgb":"source.rgb24"}});
            let manifest = work.join("sam-job.json");
            fs::write(&manifest, serde_json::to_vec_pretty(&fixture)?)?;
            super::sam_video::run(&worker_args(
                job,
                manifest,
                job.runtime.sam_models.clone(),
                *provider,
            ))?;
            Ok(JobArtifacts {
                masks_dir: Some(job.output_dir.join("masks")),
                report: Some(job.output_dir.join("inference-report.json")),
                ..Default::default()
            })
        }
    }
}

#[cfg(test)]
mod optimization_policy_tests {
    use super::*;

    #[test]
    fn directml_keeps_graph_optimization_at_4k() {
        assert_eq!(ort_optimization_for(true, Provider::DirectMl), "all");
        assert_eq!(ort_optimization_for(false, Provider::DirectMl), "all");
        assert_eq!(ort_optimization_for(true, Provider::Cuda), "none");
        assert_eq!(ort_optimization_for(false, Provider::Cuda), "all");
        assert_eq!(ort_optimization_for(true, Provider::Cpu), "all");
    }
}
