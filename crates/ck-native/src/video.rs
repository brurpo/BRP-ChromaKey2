//! Disk-backed full-clip CK validation. Runtime execution is native throughout.
mod adaptive;
use super::npy::{read_npy, write_npy};
use super::pipeline::Graph;
use super::{Args, Metrics, Tolerance, compare, initialize_runtime, pack_path};
pub(crate) use adaptive::run as run_desktop;
use anyhow::{Context, Result, ensure};
#[cfg(test)]
use ck_core::color::{linear_to_asinh, srgb_to_linear};
use ck_core::{
    color::{asinh_to_linear, linear_to_srgb},
    hints::{HintCarry, HintQualities},
    spatial::{
        Padding, resize_area_chw, resize_area_chw_cuda_order, resize_bilinear_chw,
        resize_bilinear_chw_cuda_order,
    },
    windows::{BatonAccumulator, WindowPlan},
};
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    thread,
    time::Instant,
};

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    kind: String,
    source_rgb: String,
    #[serde(default)]
    source_rgb_sha256: Option<String>,
    #[serde(default)]
    source_video: serde_json::Value,
    native_hw: [usize; 2],
    shape_hw: [usize; 2],
    total_frames: usize,
    chunk: usize,
    stride: usize,
    hint_mode: String,
    carry_hint: bool,
    carry_hint_quality: f32,
    precision: String,
    runtime_storage_rounding: bool,
    #[serde(default = "default_fps")]
    fps: String,
    references: Vec<Reference>,
    #[serde(default)]
    decoded_samples: Vec<DecodedSample>,
    tolerance: Tolerance,
}
fn default_fps() -> String {
    "30000/1001".into()
}
#[derive(Deserialize)]
struct Reference {
    frame: usize,
    file: String,
}
#[derive(Deserialize)]
struct DecodedSample {
    frame: usize,
    alpha: String,
    foreground_linear: String,
}

#[derive(Default, Serialize)]
struct Aggregate {
    frames: usize,
    elements: usize,
    failed_elements: usize,
    max_abs_error: f64,
    mean_abs_error: f64,
    root_mean_square_error: f64,
}
impl Aggregate {
    fn push(&mut self, value: &Metrics) {
        let old = self.elements as f64;
        self.elements += value.elements;
        self.frames += 1;
        self.failed_elements += value.failed_elements;
        self.max_abs_error = self.max_abs_error.max(value.max_abs_error);
        self.mean_abs_error = (self.mean_abs_error * old
            + value.mean_abs_error * value.elements as f64)
            / self.elements as f64;
        self.root_mean_square_error = ((self.root_mean_square_error.powi(2) * old
            + value.root_mean_square_error.powi(2) * value.elements as f64)
            / self.elements as f64)
            .sqrt();
    }
}

fn tensor(shape: &[usize], values: Vec<f32>) -> Result<ArrayD<f32>> {
    Ok(ArrayD::from_shape_vec(IxDyn(shape), values)?)
}

// Runtime model geometry is separate from fixed-shape golden TensorSpec data.
// A symbolic dimension must name an explicit source-derived quantity; changing
// ONNX metadata alone cannot make a fixed-shape graph pass the output checks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ModelDimension {
    Fixed(usize),
    Symbol(String),
}

#[derive(Deserialize)]
struct ModelTensorContract {
    name: String,
    dtype: String,
    shape: Vec<ModelDimension>,
}

#[derive(Deserialize)]
struct FoundationContract {
    schema_version: u32,
    component: String,
    model: String,
    inputs: Vec<ModelTensorContract>,
    outputs: Vec<ModelTensorContract>,
}

struct ResolvedFoundation {
    rgb: Vec<usize>,
    features: Vec<usize>,
    state: Option<Vec<usize>>,
}

fn foundation_contract(
    root: &Path,
    name: &str,
    height: usize,
    width: usize,
) -> Result<ResolvedFoundation> {
    ensure!(
        height.is_multiple_of(128) && width.is_multiple_of(128) && height > 0 && width > 0,
        "CorridorKey needs positive padded dimensions divisible by 128"
    );
    let grid_h = height / 64;
    let grid_w = width / 64;
    let rvm_tokens = grid_h
        .checked_mul(grid_w)
        .and_then(|n| n.checked_add(1))
        .context("RVM token count overflow")?;
    let symbols = BTreeMap::from([
        ("grid_h", grid_h),
        ("grid_w", grid_w),
        ("rvm_tokens", rvm_tokens),
        ("rvm_h", height / 4),
        ("rvm_w", width / 4),
        ("cradio_h", height / 4),
        ("cradio_w", width / 4),
        ("moge_h", height / 32 * 7),
        ("moge_w", width / 32 * 7),
    ]);
    let manifest: FoundationContract =
        serde_json::from_slice(&fs::read(root.join(name).join("manifest.json"))?)?;
    ensure!(
        manifest.schema_version == 1
            && manifest.component == name
            && manifest.model == "model.onnx",
        "unsupported {name} runtime contract"
    );
    let resolve = |spec: &ModelTensorContract| -> Result<Vec<usize>> {
        ensure!(
            spec.dtype == "float32",
            "{name}/{} must use float32",
            spec.name
        );
        spec.shape
            .iter()
            .map(|dimension| {
                let value = match dimension {
                    ModelDimension::Fixed(value) => *value,
                    ModelDimension::Symbol(symbol) => {
                        *symbols.get(symbol.as_str()).with_context(|| {
                            format!("unknown {name}/{} geometry symbol {symbol}", spec.name)
                        })?
                    }
                };
                ensure!(value > 0, "zero {name}/{} dimension", spec.name);
                Ok(value)
            })
            .collect()
    };
    let find = |specs: &[ModelTensorContract], tensor: &str| -> Result<Vec<usize>> {
        let matching: Vec<_> = specs.iter().filter(|s| s.name == tensor).collect();
        ensure!(
            matching.len() == 1,
            "{name} must declare exactly one {tensor} tensor"
        );
        resolve(matching[0])
    };
    let channels = match name {
        "rvm" => 384,
        "cradio" => 1152,
        "moge" => 768,
        _ => anyhow::bail!("unknown CorridorKey foundation {name}"),
    };
    let rgb = find(&manifest.inputs, "rgb")?;
    let features = find(&manifest.outputs, "features")?;
    let expected_rgb = if name == "rvm" {
        vec![1, 1, 3, height / 4, width / 4]
    } else if name == "moge" {
        vec![1, 3, height / 32 * 7, width / 32 * 7]
    } else {
        vec![1, 3, height / 4, width / 4]
    };
    ensure!(
        rgb == expected_rgb && features == [1, channels, grid_h, grid_w],
        "{name} model contract does not support source padding {height}x{width}; install the dynamic CorridorKey pack"
    );
    let state = if name == "rvm" {
        let state = find(&manifest.inputs, "state")?;
        ensure!(
            state == [1, rvm_tokens, channels] && find(&manifest.outputs, "next_state")? == state,
            "RVM state contract must match the source patch grid plus its class token"
        );
        Some(state)
    } else {
        None
    };
    Ok(ResolvedFoundation {
        rgb,
        features,
        state,
    })
}

fn directml_dimensions(height: usize, width: usize) -> Result<BTreeMap<String, i64>> {
    let tokens = (height / 64)
        .checked_mul(width / 64)
        .and_then(|n| n.checked_add(1))
        .context("RVM token count overflow")?;
    [
        ("height", height),
        ("width", width),
        ("grid_h", height / 64),
        ("grid_w", width / 64),
        ("baton_h", height / 16),
        ("baton_w", width / 16),
        ("aux_h", height / 2),
        ("aux_w", width / 2),
        ("rvm_h", height / 4),
        ("rvm_w", width / 4),
        ("rvm_tokens", tokens),
        ("cradio_h", height / 4),
        ("cradio_w", width / 4),
        ("moge_h", height / 32 * 7),
        ("moge_w", width / 32 * 7),
    ]
    .into_iter()
    .map(|(name, size)| Ok((name.to_owned(), i64::try_from(size)?)))
    .collect()
}

/// Independent pointwise work only: no reduction, reordering or nested GPU work.
fn map_host_pixels(
    input: &[f32],
    threads: usize,
    operation: impl Fn(f32) -> f32 + Sync,
) -> Vec<f32> {
    const MIN_PARALLEL_ELEMENTS: usize = 64 * 1024;
    let workers = threads
        .max(1)
        .min(input.len().div_ceil(MIN_PARALLEL_ELEMENTS).max(1));
    if workers == 1 {
        return input.iter().map(|&value| operation(value)).collect();
    }
    let mut output = vec![0.0; input.len()];
    let chunk = input.len().div_ceil(workers);
    thread::scope(|scope| {
        for (source, target) in input.chunks(chunk).zip(output.chunks_mut(chunk)) {
            let operation = &operation;
            scope.spawn(move || {
                for (target, &source) in target.iter_mut().zip(source) {
                    *target = operation(source);
                }
            });
        }
    });
    output
}

fn checker_review_bytes(
    alpha: &[f32],
    foreground: &[f32],
    width: usize,
    height: usize,
    threads: usize,
) -> Vec<u8> {
    let pixels = width * height;
    let mut bytes = vec![0u8; pixels * 3];
    let workers = threads.max(1).min(height.div_ceil(32).max(1));
    let rows = height.div_ceil(workers);
    let paint = |first_row: usize, target: &mut [u8]| {
        for (relative_y, row) in target.chunks_exact_mut(width * 3).enumerate() {
            let y = first_row + relative_y;
            for x in 0..width {
                let i = y * width + x;
                let background = if (x / 64 + y / 64).is_multiple_of(2) {
                    0.75
                } else {
                    0.5
                };
                for c in 0..3 {
                    row[3 * x + c] = (linear_to_srgb(
                        foreground[c * pixels + i] * alpha[i] + background * (1.0 - alpha[i]),
                    )
                    .clamp(0.0, 1.0)
                        * 255.0)
                        .round() as u8;
                }
            }
        }
    };
    if workers == 1 {
        paint(0, &mut bytes);
    } else {
        thread::scope(|scope| {
            for (index, target) in bytes.chunks_mut(rows * width * 3).enumerate() {
                let paint = &paint;
                scope.spawn(move || paint(index * rows, target));
            }
        });
    }
    bytes
}
fn take(map: &mut BTreeMap<String, ArrayD<f32>>, name: &str) -> Result<ArrayD<f32>> {
    map.remove(name)
        .with_context(|| format!("missing graph output {name}"))
}
fn frame_path(root: &Path, component: &str, index: usize) -> PathBuf {
    root.join(component).join(format!("{index:06}.npy"))
}

fn release_work_frame(root: &Path, component: &str, index: usize) -> Result<u64> {
    ensure!(
        ["rvm", "cradio", "moge", "baton"].contains(&component),
        "unknown generated work component"
    );
    let path = frame_path(root, component, index);
    let resolved_root = fs::canonicalize(root)?;
    let resolved_path = fs::canonicalize(&path)?;
    ensure!(
        resolved_path.starts_with(&resolved_root),
        "generated work path escaped its intended directory"
    );
    let bytes = fs::metadata(&resolved_path)?.len();
    fs::remove_file(resolved_path)?;
    Ok(bytes)
}

pub(crate) struct RawClip {
    file: File,
    height: usize,
    width: usize,
    frame_bytes: usize,
}
impl RawClip {
    pub(crate) fn open(path: &Path, frames: usize, height: usize, width: usize) -> Result<Self> {
        let frame_bytes = height
            .checked_mul(width)
            .and_then(|v| v.checked_mul(3))
            .context("RGB size overflow")?;
        let expected = frame_bytes
            .checked_mul(frames)
            .context("clip size overflow")?;
        ensure!(
            fs::metadata(path)?.len() == expected as u64,
            "RGB cache length mismatch; expected {expected} bytes"
        );
        Ok(Self {
            file: File::open(path)?,
            height,
            width,
            frame_bytes,
        })
    }
    pub(crate) fn read(&mut self, index: usize) -> Result<Vec<f32>> {
        self.read_mapped(index, |byte| f32::from(byte) / 255.0)
    }
    fn read_mapped(&mut self, index: usize, encode: impl Fn(u8) -> f32) -> Result<Vec<f32>> {
        self.file.seek(SeekFrom::Start(
            index
                .checked_mul(self.frame_bytes)
                .context("RGB seek overflow")? as u64,
        ))?;
        let mut packed = vec![0u8; self.frame_bytes];
        self.file.read_exact(&mut packed)?;
        let pixels = self.height * self.width;
        let mut planar = vec![0.0; self.frame_bytes];
        for (index, rgb) in packed.as_chunks::<3>().0.iter().enumerate() {
            for channel in 0..3 {
                planar[channel * pixels + index] = encode(rgb[channel]);
            }
        }
        Ok(planar)
    }
    fn padded(&mut self, index: usize, padding: &Padding) -> Result<Vec<f32>> {
        Ok(padding.pad_chw(&self.read(index)?, 3)?)
    }
    fn padded_model_rgb(&mut self, index: usize, padding: &Padding) -> Result<Vec<f32>> {
        // This reader's source is always packed RGB24, so byte lookup is exact.
        // General floating-point/HDR preprocessing lives in ck-core::color.
        Ok(padding.pad_chw(&self.read_mapped(index, ck_core::rgb8_encoding::encode)?, 3)?)
    }
}

fn hidden(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    #[cfg(not(windows))]
    {
        let _ = command;
    }
}
fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn prepare_rgb(
    args: &Args,
    fixture: &Fixture,
    root: &Path,
    output: &Path,
) -> Result<(PathBuf, serde_json::Value)> {
    let reference_rgb = pack_path(root, &fixture.source_rgb)?;
    if args.command == "desktop" {
        return Ok((
            reference_rgb.clone(),
            serde_json::json!({"mode":"desktop_native_frame_range_decode","decoded_rgb_sha256":sha256_file(&reference_rgb)?}),
        ));
    }
    let expected_rgb = if let Some(hash) = &fixture.source_rgb_sha256 {
        hash.clone()
    } else {
        sha256_file(&reference_rgb)?
    };
    if let Some(video) = &args.video {
        let ffmpeg = args
            .ffmpeg
            .as_ref()
            .context("--ffmpeg required for native video decoding")?;
        let source_hash = sha256_file(video)?;
        if let Some(expected) = fixture
            .source_video
            .get("sha256")
            .and_then(|value| value.as_str())
        {
            ensure!(
                source_hash == expected,
                "supplied video does not match the reference source SHA256"
            );
        }
        let path = output.join("work/source.rgb24");
        let [height, width] = fixture.native_hw;
        decode_rgb24(ffmpeg, video, &path, fixture.total_frames, height, width)?;
        let decoded_hash = sha256_file(&path)?;
        ensure!(
            decoded_hash == expected_rgb,
            "native decoded RGB differs from reference input SHA256"
        );
        Ok((
            path,
            serde_json::json!({"mode":"native_video_decode","source_sha256":source_hash,"decoded_rgb_sha256":decoded_hash,"independent_decode_matches_reference":true}),
        ))
    } else {
        let hash = sha256_file(&reference_rgb)?;
        ensure!(
            hash == expected_rgb,
            "prepared RGB24 fixture SHA256 mismatch"
        );
        Ok((
            reference_rgb,
            serde_json::json!({"mode":"prepared_rgb24_fixture","decoded_rgb_sha256":hash,"independent_native_decode":false}),
        ))
    }
}

pub(crate) fn decode_rgb24(
    ffmpeg: &Path,
    video: &Path,
    path: &Path,
    total_frames: usize,
    height: usize,
    width: usize,
) -> Result<()> {
    let temporary = path.with_extension("rgb24.partial");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut command = Command::new(ffmpeg);
    hidden(&mut command);
    let status = command
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i"])
        .arg(video)
        .args([
            "-an",
            "-sn",
            "-dn",
            "-vf",
            &format!("scale={width}:{height}:flags=bicubic"),
            "-frames:v",
            &total_frames.to_string(),
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
        ])
        .arg(&temporary)
        .status()?;
    ensure!(status.success(), "FFmpeg native decode failed");
    RawClip::open(&temporary, total_frames, height, width)?;
    fs::rename(temporary, path)?;
    Ok(())
}

struct Review {
    child: Child,
    input: Option<ChildStdin>,
    finished: bool,
}
impl Review {
    fn open(
        ffmpeg: &Path,
        path: &Path,
        width: usize,
        height: usize,
        fps: &str,
        gray: bool,
    ) -> Result<Self> {
        let mut command = Command::new(ffmpeg);
        hidden(&mut command);
        command
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                if gray { "gray" } else { "rgb24" },
                "-video_size",
                &format!("{width}x{height}"),
                "-framerate",
                fps,
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                if width.is_multiple_of(2) && height.is_multiple_of(2) {
                    "yuv420p"
                } else {
                    // 4:4:4 preserves literal odd source dimensions without
                    // changing source pixels or the exported PNG geometry.
                    "yuv444p"
                },
                "-threads",
                "2",
            ])
            .arg(path)
            .stdin(Stdio::piped());
        let mut child = command.spawn()?;
        let input = child.stdin.take();
        Ok(Self {
            child,
            input,
            finished: false,
        })
    }
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.input
            .as_mut()
            .context("review pipe closed")?
            .write_all(bytes)?;
        Ok(())
    }
    fn finish(mut self) -> Result<()> {
        drop(self.input.take());
        let status = self.child.wait()?;
        self.finished = true;
        ensure!(status.success(), "review encode failed");
        Ok(())
    }
}
impl Drop for Review {
    fn drop(&mut self) {
        if !self.finished {
            drop(self.input.take());
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn decoded(
    padding: &Padding,
    logits: &ArrayD<f32>,
    threads: usize,
) -> Result<(ArrayD<f32>, ArrayD<f32>)> {
    let cropped = padding.crop_chw(logits.as_slice().context("contiguous logits required")?, 4)?;
    let pixels = padding.native_height * padding.native_width;
    let alpha = tensor(
        &[1, 1, padding.native_height, padding.native_width],
        cropped[..pixels]
            .iter()
            .map(|v| v.clamp(0.0, 1.0))
            .collect(),
    )?;
    let foreground = tensor(
        &[1, 3, padding.native_height, padding.native_width],
        map_host_pixels(&cropped[pixels..], threads, asinh_to_linear),
    )?;
    Ok((alpha, foreground))
}

pub fn run(args: &Args) -> Result<()> {
    let started = Instant::now();
    let root = args.manifest.parent().unwrap_or(Path::new("."));
    let fixture: Fixture = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    let inference = args.command == "desktop";
    ensure!(
        fixture.schema_version == 1
            && fixture.kind
                == if inference {
                    "ck_video_inference_job"
                } else {
                    "ck_video_fixture"
                },
        "unsupported real video fixture"
    );
    ensure!(
        fixture.precision == "float32" && !fixture.runtime_storage_rounding,
        "real proof currently requires FP32 computation/storage"
    );
    ensure!(
        fixture.hint_mode == "none",
        "real video proof currently uses no external subject hint"
    );
    ensure!(
        fixture.carry_hint_quality.is_finite() && (0.0..=1.0).contains(&fixture.carry_hint_quality),
        "invalid carry quality"
    );
    ensure!(
        inference
            || (fixture.references.len() == fixture.total_frames
                && fixture
                    .references
                    .iter()
                    .enumerate()
                    .all(|(i, r)| i == r.frame)),
        "need exactly one ordered reference per frame"
    );
    let mut spot_frames = std::collections::BTreeSet::new();
    for sample in &fixture.decoded_samples {
        ensure!(
            sample.frame < fixture.total_frames && spot_frames.insert(sample.frame),
            "decoded sample frames must be unique and within the clip"
        );
    }
    let tolerance = Tolerance {
        atol: args.atol.unwrap_or(fixture.tolerance.atol),
        rtol: args.rtol.unwrap_or(fixture.tolerance.rtol),
    };
    ensure!(
        tolerance.atol.is_finite()
            && tolerance.rtol.is_finite()
            && tolerance.atol >= 0.0
            && tolerance.rtol >= 0.0,
        "invalid tolerance"
    );
    let [height, width] = fixture.shape_hw;
    let [native_height, native_width] = fixture.native_hw;
    let padding = Padding::new(native_height, native_width, 128)?;
    ensure!(
        [padding.height, padding.width] == fixture.shape_hw,
        "padding mismatch"
    );
    let plan = WindowPlan::new(fixture.total_frames, fixture.chunk, fixture.stride)?;
    let models_root = args
        .models_root
        .as_ref()
        .context("--models-root is required")?;
    let contracts = ["rvm", "cradio", "moge"]
        .into_iter()
        .map(|name| Ok((name, foundation_contract(models_root, name, height, width)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let output = args.output.clone().unwrap_or_else(|| root.join("native"));
    let mut specialized_args = args.clone();
    if args.provider == "directml" {
        // Specialize this session to actual source padding, while retaining one
        // spatially dynamic set of graphs and unchanged source pixel geometry.
        specialized_args
            .dimension_overrides
            .extend(directml_dimensions(height, width)?);
        specialized_args
            .dimension_overrides
            .insert("frames".into(), i64::try_from(plan.windows[0].len())?);
    }
    let args = &specialized_args;
    let (raw_path, input_provenance) = prepare_rgb(args, &fixture, root, &output)?;
    let mut clip = RawClip::open(&raw_path, fixture.total_frames, native_height, native_width)?;
    let scratch = output.join("work");
    for name in ["rvm", "moge", "cradio", "baton"] {
        fs::create_dir_all(scratch.join(name))?;
    }
    initialize_runtime(args)?;
    let mut components = Vec::new();
    let mut released_work_bytes = 0u64;
    let mut channels = BTreeMap::new();
    for name in ["rvm", "cradio", "moge"] {
        let contract = &contracts[name];
        channels.insert(name, contract.features[1]);
        let mut graph = Graph::open(args, models_root, name)?;
        let mut state = contract
            .state
            .as_deref()
            .map(|shape| ArrayD::<f32>::zeros(IxDyn(shape)));
        for frame in 0..fixture.total_frames {
            let display = clip.padded(frame, &padding)?;
            let (h, w) = if name == "moge" {
                (height * 7 / 32, width * 7 / 32)
            } else {
                (height / 4, width / 4)
            };
            let rgb = if name == "rvm" {
                resize_bilinear_chw(&display, 3, height, width, h, w)?
            } else if args.provider == "cuda" {
                resize_area_chw_cuda_order(&display, 3, height, width, h, w)?
            } else {
                resize_area_chw(&display, 3, height, width, h, w)?
            };
            let mut inputs = vec![("rgb", tensor(&contract.rgb, rgb)?)];
            if name == "rvm" {
                inputs.push(("state", state.take().context("missing recurrent state")?));
            }
            let mut result = graph.infer(inputs)?;
            let feature = take(&mut result, "features")?;
            ensure!(
                feature.shape() == contract.features,
                "{name} returned features for a different source geometry"
            );
            write_npy(frame_path(&scratch, name, frame), &feature)?;
            if name == "rvm" {
                let next_state = take(&mut result, "next_state")?;
                ensure!(
                    Some(next_state.shape()) == contract.state.as_deref(),
                    "RVM returned state for a different source patch grid"
                );
                state = Some(next_state);
            }
            println!("{name}: {}/{}", frame + 1, fixture.total_frames);
            super::desktop::progress(name, frame + 1, fixture.total_frames)?;
        }
        components.push(graph.finish()?);
    }
    let mut graph = Graph::open(args, models_root, "stage1")?;
    let capture_graph_hash = if args.diagnostic_windows.is_empty() {
        None
    } else {
        Some(sha256_file(&models_root.join("stage1/model.onnx"))?)
    };
    let mut hints = HintCarry::new(height * width)?;
    let baton_elements = 512 * (height / 16) * (width / 16);
    let mut accumulation = BatonAccumulator::new(baton_elements)?;
    for window in &plan.windows {
        let mut rgb = Vec::with_capacity(window.len() * 3 * height * width);
        for index in window.start..window.end {
            rgb.extend(clip.padded_model_rgb(index, &padding)?);
        }
        let video_hints = if inference && super::desktop::hint_is_video() {
            let mut masks = Vec::with_capacity(window.len() * height * width);
            for frame in window.start..window.end {
                let mask = super::desktop::sam_hint(frame, native_width, native_height)?
                    .context("SAM video hint missing")?;
                masks.extend(padding.pad_chw(&mask, 1)?);
            }
            Some(masks)
        } else {
            None
        };
        let first_hint = if inference && window.start == 0 && !super::desktop::hint_is_video() {
            super::desktop::sam_hint(0, native_width, native_height)?
                .map(|mask| padding.pad_chw(&mask, 1))
                .transpose()?
        } else {
            None
        };
        let quality = super::desktop::hint_quality();
        let hint = hints.build_window(
            window,
            video_hints.as_deref(),
            first_hint.as_deref(),
            HintQualities {
                carry: fixture.carry_hint_quality,
                full_video: quality,
                initial: quality,
            },
        )?;
        let mut inputs = vec![
            ("rgb", tensor(&[1, window.len(), 3, height, width], rgb)?),
            (
                "hint",
                tensor(&[1, window.len(), 1, height, width], hint.masks)?,
            ),
            ("hint_quality", tensor(&[1, window.len()], hint.quality)?),
        ];
        for (name, input) in [
            ("moge", "moge_feat"),
            ("cradio", "cradio_feat"),
            ("rvm", "rvm_feat"),
        ] {
            let mut values = Vec::new();
            for index in window.start..window.end {
                let array = read_npy(frame_path(&scratch, name, index))?;
                values.extend(array.iter().copied());
            }
            inputs.push((
                input,
                tensor(
                    &[1, window.len(), channels[name], height / 64, width / 64],
                    values,
                )?,
            ));
        }
        let capture = args.diagnostic_windows.contains(&window.index).then(|| {
            output
                .join("diagnostics")
                .join(format!("window-{:06}", window.index))
        });
        let mut captured_inputs = Vec::new();
        if let Some(directory) = &capture {
            fs::create_dir_all(directory)?;
            for (name, value) in &inputs {
                let file = format!("input_{name}.npy");
                write_npy(directory.join(&file), value)?;
                captured_inputs.push(serde_json::json!({"name":name,"file":file,"shape":value.shape(),"dtype":"float32"}));
            }
        }
        let mut result = graph.infer(inputs)?;
        let mut captured_outputs = Vec::new();
        if let Some(directory) = &capture {
            for (name, value) in &result {
                let file = format!("output_{name}.npy");
                write_npy(directory.join(&file), value)?;
                captured_outputs.push(serde_json::json!({"name":name,"file":file,"shape":value.shape(),"dtype":"float32"}));
            }
        }
        let baton = take(&mut result, "baton")?;
        ensure!(
            baton.shape() == [1, window.len(), 512, height / 16, width / 16],
            "stage1 baton geometry mismatch"
        );
        if fixture.carry_hint {
            let auxiliary = take(&mut result, "aux_alpha")?;
            let mut full = Vec::with_capacity(window.len() * height * width);
            for frame in auxiliary
                .as_slice()
                .context("contiguous auxiliary needed")?
                .chunks_exact((height / 2) * (width / 2))
            {
                let upsampled = if args.provider == "cuda" {
                    resize_bilinear_chw_cuda_order(frame, 1, height / 2, width / 2, height, width)?
                } else {
                    resize_bilinear_chw(frame, 1, height / 2, width / 2, height, width)?
                };
                full.extend(upsampled.into_iter().map(|v| v.clamp(0.0, 1.0)));
            }
            hints.commit_auxiliary(window, &full)?;
            if let Some(directory) = &capture {
                write_npy(
                    directory.join("carry_full.npy"),
                    &tensor(&[1, window.len(), 1, height, width], full)?,
                )?;
            }
        }
        if let Some(directory) = &capture {
            let manifest = serde_json::json!({
                "schema_version":1,"kind":"native_ck_window_capture","window_index":window.index,
                "absolute_frames":[window.start,window.end],"finalize_before":window.finalize_before,
                "models_root":models_root,"stage1_graph_sha256":capture_graph_hash,
                "provider":args.provider,"ort_optimization":args.ort_optimization,
                "precision":"float32","inputs":captured_inputs,"outputs":captured_outputs,
                "carry_full":"carry_full.npy","scope":"Developer-only actual native inputs/outputs; not independent reference data."
            });
            fs::write(
                directory.join("capture.json"),
                serde_json::to_string_pretty(&manifest)?,
            )?;
        }
        for frame in accumulation
            .push_window(window, baton.as_slice().context("contiguous baton needed")?)?
        {
            write_npy(
                frame_path(&scratch, "baton", frame.index),
                &tensor(&[1, 1, 512, height / 16, width / 16], frame.values)?,
            )?;
            for name in ["rvm", "cradio", "moge"] {
                released_work_bytes += release_work_frame(&scratch, name, frame.index)?;
            }
        }
        println!(
            "stage1: window {}/{} ({}..{})",
            window.index + 1,
            plan.windows.len(),
            window.start,
            window.end
        );
        super::desktop::progress("stage1", window.index + 1, plan.windows.len())?;
    }
    ensure!(accumulation.pending_frames() == 0, "unfinalized batons");
    components.push(graph.finish()?);
    let mut graph = Graph::open(args, models_root, "stage2")?;
    let mut alpha_review = if let Some(ffmpeg) = &args.ffmpeg {
        Some(Review::open(
            ffmpeg,
            &output.join("alpha.mp4"),
            native_width,
            native_height,
            &fixture.fps,
            true,
        )?)
    } else {
        None
    };
    let mut checker_review = if let Some(ffmpeg) = &args.ffmpeg {
        Some(Review::open(
            ffmpeg,
            &output.join("checker.mp4"),
            native_width,
            native_height,
            &fixture.fps,
            false,
        )?)
    } else {
        None
    };
    let mut totals: BTreeMap<String, Aggregate> = BTreeMap::new();
    let mut decoded_spot_totals: BTreeMap<String, Aggregate> = BTreeMap::new();
    let mut frame_metrics = Vec::new();
    for index in 0..fixture.total_frames {
        let rgb = clip.padded_model_rgb(index, &padding)?;
        let mut result = graph.infer(vec![
            ("rgb", tensor(&[1, 1, 3, height, width], rgb)?),
            ("baton", read_npy(frame_path(&scratch, "baton", index))?),
        ])?;
        let logits = take(&mut result, "logits")?;
        ensure!(
            logits.shape() == [1, 1, 4, height, width],
            "stage2 profile geometry mismatch"
        );
        let (alpha, foreground) = decoded(&padding, &logits, args.threads)?;
        if inference {
            let alpha_values = alpha.as_slice().context("alpha must be contiguous")?;
            let foreground_values = foreground
                .as_slice()
                .context("foreground must be contiguous")?;
            let pixels = native_width * native_height;
            let alpha_bytes: Vec<u8> = alpha_values
                .iter()
                .map(|v| (v * 255.0).round() as u8)
                .collect();
            let mut rgba = Vec::with_capacity(pixels * 4);
            for i in 0..pixels {
                for channel in 0..3 {
                    rgba.push(
                        (linear_to_srgb(foreground_values[channel * pixels + i]).clamp(0.0, 1.0)
                            * 255.0)
                            .round() as u8,
                    );
                }
                rgba.push(alpha_bytes[i]);
            }
            super::desktop::write_frame(
                index,
                native_width,
                native_height,
                Some(&rgba),
                Some(&alpha_bytes),
                Some(&checker_review_bytes(
                    alpha_values,
                    foreground_values,
                    native_width,
                    native_height,
                    args.threads,
                )),
                None,
            )?;
            if let Some(review) = alpha_review.as_mut() {
                review.write(&alpha_bytes)?;
            }
            if let Some(review) = checker_review.as_mut() {
                review.write(&checker_review_bytes(
                    alpha_values,
                    foreground_values,
                    native_width,
                    native_height,
                    args.threads,
                ))?;
            }
            released_work_bytes += release_work_frame(&scratch, "baton", index)?;
            super::desktop::progress("stage2", index + 1, fixture.total_frames)?;
            continue;
        }
        let reference = read_npy(pack_path(root, &fixture.references[index].file)?)?;
        let (ref_alpha, ref_foreground) = decoded(&padding, &reference, args.threads)?;
        let metrics = [
            compare("logits", &logits, &reference, tolerance)?,
            compare("alpha", &alpha, &ref_alpha, tolerance)?,
            compare("foreground_linear", &foreground, &ref_foreground, tolerance)?,
        ];
        for metric in &metrics {
            totals.entry(metric.name.clone()).or_default().push(metric);
        }
        let mut decoded_spot_metrics = Vec::new();
        if let Some(sample) = fixture.decoded_samples.iter().find(|s| s.frame == index) {
            for (name, actual, path) in [
                ("alpha", &alpha, &sample.alpha),
                ("foreground_linear", &foreground, &sample.foreground_linear),
            ] {
                let expected = read_npy(pack_path(root, path)?)?;
                let metric = compare(name, actual, &expected, tolerance)?;
                decoded_spot_totals
                    .entry(name.into())
                    .or_default()
                    .push(&metric);
                decoded_spot_metrics.push(metric);
            }
        }
        let frame_passed = metrics.iter().all(|metric| metric.passed)
            && decoded_spot_metrics.iter().all(|metric| metric.passed);
        frame_metrics.push(serde_json::json!({"frame":index,"metrics":metrics,"independent_decoded_metrics":decoded_spot_metrics}));
        if let Some(review) = alpha_review.as_mut() {
            let bytes: Vec<u8> = alpha.iter().map(|v| (v * 255.0).round() as u8).collect();
            review.write(&bytes)?;
        }
        if let Some(review) = checker_review.as_mut() {
            let alpha = alpha.as_slice().context("contiguous alpha needed")?;
            let fg = foreground
                .as_slice()
                .context("contiguous foreground needed")?;
            let bytes = checker_review_bytes(alpha, fg, native_width, native_height, args.threads);
            review.write(&bytes)?;
        }
        println!("stage2: {}/{}", index + 1, fixture.total_frames);
        if frame_passed {
            released_work_bytes += release_work_frame(&scratch, "baton", index)?;
        }
    }
    components.push(graph.finish()?);
    if let Some(review) = alpha_review {
        review.finish()?;
    }
    if let Some(review) = checker_review {
        review.finish()?;
    }
    if inference {
        let report = serde_json::json!({"schema_version":1,"component":"corridorkey_desktop_inference","execution_success":true,"validation_performed":false,"numerical_passed":null,"frames":fixture.total_frames,"components":components,"precision":"float32","source":fixture.source_video,"native_hw":fixture.native_hw,"shape_hw":fixture.shape_hw,"source_geometry":"native pixels; centered reflect101 padding to multiples of128; crop back to original dimensions","runtime_policy":super::runtime_policy(args),"stage1_window":{"chunk":fixture.chunk,"stride":fixture.stride,"selection":"fixed_export_temporal_contract"},"stage2_batch":1,"elapsed_seconds":started.elapsed().as_secs_f64(),"released_generated_work_bytes":released_work_bytes,"rgba":"frames","alpha":"alpha","known_numeric_note":"Negligible straight-foreground discrepancy remains documented in HANDOFF_NATIVE.md; no numerical gate was changed."});
        fs::write(
            args.report
                .as_ref()
                .context("desktop report path missing")?,
            serde_json::to_vec_pretty(&report)?,
        )?;
        return Ok(());
    }
    let passed = totals.values().all(|m| m.failed_elements == 0)
        && totals.values().all(|m| m.frames == fixture.total_frames)
        && decoded_spot_totals.values().all(|m| m.failed_elements == 0)
        && decoded_spot_totals
            .values()
            .all(|m| m.frames == fixture.decoded_samples.len());
    let report = serde_json::json!({"schema_version":1,"component":"corridorkey_real_video","source_video":fixture.source_video,"native_hw":fixture.native_hw,"shape_hw":fixture.shape_hw,"frames":fixture.total_frames,"chunk":fixture.chunk,"stride":fixture.stride,"hint_mode":fixture.hint_mode,"carry_hint":fixture.carry_hint,"precision":"float32","cuda_tf32":false,"requested_provider":args.provider,"components":components,"outputs":totals,"frame_metrics":frame_metrics,"atol":tolerance.atol,"rtol":tolerance.rtol,"elapsed_seconds":started.elapsed().as_secs_f64(),"reviews_are_lossy":true,"released_generated_work_bytes":released_work_bytes,"failed_frame_batons_retained":true,"passed":passed});
    let path = args
        .report
        .clone()
        .unwrap_or_else(|| output.join("report.json"));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut report = report;
    report["input_provenance"] = input_provenance;
    report["rgb_encoding"] = serde_json::json!("original_cpu_numpy_torch_rgb24_lut");
    report["foundation_area_interpolation"] = serde_json::json!(if args.provider == "cuda" {
        "row_major_then_divide_height_then_width"
    } else {
        "row_major_then_divide_bin_area"
    });
    report["ort_optimization"] = serde_json::json!(args.ort_optimization);
    report["hint_interpolation"] = serde_json::json!(if args.provider == "cuda" {
        "half_pixel_cuda_fma_first"
    } else {
        "half_pixel_cpu_separate"
    });
    report["host_pixel_thread_limit"] = serde_json::json!(args.threads);
    report["cuda_policy"] = serde_json::json!({"arena_limit_mib":args.cuda_memory_mib,"arena_growth":"same_as_requested","memory_pattern":false,"cudnn_search":"heuristic","cudnn_max_workspace":false,"tf32":false,"limit_scope":"provider arena, not total process/device memory"});
    report["decoded_metric_scope"] = serde_json::json!({"all_frames":"decoded from both logit arrays using tested native color/crop helpers","independent_decoded_spot_checks":"compared against original Python runtime crop and color utilities","independent_sample_frames":spot_frames});
    report["independent_decoded_outputs"] = serde_json::to_value(decoded_spot_totals)?;
    fs::write(&path, serde_json::to_string_pretty(&report)?)?;
    println!("real-video pass={passed}, report={}", path.display());
    ensure!(passed, "real clip parity failed; inspect per-frame report");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;

    #[test]
    fn dynamic_foundation_geometry_preserves_native_padding_and_recurrent_tokens() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "brp-ck-dynamic-contract-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        for (name, channels) in [("rvm", 384), ("cradio", 1152), ("moge", 768)] {
            let mut inputs = vec![
                serde_json::json!({"name":"rgb","dtype":"float32","shape":if name=="rvm" {serde_json::json!([1,1,3,"rvm_h","rvm_w"])} else if name=="moge" {serde_json::json!([1,3,"moge_h","moge_w"])} else {serde_json::json!([1,3,"cradio_h","cradio_w"])}}),
            ];
            let mut outputs = vec![
                serde_json::json!({"name":"features","dtype":"float32","shape":[1,channels,"grid_h","grid_w"]}),
            ];
            if name == "rvm" {
                inputs.push(serde_json::json!({"name":"state","dtype":"float32","shape":[1,"rvm_tokens",384]}));
                outputs.push(serde_json::json!({"name":"next_state","dtype":"float32","shape":[1,"rvm_tokens",384]}));
            }
            fs::create_dir_all(root.join(name))?;
            fs::write(
                root.join(name).join("manifest.json"),
                serde_json::to_vec(
                    &serde_json::json!({"schema_version":1,"component":name,"model":"model.onnx","inputs":inputs,"outputs":outputs}),
                )?,
            )?;
        }
        // Independent known patch grids for720p, odd dimensions, portrait,
        //1080p and4K. None of these native images is resized to another profile.
        for (native_h, native_w, padded_h, padded_w, tokens) in [
            (720, 1280, 768, 1280, 241),
            (480, 854, 512, 896, 113),
            (1280, 720, 1280, 768, 241),
            (1080, 1920, 1152, 1920, 541),
            (2160, 3840, 2176, 3840, 2041),
        ] {
            let padding = Padding::new(native_h, native_w, 128)?;
            assert_eq!((padding.height, padding.width), (padded_h, padded_w));
            let rvm = foundation_contract(&root, "rvm", padded_h, padded_w)?;
            assert_eq!(rvm.state, Some(vec![1, tokens, 384]));
            assert_eq!(rvm.rgb, vec![1, 1, 3, padded_h / 4, padded_w / 4]);
            assert_eq!(rvm.features, vec![1, 384, padded_h / 64, padded_w / 64]);
            let cradio = foundation_contract(&root, "cradio", padded_h, padded_w)?;
            assert_eq!(cradio.rgb, vec![1, 3, padded_h / 4, padded_w / 4]);
            let moge = foundation_contract(&root, "moge", padded_h, padded_w)?;
            assert_eq!(moge.rgb, vec![1, 3, padded_h / 32 * 7, padded_w / 32 * 7]);
        }
        let file = root.join("rvm/manifest.json");
        let mut fixed: serde_json::Value = serde_json::from_slice(&fs::read(&file)?)?;
        fixed["inputs"][0]["shape"] = serde_json::json!([1, 1, 3, 288, 480]);
        fs::write(&file, serde_json::to_vec(&fixed)?)?;
        assert!(foundation_contract(&root, "rvm", 768, 1280).is_err());
        assert!(foundation_contract(&root, "rvm", 1152, 1920).is_ok());
        fixed["inputs"][0]["shape"] = serde_json::json!([1, 1, 3, "unknown_h", "rvm_w"]);
        fs::write(&file, serde_json::to_vec(&fixed)?)?;
        assert!(foundation_contract(&root, "rvm", 768, 1280).is_err());
        fs::remove_dir_all(root)?;
        Ok(())
    }

    fn boundary_values() -> Vec<f32> {
        // Odd length crosses every chunk boundary, and includes both signed
        // transfer-function branches and scene-linear values above display white.
        let values = [
            -32.0,
            -16.0,
            -1.0,
            -0.18,
            -0.04045,
            -0.008,
            -0.003_130_8,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            0.003_130_8,
            0.003_130_800_3,
            0.008,
            0.04045,
            0.040_450_003,
            0.18,
            1.0,
            16.0,
            32.0,
        ];
        (0..196_613).map(|i| values[i % values.len()]).collect()
    }

    fn assert_bits_equal(expected: &[f32], actual: &[f32]) {
        assert_eq!(expected.len(), actual.len());
        for (i, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
            assert_eq!(expected.to_bits(), actual.to_bits(), "element {i}");
        }
    }

    #[test]
    fn host_color_maps_are_bit_exact_across_chunk_boundaries() {
        let values = boundary_values();
        let encoded: Vec<_> = values
            .iter()
            .map(|&value| linear_to_asinh(srgb_to_linear(value)))
            .collect();
        let decoded: Vec<_> = values.iter().map(|&value| asinh_to_linear(value)).collect();
        for workers in [0, 1, 2, 3, 4, 17] {
            assert_bits_equal(
                &encoded,
                &map_host_pixels(&values, workers, |value| {
                    linear_to_asinh(srgb_to_linear(value))
                }),
            );
            assert_bits_equal(
                &decoded,
                &map_host_pixels(&values, workers, asinh_to_linear),
            );
        }
        assert!(map_host_pixels(&[], 4, asinh_to_linear).is_empty());
    }

    #[test]
    fn checker_rows_are_bit_exact_at_tile_and_thread_boundaries() {
        let (width, height) = (131, 137);
        let pixels = width * height;
        let values = boundary_values();
        let alpha: Vec<_> = (0..pixels).map(|i| (i % 257) as f32 / 256.0).collect();
        let foreground = &values[..pixels * 3];
        // Retain the original loop as the independent serial reference.
        let mut expected = vec![0u8; pixels * 3];
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let background = if (x / 64 + y / 64).is_multiple_of(2) {
                    0.75
                } else {
                    0.5
                };
                for c in 0..3 {
                    expected[3 * i + c] = (linear_to_srgb(
                        foreground[c * pixels + i] * alpha[i] + background * (1.0 - alpha[i]),
                    )
                    .clamp(0.0, 1.0)
                        * 255.0)
                        .round() as u8;
                }
            }
        }
        for workers in [1, 2, 3, 4, 17] {
            assert_eq!(
                expected,
                checker_review_bytes(&alpha, foreground, width, height, workers)
            );
        }
    }

    /// CPU-only developer benchmark, deliberately excluded from normal CI.
    /// Run with --release --ignored --nocapture; consumes one real 1080p frame.
    #[test]
    #[ignore = "requires staged real 1080p RGB; measures host CPU work only"]
    fn benchmark_real_1080_host_pixels() -> Result<()> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let path = root.join("artifacts/highres-cases/shorter-1080/frames.rgb24");
        let (width, height) = (1920, 1080);
        let rgb = RawClip::open(&path, 50, height, width)?.read(0)?;
        let foreground: Vec<_> = rgb.iter().map(|&v| srgb_to_linear(v)).collect();
        let encoded: Vec<_> = foreground.iter().map(|&v| linear_to_asinh(v)).collect();
        let alpha: Vec<_> = rgb[..width * height].to_vec();
        let median_seconds = |work: &dyn Fn()| {
            work();
            let mut times = Vec::new();
            for _ in 0..3 {
                let started = Instant::now();
                work();
                times.push(started.elapsed().as_secs_f64());
            }
            times.sort_by(f64::total_cmp);
            times[1]
        };
        let mut results = Vec::new();
        for workers in [1, 4, 8] {
            let encode = median_seconds(&|| {
                black_box(map_host_pixels(black_box(&rgb), workers, |value| {
                    linear_to_asinh(srgb_to_linear(value))
                }));
            });
            let decode = median_seconds(&|| {
                black_box(map_host_pixels(
                    black_box(&encoded),
                    workers,
                    asinh_to_linear,
                ));
            });
            let checker = median_seconds(&|| {
                black_box(checker_review_bytes(
                    black_box(&alpha),
                    black_box(&foreground),
                    width,
                    height,
                    workers,
                ));
            });
            results.push(
                serde_json::json!({"threads":workers,"encode_seconds":encode,
                "decode_seconds":decode,"checker_seconds":checker}),
            );
        }
        let report = serde_json::json!({"kind":"host_pixel_cpu_benchmark",
            "source_rgb":path,"source_frames":50,"measured_frame":0,"shape_hw":[height,width],
            "timing":"median of three runs after one warm-up; no neural calls or GPU",
            "runs":results});
        fs::write(
            root.join("artifacts/host-pixel-benchmark.json"),
            serde_json::to_string_pretty(&report)?,
        )?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }
}
