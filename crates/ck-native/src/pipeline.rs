//! Python-free composition of the trained CK foundation/stage graphs.
//! The fixture supplies source RGB/hints and reference outputs, never features.
use super::npy::{read_npy, write_npy};
use super::{
    Args, GraphIdentity, Manifest, Metrics, TensorSpec, Tolerance, compare, create_session,
    initialize_runtime, load_tensor, pack_path, profile_counts, read_output_f32,
};
use anyhow::{Context, Result, ensure};
use ck_core::{
    color::{asinh_to_linear, linear_to_asinh, srgb_to_linear},
    hints::{HintCarry, HintQualities},
    spatial::{
        Padding, resize_area_chw, resize_area_chw_cuda_order, resize_bilinear_chw,
        resize_bilinear_chw_cuda_order,
    },
    windows::{BatonAccumulator, WindowPlan},
};
use ndarray::{ArrayD, IxDyn};
use ort::{
    session::{Session, SessionInputValue},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Deserialize)]
struct PipelineManifest {
    schema_version: u32,
    kind: String,
    source_rgb: String,
    native_hw: [usize; 2],
    shape_hw: [usize; 2],
    total_frames: usize,
    chunk: usize,
    stride: usize,
    hint_mode: String,
    initial_hint: Option<String>,
    hint_video: Option<String>,
    initial_hint_quality: f32,
    full_hint_quality: f32,
    carry_hint_quality: f32,
    carry_hint: bool,
    precision: String,
    runtime_storage_rounding: bool,
    normalization_owner: String,
    padding: String,
    outputs: Vec<TensorSpec>,
    tolerance: Tolerance,
}

#[derive(Serialize)]
struct CallTiming {
    call: usize,
    input_tensor_seconds: f64,
    ort_run_seconds: f64,
    output_readback_seconds: f64,
    finite_check_seconds: f64,
    input_host_bytes: u64,
    output_tensor_bytes: u64,
    output_materialized_f32_bytes: u64,
}

#[derive(Default, Serialize)]
struct ExecutionTimings {
    input_tensor_seconds: f64,
    ort_run_seconds: f64,
    output_readback_seconds: f64,
    finite_check_seconds: f64,
    input_host_bytes: u64,
    output_tensor_bytes: u64,
    output_materialized_f32_bytes: u64,
    /// Bounded observations distinguish session first-run work from later runs.
    first_32_calls: Vec<CallTiming>,
}
impl ExecutionTimings {
    fn record(&mut self, call: CallTiming) {
        self.input_tensor_seconds += call.input_tensor_seconds;
        self.ort_run_seconds += call.ort_run_seconds;
        self.output_readback_seconds += call.output_readback_seconds;
        self.finite_check_seconds += call.finite_check_seconds;
        self.input_host_bytes = self.input_host_bytes.saturating_add(call.input_host_bytes);
        self.output_tensor_bytes = self
            .output_tensor_bytes
            .saturating_add(call.output_tensor_bytes);
        self.output_materialized_f32_bytes = self
            .output_materialized_f32_bytes
            .saturating_add(call.output_materialized_f32_bytes);
        if self.first_32_calls.len() < 32 {
            self.first_32_calls.push(call);
        }
    }
}

#[derive(Serialize)]
pub(crate) struct ComponentReport {
    component: String,
    load_seconds: f64,
    inference_seconds: f64,
    calls: usize,
    profile: Option<String>,
    provider_node_counts: BTreeMap<String, usize>,
    directml_memory_arena: Option<bool>,
    output_dtypes: BTreeMap<String, String>,
    output_readback_precision: &'static str,
    session_loads: usize,
    session_reused_calls: usize,
    timing_scope: &'static str,
    #[serde(flatten)]
    execution_timings: ExecutionTimings,
    #[serde(flatten)]
    graph: GraphIdentity,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_plan: Option<crate::native_plan::stage::NativeReport>,
}

pub(crate) struct Graph {
    /// ONNX Runtime session; opened lazily when the native plan backend serves this graph.
    session: Option<Session>,
    native: Option<crate::native_plan::stage::NativeStage>,
    native_report: Option<crate::native_plan::stage::NativeReport>,
    open_args: Args,
    model: PathBuf,
    profile: Option<PathBuf>,
    name: String,
    load_seconds: f64,
    inference_seconds: f64,
    calls: usize,
    profiling: bool,
    require_directml: bool,
    directml_memory_arena: Option<bool>,
    output_dtypes: BTreeMap<String, String>,
    graph: GraphIdentity,
    execution_timings: ExecutionTimings,
}

pub(crate) fn graph_model_path(args: &Args, models_root: &Path, name: &str) -> PathBuf {
    let variant = args.command == "desktop"
        && args.provider == "directml"
        && args.directml_mixed_precision
        && matches!(name, "stage1" | "stage2");
    models_root.join(name).join(if variant {
        "model.directml.onnx"
    } else {
        "model.onnx"
    })
}

impl Graph {
    pub(crate) fn open(args: &Args, models_root: &Path, name: &str) -> Result<Self> {
        let profile = args
            .profile
            .as_ref()
            .map(|prefix| PathBuf::from(format!("{}-{name}", prefix.display())));
        let model = graph_model_path(args, models_root, name);
        let graph = GraphIdentity::read(&model)?;
        let started = Instant::now();
        let native = match crate::native_plan::stage::NativeStage::from_env(name, &args.provider) {
            Ok(native) => native,
            Err(error) => {
                eprintln!("native plan unavailable for {name}, using ONNX Runtime: {error:#}");
                None
            }
        };
        let session = if native.is_some() {
            None
        } else {
            Some(
                create_session(args, &model, profile.as_deref())
                    .with_context(|| format!("loading pipeline component {name}"))?,
            )
        };
        Ok(Self {
            session,
            native,
            native_report: None,
            open_args: args.clone(),
            model,
            profile: profile.clone(),
            name: name.into(),
            load_seconds: started.elapsed().as_secs_f64(),
            inference_seconds: 0.0,
            calls: 0,
            profiling: profile.is_some(),
            require_directml: args.provider == "directml",
            directml_memory_arena: (args.provider == "directml").then_some(args.dml_memory_arena),
            output_dtypes: BTreeMap::new(),
            graph,
            execution_timings: ExecutionTimings::default(),
        })
    }

    pub(crate) fn infer(
        &mut self,
        values: Vec<(&str, ArrayD<f32>)>,
    ) -> Result<BTreeMap<String, ArrayD<f32>>> {
        if let Some(native) = self.native.as_mut() {
            let started = Instant::now();
            match native.infer(&values) {
                Ok(outputs) => {
                    for (name, output) in &outputs {
                        ensure!(
                            output.iter().all(|v| v.is_finite()),
                            "{} nonfinite {name} (native plan)",
                            self.name
                        );
                        self.output_dtypes.insert(name.clone(), "float32".into());
                    }
                    self.inference_seconds += started.elapsed().as_secs_f64();
                    self.calls += 1;
                    return Ok(outputs);
                }
                Err(error) => {
                    // Fall back to ONNX Runtime for the rest of this graph's lifetime.
                    eprintln!(
                        "native plan failed for {}, falling back to ONNX Runtime: {error:#}",
                        self.name
                    );
                    let mut native = self.native.take().expect("native backend present");
                    native.report.fallback = Some(format!("{error:#}"));
                    self.native_report = Some(native.report.clone());
                    drop(native);
                    let load = Instant::now();
                    self.session = Some(
                        create_session(&self.open_args, &self.model, self.profile.as_deref())
                            .with_context(|| {
                                format!("loading pipeline component {} (fallback)", self.name)
                            })?,
                    );
                    self.load_seconds += load.elapsed().as_secs_f64();
                }
            }
        }
        let input_started = Instant::now();
        let input_host_bytes = values.iter().fold(0_u64, |total, (_, value)| {
            total.saturating_add((value.len() as u64).saturating_mul(4))
        });
        let inputs: Vec<(String, SessionInputValue<'static>)> = values
            .into_iter()
            .map(|(name, value)| Ok((name.to_owned(), Tensor::from_array(value)?.into())))
            .collect::<Result<_>>()?;
        let input_tensor_seconds = input_started.elapsed().as_secs_f64();
        let session = self
            .session
            .as_mut()
            .context("ONNX Runtime session missing")?;
        let names: Vec<String> = session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();
        let started = Instant::now();
        let outputs = session
            .run(inputs)
            .with_context(|| format!("running {}", self.name))?;
        let ort_run_seconds = started.elapsed().as_secs_f64();
        let mut output_readback_seconds = 0.0;
        let mut finite_check_seconds = 0.0;
        let mut output_tensor_bytes = 0_u64;
        let mut output_materialized_f32_bytes = 0_u64;
        let mut owned = BTreeMap::new();
        for name in names {
            let output = outputs
                .get(&name)
                .with_context(|| format!("{} missing {name}", self.name))?;
            let readback_started = Instant::now();
            let (output, dtype) = read_output_f32(output)
                .with_context(|| format!("reading {} output {name}", self.name))?;
            output_readback_seconds += readback_started.elapsed().as_secs_f64();
            output_tensor_bytes = output_tensor_bytes.saturating_add(
                (output.len() as u64).saturating_mul(if dtype == "float16" { 2 } else { 4 }),
            );
            output_materialized_f32_bytes = output_materialized_f32_bytes
                .saturating_add((output.len() as u64).saturating_mul(4));
            self.output_dtypes.insert(name.clone(), dtype.to_owned());
            let finite_started = Instant::now();
            ensure!(
                output.iter().all(|value| value.is_finite()),
                "{} nonfinite {name}",
                self.name
            );
            finite_check_seconds += finite_started.elapsed().as_secs_f64();
            owned.insert(name, output);
        }
        self.inference_seconds += started.elapsed().as_secs_f64();
        self.calls += 1;
        self.execution_timings.record(CallTiming {
            call: self.calls,
            input_tensor_seconds,
            ort_run_seconds,
            output_readback_seconds,
            finite_check_seconds,
            input_host_bytes,
            output_tensor_bytes,
            output_materialized_f32_bytes,
        });
        Ok(owned)
    }

    pub(crate) fn finish(mut self) -> Result<ComponentReport> {
        let profile = match self.session.as_mut() {
            Some(session) if self.profiling => Some(session.end_profiling()?),
            _ => None,
        };
        let native_plan = self
            .native
            .take()
            .map(|n| n.report.clone())
            .or(self.native_report.take());
        let counts = profile
            .as_deref()
            .map(profile_counts)
            .transpose()?
            .unwrap_or_default();
        if self.require_directml && self.profiling && self.calls > 0 && profile.is_some() {
            ensure!(
                counts.get("DmlExecutionProvider").copied().unwrap_or(0) > 0,
                "{} did not execute any DirectML kernels; inspect {}",
                self.name,
                profile.as_deref().unwrap_or("missing provider profile")
            );
        }
        Ok(ComponentReport {
            component: self.name,
            load_seconds: self.load_seconds,
            inference_seconds: self.inference_seconds,
            calls: self.calls,
            profile,
            provider_node_counts: counts,
            directml_memory_arena: self.directml_memory_arena,
            output_dtypes: self.output_dtypes,
            output_readback_precision: "float32",
            session_loads: 1,
            session_reused_calls: self.calls.saturating_sub(1),
            timing_scope: "ORT Run includes provider execution, synchronization, and implicit device transfers; CPU output conversion and finite validation are measured separately. Byte counters describe logical tensor storage, not measured PCIe traffic or provider-internal copies. First 32 call observations are bounded; aggregate counters cover all calls.",
            execution_timings: self.execution_timings,
            graph: self.graph,
            native_plan,
        })
    }
}

fn tensor(shape: &[usize], values: Vec<f32>) -> Result<ArrayD<f32>> {
    Ok(ArrayD::from_shape_vec(IxDyn(shape), values)?)
}

fn raw_tensor(root: &Path, file: &str, shape: &[usize], name: &str) -> Result<ArrayD<f32>> {
    load_tensor(
        root,
        &TensorSpec {
            name: name.into(),
            file: file.into(),
            dtype: "float32".into(),
            shape: shape.into(),
        },
    )
}

fn take(outputs: &mut BTreeMap<String, ArrayD<f32>>, name: &str) -> Result<ArrayD<f32>> {
    outputs
        .remove(name)
        .with_context(|| format!("missing pipeline output {name}"))
}

fn slice_frames(
    values: &[Vec<f32>],
    start: usize,
    end: usize,
    shape: &[usize],
) -> Result<ArrayD<f32>> {
    tensor(
        shape,
        values[start..end].iter().flatten().copied().collect(),
    )
}

fn optional_debug_metric(
    root: &Path,
    filename: &str,
    name: &str,
    actual: &ArrayD<f32>,
    tolerance: Tolerance,
    metrics: &mut Vec<Metrics>,
) -> Result<()> {
    let path = pack_path(root, filename)?;
    if path.is_file() {
        let reference = read_npy(path)?;
        metrics.push(compare(name, actual, &reference, tolerance)?);
    }
    Ok(())
}

pub fn run(args: &Args) -> Result<()> {
    let started = Instant::now();
    let root = args.manifest.parent().unwrap_or(Path::new("."));
    let fixture: PipelineManifest = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    ensure!(
        fixture.schema_version == 1 && fixture.kind == "ck_pipeline_fixture",
        "unsupported pipeline fixture"
    );
    ensure!(
        fixture.precision == "float32" && !fixture.runtime_storage_rounding,
        "initial pipeline proof requires FP32 math/storage; rounded production storage is not yet supported"
    );
    ensure!(
        fixture.normalization_owner == "foundation graphs",
        "foundation graph normalization contract mismatch"
    );
    ensure!(
        fixture.padding == "centered_reflect101_quantum128",
        "unsupported padding contract"
    );
    ensure!(
        matches!(fixture.hint_mode.as_str(), "none" | "first" | "video"),
        "unsupported hint mode"
    );
    ensure!(
        match fixture.hint_mode.as_str() {
            "none" => fixture.initial_hint.is_none() && fixture.hint_video.is_none(),
            "first" => fixture.initial_hint.is_some() && fixture.hint_video.is_none(),
            "video" => fixture.initial_hint.is_none() && fixture.hint_video.is_some(),
            _ => false,
        },
        "hint mode and hint paths disagree"
    );
    ensure!(
        [
            fixture.initial_hint_quality,
            fixture.full_hint_quality,
            fixture.carry_hint_quality
        ]
        .iter()
        .all(|quality| quality.is_finite() && (0.0..=1.0).contains(quality)),
        "hint qualities must be finite in 0..1"
    );
    let output_names: std::collections::HashSet<_> = fixture
        .outputs
        .iter()
        .map(|spec| spec.name.as_str())
        .collect();
    ensure!(
        fixture.outputs.len() == 3
            && output_names
                == std::collections::HashSet::from(["logits", "alpha", "foreground_linear"]),
        "pipeline requires unique logits, alpha and foreground_linear references"
    );
    let models_root = args
        .models_root
        .clone()
        .unwrap_or_else(|| root.parent().unwrap_or(root).to_owned());
    let tolerance = Tolerance {
        atol: args.atol.unwrap_or(fixture.tolerance.atol),
        rtol: args.rtol.unwrap_or(fixture.tolerance.rtol),
    };
    ensure!(
        tolerance.atol.is_finite()
            && tolerance.rtol.is_finite()
            && tolerance.atol >= 0.0
            && tolerance.rtol >= 0.0,
        "tolerances must be finite and nonnegative"
    );
    let [native_height, native_width] = fixture.native_hw;
    let [height, width] = fixture.shape_hw;
    let padding = Padding::new(native_height, native_width, 128)?;
    ensure!(
        [padding.height, padding.width] == fixture.shape_hw,
        "native dimensions do not match model padding"
    );
    let plan = WindowPlan::new(fixture.total_frames, fixture.chunk, fixture.stride)?;
    ensure!(
        plan.windows.len() >= 2,
        "multi-window proof requires at least two windows"
    );
    let source = raw_tensor(
        root,
        &fixture.source_rgb,
        &[fixture.total_frames, 3, native_height, native_width],
        "display_rgb",
    )?;
    ensure!(
        source.iter().all(|value| (0.0..=1.0).contains(value)),
        "initial proof accepts display RGB in 0..1; HDR/linear input requires a separate contract"
    );
    let native_elements = 3 * native_height * native_width;
    let display: Vec<Vec<f32>> = source
        .as_slice()
        .context("contiguous source RGB required")?
        .chunks_exact(native_elements)
        .map(|frame| padding.pad_chw(frame, 3))
        .collect::<ck_core::Result<_>>()?;
    let model_rgb: Vec<Vec<f32>> = display
        .iter()
        .map(|frame| {
            frame
                .iter()
                .map(|&value| linear_to_asinh(srgb_to_linear(value)))
                .collect()
        })
        .collect();
    let mut intermediates = Vec::new();
    optional_debug_metric(
        root,
        "debug/model_rgb.npy",
        "preprocess/model_rgb",
        &tensor(
            &[fixture.total_frames, 3, height, width],
            model_rgb.iter().flatten().copied().collect(),
        )?,
        tolerance,
        &mut intermediates,
    )?;
    initialize_runtime(args)?;
    let mut components = Vec::new();
    let mut features: BTreeMap<String, Vec<Vec<f32>>> = BTreeMap::new();
    let mut feature_channels = BTreeMap::new();
    for name in ["rvm", "cradio", "moge"] {
        println!("pipeline foundation: {name}");
        let manifest: Manifest =
            serde_json::from_slice(&fs::read(models_root.join(name).join("manifest.json"))?)?;
        let spec = manifest
            .outputs
            .iter()
            .find(|output| output.name == "features")
            .context("features contract missing")?;
        ensure!(
            spec.shape.len() == 4
                && spec.shape[0] == 1
                && spec.shape[2..] == [height / 64, width / 64],
            "invalid foundation feature geometry"
        );
        let channels = spec.shape[1];
        feature_channels.insert(name.to_owned(), channels);
        let mut graph = Graph::open(args, &models_root, name)?;
        let mut state = if name == "rvm" {
            let spec = manifest
                .inputs
                .iter()
                .find(|input| input.name == "state")
                .context("RVM state contract missing")?;
            Some(ArrayD::<f32>::zeros(IxDyn(&spec.shape)))
        } else {
            None
        };
        let mut extracted = Vec::new();
        for (index, frame) in display.iter().enumerate() {
            let (out_height, out_width) = if name == "moge" {
                (height * 7 / 32, width * 7 / 32)
            } else {
                (height / 4, width / 4)
            };
            let mut rgb = if name == "rvm" {
                resize_bilinear_chw(frame, 3, height, width, out_height, out_width)?
            } else if args.provider == "cuda" {
                resize_area_chw_cuda_order(frame, 3, height, width, out_height, out_width)?
            } else {
                resize_area_chw(frame, 3, height, width, out_height, out_width)?
            };
            let mut values = Vec::new();
            if name == "rvm" {
                for value in &mut rgb {
                    *value = value.clamp(0.0, 1.0);
                }
                values.push(("rgb", tensor(&[1, 1, 3, out_height, out_width], rgb)?));
                values.push(("state", state.take().context("RVM state missing")?));
            } else {
                values.push(("rgb", tensor(&[1, 3, out_height, out_width], rgb)?));
            }
            let mut output = graph.infer(values)?;
            let feature = take(&mut output, "features")?;
            ensure!(
                feature.shape() == spec.shape,
                "{name} feature output shape mismatch"
            );
            extracted.push(feature.iter().copied().collect());
            if name == "rvm" {
                let next = take(&mut output, "next_state")?;
                optional_debug_metric(
                    root,
                    &format!("debug/rvm_state_{index:03}.npy"),
                    &format!("rvm/state_{index:03}"),
                    &next,
                    tolerance,
                    &mut intermediates,
                )?;
                state = Some(next);
            }
        }
        optional_debug_metric(
            root,
            &format!("debug/{name}_features.npy"),
            &format!("foundation/{name}"),
            &tensor(
                &[fixture.total_frames, channels, height / 64, width / 64],
                extracted.iter().flatten().copied().collect(),
            )?,
            tolerance,
            &mut intermediates,
        )?;
        features.insert(name.to_owned(), extracted);
        components.push(graph.finish()?);
    }
    let initial = fixture
        .initial_hint
        .as_deref()
        .map(|file| raw_tensor(root, file, &[1, height, width], "initial_hint"))
        .transpose()?;
    let video = fixture
        .hint_video
        .as_deref()
        .map(|file| {
            raw_tensor(
                root,
                file,
                &[fixture.total_frames, 1, height, width],
                "hint_video",
            )
        })
        .transpose()?;
    ensure!(
        fixture.hint_mode != "first" || initial.is_some(),
        "first-frame hint missing"
    );
    ensure!(
        fixture.hint_mode != "video" || video.is_some(),
        "hint video missing"
    );
    let initial = initial
        .as_ref()
        .map(|array| array.as_slice().expect("owned contiguous input"));
    let video = video
        .as_ref()
        .map(|array| array.as_slice().expect("owned contiguous input"));
    ensure!(
        initial
            .into_iter()
            .chain(video)
            .flatten()
            .all(|value| (0.0..=1.0).contains(value)),
        "hint masks must be in 0..1"
    );
    let qualities = HintQualities {
        initial: fixture.initial_hint_quality,
        full_video: fixture.full_hint_quality,
        carry: fixture.carry_hint_quality,
    };
    let mut hints = HintCarry::new(height * width)?;
    let mut stage1 = Graph::open(args, &models_root, "stage1")?;
    let mut accumulator = BatonAccumulator::new(512 * (height / 16) * (width / 16))?;
    let mut blended: Vec<Option<Vec<f32>>> = vec![None; fixture.total_frames];
    for window in &plan.windows {
        println!(
            "pipeline window {}: {}..{}",
            window.index, window.start, window.end
        );
        let full_hint =
            video.map(|values| &values[window.start * height * width..window.end * height * width]);
        let batch = hints.build_window(window, full_hint, initial, qualities)?;
        optional_debug_metric(
            root,
            &format!("windows/window_{:03}/inputs/hint.npy", window.index),
            &format!("hint/window_{:03}/mask", window.index),
            &tensor(&[1, window.len(), 1, height, width], batch.masks.clone())?,
            tolerance,
            &mut intermediates,
        )?;
        optional_debug_metric(
            root,
            &format!("windows/window_{:03}/inputs/hint_quality.npy", window.index),
            &format!("hint/window_{:03}/quality", window.index),
            &tensor(&[1, window.len()], batch.quality.clone())?,
            tolerance,
            &mut intermediates,
        )?;
        let mut inputs = vec![
            (
                "rgb",
                slice_frames(
                    &model_rgb,
                    window.start,
                    window.end,
                    &[1, window.len(), 3, height, width],
                )?,
            ),
            (
                "hint",
                tensor(&[1, window.len(), 1, height, width], batch.masks)?,
            ),
            ("hint_quality", tensor(&[1, window.len()], batch.quality)?),
        ];
        for (name, input) in [
            ("moge", "moge_feat"),
            ("cradio", "cradio_feat"),
            ("rvm", "rvm_feat"),
        ] {
            inputs.push((
                input,
                slice_frames(
                    &features[name],
                    window.start,
                    window.end,
                    &[
                        1,
                        window.len(),
                        feature_channels[name],
                        height / 64,
                        width / 64,
                    ],
                )?,
            ));
        }
        let mut output = stage1.infer(inputs)?;
        let baton = take(&mut output, "baton")?;
        ensure!(
            baton.shape() == [1, window.len(), 512, height / 16, width / 16],
            "stage1 baton shape mismatch"
        );
        optional_debug_metric(
            root,
            &format!("windows/window_{:03}/reference/baton.npy", window.index),
            &format!("stage1/window_{:03}/baton", window.index),
            &baton,
            tolerance,
            &mut intermediates,
        )?;
        let auxiliary = take(&mut output, "aux_alpha")?;
        ensure!(
            auxiliary.shape() == [1, window.len(), 1, height / 2, width / 2],
            "stage1 auxiliary shape mismatch"
        );
        if fixture.carry_hint {
            let mut full = Vec::with_capacity(window.len() * height * width);
            for frame in auxiliary
                .as_slice()
                .context("contiguous auxiliary output required")?
                .chunks_exact(height / 2 * (width / 2))
            {
                let upsampled = if args.provider == "cuda" {
                    resize_bilinear_chw_cuda_order(frame, 1, height / 2, width / 2, height, width)?
                } else {
                    resize_bilinear_chw(frame, 1, height / 2, width / 2, height, width)?
                };
                full.extend(upsampled.into_iter().map(|value| value.clamp(0.0, 1.0)));
            }
            hints.commit_auxiliary(window, &full)?;
        }
        for frame in accumulator.push_window(
            window,
            baton.as_slice().context("contiguous baton required")?,
        )? {
            ensure!(
                blended[frame.index].is_none(),
                "baton frame finalized twice"
            );
            blended[frame.index] = Some(frame.values);
        }
    }
    components.push(stage1.finish()?);
    ensure!(
        accumulator.pending_frames() == 0,
        "unfinalized baton frames"
    );
    let blended: Vec<Vec<f32>> = blended
        .into_iter()
        .enumerate()
        .map(|(index, frame)| frame.with_context(|| format!("missing baton frame {index}")))
        .collect::<Result<_>>()?;
    optional_debug_metric(
        root,
        "debug/blended_baton.npy",
        "blend/baton",
        &tensor(
            &[fixture.total_frames, 512, height / 16, width / 16],
            blended.iter().flatten().copied().collect(),
        )?,
        tolerance,
        &mut intermediates,
    )?;
    let mut stage2 = Graph::open(args, &models_root, "stage2")?;
    let mut logits = Vec::with_capacity(fixture.total_frames * 4 * height * width);
    let mut alpha = Vec::with_capacity(fixture.total_frames * native_height * native_width);
    let mut foreground = Vec::with_capacity(fixture.total_frames * native_elements);
    for index in 0..fixture.total_frames {
        let mut output = stage2.infer(vec![
            (
                "rgb",
                tensor(&[1, 1, 3, height, width], model_rgb[index].clone())?,
            ),
            (
                "baton",
                tensor(
                    &[1, 1, 512, height / 16, width / 16],
                    blended[index].clone(),
                )?,
            ),
        ])?;
        let prediction = take(&mut output, "logits")?;
        ensure!(
            prediction.shape() == [1, 1, 4, height, width],
            "stage2 logits shape mismatch"
        );
        let values = prediction
            .as_slice()
            .context("contiguous logits required")?;
        let cropped = padding.crop_chw(values, 4)?;
        alpha.extend(
            cropped[..native_height * native_width]
                .iter()
                .map(|value| value.clamp(0.0, 1.0)),
        );
        foreground.extend(
            cropped[native_height * native_width..]
                .iter()
                .map(|&value| asinh_to_linear(value)),
        );
        logits.extend_from_slice(values);
    }
    components.push(stage2.finish()?);
    let actual = BTreeMap::from([
        (
            "logits",
            tensor(&[fixture.total_frames, 4, height, width], logits)?,
        ),
        (
            "alpha",
            tensor(
                &[fixture.total_frames, 1, native_height, native_width],
                alpha,
            )?,
        ),
        (
            "foreground_linear",
            tensor(
                &[fixture.total_frames, 3, native_height, native_width],
                foreground,
            )?,
        ),
    ]);
    if let Some(directory) = &args.output {
        fs::create_dir_all(directory)?;
    }
    let mut outputs = Vec::new();
    for spec in &fixture.outputs {
        let value = actual
            .get(spec.name.as_str())
            .with_context(|| format!("unsupported pipeline output {}", spec.name))?;
        outputs.push(compare(
            &spec.name,
            value,
            &load_tensor(root, spec)?,
            tolerance,
        )?);
        if let Some(directory) = &args.output {
            write_npy(directory.join(format!("{}.npy", spec.name)), value)?;
        }
    }
    ensure!(
        outputs.len() == 3,
        "pipeline requires all three reference outputs"
    );
    let passed = outputs
        .iter()
        .chain(intermediates.iter())
        .all(|metric| metric.passed);
    let report = serde_json::json!({
        "schema_version": 1, "component": "corridorkey_full_pipeline", "fixture": args.manifest,
        "requested_provider": args.provider, "runtime": ort::info(), "cuda_tf32": false,
        "precision": "float32", "runtime_storage_rounding": false,
        "precision_scope":"host/reference tensors; neural arithmetic is defined by the ONNX graph",
        "reference_precision":"float32",
        "hint_interpolation": if args.provider == "cuda" {"half_pixel_cuda_fma_first"} else {"half_pixel_cpu_separate"},
        "source_frames": fixture.total_frames, "native_hw": fixture.native_hw, "shape_hw": fixture.shape_hw,
        "windows": plan.windows.iter().map(|window| [window.start, window.end]).collect::<Vec<_>>(),
        "elapsed_seconds": started.elapsed().as_secs_f64(), "atol": tolerance.atol, "rtol": tolerance.rtol,
        "components": components, "outputs": outputs, "intermediates": intermediates, "passed": passed,
        "scope": "trained-model native multi-window integration on a synthetic clip; not production quality/performance qualification",
    });
    let mut report = report;
    report["ort_optimization"] = serde_json::json!(args.ort_optimization);
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &args.report {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &json)?;
    }
    println!("{json}");
    ensure!(
        passed,
        "full native pipeline parity failed; inspect output/intermediate metrics"
    );
    Ok(())
}

#[cfg(test)]
mod timing_tests {
    use super::*;

    #[test]
    fn execution_timing_samples_stay_bounded_and_totals_cover_every_call() {
        let mut timings = ExecutionTimings::default();
        for call in 1..=80 {
            timings.record(CallTiming {
                call,
                input_tensor_seconds: 0.25,
                ort_run_seconds: 2.0,
                output_readback_seconds: 0.5,
                finite_check_seconds: 0.125,
                input_host_bytes: 1024,
                output_tensor_bytes: 512,
                output_materialized_f32_bytes: 1024,
            });
        }
        assert_eq!(timings.first_32_calls.len(), 32);
        assert_eq!(timings.first_32_calls.last().unwrap().call, 32);
        assert_eq!(timings.input_tensor_seconds, 20.0);
        assert_eq!(timings.ort_run_seconds, 160.0);
        assert_eq!(timings.output_readback_seconds, 40.0);
        assert_eq!(timings.finite_check_seconds, 10.0);
        assert_eq!(timings.input_host_bytes, 81920);
        assert_eq!(timings.output_tensor_bytes, 40960);
        assert_eq!(timings.output_materialized_f32_bytes, 81920);
    }
}
