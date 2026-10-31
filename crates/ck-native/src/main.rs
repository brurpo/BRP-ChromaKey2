//! Native-only ONNX fixture runner. Python is used to prepare fixtures, never to run them.
use anyhow::{Context, Result, bail, ensure};
use ndarray::{ArrayD, ArrayViewD};
mod chain;
mod clip_tokenizer;
mod cuda_rmsnorm;
mod desktop;
mod dml_policy;
mod gpu_memory;
mod half_readback;
mod native_attention;
mod native_plan;
mod npy;
mod pipeline;
mod sam31;
mod sam_video;
mod scalar_control;
mod translate;
mod transnet;
mod video;
use npy::{read_npy, write_npy};
use ort::{
    execution_providers::{
        ArenaExtendStrategy, CUDAExecutionProvider, cuda::CuDNNConvAlgorithmSearch,
    },
    session::{Session, builder::GraphOptimizationLevel},
    tensor::TensorElementType,
    value::{DynValue, Tensor, ValueType},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    env, fs,
    path::{Component, Path, PathBuf},
    time::Instant,
};

#[derive(Debug, Deserialize)]
struct Manifest {
    schema_version: u32,
    model: String,
    component: String,
    inputs: Vec<TensorSpec>,
    outputs: Vec<TensorSpec>,
    tolerance: Option<Tolerance>,
}

#[derive(Debug, Deserialize)]
struct TensorSpec {
    name: String,
    #[serde(default)]
    file: String,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct Tolerance {
    atol: f64,
    rtol: f64,
}

#[derive(Debug, Serialize)]
struct Metrics {
    name: String,
    shape: Vec<usize>,
    elements: usize,
    max_abs_error: f64,
    mean_abs_error: f64,
    root_mean_square_error: f64,
    failed_elements: usize,
    passed: bool,
}

#[derive(Debug, Serialize, Clone)]
struct OutputObservation {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    elements: usize,
    finite_elements: usize,
    nonfinite_elements: usize,
    nonzero_elements: usize,
    min: f32,
    max: f32,
    f32_le_bytes_sha256: String,
}

#[derive(Debug, Serialize)]
struct InferenceRun {
    run: usize,
    warmup: bool,
    seconds: f64,
    outputs: Vec<OutputObservation>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum VerificationOutput {
    Compared(Metrics),
    Collected(OutputObservation),
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    component: String,
    requested_provider: String,
    ort_optimization: String,
    cuda_arena_limit_mib: Option<usize>,
    cuda_memory_pattern: Option<bool>,
    runtime: String,
    model_load_seconds: f64,
    inference_seconds: f64,
    measured_inference_seconds: f64,
    warmup_runs: usize,
    measured_runs: usize,
    inference_runs: Vec<InferenceRun>,
    validation_performed: bool,
    atol: Option<f64>,
    rtol: Option<f64>,
    outputs: Vec<VerificationOutput>,
    output_dtypes: BTreeMap<String, String>,
    output_readback_precision: &'static str,
    #[serde(flatten)]
    graph: GraphIdentity,
    profile: Option<String>,
    provider_node_counts: BTreeMap<String, usize>,
    runtime_policy: serde_json::Value,
    gpu_memory_samples: Option<gpu_memory::GpuMemorySamples>,
    gpu_memory_monitor_error: Option<String>,
    sampled_process_gpu_peak_bytes: Option<u64>,
    passed: Option<bool>,
}

#[derive(Debug, Serialize)]
struct GraphIdentity {
    graph_file: PathBuf,
    graph_sha256: String,
    declared_graph_precision: Option<serde_json::Value>,
}

impl GraphIdentity {
    fn read(model: &Path) -> Result<Self> {
        // Hash the ONNX graph only; its external learned data is not read here.
        let graph_sha256 = format!(
            "{:x}",
            Sha256::digest(
                fs::read(model)
                    .with_context(|| format!("reading graph identity {}", model.display()))?
            )
        );
        let manifest = model.with_file_name(
            if model
                .file_name()
                .is_some_and(|name| name == "model.directml.onnx")
            {
                "manifest.directml.json"
            } else {
                "manifest.json"
            },
        );
        let declared_graph_precision = match fs::read(&manifest) {
            Ok(bytes) => {
                let value: serde_json::Value = serde_json::from_slice(&bytes)
                    .with_context(|| format!("reading graph metadata {}", manifest.display()))?;
                value.pointer("/contract/precision").cloned()
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("reading graph metadata {}", manifest.display()));
            }
        };
        Ok(Self {
            graph_file: model.to_owned(),
            graph_sha256,
            declared_graph_precision,
        })
    }
}

struct VerificationMemory {
    monitor: Option<gpu_memory::GpuMemoryMonitor>,
    start_error: Option<String>,
    sidecar: Option<PathBuf>,
}
impl VerificationMemory {
    fn start(args: &Args) -> Self {
        let (monitor, start_error) =
            match gpu_memory::GpuMemoryMonitor::start(&args.provider, Instant::now()) {
                Ok(monitor) => (monitor, None),
                Err(error) => (None, Some(format!("{error:#}"))),
            };
        let sidecar = args
            .report
            .as_ref()
            .map(|path| {
                let mut name = path.as_os_str().to_os_string();
                name.push(".memory.json");
                PathBuf::from(name)
            })
            .or_else(|| {
                args.output
                    .as_ref()
                    .map(|path| path.join("memory-summary.json"))
            });
        Self {
            monitor,
            start_error,
            sidecar,
        }
    }
    fn phase(&self, phase: &str) {
        if let Some(monitor) = &self.monitor {
            monitor.set_phase(phase, None);
        }
    }
    fn observation(
        &self,
    ) -> (
        Option<gpu_memory::GpuMemorySamples>,
        Option<String>,
        Option<u64>,
    ) {
        let samples = self
            .monitor
            .as_ref()
            .map(gpu_memory::GpuMemoryMonitor::summary);
        let error = self
            .start_error
            .clone()
            .or_else(|| samples.as_ref().and_then(|value| value.last_error.clone()));
        let peak = samples
            .as_ref()
            .filter(|value| value.sample_count > 0)
            .map(|value| value.peak_process_usage_bytes);
        (samples, error, peak)
    }
}
impl Drop for VerificationMemory {
    fn drop(&mut self) {
        // Keep optional telemetry on ordinary initialization/inference errors,
        // without replacing their primary error or creating CPU-only outputs.
        if self.monitor.is_none() && self.start_error.is_none() {
            return;
        }
        let Some(path) = &self.sidecar else {
            return;
        };
        let (samples, error, peak) = self.observation();
        let record = serde_json::json!({
            "gpu_memory_samples":samples,"gpu_memory_monitor_error":error,
            "sampled_process_gpu_peak_bytes":peak,
            "measurement_scope":"This process's DXGI local allocations at100ms intervals; sampled, not a guaranteed exact peak"
        });
        let write = || -> Result<()> {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(path, serde_json::to_vec_pretty(&record)?)?;
            Ok(())
        };
        if let Err(error) = write() {
            eprintln!("Optional GPU memory summary could not be saved: {error:#}");
        }
    }
}

#[derive(Default, Clone)]
struct Args {
    command: String,
    manifest: PathBuf,
    models_root: Option<PathBuf>,
    video: Option<PathBuf>,
    ffmpeg: Option<PathBuf>,
    runtime: Option<PathBuf>,
    output: Option<PathBuf>,
    report: Option<PathBuf>,
    provider: String,
    threads: usize,
    cuda_memory_mib: usize,
    dml_memory_arena: bool,
    dml_disable_reuse: bool,
    explicit_low_vram: bool,
    directml_mixed_precision: bool,
    ort_optimization: String,
    diagnostic_windows: Vec<usize>,
    tensor_dumps: String,
    profile: Option<PathBuf>,
    no_profile: bool,
    dimension_overrides: BTreeMap<String, i64>,
    verbose: bool,
    atol: Option<f64>,
    rtol: Option<f64>,
    warmup: usize,
    runs: usize,
    collect: bool,
}

fn args() -> Result<Option<Args>> {
    parse_args(env::args().skip(1))
}

fn parse_args(mut raw: impl Iterator<Item = String>) -> Result<Option<Args>> {
    let command = raw.next().unwrap_or_else(|| "help".into());
    if matches!(command.as_str(), "help" | "--help" | "-h") {
        println!(
            "ck-native desktop --job FILE --events FILE [--start-gate FILE]\nck-native verify --manifest FILE [--runtime LIBRARY] [--provider cpu|directml|cuda]\nck-native pipeline --manifest FILE --models-root DIR\nck-native chain --manifest FILE\nck-native ck-video --manifest FILE --models-root DIR [--video FILE --ffmpeg EXE]\nck-native sam-video --manifest FILE\nck-native transnet --manifest FILE [--video FILE --ffmpeg EXE]\nOptions: --output DIR --report FILE --profile PREFIX | --no-profile\n         --threads N --atol N --rtol N --cuda-memory-mib N --ort-optimization all|none\n         --warmup 0..10 --runs 1..10 --dml-memory-arena on|off --collect (verify only)\n         --diagnostic-windows 7,8,9 (developer CK input/output capture)\n         --tensor-dumps all|masks|none (SAM; failed tensors are always saved)\n\nAll inference is native. Python is used only to prepare developer conversion/reference data.\nGPU provider registration failures are fatal; inspect profiles for provider partitioning.\nDirectML developer commands profile by default; --no-profile disables that trace collection.\n--collect checks output shapes/finite values and writes tensors without a numerical reference or pass claim."
        );
        return Ok(None);
    }
    ensure!(
        matches!(
            command.as_str(),
            "verify"
                | "pipeline"
                | "chain"
                | "ck-video"
                | "sam-video"
                | "transnet"
                | "inspect-model"
        ),
        "unknown command {command}; use --help"
    );
    let mut result = Args {
        command,
        provider: "cpu".into(),
        threads: 4,
        cuda_memory_mib: 8192,
        ort_optimization: "all".into(),
        tensor_dumps: "all".into(),
        runs: 1,
        ..Default::default()
    };
    let mut verify_options = false;
    while let Some(flag) = raw.next() {
        if flag == "--no-profile" {
            result.no_profile = true;
            continue;
        }
        if flag == "--collect" {
            result.collect = true;
            verify_options = true;
            continue;
        }
        let value = raw
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--manifest" => result.manifest = value.into(),
            "--models-root" => result.models_root = Some(value.into()),
            "--video" => result.video = Some(value.into()),
            "--ffmpeg" => result.ffmpeg = Some(value.into()),
            "--runtime" => result.runtime = Some(value.into()),
            "--output" => result.output = Some(value.into()),
            "--report" => result.report = Some(value.into()),
            "--provider" => result.provider = value,
            "--threads" => result.threads = value.parse()?,
            "--cuda-memory-mib" => result.cuda_memory_mib = value.parse()?,
            "--ort-optimization" => result.ort_optimization = value,
            "--diagnostic-windows" => {
                result.diagnostic_windows = value
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<Vec<usize>, _>>()?;
            }
            "--tensor-dumps" => result.tensor_dumps = value,
            "--profile" => result.profile = Some(value.into()),
            "--free-dimension" => {
                let (name, size) = value
                    .split_once('=')
                    .context("--free-dimension expects NAME=POSITIVE_SIZE")?;
                let size: i64 = size.parse()?;
                ensure!(!name.is_empty() && size > 0, "invalid free dimension");
                result.dimension_overrides.insert(name.into(), size);
            }
            "--log-level" => {
                ensure!(
                    matches!(value.as_str(), "verbose" | "warning"),
                    "log level must be verbose or warning"
                );
                result.verbose = value == "verbose";
            }
            "--atol" => result.atol = Some(value.parse()?),
            "--rtol" => result.rtol = Some(value.parse()?),
            "--warmup" => {
                result.warmup = value.parse()?;
                verify_options = true;
            }
            "--runs" => {
                result.runs = value.parse()?;
                verify_options = true;
            }
            "--dml-memory-arena" => {
                ensure!(
                    matches!(value.as_str(), "on" | "off"),
                    "--dml-memory-arena expects on|off"
                );
                result.dml_memory_arena = value == "on";
                verify_options = true;
            }
            _ => bail!("unknown argument {flag}"),
        }
    }
    ensure!(
        !(result.no_profile && result.profile.is_some()),
        "--no-profile and --profile are mutually exclusive"
    );
    ensure!(
        !verify_options || result.command == "verify",
        "--warmup, --runs, --dml-memory-arena and --collect are verify-only options"
    );
    ensure!(result.warmup <= 10, "--warmup must be0..10");
    ensure!((1..=10).contains(&result.runs), "--runs must be1..10");
    ensure!(
        !result.collect || (result.atol.is_none() && result.rtol.is_none()),
        "--collect does not accept numerical tolerance overrides"
    );
    ensure!(
        !result.manifest.as_os_str().is_empty(),
        "--manifest is required"
    );
    ensure!(
        matches!(result.provider.as_str(), "cpu" | "cuda" | "directml"),
        "provider must be cpu, cuda or directml"
    );
    ensure!(result.threads > 0, "--threads must be positive");
    ensure!(
        matches!(result.tensor_dumps.as_str(), "all" | "masks" | "none"),
        "--tensor-dumps must be all, masks or none"
    );
    ensure!(
        matches!(result.ort_optimization.as_str(), "all" | "none"),
        "--ort-optimization must be all or none"
    );
    ensure!(
        result.cuda_memory_mib > 0 && result.cuda_memory_mib.checked_mul(1024 * 1024).is_some(),
        "invalid CUDA arena memory limit"
    );
    if result.provider == "directml" && result.profile.is_none() && !result.no_profile {
        result.profile = Some(
            result
                .output
                .as_deref()
                .unwrap_or(Path::new("."))
                .join("directml-profile"),
        );
    }
    Ok(Some(result))
}

fn pack_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    ensure!(
        !path.as_os_str().is_empty()
            && path
                .components()
                .all(|c| matches!(c, Component::Normal(_) | Component::CurDir)),
        "pack paths must be relative without parent traversal: {relative}"
    );
    Ok(root.join(path))
}

fn load_tensor(root: &Path, spec: &TensorSpec) -> Result<ArrayD<f32>> {
    ensure!(
        spec.dtype == "float32",
        "unsupported dtype {} for {}",
        spec.dtype,
        spec.name
    );
    let path = pack_path(root, &spec.file)?;
    let array: ArrayD<f32> =
        read_npy(&path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        array.shape() == spec.shape,
        "shape mismatch for {}: file {:?}, manifest {:?}",
        spec.name,
        array.shape(),
        spec.shape
    );
    ensure!(
        array.iter().all(|v| v.is_finite()),
        "non-finite fixture tensor {}",
        spec.name
    );
    Ok(array)
}

fn compare(
    name: &str,
    actual: &ArrayD<f32>,
    reference: &ArrayD<f32>,
    tolerance: Tolerance,
) -> Result<Metrics> {
    ensure!(
        actual.shape() == reference.shape(),
        "output {name} shape mismatch: actual {:?}, reference {:?}",
        actual.shape(),
        reference.shape()
    );
    ensure!(!actual.is_empty(), "empty output {name}");
    ensure!(
        actual.iter().all(|v| v.is_finite()),
        "non-finite native output {name}"
    );
    ensure!(
        reference.iter().all(|v| v.is_finite()),
        "non-finite reference output {name}"
    );
    let (mut max, mut sum, mut square, mut failed) = (0.0_f64, 0.0, 0.0, 0);
    for (&a, &r) in actual.iter().zip(reference.iter()) {
        let error = (f64::from(a) - f64::from(r)).abs();
        max = max.max(error);
        sum += error;
        square += error * error;
        failed += usize::from(error > tolerance.atol + tolerance.rtol * f64::from(r).abs());
    }
    Ok(Metrics {
        name: name.into(),
        shape: actual.shape().into(),
        elements: actual.len(),
        max_abs_error: max,
        mean_abs_error: sum / actual.len() as f64,
        root_mean_square_error: (square / actual.len() as f64).sqrt(),
        failed_elements: failed,
        passed: failed == 0,
    })
}

fn profile_counts(path: &str) -> Result<BTreeMap<String, usize>> {
    let events: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let mut counts = BTreeMap::new();
    if let Some(events) = events.as_array() {
        for event in events {
            if let Some(provider) = event.pointer("/args/provider").and_then(|v| v.as_str()) {
                *counts.entry(provider.into()).or_insert(0) += 1;
            }
        }
    }
    Ok(counts)
}

fn initialize_runtime(args: &Args) -> Result<()> {
    if let Some(runtime) = &args.runtime {
        ort::init_from(runtime.to_string_lossy())
            .with_name("BRP CHROMA native proof")
            .commit()?;
    } else {
        ort::init().with_name("BRP CHROMA native proof").commit()?;
    }
    Ok(())
}

fn dml_sync_spinning() -> bool {
    // Developer A/B only: spinning can reduce fence latency while increasing
    // CPU use and power consumption. The portable default remains disabled.
    env::var("BRP_CHROMA_DML_SYNC_SPIN").is_ok_and(|value| value == "1")
}

/// The development-only CPU control operator is registered only when the graph
/// references its domain; canonical packs do not.
fn model_references_custom_domains(model: &Path) -> Result<(bool, bool)> {
    const CONTROL: &[u8] = b"com.brp_chroma.control";
    const NATIVE_ATTENTION: &[u8] = b"com.brp_chroma.experimental";
    let bytes = fs::read(model).with_context(|| {
        format!(
            "reading model for control-domain detection {}",
            model.display()
        )
    })?;
    Ok((
        bytes.windows(CONTROL.len()).any(|window| window == CONTROL),
        bytes
            .windows(NATIVE_ATTENTION.len())
            .any(|window| window == NATIVE_ATTENTION),
    ))
}

/// Developer A/B only: optional override of ORT parallel execution; `None`
/// keeps the portable DirectML policy (sequential execution).
fn dml_parallel_execution() -> Option<bool> {
    env::var("BRP_CHROMA_DML_PARALLEL")
        .ok()
        .map(|value| value == "1")
}

/// Developer A/B only: optional override of ORT memory pattern; `None` keeps
/// the portable DirectML policy (off).
fn dml_memory_pattern() -> Option<bool> {
    env::var("BRP_CHROMA_DML_MEMORY_PATTERN")
        .ok()
        .map(|value| value == "1")
}

/// Developer A/B only: extra ORT session config entries delimited by `;` in
/// `key=value` form. Unset by default, so the portable policy is unchanged.
fn developer_session_config_entries() -> Vec<(String, String)> {
    let Ok(raw) = env::var("BRP_CHROMA_SESSION_CONFIG") else {
        return Vec::new();
    };
    raw.split(';')
        .filter_map(|pair| pair.split_once('='))
        .filter_map(|(key, value)| {
            let (key, value) = (key.trim(), value.trim());
            (!key.is_empty() && !value.is_empty()).then(|| (key.to_owned(), value.to_owned()))
        })
        .collect()
}

fn create_session(args: &Args, model: &Path, profile: Option<&Path>) -> Result<Session> {
    create_session_with(args, model, profile, &[])
}

/// `create_session` plus per-graph session config entries (e.g. disabling DirectML graph fusion).
fn create_session_with(
    args: &Args,
    model: &Path,
    profile: Option<&Path>,
    extra_config: &[(&str, &str)],
) -> Result<Session> {
    let optimization = match args.ort_optimization.as_str() {
        "all" => GraphOptimizationLevel::Level3,
        "none" => GraphOptimizationLevel::Disable,
        _ => bail!("unsupported ORT optimization policy"),
    };
    let mut builder = Session::builder()?
        .with_intra_threads(args.threads)?
        .with_optimization_level(optimization)?;
    if args.verbose {
        builder = builder
            .with_log_level(ort::logging::LogLevel::Verbose)?
            .with_log_verbosity(1)?;
    }
    // Developer A/B only: fixed-dimension hot-path graphs have no symbolic
    // dimensions, so free-dimension overrides must not be applied to them.
    let skip_dimension_overrides =
        env::var("BRP_CHROMA_SKIP_DIM_OVERRIDES").is_ok_and(|value| value == "1");
    if !skip_dimension_overrides {
        for (name, size) in &args.dimension_overrides {
            builder = builder.with_dimension_override(name, *size)?;
        }
    }
    if env::var_os("CK_LOG_SESSIONS").is_some() {
        eprintln!(
            "session {} arena={} threads={} opt={} overrides={:?}",
            model.display(),
            args.dml_memory_arena,
            args.threads,
            args.ort_optimization,
            args.dimension_overrides
        );
    }
    let mut serialize_custom_op_session = false;
    if args.provider == "directml" {
        ensure!(cfg!(windows), "DirectML requires Windows");
        // The development QueryTileControl domain is only registered for graphs
        // that actually reference it. Registering it for every session races on
        // ORT's global custom-op schema map when sessions load concurrently.
        let (custom_control_ops, custom_attention_ops) = model_references_custom_domains(model)?;
        if custom_control_ops {
            builder = builder.with_operators(scalar_control::domain()?)?;
            // Concurrent session creation is safe for stock graphs, but only one
            // session at a time may register the shared custom-op domain.
            serialize_custom_op_session = true;
        }
        // Development graph only. The canonical pack never references this
        // domain. Both existing DLL paths must be explicit; there is no runtime
        // download, implicit system lookup or fallback to different arithmetic.
        if custom_attention_ops {
            let explicit_dll = |name: &str| -> Result<PathBuf> {
                let path = env::var_os(name)
                    .map(PathBuf::from)
                    .with_context(|| format!("experimental native attention requires {name}"))?;
                ensure!(
                    path.is_absolute() && path.is_file(),
                    "{name} must name an existing absolute DLL path"
                );
                Ok(path)
            };
            builder = builder.with_operators(native_attention::domain(
                explicit_dll("BRP_CHROMA_NATIVE_ATTENTION_DLL")?,
                explicit_dll("BRP_CHROMA_NATIVE_ATTENTION_DML_DLL")?,
                ck_desktop_core::selected_device_id()? as i32,
            )?)?;
            serialize_custom_op_session = true;
        }
        builder = builder
            .with_parallel_execution(dml_parallel_execution().unwrap_or(false))?
            .with_memory_pattern(dml_memory_pattern().unwrap_or(false))?
            // DirectML's independent pool rounds transient allocations up to
            // powers of two and retains them. Keep actual per-frame buffers
            // releasable instead of filling VRAM with pooled Loop temporaries.
            .with_config_entry(
                "ep.dml.disable_memory_arena",
                if args.dml_memory_arena { "0" } else { "1" },
            )?
            .with_config_entry(
                "ep.dml.enable_cpu_sync_spinning",
                if dml_sync_spinning() { "1" } else { "0" },
            )?;
        builder =
            dml_policy::register_directml(builder, ck_desktop_core::selected_device_id()? as i32)?;
    }
    if args.provider == "cuda" {
        // CK query Loops execute repeatedly inside one Run. Replaying a global
        // first-run memory pattern retains large loop temporaries on later
        // video windows; let the CUDA arena reuse their individual buffers.
        builder = builder.with_memory_pattern(false)?;
        builder = builder.with_execution_providers([CUDAExecutionProvider::default()
            .with_device_id(ck_desktop_core::selected_device_id()? as i32)
            .with_tf32(false)
            .with_memory_limit(args.cuda_memory_mib * 1024 * 1024)
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
            .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Heuristic)
            .with_conv_max_workspace(false)
            .build()
            .error_on_failure()])?;
        builder = builder.with_operators(cuda_rmsnorm::operator_domain()?)?;
    }
    if let Some(profile) = profile {
        if let Some(parent) = profile.parent() {
            fs::create_dir_all(parent)?;
        }
        builder = builder.with_profiling(profile)?;
    }
    for (key, value) in developer_session_config_entries() {
        builder = builder.with_config_entry(&key, &value)?;
    }
    for (key, value) in extra_config {
        builder = builder.with_config_entry(key, value)?;
    }
    if serialize_custom_op_session {
        static CUSTOM_OP_SESSION_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = CUSTOM_OP_SESSION_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        return Ok(builder.commit_from_file(model)?);
    }
    Ok(builder.commit_from_file(model)?)
}

fn runtime_policy(args: &Args) -> serde_json::Value {
    serde_json::json!({
        "requested_provider": args.provider,
        "requested_device_id": if args.provider == "cpu" { None } else { ck_desktop_core::selected_device_id().ok() },
        "provider_registration_failure": "fatal",
        "profiling_enabled": args.profile.is_some(),
        "profiling_disabled_by_cli": args.no_profile,
        "model_variant": if args.command == "desktop" && args.provider == "directml" && args.directml_mixed_precision {"desktop_directml_mixed_precision"} else {"base"},
        "explicit_low_vram": args.explicit_low_vram,
        "provider_placement_evidence": if args.profile.is_some() {"profile"} else {"not_collected"},
        "directml": (args.provider == "directml").then(|| serde_json::json!({
            "execution": "sequential",
            "memory_pattern": false,
            "memory_arena": args.dml_memory_arena,
            "force_disable_arena_reuse": args.dml_disable_reuse,
            "cpu_sync_spinning": dml_sync_spinning(),
            "memory_limit_bytes": null
        }))
    })
}

fn float_output_dtype(dtype: TensorElementType) -> Result<&'static str> {
    match dtype {
        TensorElementType::Float32 => Ok("float32"),
        TensorElementType::Float16 => Ok("float16"),
        _ => bail!("unsupported ONNX output dtype {dtype:?}; expected float32 or float16"),
    }
}

fn half_array_to_f32(array: ArrayViewD<'_, half::f16>) -> ArrayD<f32> {
    // Every finite binary16 value is exactly representable in binary32.
    // Preserve nonfinite values so the existing numerical gates reject them.
    half_readback::to_f32(array)
}

fn read_output_f32(output: &DynValue) -> Result<(ArrayD<f32>, &'static str)> {
    let ValueType::Tensor { ty, .. } = output.dtype() else {
        bail!(
            "unsupported ONNX output type {:?}; expected a float tensor",
            output.dtype()
        );
    };
    let dtype = float_output_dtype(*ty)?;
    let array = match ty {
        TensorElementType::Float32 => output.try_extract_array::<f32>()?.to_owned(),
        TensorElementType::Float16 => half_array_to_f32(output.try_extract_array::<half::f16>()?),
        _ => unreachable!("float_output_dtype rejects unsupported tensor types"),
    };
    Ok((array, dtype))
}

fn observe_output(
    spec: &TensorSpec,
    actual: &ArrayD<f32>,
    dtype: &str,
) -> Result<OutputObservation> {
    ensure!(
        matches!(spec.dtype.as_str(), "float32" | "float16"),
        "output specification {} must describe a float tensor",
        spec.name
    );
    ensure!(
        actual.shape() == spec.shape,
        "output {} shape differs from manifest: {:?} versus {:?}",
        spec.name,
        actual.shape(),
        spec.shape
    );
    ensure!(!actual.is_empty(), "empty output {}", spec.name);
    let finite_elements = actual.iter().filter(|value| value.is_finite()).count();
    ensure!(
        finite_elements == actual.len(),
        "non-finite native output {}",
        spec.name
    );
    let mut hasher = Sha256::new();
    // Hash the F32 readback bytes in explicit little-endian order. Neither
    // hashing, readback nor statistics belong to the session.run timing.
    let mut bytes = [0u8; 16384];
    let values = actual
        .as_slice()
        .context("output readback must be C-order")?;
    for chunk in values.chunks(bytes.len() / 4) {
        for (value, destination) in chunk.iter().zip(bytes.as_chunks_mut::<4>().0) {
            destination.copy_from_slice(&value.to_le_bytes());
        }
        hasher.update(&bytes[..chunk.len() * 4]);
    }
    Ok(OutputObservation {
        name: spec.name.clone(),
        dtype: dtype.to_owned(),
        shape: actual.shape().to_vec(),
        elements: actual.len(),
        finite_elements,
        nonfinite_elements: actual.len() - finite_elements,
        nonzero_elements: actual.iter().filter(|&&value| value != 0.0).count(),
        min: actual.iter().copied().fold(f32::INFINITY, f32::min),
        max: actual.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        f32_le_bytes_sha256: format!("{:x}", hasher.finalize()),
    })
}

fn verification_tolerance(args: &Args, manifest: &Manifest) -> Result<Option<Tolerance>> {
    if args.collect {
        return Ok(None);
    }
    let reference = manifest.tolerance.context(
        "verification requires reference tolerances; use --collect for reference-free collection",
    )?;
    let tolerance = Tolerance {
        atol: args.atol.unwrap_or(reference.atol),
        rtol: args.rtol.unwrap_or(reference.rtol),
    };
    ensure!(
        tolerance.atol.is_finite()
            && tolerance.rtol.is_finite()
            && tolerance.atol >= 0.0
            && tolerance.rtol >= 0.0,
        "tolerances must be finite and nonnegative"
    );
    Ok(Some(tolerance))
}

fn verification_passed(collect: bool, outputs: &[VerificationOutput]) -> Option<bool> {
    if collect {
        None
    } else {
        Some(!outputs.is_empty() && outputs.iter().all(|output| matches!(output, VerificationOutput::Compared(metric) if metric.passed)))
    }
}

fn verify(args: Args) -> Result<()> {
    let root = args.manifest.parent().unwrap_or(Path::new("."));
    let manifest: Manifest = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    ensure!(
        manifest.schema_version == 1,
        "unsupported manifest schema {}",
        manifest.schema_version
    );
    let tolerance = verification_tolerance(&args, &manifest)?;
    ensure!(
        !manifest.outputs.is_empty(),
        "manifest has no output specifications"
    );
    let output_names: HashSet<_> = manifest
        .outputs
        .iter()
        .map(|value| value.name.as_str())
        .collect();
    ensure!(
        output_names.len() == manifest.outputs.len(),
        "duplicate output specification names"
    );
    let memory = VerificationMemory::start(&args);
    memory.phase("runtime_initialization");
    initialize_runtime(&args)?;
    memory.phase("model_load");
    let model = pack_path(root, &manifest.model)?;
    let graph = GraphIdentity::read(&model)?;
    let load_start = Instant::now();
    let mut session = create_session(&args, &model, args.profile.as_deref())?;
    let model_load_seconds = load_start.elapsed().as_secs_f64();
    let expected_names: HashSet<_> = manifest.inputs.iter().map(|i| i.name.as_str()).collect();
    let model_names: HashSet<_> = session.inputs.iter().map(|i| i.name.as_str()).collect();
    ensure!(
        expected_names.len() == manifest.inputs.len() && expected_names == model_names,
        "fixture input names {:?} do not match ONNX names {:?}",
        expected_names,
        model_names
    );
    let mut inputs: Vec<(String, DynValue)> = Vec::new();
    memory.phase("input_load");
    for spec in &manifest.inputs {
        inputs.push((
            spec.name.clone(),
            Tensor::from_array(load_tensor(root, spec)?)?.into_dyn(),
        ));
    }
    let mut metrics = Vec::new();
    let mut output_dtypes = BTreeMap::new();
    let mut inference_runs = Vec::with_capacity(args.warmup + args.runs);
    let mut inference_seconds = 0.0;
    let mut measured_inference_seconds = 0.0;
    if let Some(dir) = &args.output {
        fs::create_dir_all(dir)?;
    }
    for run in 0..args.warmup + args.runs {
        let warmup = run < args.warmup;
        memory.phase(if warmup {
            "warmup_inference"
        } else {
            "inference"
        });
        // Reborrow the same loaded native tensors. No input decode, allocation,
        // file I/O, output readback or observation work occurs in this timer.
        let borrowed_inputs: Vec<_> = inputs
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect();
        let start = Instant::now();
        let outputs = session.run(borrowed_inputs)?;
        inference_seconds = start.elapsed().as_secs_f64();
        if !warmup {
            measured_inference_seconds += inference_seconds;
        }
        memory.phase("output_observation");
        let mut observations = Vec::with_capacity(manifest.outputs.len());
        let last = run + 1 == args.warmup + args.runs;
        for spec in &manifest.outputs {
            let output = outputs
                .get(&spec.name)
                .with_context(|| format!("ONNX output missing: {}", spec.name))?;
            let (actual, dtype) = read_output_f32(output)
                .with_context(|| format!("reading ONNX output {}", spec.name))?;
            let observation = observe_output(spec, &actual, dtype)?;
            output_dtypes.insert(spec.name.clone(), dtype.to_owned());
            if last {
                if let Some(tolerance) = tolerance {
                    memory.phase("output_validation");
                    let reference = load_tensor(root, spec)?;
                    let metric = compare(&spec.name, &actual, &reference, tolerance)?;
                    println!(
                        "{}: {} max_abs={:.8e} mean_abs={:.8e} failed={}/{}",
                        spec.name,
                        if metric.passed { "PASS" } else { "FAIL" },
                        metric.max_abs_error,
                        metric.mean_abs_error,
                        metric.failed_elements,
                        metric.elements
                    );
                    metrics.push(VerificationOutput::Compared(metric));
                } else {
                    metrics.push(VerificationOutput::Collected(observation.clone()));
                }
                if let Some(dir) = &args.output {
                    ensure!(
                        !spec.name.contains(['/', '\\']) && spec.name != "." && spec.name != "..",
                        "unsafe output name"
                    );
                    write_npy(dir.join(format!("{}.npy", spec.name)), &actual)?;
                }
            }
            observations.push(observation);
        }
        inference_runs.push(InferenceRun {
            run: run + 1,
            warmup,
            seconds: inference_seconds,
            outputs: observations,
        });
        // SessionOutputs and each transient F32 readback are dropped before the
        // next run; only compact observations and last-run output files persist.
        drop(outputs);
    }
    memory.phase("profile_finalize");
    let profile = if args.profile.is_some() {
        Some(session.end_profiling()?)
    } else {
        None
    };
    let provider_node_counts = profile
        .as_deref()
        .map(profile_counts)
        .transpose()?
        .unwrap_or_default();
    let passed = verification_passed(args.collect, &metrics);
    memory.phase("report");
    let (gpu_memory_samples, gpu_memory_monitor_error, sampled_process_gpu_peak_bytes) =
        memory.observation();
    let resolved_policy = runtime_policy(&args);
    let report = Report {
        schema_version: 1,
        component: manifest.component,
        ort_optimization: args.ort_optimization,
        cuda_arena_limit_mib: (args.provider == "cuda").then_some(args.cuda_memory_mib),
        cuda_memory_pattern: (args.provider == "cuda").then_some(false),
        requested_provider: args.provider,
        runtime: ort::info().to_string(),
        model_load_seconds,
        inference_seconds,
        measured_inference_seconds,
        warmup_runs: args.warmup,
        measured_runs: args.runs,
        inference_runs,
        validation_performed: !args.collect,
        atol: tolerance.map(|value| value.atol),
        rtol: tolerance.map(|value| value.rtol),
        outputs: metrics,
        output_dtypes,
        output_readback_precision: "float32",
        graph,
        profile,
        provider_node_counts,
        runtime_policy: resolved_policy,
        gpu_memory_samples,
        gpu_memory_monitor_error,
        sampled_process_gpu_peak_bytes,
        passed,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &args.report {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &json)?;
    }
    println!("load={model_load_seconds:.3}s inference={inference_seconds:.3}s");
    println!("{json}");
    ensure!(
        passed != Some(false),
        "native/reference parity failed (report written)"
    );
    Ok(())
}

fn main() -> Result<()> {
    if env::args().nth(1).as_deref() == Some("desktop") {
        return desktop::run_cli();
    }
    if env::args().nth(1).as_deref() == Some("dml-selftest") {
        return dml_policy::run_cli(&env::args().skip(2).collect::<Vec<_>>());
    }
    if env::args().nth(1).as_deref() == Some("translate") {
        return translate::run_cli(&env::args().skip(2).collect::<Vec<_>>());
    }
    if env::args().nth(1).as_deref() == Some("sam31-track") {
        return sam31::run_cli(&env::args().skip(2).collect::<Vec<_>>());
    }
    if env::args().nth(1).as_deref() == Some("native-plan") {
        return native_plan::run_cli(&env::args().skip(2).collect::<Vec<_>>());
    }
    if let Some(args) = args()? {
        if args.command == "inspect-model" {
            initialize_runtime(&args)?;
            let start = Instant::now();
            let session = create_session(&args, &args.manifest, args.profile.as_deref())?;
            let report = serde_json::json!({
                "session_initialized":true,"inference_performed":false,"validation_performed":false,
                "model":args.manifest,"requested_provider":args.provider,
                "load_seconds":start.elapsed().as_secs_f64(),
                "inputs":session.inputs.iter().map(|v|serde_json::json!({"name":v.name,"type":format!("{:?}",v.input_type)})).collect::<Vec<_>>(),
                "outputs":session.outputs.iter().map(|v|serde_json::json!({"name":v.name,"type":format!("{:?}",v.output_type)})).collect::<Vec<_>>()
            });
            println!("{}", serde_json::to_string_pretty(&report)?);
            return Ok(());
        }
        if args.command == "ck-video" {
            return video::run(&args);
        }
        if args.command == "sam-video" {
            return sam_video::run(&args);
        }
        if args.command == "transnet" {
            return transnet::run(&args);
        }
        if args.command == "chain" {
            return chain::run(&args);
        }
        if args.command == "pipeline" {
            pipeline::run(&args)?;
        } else {
            verify(args)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn tolerance_is_elementwise_and_reference_scaled() {
        let a = array![0.00005, 100.005].into_dyn();
        let r = array![0.0, 100.0].into_dyn();
        assert!(
            compare(
                "x",
                &a,
                &r,
                Tolerance {
                    atol: 0.0001,
                    rtol: 0.0001
                }
            )
            .unwrap()
            .passed
        );
        assert_eq!(
            compare(
                "x",
                &a,
                &r,
                Tolerance {
                    atol: 0.00001,
                    rtol: 0.00001
                }
            )
            .unwrap()
            .failed_elements,
            2
        );
    }
    #[test]
    fn nonfinite_output_never_passes() {
        assert!(
            compare(
                "x",
                &array![f32::NAN].into_dyn(),
                &array![0.0].into_dyn(),
                Tolerance {
                    atol: 1.0,
                    rtol: 1.0
                }
            )
            .is_err()
        );
    }
    #[test]
    fn half_readback_preserves_shape_and_binary32_values_exactly() {
        let values = [
            (0x0000, 0x0000_0000), // Positive zero.
            (0x8000, 0x8000_0000), // Negative zero.
            (0x0001, 0x3380_0000), // Smallest positive subnormal.
            (0x03ff, 0x387f_c000), // Largest subnormal.
            (0x0400, 0x3880_0000), // Smallest positive normal.
            (0x3c00, 0x3f80_0000), // One.
            (0xc200, 0xc040_0000), // Negative three.
            (0x7bff, 0x477f_e000), // Largest finite value.
            (0x7c00, 0x7f80_0000), // Positive infinity.
            (0xfc00, 0xff80_0000), // Negative infinity.
            (0x7e00, 0x7fc0_0000), // Quiet NaN.
            (0xfe00, 0xffc0_0000), // Negative quiet NaN.
        ];
        let array = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3, 4]),
            values
                .iter()
                .map(|(bits, _)| half::f16::from_bits(*bits))
                .collect(),
        )
        .unwrap();
        let converted = half_array_to_f32(array.view());
        assert_eq!(converted.shape(), [3, 4]);
        for (value, (_, expected)) in converted.iter().zip(values) {
            assert_eq!(value.to_bits(), expected);
        }
        // Readback does not sanitize infinity/NaN into a passing finite value.
        assert!(
            compare(
                "half",
                &converted,
                &ArrayD::zeros(converted.raw_dim()),
                Tolerance {
                    atol: 1.0,
                    rtol: 1.0
                }
            )
            .is_err()
        );
    }
    #[test]
    fn output_readback_rejects_other_onnx_dtypes() {
        assert_eq!(
            float_output_dtype(TensorElementType::Float32).unwrap(),
            "float32"
        );
        assert_eq!(
            float_output_dtype(TensorElementType::Float16).unwrap(),
            "float16"
        );
        for dtype in [
            TensorElementType::Bfloat16,
            TensorElementType::Float64,
            TensorElementType::Int64,
            TensorElementType::Bool,
            TensorElementType::Undefined,
        ] {
            assert!(
                float_output_dtype(dtype)
                    .unwrap_err()
                    .to_string()
                    .contains("unsupported")
            );
        }
    }
    #[test]
    fn graph_identity_hashes_graph_only_and_uses_explicit_contract_precision() {
        let root = std::env::temp_dir().join(format!(
            "brp-graph-identity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let model = root.join("model.onnx");
        fs::write(&model, b"abc").unwrap();
        // No adjacent external data is present or required for this metadata read.
        fs::write(root.join("manifest.json"), r#"{"precision":"ignored","contract":{"precision":"mixed_float16_compute_float32_input_float16_output"}}"#).unwrap();
        let identity = GraphIdentity::read(&model).unwrap();
        assert_eq!(identity.graph_file, model);
        assert_eq!(
            identity.graph_sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            identity.declared_graph_precision.unwrap(),
            "mixed_float16_compute_float32_input_float16_output"
        );
        fs::write(
            root.join("manifest.json"),
            r#"{"precision":"do not infer from this field"}"#,
        )
        .unwrap();
        assert!(
            GraphIdentity::read(&model)
                .unwrap()
                .declared_graph_precision
                .is_none()
        );
        fs::remove_file(root.join("manifest.json")).unwrap();
        assert!(
            GraphIdentity::read(&model)
                .unwrap()
                .declared_graph_precision
                .is_none()
        );
        let variant = root.join("model.directml.onnx");
        fs::write(&variant, b"variant").unwrap();
        fs::write(
            root.join("manifest.directml.json"),
            r#"{"contract":{"precision":"mixed_float16_compute_float32_io"}}"#,
        )
        .unwrap();
        assert_eq!(
            GraphIdentity::read(&variant)
                .unwrap()
                .declared_graph_precision
                .unwrap(),
            "mixed_float16_compute_float32_io"
        );
        fs::remove_dir_all(&root).unwrap();
    }
    #[test]
    fn pack_cannot_escape_its_root() {
        assert!(pack_path(Path::new("root"), "../outside.npy").is_err());
        assert!(pack_path(Path::new("root"), "inputs/rgb.npy").is_ok());
    }
    fn parsed(values: &[&str]) -> Result<Args> {
        parse_args(values.iter().map(|value| (*value).to_owned()))?.context("expected command")
    }
    #[test]
    fn directml_developer_profile_default_is_preserved() {
        let args = parsed(&[
            "verify",
            "--manifest",
            "fixture.json",
            "--provider",
            "directml",
        ])
        .unwrap();
        assert_eq!(args.profile, Some(PathBuf::from("./directml-profile")));
        assert_eq!(runtime_policy(&args)["profiling_enabled"], true);
    }
    #[test]
    fn explicit_no_profile_disables_trace_and_placement_claim() {
        let args = parsed(&[
            "verify",
            "--no-profile",
            "--manifest",
            "fixture.json",
            "--provider",
            "directml",
        ])
        .unwrap();
        assert!(args.profile.is_none());
        let policy = runtime_policy(&args);
        assert_eq!(policy["profiling_enabled"], false);
        assert_eq!(policy["profiling_disabled_by_cli"], true);
        assert_eq!(policy["provider_placement_evidence"], "not_collected");
    }
    #[test]
    fn no_profile_conflicts_with_explicit_profile_in_both_orders() {
        for options in [
            ["--no-profile", "--profile", "trace"],
            ["--profile", "trace", "--no-profile"],
        ] {
            let values = ["verify", "--manifest", "fixture.json"]
                .into_iter()
                .chain(options)
                .map(str::to_owned);
            assert!(
                parse_args(values)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("mutually exclusive")
            );
        }
    }
    #[test]
    fn verify_repeat_defaults_preserve_one_call_and_arena_off() {
        let args = parsed(&["verify", "--manifest", "fixture.json"]).unwrap();
        assert_eq!(args.warmup, 0);
        assert_eq!(args.runs, 1);
        assert!(!args.collect && !args.dml_memory_arena);
        let args = parsed(&[
            "verify",
            "--manifest",
            "fixture.json",
            "--warmup",
            "10",
            "--runs",
            "10",
            "--dml-memory-arena",
            "on",
            "--collect",
        ])
        .unwrap();
        assert_eq!((args.warmup, args.runs), (10, 10));
        assert!(args.collect && args.dml_memory_arena);
    }
    #[test]
    fn verify_repeat_limits_and_restrictions_are_enforced() {
        for option in [
            ["--warmup", "11"],
            ["--runs", "0"],
            ["--runs", "11"],
            ["--dml-memory-arena", "yes"],
        ] {
            assert!(
                parsed(&["verify", "--manifest", "fixture.json", option[0], option[1]]).is_err()
            );
        }
        for option in [
            ["--warmup", "0"],
            ["--runs", "1"],
            ["--dml-memory-arena", "off"],
        ] {
            assert!(
                parsed(&[
                    "pipeline",
                    "--manifest",
                    "fixture.json",
                    option[0],
                    option[1]
                ])
                .err()
                .unwrap()
                .to_string()
                .contains("verify-only")
            );
        }
        assert!(parsed(&["pipeline", "--manifest", "fixture.json", "--collect"]).is_err());
        assert!(
            parsed(&[
                "verify",
                "--manifest",
                "fixture.json",
                "--collect",
                "--atol",
                "1"
            ])
            .is_err()
        );
        assert!(
            parsed(&[
                "verify",
                "--manifest",
                "fixture.json",
                "--rtol",
                "1",
                "--collect"
            ])
            .is_err()
        );
    }
    #[test]
    fn collection_requires_no_reference_and_never_claims_parity() {
        let manifest: Manifest = serde_json::from_str(r#"{"schema_version":1,"model":"model.onnx","component":"collection","inputs":[],"outputs":[{"name":"x","dtype":"float32","shape":[1]}]}"#).unwrap();
        let args = parsed(&["verify", "--manifest", "fixture.json", "--collect"]).unwrap();
        assert!(verification_tolerance(&args, &manifest).unwrap().is_none());
        assert_eq!(verification_passed(true, &[]), None);
        assert_eq!(
            serde_json::to_value(verification_passed(true, &[])).unwrap(),
            serde_json::Value::Null
        );
        let normal = parsed(&["verify", "--manifest", "fixture.json"]).unwrap();
        assert!(verification_tolerance(&normal, &manifest).is_err());
        assert_eq!(verification_passed(false, &[]), Some(false));
        assert_eq!(manifest.outputs[0].file, "");
    }
    #[test]
    fn observed_outputs_check_shapes_finite_values_and_hash_f32_le_bytes() {
        let spec = TensorSpec {
            name: "x".into(),
            file: String::new(),
            dtype: "float32".into(),
            shape: vec![3],
        };
        let actual = array![0.0, -0.0, 1.0].into_dyn();
        let observed = observe_output(&spec, &actual, "float16").unwrap();
        assert_eq!(observed.dtype, "float16");
        assert_eq!(observed.shape, [3]);
        assert_eq!(observed.nonzero_elements, 1);
        assert_eq!(observed.finite_elements, 3);
        assert_eq!(observed.nonfinite_elements, 0);
        let bytes: Vec<u8> = [0.0_f32, -0.0, 1.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(
            observed.f32_le_bytes_sha256,
            format!("{:x}", Sha256::digest(bytes))
        );
        assert!(observe_output(&spec, &array![f32::NAN, 0.0, 1.0].into_dyn(), "float32").is_err());
        assert!(observe_output(&spec, &array![1.0].into_dyn(), "float32").is_err());
        let collected = vec![VerificationOutput::Collected(observed)];
        assert_eq!(verification_passed(true, &collected), None);
        assert_eq!(verification_passed(false, &collected), Some(false));
    }
    #[test]
    fn cpu_verification_memory_has_no_sampler_or_side_effect_output() {
        let path =
            std::env::temp_dir().join(format!("brp-cpu-memory-report-{}.json", std::process::id()));
        let args = Args {
            provider: "cpu".into(),
            report: Some(path.clone()),
            ..Default::default()
        };
        let memory = VerificationMemory::start(&args);
        memory.phase("inference");
        let (samples, error, peak) = memory.observation();
        assert!(samples.is_none() && error.is_none() && peak.is_none());
        let sidecar = memory.sidecar.clone().unwrap();
        drop(memory);
        assert!(!sidecar.exists());
        assert!(!path.exists());
    }
}
