//! Production streaming/staged CK scheduling. Golden validation remains in the
//! parent module and never passes through allocation-recovery fallbacks.
use super::*;
use ck_core::window_policy::{
    MemoryEstimate, WindowChoice, WindowMode, WindowRequest, resolve_window, retry_choices,
};
use ck_desktop_core::{CkSettings, CkWindowMode, FrameRange, JobArtifacts, JobRequest, Provider};
use std::{collections::VecDeque, error::Error, fmt, process::ChildStdout};

const GPU_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
const DISK_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;
const ESTIMATE: MemoryEstimate = MemoryEstimate {
    normal_scale: 1.5,
    normal_extra_gib: 0.0,
    low_vram_scale: 2.0,
    low_vram_extra_gib: 0.0,
};

#[derive(Debug)]
struct MemoryCapacity(String);
impl fmt::Display for MemoryCapacity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl Error for MemoryCapacity {}

#[derive(Debug)]
struct CapacityGuard {
    reason: &'static str,
    message: String,
}
impl fmt::Display for CapacityGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}
impl Error for CapacityGuard {}

fn allocation_oom(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        if cause.is::<MemoryCapacity>() {
            return true;
        }
        if let Some(io) = cause.downcast_ref::<std::io::Error>() {
            return io.kind() == std::io::ErrorKind::OutOfMemory;
        }
        // Only an ORT allocator failure can trigger a provider retry. Context
        // strings can contain arbitrary source/model filenames.
        let Some(ort) = cause.downcast_ref::<ort::Error>() else {
            return false;
        };
        if matches!(
            ort.code(),
            ort::ErrorCode::NoSuchFile
                | ort::ErrorCode::NoModel
                | ort::ErrorCode::InvalidGraph
                | ort::ErrorCode::InvalidProtobuf
        ) {
            return false;
        }
        let message = ort.to_string().to_ascii_lowercase();
        if message
            .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .any(|token| matches!(token, "8007000e" | "0x8007000e" | "e_outofmemory"))
        {
            return true;
        }
        [
            "out of memory",
            "cuda_error_out_of_memory",
            "cudnn_status_alloc_failed",
            "std::bad_alloc",
            "failed to allocate memory",
            "memory allocation failed",
        ]
        .iter()
        .any(|needle| message.contains(needle))
    })
}

#[derive(Clone, Debug, Serialize)]
struct ArenaReusePolicy {
    enabled: bool,
    source: &'static str,
    estimated_required_gib: Option<f64>,
    usable_budget_gib: Option<f64>,
    budget_query_error: Option<String>,
}

fn stage2_arena_policy(
    args: &Args,
    low_vram: bool,
    padding: Padding,
    budget: Option<f64>,
    developer_arena: bool,
    developer_staged_arena: bool,
) -> ArenaReusePolicy {
    let staged_mixed = args.provider == "directml" && args.directml_mixed_precision && low_vram;
    let required = staged_mixed
        .then(|| 1.5 + 5.0 * (padding.height as f64 * padding.width as f64 / 2_211_840.0));
    let (enabled, source) = if args.provider != "directml" {
        (false, "not_directml")
    } else if args.dml_disable_reuse {
        (false, "disabled_after_allocation_oom")
    } else if args.explicit_low_vram {
        (false, "explicit_low_vram")
    } else if developer_arena && (!low_vram || developer_staged_arena) {
        (true, "explicit_developer_override")
    } else if let Some(required) = required {
        if budget.is_some_and(|budget| budget >= required) {
            (true, "staged_mixed_live_budget")
        } else {
            (
                false,
                if budget.is_some() {
                    "insufficient_live_budget"
                } else {
                    "live_budget_unavailable"
                },
            )
        }
    } else {
        (false, "default_off")
    };
    ArenaReusePolicy {
        enabled,
        source,
        estimated_required_gib: required,
        usable_budget_gib: budget,
        budget_query_error: None,
    }
}

fn resolve_stage2_arena(args: &Args, low_vram: bool, padding: Padding) -> ArenaReusePolicy {
    let (memory, error) = if args.provider == "directml"
        && args.directml_mixed_precision
        && low_vram
        && !args.dml_disable_reuse
        && !args.explicit_low_vram
    {
        memory_snapshot(args)
    } else {
        (None, None)
    };
    let mut policy = stage2_arena_policy(
        args,
        low_vram,
        padding,
        memory.as_ref().map(usable_budget),
        std::env::var("BRP_CHROMA_STAGE2_ARENA").is_ok_and(|value| value == "1"),
        std::env::var("BRP_CHROMA_STAGE2_STAGED_ARENA").is_ok_and(|value| value == "1"),
    );
    policy.budget_query_error = error;
    policy
}

fn stage2_args(args: &Args, policy: &ArenaReusePolicy) -> Args {
    let mut stage2 = args.clone();
    stage2.dml_memory_arena = policy.enabled;
    stage2
}

fn stage1_arena_policy(
    args: &Args,
    low_vram: bool,
    padding: Padding,
    actual_frames: usize,
    budget: Option<f64>,
    developer_arena: bool,
) -> ArenaReusePolicy {
    let pixels = padding.height as f64 * padding.width as f64;
    let staged_mixed = args.provider == "directml" && args.directml_mixed_precision && low_vram;
    let measured_domain = (1..=8).contains(&actual_frames) && pixels <= 2_211_840.0;
    let required = (staged_mixed && measured_domain)
        .then(|| 3.0 + 9.0 * (pixels / 2_211_840.0) * (actual_frames as f64 / 8.0));
    let (enabled, source) = if args.provider != "directml" {
        (false, "not_directml")
    } else if args.dml_disable_reuse {
        (false, "disabled_after_allocation_oom")
    } else if args.explicit_low_vram {
        (false, "explicit_low_vram")
    } else if developer_arena {
        (true, "explicit_developer_override")
    } else if let Some(required) = required {
        if budget.is_some_and(|budget| budget >= required) {
            (true, "staged_mixed_live_budget")
        } else {
            (
                false,
                if budget.is_some() {
                    "insufficient_live_budget"
                } else {
                    "live_budget_unavailable"
                },
            )
        }
    } else {
        (
            false,
            if staged_mixed {
                "outside_measured_domain"
            } else {
                "default_off"
            },
        )
    };
    ArenaReusePolicy {
        enabled,
        source,
        estimated_required_gib: required,
        usable_budget_gib: budget,
        budget_query_error: None,
    }
}

fn resolve_stage1_arena(
    args: &Args,
    low_vram: bool,
    padding: Padding,
    actual_frames: usize,
) -> ArenaReusePolicy {
    let (memory, error) = if args.provider == "directml"
        && args.directml_mixed_precision
        && low_vram
        && !args.dml_disable_reuse
        && !args.explicit_low_vram
    {
        memory_snapshot(args)
    } else {
        (None, None)
    };
    let mut policy = stage1_arena_policy(
        args,
        low_vram,
        padding,
        actual_frames,
        memory.as_ref().map(usable_budget),
        std::env::var("BRP_CHROMA_STAGE1_ARENA").is_ok_and(|value| value == "1"),
    );
    policy.budget_query_error = error;
    policy
}

fn balanced_staged_reuse(
    args: &Args,
    request: WindowRequest,
    choice: WindowChoice,
    padding: Padding,
    total_frames: usize,
    budget: Option<f64>,
) -> WindowChoice {
    let actual_frames = choice.chunk.min(total_frames);
    if request.mode != WindowMode::Balanced
        || choice.low_vram
        || !stage1_arena_policy(args, true, padding, actual_frames, budget, false).enabled
    {
        return choice;
    }
    // Staged reuse pays off for ONNX Runtime Stage1/Stage2 (arena reuse). The native plan
    // backend has no arenas and is faster resident; keep the budget-checked normal window.
    if native_plan_serves(args) {
        return choice;
    }
    // Preserve the selected Hann window exactly. Only residency changes.
    WindowChoice {
        low_vram: true,
        source: "balanced_staged_reuse",
        estimated_peak_gib: memory_estimate(args).peak_gib(
            (padding.native_width as f64 * padding.native_height as f64 / 1e6).max(0.1),
            actual_frames,
            true,
        ),
        ..choice
    }
}

/// Auto temporal window (28 October 2026). Larger windows cost more per frame on both
/// backends (temporal attention grows with T^2), so extra memory never grows the window:
/// start at the validated 8 frames, shrink to a minimum of 4 while resident does not fit,
/// then fall back to staged (sequential) inference at 8, 6, 4. Custom/Wide are unchanged.
const AUTO_WINDOWS: [usize; 3] = [8, 6, 4];

/// Native static-plan process peak (GiB) for a padded geometry, calibrated on CARTOLAS
/// 720p/1080p/4K 16-frame jobs and the Stage1 window sweep (RTX 5070 Ti, 28-29 October 2026):
/// resident foundation sessions 4.1 GiB at <= 1080p (4K: ~4.7), staged foundation pass (one
/// model at a time) 3.3 GiB (4K: 3.7), Stage1 executor 1.35 + 0.06*MP*T, Stage2 executor
/// 0.68 + 0.14*MP, batched-init cap 0.5, margin 0.25. Measured peaks: 1080p T8 resident 7.46,
/// 720p T8 resident 6.69, 4K T4 resident 9.88 / staged 5.14 GiB.
fn native_peak_gib(padding: Padding, frames: usize, staged: bool) -> f64 {
    let mp = padding.height as f64 * padding.width as f64 / 1e6;
    let above = (mp - 2.21).max(0.0);
    let stage1 = 1.35 + 0.06 * mp * frames as f64;
    let stage2 = 0.68 + 0.14 * mp;
    let native = stage1 + stage2 + 0.5;
    0.25 + if staged {
        (3.3 + 0.1 * above).max(native)
    } else {
        4.1 + 0.15 * above + native
    }
}

/// Largest native Stage1 window ORT-DirectML can fold: kernel creation fails above roughly
/// 4e8 masked-attention elements per block (windows * (256*T)^2); 1080p passes T12, fails T16.
fn native_window_limit(padding: Padding) -> usize {
    let windows = padding.height.div_ceil(256) * padding.width.div_ceil(256);
    (1..=64)
        .take_while(|&t| windows as f64 * (256.0 * t as f64).powi(2) <= 3.8e8)
        .last()
        .unwrap_or(1)
}

fn app_auto_window(
    args: &Args,
    request: WindowRequest,
    choice: WindowChoice,
    padding: Padding,
    total_frames: usize,
    budget: Option<f64>,
    temporal_max: usize,
) -> WindowChoice {
    if request.mode != WindowMode::Balanced {
        return choice;
    }
    auto_window_choice(
        native_plan_serves(args),
        args,
        request,
        padding,
        total_frames,
        budget,
        temporal_max,
    )
}

#[allow(clippy::too_many_arguments)]
fn auto_window_choice(
    native: bool,
    args: &Args,
    request: WindowRequest,
    padding: Padding,
    total_frames: usize,
    budget: Option<f64>,
    temporal_max: usize,
) -> WindowChoice {
    let mp = (padding.native_width as f64 * padding.native_height as f64 / 1e6).max(0.1);
    let limit = if native {
        native_window_limit(padding).max(4)
    } else {
        usize::MAX
    };
    let estimate = |chunk: usize, staged: bool| {
        let frames = chunk.min(total_frames);
        if native {
            native_peak_gib(padding, frames, staged)
        } else {
            memory_estimate(args).peak_gib(mp, frames, staged)
        }
    };
    let candidates: Vec<usize> = AUTO_WINDOWS
        .into_iter()
        .map(|c| c.min(temporal_max))
        .filter(|&c| c <= limit)
        .collect();
    let pick = |chunk: usize, staged: bool, source: &'static str| WindowChoice {
        chunk,
        stride: (chunk / 2).max(1),
        low_vram: staged,
        estimated_peak_gib: estimate(chunk, staged),
        budget_gib: budget,
        source,
    };
    let Some(budget) = budget else {
        // Unknown budget: keep the default window, staged for safety.
        return pick(
            candidates.first().copied().unwrap_or(4),
            true,
            "auto_unknown_budget",
        );
    };
    if !request.low_vram {
        for &chunk in &candidates {
            if estimate(chunk, false) <= budget {
                return pick(
                    chunk,
                    false,
                    if native {
                        "auto_native_resident"
                    } else {
                        "auto_resident"
                    },
                );
            }
        }
    }
    for &chunk in &candidates {
        if estimate(chunk, true) <= budget {
            return pick(
                chunk,
                true,
                if native {
                    "auto_native_staged"
                } else {
                    "auto_staged"
                },
            );
        }
    }
    pick(
        candidates.last().copied().unwrap_or(4),
        true,
        "auto_minimum_window",
    )
}

/// Whether the native static-plan backend will serve both Stage1 and Stage2 of this job.
fn native_plan_serves(args: &Args) -> bool {
    ["stage1", "stage2"].iter().all(|stage| {
        crate::native_plan::stage::NativeStage::from_env(stage, &args.provider)
            .is_ok_and(|native| native.is_some())
    })
}

fn memory_estimate(args: &Args) -> MemoryEstimate {
    MemoryEstimate {
        normal_scale: if args.provider == "directml" && args.directml_mixed_precision {
            1.75
        } else {
            ESTIMATE.normal_scale
        },
        ..ESTIMATE
    }
}

#[derive(Clone, Copy, Debug)]
struct AttemptPolicy {
    window: WindowChoice,
    disable_reuse: bool,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct ReuseInAttempt {
    stage1: bool,
    stage2: bool,
}
impl fmt::Display for ReuseInAttempt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DirectML buffer reuse requested: Stage1={}, Stage2={}",
            self.stage1, self.stage2
        )
    }
}
impl Error for ReuseInAttempt {}

fn retry_without_reuse(
    policy: AttemptPolicy,
    error: &anyhow::Error,
    pending: &mut VecDeque<AttemptPolicy>,
) -> bool {
    // anyhow's typed contexts are available through its own downcast API;
    // std::error::Error::source traverses the wrapped provider error instead.
    let reused = error
        .downcast_ref::<ReuseInAttempt>()
        .is_some_and(|reuse| reuse.stage1 || reuse.stage2);
    let provider_error = error.chain().any(|cause| cause.is::<ort::Error>());
    if policy.disable_reuse || !reused || !provider_error || !allocation_oom(error) {
        return false;
    }
    // A pool failure must not reduce temporal context. Once disabled, retain
    // the safe arena-OFF policy through every remaining bounded recovery step.
    for next in pending.iter_mut() {
        next.disable_reuse = true;
    }
    pending.push_front(AttemptPolicy {
        disable_reuse: true,
        ..policy
    });
    true
}

fn reserved<T>(elements: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|error| MemoryCapacity(format!("Host tensor allocation failed: {error}")))?;
    Ok(values)
}

fn choice_json(choice: WindowChoice, args: &Args) -> serde_json::Value {
    let estimate = memory_estimate(args);
    serde_json::json!({
        "chunk":choice.chunk,"stride":choice.stride,"low_vram":choice.low_vram,
        "estimated_peak_gib":choice.estimated_peak_gib,"usable_budget_gib":choice.budget_gib,
        "selection":choice.source,
        "estimate_basis":"profile-specific conservative multipliers on upstream BF16 peak tables; not an exact measured peak or device guarantee",
        "estimate_profile":if args.provider=="directml" && args.directml_mixed_precision {"directml_mixed_precision"} else {"base_float32"},
        "normal_estimate_scale":estimate.normal_scale,"staged_estimate_scale":estimate.low_vram_scale
        ,"explicit_low_vram":args.explicit_low_vram
    })
}

fn memory_snapshot(args: &Args) -> (Option<crate::gpu_memory::GpuMemorySnapshot>, Option<String>) {
    match crate::gpu_memory::snapshot(&args.provider) {
        Ok(memory) => (memory, None),
        Err(error) => (None, Some(format!("{error:#}"))),
    }
}

fn usable_budget(memory: &crate::gpu_memory::GpuMemorySnapshot) -> f64 {
    // The estimate includes resident models. Compare its TOTAL process peak
    // with the TOTAL WDDM process budget, not the free remainder after loading.
    memory.local_budget_bytes.saturating_sub(GPU_RESERVE_BYTES) as f64 / 1024_f64.powi(3)
}

fn gpu_budget_guard(
    choice: WindowChoice,
    memory: &crate::gpu_memory::GpuMemorySnapshot,
) -> Result<()> {
    // Explicit Custom settings retain the original force-window semantics.
    // Estimates only guide automatic modes; actual allocation failures and
    // measured host capacity still use the recovery ladder for Custom.
    if choice.source == "custom" {
        return Ok(());
    }
    let current_budget = usable_budget(memory);
    ensure!(
        choice.estimated_peak_gib <= current_budget,
        CapacityGuard {
            reason: "memory_budget_reselection",
            message: format!(
                "The live WDDM process budget changed: estimated total {:.2} GiB exceeds the current usable {:.2} GiB",
                choice.estimated_peak_gib, current_budget
            )
        }
    );
    Ok(())
}

#[cfg(windows)]
fn host_available_bytes() -> Option<u64> {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut memory = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    unsafe { GlobalMemoryStatusEx(&mut memory).ok()? };
    Some(memory.ullAvailPhys)
}
#[cfg(not(windows))]
fn host_available_bytes() -> Option<u64> {
    None
}

fn host_preflight(padding: Padding, frames: usize) -> Result<()> {
    let required = padding
        .height
        .checked_mul(padding.width)
        .and_then(|value| value.checked_mul(frames))
        .and_then(|value| value.checked_mul(12 * 4))
        .context("host window tensor size overflow")? as u64;
    if let Some(available) = host_available_bytes() {
        ensure!(
            required <= available.saturating_mul(3) / 5,
            CapacityGuard {
                reason: "host_capacity",
                message: format!(
                    "Window needs approximately {} MiB of host tensors; only {} MiB of RAM is currently available",
                    required / 1024 / 1024,
                    available / 1024 / 1024
                )
            }
        );
    }
    Ok(())
}

struct PhaseTrace {
    file: File,
    started: Instant,
    peak_usage: u64,
    source_start: u64,
    monitor: Option<crate::gpu_memory::GpuMemoryMonitor>,
    monitor_start_error: Option<String>,
    aggregate_path: PathBuf,
}
impl PhaseTrace {
    fn open(output: &Path, range: &FrameRange, started: Instant, provider: &str) -> Result<Self> {
        let (monitor, monitor_start_error) =
            match crate::gpu_memory::GpuMemoryMonitor::start(provider, started) {
                Ok(monitor) => (monitor, None),
                Err(error) => (None, Some(format!("{error:#}"))),
            };
        Ok(Self {
            file: File::create(output.join("memory-phases.jsonl"))?,
            started,
            peak_usage: 0,
            source_start: range.start,
            monitor,
            monitor_start_error,
            aggregate_path: output.join("memory-summary.json"),
        })
    }
    fn record(&mut self, args: &Args, phase: &str, current: usize, total: usize) -> Result<()> {
        ensure!(
            current < total,
            "runtime phase frame lies outside its source interval"
        );
        let source_frame = self
            .source_start
            .checked_add(current as u64)
            .context("runtime phase source index overflow")?;
        if let Some(monitor) = &self.monitor {
            monitor.set_phase(phase, Some(source_frame));
        }
        let (memory, error) = memory_snapshot(args);
        if let Some(memory) = &memory {
            self.peak_usage = self.peak_usage.max(memory.current_usage_bytes);
        }
        serde_json::to_writer(
            &mut self.file,
            &serde_json::json!({
                "phase":phase,"position":current,"total":total,
                "source_frame":source_frame,"position_is_zero_based":true,
                "elapsed_seconds":self.started.elapsed().as_secs_f64(),
                "gpu_memory":memory,"gpu_memory_error":error
            }),
        )?;
        self.file.write_all(b"\n")?;
        self.file.flush()?;
        let usage = memory
            .as_ref()
            .map(|memory| {
                format!(
                    "; process GPU {:.0} MiB / WDDM budget {:.0} MiB ({:.0} MiB free)",
                    memory.current_usage_bytes as f64 / 1048576.0,
                    memory.local_budget_bytes as f64 / 1048576.0,
                    memory.free_budget_bytes() as f64 / 1048576.0
                )
            })
            .unwrap_or_default();
        crate::desktop::progress_message("runtime_phase", current, total, format!("{phase}{usage}"))
    }
    fn samples(&self) -> Option<crate::gpu_memory::GpuMemorySamples> {
        self.monitor
            .as_ref()
            .map(crate::gpu_memory::GpuMemoryMonitor::summary)
    }
    fn sampled_peak(&self) -> u64 {
        self.peak_usage.max(
            self.samples()
                .map_or(0, |samples| samples.peak_process_usage_bytes),
        )
    }
}
impl Drop for PhaseTrace {
    fn drop(&mut self) {
        // Aggregate only: no unbounded100ms sample log is written. Persist on
        // ordinary failures too, after NN/output workers have unwound.
        let summary = serde_json::json!({
            "endpoint_peak_process_usage_bytes":self.peak_usage,
            "sampled_process_peak_bytes":self.sampled_peak(),
            "background_samples":self.samples(),
            "monitor_start_error":self.monitor_start_error,
            "scope":"This process's DXGI local allocations; sampled, not a guaranteed exact peak"
        });
        if let Ok(bytes) = serde_json::to_vec_pretty(&summary)
            && let Err(error) = fs::write(&self.aggregate_path, bytes)
        {
            eprintln!("Optional GPU memory summary could not be saved: {error}");
        }
    }
}

#[cfg(windows)]
fn disk_available_bytes(directory: &Path) -> Result<Option<u64>> {
    use std::os::windows::ffi::OsStrExt;
    use windows::{Win32::Storage::FileSystem::GetDiskFreeSpaceExW, core::PCWSTR};
    let directory: Vec<u16> = directory.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut free = 0;
    unsafe { GetDiskFreeSpaceExW(PCWSTR(directory.as_ptr()), Some(&mut free), None, None) }
        .context("querying available scratch disk capacity")?;
    Ok(Some(free))
}
#[cfg(not(windows))]
fn disk_available_bytes(_directory: &Path) -> Result<Option<u64>> {
    Ok(None)
}

fn disk_preflight(directory: &Path, scratch_bytes: u64) -> Result<()> {
    if let Some(available) = disk_available_bytes(directory)? {
        let required = scratch_bytes
            .checked_add(DISK_RESERVE_BYTES)
            .context("scratch disk estimate overflow")?;
        ensure!(
            available >= required,
            "Insufficient scratch disk capacity: need {} MiB plus a 1024 MiB reserve; {} MiB available. Free disk space or choose a shorter range",
            scratch_bytes / 1024 / 1024,
            available / 1024 / 1024
        );
    }
    Ok(())
}

fn staged_scratch_bound(
    contracts: &BTreeMap<&str, ResolvedFoundation>,
    p: Padding,
    frames: usize,
) -> Result<u64> {
    let features = contracts
        .values()
        .try_fold(0u64, |sum, contract| {
            let elements = contract
                .features
                .iter()
                .try_fold(1u64, |n, &dim| n.checked_mul(dim as u64))?;
            sum.checked_add(elements.checked_mul(4)?.checked_add(4096)?)
        })
        .context("foundation scratch size overflow")?;
    let baton = (p.height as u64 / 16)
        .checked_mul(p.width as u64 / 16)
        .and_then(|n| n.checked_mul(512 * 4))
        .and_then(|n| n.checked_add(4096))
        .context("baton scratch size overflow")?;
    features
        .checked_add(baton)
        .and_then(|n| n.checked_mul(frames as u64))
        .context("scene scratch size overflow")
}

struct Decoder {
    child: Child,
    output: Option<ChildStdout>,
    stderr: PathBuf,
    frame_bytes: usize,
    frames: usize,
    read: usize,
    hash: Sha256,
    finished: bool,
}
impl Decoder {
    fn open(job: &JobRequest, range: &FrameRange, work: &Path, label: &str) -> Result<Self> {
        job.media.verify_source_unchanged()?;
        let stderr = work.join(format!("decode-{label}.stderr.log"));
        let mut command = Command::new(&job.runtime.ffmpeg);
        hidden(&mut command);
        let mut child = command
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-xerror",
                "-i",
            ])
            .arg(&job.media.path)
            .args(["-map", "0:v:0", "-vf"])
            .arg(format!(
                "trim=start_frame={}:end_frame={},setpts=PTS-STARTPTS,format=rgb24",
                range.start, range.end_exclusive
            ))
            .args(["-an", "-sn", "-dn", "-fps_mode", "passthrough", "-frames:v"])
            .arg((range.end_exclusive - range.start).to_string())
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::from(File::create(&stderr)?))
            .spawn()
            .context("starting exact-frame native decoder")?;
        let output = child.stdout.take().context("decoder stdout missing")?;
        let frame_bytes = (job.media.width as usize)
            .checked_mul(job.media.height as usize)
            .and_then(|value| value.checked_mul(3))
            .context("source frame size overflow")?;
        Ok(Self {
            child,
            output: Some(output),
            stderr,
            frame_bytes,
            frames: usize::try_from(range.end_exclusive - range.start)?,
            read: 0,
            hash: Sha256::new(),
            finished: false,
        })
    }
    fn next(&mut self) -> Result<Vec<u8>> {
        ensure!(self.read < self.frames, "decoder frame range exhausted");
        let mut bytes = reserved(self.frame_bytes)?;
        bytes.resize(self.frame_bytes, 0);
        let read = self
            .output
            .as_mut()
            .context("decoder is closed")?
            .read_exact(&mut bytes);
        if let Err(error) = read {
            let mut stderr = String::new();
            if let Ok(file) = File::open(&self.stderr) {
                let _ = file.take(32 * 1024).read_to_string(&mut stderr);
            }
            return Err(error).with_context(|| {
                format!(
                    "native source decode ended before the exact requested frame count: {stderr}"
                )
            });
        }
        self.hash.update(&bytes);
        self.read += 1;
        Ok(bytes)
    }
    fn finish(mut self) -> Result<String> {
        ensure!(
            self.read == self.frames,
            "source decoder was not fully consumed"
        );
        let mut extra = [0u8; 1];
        ensure!(
            self.output
                .as_mut()
                .context("decoder output missing")?
                .read(&mut extra)?
                == 0,
            "native decoder returned extra source frames"
        );
        drop(self.output.take());
        let status = self.child.wait()?;
        self.finished = true;
        if !status.success() {
            let mut message = String::new();
            File::open(&self.stderr)?
                .take(32 * 1024)
                .read_to_string(&mut message)?;
            anyhow::bail!("native source decode failed: {message}");
        }
        Ok(format!("{:x}", self.hash.clone().finalize()))
    }
}
impl Drop for Decoder {
    fn drop(&mut self) {
        if !self.finished {
            drop(self.output.take());
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn planar(bytes: &[u8], width: usize, height: usize, encoded: bool) -> Result<Vec<f32>> {
    let pixels = width
        .checked_mul(height)
        .context("source pixel count overflow")?;
    ensure!(
        bytes.len() == pixels * 3,
        "packed source frame length differs"
    );
    let mut values = reserved(bytes.len())?;
    values.resize(bytes.len(), 0.0);
    for (pixel, rgb) in bytes.as_chunks::<3>().0.iter().enumerate() {
        for channel in 0..3 {
            values[channel * pixels + pixel] = if encoded {
                ck_core::rgb8_encoding::encode(rgb[channel])
            } else {
                f32::from(rgb[channel]) / 255.0
            };
        }
    }
    Ok(values)
}

type OutputStats = (usize, Option<f64>);

struct Render {
    sender: Option<std::sync::mpsc::SyncSender<(usize, ArrayD<f32>)>>,
    worker: Option<std::thread::JoinHandle<Result<OutputStats>>>,
    padding: Padding,
}
impl Render {
    fn open(
        args: &Args,
        job: &JobRequest,
        padding: Padding,
        total: usize,
        started: Instant,
    ) -> Result<Self> {
        let mut writer = OutputWriter::open(args, job, padding, total)?;
        writer.started = started;
        // One waiting frame plus the frame being written. Backpressure bounds
        // host memory while PNG/review encoding overlaps the next GPU run.
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::Builder::new()
            .name("ck-output".into())
            .spawn(move || {
                while let Ok((frame, logits)) = receiver.recv() {
                    writer.frame(frame, logits)?;
                }
                writer.finish()
            })
            .context("starting bounded native output writer")?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
            padding,
        })
    }
    fn frame(
        &mut self,
        graph: &mut Graph,
        frame: usize,
        rgb: Vec<f32>,
        baton: Vec<f32>,
    ) -> Result<()> {
        let p = self.padding;
        let mut result = graph.infer(vec![
            ("rgb", tensor(&[1, 1, 3, p.height, p.width], rgb)?),
            (
                "baton",
                tensor(&[1, 1, 512, p.height / 16, p.width / 16], baton)?,
            ),
        ])?;
        let logits = take(&mut result, "logits")?;
        ensure!(
            logits.shape() == [1, 1, 4, p.height, p.width],
            "stage2 native geometry differs"
        );
        if self
            .sender
            .as_ref()
            .context("output queue is closed")?
            .send((frame, logits))
            .is_err()
        {
            self.join()?;
            anyhow::bail!("native output writer stopped before the scene completed");
        }
        Ok(())
    }
    fn join(&mut self) -> Result<OutputStats> {
        drop(self.sender.take());
        self.worker
            .take()
            .context("output writer already joined")?
            .join()
            .map_err(|_| anyhow::anyhow!("native output writer panicked"))?
    }
    fn finish(mut self) -> Result<OutputStats> {
        self.join()
    }
}
impl Drop for Render {
    fn drop(&mut self) {
        // Retries may only emit memory_retry after the previous writer and
        // its FrameReady events have stopped, including on inference errors.
        if self.worker.is_some() {
            let _ = self.join();
        }
    }
}

struct OutputWriter {
    padding: Padding,
    threads: usize,
    alpha: Option<Review>,
    checker: Option<Review>,
    rendered: usize,
    first_frame_seconds: Option<f64>,
    started: Instant,
    total: usize,
    validation_logits: Option<PathBuf>,
    source_start: u64,
}
impl OutputWriter {
    fn open(args: &Args, job: &JobRequest, padding: Padding, total: usize) -> Result<Self> {
        let output = args.output.as_ref().context("render output missing")?;
        let fps = format!("{}/{}", job.media.fps.num, job.media.fps.den);
        let source_start = match &job.kind {
            ck_desktop_core::JobKind::CorridorKey { range, .. } => range.start,
            _ => anyhow::bail!("CorridorKey output writer requires a CorridorKey job"),
        };
        Ok(Self {
            padding,
            threads: args.threads,
            rendered: 0,
            first_frame_seconds: None,
            started: Instant::now(),
            total,
            validation_logits: validation_raw_enabled().then(|| output.join("validation-logits")),
            source_start,
            alpha: Some(Review::open(
                &job.runtime.ffmpeg,
                &output.join("alpha.mp4"),
                padding.native_width,
                padding.native_height,
                &fps,
                true,
            )?),
            checker: Some(Review::open(
                &job.runtime.ffmpeg,
                &output.join("checker.mp4"),
                padding.native_width,
                padding.native_height,
                &fps,
                false,
            )?),
        })
    }
    fn frame(&mut self, frame: usize, logits: ArrayD<f32>) -> Result<()> {
        let p = self.padding;
        if let Some(directory) = &self.validation_logits {
            fs::create_dir_all(directory)?;
            disk_preflight(
                directory,
                (logits.len() as u64).saturating_mul(4).saturating_add(4096),
            )?;
            let absolute = self
                .source_start
                .checked_add(frame as u64)
                .context("validation source frame overflow")?;
            write_npy(directory.join(format!("{absolute:06}.npy")), &logits)?;
        }
        let (alpha, foreground) = decoded(&p, &logits, self.threads)?;
        let alpha = alpha.as_slice().context("contiguous alpha required")?;
        let foreground = foreground
            .as_slice()
            .context("contiguous foreground required")?;
        let mut alpha_bytes = reserved(alpha.len())?;
        alpha_bytes.extend(alpha.iter().map(|value| (value * 255.0).round() as u8));
        let pixels = alpha.len();
        let mut rgba = reserved(pixels * 4)?;
        for (i, &a) in alpha_bytes.iter().enumerate() {
            for channel in 0..3 {
                rgba.push(
                    (linear_to_srgb(foreground[channel * pixels + i]).clamp(0.0, 1.0) * 255.0)
                        .round() as u8,
                );
            }
            rgba.push(a);
        }
        let checker = checker_review_bytes(
            alpha,
            foreground,
            p.native_width,
            p.native_height,
            self.threads,
        );
        crate::desktop::write_frame(
            frame,
            p.native_width,
            p.native_height,
            Some(&rgba),
            Some(&alpha_bytes),
            Some(&checker),
            None,
        )?;
        if let Some(review) = self.alpha.as_mut() {
            review.write(&alpha_bytes)?;
        }
        if let Some(review) = self.checker.as_mut() {
            review.write(&checker)?;
        }
        self.rendered += 1;
        self.first_frame_seconds
            .get_or_insert_with(|| self.started.elapsed().as_secs_f64());
        crate::desktop::progress("stage2", frame + 1, self.total)?;
        Ok(())
    }
    fn finish(self) -> Result<(usize, Option<f64>)> {
        if let Some(review) = self.alpha {
            review.finish()?;
        }
        if let Some(review) = self.checker {
            review.finish()?;
        }
        Ok((self.rendered, self.first_frame_seconds))
    }
}

fn validation_raw_enabled() -> bool {
    std::env::var("BRP_CHROMA_VALIDATION_RAW").is_ok_and(|value| value == "1")
}

struct CachedFrame {
    index: usize,
    rgb: Vec<f32>,
    features: BTreeMap<&'static str, ArrayD<f32>>,
}
#[derive(Default)]
struct FrameCache {
    frames: VecDeque<CachedFrame>,
    peak_frames: usize,
    peak_bytes: usize,
}
impl FrameCache {
    fn push(&mut self, frame: CachedFrame) {
        self.frames.push_back(frame);
        self.peak_frames = self.peak_frames.max(self.frames.len());
        let bytes = self
            .frames
            .iter()
            .map(|frame| {
                (frame.rgb.len() + frame.features.values().map(ArrayD::len).sum::<usize>()) * 4
            })
            .sum();
        self.peak_bytes = self.peak_bytes.max(bytes);
    }
    fn get(&self, index: usize) -> Result<&CachedFrame> {
        let first = self
            .frames
            .front()
            .context("source window cache is empty")?
            .index;
        let frame = self
            .frames
            .get(
                index
                    .checked_sub(first)
                    .context("source frame was already finalized")?,
            )
            .context("source frame not decoded into window")?;
        ensure!(
            frame.index == index,
            "source window cache lost frame identity"
        );
        Ok(frame)
    }
    fn take(&mut self, index: usize) -> Result<CachedFrame> {
        ensure!(
            self.frames
                .front()
                .is_some_and(|frame| frame.index == index),
            "finalized source frames must be consumed in original order"
        );
        self.frames.pop_front().context("source window cache empty")
    }
}

fn foundation_input(
    display: &[f32],
    name: &str,
    padding: Padding,
    args: &Args,
) -> Result<Vec<f32>> {
    let (h, w) = if name == "moge" {
        (padding.height / 32 * 7, padding.width / 32 * 7)
    } else {
        (padding.height / 4, padding.width / 4)
    };
    if name == "rvm" {
        resize_bilinear_chw(display, 3, padding.height, padding.width, h, w).map_err(Into::into)
    } else if args.provider == "cuda" {
        resize_area_chw_cuda_order(display, 3, padding.height, padding.width, h, w)
            .map_err(Into::into)
    } else {
        resize_area_chw(display, 3, padding.height, padding.width, h, w).map_err(Into::into)
    }
}

fn foundation_frame(
    graph: &mut Graph,
    name: &str,
    contract: &ResolvedFoundation,
    state: &mut Option<ArrayD<f32>>,
    display: &[f32],
    padding: Padding,
    args: &Args,
) -> Result<ArrayD<f32>> {
    let mut inputs = vec![(
        "rgb",
        tensor(
            &contract.rgb,
            foundation_input(display, name, padding, args)?,
        )?,
    )];
    if name == "rvm" {
        inputs.push((
            "state",
            state.take().context("RVM recurrent state missing")?,
        ));
    }
    let mut result = graph.infer(inputs)?;
    let feature = take(&mut result, "features")?;
    ensure!(
        feature.shape() == contract.features,
        "foundation source geometry mismatch"
    );
    if name == "rvm" {
        let next = take(&mut result, "next_state")?;
        ensure!(
            Some(next.shape()) == contract.state.as_deref(),
            "RVM state source geometry mismatch"
        );
        *state = Some(next);
    }
    Ok(feature)
}

fn stage1_args(args: &Args, frames: usize, policy: &ArenaReusePolicy) -> Result<Args> {
    let mut specialized = args.clone();
    specialized.dml_memory_arena = policy.enabled;
    specialized
        .dimension_overrides
        .insert("frames".into(), i64::try_from(frames)?);
    Ok(specialized)
}

/// Automatic session-load overlap policy. The portable DirectML mixed-precision
/// profile loads the Stage1 session while the staged foundation pass runs and
/// prefetches the Stage2 session while the Stage1 windows run. Explicit Low VRAM
/// keeps the sequential minimum-memory order, and both overlaps are disabled for
/// the resident-streaming mode (which already loads Stage2 up front).
#[derive(Clone, Debug, Serialize)]
struct SessionLoadOverlap {
    parallel_stage1_load: bool,
    stage2_prefetch: bool,
    usable_budget_gib: Option<f64>,
    reason: &'static str,
}

fn session_load_overlap(
    args: &Args,
    choice: &WindowChoice,
    usable_budget_gib: Option<f64>,
) -> SessionLoadOverlap {
    let eligible =
        args.provider == "directml" && args.directml_mixed_precision && !args.explicit_low_vram;
    let serial = std::env::var("BRP_CHROMA_SERIAL_LOAD").is_ok_and(|value| value == "1");
    let no_prefetch =
        std::env::var("BRP_CHROMA_NO_STAGE2_PREFETCH").is_ok_and(|value| value == "1");
    let parallel_stage1_load = eligible && !serial;
    // Prefetch keeps the Stage2 session resident during the Stage1 windows, so
    // it needs headroom beyond the staged Stage1 peak. The live WDDM budget is
    // authoritative; when it cannot be queried the portable order is retained.
    let stage2_prefetch = parallel_stage1_load
        && choice.low_vram
        && !no_prefetch
        && usable_budget_gib.is_some_and(|budget| budget >= 13.0);
    let reason = if !eligible {
        "directml_auto_mixed_precision_only"
    } else if serial {
        "developer_serial_override"
    } else if !choice.low_vram {
        "resident_streaming_loads_stage2_up_front"
    } else {
        "overlapped_auto_staged"
    };
    SessionLoadOverlap {
        parallel_stage1_load,
        stage2_prefetch,
        usable_budget_gib,
        reason,
    }
}

fn validation_stage1_enabled() -> bool {
    std::env::var("BRP_CHROMA_VALIDATION_STAGE1").is_ok_and(|value| value == "1")
}

/// Opt-in numerical evidence from the actual adaptive pipeline, before Hann
/// blending and hint carry. Hashes preserve every F32 bit (including -0) while
/// avoiding large tensor dumps. This CPU work is excluded from ordinary timing
/// runs by leaving BRP_CHROMA_VALIDATION_STAGE1 unset.
fn write_stage1_observation(
    output: &Path,
    window: &ck_core::windows::Window,
    baton: &ArrayD<f32>,
    auxiliary: &ArrayD<f32>,
) -> Result<()> {
    let observations = [("baton", baton), ("aux_alpha", auxiliary)]
        .into_iter()
        .map(|(name, values)| {
            crate::observe_output(
                &crate::TensorSpec {
                    name: name.to_owned(),
                    file: String::new(),
                    dtype: "float32".to_owned(),
                    shape: values.shape().to_vec(),
                },
                values,
                "float32",
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let directory = output.join("validation-stage1");
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!("window-{:06}.json", window.index));
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| format!("creating Stage1 evidence {}", path.display()))?;
    serde_json::to_writer_pretty(
        file,
        &serde_json::json!({
            "schema_version":1,
            "development_only":true,
            "window_index":window.index,
            "frame_index_space":"job_local_zero_based_half_open",
            "start":window.start,
            "end_exclusive":window.end,
            "finalize_before":window.finalize_before,
            "readback_dtype":"float32",
            "scope":"Actual adaptive Stage1 outputs before Hann blending and auxiliary hint carry; observation overhead is not an ordinary render benchmark.",
            "outputs":observations,
        }),
    )?;
    Ok(())
}

fn stage1_window(
    graph: &mut Graph,
    window: &ck_core::windows::Window,
    cache: &FrameCache,
    contracts: &BTreeMap<&str, ResolvedFoundation>,
    scratch: Option<&Path>,
    hints: &mut HintCarry,
    pixel_context: (Padding, &Args),
) -> Result<ArrayD<f32>> {
    let (p, args) = pixel_context;
    let mut rgb = reserved(window.len() * 3 * p.height * p.width)?;
    for frame in window.start..window.end {
        rgb.extend_from_slice(&cache.get(frame)?.rgb);
    }
    let video_hints = if crate::desktop::hint_is_video() {
        let mut values = reserved(window.len() * p.height * p.width)?;
        for frame in window.start..window.end {
            let mask = crate::desktop::sam_hint(frame, p.native_width, p.native_height)?
                .context("SAM video hint missing")?;
            values.extend(p.pad_chw(&mask, 1)?);
        }
        Some(values)
    } else {
        None
    };
    let initial = if window.start == 0 && !crate::desktop::hint_is_video() {
        crate::desktop::sam_hint(0, p.native_width, p.native_height)?
            .map(|mask| p.pad_chw(&mask, 1))
            .transpose()?
    } else {
        None
    };
    let quality = crate::desktop::hint_quality();
    let hint = hints.build_window(
        window,
        video_hints.as_deref(),
        initial.as_deref(),
        HintQualities {
            carry: 0.95,
            full_video: quality,
            initial: quality,
        },
    )?;
    let mut inputs = vec![
        (
            "rgb",
            tensor(&[1, window.len(), 3, p.height, p.width], rgb)?,
        ),
        (
            "hint",
            tensor(&[1, window.len(), 1, p.height, p.width], hint.masks)?,
        ),
        ("hint_quality", tensor(&[1, window.len()], hint.quality)?),
    ];
    for (name, input) in [
        ("moge", "moge_feat"),
        ("cradio", "cradio_feat"),
        ("rvm", "rvm_feat"),
    ] {
        let channels = contracts[name].features[1];
        let mut values = reserved(window.len() * channels * (p.height / 64) * (p.width / 64))?;
        for frame in window.start..window.end {
            if let Some(scratch) = scratch {
                let feature = read_npy(frame_path(scratch, name, frame))?;
                values.extend(feature.iter().copied());
            } else {
                values.extend(
                    cache
                        .get(frame)?
                        .features
                        .get(name)
                        .context("cached foundation feature missing")?
                        .iter()
                        .copied(),
                );
            }
        }
        inputs.push((
            input,
            tensor(
                &[1, window.len(), channels, p.height / 64, p.width / 64],
                values,
            )?,
        ));
    }
    let mut result = graph.infer(inputs)?;
    let baton = take(&mut result, "baton")?;
    ensure!(
        baton.shape() == [1, window.len(), 512, p.height / 16, p.width / 16],
        "stage1 temporal/source geometry differs from requested window"
    );
    let auxiliary = take(&mut result, "aux_alpha")?;
    ensure!(
        auxiliary.shape() == [1, window.len(), 1, p.height / 2, p.width / 2],
        "stage1 auxiliary geometry mismatch"
    );
    if validation_stage1_enabled() {
        write_stage1_observation(
            args.output
                .as_deref()
                .context("Stage1 evidence output missing")?,
            window,
            &baton,
            &auxiliary,
        )?;
    }
    let mut future = Vec::new();
    for (relative, frame) in auxiliary
        .as_slice()
        .context("contiguous auxiliary needed")?
        .chunks_exact((p.height / 2) * (p.width / 2))
        .enumerate()
    {
        let absolute = window.start + relative;
        if absolute < window.finalize_before {
            continue;
        }
        let upsampled = if args.provider == "cuda" {
            resize_bilinear_chw_cuda_order(frame, 1, p.height / 2, p.width / 2, p.height, p.width)?
        } else {
            resize_bilinear_chw(frame, 1, p.height / 2, p.width / 2, p.height, p.width)?
        };
        future.push((
            absolute,
            upsampled
                .into_iter()
                .map(|value| value.clamp(0.0, 1.0))
                .collect(),
        ));
    }
    hints.commit_auxiliary_overlap(window, future)?;
    Ok(baton)
}

fn temporal_contract(models: &Path, directml_variant: bool) -> Result<(usize, serde_json::Value)> {
    let manifest: serde_json::Value = serde_json::from_slice(&fs::read(
        models.join("stage1").join(if directml_variant {
            "manifest.directml.json"
        } else {
            "manifest.json"
        }),
    )?)?;
    let contract = &manifest["contract"];
    ensure!(
        contract["dynamic_temporal"] == true,
        "This CorridorKey pack has a fixed temporal window. Update the adaptive model pack before rendering."
    );
    let minimum = contract["temporal_min"]
        .as_u64()
        .context("temporal minimum missing")?;
    let maximum = contract["temporal_max"]
        .as_u64()
        .context("temporal maximum missing")?;
    ensure!(
        minimum == 1 && (1..=256).contains(&maximum),
        "unsupported temporal model range"
    );
    let inputs = manifest["inputs"]
        .as_array()
        .context("Stage1 inputs missing")?;
    for name in [
        "rgb",
        "hint",
        "moge_feat",
        "cradio_feat",
        "rvm_feat",
        "hint_quality",
    ] {
        let input = inputs
            .iter()
            .find(|value| value["name"] == name)
            .context("Stage1 temporal input missing")?;
        let dimension = &input["shape"][1];
        ensure!(
            dimension.as_i64().is_some_and(|value| value < 0) || dimension == "frames",
            "Stage1 {name} must declare a dynamic frames dimension"
        );
    }
    Ok((usize::try_from(maximum)?, contract.clone()))
}

fn encoder_query_experiment_requested() -> bool {
    std::env::var("BRP_CHROMA_DML_ENCODER_QUERY_EXPERIMENT").is_ok_and(|value| value == "512")
}

fn encoder_query_experiment_contract(bounded: &serde_json::Value, requested: bool) -> bool {
    requested
        && bounded["query_chunk_tokens"] == 512
        && bounded["default_query_chunk_tokens"] == 128
        && bounded["query_chunk_overrides"]
            == serde_json::json!({
                "/model/enc_16.0/attn/Loop": 512,
                "/model/enc_16.1/attn/Loop": 512
            })
}

fn check_bounded_contract_with_experiment(
    contract: &serde_json::Value,
    requested: bool,
) -> Result<()> {
    let bounded = &contract["bounded_inference"];
    // Development-only comparison of two encoder query tiles. The installed
    // model and ordinary jobs retain their verified 128-query bound. Never
    // silently accept a larger general/window-attention bound here.
    let query_bound = bounded["query_chunk_tokens"]
        .as_u64()
        .is_some_and(|value| (1..=128).contains(&value))
        || encoder_query_experiment_contract(bounded, requested);
    ensure!(
        bounded["encoder_frame_chunk"] == 1
            && bounded["mlp_chunk_tokens"]
                .as_u64()
                .is_some_and(|value| (1..=65536).contains(&value))
            && bounded["swin_window_batch"]
                .as_u64()
                .is_some_and(|value| (1..=16).contains(&value))
            && query_bound
            && bounded["window_mask_queries_bounded"] == true,
        "Low VRAM requires the verified bounded encoder/MLP/window/query model contract"
    );
    Ok(())
}

fn check_bounded_contract(contract: &serde_json::Value) -> Result<()> {
    check_bounded_contract_with_experiment(contract, encoder_query_experiment_requested())
}

fn configure_desktop_model_profile(args: &mut Args, models: &Path) -> Result<()> {
    args.directml_mixed_precision = false;
    if args.command != "desktop" || args.provider != "directml" {
        return Ok(());
    }
    let pack_file = models.join("pack.json");
    let marker: serde_json::Value = match fs::read(&pack_file) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .with_context(|| format!("reading CorridorKey pack {}", pack_file.display()))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    let enabled = &marker["execution_profiles"]["directml_mixed_precision"];
    ensure!(
        enabled.is_null() || enabled.is_boolean(),
        "DirectML mixed-precision pack marker must be boolean"
    );
    if enabled != true {
        return Ok(());
    }
    ensure!(
        marker["schema_version"] == 2 && marker["kind"] == "corridorkey_dynamic",
        "DirectML mixed precision requires a schema2 dynamic CorridorKey pack"
    );
    for component in ["stage1", "stage2"] {
        let directory = models.join(component);
        for name in [
            "model.onnx",
            "model.onnx.data",
            "manifest.json",
            "model.directml.onnx",
            "manifest.directml.json",
        ] {
            ensure!(
                directory.join(name).is_file(),
                "Incomplete DirectML mixed-precision pack: missing {component}/{name}"
            );
        }
        let read_manifest = |name: &str| -> Result<serde_json::Value> {
            Ok(serde_json::from_slice(&fs::read(directory.join(name))?)?)
        };
        let base = read_manifest("manifest.json")?;
        let variant = read_manifest("manifest.directml.json")?;
        let precision = variant["contract"]["precision"]
            .as_str()
            .context("DirectML variant contract.precision missing")?;
        ensure!(
            matches!(
                precision,
                "mixed_float16_compute_float32_io"
                    | "mixed_float16_compute_float32_input_float16_output"
            ),
            "Unsupported declared DirectML {component} precision {precision}"
        );
        if component == "stage1" {
            for field in ["dynamic_temporal", "temporal_min", "temporal_max"] {
                ensure!(
                    !base["contract"][field].is_null()
                        && base["contract"][field] == variant["contract"][field],
                    "DirectML Stage1 {field} differs from the base pack"
                );
            }
            let base_bounds = base["contract"]["bounded_inference"]
                .as_object()
                .context("Base Stage1 bounds missing")?;
            for (field, value) in base_bounds {
                let query_experiment = field == "query_chunk_tokens"
                    && *value == 128
                    && encoder_query_experiment_contract(
                        &variant["contract"]["bounded_inference"],
                        encoder_query_experiment_requested(),
                    );
                ensure!(
                    variant["contract"]["bounded_inference"][field] == *value || query_experiment,
                    "DirectML Stage1 bound {field} differs from the base pack"
                );
            }
            check_bounded_contract(&variant["contract"])?;
            for field in ["complete_swin_window_batch", "cswin_strip_batch"] {
                ensure!(
                    variant["contract"]["bounded_inference"][field]
                        .as_u64()
                        .is_some_and(|value| (1..=16).contains(&value)),
                    "DirectML mixed-precision Low VRAM profile requires {field} between1 and16"
                );
            }
        }
    }
    // Validate dynamic axes/range before any session or GPU budget sampling.
    temporal_contract(models, true)?;
    args.directml_mixed_precision = true;
    Ok(())
}

fn new_attempt(root: &Path, job_id: &str, sequence: usize) -> Result<PathBuf> {
    for suffix in 0..1000 {
        let path = root.join(format!("attempt-{sequence:03}-{suffix:03}"));
        match fs::create_dir(&path) {
            Ok(()) => {
                fs::write(path.join(".brp-native-attempt"), job_id.as_bytes())?;
                return Ok(path);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    anyhow::bail!("No fresh native attempt directory is available")
}

fn remove_owned_attempt(root: &Path, path: &Path, job_id: &str) -> Result<()> {
    let root = fs::canonicalize(root)?;
    let path = fs::canonicalize(path)?;
    ensure!(
        path.parent() == Some(root.as_path()),
        "attempt cleanup escaped the selected output"
    );
    ensure!(
        fs::read(path.join(".brp-native-attempt"))? == job_id.as_bytes(),
        "attempt cleanup ownership marker differs"
    );
    fs::remove_dir_all(path)?;
    Ok(())
}

fn remove_owned_work(output: &Path, job_id: &str) -> Result<()> {
    ensure!(
        fs::read(output.join(".brp-native-attempt"))? == job_id.as_bytes(),
        "scratch cleanup ownership marker differs"
    );
    let work = output.join("work");
    if work.exists() {
        ensure!(
            work.canonicalize()?.parent() == Some(output.canonicalize()?.as_path()),
            "scratch cleanup escaped the owned attempt"
        );
        fs::remove_dir_all(work)?;
    }
    Ok(())
}

pub(crate) fn run(
    job: &JobRequest,
    range: &FrameRange,
    settings: &CkSettings,
    models: PathBuf,
    provider: Provider,
) -> Result<JobArtifacts> {
    settings.validate()?;
    let started = Instant::now();
    let total = usize::try_from(range.end_exclusive - range.start)?;
    let mut base = crate::desktop::worker_args(
        job,
        job.output_dir.join("adaptive-job.json"),
        models.clone(),
        provider,
    );
    base.explicit_low_vram = settings.low_vram;
    configure_desktop_model_profile(&mut base, &models)?;
    let (temporal_max, contract) = temporal_contract(&models, base.directml_mixed_precision)?;
    let (initial_memory, budget_error) = memory_snapshot(&base);
    let budget = initial_memory.as_ref().map(usable_budget);
    let request = WindowRequest {
        mode: match settings.window_mode {
            CkWindowMode::Balanced => WindowMode::Balanced,
            CkWindowMode::Wide => WindowMode::Wide,
            CkWindowMode::Custom => WindowMode::Custom,
        },
        chunk: settings.hann_chunk,
        stride: settings.hann_stride,
        low_vram: settings.low_vram,
    };
    let initial = resolve_window(
        request,
        job.media.width as usize,
        job.media.height as usize,
        total,
        budget,
        temporal_max,
        memory_estimate(&base),
    )?;
    let auto_padding = Padding::new(job.media.height as usize, job.media.width as usize, 128)?;
    let initial = app_auto_window(
        &base,
        request,
        initial,
        auto_padding,
        total,
        budget,
        temporal_max,
    );
    let initial = balanced_staged_reuse(&base, request, initial, auto_padding, total, budget);
    let choices = retry_choices(
        initial,
        temporal_max,
        memory_estimate(&base),
        job.media.width as usize,
        job.media.height as usize,
        total,
    );
    let mut pending: VecDeque<_> = choices
        .into_iter()
        .map(|window| AttemptPolicy {
            window,
            disable_reuse: false,
        })
        .collect();
    let mut history = Vec::new();
    let mut attempt = 0;
    while let Some(policy) = pending.pop_front() {
        let choice = policy.window;
        if choice.low_vram {
            check_bounded_contract(&contract)?;
        }
        job.media.verify_source_unchanged()?;
        let output = new_attempt(&job.output_dir, &job.job_id, attempt)?;
        crate::desktop::set_output(&output)?;
        let mut args = base.clone();
        args.dml_disable_reuse = policy.disable_reuse;
        args.output = Some(output.clone());
        args.report = Some(output.join("inference-report.json"));
        args.manifest = output.join("adaptive-job.json");
        if args.profile.is_some() {
            args.profile = Some(output.join("provider-profile"));
        }
        fs::write(
            &args.manifest,
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema_version":1,"source":job.media,"source_range":range,
                "requested_settings":settings,"resolved_window":choice_json(choice,&args),
                "stage1_arena_policy":"resolved from the live WDDM budget immediately before Stage1 load",
                "stage2_arena_policy":"resolved from the live WDDM budget immediately before Stage2 load",
                "force_disable_arena_reuse":policy.disable_reuse,
                "explicit_low_vram":args.explicit_low_vram,
                "stage1_temporal_contract":contract
            }))?,
        )?;
        let actual_t = choice.chunk.min(total);
        crate::desktop::progress_message(
            "window_policy",
            attempt + 1,
            attempt + 1 + pending.len(),
            format!(
                "{}: window {}/stride {} (actual T{}), Stage2 batch1; estimated {:.2} GiB, usable WDDM budget {}",
                if choice.low_vram {
                    "Low VRAM staged offload"
                } else {
                    "Resident streaming"
                },
                choice.chunk,
                choice.stride,
                actual_t,
                choice.estimated_peak_gib,
                budget
                    .map(|value| format!("{value:.2} GiB"))
                    .unwrap_or_else(|| "unavailable".into())
            ),
        )?;
        match execute_attempt(&args, job, range, &models, choice, started) {
            Ok(stats) => {
                let cleanup_error = remove_owned_work(&output, &job.job_id)
                    .err()
                    .map(|error| format!("{error:#}"));
                history.push(serde_json::json!({"attempt":attempt,"window":choice_json(choice,&args),"completed":true,"force_disable_arena_reuse":policy.disable_reuse}));
                let free_budget = initial_memory
                    .as_ref()
                    .map(crate::gpu_memory::GpuMemorySnapshot::free_budget_bytes);
                let mut report = serde_json::json!({
                    "schema_version":1,"component":"corridorkey_desktop_inference",
                    "execution_success":true,"validation_performed":false,"numerical_passed":null,
                    "source":job.media.path,"source_range":range,"frames":total,
                    "native_hw":[job.media.height,job.media.width],
                    "shape_hw":[Padding::new(job.media.height as usize,job.media.width as usize,128)?.height,
                        Padding::new(job.media.height as usize,job.media.width as usize,128)?.width],
                    "precision":"float32",
                    "precision_scope":"host inputs/readback/reference tensors; neural arithmetic defined by ONNX graph",
                    "runtime_policy":crate::runtime_policy(&args),
                    "requested_settings":settings,"window_policy":choice_json(choice,&args),
                    "actual_stage1_frames":actual_t,"stage2_batch":1,
                    "execution_mode":if choice.low_vram{"staged_offload"}else{"resident_streaming"},
                    "source_geometry":"native pixels, original centered reflect101 padding/crop; no profile resize",
                    "stage1_temporal_contract":contract,"gpu_budget_initial":initial_memory,
                    "gpu_free_budget_initial_bytes":free_budget,"gpu_budget_error":budget_error,
                    "gpu_budget_reserve_bytes":GPU_RESERVE_BYTES,"attempts":history,
                    "elapsed_seconds":started.elapsed().as_secs_f64(),"statistics":stats,
                    "components":stats.components,
                    "input_provenance":{"mode":"native_ffmpeg_exact_frame_stream","decoded_rgb_sha256":stats.decoded_rgb_sha256},
                    "rgba":"frames","alpha":"alpha","checker":"checker",
                    "full_scene_rgb_dump":false
                    ,"memory_summary":"memory-summary.json"
                    ,"scratch_cleanup_error":cleanup_error,"output_queue_capacity":1,
                    "stage2_memory_arena":stats.stage2_arena_policy.enabled
                    ,"stage2_arena_policy":stats.stage2_arena_policy
                    ,"stage1_memory_arena":stats.stage1_arena_policy.enabled
                    ,"stage1_arena_policy":stats.stage1_arena_policy
                    ,"developer_validation_raw_logits":validation_raw_enabled()
                });
                report["developer_validation_stage1_observations"] =
                    validation_stage1_enabled().into();
                report["explicit_low_vram"] = args.explicit_low_vram.into();
                report["developer_encoder_query_512"] = encoder_query_experiment_requested().into();
                report["session_load_overlap"] = serde_json::to_value(&stats.session_load_overlap)?;
                fs::write(
                    args.report.as_ref().context("report path missing")?,
                    serde_json::to_vec_pretty(&report)?,
                )?;
                return Ok(JobArtifacts {
                    frames_dir: Some(output.join("frames")),
                    alpha_dir: Some(output.join("alpha")),
                    checker_dir: Some(output.join("checker")),
                    report: args.report,
                    ..Default::default()
                });
            }
            Err(error) => {
                let oom = allocation_oom(&error);
                let reuse_retry = retry_without_reuse(policy, &error, &mut pending);
                let guard = error
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<CapacityGuard>())
                    .map(|guard| guard.reason);
                let retryable = oom || guard.is_some();
                let retry_reason = guard.unwrap_or(if reuse_retry {
                    "allocation_oom_retry_without_arena"
                } else if oom {
                    "allocation_oom"
                } else {
                    "non_memory_failure"
                });
                history.push(
                    serde_json::json!({"attempt":attempt,"window":choice_json(choice,&args),
                    "completed":false,"allocation_oom":oom,"retry_reason":retry_reason,
                    "force_disable_arena_reuse":policy.disable_reuse,"retry_without_arena":reuse_retry,"error":format!("{error:#}")}),
                );
                if !retryable || pending.is_empty() {
                    let cleanup_error = remove_owned_work(&output, &job.job_id)
                        .err()
                        .map(|error| format!("{error:#}"));
                    fs::write(
                        output.join("failure-report.json"),
                        serde_json::to_vec_pretty(
                            &serde_json::json!({"execution_success":false,"attempts":history,"scratch_cleanup_error":cleanup_error}),
                        )?,
                    )?;
                    return Err(error.context(if retryable {
                        "Allocation-memory recovery exhausted; select CPU explicitly or reduce the source size"
                    }else{"CorridorKey job failed; this is not an allocation-memory error and was not retried"}));
                }
                crate::desktop::progress_message(
                    "memory_retry",
                    attempt + 1,
                    attempt + 1 + pending.len(),
                    format!(
                        "{retry_reason}: restarting the complete scene in a fresh output attempt. {}",
                        error
                    ),
                )?;
                for name in ["memory-phases.jsonl", "memory-summary.json"] {
                    let trace = output.join(name);
                    if trace.is_file() {
                        let archive = job.output_dir.join(format!("attempt-{attempt:03}-{name}"));
                        if let Err(error) = fs::copy(trace, &archive) {
                            history.push(serde_json::json!({"memory_trace_archive":archive,"error":error.to_string()}));
                        }
                    }
                }
                if let Err(cleanup) = remove_owned_attempt(&job.output_dir, &output, &job.job_id) {
                    history.push(serde_json::json!({"retained_failed_attempt":output,"cleanup_error":format!("{cleanup:#}")}));
                }
            }
        }
        attempt += 1;
    }
    anyhow::bail!("No CorridorKey execution policy was available")
}

#[derive(Serialize)]
struct AttemptStats {
    components: Vec<super::super::pipeline::ComponentReport>,
    peak_cached_frames: usize,
    peak_cached_tensor_bytes: usize,
    rendered_frames: usize,
    first_frame_seconds: Option<f64>,
    decoded_rgb_sha256: String,
    generated_scratch_bytes: u64,
    memory_after_model_load: Option<crate::gpu_memory::GpuMemorySnapshot>,
    sampled_process_gpu_peak_bytes: u64,
    memory_trace: String,
    gpu_memory_samples: Option<crate::gpu_memory::GpuMemorySamples>,
    gpu_monitor_start_error: Option<String>,
    stage1_arena_policy: ArenaReusePolicy,
    stage2_arena_policy: ArenaReusePolicy,
    session_load_overlap: SessionLoadOverlap,
}

fn execute_attempt(
    args: &Args,
    job: &JobRequest,
    range: &FrameRange,
    models: &Path,
    choice: WindowChoice,
    started: Instant,
) -> Result<AttemptStats> {
    let mut reuse = ReuseInAttempt::default();
    execute_attempt_inner(args, job, range, models, choice, started, &mut reuse).map_err(|error| {
        if reuse.stage1 || reuse.stage2 {
            error.context(reuse)
        } else {
            error
        }
    })
}

fn execute_attempt_inner(
    args: &Args,
    job: &JobRequest,
    range: &FrameRange,
    models: &Path,
    choice: WindowChoice,
    started: Instant,
    reuse: &mut ReuseInAttempt,
) -> Result<AttemptStats> {
    let p = Padding::new(job.media.height as usize, job.media.width as usize, 128)?;
    let total = usize::try_from(range.end_exclusive - range.start)?;
    let plan = WindowPlan::new(total, choice.chunk, choice.stride)?;
    let temporal = plan.windows[0].len();
    let work = args
        .output
        .as_ref()
        .context("attempt output missing")?
        .join("work");
    fs::create_dir_all(&work)?;
    let mut trace = PhaseTrace::open(
        args.output.as_ref().context("attempt output missing")?,
        range,
        started,
        &args.provider,
    )?;
    let contracts = ["rvm", "cradio", "moge"]
        .into_iter()
        .map(|name| Ok((name, foundation_contract(models, name, p.height, p.width)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut args = args.clone();
    if args.provider == "directml" {
        args.dimension_overrides
            .extend(directml_dimensions(p.height, p.width)?);
    }
    initialize_runtime(&args)?;
    trace.record(&args, "runtime_initialized", 0, total)?;
    let mut components = Vec::new();
    let mut scratch_bytes = 0u64;
    let mut reference_hash = None;
    let mut resident = BTreeMap::new();
    let mut state = contracts["rvm"]
        .state
        .as_deref()
        .map(|shape| ArrayD::zeros(IxDyn(shape)));
    // Developer A/B: preload the Stage1 session while the staged foundation
    // pass runs. The portable default remains a sequential load; the final
    // memory guard is unchanged and still samples after every session exists.
    let stage1_policy = resolve_stage1_arena(&args, choice.low_vram, p, temporal);
    let stage1_options = stage1_args(&args, temporal, &stage1_policy)?;
    reuse.stage1 = stage1_options.dml_memory_arena;
    let overlap = {
        let (memory, _) = memory_snapshot(&args);
        session_load_overlap(&args, &choice, memory.as_ref().map(usable_budget))
    };
    let mut stage1_loader = None;
    if overlap.parallel_stage1_load {
        let options = stage1_options.clone();
        let loader_models = models.to_owned();
        trace.record(&args, "load_stage1_begin", 0, total)?;
        stage1_loader = Some(std::thread::spawn(move || {
            Graph::open(&options, &loader_models, "stage1")
        }));
    }
    if choice.low_vram {
        disk_preflight(&work, staged_scratch_bound(&contracts, p, total)?)?;
        for name in ["rvm", "cradio", "moge"] {
            disk_preflight(&work, 0)?;
            fs::create_dir_all(work.join(name))?;
            trace.record(&args, &format!("load_{name}_begin"), 0, total)?;
            let mut graph = Graph::open(&args, models, name)?;
            trace.record(&args, &format!("load_{name}_end"), 0, total)?;
            let mut decoder = Decoder::open(job, range, &work, name)?;
            for frame in 0..total {
                let bytes = decoder.next()?;
                let display =
                    p.pad_chw(&planar(&bytes, p.native_width, p.native_height, false)?, 3)?;
                trace.record(&args, &format!("{name}_begin"), frame, total)?;
                let feature = foundation_frame(
                    &mut graph,
                    name,
                    &contracts[name],
                    &mut state,
                    &display,
                    p,
                    &args,
                )?;
                trace.record(&args, &format!("{name}_end"), frame, total)?;
                let path = frame_path(&work, name, frame);
                disk_preflight(
                    &work,
                    (feature.len() as u64)
                        .saturating_mul(4)
                        .saturating_add(4096),
                )?;
                write_npy(&path, &feature)?;
                scratch_bytes += fs::metadata(path)?.len();
                crate::desktop::progress(name, frame + 1, total)?;
            }
            let hash = decoder.finish()?;
            if let Some(reference) = &reference_hash {
                ensure!(
                    reference == &hash,
                    "source decode changed between foundation passes"
                );
            } else {
                reference_hash = Some(hash);
            }
            components.push(graph.finish()?);
        }
    } else {
        for name in ["rvm", "cradio", "moge"] {
            trace.record(&args, &format!("load_{name}_begin"), 0, total)?;
            resident.insert(name, Graph::open(&args, models, name)?);
            trace.record(&args, &format!("load_{name}_end"), 0, total)?;
        }
    }
    let mut stage1 = match stage1_loader {
        Some(loader) => match loader.join() {
            Ok(Ok(graph)) => graph,
            Ok(Err(error)) => {
                eprintln!("background Stage1 load failed; retrying sequentially: {error:#}");
                Graph::open(&stage1_options, models, "stage1")?
            }
            Err(_) => {
                eprintln!("background Stage1 loader panicked; retrying sequentially");
                Graph::open(&stage1_options, models, "stage1")?
            }
        },
        None => {
            trace.record(&args, "load_stage1_begin", 0, total)?;
            Graph::open(&stage1_options, models, "stage1")?
        }
    };
    trace.record(&args, "load_stage1_end", 0, total)?;
    let mut stage2_policy = stage2_arena_policy(&args, choice.low_vram, p, None, false, false);
    let mut stage2 = if choice.low_vram {
        None
    } else {
        trace.record(&args, "load_stage2_begin", 0, total)?;
        stage2_policy = resolve_stage2_arena(&args, false, p);
        reuse.stage2 = stage2_policy.enabled;
        Some(Graph::open(
            &stage2_args(&args, &stage2_policy),
            models,
            "stage2",
        )?)
    };
    trace.record(&args, "model_loads_complete", 0, total)?;
    let (memory_after_model_load, _) = memory_snapshot(&args);
    if let Some(memory) = &memory_after_model_load {
        gpu_budget_guard(choice, memory)?;
    }
    host_preflight(p, temporal)?;
    let mut renderer = if choice.low_vram {
        None
    } else {
        Some(Render::open(&args, job, p, total, started)?)
    };
    // Prefetch the Stage2 session while the Stage1 windows run. The live-budget
    // gate keeps this off weak adapters; the returned session is joined before
    // the first Stage2 frame, so only session initialization overlaps.
    let mut stage2_loader = None;
    if overlap.stage2_prefetch {
        stage2_policy = resolve_stage2_arena(&args, true, p);
        reuse.stage2 = stage2_policy.enabled;
        let options = stage2_args(&args, &stage2_policy);
        let loader_models = models.to_owned();
        trace.record(&args, "load_stage2_begin", 0, total)?;
        stage2_loader = Some(std::thread::spawn(move || {
            Graph::open(&options, &loader_models, "stage2")
        }));
    }
    let result = (|| -> Result<AttemptStats> {
        if choice.low_vram {
            fs::create_dir_all(work.join("baton"))?;
        }
        let mut cache = FrameCache::default();
        let mut decoder = Decoder::open(job, range, &work, "windows")?;
        let mut next = 0usize;
        let mut hints = HintCarry::new(p.height * p.width)?;
        let mut accumulation = BatonAccumulator::new(512 * (p.height / 16) * (p.width / 16))?;
        for window in &plan.windows {
            while next < window.end {
                let bytes = decoder.next()?;
                let rgb = p.pad_chw(&planar(&bytes, p.native_width, p.native_height, true)?, 3)?;
                let mut features = BTreeMap::new();
                if !choice.low_vram {
                    let display =
                        p.pad_chw(&planar(&bytes, p.native_width, p.native_height, false)?, 3)?;
                    for name in ["rvm", "cradio", "moge"] {
                        trace.record(&args, &format!("{name}_begin"), next, total)?;
                        features.insert(
                            name,
                            foundation_frame(
                                resident
                                    .get_mut(name)
                                    .context("resident foundation missing")?,
                                name,
                                &contracts[name],
                                &mut state,
                                &display,
                                p,
                                &args,
                            )?,
                        );
                        trace.record(&args, &format!("{name}_end"), next, total)?;
                    }
                }
                cache.push(CachedFrame {
                    index: next,
                    rgb,
                    features,
                });
                next += 1;
            }
            trace.record(&args, "stage1_window_begin", window.start, total)?;
            let baton = stage1_window(
                &mut stage1,
                window,
                &cache,
                &contracts,
                choice.low_vram.then_some(work.as_path()),
                &mut hints,
                (p, &args),
            )?;
            trace.record(&args, "stage1_window_end", window.start, total)?;
            let finalized = accumulation
                .push_window(window, baton.as_slice().context("contiguous baton needed")?)?;
            drop(baton);
            for frame in finalized {
                let CachedFrame { rgb, features, .. } = cache.take(frame.index)?;
                drop(features);
                if choice.low_vram {
                    drop(rgb);
                    let path = frame_path(&work, "baton", frame.index);
                    disk_preflight(
                        &work,
                        (frame.values.len() as u64)
                            .saturating_mul(4)
                            .saturating_add(4096),
                    )?;
                    write_npy(
                        &path,
                        &tensor(&[1, 1, 512, p.height / 16, p.width / 16], frame.values)?,
                    )?;
                    scratch_bytes += fs::metadata(path)?.len();
                    for name in ["rvm", "cradio", "moge"] {
                        release_work_frame(&work, name, frame.index)?;
                    }
                } else {
                    trace.record(&args, "stage2_frame_begin", frame.index, total)?;
                    renderer
                        .as_mut()
                        .context("stream renderer missing")?
                        .frame(
                            stage2.as_mut().context("resident stage2 missing")?,
                            frame.index,
                            rgb,
                            frame.values,
                        )?;
                    trace.record(&args, "stage2_frame_end", frame.index, total)?;
                }
            }
            crate::desktop::progress("stage1", window.index + 1, plan.windows.len())?;
        }
        ensure!(
            accumulation.pending_frames() == 0 && cache.frames.is_empty(),
            "unfinalized temporal/source cache"
        );
        let hash = decoder.finish()?;
        if let Some(reference) = &reference_hash {
            ensure!(reference == &hash, "source decode changed for stage1");
        }
        components.push(stage1.finish()?);
        for (_, graph) in resident {
            components.push(graph.finish()?);
        }
        if choice.low_vram {
            // Stage1 is consumed/dropped above before Stage2 is even initialized,
            // unless the developer prefetch already loaded it concurrently.
            host_preflight(p, 1)?;
            disk_preflight(&work, 0)?;
            stage2 = Some(if let Some(loader) = stage2_loader.take() {
                match loader.join() {
                    Ok(Ok(graph)) => graph,
                    Ok(Err(error)) => {
                        eprintln!(
                            "background Stage2 load failed; retrying sequentially: {error:#}"
                        );
                        Graph::open(&stage2_args(&args, &stage2_policy), models, "stage2")?
                    }
                    Err(_) => {
                        eprintln!("background Stage2 loader panicked; retrying sequentially");
                        Graph::open(&stage2_args(&args, &stage2_policy), models, "stage2")?
                    }
                }
            } else {
                trace.record(&args, "load_stage2_begin", 0, total)?;
                stage2_policy = resolve_stage2_arena(&args, true, p);
                reuse.stage2 = stage2_policy.enabled;
                Graph::open(&stage2_args(&args, &stage2_policy), models, "stage2")?
            });
            trace.record(&args, "load_stage2_end", 0, total)?;
            let mut reader = Decoder::open(job, range, &work, "stage2")?;
            renderer = Some(Render::open(&args, job, p, total, started)?);
            for index in 0..total {
                let bytes = reader.next()?;
                let rgb = p.pad_chw(&planar(&bytes, p.native_width, p.native_height, true)?, 3)?;
                let baton = read_npy(frame_path(&work, "baton", index))?;
                trace.record(&args, "stage2_frame_begin", index, total)?;
                renderer
                    .as_mut()
                    .context("staged output writer missing")?
                    .frame(
                        stage2.as_mut().context("staged stage2 missing")?,
                        index,
                        rgb,
                        baton.into_raw_vec_and_offset().0,
                    )?;
                trace.record(&args, "stage2_frame_end", index, total)?;
                release_work_frame(&work, "baton", index)?;
            }
            ensure!(reader.finish()? == hash, "source decode changed for stage2");
        }
        components.push(stage2.context("stage2 graph missing")?.finish()?);
        let (rendered_frames, first_frame_seconds) =
            renderer.take().context("renderer missing")?.finish()?;
        ensure!(
            rendered_frames == total,
            "rendered frame count differs from source interval"
        );
        job.media.verify_source_unchanged()?;
        Ok(AttemptStats {
            components,
            peak_cached_frames: cache.peak_frames,
            peak_cached_tensor_bytes: cache.peak_bytes,
            rendered_frames,
            first_frame_seconds,
            decoded_rgb_sha256: hash,
            generated_scratch_bytes: scratch_bytes,
            memory_after_model_load,
            sampled_process_gpu_peak_bytes: trace.sampled_peak(),
            memory_trace: "memory-phases.jsonl".into(),
            gpu_memory_samples: trace.samples(),
            gpu_monitor_start_error: trace.monitor_start_error.clone(),
            stage1_arena_policy: stage1_policy,
            stage2_arena_policy: stage2_policy,
            session_load_overlap: overlap.clone(),
        })
    })();
    if let Err(error) = result {
        // Preserve a simultaneous disk/encoder failure rather than hiding it
        // behind a GPU allocation retry. Joining also orders all FrameReady
        // events before the caller can emit memory_retry.
        if let Some(render) = renderer.take() {
            render.finish().with_context(|| {
                format!("output finalization failed while stopping the pipeline ({error:#})")
            })?;
        }
        return Err(error);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_low_vram_vetoes_reuse_without_changing_window_or_oom_reason() {
        assert!(!Args::default().explicit_low_vram);
        let automatic = Args {
            provider: "directml".into(),
            directml_mixed_precision: true,
            ..Default::default()
        };
        let p = Padding::new(1080, 1920, 128).unwrap();
        assert!(stage1_arena_policy(&automatic, true, p, 8, Some(16.0), false).enabled);
        assert!(stage2_arena_policy(&automatic, true, p, Some(16.0), false, false).enabled);
        let explicit = Args {
            explicit_low_vram: true,
            ..automatic
        };
        let stage1 = stage1_arena_policy(&explicit, true, p, 8, Some(16.0), true);
        let stage2 = stage2_arena_policy(&explicit, true, p, Some(16.0), true, true);
        assert!(!stage1.enabled && !stage2.enabled);
        assert_eq!(
            (stage1.source, stage2.source),
            ("explicit_low_vram", "explicit_low_vram")
        );
        assert!(!stage1_args(&explicit, 8, &stage1).unwrap().dml_memory_arena);
        assert!(!stage2_args(&explicit, &stage2).dml_memory_arena);
        assert!(!explicit.dml_disable_reuse);
        let request = WindowRequest {
            mode: WindowMode::Custom,
            chunk: 8,
            stride: 3,
            low_vram: true,
        };
        let initial = resolve_window(
            request,
            1920,
            1080,
            69,
            Some(14.37),
            256,
            memory_estimate(&explicit),
        )
        .unwrap();
        let choice = balanced_staged_reuse(&explicit, request, initial, p, 69, Some(14.37));
        assert_eq!((choice.chunk, choice.stride, choice.low_vram), (8, 3, true));
        assert_eq!(crate::runtime_policy(&explicit)["explicit_low_vram"], true);
        assert_eq!(choice_json(choice, &explicit)["explicit_low_vram"], true);
    }
    #[test]
    fn staged_mixed_stage1_reuse_is_bounded_by_measured_domain_and_live_budget() {
        let args = Args {
            provider: "directml".into(),
            directml_mixed_precision: true,
            ..Default::default()
        };
        let p = Padding::new(1080, 1920, 128).unwrap();
        let policy = stage1_arena_policy(&args, true, p, 8, Some(12.0), false);
        assert!(policy.enabled);
        assert_eq!(policy.estimated_required_gib, Some(12.0));
        assert!(!stage1_arena_policy(&args, true, p, 8, Some(11.99), false).enabled);
        assert!(!stage1_arena_policy(&args, true, p, 8, None, false).enabled);
        assert!(!stage1_arena_policy(&args, true, p, 9, Some(32.0), false).enabled);
        assert!(
            !stage1_arena_policy(
                &args,
                true,
                Padding::new(2160, 3840, 128).unwrap(),
                8,
                Some(32.0),
                false
            )
            .enabled
        );
        assert!(!stage1_arena_policy(&args, false, p, 8, Some(16.0), false).enabled);
        assert!(
            !stage1_arena_policy(
                &Args {
                    directml_mixed_precision: false,
                    ..args.clone()
                },
                true,
                p,
                8,
                Some(16.0),
                false
            )
            .enabled
        );
        assert!(
            !stage1_arena_policy(
                &Args {
                    provider: "cpu".into(),
                    ..args.clone()
                },
                true,
                p,
                8,
                Some(16.0),
                true
            )
            .enabled
        );
        let disabled = Args {
            dml_disable_reuse: true,
            ..args
        };
        let policy = stage1_arena_policy(&disabled, true, p, 8, Some(16.0), true);
        assert!(!policy.enabled);
        assert!(!stage1_args(&disabled, 8, &policy).unwrap().dml_memory_arena);
        assert_eq!(
            stage1_args(&disabled, 8, &policy)
                .unwrap()
                .dimension_overrides["frames"],
            8
        );
    }
    #[test]
    fn balanced_reuse_changes_residency_without_changing_hann_context_or_custom_wide() {
        let args = Args {
            provider: "directml".into(),
            directml_mixed_precision: true,
            ..Default::default()
        };
        let p = Padding::new(1080, 1920, 128).unwrap();
        let request = WindowRequest::default();
        let initial = resolve_window(
            request,
            1920,
            1080,
            69,
            Some(14.37),
            256,
            memory_estimate(&args),
        )
        .unwrap();
        let choice = balanced_staged_reuse(&args, request, initial, p, 69, Some(14.37));
        assert_eq!(
            (choice.chunk, choice.stride),
            (initial.chunk, initial.stride)
        );
        assert_eq!((choice.chunk, choice.stride), (8, 4));
        assert!(choice.low_vram);
        assert_eq!(choice.source, "balanced_staged_reuse");
        for mode in [WindowMode::Custom, WindowMode::Wide] {
            let kept = balanced_staged_reuse(
                &args,
                WindowRequest { mode, ..request },
                initial,
                p,
                69,
                Some(14.37),
            );
            assert!(!kept.low_vram);
            assert_eq!((kept.chunk, kept.stride), (initial.chunk, initial.stride));
        }
        for budget in [None, Some(11.99)] {
            assert!(!balanced_staged_reuse(&args, request, initial, p, 69, budget).low_vram);
        }
        assert!(
            !balanced_staged_reuse(
                &Args {
                    directml_mixed_precision: false,
                    ..args.clone()
                },
                request,
                initial,
                p,
                69,
                Some(16.0)
            )
            .low_vram
        );
        assert!(
            !balanced_staged_reuse(
                &args,
                request,
                WindowChoice {
                    chunk: 16,
                    ..initial
                },
                p,
                69,
                Some(32.0)
            )
            .low_vram
        );
        // A short clip uses its actual temporal length, never the nominal chunk.
        let one = balanced_staged_reuse(
            &args,
            request,
            WindowChoice {
                chunk: 24,
                stride: 12,
                ..initial
            },
            p,
            1,
            Some(12.0),
        );
        assert!(one.low_vram);
        assert_eq!((one.chunk, one.stride), (24, 12));
    }
    #[test]
    fn staged_mixed_stage2_reuse_requires_live_total_budget() {
        let args = Args {
            provider: "directml".into(),
            directml_mixed_precision: true,
            ..Default::default()
        };
        let p = Padding::new(1080, 1920, 128).unwrap();
        let policy = stage2_arena_policy(&args, true, p, Some(6.5), false, false);
        assert!(policy.enabled);
        assert_eq!(policy.estimated_required_gib, Some(6.5));
        for budget in [None, Some(6.49)] {
            assert!(!stage2_arena_policy(&args, true, p, budget, false, false).enabled);
        }
        assert!(!stage2_arena_policy(&args, false, p, Some(16.0), false, false).enabled);
        assert!(
            !stage2_arena_policy(
                &Args {
                    directml_mixed_precision: false,
                    ..args.clone()
                },
                true,
                p,
                Some(16.0),
                false,
                false
            )
            .enabled
        );
        assert!(
            !stage2_arena_policy(
                &Args {
                    provider: "cpu".into(),
                    ..args.clone()
                },
                true,
                p,
                Some(16.0),
                true,
                true
            )
            .enabled
        );
        assert!(!stage2_arena_policy(&args, true, p, None, false, true).enabled);
        assert!(stage2_arena_policy(&args, true, p, None, true, true).enabled);
        assert!(
            !stage2_arena_policy(
                &Args {
                    dml_disable_reuse: true,
                    ..args
                },
                true,
                p,
                Some(16.0),
                true,
                true
            )
            .enabled
        );
    }
    #[test]
    fn mixed_estimate_is_conservative_without_overriding_explicit_custom() {
        let args = Args {
            provider: "directml".into(),
            directml_mixed_precision: true,
            ..Default::default()
        };
        assert_eq!(memory_estimate(&args).normal_scale, 1.75);
        assert_eq!(memory_estimate(&args).low_vram_scale, 2.0);
        assert_eq!(
            memory_estimate(&Args {
                directml_mixed_precision: false,
                ..args.clone()
            })
            .normal_scale,
            1.5
        );
        assert_eq!(
            memory_estimate(&Args {
                provider: "cpu".into(),
                ..args.clone()
            })
            .normal_scale,
            1.5
        );
        let automatic = resolve_window(
            WindowRequest::default(),
            1920,
            1080,
            69,
            Some(14.37),
            256,
            memory_estimate(&args),
        )
        .unwrap();
        assert_eq!(automatic.chunk, 8);
        let custom = resolve_window(
            WindowRequest {
                mode: WindowMode::Custom,
                chunk: 8,
                stride: 4,
                low_vram: false,
            },
            1920,
            1080,
            69,
            Some(4.0),
            256,
            memory_estimate(&args),
        )
        .unwrap();
        assert_eq!((custom.chunk, custom.stride), (8, 4));
        assert_eq!(
            choice_json(automatic, &args)["estimate_profile"],
            "directml_mixed_precision"
        );
    }
    #[test]
    fn typed_provider_oom_retries_same_window_without_reuse_only_once() {
        let window = WindowChoice {
            chunk: 8,
            stride: 4,
            low_vram: true,
            estimated_peak_gib: 8.0,
            budget_gib: Some(12.0),
            source: "custom",
        };
        let first = AttemptPolicy {
            window,
            disable_reuse: false,
        };
        let smaller = AttemptPolicy {
            window: WindowChoice {
                chunk: 4,
                stride: 2,
                ..window
            },
            disable_reuse: false,
        };
        let mut pending = VecDeque::from([smaller]);
        let error = anyhow::Error::from(ort::Error::new("8007000E allocation failure")).context(
            ReuseInAttempt {
                stage1: false,
                stage2: true,
            },
        );
        assert!(retry_without_reuse(first, &error, &mut pending));
        let retry = pending.pop_front().unwrap();
        assert!(retry.disable_reuse);
        assert_eq!(
            (
                retry.window.chunk,
                retry.window.stride,
                retry.window.low_vram
            ),
            (8, 4, true)
        );
        assert!(!retry_without_reuse(retry, &error, &mut pending));
        assert_eq!(pending.len(), 1);
        assert!(pending.front().unwrap().disable_reuse);
        assert_eq!(pending.front().unwrap().window.chunk, 4);
        let disk = anyhow::anyhow!("disk error while writing source 8007000E.png").context(
            ReuseInAttempt {
                stage1: true,
                stage2: true,
            },
        );
        assert!(!retry_without_reuse(first, &disk, &mut pending));
        let not_reused = anyhow::Error::from(ort::Error::new("8007000E allocation failure"));
        assert!(!retry_without_reuse(first, &not_reused, &mut pending));
        assert_eq!(pending.len(), 1);
    }
    #[test]
    fn larger_encoder_query_contract_requires_exact_developer_exception() {
        let mut contract = serde_json::json!({"bounded_inference": {
            "encoder_frame_chunk":1,"mlp_chunk_tokens":65536,
            "swin_window_batch":16,"query_chunk_tokens":512,
            "default_query_chunk_tokens":128,"window_mask_queries_bounded":true,
            "query_chunk_overrides": {
                "/model/enc_16.0/attn/Loop":512,
                "/model/enc_16.1/attn/Loop":512
            }
        }});
        assert!(check_bounded_contract_with_experiment(&contract, false).is_err());
        check_bounded_contract_with_experiment(&contract, true).unwrap();
        for field in ["default_query_chunk_tokens", "query_chunk_tokens"] {
            let mut changed = contract.clone();
            changed["bounded_inference"][field] = 1024.into();
            assert!(check_bounded_contract_with_experiment(&changed, true).is_err());
        }
        let overrides = &mut contract["bounded_inference"]["query_chunk_overrides"];
        overrides["/model/dec_16.2/attn/Loop"] = 512.into();
        assert!(check_bounded_contract_with_experiment(&contract, true).is_err());
        let mut ordinary = contract;
        ordinary["bounded_inference"]["query_chunk_tokens"] = 128.into();
        ordinary["bounded_inference"]
            .as_object_mut()
            .unwrap()
            .remove("query_chunk_overrides");
        check_bounded_contract_with_experiment(&ordinary, false).unwrap();
    }

    fn profile_test_pack() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "brp-profile-contract-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("pack.json"), br#"{"schema_version":2,"kind":"corridorkey_dynamic","execution_profiles":{"directml_mixed_precision":true}}"#).unwrap();
        for component in ["stage1", "stage2"] {
            let directory = root.join(component);
            fs::create_dir(&directory).unwrap();
            // Contract-only test: no session is initialized and no reference tensors are fabricated.
            for file in ["model.onnx", "model.directml.onnx", "model.onnx.data"] {
                fs::write(directory.join(file), b"contract fixture").unwrap();
            }
            let mut value = serde_json::json!({"contract":{"precision":"float32"}});
            if component == "stage1" {
                value["inputs"] = serde_json::json!(
                    [
                        "rgb",
                        "hint",
                        "moge_feat",
                        "cradio_feat",
                        "rvm_feat",
                        "hint_quality"
                    ]
                    .map(|name| serde_json::json!({"name":name,"shape":[1,-1]}))
                );
                let contract = &mut value["contract"];
                contract["dynamic_temporal"] = true.into();
                contract["temporal_min"] = 1.into();
                contract["temporal_max"] = 256.into();
                contract["bounded_inference"] = serde_json::json!({"encoder_frame_chunk":1,"mlp_chunk_tokens":65536,
                    "swin_window_batch":16,"query_chunk_tokens":128,"window_mask_queries_bounded":true,
                    "complete_swin_window_batch":16,"cswin_strip_batch":16});
            }
            fs::write(
                directory.join("manifest.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
            value["contract"]["precision"] = "mixed_float16_compute_float32_io".into();
            fs::write(
                directory.join("manifest.directml.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }
        root
    }
    #[test]
    fn directml_profile_is_desktop_only_and_keeps_foundations_on_base() {
        let root = profile_test_pack();
        for (command, provider) in [
            ("verify", "directml"),
            ("pipeline", "directml"),
            ("ck-video", "directml"),
            ("desktop", "cpu"),
        ] {
            let mut args = Args {
                command: command.into(),
                provider: provider.into(),
                directml_mixed_precision: true,
                ..Default::default()
            };
            configure_desktop_model_profile(&mut args, &root).unwrap();
            assert!(!args.directml_mixed_precision);
            assert_eq!(
                crate::pipeline::graph_model_path(&args, &root, "stage1"),
                root.join("stage1/model.onnx")
            );
        }
        let mut args = Args {
            command: "desktop".into(),
            provider: "directml".into(),
            ..Default::default()
        };
        configure_desktop_model_profile(&mut args, &root).unwrap();
        assert!(args.directml_mixed_precision);
        for name in ["stage1", "stage2"] {
            assert_eq!(
                crate::pipeline::graph_model_path(&args, &root, name),
                root.join(name).join("model.directml.onnx")
            );
        }
        assert_eq!(
            crate::pipeline::graph_model_path(&args, &root, "moge"),
            root.join("moge/model.onnx")
        );
        fs::remove_file(root.join("pack.json")).unwrap();
        configure_desktop_model_profile(&mut args, &root).unwrap();
        assert!(!args.directml_mixed_precision);
        assert_eq!(
            crate::pipeline::graph_model_path(&args, &root, "stage2"),
            root.join("stage2/model.onnx")
        );
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn incomplete_or_mismatched_directml_profile_fails_before_session_creation() {
        let root = profile_test_pack();
        let mut args = Args {
            command: "desktop".into(),
            provider: "directml".into(),
            ..Default::default()
        };
        fs::remove_file(root.join("stage2/model.directml.onnx")).unwrap();
        assert!(
            configure_desktop_model_profile(&mut args, &root)
                .unwrap_err()
                .to_string()
                .contains("stage2/model.directml.onnx")
        );
        assert!(!args.directml_mixed_precision);
        fs::write(root.join("stage2/model.directml.onnx"), b"contract fixture").unwrap();
        let manifest = root.join("stage1/manifest.directml.json");
        let original: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
        let mut changed = original.clone();
        changed["contract"]["temporal_max"] = 128.into();
        fs::write(&manifest, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(
            configure_desktop_model_profile(&mut args, &root)
                .unwrap_err()
                .to_string()
                .contains("temporal_max")
        );
        changed = original.clone();
        changed["contract"]["bounded_inference"]["cswin_strip_batch"] = 17.into();
        fs::write(&manifest, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(configure_desktop_model_profile(&mut args, &root).is_err());
        changed = original;
        changed["contract"]
            .as_object_mut()
            .unwrap()
            .remove("precision");
        fs::write(&manifest, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(
            configure_desktop_model_profile(&mut args, &root)
                .unwrap_err()
                .to_string()
                .contains("precision")
        );
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn allocation_recovery_never_retries_disk_source_or_bad_tensor_errors() {
        for message in [
            "No space left on device",
            "Source metadata changed",
            "missing model.onnx",
            "invalid tensor dimensions",
            "0x887a0005 DXGI_ERROR_DEVICE_REMOVED",
            "opening source D:/Out of Memory/video.mp4",
            "loading model E_OUTOFMEMORY.onnx",
        ] {
            assert!(!allocation_oom(&anyhow::anyhow!(message)));
        }
        for message in [
            "E_OUTOFMEMORY 0x8007000e",
            "CUDA_ERROR_OUT_OF_MEMORY",
            "std::bad_alloc",
            "Failed to allocate memory for requested buffer",
            "Exception(2) tid(123) 8007000E Not enough memory resources are available to complete this operation.",
            "Exception(2) tid(123) 8007000E Recursos de memória insuficientes.",
        ] {
            assert!(allocation_oom(&ort::Error::new(message).into()));
        }
        assert!(allocation_oom(
            &MemoryCapacity("host allocation".into()).into()
        ));
        assert!(allocation_oom(
            &std::io::Error::from(std::io::ErrorKind::OutOfMemory).into()
        ));
        assert!(!allocation_oom(
            &ort::Error::new_with_code(
                ort::ErrorCode::NoSuchFile,
                "D:/8007000E/model.onnx does not exist"
            )
            .into()
        ));
    }
    #[test]
    fn explicit_custom_is_not_rejected_by_an_uncalibrated_gpu_estimate() {
        let memory = crate::gpu_memory::GpuMemorySnapshot {
            adapter_index: 0,
            adapter_name: "test".into(),
            dedicated_video_memory_bytes: 16 << 30,
            local_budget_bytes: 12 << 30,
            current_usage_bytes: 8 << 30,
            available_for_reservation_bytes: 0,
            current_reservation_bytes: 0,
        };
        let choice = WindowChoice {
            chunk: 24,
            stride: 12,
            low_vram: false,
            estimated_peak_gib: 100.0,
            budget_gib: Some(11.5),
            source: "custom",
        };
        assert!(gpu_budget_guard(choice, &memory).is_ok());
        let error = gpu_budget_guard(
            WindowChoice {
                source: "balanced",
                ..choice
            },
            &memory,
        )
        .unwrap_err();
        assert!(!allocation_oom(&error));
        assert_eq!(
            error.downcast_ref::<CapacityGuard>().unwrap().reason,
            "memory_budget_reselection"
        );
    }
    #[test]
    fn runtime_phase_records_absolute_source_indices_without_gpu_initialization() {
        let directory =
            std::env::temp_dir().join(format!("brp-phase-trace-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let mut trace = PhaseTrace::open(
            &directory,
            &FrameRange {
                start: 46,
                end_exclusive: 50,
            },
            Instant::now(),
            "cpu",
        )
        .unwrap();
        trace
            .record(
                &Args {
                    provider: "cpu".into(),
                    ..Default::default()
                },
                "stage1_window_begin",
                2,
                4,
            )
            .unwrap();
        drop(trace);
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("memory-phases.jsonl")).unwrap())
                .unwrap();
        assert_eq!(value["source_frame"], 48);
        assert!(value["gpu_memory"].is_null());
        let summary: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("memory-summary.json")).unwrap())
                .unwrap();
        assert!(summary["background_samples"].is_null());
        fs::remove_file(directory.join("memory-phases.jsonl")).unwrap();
        fs::remove_file(directory.join("memory-summary.json")).unwrap();
        fs::remove_dir(directory).unwrap();
    }
    #[test]
    fn stage1_evidence_preserves_bits_and_window_coordinates_without_tensor_dumps() {
        let directory = std::env::temp_dir().join(format!(
            "brp-stage1-evidence-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let window = ck_core::windows::Window {
            index: 2,
            start: 8,
            end: 16,
            weights: vec![1.0; 8],
            finalize_before: 16,
        };
        let baton = tensor(&[2], vec![0.0, -0.0]).unwrap();
        let auxiliary = tensor(&[2], vec![0.0, 0.0]).unwrap();
        write_stage1_observation(&directory, &window, &baton, &auxiliary).unwrap();
        let path = directory.join("validation-stage1/window-000002.json");
        let report: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(report["start"], 8);
        assert_eq!(report["end_exclusive"], 16);
        assert_eq!(report["outputs"][0]["finite_elements"], 2);
        assert_ne!(
            report["outputs"][0]["f32_le_bytes_sha256"],
            report["outputs"][1]["f32_le_bytes_sha256"]
        );
        assert_eq!(baton.as_slice().unwrap()[1].to_bits(), (-0.0f32).to_bits());
        // Evidence must never silently overwrite a prior window observation.
        assert!(write_stage1_observation(&directory, &window, &baton, &auxiliary).is_err());
        let nonfinite = tensor(&[1], vec![f32::NAN]).unwrap();
        assert!(write_stage1_observation(&directory, &window, &nonfinite, &auxiliary).is_err());
        assert_eq!(
            fs::read_dir(directory.join("validation-stage1"))
                .unwrap()
                .count(),
            1
        );
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory.join("validation-stage1")).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn developer_raw_capture_preserves_exact_logits_and_absolute_indices() {
        let directory =
            std::env::temp_dir().join(format!("brp-validation-capture-{}", std::process::id()));
        let mut writer = OutputWriter {
            padding: Padding::new(2, 2, 128).unwrap(),
            threads: 1,
            alpha: None,
            checker: None,
            rendered: 0,
            first_frame_seconds: None,
            started: Instant::now(),
            total: 4,
            validation_logits: Some(directory.clone()),
            source_start: 46,
        };
        let logits = ArrayD::from_elem(IxDyn(&[1, 1, 4, 128, 128]), 0.25f32);
        writer.frame(1, logits.clone()).unwrap();
        assert_eq!(read_npy(directory.join("000047.npy")).unwrap(), logits);
        assert!(!directory.join("000001.npy").exists());
        fs::remove_file(directory.join("000047.npy")).unwrap();
        fs::remove_dir(directory).unwrap();
    }
    #[test]
    fn finalized_cache_preserves_nonzero_absolute_local_frame_order() {
        let mut cache = FrameCache::default();
        for index in 4..7 {
            cache.push(CachedFrame {
                index,
                rgb: vec![index as f32],
                features: BTreeMap::new(),
            });
        }
        assert_eq!(cache.get(5).unwrap().rgb, [5.0]);
        assert!(cache.take(5).is_err());
        assert_eq!(cache.take(4).unwrap().index, 4);
        assert_eq!(cache.get(5).unwrap().index, 5);
        assert_eq!(cache.peak_frames, 3);
    }
    #[test]
    fn output_drop_joins_all_prior_writes_before_a_retry_can_start() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let completed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let written = completed.clone();
        let worker = std::thread::spawn(move || {
            while let Ok((index, _logits)) = receiver.recv() {
                written.store(index + 1, std::sync::atomic::Ordering::Release);
            }
            Ok((1, None))
        });
        sender.send((46, ArrayD::zeros(IxDyn(&[1])))).unwrap();
        let render = Render {
            sender: Some(sender),
            worker: Some(worker),
            padding: Padding::new(1, 1, 128).unwrap(),
        };
        drop(render);
        assert_eq!(completed.load(std::sync::atomic::Ordering::Acquire), 47);
    }
    #[test]
    fn output_writer_failure_is_returned_instead_of_an_allocation_retry() {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let worker = std::thread::spawn(move || {
            drop(receiver);
            anyhow::bail!("disk-capacity failure during PNG output")
        });
        let render = Render {
            sender: Some(sender),
            worker: Some(worker),
            padding: Padding::new(1, 1, 128).unwrap(),
        };
        let error = render.finish().unwrap_err();
        assert!(error.to_string().contains("disk-capacity"));
        assert!(!allocation_oom(&error));
    }
    #[test]
    fn auto_window_never_grows_with_memory_and_keeps_a_minimum_of_four() {
        let mut args = Args {
            provider: "directml".into(),
            directml_mixed_precision: true,
            ..Args::default()
        };
        args.command = "desktop".into();
        let request = WindowRequest {
            mode: WindowMode::Balanced,
            chunk: 0,
            stride: 0,
            low_vram: false,
        };
        let p1080 = Padding::new(1080, 1920, 128).unwrap();
        let pick = |native: bool, budget: f64, p: Padding| {
            auto_window_choice(native, &args, request, p, 100, Some(budget), 256)
        };
        // native: 24 GB and 16 GB cards both get the default 8-frame window, resident
        for budget in [22.3, 14.37, 9.0] {
            let c = pick(true, budget, p1080);
            assert_eq!(
                (c.chunk, c.stride, c.low_vram),
                (8, 4, false),
                "budget {budget}"
            );
        }
        // 8 GB / 6 GB cards: staged before shrinking below the default where staged fits
        let c = pick(true, 6.7, p1080);
        assert_eq!(
            (c.chunk, c.low_vram, c.source),
            (8, true, "auto_native_staged")
        );
        let c = pick(true, 4.5, p1080);
        assert_eq!((c.chunk, c.low_vram), (8, true));
        // nothing fits: the minimum window is 4, staged
        let c = pick(true, 2.0, p1080);
        assert_eq!(
            (c.chunk, c.stride, c.low_vram, c.source),
            (4, 2, true, "auto_minimum_window")
        );
        // ORT keeps 8 on large cards instead of the old 24-frame choice
        let c = pick(false, 22.3, p1080);
        assert_eq!((c.chunk, c.low_vram), (8, false));
        // native 4K: the fold ceiling limits the window to 6
        let p4k = Padding::new(2160, 3840, 128).unwrap();
        assert_eq!(native_window_limit(p4k), 6);
        assert_eq!(native_window_limit(p1080), 12);
        assert!(pick(true, 30.0, p4k).chunk <= 6);
        // explicit Low VRAM stays staged
        let low = WindowRequest {
            low_vram: true,
            ..request
        };
        assert!(auto_window_choice(true, &args, low, p1080, 100, Some(22.3), 256).low_vram);
        // short clips keep the nominal chunk; the estimate uses the actual frames
        let short = auto_window_choice(true, &args, request, p1080, 3, Some(22.3), 256);
        assert_eq!(short.chunk, 8);
    }

    #[test]
    fn automatic_staged_overlap_needs_directml_mixed_precision_and_budget() {
        let directml = Args {
            provider: "directml".into(),
            directml_mixed_precision: true,
            ..Default::default()
        };
        let staged = WindowChoice {
            chunk: 8,
            stride: 4,
            low_vram: true,
            estimated_peak_gib: 8.0,
            budget_gib: Some(14.0),
            source: "balanced_staged_reuse",
        };
        let overlap = session_load_overlap(&directml, &staged, Some(14.37));
        assert!(overlap.parallel_stage1_load);
        assert!(overlap.stage2_prefetch);
        assert_eq!(overlap.reason, "overlapped_auto_staged");
        // Prefetch needs a queryable budget with headroom; otherwise the
        // portable sequential order is retained.
        let low = session_load_overlap(&directml, &staged, Some(12.0));
        assert!(low.parallel_stage1_load);
        assert!(!low.stage2_prefetch);
        let unknown = session_load_overlap(&directml, &staged, None);
        assert!(unknown.parallel_stage1_load);
        assert!(!unknown.stage2_prefetch);
        // Explicit Low VRAM keeps the sequential minimum-memory order.
        let explicit = Args {
            explicit_low_vram: true,
            ..directml.clone()
        };
        let explicit_overlap = session_load_overlap(&explicit, &staged, Some(14.37));
        assert!(!explicit_overlap.parallel_stage1_load);
        assert!(!explicit_overlap.stage2_prefetch);
        // Non-DirectML providers and the FP32 base profile stay sequential.
        let cpu = Args {
            provider: "cpu".into(),
            ..Default::default()
        };
        assert!(!session_load_overlap(&cpu, &staged, Some(14.37)).parallel_stage1_load);
        let base = Args {
            provider: "directml".into(),
            directml_mixed_precision: false,
            ..Default::default()
        };
        assert!(!session_load_overlap(&base, &staged, Some(14.37)).parallel_stage1_load);
        // Resident streaming already loads Stage2 up front, so it must not
        // prefetch a second session.
        let resident = WindowChoice {
            low_vram: false,
            ..staged
        };
        let resident_overlap = session_load_overlap(&directml, &resident, Some(14.37));
        assert!(resident_overlap.parallel_stage1_load);
        assert!(!resident_overlap.stage2_prefetch);
        assert_eq!(
            resident_overlap.reason,
            "resident_streaming_loads_stage2_up_front"
        );
    }
    #[test]
    fn cleanup_preserves_unrelated_files_and_rejects_wrong_attempt_ownership() {
        let root = std::env::temp_dir().join(format!(
            "brp-owned-attempt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("user-file.txt"), b"preserve").unwrap();
        let attempt = new_attempt(&root, "job-a", 0).unwrap();
        fs::write(attempt.join("partial.png"), b"generated").unwrap();
        fs::create_dir(attempt.join("work")).unwrap();
        fs::write(attempt.join("work/tensor.npy"), b"scratch").unwrap();
        assert!(remove_owned_work(&attempt, "job-b").is_err());
        remove_owned_work(&attempt, "job-a").unwrap();
        assert!(!attempt.join("work").exists());
        assert!(attempt.join("partial.png").is_file());
        assert!(remove_owned_attempt(&root, &attempt, "job-b").is_err());
        assert!(attempt.join("partial.png").is_file());
        assert!(remove_owned_attempt(&root, &root, "job-a").is_err());
        remove_owned_attempt(&root, &attempt, "job-a").unwrap();
        assert_eq!(fs::read(root.join("user-file.txt")).unwrap(), b"preserve");
        fs::remove_file(root.join("user-file.txt")).unwrap();
        fs::remove_dir(root).unwrap();
    }
}
