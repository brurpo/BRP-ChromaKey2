//! Native WDDM budget information for the adapter selected by DirectML.
//!
//! DXGI CurrentUsage is this process's local allocation usage; it is not total
//! card usage. Budget is the OS-assigned process budget and can change while a
//! job is running. Neither number is a promised model allocation limit.

use serde::Serialize;
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Instant,
};

const SAMPLE_MILLISECONDS: u64 = 100;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct GpuMemorySnapshot {
    pub adapter_index: u32,
    pub adapter_name: String,
    pub dedicated_video_memory_bytes: u64,
    pub local_budget_bytes: u64,
    pub current_usage_bytes: u64,
    pub available_for_reservation_bytes: u64,
    pub current_reservation_bytes: u64,
}

impl GpuMemorySnapshot {
    pub(crate) fn free_budget_bytes(&self) -> u64 {
        self.local_budget_bytes
            .saturating_sub(self.current_usage_bytes)
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub(crate) struct GpuMemorySamples {
    pub sample_count: u64,
    pub peak_process_usage_bytes: u64,
    pub minimum_local_budget_bytes: Option<u64>,
    pub minimum_free_budget_bytes: Option<u64>,
    pub peak_process_usage_by_phase: BTreeMap<String, u64>,
    pub peak_sample: Option<GpuMemoryPeak>,
    pub peak_sample_by_phase: BTreeMap<String, GpuMemoryPeak>,
    pub last_error: Option<String>,
    pub sample_interval_milliseconds: u64,
}

/// Coordinates of an observed high-water sample, not an allocation trace or a
/// guaranteed hardware peak. One record per phase keeps storage bounded by the
/// fixed set of worker phases rather than the footage duration.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct GpuMemoryPeak {
    pub phase: String,
    pub source_frame: Option<u64>,
    pub elapsed_seconds: f64,
    pub current_usage_bytes: u64,
    pub local_budget_bytes: u64,
}

impl GpuMemorySamples {
    #[cfg(any(windows, test))]
    fn record(
        &mut self,
        phase: &str,
        source_frame: Option<u64>,
        elapsed_seconds: f64,
        snapshot: &GpuMemorySnapshot,
    ) {
        self.sample_count += 1;
        let observation = GpuMemoryPeak {
            phase: phase.to_owned(),
            source_frame,
            elapsed_seconds,
            current_usage_bytes: snapshot.current_usage_bytes,
            local_budget_bytes: snapshot.local_budget_bytes,
        };
        if self
            .peak_sample
            .as_ref()
            .is_none_or(|previous| snapshot.current_usage_bytes > previous.current_usage_bytes)
        {
            self.peak_sample = Some(observation.clone());
        }
        let phase_peak = self
            .peak_sample_by_phase
            .entry(phase.to_owned())
            .or_insert_with(|| observation.clone());
        if snapshot.current_usage_bytes > phase_peak.current_usage_bytes {
            *phase_peak = observation;
        }
        self.peak_process_usage_bytes = self
            .peak_process_usage_bytes
            .max(snapshot.current_usage_bytes);
        self.minimum_local_budget_bytes = Some(
            self.minimum_local_budget_bytes
                .map_or(snapshot.local_budget_bytes, |old| {
                    old.min(snapshot.local_budget_bytes)
                }),
        );
        let free = snapshot.free_budget_bytes();
        self.minimum_free_budget_bytes = Some(
            self.minimum_free_budget_bytes
                .map_or(free, |old| old.min(free)),
        );
        self.peak_process_usage_by_phase
            .entry(phase.to_owned())
            .and_modify(|peak| *peak = (*peak).max(snapshot.current_usage_bytes))
            .or_insert(snapshot.current_usage_bytes);
    }
}

#[derive(Default)]
struct SampleState {
    phase: String,
    source_frame: Option<u64>,
    samples: GpuMemorySamples,
}

/// Samples this worker's DXGI usage during NN calls. Phase-end snapshots miss
/// transient workspaces; these samples are still not a guaranteed exact peak.
pub(crate) struct GpuMemoryMonitor {
    stop: Arc<AtomicBool>,
    state: Arc<Mutex<SampleState>>,
    worker: Option<JoinHandle<()>>,
}

impl GpuMemoryMonitor {
    pub(crate) fn start(provider: &str, started: Instant) -> anyhow::Result<Option<Self>> {
        if provider != "directml" || !cfg!(windows) {
            return Ok(None);
        }
        let adapter_index = ck_desktop_core::selected_device_id()?;
        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(SampleState {
            phase: "initialization".into(),
            source_frame: None,
            samples: GpuMemorySamples {
                sample_interval_milliseconds: SAMPLE_MILLISECONDS,
                ..Default::default()
            },
        }));
        let thread_stop = stop.clone();
        let thread_state = state.clone();
        let worker = thread::Builder::new()
            .name("gpu-budget".into())
            .spawn(move || {
                sample_worker(thread_stop, thread_state, adapter_index, started);
            })?;
        Ok(Some(Self {
            stop,
            state,
            worker: Some(worker),
        }))
    }

    pub(crate) fn set_phase(&self, phase: &str, source_frame: Option<u64>) {
        if let Ok(mut state) = self.state.lock() {
            state.phase = phase.to_owned();
            state.source_frame = source_frame;
        }
    }

    pub(crate) fn summary(&self) -> GpuMemorySamples {
        self.state
            .lock()
            .map(|state| state.samples.clone())
            .unwrap_or_else(|_| GpuMemorySamples {
                last_error: Some("GPU memory sampler lock poisoned".into()),
                ..Default::default()
            })
    }
}

impl Drop for GpuMemoryMonitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.thread().unpark();
            let _ = worker.join();
        }
    }
}

#[cfg(not(windows))]
fn sample_worker(
    _stop: Arc<AtomicBool>,
    _state: Arc<Mutex<SampleState>>,
    _adapter_index: u32,
    _started: Instant,
) {
}

#[cfg(windows)]
fn sample_worker(
    stop: Arc<AtomicBool>,
    state: Arc<Mutex<SampleState>>,
    adapter_index: u32,
    started: Instant,
) {
    let run = || -> anyhow::Result<()> {
        // Keep one factory/adapter handle on this thread; queries do not create
        // a new DXGI factory every100ms or require an NVIDIA-specific utility.
        let reader = AdapterMemoryReader::open(adapter_index)?;
        while !stop.load(Ordering::Acquire) {
            if let Ok(mut state) = state.lock() {
                // Read under the phase lock so a completed query cannot be
                // attributed to a new window entered while DXGI was queried.
                let snapshot = reader.read()?;
                let phase = state.phase.clone();
                let source_frame = state.source_frame;
                state.samples.record(
                    &phase,
                    source_frame,
                    started.elapsed().as_secs_f64(),
                    &snapshot,
                );
            }
            thread::park_timeout(std::time::Duration::from_millis(SAMPLE_MILLISECONDS));
        }
        Ok(())
    };
    if let Err(error) = run()
        && let Ok(mut state) = state.lock()
    {
        state.samples.last_error = Some(format!("{error:#}"));
    }
}

pub(crate) fn snapshot(provider: &str) -> anyhow::Result<Option<GpuMemorySnapshot>> {
    if provider != "directml" {
        return Ok(None);
    }
    platform_snapshot(ck_desktop_core::selected_device_id()?)
}

/// Identity of the DXGI adapter DirectML uses for `adapter_index` (same EnumAdapters1 order).
/// `key` changes with the device or the user-mode driver version.
pub(crate) struct AdapterIdentity {
    pub vendor: u32,
    pub name: String,
    pub key: String,
}

#[cfg(windows)]
pub(crate) fn adapter_identity(adapter_index: u32) -> Option<AdapterIdentity> {
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIDevice, IDXGIFactory1};
    use windows::core::Interface;
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let adapter = factory.EnumAdapters1(adapter_index).ok()?;
        let d = adapter.GetDesc1().ok()?;
        let driver = adapter
            .CheckInterfaceSupport(&IDXGIDevice::IID)
            .unwrap_or(0) as u64;
        let end = d
            .Description
            .iter()
            .position(|c| *c == 0)
            .unwrap_or(d.Description.len());
        let name = String::from_utf16_lossy(&d.Description[..end]);
        let key = format!(
            "{:04x}:{:04x}:{:08x}:{:x}:{}.{}.{}.{}",
            d.VendorId,
            d.DeviceId,
            d.SubSysId,
            d.Revision,
            driver >> 48,
            (driver >> 32) & 0xffff,
            (driver >> 16) & 0xffff,
            driver & 0xffff
        );
        Some(AdapterIdentity {
            vendor: d.VendorId,
            name,
            key,
        })
    }
}

#[cfg(not(windows))]
pub(crate) fn adapter_identity(_adapter_index: u32) -> Option<AdapterIdentity> {
    None
}

#[cfg(not(windows))]
fn platform_snapshot(_adapter_index: u32) -> anyhow::Result<Option<GpuMemorySnapshot>> {
    Ok(None)
}

#[cfg(windows)]
fn platform_snapshot(adapter_index: u32) -> anyhow::Result<Option<GpuMemorySnapshot>> {
    Ok(Some(AdapterMemoryReader::open(adapter_index)?.read()?))
}

#[cfg(windows)]
struct AdapterMemoryReader {
    index: u32,
    adapter: windows::Win32::Graphics::Dxgi::IDXGIAdapter3,
    name: String,
    dedicated_bytes: u64,
}

#[cfg(windows)]
impl AdapterMemoryReader {
    fn open(adapter_index: u32) -> anyhow::Result<Self> {
        use anyhow::Context;
        use windows::{
            Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter3, IDXGIFactory1},
            core::Interface,
        };

        // DirectML selects this original adapter index through EnumAdapters1.
        // Query that same adapter, rather than assuming an NVIDIA device or probing
        // an unrelated high-performance adapter with a separate selection policy.
        unsafe {
            let factory: IDXGIFactory1 =
                CreateDXGIFactory1().context("create DXGI budget factory")?;
            let adapter = factory
                .EnumAdapters1(adapter_index)
                .context("select requested DirectML adapter")?;
            let description = adapter
                .GetDesc1()
                .context("read DirectML adapter description")?;
            let budget_adapter: IDXGIAdapter3 =
                adapter.cast().context("query DXGI budget interface")?;
            let name_end = description
                .Description
                .iter()
                .position(|value| *value == 0)
                .unwrap_or(description.Description.len());
            Ok(Self {
                index: adapter_index,
                adapter: budget_adapter,
                name: String::from_utf16_lossy(&description.Description[..name_end]),
                dedicated_bytes: description.DedicatedVideoMemory as u64,
            })
        }
    }

    fn read(&self) -> anyhow::Result<GpuMemorySnapshot> {
        use anyhow::Context;
        use windows::Win32::Graphics::Dxgi::{
            DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
        };
        let mut memory = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        unsafe {
            self.adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut memory)
        }
        .context("query local WDDM memory budget")?;
        Ok(GpuMemorySnapshot {
            adapter_index: self.index,
            adapter_name: self.name.clone(),
            dedicated_video_memory_bytes: self.dedicated_bytes,
            local_budget_bytes: memory.Budget,
            current_usage_bytes: memory.CurrentUsage,
            available_for_reservation_bytes: memory.AvailableForReservation,
            current_reservation_bytes: memory.CurrentReservation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_above_budget_never_wraps_to_an_enormous_free_value() {
        let memory = GpuMemorySnapshot {
            adapter_index: 0,
            adapter_name: "test".into(),
            dedicated_video_memory_bytes: 16,
            local_budget_bytes: 10,
            current_usage_bytes: 11,
            available_for_reservation_bytes: 0,
            current_reservation_bytes: 0,
        };
        assert_eq!(memory.free_budget_bytes(), 0);
    }

    #[test]
    fn cpu_provider_does_not_require_dxgi_or_a_gpu() {
        assert!(snapshot("cpu").unwrap().is_none());
        assert!(
            GpuMemoryMonitor::start("cpu", Instant::now())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn sampled_peak_and_budget_minima_are_separate_from_phase_end_usage() {
        let mut samples = GpuMemorySamples::default();
        let mut memory = GpuMemorySnapshot {
            adapter_index: 0,
            adapter_name: "test".into(),
            dedicated_video_memory_bytes: 16,
            local_budget_bytes: 12,
            current_usage_bytes: 4,
            available_for_reservation_bytes: 0,
            current_reservation_bytes: 0,
        };
        samples.record("stage1", Some(0), 1.0, &memory);
        memory.current_usage_bytes = 10;
        samples.record("stage1", Some(4), 2.0, &memory);
        memory.current_usage_bytes = 3;
        memory.local_budget_bytes = 8;
        samples.record("stage2", Some(9), 3.0, &memory);
        assert_eq!(samples.sample_count, 3);
        assert_eq!(samples.peak_process_usage_bytes, 10);
        assert_eq!(samples.minimum_local_budget_bytes, Some(8));
        assert_eq!(samples.minimum_free_budget_bytes, Some(2));
        assert_eq!(samples.peak_process_usage_by_phase["stage1"], 10);
        assert_eq!(samples.peak_process_usage_by_phase["stage2"], 3);
        let peak = samples.peak_sample.as_ref().unwrap();
        assert_eq!(peak.phase, "stage1");
        assert_eq!(peak.source_frame, Some(4));
        assert_eq!(peak.elapsed_seconds, 2.0);
        assert_eq!(peak.current_usage_bytes, 10);
        assert_eq!(peak.local_budget_bytes, 12);
        assert_eq!(samples.peak_sample_by_phase["stage2"].source_frame, Some(9));
        // Repeated windows do not grow the record set or replace the original
        // high-water coordinates when usage ties. Null means no frame yet.
        memory.current_usage_bytes = 10;
        for frame in 10..10_000 {
            samples.record("stage1", Some(frame), 4.0, &memory);
        }
        assert_eq!(samples.peak_sample_by_phase.len(), 2);
        assert_eq!(samples.peak_sample_by_phase["stage1"].source_frame, Some(4));
        memory.current_usage_bytes = 11;
        samples.record("initialization", None, 5.0, &memory);
        assert_eq!(samples.peak_sample.as_ref().unwrap().source_frame, None);
        assert_eq!(
            samples.peak_sample.as_ref().unwrap().phase,
            "initialization"
        );
        let serialized = serde_json::to_value(&samples).unwrap();
        assert!(serialized["peak_sample"]["source_frame"].is_null());
    }
}
