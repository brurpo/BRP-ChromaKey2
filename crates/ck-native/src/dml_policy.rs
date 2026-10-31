//! DirectML metacommand policy.
//!
//! Metacommands are driver-supplied fast kernels that DirectML uses in place of its own
//! shaders. On an AMD Radeon integrated GPU (driver 32.0.21030) the normalization
//! metacommand returns wrong values (LayerNormalization off by up to 0.1, NaN for
//! zero-variance rows), which made every RVM feature NaN. Instead of guessing by vendor,
//! the first DirectML job on an adapter runs one LayerNormalization with metacommands on
//! and compares it with the CPU provider; metacommands stay on only if it matches. The
//! verdict is cached per adapter and driver version, so the test reruns after a driver
//! update. `BRP_CHROMA_DML_METACOMMANDS=on|off` overrides the test.

use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::OnceLock,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use ort::{
    AsPointer,
    execution_providers::{DirectMLExecutionProvider, ExecutionProvider},
    session::{Session, builder::SessionBuilder},
    value::Tensor,
};
use serde::{Deserialize, Serialize};

pub const METACOMMANDS_ENV: &str = "BRP_CHROMA_DML_METACOMMANDS";
const AMD_VENDOR_ID: u32 = 0x1002;
/// One LayerNormalization [1,541,384], opset 17 (RVM's state norm geometry).
const SELFTEST_MODEL: &[u8] = include_bytes!("dml_selftest.onnx");
const SELFTEST_ROWS: usize = 541;
const SELFTEST_CHANNELS: usize = 384;
/// Correct DirectML agrees with the CPU to ~1e-5; the broken metacommand is off by ~0.1.
const SELFTEST_TOLERANCE: f32 = 1e-2;

fn env_override(env: Option<&str>) -> Result<Option<bool>> {
    match env.map(str::trim) {
        None | Some("") => Ok(None),
        Some("off" | "0") => Ok(Some(true)),
        Some("on" | "1") => Ok(Some(false)),
        Some(other) => bail!("{METACOMMANDS_ENV} must be on or off, not {other:?}"),
    }
}

/// Whether DirectML work on the selected adapter must avoid vendor metacommands.
pub fn metacommands_disabled() -> Result<bool> {
    static CHOICE: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    CHOICE
        .get_or_init(|| decide().map_err(|e| format!("{e:#}")))
        .clone()
        .map_err(anyhow::Error::msg)
}

fn decide() -> Result<bool> {
    if let Some(forced) = env_override(std::env::var(METACOMMANDS_ENV).ok().as_deref())? {
        return Ok(forced);
    }
    let device = ck_desktop_core::selected_device_id()?;
    let Some(adapter) = crate::gpu_memory::adapter_identity(device) else {
        return Ok(false); // no DXGI adapter information (non-Windows): DirectML is not used
    };
    let cache_path = cache_path();
    let mut cache = cache_path
        .as_ref()
        .map(|p| load_cache(p))
        .unwrap_or_default();
    if let Some(entry) = cache.get(&adapter.key) {
        return Ok(!entry.metacommands_ok);
    }
    let started = Instant::now();
    match selftest(device as i32) {
        Ok(max_abs) => {
            let ok = max_abs.is_finite() && max_abs <= SELFTEST_TOLERANCE;
            eprintln!(
                "DirectML metacommand self-test on {}: max error {max_abs:e} in {:.2} s -> metacommands {}",
                adapter.name,
                started.elapsed().as_secs_f64(),
                if ok { "on" } else { "OFF" }
            );
            cache.insert(
                adapter.key.clone(),
                CacheEntry {
                    adapter: adapter.name.clone(),
                    metacommands_ok: ok,
                    max_abs: max_abs.is_finite().then_some(max_abs),
                    tested_unix: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0),
                },
            );
            if let Some(path) = cache_path
                && let Err(error) = save_cache(&path, &cache)
            {
                eprintln!("could not save the DirectML self-test result: {error:#}");
            }
            Ok(!ok)
        }
        Err(error) => {
            // Not cached: retry on the next job. Keep the known-bad vendor safe meanwhile.
            let disable = adapter.vendor == AMD_VENDOR_ID;
            eprintln!(
                "DirectML metacommand self-test failed to run on {} ({error:#}); metacommands {}",
                adapter.name,
                if disable { "OFF (AMD default)" } else { "on" }
            );
            Ok(disable)
        }
    }
}

/// `ck-native dml-selftest [--runtime onnxruntime.dll]`: runs the test on the selected
/// adapter (`BRP_CHROMA_DEVICE_ID`) without reading or writing the cache.
pub fn run_cli(args: &[String]) -> Result<()> {
    let runtime = match args {
        [] => std::env::current_exe()?.with_file_name("onnxruntime.dll"),
        [flag, path] if flag == "--runtime" => PathBuf::from(path),
        _ => bail!("usage: ck-native dml-selftest [--runtime onnxruntime.dll]"),
    };
    ort::init_from(runtime.to_string_lossy()).commit()?;
    let device = ck_desktop_core::selected_device_id()?;
    let adapter = crate::gpu_memory::adapter_identity(device).context("no DXGI adapter")?;
    let started = Instant::now();
    let max_abs = selftest(device as i32)?;
    let report = serde_json::json!({
        "adapter": adapter.name, "key": adapter.key, "device": device,
        "max_abs": max_abs.is_finite().then_some(max_abs), "nonfinite": !max_abs.is_finite(),
        "metacommands_ok": max_abs.is_finite() && max_abs <= SELFTEST_TOLERANCE,
        "seconds": started.elapsed().as_secs_f64(),
    });
    println!("{report}");
    Ok(())
}

/// Runs the LayerNormalization on DirectML (metacommands on) and on the CPU; returns the
/// largest absolute difference, or infinity when DirectML produced a nonfinite value.
fn selftest(device_id: i32) -> Result<f32> {
    let input = selftest_input();
    let run = |builder: SessionBuilder| -> Result<Vec<f32>> {
        let mut session: Session = builder.commit_from_memory(SELFTEST_MODEL)?;
        let x = Tensor::from_array(([1usize, SELFTEST_ROWS, SELFTEST_CHANNELS], input.clone()))?;
        let outputs = session.run(ort::inputs!["x" => x])?;
        let (_, y) = outputs["y"].try_extract_tensor::<f32>()?;
        Ok(y.to_vec())
    };
    let cpu = run(Session::builder()?)?;
    let mut dml_builder = Session::builder()?
        .with_memory_pattern(false)?
        .with_parallel_execution(false)?;
    DirectMLExecutionProvider::default()
        .with_device_id(device_id)
        .register(&mut dml_builder)
        .map_err(ort::Error::from)?;
    let dml = run(dml_builder)?;
    ensure!(cpu.len() == dml.len(), "self-test output size mismatch");
    let mut max_abs = 0f32;
    for (a, b) in cpu.iter().zip(&dml) {
        if !b.is_finite() {
            return Ok(f32::INFINITY);
        }
        max_abs = max_abs.max((a - b).abs());
    }
    Ok(max_abs)
}

/// Row 0 is constant (zero variance: the NaN case); the rest are deterministic values with
/// varied means and scales, like real activations.
fn selftest_input() -> Vec<f32> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut out = Vec::with_capacity(SELFTEST_ROWS * SELFTEST_CHANNELS);
    for row in 0..SELFTEST_ROWS {
        let mean = ((row % 7) as f32 - 3.0) * 0.8;
        let spread = 0.25 + (row % 5) as f32 * 0.5;
        for _ in 0..SELFTEST_CHANNELS {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
            out.push(if row == 0 { 0.0 } else { mean + spread * unit });
        }
    }
    out
}

#[derive(Serialize, Deserialize, Clone)]
struct CacheEntry {
    adapter: String,
    metacommands_ok: bool,
    max_abs: Option<f32>,
    tested_unix: u64,
}

fn cache_path() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|dir| {
        PathBuf::from(dir)
            .join("BRP-Chroma")
            .join("dml-selftest.json")
    })
}

fn load_cache(path: &std::path::Path) -> BTreeMap<String, CacheEntry> {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

fn save_cache(path: &std::path::Path, cache: &BTreeMap<String, CacheEntry>) -> Result<()> {
    let dir = path.parent().context("cache path has no parent")?;
    std::fs::create_dir_all(dir)?;
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, serde_json::to_vec_pretty(cache)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Registers DirectML on `builder` for `device_id`. With metacommands enabled this is the
/// same legacy registration as `DirectMLExecutionProvider`; otherwise it uses ORT's generic
/// provider-options call with `disable_metacommands`.
pub fn register_directml(builder: SessionBuilder, device_id: i32) -> Result<SessionBuilder> {
    let mut builder = builder;
    if !metacommands_disabled()? {
        DirectMLExecutionProvider::default()
            .with_device_id(device_id)
            .register(&mut builder)
            .map_err(ort::Error::from)?;
        return Ok(builder);
    }
    let id = std::ffi::CString::new(device_id.to_string())?;
    let keys = [c"device_id".as_ptr(), c"disable_metacommands".as_ptr()];
    let values = [id.as_ptr(), c"true".as_ptr()];
    // SAFETY: the builder pointer is live, and the key/value arrays outlive the call.
    unsafe {
        let status = (ort::api().SessionOptionsAppendExecutionProvider)(
            builder.ptr_mut(),
            c"DML".as_ptr(),
            keys.as_ptr(),
            values.as_ptr(),
            keys.len(),
        );
        ort::error::status_to_result(status)?;
    }
    Ok(builder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_parses_on_off_and_rejects_other_values() {
        assert_eq!(env_override(None).unwrap(), None);
        assert_eq!(env_override(Some(" ")).unwrap(), None);
        assert_eq!(env_override(Some("off")).unwrap(), Some(true));
        assert_eq!(env_override(Some("on")).unwrap(), Some(false));
        assert!(env_override(Some("maybe")).is_err());
    }

    #[test]
    fn selftest_input_has_a_zero_variance_row_and_varied_rows() {
        let x = selftest_input();
        assert_eq!(x.len(), SELFTEST_ROWS * SELFTEST_CHANNELS);
        assert!(x[..SELFTEST_CHANNELS].iter().all(|&v| v == 0.0));
        let row1 = &x[SELFTEST_CHANNELS..2 * SELFTEST_CHANNELS];
        assert!(row1.iter().any(|&v| v != row1[0]));
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn cache_round_trips() {
        let dir = std::env::temp_dir().join(format!("ck-dml-selftest-{}", std::process::id()));
        let path = dir.join("dml-selftest.json");
        let mut cache = BTreeMap::new();
        cache.insert(
            "1002:164e".into(),
            CacheEntry {
                adapter: "AMD".into(),
                metacommands_ok: false,
                max_abs: None,
                tested_unix: 1,
            },
        );
        save_cache(&path, &cache).unwrap();
        let back = load_cache(&path);
        assert!(!back["1002:164e"].metacommands_ok);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
