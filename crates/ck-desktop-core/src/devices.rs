//! Graphics adapter metadata and the worker's process-local device selection.
//! Enumeration creates no ONNX Runtime sessions or inference/model resources.

use anyhow::{Context, Result, ensure};
use serde::Serialize;

pub const DEVICE_ID_ENV: &str = "BRP_CHROMA_DEVICE_ID";

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct GpuDevice {
    /// Original DXGI EnumAdapters1 index. Software filtering never renumbers it.
    pub id: u32,
    pub name: String,
    pub dedicated_bytes: u64,
}

/// Parse without changing process environment, so caller and CPU tests share
/// one contract. ORT provider APIs accept nonnegative signed 32-bit indices.
pub fn parse_device_id(value: Option<&str>) -> Result<u32> {
    let Some(value) = value else {
        return Ok(0);
    };
    ensure!(
        !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
        "{DEVICE_ID_ENV} must be a nonnegative integer device index"
    );
    let device: u32 = value.parse().context("GPU device index is too large")?;
    ensure!(
        device <= i32::MAX as u32,
        "GPU device index exceeds the native provider limit"
    );
    Ok(device)
}

/// DirectML interprets this as a DXGI index; optional CUDA interprets it as its
/// own device ordinal. Those namespaces must not be interchanged by the UI.
pub fn selected_device_id() -> Result<u32> {
    match std::env::var(DEVICE_ID_ENV) {
        Ok(value) => parse_device_id(Some(&value)),
        Err(std::env::VarError::NotPresent) => Ok(0),
        Err(error) => Err(error).context("Reading native GPU device selection"),
    }
}

#[cfg(windows)]
pub fn enumerate_gpus() -> Result<Vec<GpuDevice>> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND, IDXGIFactory1,
    };
    let factory: IDXGIFactory1 =
        unsafe { CreateDXGIFactory1() }.context("Enumerating Windows graphics adapters")?;
    let mut devices = Vec::new();
    for index in 0..u32::MAX {
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(adapter) => adapter,
            Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(error) => return Err(error).context("Reading Windows graphics adapter"),
        };
        let description =
            unsafe { adapter.GetDesc1() }.context("Reading graphics adapter description")?;
        if description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
            continue;
        }
        let end = description
            .Description
            .iter()
            .position(|character| *character == 0)
            .unwrap_or(description.Description.len());
        devices.push(GpuDevice {
            id: index,
            name: String::from_utf16_lossy(&description.Description[..end]),
            dedicated_bytes: description.DedicatedVideoMemory as u64,
        });
    }
    Ok(devices)
}

#[cfg(not(windows))]
pub fn enumerate_gpus() -> Result<Vec<GpuDevice>> {
    // DirectML is Windows-specific. Do not fabricate a CUDA/Metal ordinal from
    // another platform's unrelated adapter ordering.
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_device_index_defaults_to_zero_and_preserves_explicit_nonzero() {
        assert_eq!(parse_device_id(None).unwrap(), 0);
        assert_eq!(parse_device_id(Some("0")).unwrap(), 0);
        assert_eq!(parse_device_id(Some("2")).unwrap(), 2);
        assert_eq!(
            parse_device_id(Some("2147483647")).unwrap(),
            i32::MAX as u32
        );
    }

    #[test]
    fn invalid_provider_device_indices_do_not_silently_select_another_adapter() {
        for value in [
            "",
            "-1",
            "+1",
            " 1",
            "1 ",
            "1.0",
            "gpu0",
            "2147483648",
            "4294967296",
        ] {
            assert!(parse_device_id(Some(value)).is_err(), "{value}");
        }
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "Explicit metadata-only DXGI enumeration; creates no model or inference session"]
    fn enumerate_windows_gpus_metadata_only() {
        let devices = enumerate_gpus().unwrap();
        assert!(devices.windows(2).all(|pair| pair[0].id < pair[1].id));
        assert!(devices.iter().all(|device| !device.name.is_empty()));
        println!(
            "GPU_ENUMERATION {}",
            serde_json::json!({ "devices":devices, "models_initialized":false, "inference_executed":false })
        );
    }
}
