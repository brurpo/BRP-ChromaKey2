#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app_history;
mod asset_setup;
mod despill;
mod frame_cache;
mod hann_advice;
mod icons;
mod job_progress;
mod mask_cache;
mod mask_composite;
mod mask_editor;
mod mask_history;
mod mask_overlay;
mod media_tasks;
mod memory_budget;
mod preview;
mod project_store;
#[cfg(test)]
mod project_workflow_tests;
mod ram_cache;
mod range_export;
mod sam_layers;
mod studio;
mod theme;
mod timeline;
mod translation;
mod update_ui;
mod viewport;

use eframe::egui;
use std::path::PathBuf;

fn main() -> eframe::Result {
    let mut workspace = std::env::current_dir().unwrap_or_default();
    let mut open = None;
    let mut screenshot = None;
    let mut software_renderer = false;
    let mut ui_only = false;
    let mut renderer_report = None;
    let mut window_size = None;
    let mut ui_report = None;
    let mut raw = std::env::args().skip(1);
    while let Some(argument) = raw.next() {
        match argument.as_str() {
            "--workspace" => workspace = raw.next().map(PathBuf::from).unwrap_or(workspace),
            "--open" => open = raw.next().map(PathBuf::from),
            "--screenshot" => screenshot = raw.next().map(PathBuf::from),
            "--software-renderer" => software_renderer = true,
            "--ui-only" => ui_only = true,
            "--renderer-report" => renderer_report = raw.next().map(PathBuf::from),
            "--ui-report" => ui_report = raw.next().map(PathBuf::from),
            "--window-size" => {
                window_size = raw.next().and_then(|size| {
                    let (width, height) = size.split_once('x')?;
                    let width = width.parse::<u16>().ok()?;
                    let height = height.parse::<u16>().ok()?;
                    (width >= 800 && height >= 480).then_some([width as f32, height as f32])
                })
            }
            _ => {}
        }
    }
    let mut native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("BRP CHROMA — CorridorKey Studio")
            // The Studio draws its own title bar (toolbar + window buttons).
            .with_decorations(false)
            .with_inner_size(window_size.unwrap_or([1520.0, 980.0]))
            // Maximize after native creation: an initial inner-size request can
            // otherwise leave Windows' maximized flag and client bounds apart.
            .with_maximized(false)
            // Logical points are scaled by Windows DPI. A 1060×720 minimum can
            // exceed a laptop's entire work area and prevent real maximization.
            .with_min_inner_size([480.0, 240.0]),
        renderer: eframe::Renderer::Wgpu,
        persist_window: false,
        ..Default::default()
    };
    if software_renderer {
        let mut setup = eframe::egui_wgpu::WgpuSetupCreateNew::default();
        #[cfg(windows)]
        {
            setup.instance_descriptor.backends = wgpu::Backends::DX12;
        }
        setup.native_adapter_selector = Some(std::sync::Arc::new(move |adapters, surface| {
            let adapter = adapters
                .iter()
                .find(|adapter| {
                    adapter.get_info().device_type == wgpu::DeviceType::Cpu
                        && surface.is_none_or(|surface| adapter.is_surface_supported(surface))
                })
                .cloned()
                .ok_or_else(|| {
                    "No compatible CPU rendering adapter; refusing hardware GPU fallback"
                        .to_string()
                })?;
            if let Some(path) = &renderer_report {
                let info = adapter.get_info();
                let report = serde_json::json!({"name":info.name,"device_type":format!("{:?}",info.device_type),"backend":format!("{:?}",info.backend),"software_only":true});
                std::fs::write(
                    path,
                    serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
            }
            Ok(adapter)
        }));
        native.wgpu_options.wgpu_setup = setup.into();
    }
    eframe::run_native(
        "BRP CHROMA",
        native,
        Box::new(move |creation| {
            Ok(Box::new(studio::Studio::new(
                creation,
                workspace,
                open,
                screenshot,
                ui_only,
                window_size.is_none(),
                ui_report,
            )))
        }),
    )
}
