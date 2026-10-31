//! Native static-plan backend for the Stage1/Stage2 pipeline graphs.
//!
//! Active on DirectML jobs when the native models are installed beside the worker
//! (`models/corridorkey/native/stage{1,2}/dyn16.onnx`) or `BRP_CHROMA_NATIVE_PLAN=1`;
//! `BRP_CHROMA_NATIVE_PLAN=0` always uses ONNX Runtime. Each stage keeps ONE live
//! executor for the current geometry; plans are specialised on first use per geometry and
//! cached on disk (at most `KEEP_PER_STAGE` geometries per stage). Any failure is returned
//! to the caller, which falls back to the ONNX Runtime session.
//!
//! Environment:
//! - `BRP_CHROMA_NATIVE_PLAN_MODELS`: folder with `stage1/dyn16.onnx` and `stage2/dyn16.onnx`
//!   (symbolic-geometry exports, external data beside them; default
//!   `<worker>/models/corridorkey/native`).
//! - `BRP_CHROMA_NATIVE_PLAN_EXECUTOR`: `ck_plan_executor.dll` (default: beside the worker).
//! - `BRP_CHROMA_NATIVE_PLAN_DIRECTML`: `DirectML.dll` (default: beside the worker).
//! - `BRP_CHROMA_NATIVE_PLAN_CACHE`: plan cache (default `<worker>/work/native-plans` when the
//!   app has a `work` folder, else `%LOCALAPPDATA%/BRP-Chroma/native-plans`).

use super::executor::PlanExecutor;
use super::specialise;
use anyhow::{Context, Result, bail, ensure};
use ndarray::{ArrayD, IxDyn};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const KEEP_PER_STAGE: usize = 2;

/// Debug aid (`CK_NATIVE_MEM=1`): process GPU usage at native setup/run boundaries.
fn trace_mem(stage: &str, what: &str) {
    if std::env::var_os("CK_NATIVE_MEM").is_none() {
        return;
    }
    if let Ok(Some(m)) = crate::gpu_memory::snapshot("directml") {
        eprintln!(
            "native-mem {stage} {what} {:.3} GiB",
            m.current_usage_bytes as f64 / (1u64 << 30) as f64
        );
    }
}
/// Bump when the planner or plan text format changes so stale cached plans are rebuilt.
const PLAN_FORMAT: u32 = 2;

#[derive(Default, Serialize, Clone)]
pub struct NativeReport {
    pub model: String,
    pub specialisations: Vec<SpecialiseRecord>,
    pub executor_setups: usize,
    pub executor_setup_seconds: f64,
    pub device_bytes: u64,
    pub gpu_ms: Vec<f64>,
    pub fallback: Option<String>,
}

#[derive(Serialize, Clone)]
pub struct SpecialiseRecord {
    pub geometry: [i64; 3],
    pub cached: bool,
    pub seconds: f64,
}

pub struct NativeStage {
    stage: String,
    model: PathBuf,
    cache: PathBuf,
    executor_dll: PathBuf,
    directml_dll: PathBuf,
    device: i32,
    current: Option<([i64; 3], PlanExecutor)>,
    pub report: NativeReport,
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Native plan models shipped beside the worker.
fn bundled_models(exe_dir: &Path) -> PathBuf {
    exe_dir.join("models").join("corridorkey").join("native")
}

impl NativeStage {
    /// `None` when the native path is not enabled for this stage.
    pub fn from_env(stage: &str, provider: &str) -> Result<Option<Self>> {
        if provider != "directml" || !matches!(stage, "stage1" | "stage2") {
            return Ok(None);
        }
        let exe_dir = std::env::current_exe()?
            .parent()
            .context("worker folder")?
            .to_path_buf();
        let models =
            env_path("BRP_CHROMA_NATIVE_PLAN_MODELS").unwrap_or_else(|| bundled_models(&exe_dir));
        let installed = ["stage1", "stage2"]
            .iter()
            .all(|s| models.join(s).join("dyn16.onnx").is_file());
        match std::env::var("BRP_CHROMA_NATIVE_PLAN").ok().as_deref() {
            Some("0") => return Ok(None),
            Some("1") => {}
            _ if installed => {}
            _ => return Ok(None),
        }
        let model = models.join(stage).join("dyn16.onnx");
        ensure!(
            model.is_file(),
            "native plan model {} missing",
            model.display()
        );
        let executor_dll = env_path("BRP_CHROMA_NATIVE_PLAN_EXECUTOR")
            .unwrap_or_else(|| exe_dir.join("ck_plan_executor.dll"));
        let directml_dll = env_path("BRP_CHROMA_NATIVE_PLAN_DIRECTML")
            .unwrap_or_else(|| exe_dir.join("DirectML.dll"));
        ensure!(
            executor_dll.is_file() && directml_dll.is_file(),
            "native plan executor or DirectML library missing"
        );
        let cache = match env_path("BRP_CHROMA_NATIVE_PLAN_CACHE") {
            Some(c) => c,
            None if exe_dir.join("work").is_dir() => exe_dir.join("work").join("native-plans"),
            None => env_path("LOCALAPPDATA")
                .context("LOCALAPPDATA")?
                .join("BRP-Chroma")
                .join("native-plans"),
        };
        Ok(Some(Self {
            stage: stage.to_owned(),
            model: std::path::absolute(&model)?,
            cache: std::path::absolute(cache.join(stage))?,
            executor_dll: std::path::absolute(executor_dll)?,
            directml_dll: std::path::absolute(directml_dll)?,
            device: ck_desktop_core::selected_device_id()? as i32,
            current: None,
            report: NativeReport {
                model: model.display().to_string(),
                ..Default::default()
            },
        }))
    }

    /// Identity of the symbolic model, so a re-exported model invalidates cached plans.
    fn model_stamp(&self) -> Result<String> {
        let mut parts = vec![format!("format={PLAN_FORMAT}")];
        for p in [self.model.clone(), self.model.with_extension("onnx.data")] {
            if let Ok(m) = fs::metadata(&p) {
                let t = m
                    .modified()?
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs();
                parts.push(format!(
                    "{}:{}:{t}",
                    p.file_name().unwrap_or_default().to_string_lossy(),
                    m.len()
                ));
            }
        }
        Ok(parts.join(";"))
    }

    fn plan_dir(&mut self, geo: [i64; 3]) -> Result<PathBuf> {
        let dir = self.cache.join(format!("{}x{}x{}", geo[0], geo[1], geo[2]));
        let stamp_file = dir.join("ready.txt");
        let stamp = self.model_stamp()?;
        let started = Instant::now();
        let cached = fs::read_to_string(&stamp_file).is_ok_and(|s| s == stamp);
        if !cached {
            if dir.exists() {
                fs::remove_dir_all(&dir)
                    .with_context(|| format!("clear stale plan {}", dir.display()))?;
            }
            specialise::specialise(&self.model, &dir, geo[0], geo[1], geo[2], self.device)
                .with_context(|| format!("specialising {} for {geo:?}", self.stage))?;
            fs::write(&stamp_file, &stamp)?;
            self.prune(&dir)?;
        } else {
            // refresh recency for pruning
            fs::write(&stamp_file, &stamp)?;
        }
        self.report.specialisations.push(SpecialiseRecord {
            geometry: geo,
            cached,
            seconds: started.elapsed().as_secs_f64(),
        });
        Ok(dir.join("plan"))
    }

    /// Keeps the most recently used geometries of this stage; removes the rest.
    fn prune(&self, keep: &Path) -> Result<()> {
        let mut entries: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();
        for e in fs::read_dir(&self.cache)? {
            let p = e?.path();
            if p == keep || !p.is_dir() {
                continue;
            }
            let t = fs::metadata(p.join("ready.txt"))
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            entries.push((t, p));
        }
        entries.sort_by_key(|e| std::cmp::Reverse(e.0));
        for (_, p) in entries.into_iter().skip(KEEP_PER_STAGE - 1) {
            fs::remove_dir_all(&p).with_context(|| format!("prune plan {}", p.display()))?;
        }
        Ok(())
    }

    pub fn infer(
        &mut self,
        values: &[(&str, ArrayD<f32>)],
    ) -> Result<BTreeMap<String, ArrayD<f32>>> {
        let rgb = values
            .iter()
            .find(|(n, _)| *n == "rgb")
            .context("native stage needs rgb")?
            .1
            .shape();
        ensure!(
            rgb.len() == 5 && rgb[0] == 1,
            "unexpected rgb layout {rgb:?}"
        );
        let geo = [rgb[1] as i64, rgb[3] as i64, rgb[4] as i64];
        if geo[1] % 64 != 0 || geo[2] % 64 != 0 || geo[1] < 256 || geo[2] < 256 {
            bail!("geometry {geo:?} unsupported by the native plan (needs /64 and >= 256 px)");
        }
        if self.current.as_ref().is_none_or(|(g, _)| *g != geo) {
            self.current = None; // release the previous geometry's device memory first
            trace_mem(&self.stage, "before_plan");
            let plan = self.plan_dir(geo)?;
            trace_mem(&self.stage, "before_create");
            let started = Instant::now();
            let exec =
                PlanExecutor::create(&self.executor_dll, &self.directml_dll, &plan, self.device)?;
            self.report.executor_setups += 1;
            self.report.executor_setup_seconds += started.elapsed().as_secs_f64();
            self.report.device_bytes = self.report.device_bytes.max(exec.stats.device_bytes);
            self.current = Some((geo, exec));
            trace_mem(&self.stage, "after_create");
        }
        let exec = &mut self.current.as_mut().unwrap().1;
        let mut owned_inputs = Vec::with_capacity(exec.inputs.len());
        for info in &exec.inputs {
            let value = &values
                .iter()
                .find(|(n, _)| *n == info.name)
                .with_context(|| format!("missing input {}", info.name))?
                .1;
            ensure!(
                value.shape() == info.shape.as_slice(),
                "input {} shape {:?} != plan {:?}",
                info.name,
                value.shape(),
                info.shape
            );
            owned_inputs.push(value.as_standard_layout());
        }
        let ins: Vec<&[f32]> = owned_inputs
            .iter()
            .map(|v| v.as_slice().expect("standard layout"))
            .collect();
        let mut outs: Vec<Vec<f32>> = exec
            .outputs
            .iter()
            .map(|o| vec![0.0; o.elements()])
            .collect();
        let mut out_refs: Vec<&mut [f32]> = outs.iter_mut().map(|v| v.as_mut_slice()).collect();
        let gpu_ms = exec.run(&ins, &mut out_refs)?;
        trace_mem(&self.stage, "after_run");
        if self.report.gpu_ms.len() < 64 {
            self.report.gpu_ms.push(gpu_ms);
        }
        let mut result = BTreeMap::new();
        for (info, data) in exec.outputs.iter().zip(outs) {
            result.insert(
                info.name.clone(),
                ArrayD::from_shape_vec(IxDyn(&info.shape), data)?,
            );
        }
        Ok(result)
    }
}
