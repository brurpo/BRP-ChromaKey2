//! Job-start specialisation of a symbolic-geometry stage graph (Rust port of the
//! development `specialise.py`): bake the Swin region ids for one geometry, write the
//! concrete input sizes, constant-fold with ONNX Runtime on the DirectML provider (keeps
//! FP16 arithmetic; the CPU provider would insert FP32 up-casts) and lower to a plan.

use super::onnx::{self, TensorProto};
use super::planner::{Plan, Planner};
use anyhow::{Context, Result, ensure};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Swin window used by every attention block of both stages.
const WINDOW: usize = 16;

/// Region id per token for one shifted (or unshifted) window layout, `[windows, w*w]`.
/// Port of upstream `_shifted_region_ids_2d`; padded tokens are -1.
pub fn region_ids(gh: usize, gw: usize, w: usize, sh: usize, sw: usize) -> Vec<i32> {
    let (hp, wp) = (gh.div_ceil(w) * w, gw.div_ceil(w) * w);
    let mut region = vec![-1i32; hp * wp];
    let real = |r: usize, c: usize| r < gh && c < gw;
    if sh == 0 && sw == 0 {
        for r in 0..gh {
            for c in 0..gw {
                region[r * wp + c] = 0;
            }
        }
    } else {
        let bands = |n: usize, w: usize, s: usize| -> Vec<(usize, usize)> {
            if s > 0 {
                vec![(0, n - w), (n - w, n - s), (n - s, n)]
            } else {
                vec![(0, n)]
            }
        };
        let mut idx = 0;
        for (r0, r1) in bands(hp, w, sh) {
            for &(c0, c1) in &bands(wp, w, sw) {
                for r in r0..r1 {
                    for c in c0..c1 {
                        if real(r, c) {
                            region[r * wp + c] = idx;
                        }
                    }
                }
                idx += 1;
            }
        }
        let old = region.clone();
        for r in 0..hp {
            for c in 0..wp {
                region[r * wp + c] = old[((r + sh) % hp) * wp + (c + sw) % wp];
            }
        }
    }
    let (ny, nx) = (hp / w, wp / w);
    let mut out = Vec::with_capacity(hp * wp);
    for wy in 0..ny {
        for wx in 0..nx {
            for iy in 0..w {
                for ix in 0..w {
                    out.push(region[(wy * w + iy) * wp + wx * w + ix]);
                }
            }
        }
    }
    out
}

/// Concrete values of the symbolic dimensions for one job geometry (padded H, W).
pub fn concrete_dims(frames: i64, height: i64, width: i64) -> Vec<(&'static str, i64)> {
    let (gh, gw) = (height / 16, width / 16);
    let windows = ((gh + 15) / 16) * ((gw + 15) / 16);
    vec![
        ("frames", frames),
        ("height", height),
        ("width", width),
        ("grid_h", gh),
        ("grid_w", gw),
        ("grid64_h", height / 64),
        ("grid64_w", width / 64),
        ("windows", windows),
    ]
}

pub struct Timings {
    pub inject_s: f64,
    pub ort_fold_s: f64,
    pub plan_s: f64,
}

/// Steps 1+2: returns the path of the folded graph (`out/fold16.onnx` + `.data`).
/// ONNX Runtime must already be initialised with the shipped runtime library.
pub fn fold(
    dyn_model: &Path,
    out: &Path,
    frames: i64,
    height: i64,
    width: i64,
    device_id: i32,
) -> Result<(PathBuf, f64, f64)> {
    ensure!(
        height % 64 == 0 && width % 64 == 0,
        "geometry must be padded to a multiple of 64"
    );
    ensure!(
        height / 64 > 3 && width / 64 > 3,
        "native path needs at least 256 px per side"
    );
    fs::create_dir_all(out)?;
    let t0 = Instant::now();
    let mut m = onnx::load_model(dyn_model)?;
    let g = m.graph.as_mut().context("model without graph")?;
    let (gh, gw) = ((height / 16) as usize, (width / 16) as usize);
    // Region ids stay graph inputs through the ORT fold: as constants ORT would expand them
    // into [windows,1,L,L] boolean masks (L = frames*256), O(frames^2) bytes. They are baked
    // into the folded graph afterwards, where the planner reads them as id tables.
    let mut tables = Vec::new();
    for i in &g.input {
        if let Some(shift) = i.name.strip_prefix("region_s") {
            let (sh, sw) = shift
                .split_once('_')
                .with_context(|| format!("bad region input {}", i.name))?;
            let ids = region_ids(gh, gw, WINDOW, sh.parse()?, sw.parse()?);
            let n = ids.len() as i64 / (WINDOW * WINDOW) as i64;
            tables.push(TensorProto {
                dims: vec![n, (WINDOW * WINDOW) as i64],
                data_type: onnx::INT32,
                name: i.name.clone(),
                raw_data: ids.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ..Default::default()
            });
        }
    }
    // Concrete sizes go into the input shapes: ORT's free-dimension override path hits an
    // internal reshape-fusion bug for frames > 1.
    let dims: HashMap<&str, i64> = concrete_dims(frames, height, width).into_iter().collect();
    for i in &mut g.input {
        if let Some(shape) = i
            .r#type
            .as_mut()
            .and_then(|t| t.tensor_type.as_mut())
            .and_then(|t| t.shape.as_mut())
        {
            for d in &mut shape.dim {
                if let Some(v) = d.dim_param.as_deref().and_then(|p| dims.get(p)) {
                    d.dim_value = Some(*v);
                    d.dim_param = None;
                }
            }
        }
    }
    // External data may only be referenced from the model's own folder.
    let dir = dyn_model.parent().context("model path without folder")?;
    let spec = dir.join(format!("spec_{frames}x{height}x{width}.onnx"));
    onnx::save_model(&m, &spec)?;
    drop(m);
    let inject_s = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let folded = out.join("fold16.onnx");
    let result = (|| -> Result<()> {
        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_optimized_model_path(&folded)?
            .with_config_entry(
                "session.optimized_model_external_initializers_file_name",
                "fold16.onnx.data",
            )?
            .with_config_entry(
                "session.optimized_model_external_initializers_min_size_in_bytes",
                "1024",
            )?
            .with_config_entry("ep.dml.disable_graph_fusion", "1")?
            // ORT bug: ReshapeFusion drops a Shape output when frames > 1.
            .with_config_entry("optimization.disable_specified_optimizers", "ReshapeFusion")?
            .with_parallel_execution(false)?
            .with_memory_pattern(false)?;
        let _session =
            crate::dml_policy::register_directml(builder, device_id)?.commit_from_file(&spec)?;
        Ok(())
    })();
    let _ = fs::remove_file(&spec);
    result.context("ONNX Runtime constant folding")?;
    let mut m = onnx::load_model(&folded)?;
    let g = m.graph.as_mut().context("folded model without graph")?;
    for t in tables {
        if let Some(k) = g.input.iter().position(|i| i.name == t.name) {
            g.input.remove(k);
            g.initializer.push(t);
        }
    }
    onnx::save_model(&m, &folded)?;
    Ok((folded, inject_s, t0.elapsed().as_secs_f64()))
}

/// Step 3: lower a folded graph and write `plan.txt` + `weights.bin` into `out`.
pub fn plan_to_dir(folded: &Path, out: &Path, dims: &HashMap<String, i64>) -> Result<Plan> {
    let m = onnx::load_model(folded)?;
    let mut p = Planner::new(&m, folded.parent().unwrap_or(Path::new(".")), dims)?;
    drop(m);
    p.lower()?;
    let plan = p.plan()?;
    fs::create_dir_all(out)?;
    fs::write(out.join("plan.txt"), plan.to_text())?;
    fs::write(out.join("weights.bin"), &plan.weights)?;
    Ok(plan)
}

pub fn specialise(
    dyn_model: &Path,
    out: &Path,
    frames: i64,
    height: i64,
    width: i64,
    device_id: i32,
) -> Result<(Plan, Timings)> {
    let (folded, inject_s, ort_fold_s) = fold(dyn_model, out, frames, height, width, device_id)?;
    let t0 = Instant::now();
    let plan = plan_to_dir(&folded, &out.join("plan"), &HashMap::new());
    // The folded graph is a large transient (Stage1 T8 ~2.3 GB incl. materialised masks).
    if std::env::var_os("CK_KEEP_FOLD").is_none() {
        let _ = fs::remove_file(&folded);
        let _ = fs::remove_file(out.join("fold16.onnx.data"));
    }
    let plan = plan?;
    Ok((
        plan,
        Timings {
            inject_s,
            ort_fold_s,
            plan_s: t0.elapsed().as_secs_f64(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unshifted_regions_mark_padding() {
        let ids = region_ids(20, 16, 16, 0, 0);
        assert_eq!(ids.len(), 2 * 256);
        assert!(ids[..256].iter().all(|&v| v == 0));
        // second window row: 4 real rows of 16, rest padding
        assert!(ids[256..256 + 64].iter().all(|&v| v == 0));
        assert!(ids[256 + 64..].iter().all(|&v| v == -1));
    }

    #[test]
    fn shifted_regions_match_numpy_reference() {
        // Reference values from the NumPy port (specialise.py) for a padded 20x24 grid.
        let ids = region_ids(20, 24, 16, 8, 8);
        assert_eq!(ids.len(), 4 * 256);
        let expect: [&[i32]; 4] = [&[-1, 0, 1, 3, 4], &[-1, 0, 3], &[-1, 0, 1], &[-1, 0]];
        let pads = [64, 160, 128, 192];
        for (w, win) in ids.chunks(256).enumerate() {
            let mut distinct = win.to_vec();
            distinct.sort();
            distinct.dedup();
            assert_eq!(distinct, expect[w], "window {w}");
            assert_eq!(
                win.iter().filter(|&&v| v == -1).count(),
                pads[w],
                "window {w}"
            );
        }
        assert_eq!(ids.iter().map(|&v| v as i64).sum::<i64>(), -96);
    }

    #[test]
    fn concrete_dims_for_1080p() {
        let d: HashMap<_, _> = concrete_dims(8, 1152, 1920).into_iter().collect();
        assert_eq!(d["grid_h"], 72);
        assert_eq!(d["windows"], 5 * 8);
    }
}
