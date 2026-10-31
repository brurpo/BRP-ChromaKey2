//! Development native static-plan path: specialise a symbolic loop-free stage graph to one
//! job geometry and lower it to the plan run by the raw-DirectML executor.

pub mod executor;
pub mod onnx;
pub mod planner;
pub mod specialise;
pub mod stage;

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::PathBuf;

/// `ck-native native-plan --model dyn16.onnx --geometry T,H,W --out DIR [--runtime DLL] [--device N]`
/// `ck-native native-plan --folded fold16.onnx --out DIR [--dims k=v,...]` (planner only)
/// `ck-native native-plan --run PLAN_DIR --data DIR --executor DLL --directml DLL [--device N] [--reps N]`
pub fn run_cli(args: &[String]) -> Result<()> {
    let mut opts: HashMap<&str, &str> = HashMap::new();
    let mut it = args.iter();
    while let Some(k) = it.next() {
        let Some(name) = k.strip_prefix("--") else {
            bail!("unexpected argument {k}")
        };
        opts.insert(
            name,
            it.next().with_context(|| format!("{k} needs a value"))?,
        );
    }
    if let Some(plan_dir) = opts.get("run") {
        return run_plan(&opts, plan_dir);
    }
    let out = PathBuf::from(opts.get("out").context("--out DIR is required")?);
    let summary = |plan: &planner::Plan| {
        let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
        for op in &plan.ops {
            *kinds.entry(format!("{:?}", op.kind)).or_default() += 1;
        }
        serde_json::json!({"ops": plan.ops.len(), "kinds": kinds, "heap_MiB": plan.heap_bytes as f64 / 1048576.0,
            "naive_MiB": plan.naive_bytes as f64 / 1048576.0, "weights_MiB": plan.weights.len() as f64 / 1048576.0})
    };
    if let Some(folded) = opts.get("folded") {
        let mut dims = HashMap::new();
        for kv in opts
            .get("dims")
            .copied()
            .unwrap_or("")
            .split(',')
            .filter(|s| !s.is_empty())
        {
            let (k, v) = kv.split_once('=').context("--dims expects name=value")?;
            dims.insert(k.to_owned(), v.parse()?);
        }
        let t0 = std::time::Instant::now();
        let plan = specialise::plan_to_dir(PathBuf::from(folded).as_path(), &out, &dims)?;
        println!(
            "{}",
            serde_json::json!({"plan_s": t0.elapsed().as_secs_f64(), "plan": summary(&plan)})
        );
        return Ok(());
    }
    let model = PathBuf::from(
        opts.get("model")
            .context("--model or --folded is required")?,
    );
    let geo: Vec<i64> = opts
        .get("geometry")
        .context("--geometry T,H,W is required")?
        .split(',')
        .map(|v| v.parse())
        .collect::<Result<_, _>>()?;
    let [t, h, w] = geo[..] else {
        bail!("--geometry expects T,H,W")
    };
    match opts.get("runtime") {
        Some(dll) => ort::init_from(*dll)
            .with_name("BRP CHROMA native plan")
            .commit()?,
        None => ort::init().with_name("BRP CHROMA native plan").commit()?,
    };
    let device = match opts.get("device") {
        Some(d) => d.parse()?,
        None => ck_desktop_core::selected_device_id()? as i32,
    };
    let (plan, tm) = specialise::specialise(&model, &out, t, h, w, device)?;
    println!(
        "{}",
        serde_json::json!({"timings": {"inject_s": tm.inject_s, "ort_fold_s": tm.ort_fold_s, "plan_s": tm.plan_s}, "plan": summary(&plan)})
    );
    Ok(())
}

/// Runs a plan through the executor DLL on `DATA/inputs/<name>.npy`; compares with
/// `DATA/ref_<name>.npy` when present.
fn run_plan(opts: &HashMap<&str, &str>, plan_dir: &str) -> Result<()> {
    let abs = |k: &str| -> Result<PathBuf> {
        Ok(std::path::absolute(
            opts.get(k).with_context(|| format!("--{k} is required"))?,
        )?)
    };
    let data = PathBuf::from(opts.get("data").context("--data DIR is required")?);
    let device = match opts.get("device") {
        Some(d) => d.parse()?,
        None => ck_desktop_core::selected_device_id()? as i32,
    };
    let reps: usize = opts
        .get("reps")
        .map(|r| r.parse())
        .transpose()?
        .unwrap_or(3);
    let t0 = std::time::Instant::now();
    let mut exec = executor::PlanExecutor::create(
        &abs("executor")?,
        &abs("directml")?,
        &std::path::absolute(plan_dir)?,
        device,
    )?;
    let setup_s = t0.elapsed().as_secs_f64();
    let inputs = exec
        .inputs
        .iter()
        .map(|i| {
            crate::npy::read_npy(data.join("inputs").join(format!("{}.npy", i.name)))
                .map(|a| a.into_raw_vec_and_offset().0)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut outputs: Vec<Vec<f32>> = exec
        .outputs
        .iter()
        .map(|o| vec![0.0; o.elements()])
        .collect();
    let mut gpu_ms = Vec::new();
    for _ in 0..reps {
        let ins: Vec<&[f32]> = inputs.iter().map(|v| v.as_slice()).collect();
        let mut outs: Vec<&mut [f32]> = outputs.iter_mut().map(|v| v.as_mut_slice()).collect();
        gpu_ms.push(exec.run(&ins, &mut outs)?);
    }
    let mut cmp = serde_json::Map::new();
    for (o, got) in exec.outputs.iter().zip(&outputs) {
        let r = data.join(format!("ref_{}.npy", o.name));
        if !r.is_file() {
            continue;
        }
        let want = crate::npy::read_npy(&r)?;
        let want = want.as_slice().context("reference not contiguous")?;
        let (mut max, mut sum, mut finite) = (0f64, 0f64, true);
        for (a, b) in got.iter().zip(want) {
            finite &= a.is_finite();
            let d = (*a as f64 - *b as f64).abs();
            max = max.max(d);
            sum += d;
        }
        cmp.insert(o.name.clone(), serde_json::json!({"finite": finite, "max_abs": max, "mean_abs": sum / got.len() as f64}));
    }
    println!(
        "{}",
        serde_json::json!({"setup_s": setup_s, "device_bytes": exec.stats.device_bytes, "compile_ms": exec.stats.compile_ms,
            "bind_ms": exec.stats.bind_ms, "gpu_ms": gpu_ms, "vs_reference": cmp})
    );
    Ok(())
}
