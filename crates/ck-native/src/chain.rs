//! Named ONNX graph composition; computed embeddings stay native and in memory.
use super::npy::write_npy;
use super::{
    Args, TensorSpec, Tolerance, compare, create_session, initialize_runtime, load_tensor,
    pack_path, profile_counts,
};
use anyhow::{Context, Result, ensure};
use ndarray::ArrayD;
use ort::{session::SessionInputValue, value::Tensor};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Deserialize)]
struct Chain {
    schema_version: u32,
    kind: String,
    component: String,
    inputs: Vec<TensorSpec>,
    steps: Vec<Step>,
    outputs: Vec<TensorSpec>,
    tolerance: Tolerance,
    #[serde(default)]
    postprocess: Option<Postprocess>,
}
#[derive(Deserialize)]
struct Postprocess {
    kind: String,
    masks: String,
    original_hw: [usize; 2],
    threshold: f32,
    output: String,
}
#[derive(Deserialize)]
struct Step {
    name: String,
    model: String,
    inputs: Vec<InputLink>,
    outputs: Vec<OutputLink>,
}
#[derive(Deserialize)]
struct InputLink {
    name: String,
    source: String,
}
#[derive(Deserialize)]
struct OutputLink {
    name: String,
    target: String,
}

fn identifier(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-'))
}

pub fn run(args: &Args) -> Result<()> {
    let started = Instant::now();
    let root = args.manifest.parent().unwrap_or(Path::new("."));
    let fixture: Chain = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    ensure!(
        fixture.schema_version == 1 && fixture.kind == "onnx_chain_fixture",
        "unsupported chain fixture"
    );
    ensure!(
        !fixture.steps.is_empty() && !fixture.outputs.is_empty(),
        "chain needs steps and reference outputs"
    );
    let mut declared = HashSet::new();
    for step in &fixture.steps {
        ensure!(
            !step.outputs.is_empty(),
            "every chain step must expose a computed output"
        );
        declared.extend(step.outputs.iter().map(|link| link.target.as_str()));
    }
    if let Some(postprocess) = &fixture.postprocess {
        ensure!(
            declared.contains(postprocess.masks.as_str()),
            "postprocess logits must be computed by a graph"
        );
        declared.insert(postprocess.output.as_str());
    }
    ensure!(
        fixture
            .outputs
            .iter()
            .all(|spec| declared.contains(spec.name.as_str())),
        "final reference tensors must be produced by the graph chain, not supplied as initial inputs"
    );
    let tolerance = Tolerance {
        atol: args.atol.unwrap_or(fixture.tolerance.atol),
        rtol: args.rtol.unwrap_or(fixture.tolerance.rtol),
    };
    ensure!(
        tolerance.atol.is_finite()
            && tolerance.rtol.is_finite()
            && tolerance.atol >= 0.0
            && tolerance.rtol >= 0.0,
        "invalid chain tolerance"
    );
    let mut values: BTreeMap<String, ArrayD<f32>> = BTreeMap::new();
    for input in &fixture.inputs {
        ensure!(
            values
                .insert(input.name.clone(), load_tensor(root, input)?)
                .is_none(),
            "duplicate initial chain input {}",
            input.name
        );
    }
    initialize_runtime(args)?;
    let mut steps = Vec::new();
    for step in &fixture.steps {
        ensure!(
            identifier(&step.name),
            "step names must be safe identifiers"
        );
        let profile = args
            .profile
            .as_ref()
            .map(|prefix| PathBuf::from(format!("{}-{}", prefix.display(), step.name)));
        let load_started = Instant::now();
        let mut session = create_session(args, &pack_path(root, &step.model)?, profile.as_deref())?;
        let load_seconds = load_started.elapsed().as_secs_f64();
        let linked_names: HashSet<_> = step.inputs.iter().map(|link| link.name.as_str()).collect();
        let model_names: HashSet<_> = session
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect();
        ensure!(
            linked_names.len() == step.inputs.len() && linked_names == model_names,
            "{} input links do not match model inputs",
            step.name
        );
        let mut inputs: Vec<(String, SessionInputValue<'static>)> = Vec::new();
        for link in &step.inputs {
            let array = values
                .get(&link.source)
                .with_context(|| format!("{} needs missing tensor {}", step.name, link.source))?
                .clone();
            inputs.push((link.name.clone(), Tensor::from_array(array)?.into()));
        }
        let inference_started = Instant::now();
        let output = session.run(inputs)?;
        let inference_seconds = inference_started.elapsed().as_secs_f64();
        let mut targets = HashSet::new();
        for link in &step.outputs {
            ensure!(
                targets.insert(&link.target),
                "duplicate target {} in step {}",
                link.target,
                step.name
            );
            let tensor = output
                .get(&link.name)
                .with_context(|| format!("{} missing output {}", step.name, link.name))?
                .try_extract_array::<f32>()?
                .to_owned();
            ensure!(
                tensor.iter().all(|value| value.is_finite()),
                "nonfinite chain output {}",
                link.name
            );
            values.insert(link.target.clone(), tensor);
        }
        drop(output);
        let profile = if profile.is_some() {
            Some(session.end_profiling()?)
        } else {
            None
        };
        let providers = profile
            .as_deref()
            .map(profile_counts)
            .transpose()?
            .unwrap_or_default();
        println!(
            "chain {}: load={load_seconds:.3}s inference={inference_seconds:.3}s",
            step.name
        );
        steps.push(serde_json::json!({"name":step.name,"model":step.model,"load_seconds":load_seconds,"inference_seconds":inference_seconds,"profile":profile,"provider_node_counts":providers}));
    }
    if let Some(postprocess) = &fixture.postprocess {
        ensure!(
            postprocess.kind == "sam2_masks"
                && postprocess.threshold.is_finite()
                && identifier(&postprocess.output),
            "unsupported/invalid chain postprocessing"
        );
        let masks = values
            .get(&postprocess.masks)
            .context("SAM postprocess source logits missing")?;
        ensure!(
            masks.ndim() == 4 && masks.shape()[0] == 1,
            "SAM mask logits must have shape [1,M,H,W]"
        );
        let shape = masks.shape();
        let [height, width] = postprocess.original_hw;
        let resized = ck_core::spatial::resize_bilinear_chw(
            masks.as_slice().context("contiguous SAM logits required")?,
            shape[1],
            shape[2],
            shape[3],
            height,
            width,
        )?;
        let binary: Vec<f32> = resized
            .into_iter()
            .map(|value| {
                if value > postprocess.threshold {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let masks = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, shape[1], height, width]), binary)?;
        ensure!(
            values.insert(postprocess.output.clone(), masks).is_none(),
            "SAM postprocess output overwrites an existing tensor"
        );
    }
    if let Some(directory) = &args.output {
        fs::create_dir_all(directory)?;
    }
    let mut metrics = Vec::new();
    let mut names = HashSet::new();
    for spec in &fixture.outputs {
        ensure!(
            identifier(&spec.name) && names.insert(&spec.name),
            "invalid/duplicate reference output name"
        );
        let actual = values
            .get(&spec.name)
            .with_context(|| format!("chain final tensor missing {}", spec.name))?;
        metrics.push(compare(
            &spec.name,
            actual,
            &load_tensor(root, spec)?,
            tolerance,
        )?);
        if let Some(directory) = &args.output {
            write_npy(directory.join(format!("{}.npy", spec.name)), actual)?;
        }
    }
    let passed = metrics.iter().all(|metric| metric.passed);
    let report = serde_json::json!({"schema_version":1,"component":fixture.component,"requested_provider":args.provider,"precision":"float32","cuda_tf32":false,"runtime":ort::info(),"steps":steps,"outputs":metrics,"atol":tolerance.atol,"rtol":tolerance.rtol,"elapsed_seconds":started.elapsed().as_secs_f64(),"passed":passed});
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
    ensure!(passed, "native chain parity failed (report written)");
    Ok(())
}
