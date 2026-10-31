//! Native single-object SAM2 video memory, propagation and point correction.
//! All image, attention, decoder and memory-encoder inference stays in ONNX.
use super::npy::write_npy;
use super::{
    Args, Metrics, TensorSpec, Tolerance, compare, create_session, initialize_runtime, load_tensor,
    pack_path, profile_counts,
};
use anyhow::{Context, Result, ensure};
use ck_core::sam_memory::{MemoryPolicy, MemorySelection, sine_position_encoding};
use ck_core::spatial::{resize_bilinear_antialias_chw, resize_bilinear_chw};
use ndarray::{ArrayD, IxDyn};
use ort::{
    session::{RunOptions, Session, SessionInputValue},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Instant, UNIX_EPOCH},
};

#[derive(Debug, Deserialize)]
struct Config {
    num_maskmem: usize,
    memory_temporal_stride_for_eval: usize,
    max_cond_frames_in_attn: isize,
    max_obj_ptrs_in_encoder: usize,
    use_obj_ptrs_in_encoder: bool,
    add_tpos_enc_to_obj_ptrs: bool,
    proj_tpos_enc_in_obj_ptrs: bool,
    use_signed_tpos_enc_to_obj_ptrs: bool,
    only_obj_ptrs_in_the_past_for_eval: bool,
    directly_add_no_mem_embed: bool,
    multimask_output_in_sam: bool,
    multimask_output_for_tracking: bool,
    multimask_min_pt_num: usize,
    multimask_max_pt_num: usize,
    hidden_dim: usize,
    mem_dim: usize,
    memory_storage_dtype: String,
    fill_hole_area: usize,
    clear_non_cond_mem_around_input: bool,
    add_all_frames_to_correct_as_cond: bool,
    non_overlap_masks_for_mem_enc: bool,
}

#[derive(Deserialize)]
struct FrameInput {
    frame_idx: usize,
    rgb: TensorSpec,
}

#[derive(Deserialize)]
struct Event {
    name: String,
    frame_idx: usize,
    is_init_cond_frame: bool,
    store_as_conditioning: bool,
    point_coords: Option<TensorSpec>,
    point_labels: Option<TensorSpec>,
    outputs: Vec<TensorSpec>,
}

#[derive(Deserialize)]
struct VideoFixture {
    schema_version: u32,
    kind: String,
    image_size: usize,
    original_hw: [usize; 2],
    total_frames: usize,
    point_coordinate_space: String,
    frames: Vec<FrameInput>,
    events: Vec<Event>,
    constants: BTreeMap<String, TensorSpec>,
    config: Config,
    models: BTreeMap<String, String>,
    tolerance: Tolerance,
    #[serde(default)]
    source: serde_json::Value,
}

fn should_dump_tensor(policy: &str, name: &str, passed: bool) -> bool {
    !passed
        || policy == "all"
        || (policy == "masks" && matches!(name, "original_mask" | "low_res_masks"))
}

#[derive(Serialize)]
struct GraphReport {
    name: String,
    calls: usize,
    load_seconds: f64,
    inference_seconds: f64,
    profile: Option<String>,
    provider_node_counts: BTreeMap<String, usize>,
    cuda_arena_shrinkage: bool,
}

struct Graph {
    session: Session,
    name: String,
    calls: usize,
    load_seconds: f64,
    inference_seconds: f64,
    profiling: bool,
    run_options: Option<RunOptions>,
    require_directml: bool,
}

impl Graph {
    fn open(args: &Args, root: &Path, name: &str, relative: &str) -> Result<Self> {
        let profile = args
            .profile
            .as_ref()
            .map(|prefix| PathBuf::from(format!("{}-{name}", prefix.display())));
        let started = Instant::now();
        let session = create_session(args, &pack_path(root, relative)?, profile.as_deref())
            .with_context(|| format!("loading SAM video graph {name}"))?;
        let run_options = if args.provider == "cuda" && name == "memory_attention" {
            // Dynamic memory-token lengths otherwise accumulate unusable arena
            // regions across frames. Public ORT RunOptions release free regions.
            let mut options = RunOptions::new()?;
            options.add_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")?;
            Some(options)
        } else {
            None
        };
        Ok(Self {
            session,
            name: name.into(),
            calls: 0,
            load_seconds: started.elapsed().as_secs_f64(),
            inference_seconds: 0.0,
            profiling: profile.is_some(),
            run_options,
            require_directml: args.provider == "directml",
        })
    }

    fn infer(&mut self, values: Vec<(&str, ArrayD<f32>)>) -> Result<BTreeMap<String, ArrayD<f32>>> {
        let linked: HashSet<_> = values.iter().map(|(name, _)| *name).collect();
        let required: HashSet<_> = self
            .session
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect();
        ensure!(
            linked.len() == values.len() && linked == required,
            "{} input links differ from exported graph",
            self.name
        );
        ensure!(
            values
                .iter()
                .all(|(_, value)| value.iter().all(|value| value.is_finite())),
            "nonfinite {} input",
            self.name
        );
        let inputs: Vec<(String, SessionInputValue<'static>)> = values
            .into_iter()
            .map(|(name, value)| Ok((name.into(), Tensor::from_array(value)?.into())))
            .collect::<Result<_>>()?;
        let names: Vec<_> = self
            .session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();
        let started = Instant::now();
        let outputs = if let Some(options) = &self.run_options {
            self.session.run_with_options(inputs, options)
        } else {
            self.session.run(inputs)
        }
        .with_context(|| format!("running SAM video {}", self.name))?;
        let mut owned = BTreeMap::new();
        for name in names {
            let value = outputs
                .get(&name)
                .context("missing SAM graph output")?
                .try_extract_array::<f32>()?
                .to_owned();
            ensure!(
                value.iter().all(|value| value.is_finite()),
                "nonfinite {} output {name}",
                self.name
            );
            owned.insert(name, value);
        }
        self.calls += 1;
        self.inference_seconds += started.elapsed().as_secs_f64();
        Ok(owned)
    }

    fn finish(mut self) -> Result<GraphReport> {
        let profile = if self.profiling {
            Some(self.session.end_profiling()?)
        } else {
            None
        };
        let provider_node_counts = profile
            .as_deref()
            .map(profile_counts)
            .transpose()?
            .unwrap_or_default();
        if self.require_directml && self.profiling && self.calls > 0 {
            ensure!(
                provider_node_counts
                    .get("DmlExecutionProvider")
                    .copied()
                    .unwrap_or(0)
                    > 0,
                "SAM {} did not execute any DirectML kernels; inspect {}",
                self.name,
                profile.as_deref().unwrap_or("missing provider profile")
            );
        }
        Ok(GraphReport {
            name: self.name,
            calls: self.calls,
            load_seconds: self.load_seconds,
            inference_seconds: self.inference_seconds,
            profile,
            provider_node_counts,
            cuda_arena_shrinkage: self.run_options.is_some(),
        })
    }
}

fn tensor(shape: &[usize], values: Vec<f32>) -> Result<ArrayD<f32>> {
    Ok(ArrayD::from_shape_vec(IxDyn(shape), values)?)
}

fn take(
    values: &mut BTreeMap<String, ArrayD<f32>>,
    name: &str,
    shape: &[usize],
) -> Result<ArrayD<f32>> {
    let value = values
        .remove(name)
        .with_context(|| format!("SAM output {name} missing"))?;
    ensure!(
        value.shape() == shape,
        "SAM {name} shape {:?}, expected {shape:?}",
        value.shape()
    );
    Ok(value)
}

fn identifier(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn binary_overlap(
    actual: &ArrayD<f32>,
    reference: &ArrayD<f32>,
) -> Result<(usize, f64, serde_json::Value)> {
    ensure!(
        actual.shape() == reference.shape(),
        "binary mask shape mismatch"
    );
    let (mut changed, mut intersection, mut union, mut actual_positive, mut reference_positive) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for (&actual, &reference) in actual.iter().zip(reference) {
        ensure!(
            [0.0, 1.0].contains(&actual) && [0.0, 1.0].contains(&reference),
            "binary mask contains nonbinary values"
        );
        let actual = actual > 0.0;
        let reference = reference > 0.0;
        changed += usize::from(actual != reference);
        intersection += usize::from(actual && reference);
        union += usize::from(actual || reference);
        actual_positive += usize::from(actual);
        reference_positive += usize::from(reference);
    }
    let iou = if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    };
    let dice = if actual_positive + reference_positive == 0 {
        1.0
    } else {
        2.0 * intersection as f64 / (actual_positive + reference_positive) as f64
    };
    Ok((
        changed,
        iou,
        serde_json::json!({"elements":actual.len(),"changed_pixels":changed,"intersection_pixels":intersection,"union_pixels":union,"iou":iou,"dice":dice,"exact":changed == 0}),
    ))
}

#[derive(Clone)]
struct Features {
    embed: ArrayD<f32>,
    raw: ArrayD<f32>,
    high0: ArrayD<f32>,
    high1: ArrayD<f32>,
}

#[derive(Clone)]
struct FrameState {
    conditioning: bool,
    memory: StoredMemory,
    position: Arc<ArrayD<f32>>,
    pointer: ArrayD<f32>,
    low_mask: ArrayD<f32>,
}

#[derive(Clone)]
enum MemoryValues {
    Float32(Vec<f32>),
    Bfloat16(Vec<u16>),
}

#[derive(Clone)]
struct StoredMemory {
    shape: Vec<usize>,
    values: MemoryValues,
}

fn bfloat16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn bfloat16_value(value: u16) -> f32 {
    f32::from_bits(u32::from(value) << 16)
}

impl StoredMemory {
    fn new(value: &ArrayD<f32>, dtype: &str) -> Self {
        let values = if dtype == "bfloat16" {
            MemoryValues::Bfloat16(value.iter().copied().map(bfloat16_bits).collect())
        } else {
            MemoryValues::Float32(value.iter().copied().collect())
        };
        Self {
            shape: value.shape().to_vec(),
            values,
        }
    }
    fn at(&self, index: usize) -> f32 {
        match &self.values {
            MemoryValues::Float32(values) => values[index],
            MemoryValues::Bfloat16(values) => bfloat16_value(values[index]),
        }
    }
    fn bytes(&self) -> usize {
        match &self.values {
            MemoryValues::Float32(values) => values.len() * 4,
            MemoryValues::Bfloat16(values) => values.len() * 2,
        }
    }
}

struct Constants {
    no_memory: Vec<f32>,
    memory_temporal: Vec<f32>,
    current_pos: ArrayD<f32>,
    pointer_projection: Option<Vec<f32>>,
    pointer_bias: Option<Vec<f32>>,
}

impl Constants {
    fn load(root: &Path, fixture: &VideoFixture) -> Result<Self> {
        let read = |name: &str| -> Result<ArrayD<f32>> {
            load_tensor(
                root,
                fixture
                    .constants
                    .get(name)
                    .with_context(|| format!("SAM constant {name} missing"))?,
            )
        };
        let cfg = &fixture.config;
        let grid = fixture.image_size / 16;
        let no_memory = read("no_mem_embed")?;
        ensure!(
            no_memory.len() == cfg.hidden_dim,
            "invalid no_mem_embed size"
        );
        let temporal = read("maskmem_tpos_enc")?;
        ensure!(
            temporal.len() == cfg.num_maskmem * cfg.mem_dim,
            "invalid memory temporal encoding size"
        );
        let current_pos = read("current_pos")?;
        ensure!(
            current_pos.shape() == [1, cfg.hidden_dim, grid, grid],
            "invalid image positional encoding shape"
        );
        let (pointer_projection, pointer_bias) = if cfg.proj_tpos_enc_in_obj_ptrs {
            let weight = read("obj_ptr_tpos_proj_weight")?;
            let bias = read("obj_ptr_tpos_proj_bias")?;
            ensure!(
                weight.shape() == [cfg.mem_dim, cfg.hidden_dim] && bias.len() == cfg.mem_dim,
                "invalid pointer projection geometry"
            );
            (
                Some(weight.iter().copied().collect()),
                Some(bias.iter().copied().collect()),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            no_memory: no_memory.iter().copied().collect(),
            memory_temporal: temporal.iter().copied().collect(),
            current_pos,
            pointer_projection,
            pointer_bias,
        })
    }
}

fn pointer_positions(
    selection: &MemorySelection,
    config: &Config,
    constants: &Constants,
) -> Result<Vec<Vec<f32>>> {
    if !config.add_tpos_enc_to_obj_ptrs {
        return Ok(vec![vec![0.0; config.mem_dim]; selection.pointers.len()]);
    }
    let dimension = if config.proj_tpos_enc_in_obj_ptrs {
        config.hidden_dim
    } else {
        config.mem_dim
    };
    ensure!(
        selection.pointer_distance_denominator > 0 || selection.pointers.is_empty(),
        "invalid pointer temporal denominator"
    );
    let positions: Vec<_> = selection
        .pointers
        .iter()
        .map(|source| source.distance as f32 / selection.pointer_distance_denominator as f32)
        .collect();
    let sine = sine_position_encoding(&positions, dimension)?;
    if let (Some(weight), Some(bias)) = (&constants.pointer_projection, &constants.pointer_bias) {
        Ok(sine
            .chunks_exact(dimension)
            .map(|row| {
                (0..config.mem_dim)
                    .map(|output| {
                        let mut value = bias[output];
                        for input in 0..dimension {
                            value += row[input] * weight[output * dimension + input];
                        }
                        value
                    })
                    .collect()
            })
            .collect())
    } else {
        Ok(sine
            .chunks_exact(config.mem_dim)
            .map(|row| row.to_vec())
            .collect())
    }
}

fn memory_tensors(
    selection: &MemorySelection,
    states: &BTreeMap<usize, FrameState>,
    cfg: &Config,
    constants: &Constants,
    grid: usize,
) -> Result<(ArrayD<f32>, ArrayD<f32>, usize)> {
    let spatial = grid * grid;
    let mut memory = Vec::new();
    let mut positions = Vec::new();
    for source in &selection.spatial {
        let state = states
            .get(&source.frame)
            .context("selected SAM spatial memory missing")?;
        ensure!(
            state.memory.shape == [1, cfg.mem_dim, grid, grid]
                && state.position.shape() == state.memory.shape.as_slice(),
            "invalid stored SAM memory geometry"
        );
        let position = state
            .position
            .as_slice()
            .context("contiguous memory position required")?;
        let temporal = &constants.memory_temporal[source.temporal_embedding_index * cfg.mem_dim
            ..(source.temporal_embedding_index + 1) * cfg.mem_dim];
        for pixel in 0..spatial {
            for channel in 0..cfg.mem_dim {
                memory.push(state.memory.at(channel * spatial + pixel));
                positions.push(position[channel * spatial + pixel] + temporal[channel]);
            }
        }
    }
    let mut pointer_tokens = 0;
    if cfg.use_obj_ptrs_in_encoder {
        ensure!(
            cfg.hidden_dim.is_multiple_of(cfg.mem_dim),
            "pointer channels must divide memory channels"
        );
        let encoded_positions = pointer_positions(selection, cfg, constants)?;
        for (source, position) in selection.pointers.iter().zip(encoded_positions) {
            let state = states
                .get(&source.frame)
                .context("selected SAM object pointer missing")?;
            ensure!(
                state.pointer.shape() == [1, cfg.hidden_dim],
                "invalid stored object pointer shape"
            );
            for token in state
                .pointer
                .as_slice()
                .context("contiguous pointer required")?
                .chunks_exact(cfg.mem_dim)
            {
                memory.extend_from_slice(token);
                positions.extend_from_slice(&position);
                pointer_tokens += 1;
            }
        }
    }
    ensure!(
        !memory.is_empty() && memory.len() == positions.len(),
        "empty or malformed SAM memory bank"
    );
    let count = memory.len() / cfg.mem_dim;
    Ok((
        tensor(&[count, 1, cfg.mem_dim], memory)?,
        tensor(&[count, 1, cfg.mem_dim], positions)?,
        pointer_tokens,
    ))
}

pub fn run(args: &Args) -> Result<()> {
    let started = Instant::now();
    let root = args.manifest.parent().unwrap_or(Path::new("."));
    let fixture: VideoFixture = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    let inference = args.command == "desktop";
    ensure!(
        fixture.schema_version == 1
            && fixture.kind
                == if inference {
                    "sam2_video_inference_job"
                } else {
                    "sam2_video_fixture"
                },
        "unsupported SAM video fixture"
    );
    ensure!(
        fixture.image_size == 1024
            && fixture.config.hidden_dim == 256
            && fixture.config.mem_dim == 64,
        "initial SAM2.1-small proof requires 1024/256/64 profile"
    );
    ensure!(
        fixture.config.directly_add_no_mem_embed,
        "initial no-memory attention-token route not supported by this profile"
    );
    ensure!(
        matches!(
            fixture.config.memory_storage_dtype.as_str(),
            "float32" | "bfloat16"
        ) && fixture.config.fill_hole_area == 0,
        "this video proof requires explicit FP32/BF16 memory and disabled connected-component hole filling"
    );
    ensure!(
        !fixture.config.clear_non_cond_mem_around_input
            && !fixture.config.add_all_frames_to_correct_as_cond
            && !fixture.config.non_overlap_masks_for_mem_enc,
        "this single-object core profile does not implement predictor consolidation/multi-object memory cleanup"
    );
    ensure!(
        if inference {
            fixture.total_frames >= 1 && !fixture.events.is_empty()
        } else {
            fixture.total_frames >= 2
                && fixture.frames.len() == fixture.total_frames
                && fixture.events.len() >= 3
        },
        "video proof requires multiple frames and recurrent events"
    );
    ensure!(
        fixture.original_hw.iter().all(|&value| value > 0),
        "positive original video dimensions required"
    );
    ensure!(
        matches!(
            fixture.point_coordinate_space.as_str(),
            "model_pixels" | "original_pixels"
        ),
        "unsupported prompt coordinate space"
    );
    let cfg = &fixture.config;
    ensure!(
        cfg.max_cond_frames_in_attn == -1 || cfg.max_cond_frames_in_attn >= 2,
        "invalid conditioning frame limit"
    );
    let policy = MemoryPolicy {
        num_maskmem: cfg.num_maskmem,
        temporal_stride: cfg.memory_temporal_stride_for_eval,
        max_conditioning_frames: if cfg.max_cond_frames_in_attn == -1 {
            None
        } else {
            Some(cfg.max_cond_frames_in_attn as usize)
        },
        max_object_pointers: cfg.max_obj_ptrs_in_encoder,
        only_past_object_pointers: cfg.only_obj_ptrs_in_the_past_for_eval,
        signed_pointer_distance: cfg.use_signed_tpos_enc_to_obj_ptrs,
    };
    let tolerance = Tolerance {
        atol: args.atol.unwrap_or(fixture.tolerance.atol),
        rtol: args.rtol.unwrap_or(fixture.tolerance.rtol),
    };
    ensure!(
        tolerance.atol.is_finite()
            && tolerance.rtol.is_finite()
            && tolerance.atol >= 0.0
            && tolerance.rtol >= 0.0,
        "invalid SAM video tolerance"
    );
    let mut frame_ids = HashSet::new();
    ensure!(
        fixture.frames.iter().all(
            |frame| frame.frame_idx < fixture.total_frames && frame_ids.insert(frame.frame_idx)
        ),
        "invalid or duplicate SAM video frame index"
    );
    let mut event_names = HashSet::new();
    ensure!(
        fixture.events.iter().all(|event| identifier(&event.name)
            && event_names.insert(event.name.clone())
            && event.frame_idx < fixture.total_frames),
        "invalid or duplicate SAM video event"
    );
    let constants = Constants::load(root, &fixture)?;
    let mut source_clip = if inference {
        let source = fixture
            .source
            .get("desktop_rgb")
            .and_then(serde_json::Value::as_str)
            .context("desktop SAM source missing")?;
        Some(super::video::RawClip::open(
            &pack_path(root, source)?,
            fixture.total_frames,
            fixture.original_hw[0],
            fixture.original_hw[1],
        )?)
    } else if let Some(video) = &args.video {
        let ffmpeg = args
            .ffmpeg
            .as_ref()
            .context("SAM --video requires --ffmpeg")?;
        let work = args
            .output
            .as_ref()
            .context("SAM --video requires --output for its owned work cache")?
            .join("work");
        fs::create_dir_all(&work)?;
        let source_path = work.join("source.rgb24");
        let identity_path = work.join("source_identity.json");
        let metadata = fs::metadata(video)?;
        let identity = serde_json::json!({"path":fs::canonicalize(video)?,"bytes":metadata.len(),
            "modified_nanos":metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos().to_string(),
            "frames":fixture.total_frames,"height":fixture.original_hw[0],"width":fixture.original_hw[1]});
        if source_path.is_file() {
            ensure!(
                identity_path.is_file(),
                "unverified SAM RGB cache; use a fresh output directory"
            );
            let prior: serde_json::Value = serde_json::from_slice(&fs::read(&identity_path)?)?;
            ensure!(
                prior == identity,
                "SAM RGB cache belongs to a different source/configuration; use a fresh output directory"
            );
        }
        super::video::decode_rgb24(
            ffmpeg,
            video,
            &source_path,
            fixture.total_frames,
            fixture.original_hw[0],
            fixture.original_hw[1],
        )?;
        fs::write(identity_path, serde_json::to_vec_pretty(&identity)?)?;
        Some(super::video::RawClip::open(
            &source_path,
            fixture.total_frames,
            fixture.original_hw[0],
            fixture.original_hw[1],
        )?)
    } else {
        None
    };
    initialize_runtime(args)?;
    let model_root = if inference {
        args.models_root
            .as_deref()
            .context("SAM model root missing")?
    } else {
        root
    };
    let graph_path = |name: &str| {
        fixture
            .models
            .get(name)
            .with_context(|| format!("model {name} missing from SAM video fixture"))
    };
    let mut image_encoder = Graph::open(
        args,
        model_root,
        "image_encoder",
        graph_path("image_encoder")?,
    )?;
    let mut memory_encoder = Graph::open(
        args,
        model_root,
        "memory_encoder",
        graph_path("memory_encoder")?,
    )?;
    let mut memory_attention = Graph::open(
        args,
        model_root,
        "memory_attention",
        graph_path("memory_attention")?,
    )?;
    let mut heads = Graph::open(args, model_root, "video_heads", graph_path("video_heads")?)?;
    let grid = fixture.image_size / 16;
    let low = fixture.image_size / 4;
    let mut cached: BTreeMap<usize, Features> = BTreeMap::new();
    let mut cache_order = VecDeque::new();
    const IMAGE_CACHE_FRAMES: usize = 2;
    let mut states: BTreeMap<usize, FrameState> = BTreeMap::new();
    let mut shared_memory_position: Option<Arc<ArrayD<f32>>> = None;
    let mut peak_state_bytes = 0;
    let mut peak_state_frames = 0;
    let mut evicted_state_frames = 0;
    let mut conditioning_order = Vec::new();
    let mut event_reports = Vec::new();
    let mut preprocessing_metrics = Vec::new();
    let mut all_passed = true;
    let mut floating_outputs_passed = true;
    let mut binary_changed_pixels = 0usize;
    let mut binary_pixel_comparisons = 0usize;
    let mut minimum_mask_iou = 1.0f64;
    let mut compared_binary_masks = 0usize;
    let mut correction_count = 0;
    for (event_index, event) in fixture.events.iter().enumerate() {
        println!("SAM video event {} frame {}", event.name, event.frame_idx);
        if !cached.contains_key(&event.frame_idx) {
            while cached.len() >= IMAGE_CACHE_FRAMES {
                let oldest = cache_order
                    .pop_front()
                    .context("image cache LRU bookkeeping lost")?;
                cached.remove(&oldest);
            }
            let rgb = if let Some(clip) = &mut source_clip {
                let original = clip.read(event.frame_idx)?;
                let mut resized = resize_bilinear_antialias_chw(
                    &original,
                    3,
                    fixture.original_hw[0],
                    fixture.original_hw[1],
                    fixture.image_size,
                    fixture.image_size,
                )?;
                for value in &mut resized {
                    *value = value.clamp(0.0, 1.0);
                }
                tensor(&[1, 3, fixture.image_size, fixture.image_size], resized)?
            } else {
                let input = fixture
                    .frames
                    .iter()
                    .find(|frame| frame.frame_idx == event.frame_idx)
                    .context("frame input missing")?;
                load_tensor(root, &input.rgb)?
            };
            if args.video.is_some() {
                let reference = fixture
                    .frames
                    .iter()
                    .find(|frame| frame.frame_idx == event.frame_idx)
                    .context("preprocess reference frame missing")?;
                let metric = compare(
                    &format!("input_rgb/frame_{:06}", event.frame_idx),
                    &rgb,
                    &load_tensor(root, &reference.rgb)?,
                    Tolerance {
                        atol: 2e-6,
                        rtol: 2e-6,
                    },
                )?;
                ensure!(
                    metric.passed,
                    "native RGB decode/AA resize differs from pinned input reference: frame {} max_abs={} failed={}",
                    event.frame_idx,
                    metric.max_abs_error,
                    metric.failed_elements
                );
                preprocessing_metrics.push(metric);
            }
            ensure!(
                rgb.shape() == [1, 3, fixture.image_size, fixture.image_size]
                    && rgb.iter().all(|&value| (0.0..=1.0).contains(&value)),
                "SAM encoder RGB must be resized display [1,3,1024,1024] in [0,1]"
            );
            let mut outputs = image_encoder.infer(vec![("rgb", rgb)])?;
            let embed = take(
                &mut outputs,
                "image_embed",
                &[1, cfg.hidden_dim, grid, grid],
            )?;
            let mut raw = embed.clone();
            for (index, value) in raw.iter_mut().enumerate() {
                *value -= constants.no_memory[(index / (grid * grid)) % cfg.hidden_dim];
            }
            let high0 = take(&mut outputs, "high_res_0", &[1, 32, grid * 4, grid * 4])?;
            let high1 = take(&mut outputs, "high_res_1", &[1, 64, grid * 2, grid * 2])?;
            cached.insert(
                event.frame_idx,
                Features {
                    embed,
                    raw,
                    high0,
                    high1,
                },
            );
        }
        cache_order.retain(|&index| index != event.frame_idx);
        cache_order.push_back(event.frame_idx);
        let features = cached
            .get(&event.frame_idx)
            .context("cached image features missing")?;
        ensure!(
            event.point_coords.is_some() == event.point_labels.is_some(),
            "prompt coordinates/labels must be supplied together"
        );
        let has_points = event.point_coords.is_some();
        if event.is_init_cond_frame {
            ensure!(
                has_points && event.store_as_conditioning && !states.contains_key(&event.frame_idx),
                "initial conditioning event requires new-frame point prompt"
            );
        } else if has_points {
            ensure!(
                states.contains_key(&event.frame_idx),
                "correction requires an already inferred frame for native mask feedback"
            );
            correction_count += 1;
        }
        let (conditioned, selected, token_count) = if event.is_init_cond_frame {
            (features.embed.clone(), None, 0)
        } else {
            let non_conditioning: BTreeSet<_> = states
                .iter()
                .filter_map(|(&index, state)| (!state.conditioning).then_some(index))
                .collect();
            let selection = policy.select(
                event.frame_idx,
                fixture.total_frames,
                &conditioning_order,
                &non_conditioning,
            )?;
            let (memory, memory_pos, pointers) =
                memory_tensors(&selection, &states, cfg, &constants, grid)?;
            let mut output = memory_attention.infer(vec![
                ("backbone_features", features.raw.clone()),
                ("current_pos", constants.current_pos.clone()),
                ("memory", memory),
                ("memory_pos", memory_pos),
                ("num_obj_ptr_tokens", tensor(&[1], vec![pointers as f32])?),
            ])?;
            (
                take(
                    &mut output,
                    "conditioned_features",
                    &[1, cfg.hidden_dim, grid, grid],
                )?,
                Some(selection),
                pointers,
            )
        };
        let (mut coordinates, labels, real_points) = if has_points {
            let coords = load_tensor(root, event.point_coords.as_ref().unwrap())?;
            let labels = load_tensor(root, event.point_labels.as_ref().unwrap())?;
            ensure!(
                coords.ndim() == 3
                    && coords.shape()[0] == 1
                    && coords.shape()[2] == 2
                    && labels.shape() == [1, coords.shape()[1]]
                    && coords.shape()[1] > 0,
                "invalid SAM point geometry"
            );
            ensure!(
                labels
                    .iter()
                    .all(|&value| [-1.0, 0.0, 1.0, 2.0, 3.0].contains(&value)),
                "invalid SAM point label"
            );
            let count = labels.shape()[1];
            (coords, labels, count)
        } else {
            (
                tensor(&[1, 1, 2], vec![0.0, 0.0])?,
                tensor(&[1, 1], vec![-1.0])?,
                0,
            )
        };
        if fixture.point_coordinate_space == "original_pixels" {
            for (index, value) in coordinates.iter_mut().enumerate() {
                *value = (*value / fixture.original_hw[if index % 2 == 0 { 1 } else { 0 }] as f32)
                    * fixture.image_size as f32;
            }
        }
        let prior = if has_points {
            states.get(&event.frame_idx)
        } else {
            None
        };
        let has_previous = prior.is_some();
        let previous = if let Some(prior) = prior {
            prior.low_mask.mapv(|value| value.clamp(-32.0, 32.0))
        } else {
            ArrayD::<f32>::zeros(IxDyn(&[1, 1, low, low]))
        };
        let multimask = cfg.multimask_output_in_sam
            && (event.is_init_cond_frame || cfg.multimask_output_for_tracking)
            && real_points >= cfg.multimask_min_pt_num
            && real_points <= cfg.multimask_max_pt_num;
        let mut outputs = heads.infer(vec![
            ("image_embed", conditioned),
            ("high_res_0", features.high0.clone()),
            ("high_res_1", features.high1.clone()),
            ("point_coords", coordinates),
            ("point_labels", labels),
            ("previous_mask_logits", previous),
            (
                "has_previous_mask",
                tensor(&[1], vec![if has_previous { 1.0 } else { 0.0 }])?,
            ),
            (
                "multimask_output",
                tensor(&[1], vec![if multimask { 1.0 } else { 0.0 }])?,
            ),
        ])?;
        let low_masks = take(&mut outputs, "low_res_masks", &[1, 1, low, low])?;
        let high_masks = take(
            &mut outputs,
            "high_res_masks",
            &[1, 1, fixture.image_size, fixture.image_size],
        )?;
        let pointer = take(&mut outputs, "obj_ptr", &[1, cfg.hidden_dim])?;
        let score = take(&mut outputs, "object_score_logits", &[1, 1])?;
        let mut encoded = memory_encoder.infer(vec![
            ("backbone_features", features.raw.clone()),
            ("mask_logits", high_masks.clone()),
            ("object_score_logits", score.clone()),
            (
                "mask_from_points",
                tensor(&[1], vec![if has_points { 1.0 } else { 0.0 }])?,
            ),
        ])?;
        let mut memory = take(
            &mut encoded,
            "maskmem_features",
            &[1, cfg.mem_dim, grid, grid],
        )?;
        if cfg.memory_storage_dtype == "bfloat16" {
            memory.mapv_inplace(|value| bfloat16_value(bfloat16_bits(value)));
            ensure!(
                memory.iter().all(|value| value.is_finite()),
                "BF16 memory storage overflow"
            );
        }
        let position = take(
            &mut encoded,
            "maskmem_pos_enc",
            &[1, cfg.mem_dim, grid, grid],
        )?;
        let shared_position = if let Some(shared) = &shared_memory_position {
            ensure!(
                shared.as_ref() == position,
                "spatial memory position changed across fixed-shape frames"
            );
            Arc::clone(shared)
        } else {
            let shared = Arc::new(position.clone());
            shared_memory_position = Some(Arc::clone(&shared));
            shared
        };
        let original_logits = resize_bilinear_chw(
            low_masks.as_slice().context("contiguous mask required")?,
            1,
            low,
            low,
            fixture.original_hw[0],
            fixture.original_hw[1],
        )?;
        let original_mask: Vec<_> = original_logits
            .iter()
            .map(|&value| if value > 0.0 { 1.0 } else { 0.0 })
            .collect();
        if inference {
            let bytes: Vec<u8> = original_mask
                .iter()
                .map(|&value| if value > 0.0 { 255 } else { 0 })
                .collect();
            super::desktop::write_frame(
                event.frame_idx,
                fixture.original_hw[1],
                fixture.original_hw[0],
                None,
                None,
                None,
                Some(&bytes),
            )?;
            super::desktop::progress("sam_video", event_index + 1, fixture.events.len())?;
        }
        let mut computed = BTreeMap::from([
            ("low_res_masks".to_owned(), low_masks.clone()),
            ("high_res_masks".to_owned(), high_masks),
            ("obj_ptr".to_owned(), pointer.clone()),
            ("object_score_logits".to_owned(), score),
            ("maskmem_features".to_owned(), memory.clone()),
            ("maskmem_pos_enc".to_owned(), position.clone()),
            (
                "original_mask_logits".to_owned(),
                tensor(
                    &[1, 1, fixture.original_hw[0], fixture.original_hw[1]],
                    original_logits,
                )?,
            ),
            (
                "original_mask".to_owned(),
                tensor(
                    &[1, 1, fixture.original_hw[0], fixture.original_hw[1]],
                    original_mask,
                )?,
            ),
        ]);
        computed.extend(outputs);
        let mut seen = HashSet::new();
        let mut metrics: Vec<Metrics> = Vec::new();
        let mut event_overlap = None;
        for reference in &event.outputs {
            ensure!(
                identifier(&reference.name) && seen.insert(reference.name.clone()),
                "unsafe/duplicate SAM event reference name"
            );
            let actual = computed.get(&reference.name).with_context(|| {
                format!(
                    "event {} asks for uncomputed output {}",
                    event.name, reference.name
                )
            })?;
            let reference_value = load_tensor(root, reference)?;
            let metric = compare(&reference.name, actual, &reference_value, tolerance)?;
            if reference.name == "original_mask" {
                let (changed, iou, overlap) = binary_overlap(actual, &reference_value)?;
                binary_changed_pixels += changed;
                binary_pixel_comparisons += actual.len();
                minimum_mask_iou = minimum_mask_iou.min(iou);
                compared_binary_masks += 1;
                event_overlap = Some(overlap);
            } else {
                floating_outputs_passed &= metric.passed;
            }
            let dump = should_dump_tensor(&args.tensor_dumps, &reference.name, metric.passed);
            metrics.push(metric);
            if let Some(directory) = args.output.as_ref().filter(|_| dump) {
                let directory = directory.join(&event.name);
                fs::create_dir_all(&directory)?;
                write_npy(directory.join(format!("{}.npy", reference.name)), actual)?;
            }
        }
        ensure!(
            inference
                || ["low_res_masks", "obj_ptr", "maskmem_features"]
                    .iter()
                    .all(|name| seen.contains(*name)),
            "each SAM event must validate mask, pointer and encoded memory"
        );
        let passed = metrics.iter().all(|metric| metric.passed);
        all_passed &= passed;
        if event.store_as_conditioning {
            if !conditioning_order.contains(&event.frame_idx) {
                conditioning_order.push(event.frame_idx);
            }
        } else {
            ensure!(
                !conditioning_order.contains(&event.frame_idx),
                "downgrading a conditioning frame requires explicit predictor consolidation, unsupported by this profile"
            );
        }
        states.insert(
            event.frame_idx,
            FrameState {
                conditioning: event.store_as_conditioning,
                memory: StoredMemory::new(&memory, &cfg.memory_storage_dtype),
                position: shared_position,
                pointer,
                low_mask: low_masks,
            },
        );
        peak_state_frames = peak_state_frames.max(states.len());
        let state_bytes: usize = states
            .values()
            .map(|state| state.memory.bytes() + (state.pointer.len() + state.low_mask.len()) * 4)
            .sum();
        peak_state_bytes = peak_state_bytes.max(state_bytes + grid * grid * cfg.mem_dim * 4);
        let future = &fixture.events[event_index + 1..];
        let future_schedule: Vec<_> = future
            .iter()
            .map(|event| {
                (
                    event.frame_idx,
                    !event.is_init_cond_frame && event.point_coords.is_some(),
                )
            })
            .collect();
        if let Some(retention) = policy.retention_after(event.frame_idx, &future_schedule) {
            // A later correction/replay disables eviction until its history has
            // been consumed. Once remaining events are forward-only, retain
            // conditioning frames, the required stride/pointer horizon and any
            // old future prediction explicitly needed as correction feedback.
            let before = states.len();
            states.retain(|&index, state| retention.keeps(index, state.conditioning));
            evicted_state_frames += before - states.len();
        }
        let mut event_report = serde_json::json!({"name":event.name,"frame_idx":event.frame_idx,"is_init_cond_frame":event.is_init_cond_frame,"store_as_conditioning":event.store_as_conditioning,"has_previous_mask":has_previous,"multimask_output":multimask,
            "spatial_memory_frames":selected.as_ref().map(|selection|selection.spatial.iter().map(|source|source.frame).collect::<Vec<_>>()).unwrap_or_default(),
            "object_pointer_frames":selected.as_ref().map(|selection|selection.pointers.iter().map(|source|source.frame).collect::<Vec<_>>()).unwrap_or_default(),"num_object_pointer_tokens":token_count,"outputs":metrics,"binary_mask_overlap":event_overlap,"passed":passed});
        if inference {
            let object = event_report.as_object_mut().expect("event object");
            object.remove("passed");
            object.remove("outputs");
            object.remove("binary_mask_overlap");
            object.insert("execution_success".into(), serde_json::json!(true));
        }
        event_reports.push(event_report);
    }
    ensure!(
        (inference && fixture.total_frames == 1 || memory_attention.calls > 0)
            && memory_encoder.calls == fixture.events.len(),
        "SAM video proof must execute neural memory encoding and recurrent attention"
    );
    let graphs = vec![
        image_encoder.finish()?,
        memory_attention.finish()?,
        heads.finish()?,
        memory_encoder.finish()?,
    ];
    if inference {
        let report = serde_json::json!({"schema_version":1,"component":"sam2_video_desktop_inference","execution_success":true,"validation_performed":false,"numerical_passed":null,"source_frames":fixture.total_frames,"events":event_reports,"graphs":graphs,"corrections":correction_count,"elapsed_seconds":started.elapsed().as_secs_f64(),"requested_provider":args.provider,"runtime_policy":super::runtime_policy(args),"scope":"Single-object forward propagation with correction/replay; binary source-resolution PNG masks"});
        fs::write(
            args.report.as_ref().context("desktop report missing")?,
            serde_json::to_vec_pretty(&report)?,
        )?;
        return Ok(());
    }
    let report = serde_json::json!({"schema_version":1,"component":"sam2_video_native","fixture":args.manifest,"source_video":args.video,"input_mode":if args.video.is_some() {"native_ffmpeg_rgb24_and_native_antialias_resize"} else {"pre_resized_rgb_npy"},"input_preprocessing":preprocessing_metrics,"requested_provider":args.provider,"runtime":format!("{:?}",ort::info()),"scope":"single-object forward SAM2 core propagation and point corrections, FP32 neural math, explicit memory-storage dtype; no predictor UI, reverse traversal or hole filling","memory_storage_dtype":cfg.memory_storage_dtype,"image_feature_cache_frames":IMAGE_CACHE_FRAMES,"peak_image_feature_tensor_bytes":IMAGE_CACHE_FRAMES * grid * grid * (cfg.hidden_dim * 2 + 32 * 16 + 64 * 4) * 4,"peak_state_tensor_bytes":peak_state_bytes,"peak_state_frames":peak_state_frames,"evicted_state_frames":evicted_state_frames,"state_eviction":"preserve correction/replay history, then conditioning plus stride/pointer horizon","source_frames":fixture.total_frames,"events":event_reports,"corrections":correction_count,"graphs":graphs,"elapsed_seconds":started.elapsed().as_secs_f64(),"atol":tolerance.atol,"rtol":tolerance.rtol,"floating_outputs_passed":floating_outputs_passed,"binary_changed_pixels":binary_changed_pixels,"binary_pixel_comparisons":binary_pixel_comparisons,"compared_binary_masks":compared_binary_masks,"minimum_mask_iou":if compared_binary_masks > 0 {Some(minimum_mask_iou)} else {None},"strict_passed":all_passed,"passed":all_passed});
    let mut report = report;
    report["ort_optimization"] = serde_json::json!(args.ort_optimization);
    report["reference_source"] = fixture.source.clone();
    report["same_device_reference"] = serde_json::json!(
        fixture
            .source
            .get("reference_device")
            .and_then(serde_json::Value::as_str)
            .map(|device| device == args.provider)
    );
    report["cuda_arena_limit_mib"] =
        serde_json::json!((args.provider == "cuda").then_some(args.cuda_memory_mib));
    report["cuda_memory_pattern"] = serde_json::json!((args.provider == "cuda").then_some(false));
    report["tensor_dump_policy"] = serde_json::json!({"requested":args.tensor_dumps,
        "failed_tensors_always_retained":true});
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &args.report {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &json)?;
    }
    println!("{json}");
    ensure!(all_passed, "native SAM2 video reference parity failed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dump_policy_never_discards_failed_tensor_evidence() {
        for policy in ["all", "masks", "none"] {
            for name in [
                "original_mask",
                "low_res_masks",
                "obj_ptr",
                "maskmem_features",
            ] {
                assert!(should_dump_tensor(policy, name, false));
                assert_eq!(
                    should_dump_tensor(policy, name, true),
                    policy == "all"
                        || (policy == "masks" && matches!(name, "original_mask" | "low_res_masks"))
                );
            }
        }
    }

    #[test]
    fn bfloat16_storage_rounds_halfway_values_to_even_for_both_signs() {
        for (input, expected) in [
            (1.003_906_3, 1.0),
            (1.011_718_8, 1.015_625),
            (-1.003_906_3, -1.0),
            (-1.011_718_8, -1.015_625),
        ] {
            assert_eq!(bfloat16_value(bfloat16_bits(input)), expected);
        }
        assert_eq!(bfloat16_value(bfloat16_bits(0.0)), 0.0);
    }

    #[test]
    fn compact_memory_really_uses_two_bytes_per_element_and_reconstructs_values() {
        let values = tensor(&[1, 1, 1, 3], vec![0.1, -1.1, 1.003_906_3]).unwrap();
        let packed = StoredMemory::new(&values, "bfloat16");
        let full = StoredMemory::new(&values, "float32");
        assert_eq!(packed.bytes(), 6);
        assert_eq!(full.bytes(), 12);
        assert_eq!(packed.at(0), 0.100_097_656);
        assert_eq!(packed.at(1), -1.101_562_5);
        assert_eq!(packed.at(2), 1.0);
        assert_eq!(full.at(0), 0.1);
    }

    #[test]
    fn binary_overlap_reports_changed_pixels_without_changing_strict_comparison() {
        let actual = tensor(&[1, 1, 1, 4], vec![1.0, 1.0, 0.0, 0.0]).unwrap();
        let reference = tensor(&[1, 1, 1, 4], vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let (changed, iou, _) = binary_overlap(&actual, &reference).unwrap();
        assert_eq!(changed, 1);
        assert_eq!(iou, 0.5);
        assert!(
            !compare(
                "original_mask",
                &actual,
                &reference,
                Tolerance {
                    atol: 0.0005,
                    rtol: 0.0005
                }
            )
            .unwrap()
            .passed
        );
        let empty = tensor(&[1, 1, 1, 1], vec![0.0]).unwrap();
        assert_eq!(binary_overlap(&empty, &empty).unwrap().1, 1.0);
    }
}
