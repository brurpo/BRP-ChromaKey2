//! SAM 3.1 video tracker host: upstream's multiplex tracker state machine around the exported graphs.
//!
//! Port of `tools/model_export/sam31_host_ref.py` (itself checked against upstream PyTorch). The neural
//! parts are ONNX graphs (`vision`, `tracker_features`, `interactive_{multi,single}`, `propagation`,
//! `memory_encoder`, `memory_attention_mem{7,10}`); this module owns frame preprocessing, the
//! conditioning/non-conditioning memory bank, temporal and object-pointer encodings, the 16-slot
//! multiplex packing and mask output. One multiplex bucket (up to 16 objects).
use crate::npy::{read_npy, write_npy};
use anyhow::{Context, Result, bail, ensure};
use ck_core::spatial::{
    resize_bilinear_antialias_chw, resize_bilinear_chw, resize_bilinear_pil_u8,
};
use half::{bf16, f16};
use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

/// 72 x 72 tokens of the lowest feature level.
pub const TOKENS: usize = 72 * 72;
pub const CHANNELS: usize = 256;
/// Objects per multiplex bucket.
pub const SLOTS: usize = 16;
/// Object-pointer token capacity of the memory-attention graphs (16 frames x 16 slots).
pub const POINTER_TOKENS: usize = 256;
pub const LOW_RES: usize = 288;

/// A graph input.
pub enum Input {
    F32(ArrayD<f32>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    Bool(ArrayD<bool>),
}

fn f32_input(shape: &[usize], data: Vec<f32>) -> Result<Input> {
    Ok(Input::F32(ArrayD::from_shape_vec(IxDyn(shape), data)?))
}

fn bool_input(shape: &[usize], data: Vec<bool>) -> Result<Input> {
    Ok(Input::Bool(ArrayD::from_shape_vec(IxDyn(shape), data)?))
}

/// Runs one exported graph by name with named inputs; every output is returned as FP32.
pub trait GraphRunner {
    fn run(
        &mut self,
        graph: &str,
        inputs: Vec<(&'static str, Input)>,
    ) -> Result<BTreeMap<String, ArrayD<f32>>>;

    /// Records host-side time of a named step (developer reports only).
    fn note(&mut self, _step: &str, _seconds: f64) {}

    /// Like `run`, returning only `outputs` (others stay on the device and may be skipped).
    fn run_select(
        &mut self,
        graph: &str,
        inputs: Vec<(&'static str, Input)>,
        outputs: &[&str],
    ) -> Result<BTreeMap<String, ArrayD<f32>>> {
        let mut all = self.run(graph, inputs)?;
        all.retain(|name, _| outputs.contains(&name.as_str()));
        Ok(all)
    }

    /// Whether an optional graph (the merged `frame` graph, `memory_encoder_logits`) is installed.
    fn has_graph(&self, _graph: &str) -> bool {
        false
    }
}

/// Tracker settings exported with the graphs (`tracker/config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub image_size: usize,
    pub num_maskmem: usize,
    pub max_cond_frames_in_attn: i64,
    pub max_obj_ptrs_in_encoder: usize,
    pub multimask_max_pt_num: usize,
    pub sigmoid_scale_for_mem_enc: f32,
    pub sigmoid_bias_for_mem_enc: f32,
    pub condition_as_mask_input_fg: f32,
    pub condition_as_mask_input_bg: f32,
    pub keep_first_cond_frame: bool,
    pub memory_temporal_stride_for_eval: usize,
    pub use_signed_tpos_enc_to_obj_ptrs: bool,
    pub only_obj_ptrs_in_the_past_for_eval: bool,
    pub use_maskmem_tpos_v2: bool,
    pub use_memory_selection: bool,
    pub multiplex_count: usize,
}

impl Config {
    /// The host implements exactly the settings SAM 3.1 ships with; refuse anything else.
    fn check(&self) -> Result<()> {
        ensure!(
            self.num_maskmem == 7
                && self.memory_temporal_stride_for_eval == 1
                && !self.keep_first_cond_frame
                && !self.use_signed_tpos_enc_to_obj_ptrs
                && !self.only_obj_ptrs_in_the_past_for_eval
                && self.use_maskmem_tpos_v2
                && !self.use_memory_selection
                && self.multiplex_count == SLOTS
                && self.max_obj_ptrs_in_encoder <= 16
                && self.max_cond_frames_in_attn >= 2,
            "unsupported SAM 3.1 tracker configuration {self:?}"
        );
        Ok(())
    }
}

/// Host-side learned tensors (`tracker/constants/*.npy`).
pub struct Constants {
    /// [num_maskmem, 256]
    pub maskmem_tpos_enc: Vec<f32>,
    /// [256, 256] (out, in)
    pub obj_ptr_tpos_proj_weight: Vec<f32>,
    pub obj_ptr_tpos_proj_bias: Vec<f32>,
    /// [5184, 256] token-major position encoding of the propagation level.
    pub image_pos: Vec<f32>,
}

impl Constants {
    pub fn load(dir: &Path, config: &Config) -> Result<Self> {
        let load = |name: &str, len: usize| -> Result<Vec<f32>> {
            let array = read_npy(dir.join(format!("{name}.npy")))
                .with_context(|| format!("reading SAM 3.1 constant {name}"))?;
            ensure!(
                array.len() == len,
                "SAM 3.1 constant {name} has {} values, expected {len}",
                array.len()
            );
            Ok(into_vec(array))
        };
        Ok(Self {
            maskmem_tpos_enc: load("maskmem_tpos_enc", config.num_maskmem * CHANNELS)?,
            obj_ptr_tpos_proj_weight: load("obj_ptr_tpos_proj_weight", CHANNELS * CHANNELS)?,
            obj_ptr_tpos_proj_bias: load("obj_ptr_tpos_proj_bias", CHANNELS)?,
            image_pos: load("image_pos", TOKENS * CHANNELS)?,
        })
    }
}

/// Upstream's video loader for one RGB24 frame: PIL bilinear resize to `size`, `to_tensor`, FP16
/// normalisation `(x - 0.5) / 0.5`; returned as the vision graph's equivalent [0, 1] input
/// `[1, 3, size, size]` (the graph normalises itself).
pub fn model_input(rgb: &[u8], height: usize, width: usize, size: usize) -> Result<ArrayD<f32>> {
    let resized = resize_bilinear_pil_u8(rgb, 3, height, width, size, size)?;
    let plane = size * size;
    let mut out = vec![0.0f32; 3 * plane];
    // One row of one channel plane per task (same per-value arithmetic as a sequential loop).
    ck_core::spatial::parallel_rows(&mut out, size, |row, line| {
        let (channel, y) = (row / size, row % size);
        for (x, value) in line.iter_mut().enumerate() {
            let v = f16::from_f32(f32::from(resized[(y * size + x) * 3 + channel]) / 255.0);
            let centered = f16::from_f32(v.to_f32() - 0.5);
            *value = centered.to_f32() + 0.5;
        }
    });
    Ok(ArrayD::from_shape_vec(IxDyn(&[1, 3, size, size]), out)?)
}

/// `get_1d_sine_pe(pos, dim)`: sin half then cos half.
fn sine_pe(position: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0; dim];
    for i in 0..half {
        let dim_t = 10000f32.powf(2.0 * (i / 2) as f32 / half as f32);
        let x = position / dim_t;
        out[i] = x.sin();
        out[half + i] = x.cos();
    }
    out
}

/// [1, C, 72, 72] channel-major -> [5184, C] token-major.
fn tokens_from_chw(chw: &[f32]) -> Vec<f32> {
    tokens_from_chw_map(chw, |x| x)
}

/// [1, C, 72, 72] -> [5184, C], applying `map` to every value (parallel over tokens).
fn tokens_from_chw_map(chw: &[f32], map: impl Fn(f32) -> f32 + Sync) -> Vec<f32> {
    // Blocks of 64 tokens (5184 = 81 x 64): contiguous reads per channel, writes stay in-cache.
    const BLOCK: usize = 64;
    let mut out = vec![0.0; TOKENS * CHANNELS];
    ck_core::spatial::parallel_rows(&mut out, BLOCK * CHANNELS, |block, rows| {
        let t0 = block * BLOCK;
        for c in 0..CHANNELS {
            let src = &chw[c * TOKENS + t0..c * TOKENS + t0 + BLOCK];
            for (t, &x) in src.iter().enumerate() {
                rows[t * CHANNELS + c] = map(x);
            }
        }
    });
    out
}

/// [5184, C] token-major -> [1, C, 72, 72] channel-major.
fn chw_from_tokens(tokens: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; TOKENS * CHANNELS];
    ck_core::spatial::parallel_rows(&mut out, TOKENS, |c, plane| {
        for (t, value) in plane.iter_mut().enumerate() {
            *value = tokens[t * CHANNELS + c];
        }
    });
    out
}

/// The values of an owned array in logical order without the (slow) element iterator.
fn into_vec(array: ArrayD<f32>) -> Vec<f32> {
    if array.is_standard_layout() {
        let len = array.len();
        let (values, offset) = array.into_raw_vec_and_offset();
        let start = offset.unwrap_or(0);
        if start == 0 && values.len() == len {
            return values;
        }
        // Standard layout inside a larger buffer: copy the logical range.
        return values[start..start + len].to_vec();
    }
    array.iter().copied().collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn take(
    outputs: &mut BTreeMap<String, ArrayD<f32>>,
    graph: &str,
    name: &str,
) -> Result<ArrayD<f32>> {
    outputs
        .remove(name)
        .with_context(|| format!("SAM 3.1 {graph} graph has no output {name}"))
}

/// Per-frame features (only the current frame is kept).
struct Features {
    frame: usize,
    int_pix: ArrayD<f32>,
    int_hr0: ArrayD<f32>,
    int_hr1: ArrayD<f32>,
    prop_hr0: ArrayD<f32>,
    prop_hr1: ArrayD<f32>,
    /// [1, 256, 72, 72]
    prop_pix: Vec<f32>,
    /// [5184, 256], shared with the frames' stored memory.
    prop_tokens: Arc<Vec<f32>>,
    /// Detector features (det_fpn0..2), only on frames that run the detector.
    det_fpn: Option<[ArrayD<f32>; 3]>,
}

/// One frame's spatial memory: ([5184, 256] BF16-rounded maskmem features, [5184, 256] maskmem
/// position encoding (shared, identical across frames), [5184, 256] image features (shared)).
type SpatialMemory = (Vec<f32>, Arc<Vec<f32>>, Arc<Vec<f32>>);

/// One frame's stored tracker output (upstream's compact per-frame output).
struct Stored {
    /// Spatial memory, dropped once it can no longer be selected.
    spatial: Option<SpatialMemory>,
    /// [16, 256] multiplexed object pointers.
    obj_ptr: Vec<f32>,
}

/// Mask prompts are binary 0/1 at this size (4 x the 288 low-res masks, upstream's tracker size).
pub const MASK_SIZE: usize = 1152;
/// CLIP context length of the text encoder.
pub const TEXT_TOKENS: usize = 32;
/// Fixed (padded) box and point slots of the detector graph.
pub const DETECTOR_SLOTS: usize = 8;

/// What defines one object on a frame: points in source pixels (labels 1 positive, 0 negative,
/// 2/3 box top-left/bottom-right corners) and/or a binary mask at `MASK_SIZE` (e.g. a detection).
#[derive(Clone, Debug, Default)]
pub struct ObjectPrompt {
    pub points: Vec<([f32; 2], i32)>,
    pub mask: Option<Vec<f32>>,
}

struct HeadOutput {
    low: Vec<f32>,
    score: f32,
    obj_ptr: Vec<f32>,
}

/// Encoded text prompt.
pub struct TextFeatures {
    /// [32, 1, 256]
    features: Vec<f32>,
    /// [1, 32], true on padding tokens.
    padding: Vec<bool>,
}

/// One detector result.
pub struct Detection {
    pub score: f32,
    /// [288, 288] mask logits over the whole frame.
    pub low_res: Vec<f32>,
}

/// Head output for the tracked objects (demultiplexed).
pub struct FrameMasks {
    /// [objects, 288, 288] mask logits.
    pub low_res: Vec<f32>,
    pub object_scores: Vec<f32>,
}

/// What a memory-attention call was assembled from (for checks and reports).
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryLayout {
    pub graph: String,
    pub frames: usize,
    pub pointer_tokens: usize,
    pub conditioning: Vec<usize>,
    /// (temporal position, frame) of each spatial memory frame, in order.
    pub spatial: Vec<(i64, usize)>,
    /// (pointer distance, frame) of each pointer frame, in order.
    pub pointers: Vec<(usize, usize)>,
}

pub struct Tracker<R: GraphRunner> {
    pub runner: R,
    config: Config,
    constants: Constants,
    num_frames: usize,
    source_height: usize,
    source_width: usize,
    objects: usize,
    /// Conditioning frames in insertion order (upstream dict order).
    cond: Vec<(usize, Stored)>,
    non_cond: BTreeMap<usize, Stored>,
    features: Option<Features>,
    /// Last memory-encoder position output (raw) and its token layout.
    maskmem_pos: Option<(Vec<f32>, Arc<Vec<f32>>)>,
    /// Tracking in reverse time order (upstream `track_in_reverse`).
    reverse: bool,
    /// Non-conditioning frames below this keep their spatial memory during forward tracking,
    /// because the backward pass from the first prompted frame selects them again.
    keep_spatial_below: usize,
}

impl<R: GraphRunner> Tracker<R> {
    pub fn new(
        runner: R,
        config: Config,
        constants: Constants,
        num_frames: usize,
        source_height: usize,
        source_width: usize,
    ) -> Result<Self> {
        config.check()?;
        ensure!(
            num_frames > 0 && source_height > 0 && source_width > 0,
            "empty SAM 3.1 video"
        );
        Ok(Self {
            runner,
            config,
            constants,
            num_frames,
            source_height,
            source_width,
            objects: 0,
            cond: Vec::new(),
            non_cond: BTreeMap::new(),
            features: None,
            maskmem_pos: None,
            reverse: false,
            keep_spatial_below: 0,
        })
    }

    /// Switch the tracking direction (upstream runs forward, then reverse from the first prompt).
    pub fn set_reverse(&mut self, reverse: bool) {
        self.reverse = reverse;
    }

    /// Keep spatial memory of the frames a backward pass from `anchor` will select again.
    pub fn plan_backward_from(&mut self, anchor: usize) {
        self.keep_spatial_below = if anchor > 0 {
            anchor + self.config.num_maskmem - 1
        } else {
            0
        };
    }

    pub fn objects(&self) -> usize {
        self.objects
    }

    fn features(&mut self, frame: usize, rgb: &[u8], detector: bool) -> Result<&Features> {
        if self
            .features
            .as_ref()
            .is_none_or(|f| f.frame != frame || (detector && f.det_fpn.is_none()))
        {
            let size = self.config.image_size;
            let clock = std::time::Instant::now();
            let input = model_input(rgb, self.source_height, self.source_width, size)?;
            self.runner
                .note("host:model_input", clock.elapsed().as_secs_f64());
            const TRACKER: [&str; 6] = [
                "prop_fpn2",
                "int_pix",
                "int_hr0",
                "int_hr1",
                "prop_hr0",
                "prop_hr1",
            ];
            const DETECTOR: [&str; 3] = ["det_fpn0", "det_fpn1", "det_fpn2"];
            let mut out = if self.runner.has_graph("frame") {
                // One graph: the FPN maps go from the backbone to the tracker projections on the
                // device; the detector maps are only fetched when needed.
                let mut wanted: Vec<&str> = TRACKER.to_vec();
                if detector {
                    wanted.extend(DETECTOR);
                }
                self.runner
                    .run_select("frame", vec![("rgb", Input::F32(input))], &wanted)?
            } else {
                let mut vision = self
                    .runner
                    .run("vision", vec![("rgb", Input::F32(input))])?;
                let mut inputs = Vec::new();
                for name in ["int_fpn0", "int_fpn1", "int_fpn2", "prop_fpn0", "prop_fpn1"] {
                    inputs.push((name, Input::F32(take(&mut vision, "vision", name)?)));
                }
                let mut out = self.runner.run("tracker_features", inputs)?;
                for name in ["prop_fpn2", "det_fpn0", "det_fpn1", "det_fpn2"] {
                    out.insert(name.to_owned(), take(&mut vision, "vision", name)?);
                }
                out
            };
            let prop_fpn2 = take(&mut out, "frame", "prop_fpn2")?;
            ensure!(
                prop_fpn2.shape() == [1, CHANNELS, 72, 72],
                "unexpected prop_fpn2 shape {:?}",
                prop_fpn2.shape()
            );
            let det_fpn = if detector {
                Some([
                    take(&mut out, "frame", "det_fpn0")?,
                    take(&mut out, "frame", "det_fpn1")?,
                    take(&mut out, "frame", "det_fpn2")?,
                ])
            } else {
                None
            };
            let prop_pix = into_vec(prop_fpn2);
            self.features = Some(Features {
                frame,
                det_fpn,
                int_pix: take(&mut out, "frame", "int_pix")?,
                int_hr0: take(&mut out, "frame", "int_hr0")?,
                int_hr1: take(&mut out, "frame", "int_hr1")?,
                prop_hr0: take(&mut out, "frame", "prop_hr0")?,
                prop_hr1: take(&mut out, "frame", "prop_hr1")?,
                prop_tokens: Arc::new(tokens_from_chw(&prop_pix)),
                prop_pix,
            });
        }
        Ok(self.features.as_ref().expect("features set above"))
    }

    fn mux(&self, per_object: &[f32], item: usize) -> Vec<f32> {
        let mut out = vec![0.0; SLOTS * item];
        out[..per_object.len()].copy_from_slice(per_object);
        out
    }

    fn valid_slots(&self) -> Vec<bool> {
        (0..SLOTS).map(|slot| slot < self.objects).collect()
    }

    /// `_encode_new_memory` for one bucket: `logits` is [objects, size, size] mask logits, either the
    /// tracker's low-res masks or a consolidated image-size mask. Upstream resizes them to the image
    /// size, then applies sigmoid * scale + bias; `memory_encoder_logits` does both on the GPU (5 MB
    /// upload instead of 16 image-size planes), else the host does.
    fn encode_memory(
        &mut self,
        logits: &[f32],
        size: usize,
        scores: &[f32],
        conditioning: &[bool],
    ) -> Result<(Vec<f32>, Arc<Vec<f32>>)> {
        let n = self.objects;
        ensure!(
            logits.len() == n * size * size && scores.len() == n && conditioning.len() == n,
            "memory encoder input size mismatch"
        );
        let clock = std::time::Instant::now();
        let on_device = self.runner.has_graph("memory_encoder_logits");
        let (graph, mask_name, mask_size, masks) = if on_device {
            (
                "memory_encoder_logits",
                "mask_logits",
                size,
                logits.to_vec(),
            )
        } else {
            let image = self.config.image_size;
            let high = if size == image {
                logits.to_vec()
            } else {
                resize_bilinear_chw(logits, n, size, size, image, image)?
            };
            let scale = self.config.sigmoid_scale_for_mem_enc;
            let bias = self.config.sigmoid_bias_for_mem_enc;
            let masks = high.iter().map(|&x| sigmoid(x) * scale + bias).collect();
            ("memory_encoder", "slot_masks", image, masks)
        };
        let plane = mask_size * mask_size;
        let cond: Vec<f32> = conditioning
            .iter()
            .map(|&c| {
                if c {
                    self.config.condition_as_mask_input_fg
                } else {
                    self.config.condition_as_mask_input_bg
                }
            })
            .collect();
        let pix = self
            .features
            .as_ref()
            .context("no frame features")?
            .prop_pix
            .clone();
        let inputs = vec![
            ("pix_feat", f32_input(&[1, CHANNELS, 72, 72], pix)?),
            (
                mask_name,
                f32_input(&[1, SLOTS, mask_size, mask_size], self.mux(&masks, plane))?,
            ),
            ("slot_cond", f32_input(&[1, SLOTS], self.mux(&cond, 1))?),
            (
                "slot_scores",
                f32_input(&[1, SLOTS, 1], self.mux(scores, 1))?,
            ),
        ];
        self.runner
            .note("host:memory_encoder_inputs", clock.elapsed().as_secs_f64());
        let mut out = self.runner.run(graph, inputs)?;
        let clock = std::time::Instant::now();
        let features = take(&mut out, graph, "maskmem_features")?;
        let pos = take(&mut out, graph, "maskmem_pos")?;
        ensure!(
            features.len() == TOKENS * CHANNELS && pos.len() == TOKENS * CHANNELS,
            "unexpected memory encoder output size"
        );
        // Upstream stores maskmem_features as BF16.
        let features = tokens_from_chw_map(&into_vec(features), |x| bf16::from_f32(x).to_f32());
        // The position encoding depends only on the feature size: reuse the token copy while the
        // raw output is unchanged (checked every time).
        let pos = into_vec(pos);
        let pos = match &self.maskmem_pos {
            Some((raw, tokens)) if *raw == pos => tokens.clone(),
            _ => {
                let tokens = Arc::new(tokens_from_chw(&pos));
                self.maskmem_pos = Some((pos, tokens.clone()));
                tokens
            }
        };
        let tokens = (features, pos);
        self.runner
            .note("host:memory_encoder_outputs", clock.elapsed().as_secs_f64());
        Ok(tokens)
    }

    fn store(
        &mut self,
        logits: &[f32],
        size: usize,
        scores: &[f32],
        obj_ptr: &[f32],
        conditioning: &[bool],
    ) -> Result<Stored> {
        let (features, pos) = self.encode_memory(logits, size, scores, conditioning)?;
        let image = self
            .features
            .as_ref()
            .context("no frame features")?
            .prop_tokens
            .clone();
        Ok(Stored {
            spatial: Some((features, pos, image)),
            obj_ptr: self.mux(obj_ptr, CHANNELS),
        })
    }

    /// Initial conditioning frame with point-only objects (x, y in source pixels, labels 1/0).
    #[cfg(test)]
    pub fn add_points(
        &mut self,
        frame: usize,
        rgb: &[u8],
        objects: &[Vec<([f32; 2], i32)>],
    ) -> Result<FrameMasks> {
        let prompts: Vec<ObjectPrompt> = objects
            .iter()
            .map(|points| ObjectPrompt {
                points: points.clone(),
                mask: None,
            })
            .collect();
        self.add_objects(frame, rgb, &prompts)
    }

    /// Initial conditioning frame: one object per prompt (points/box corners and/or a mask).
    pub fn add_objects(
        &mut self,
        frame: usize,
        rgb: &[u8],
        objects: &[ObjectPrompt],
    ) -> Result<FrameMasks> {
        ensure!(
            self.objects == 0 && self.cond.is_empty(),
            "SAM 3.1 objects were already added"
        );
        ensure!(
            !objects.is_empty() && objects.len() <= SLOTS,
            "SAM 3.1 needs 1..={SLOTS} objects"
        );
        ensure!(frame < self.num_frames, "prompt frame outside the video");
        self.features(frame, rgb, false)?;
        let mut low = Vec::new();
        let mut scores = Vec::new();
        let mut pointers = Vec::new();
        for prompt in objects {
            let head = self.prompt_head(prompt, None)?;
            low.extend(head.low);
            scores.push(head.score);
            pointers.extend(head.obj_ptr);
        }
        self.objects = objects.len();
        let n = self.objects;
        let high = self.consolidated_high(&low)?;
        let stored = self.store(
            &high,
            self.config.image_size,
            &scores,
            &pointers,
            &vec![true; n],
        )?;
        self.cond.push((frame, stored));
        Ok(FrameMasks {
            low_res: low,
            object_scores: scores,
        })
    }

    /// Upstream's preflight consolidation encodes a prompted frame from its video-resolution mask:
    /// low-res -> source size (bilinear) -> 288 (antialiased) -> image size (bilinear), with the
    /// non-overlap rule across objects.
    fn consolidated_high(&self, low: &[f32]) -> Result<Vec<f32>> {
        let n = low.len() / (LOW_RES * LOW_RES);
        let video = resize_bilinear_chw(
            low,
            n,
            LOW_RES,
            LOW_RES,
            self.source_height,
            self.source_width,
        )?;
        let back = resize_bilinear_antialias_chw(
            &video,
            n,
            self.source_height,
            self.source_width,
            LOW_RES,
            LOW_RES,
        )?;
        let image = self.config.image_size;
        let mut high = resize_bilinear_chw(&back, n, LOW_RES, LOW_RES, image, image)?;
        if n > 1 {
            non_overlap(&mut high, n);
        }
        Ok(high)
    }

    /// One object's SAM head output for a prompt on the current frame.
    ///
    /// - mask only: upstream `_use_mask_as_output` (logits +-10, pointer from the mask prompt);
    /// - points (box corners are labels 2/3): the interactive head, with the mask (or `previous`,
    ///   the object's propagated low-res logits on a corrected frame) as its mask prompt.
    fn prompt_head(
        &mut self,
        prompt: &ObjectPrompt,
        previous: Option<&[f32]>,
    ) -> Result<HeadOutput> {
        let plane = LOW_RES * LOW_RES;
        let mask_prompt = match &prompt.mask {
            Some(mask) => {
                ensure!(
                    mask.len() == MASK_SIZE * MASK_SIZE,
                    "SAM 3.1 mask prompt size"
                );
                let mut out = self.runner.run(
                    "mask_downsample",
                    vec![(
                        "mask",
                        f32_input(&[1, 1, MASK_SIZE, MASK_SIZE], mask.clone())?,
                    )],
                )?;
                Some(
                    take(&mut out, "mask_downsample", "mask_prompt")?
                        .into_iter()
                        .collect::<Vec<_>>(),
                )
            }
            None => previous.map(<[f32]>::to_vec),
        };
        let only_mask = prompt.points.is_empty();
        ensure!(
            !only_mask || prompt.mask.is_some(),
            "SAM 3.1 object without points, box or mask"
        );
        // A mask-only prompt is padded with one empty point (label -1), as upstream does.
        let points: Vec<([f32; 2], i32)> = if only_mask {
            vec![([0.0, 0.0], -1)]
        } else {
            prompt.points.clone()
        };
        let size = self.config.image_size as f32;
        let (sx, sy) = (
            size / self.source_width as f32,
            size / self.source_height as f32,
        );
        let coords: Vec<f32> = points
            .iter()
            .flat_map(|(p, l)| {
                if *l < 0 {
                    [0.0, 0.0]
                } else {
                    [p[0] * sx, p[1] * sy]
                }
            })
            .collect();
        let labels: Vec<i32> = points.iter().map(|(_, l)| *l).collect();
        let multi = !only_mask && points.len() <= self.config.multimask_max_pt_num;
        let has_mask = mask_prompt.is_some();
        let f = self.features.as_ref().context("no frame features")?;
        let inputs = vec![
            ("backbone_features", Input::F32(f.int_pix.clone())),
            ("hr0", Input::F32(f.int_hr0.clone())),
            ("hr1", Input::F32(f.int_hr1.clone())),
            ("point_coords", f32_input(&[1, points.len(), 2], coords)?),
            (
                "point_labels",
                Input::I32(ArrayD::from_shape_vec(IxDyn(&[1, points.len()]), labels)?),
            ),
            (
                "mask_prompt",
                f32_input(
                    &[1, 1, LOW_RES, LOW_RES],
                    mask_prompt.unwrap_or_else(|| vec![0.0; plane]),
                )?,
            ),
            ("has_mask", bool_input(&[1], vec![has_mask])?),
        ];
        let graph = if multi {
            "interactive_multi"
        } else {
            "interactive_single"
        };
        let mut out = self.runner.run(graph, inputs)?;
        let mut low: Vec<f32> = take(&mut out, graph, "low_res_masks")?
            .into_iter()
            .collect();
        let mut score = take(&mut out, graph, "object_score_logits")?
            .iter()
            .copied()
            .next()
            .context("missing object score")?;
        let obj_ptr = into_vec(take(&mut out, graph, "obj_ptr")?);
        ensure!(
            low.len() == plane && obj_ptr.len() == CHANNELS,
            "interactive output size mismatch"
        );
        if only_mask {
            let mask = prompt.mask.as_ref().expect("checked above");
            let high: Vec<f32> = mask.iter().map(|&m| m * 20.0 - 10.0).collect();
            low = resize_bilinear_antialias_chw(&high, 1, MASK_SIZE, MASK_SIZE, LOW_RES, LOW_RES)?;
            score = if mask.iter().any(|&m| m > 0.0) {
                10.0
            } else {
                -10.0
            };
        }
        Ok(HeadOutput {
            low,
            score,
            obj_ptr,
        })
    }

    /// `select_closest_cond_frames`: (selected frames in upstream order, unselected frames).
    fn select_conditioning(&self, frame: usize) -> (Vec<usize>, Vec<usize>) {
        let all: Vec<usize> = self.cond.iter().map(|(t, _)| *t).collect();
        let limit = self.config.max_cond_frames_in_attn;
        if limit == -1 || all.len() <= limit as usize {
            return (all, Vec::new());
        }
        let mut selected = Vec::new();
        if let Some(before) = all.iter().copied().filter(|&t| t < frame).max() {
            selected.push(before);
        }
        if let Some(after) = all.iter().copied().filter(|&t| t >= frame).min() {
            selected.push(after);
        }
        let mut rest: Vec<usize> = all
            .iter()
            .copied()
            .filter(|t| !selected.contains(t))
            .collect();
        rest.sort_by_key(|&t| t.abs_diff(frame)); // stable, like Python's sorted
        let room = (limit as usize).saturating_sub(selected.len());
        selected.extend(rest.into_iter().take(room));
        let unselected = all.into_iter().filter(|t| !selected.contains(t)).collect();
        (selected, unselected)
    }

    /// Whether `frame` is a conditioning (prompted) frame.
    pub fn is_conditioning(&self, frame: usize) -> bool {
        self.lookup(frame).is_some()
    }

    fn lookup(&self, frame: usize) -> Option<&Stored> {
        self.cond.iter().find(|(t, _)| *t == frame).map(|(_, s)| s)
    }

    /// Assemble the memory-attention feed for `frame` (`_prepare_memory_conditioned_features`).
    fn memory_feed(&self, frame: usize) -> Result<(Vec<(&'static str, Input)>, MemoryLayout)> {
        let n_mem = self.config.num_maskmem;
        let (selected, unselected) = self.select_conditioning(frame);
        let from_cond_or_unselected = |t: usize| -> Option<&Stored> {
            self.non_cond
                .get(&t)
                .or_else(|| unselected.contains(&t).then(|| self.lookup(t)).flatten())
        };
        let sign = if self.reverse { -1 } else { 1 };
        // The frame `distance` steps back in tracking order (after `frame` when reversed).
        let behind = |distance: usize| -> Option<usize> {
            if self.reverse {
                Some(frame + distance).filter(|&t| t < self.num_frames)
            } else {
                frame.checked_sub(distance)
            }
        };
        let mut spatial: Vec<(i64, usize, &Stored)> = selected
            .iter()
            .map(|&t| {
                (
                    (frame as i64 - t as i64) * sign,
                    t,
                    self.lookup(t).expect("selected cond frame exists"),
                )
            })
            .collect();
        for t_pos in 1..n_mem {
            if let Some(prev) = behind(n_mem - t_pos)
                && let Some(stored) = from_cond_or_unselected(prev)
            {
                spatial.push((t_pos as i64, prev, stored));
            }
        }
        spatial.retain(|(_, _, s)| s.spatial.is_some());
        let frames = spatial.len();
        ensure!(
            (1..=10).contains(&frames),
            "SAM 3.1 memory has {frames} frames (supported 1..=10)"
        );
        let capacity = if frames <= 7 { 7 } else { 10 };
        let rows = capacity * TOKENS;
        let tpos = |t_pos: i64| -> &[f32] {
            let index = if t_pos <= 0 || t_pos >= n_mem as i64 {
                n_mem - 1
            } else {
                n_mem - t_pos as usize - 1
            };
            &self.constants.maskmem_tpos_enc[index * CHANNELS..(index + 1) * CHANNELS]
        };
        let row_values = rows * CHANNELS;
        let mut memory = vec![0.0f32; (rows + POINTER_TOKENS) * CHANNELS];
        let mut memory_pos = vec![0.0f32; (rows + POINTER_TOKENS) * CHANNELS];
        let mut memory_image = vec![0.0f32; row_values];
        let mut memory_image_pos = vec![0.0f32; row_values];
        {
            let frame_len = TOKENS * CHANNELS;
            let image_pos = &self.constants.image_pos;
            std::thread::scope(|scope| {
                for ((((entry, mem), mem_pos), mem_img), mem_img_pos) in spatial
                    .iter()
                    .zip(memory.chunks_mut(frame_len))
                    .zip(memory_pos.chunks_mut(frame_len))
                    .zip(memory_image.chunks_mut(frame_len))
                    .zip(memory_image_pos.chunks_mut(frame_len))
                {
                    let (t_pos, _, stored) = entry;
                    let enc = tpos(*t_pos);
                    scope.spawn(move || {
                        let (features, pos, image) =
                            stored.spatial.as_ref().expect("retained above");
                        mem.copy_from_slice(features);
                        mem_img.copy_from_slice(image);
                        for token in 0..TOKENS {
                            let at = token * CHANNELS;
                            for c in 0..CHANNELS {
                                mem_pos[at + c] = pos[at + c] + enc[c];
                                mem_img_pos[at + c] = image_pos[at + c] + enc[c];
                            }
                        }
                    });
                }
            });
        }
        // Object pointers: selected conditioning frames, then up to max_ptr - 1 previous frames.
        let max_ptr = self.num_frames.min(self.config.max_obj_ptrs_in_encoder);
        let mut pointers: Vec<(usize, usize, &Stored)> = selected
            .iter()
            .map(|&t| {
                (
                    frame.abs_diff(t),
                    t,
                    self.lookup(t).expect("selected cond frame exists"),
                )
            })
            .collect();
        for t_diff in 1..max_ptr {
            let Some(prev) = behind(t_diff) else {
                break;
            };
            if let Some(stored) = from_cond_or_unselected(prev) {
                pointers.push((t_diff, prev, stored));
            }
        }
        let pointer_tokens = pointers.len() * SLOTS;
        ensure!(
            pointer_tokens <= POINTER_TOKENS,
            "too many SAM 3.1 object pointers"
        );
        let denominator = max_ptr.saturating_sub(1).max(1) as f32;
        let (w, b) = (
            &self.constants.obj_ptr_tpos_proj_weight,
            &self.constants.obj_ptr_tpos_proj_bias,
        );
        for (i, (distance, _, stored)) in pointers.iter().enumerate() {
            let pe = sine_pe(*distance as f32 / denominator, CHANNELS);
            let encoded: Vec<f32> = (0..CHANNELS)
                .map(|o| {
                    b[o] + (0..CHANNELS)
                        .map(|k| w[o * CHANNELS + k] * pe[k])
                        .sum::<f32>()
                })
                .collect();
            for slot in 0..SLOTS {
                let at = row_values + (i * SLOTS + slot) * CHANNELS;
                memory[at..at + CHANNELS]
                    .copy_from_slice(&stored.obj_ptr[slot * CHANNELS..(slot + 1) * CHANNELS]);
                memory_pos[at..at + CHANNELS].copy_from_slice(&encoded);
            }
        }
        let mut key_valid = vec![false; rows + POINTER_TOKENS];
        key_valid[..frames * TOKENS]
            .iter_mut()
            .for_each(|v| *v = true);
        key_valid[rows..rows + pointer_tokens]
            .iter_mut()
            .for_each(|v| *v = true);
        let f = self.features.as_ref().context("no frame features")?;
        let graph = format!("memory_attention_mem{capacity}");
        let layout = MemoryLayout {
            graph: graph.clone(),
            frames,
            pointer_tokens,
            conditioning: {
                let mut c = selected.clone();
                c.sort_unstable();
                c
            },
            spatial: spatial.iter().map(|(p, t, _)| (*p, *t)).collect(),
            pointers: pointers.iter().map(|(d, t, _)| (*d, *t)).collect(),
        };
        let inputs = vec![
            (
                "image",
                f32_input(&[TOKENS, 1, CHANNELS], f.prop_tokens.to_vec())?,
            ),
            (
                "image_pos",
                f32_input(&[TOKENS, 1, CHANNELS], self.constants.image_pos.clone())?,
            ),
            (
                "memory_image",
                f32_input(&[rows, 1, CHANNELS], memory_image)?,
            ),
            (
                "memory_image_pos",
                f32_input(&[rows, 1, CHANNELS], memory_image_pos)?,
            ),
            (
                "memory",
                f32_input(&[rows + POINTER_TOKENS, 1, CHANNELS], memory)?,
            ),
            (
                "memory_pos",
                f32_input(&[rows + POINTER_TOKENS, 1, CHANNELS], memory_pos)?,
            ),
            (
                "key_valid",
                bool_input(&[1, rows + POINTER_TOKENS], key_valid)?,
            ),
        ];
        Ok((inputs, layout))
    }

    /// Track the objects into `frame` from memory (non-conditioning frame).
    #[cfg(test)]
    pub fn propagate(&mut self, frame: usize, rgb: &[u8]) -> Result<(FrameMasks, MemoryLayout)> {
        let (masks, layout, _) = self.propagate_with(frame, rgb, None)?;
        Ok((masks, layout))
    }

    /// Track into `frame`; with a `correction` (upstream "propagation and interaction"), the prompt
    /// is applied to the object it lands on (the one whose propagated mask contains its first
    /// positive point or box centre, else object 0) and the frame becomes a conditioning frame.
    /// Returns the corrected object index.
    pub fn propagate_with(
        &mut self,
        frame: usize,
        rgb: &[u8],
        correction: Option<&ObjectPrompt>,
    ) -> Result<(FrameMasks, MemoryLayout, Option<usize>)> {
        ensure!(self.objects > 0, "add SAM 3.1 objects before propagating");
        ensure!(
            frame < self.num_frames && self.lookup(frame).is_none(),
            "invalid SAM 3.1 propagation frame {frame}"
        );
        self.features(frame, rgb, false)?;
        let clock = std::time::Instant::now();
        let (inputs, layout) = self.memory_feed(frame)?;
        self.runner
            .note("host:memory_feed", clock.elapsed().as_secs_f64());
        let mut out = self.runner.run(&layout.graph, inputs)?;
        let conditioned = take(&mut out, &layout.graph, "conditioned")?;
        ensure!(
            conditioned.len() == TOKENS * CHANNELS,
            "unexpected memory attention output size"
        );
        let tokens = into_vec(conditioned);
        let f = self.features.as_ref().expect("features loaded");
        let inputs = vec![
            (
                "pix_feat_with_mem",
                f32_input(&[1, CHANNELS, 72, 72], chw_from_tokens(&tokens))?,
            ),
            ("hr0", Input::F32(f.prop_hr0.clone())),
            ("hr1", Input::F32(f.prop_hr1.clone())),
            ("valid_slots", bool_input(&[1, SLOTS], self.valid_slots())?),
        ];
        let mut out = self.runner.run("propagation", inputs)?;
        let n = self.objects;
        let plane = LOW_RES * LOW_RES;
        let low: Vec<f32> = take(&mut out, "propagation", "low_res_masks")?
            .iter()
            .take(n * plane)
            .copied()
            .collect();
        let scores: Vec<f32> = take(&mut out, "propagation", "object_score_logits")?
            .iter()
            .take(n)
            .copied()
            .collect();
        let pointers: Vec<f32> = take(&mut out, "propagation", "obj_ptr")?
            .iter()
            .take(n * CHANNELS)
            .copied()
            .collect();
        ensure!(
            low.len() == n * plane && scores.len() == n && pointers.len() == n * CHANNELS,
            "propagation output size mismatch"
        );
        let image = self.config.image_size;
        let (mut low, mut scores, mut pointers) = (low, scores, pointers);
        let mut corrected = None;
        if let Some(prompt) = correction {
            let target = self.correction_target(prompt, &low);
            let head =
                self.prompt_head(prompt, Some(&low[target * plane..(target + 1) * plane]))?;
            low[target * plane..(target + 1) * plane].copy_from_slice(&head.low);
            scores[target] = head.score;
            pointers[target * CHANNELS..(target + 1) * CHANNELS].copy_from_slice(&head.obj_ptr);
            corrected = Some(target);
        }
        if let Some(target) = corrected {
            let high = self.consolidated_high(&low)?;
            let cond: Vec<bool> = (0..n).map(|o| o == target).collect();
            let stored = self.store(&high, image, &scores, &pointers, &cond)?;
            self.cond.push((frame, stored));
        } else {
            let stored = self.store(&low, LOW_RES, &scores, &pointers, &vec![false; n])?;
            self.non_cond.insert(frame, stored);
        }
        // Tracking never selects spatial memory more than num_maskmem - 1 frames behind again
        // (except the frames a later backward pass needs).
        let keep = self.config.num_maskmem - 1;
        if self.reverse {
            for (_, stored) in self.non_cond.range_mut(frame + keep + 1..) {
                stored.spatial = None;
            }
        } else if frame >= keep {
            let floor = self.keep_spatial_below;
            for (_, stored) in self
                .non_cond
                .range_mut(floor.min(frame - keep)..frame - keep)
            {
                stored.spatial = None;
            }
        }
        Ok((
            FrameMasks {
                low_res: low,
                object_scores: scores,
            },
            layout,
            corrected,
        ))
    }

    /// The object a correction prompt lands on: the one whose low-res mask contains the first
    /// positive point (or the box centre), else the first object.
    fn correction_target(&self, prompt: &ObjectPrompt, low: &[f32]) -> usize {
        let anchor = prompt
            .points
            .iter()
            .find(|(_, l)| *l == 1)
            .map(|(p, _)| *p)
            .or_else(|| {
                let corners: Vec<[f32; 2]> = prompt
                    .points
                    .iter()
                    .filter(|(_, l)| *l == 2 || *l == 3)
                    .map(|(p, _)| *p)
                    .collect();
                (corners.len() == 2).then(|| {
                    [
                        (corners[0][0] + corners[1][0]) / 2.0,
                        (corners[0][1] + corners[1][1]) / 2.0,
                    ]
                })
            })
            // Background-only corrections act on the object under the first background point.
            .or_else(|| prompt.points.iter().find(|(_, l)| *l == 0).map(|(p, _)| *p));
        let plane = LOW_RES * LOW_RES;
        let from_mask = prompt.mask.as_ref().map(|mask| {
            // Mask prompt: the object overlapping it most (at low resolution).
            let small = resize_bilinear_chw(mask, 1, MASK_SIZE, MASK_SIZE, LOW_RES, LOW_RES)
                .unwrap_or_default();
            (0..self.objects)
                .max_by_key(|&o| {
                    small
                        .iter()
                        .zip(&low[o * plane..(o + 1) * plane])
                        .filter(|(m, l)| **m > 0.5 && **l > 0.0)
                        .count()
                })
                .unwrap_or(0)
        });
        anchor
            .and_then(|p| {
                let x = ((p[0] / self.source_width as f32) * LOW_RES as f32) as usize;
                let y = ((p[1] / self.source_height as f32) * LOW_RES as f32) as usize;
                let cell = y.min(LOW_RES - 1) * LOW_RES + x.min(LOW_RES - 1);
                (0..self.objects)
                    .filter(|&o| low[o * plane + cell] > 0.0)
                    .max_by(|&a, &b| low[a * plane + cell].total_cmp(&low[b * plane + cell]))
            })
            .or(from_mask)
            .unwrap_or(0)
    }

    /// Text features for a prompt (`text` graph, CPU FP32): ([32, 1, 256], padding mask [1, 32]).
    pub fn encode_text(&mut self, tokens: &[i64]) -> Result<TextFeatures> {
        ensure!(
            tokens.len() == TEXT_TOKENS,
            "SAM 3.1 text needs {TEXT_TOKENS} tokens"
        );
        let mut out = self.runner.run(
            "text",
            vec![(
                "tokens",
                Input::I64(ArrayD::from_shape_vec(
                    IxDyn(&[1, TEXT_TOKENS]),
                    tokens.to_vec(),
                )?),
            )],
        )?;
        Ok(TextFeatures {
            features: take(&mut out, "text", "text_features")?
                .into_iter()
                .collect(),
            padding: take(&mut out, "text", "text_mask")?
                .iter()
                .map(|&v| v > 0.5)
                .collect(),
        })
    }

    /// Run the detector on `frame` for a text prompt: detections with sigmoid(score) > 0.5, best
    /// first, each as 288x288 mask logits over the whole frame.
    pub fn detect(
        &mut self,
        frame: usize,
        rgb: &[u8],
        text: &TextFeatures,
    ) -> Result<Vec<Detection>> {
        self.features(frame, rgb, true)?;
        let f = self.features.as_ref().expect("features loaded");
        let det = f.det_fpn.as_ref().expect("detector features requested");
        let slots = DETECTOR_SLOTS;
        let inputs = vec![
            ("det_fpn0", Input::F32(det[0].clone())),
            ("det_fpn1", Input::F32(det[1].clone())),
            ("det_fpn2", Input::F32(det[2].clone())),
            (
                "text_features",
                f32_input(&[TEXT_TOKENS, 1, CHANNELS], text.features.clone())?,
            ),
            (
                "text_mask",
                bool_input(&[1, TEXT_TOKENS], text.padding.clone())?,
            ),
            // No geometric prompts: all slots are padding (upstream filler values).
            (
                "boxes",
                f32_input(&[slots, 1, 4], [0.5, 0.5, 0.1, 0.1].repeat(slots))?,
            ),
            ("box_mask", bool_input(&[1, slots], vec![true; slots])?),
            ("box_labels", Input::I64(ArrayD::zeros(IxDyn(&[slots, 1])))),
            (
                "points",
                f32_input(&[slots, 1, 2], [0.5, 0.5].repeat(slots))?,
            ),
            ("point_mask", bool_input(&[1, slots], vec![true; slots])?),
            (
                "point_labels",
                Input::I64(ArrayD::zeros(IxDyn(&[slots, 1]))),
            ),
        ];
        let mut out = self.runner.run("detector", inputs)?;
        let logits = take(&mut out, "detector", "pred_logits")?;
        let masks = take(&mut out, "detector", "pred_masks")?;
        let plane = LOW_RES * LOW_RES;
        let mut detections: Vec<Detection> = logits
            .iter()
            .enumerate()
            .map(|(query, &logit)| (query, sigmoid(logit)))
            .filter(|(_, score)| *score > 0.5)
            .map(|(query, score)| Detection {
                score,
                low_res: masks
                    .iter()
                    .skip(query * plane)
                    .take(plane)
                    .copied()
                    .collect(),
            })
            .collect();
        detections.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(detections)
    }

    /// A detection as a binary mask prompt at `MASK_SIZE` (logits resized, then > 0).
    pub fn detection_mask(detection: &Detection) -> Result<Vec<f32>> {
        Ok(resize_bilinear_chw(
            &detection.low_res,
            1,
            LOW_RES,
            LOW_RES,
            MASK_SIZE,
            MASK_SIZE,
        )?
        .into_iter()
        .map(|v| if v > 0.0 { 1.0 } else { 0.0 })
        .collect())
    }

    /// Whether a detection covers a source-pixel point.
    pub fn detection_contains(&self, detection: &Detection, point: [f32; 2]) -> bool {
        let x = ((point[0] / self.source_width as f32) * LOW_RES as f32) as usize;
        let y = ((point[1] / self.source_height as f32) * LOW_RES as f32) as usize;
        detection.low_res[y.min(LOW_RES - 1) * LOW_RES + x.min(LOW_RES - 1)] > 0.0
    }
}

/// [objects, 288, 288] logits -> [objects, H, W] bool at source resolution, non-overlapping.
fn source_masks(low_res: &[f32], objects: usize, height: usize, width: usize) -> Result<Vec<bool>> {
    let mut video = resize_bilinear_chw(low_res, objects, LOW_RES, LOW_RES, height, width)?;
    if objects > 1 {
        non_overlap(&mut video, objects);
    }
    Ok(video.into_iter().map(|v| v > 0.0).collect())
}

/// Union of the objects' source-resolution masks as 0/255 bytes.
fn source_union(low_res: &[f32], objects: usize, height: usize, width: usize) -> Result<Vec<u8>> {
    let source = source_masks(low_res, objects, height, width)?;
    let plane = height * width;
    Ok((0..plane)
        .map(|p| {
            if (0..objects).any(|o| source[o * plane + p]) {
                255
            } else {
                0
            }
        })
        .collect())
}

/// `_apply_non_overlapping_constraints`: keep only the highest-scoring object per pixel; others are
/// clamped to at most -10.
fn non_overlap(masks: &mut [f32], objects: usize) {
    let plane = masks.len() / objects;
    for p in 0..plane {
        let best = (0..objects)
            .max_by(|&a, &b| {
                masks[a * plane + p]
                    .total_cmp(&masks[b * plane + p])
                    .then(b.cmp(&a))
            })
            .unwrap_or(0);
        for o in 0..objects {
            if o != best {
                let v = &mut masks[o * plane + p];
                *v = v.min(-10.0);
            }
        }
    }
}

// ----- ONNX Runtime runner and developer CLI -----

/// Graphs under `<models>/vision/<precision>` and `<models>/tracker/<graph>/<precision>`.
pub struct OrtGraphs {
    models: PathBuf,
    precision: String,
    args: crate::Args,
    sessions: BTreeMap<String, ort::session::Session>,
    pub timing: BTreeMap<String, Vec<f64>>,
    pub detail: BTreeMap<String, Vec<f64>>,
    pub load_seconds: BTreeMap<String, f64>,
    pub pool: MemoryPool,
}

/// Which graphs keep DirectML's memory pool (arena). Without it every run allocates and frees its
/// transient buffers (slower); with it the pool keeps power-of-two buckets (more VRAM).
/// 16 frames 1080p, process peak: Lean ~4.1 GiB, Attention ~5.5 GiB (the default, ~22% faster),
/// Fast ~5.9 GiB.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPool {
    Lean,
    Attention,
    Fast,
}

impl MemoryPool {
    /// `Attention`, unless `BRP_SAM31_POOL=lean|fast` (developer comparisons).
    pub fn choose() -> Self {
        match std::env::var("BRP_SAM31_POOL").as_deref() {
            Ok("lean") => Self::Lean,
            Ok("fast") => Self::Fast,
            _ => Self::Attention,
        }
    }

    fn pooled(self, graph: &str) -> bool {
        match self {
            Self::Lean => false,
            Self::Attention => graph.starts_with("memory_attention"),
            Self::Fast => {
                graph.starts_with("memory_attention") || graph.starts_with("memory_encoder")
            }
        }
    }
}

impl OrtGraphs {
    pub fn new(models: PathBuf, precision: String, args: crate::Args) -> Self {
        Self {
            models,
            precision,
            args,
            sessions: BTreeMap::new(),
            timing: BTreeMap::new(),
            detail: BTreeMap::new(),
            load_seconds: BTreeMap::new(),
            pool: MemoryPool::Lean,
        }
    }
}

impl GraphRunner for OrtGraphs {
    fn note(&mut self, step: &str, seconds: f64) {
        self.detail
            .entry(step.to_owned())
            .or_default()
            .push(seconds);
    }

    fn has_graph(&self, graph: &str) -> bool {
        self.graph_path(graph).is_file()
    }

    fn run(
        &mut self,
        graph: &str,
        inputs: Vec<(&'static str, Input)>,
    ) -> Result<BTreeMap<String, ArrayD<f32>>> {
        self.run_outputs(graph, inputs, None)
    }

    fn run_select(
        &mut self,
        graph: &str,
        inputs: Vec<(&'static str, Input)>,
        outputs: &[&str],
    ) -> Result<BTreeMap<String, ArrayD<f32>>> {
        self.run_outputs(graph, inputs, Some(outputs))
    }
}

impl OrtGraphs {
    fn graph_path(&self, graph: &str) -> PathBuf {
        if graph == "text" {
            // The text encoder runs once per prompt on the CPU in FP32 (45 ms, no VRAM).
            self.models.join("text").join("fp32c").join("model.onnx")
        } else if graph == "vision" || graph == "detector" || graph == "frame" {
            self.models
                .join(graph)
                .join(&self.precision)
                .join("model.onnx")
        } else {
            self.models
                .join("tracker")
                .join(graph)
                .join(&self.precision)
                .join("model.onnx")
        }
    }

    fn run_outputs(
        &mut self,
        graph: &str,
        inputs: Vec<(&'static str, Input)>,
        select: Option<&[&str]>,
    ) -> Result<BTreeMap<String, ArrayD<f32>>> {
        use ort::{
            session::{
                SessionInputValue,
                run_options::{OutputSelector, RunOptions},
            },
            value::Tensor,
        };
        if !self.sessions.contains_key(graph) {
            let path = self.graph_path(graph);
            // DirectML graph fusion compiles memory attention into one workspace that holds every
            // query chunk's scores at once (~9 GiB); unfused it peaks ~1.8 GiB and is faster at FP32.
            let extra: &[(&str, &str)] = if graph.starts_with("memory_attention") {
                &[("ep.dml.disable_graph_fusion", "1")]
            } else {
                &[]
            };
            let args = if graph == "text" {
                crate::Args {
                    provider: "cpu".into(),
                    ..self.args.clone()
                }
            } else {
                crate::Args {
                    dml_memory_arena: self.pool.pooled(graph),
                    ..self.args.clone()
                }
            };
            let loading = std::time::Instant::now();
            let session = crate::create_session_with(&args, &path, None, extra)
                .with_context(|| format!("loading SAM 3.1 graph {}", path.display()))?;
            self.load_seconds
                .insert(graph.to_owned(), loading.elapsed().as_secs_f64());
            self.sessions.insert(graph.to_owned(), session);
        }
        let session = self.sessions.get_mut(graph).expect("inserted above");
        let prepare = std::time::Instant::now();
        let values: Vec<(String, SessionInputValue<'static>)> = inputs
            .into_iter()
            .map(|(name, value)| {
                let value: SessionInputValue<'static> = match value {
                    Input::F32(a) => Tensor::from_array(a)?.into(),
                    Input::I32(a) => Tensor::from_array(a)?.into(),
                    Input::I64(a) => Tensor::from_array(a)?.into(),
                    Input::Bool(a) => Tensor::from_array(a)?.into(),
                };
                Ok((name.to_owned(), value))
            })
            .collect::<Result<_>>()?;
        let names: Vec<String> = match select {
            Some(wanted) => wanted.iter().map(|n| (*n).to_owned()).collect(),
            None => session.outputs.iter().map(|o| o.name.clone()).collect(),
        };
        let prepared = prepare.elapsed().as_secs_f64();
        let started = std::time::Instant::now();
        // Unrequested outputs are not copied to the host.
        let options = match select {
            Some(_) => Some(
                RunOptions::new()?.with_outputs(
                    names
                        .iter()
                        .fold(OutputSelector::no_default(), |s, n| s.with(n.as_str())),
                ),
            ),
            None => None,
        };
        let outputs = match &options {
            Some(options) => session.run_with_options(values, options),
            None => session.run(values),
        }
        .with_context(|| format!("running SAM 3.1 {graph}"))?;
        let ran = started.elapsed().as_secs_f64();
        let mut owned = BTreeMap::new();
        for name in names {
            let output = outputs
                .get(name.as_str())
                .with_context(|| format!("SAM 3.1 {graph} graph has no output {name}"))?;
            // Bool outputs (the text padding mask) come back as 0/1.
            let value = match output.try_extract_array::<f32>() {
                Ok(array) => array.to_owned(),
                Err(_) => output
                    .try_extract_array::<bool>()?
                    .mapv(|b| if b { 1.0 } else { 0.0 }),
            };
            ensure!(
                value.iter().all(|v| v.is_finite()),
                "nonfinite SAM 3.1 {graph} output {name}"
            );
            owned.insert(name, value);
        }
        let total = started.elapsed().as_secs_f64();
        self.timing.entry(graph.to_owned()).or_default().push(total);
        // Developer breakdown: input tensor creation, session.run, output copies + checks.
        for (part, seconds) in [
            ("prepare", prepared),
            ("run", ran),
            ("extract", total - ran),
        ] {
            self.detail
                .entry(format!("{graph}:{part}"))
                .or_default()
                .push(seconds);
        }
        Ok(owned)
    }
}

/// `ck-native sam31-track --models DIR --frames FILE.rgb24 --width W --height H --object "x,y,l;x,y,l"
///   [--object ...] [--count N] [--precision fp16] [--provider directml|cpu] [--runtime DLL] --output DIR`
///
/// Developer check: points (relative 0..1) on frame 0, forward propagation, per-frame low-res logits
/// (`lowres-NNNN.npy`), union masks (`mask-NNNN.png`) and `report.json`.
pub fn run_cli(raw: &[String]) -> Result<()> {
    let mut models = None;
    let mut frames = None;
    let mut output = None;
    let mut runtime = None;
    let (mut width, mut height, mut count) = (0usize, 0usize, 0usize);
    let mut precision = "fp16".to_owned();
    let mut provider = "directml".to_owned();
    let mut objects: Vec<Vec<([f32; 2], i32)>> = Vec::new();
    // Relative prompts: (frame, x, y, positive, text) points and (frame, box) boxes.
    let mut points: Vec<(usize, [f32; 2], bool, Option<String>)> = Vec::new();
    let mut boxes: Vec<(usize, [f32; 4])> = Vec::new();
    let mut text: Option<String> = None;
    let mut it = raw.iter();
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--models" => models = Some(PathBuf::from(value)),
            "--frames" => frames = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--runtime" => runtime = Some(PathBuf::from(value)),
            "--width" => width = value.parse()?,
            "--height" => height = value.parse()?,
            "--count" => count = value.parse()?,
            "--precision" => precision = value.clone(),
            "--provider" => provider = value.clone(),
            "--object" => objects.push(parse_points(value)?),
            "--text" => text = Some(value.clone()),
            "--point" => {
                // frame:x,y,label[,text]
                let (frame, rest) = value
                    .split_once(':')
                    .context("--point is frame:x,y,label[,text]")?;
                let parts: Vec<&str> = rest.splitn(4, ',').collect();
                ensure!(parts.len() >= 3, "--point is frame:x,y,label[,text]");
                points.push((
                    frame.parse()?,
                    [parts[0].parse()?, parts[1].parse()?],
                    parts[2].trim() == "1",
                    parts.get(3).map(|t| t.to_string()),
                ));
            }
            "--box" => {
                let (frame, rest) = value
                    .split_once(':')
                    .context("--box is frame:x0,y0,x1,y1")?;
                let v: Vec<f32> = rest
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect::<std::result::Result<_, _>>()?;
                ensure!(v.len() == 4, "--box is frame:x0,y0,x1,y1");
                boxes.push((frame.parse()?, [v[0], v[1], v[2], v[3]]));
            }
            _ => bail!("unknown sam31-track argument {flag}"),
        }
    }
    let models = models.context("--models is required")?;
    let frames = frames.context("--frames is required")?;
    let output = output.context("--output is required")?;
    ensure!(width > 0 && height > 0, "--width and --height are required");
    ensure!(
        !objects.is_empty() || !points.is_empty() || !boxes.is_empty() || text.is_some(),
        "give --object, --point, --box or --text"
    );
    ensure!(
        matches!(provider.as_str(), "cpu" | "directml"),
        "provider must be cpu or directml"
    );
    let frame_bytes = width * height * 3;
    let data = std::fs::read(&frames).with_context(|| format!("reading {}", frames.display()))?;
    ensure!(
        data.len() % frame_bytes == 0,
        "frame file is not a whole number of {width}x{height} RGB24 frames"
    );
    let available = data.len() / frame_bytes;
    let count = if count == 0 {
        available
    } else {
        count.min(available)
    };
    std::fs::create_dir_all(&output)?;
    let args = crate::Args {
        command: "sam31-track".into(),
        runtime,
        provider: provider.clone(),
        threads: 4,
        ort_optimization: "all".into(),
        ..Default::default()
    };
    crate::initialize_runtime(&args)?;
    let config: Config =
        serde_json::from_slice(&std::fs::read(models.join("tracker/config.json"))?)?;
    let constants = Constants::load(&models.join("tracker/constants"), &config)?;
    let merges = std::fs::read_to_string(models.join("text/bpe_merges.txt")).unwrap_or_default();
    let mut runner = OrtGraphs::new(models, precision.clone(), args);
    let pool = MemoryPool::choose();
    runner.pool = pool;
    let mut tracker = Tracker::new(runner, config, constants, count, height, width)?;
    let mut resolver = PromptResolver::new(&merges)?;
    let (w, h) = (width as f32, height as f32);
    let mut layer = LayerPrompts {
        text,
        frames: BTreeMap::new(),
    };
    for object in &objects {
        for (p, l) in object {
            layer.frames.entry(0).or_default().points.push(PointPrompt {
                xy: [p[0] * w, p[1] * h],
                positive: *l == 1,
                text: None,
            });
        }
    }
    for (frame, p, positive, text) in points {
        layer
            .frames
            .entry(frame)
            .or_default()
            .points
            .push(PointPrompt {
                xy: [p[0] * w, p[1] * h],
                positive,
                text,
            });
    }
    for (frame, b) in boxes {
        layer
            .frames
            .entry(frame)
            .or_default()
            .boxes
            .push([b[0] * w, b[1] * h, b[2] * w, b[3] * h]);
    }
    let started = std::time::Instant::now();
    let mut rows = Vec::new();
    track_layer(
        &mut tracker,
        &mut resolver,
        &layer,
        count,
        |t| Ok(data[t * frame_bytes..(t + 1) * frame_bytes].to_vec()),
        |t, tracker, masks| {
            let n = masks.object_scores.len();
            write_npy(
                output.join(format!("lowres-{t:04}.npy")),
                &ArrayD::from_shape_vec(IxDyn(&[n, 1, LOW_RES, LOW_RES]), masks.low_res.clone())?,
            )?;
            let union = union_mask(tracker, masks)?;
            image::save_buffer(
                output.join(format!("mask-{t:04}.png")),
                &union,
                width as u32,
                height as u32,
                image::ExtendedColorType::L8,
            )?;
            rows.push(serde_json::json!({
                "frame": t, "scores": masks.object_scores,
                "pixels": union.iter().filter(|&&v| v > 0).count(),
                "conditioning": tracker.is_conditioning(t),
            }));
            Ok(())
        },
    )?;
    let wall = started.elapsed().as_secs_f64();
    let median = |v: &Vec<f64>| {
        let mut s = v.clone();
        s.sort_by(f64::total_cmp);
        s[s.len() / 2]
    };
    let report = serde_json::json!({
        "frames": count, "objects": tracker.objects(), "precision": precision, "provider": provider,
        "wall_seconds": wall, "memory_pool": pool,
        "median_seconds": tracker.runner.timing.iter().map(|(k, v)| (k.clone(), median(v))).collect::<BTreeMap<_, _>>(),
        "total_seconds": tracker.runner.timing.iter().map(|(k, v)| (k.clone(), v.iter().sum::<f64>())).collect::<BTreeMap<_, _>>(),
        "calls": tracker.runner.timing.iter().map(|(k, v)| (k.clone(), v.len())).collect::<BTreeMap<_, _>>(),
        "detail_total_seconds": tracker.runner.detail.iter().map(|(k, v)| (k.clone(), v.iter().sum::<f64>())).collect::<BTreeMap<_, _>>(),
        "load_seconds": tracker.runner.load_seconds,
        "per_frame": rows,
    });
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report["median_seconds"])?
    );
    println!("frames={count} wall={wall:.2}s");
    Ok(())
}

/// Union of the objects' source-resolution masks as 0/255 bytes.
fn union_mask<R: GraphRunner>(tracker: &Tracker<R>, masks: &FrameMasks) -> Result<Vec<u8>> {
    source_union(
        &masks.low_res,
        masks.object_scores.len(),
        tracker.source_height,
        tracker.source_width,
    )
}

/// Desktop job: track one layer's prompts through `count` RGB24 frames of `source` (the decoded
/// range), streaming binary source-resolution masks to the app. FP16 graphs, text encoder on CPU.
pub(crate) fn run_desktop(
    args: &crate::Args,
    models: &Path,
    source: &Path,
    count: usize,
    height: usize,
    width: usize,
    layer: &LayerPrompts,
) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let started = std::time::Instant::now();
    crate::initialize_runtime(args)?;
    let config: Config =
        serde_json::from_slice(&std::fs::read(models.join("tracker/config.json"))?)?;
    let constants = Constants::load(&models.join("tracker/constants"), &config)?;
    let mut runner = OrtGraphs::new(models.to_path_buf(), "fp16".into(), args.clone());
    let pool = MemoryPool::choose();
    runner.pool = pool;
    let mut tracker = Tracker::new(runner, config, constants, count, height, width)?;
    let mut resolver = PromptResolver::new(
        &std::fs::read_to_string(models.join("text/bpe_merges.txt"))
            .context("reading the SAM 3.1 text vocabulary")?,
    )?;
    let frame_bytes = height * width * 3;
    let mut file = std::fs::File::open(source)?;
    ensure!(
        file.metadata()?.len() == (frame_bytes * count) as u64,
        "SAM 3.1 RGB cache length mismatch"
    );
    // Source-size masks and PNG writes run on a writer thread while the next frame tracks.
    let (sender, receiver) = std::sync::mpsc::sync_channel::<(usize, Vec<f32>, Vec<f32>)>(4);
    let writer = std::thread::spawn(move || -> Result<Vec<serde_json::Value>> {
        let mut frames = Vec::new();
        for (t, low_res, scores) in receiver {
            let union = source_union(&low_res, scores.len(), height, width)?;
            crate::desktop::write_frame(t, width, height, None, None, None, Some(&union))?;
            frames.push(serde_json::json!({"frame": t, "scores": scores}));
            crate::desktop::progress("sam_video", frames.len(), count)?;
        }
        Ok(frames)
    });
    let tracked = track_layer(
        &mut tracker,
        &mut resolver,
        layer,
        count,
        |t| {
            let mut rgb = vec![0u8; frame_bytes];
            file.seek(SeekFrom::Start((t * frame_bytes) as u64))?;
            file.read_exact(&mut rgb)?;
            Ok(rgb)
        },
        |t, _, masks| {
            sender
                .send((t, masks.low_res.clone(), masks.object_scores.clone()))
                .map_err(|_| anyhow::anyhow!("SAM 3.1 mask writer stopped"))
        },
    );
    drop(sender);
    let written = writer
        .join()
        .map_err(|_| anyhow::anyhow!("SAM 3.1 mask writer panicked"))?;
    tracked?;
    let frames = written?;
    let report = serde_json::json!({
        "schema_version": 1, "component": "sam31_video_desktop_inference", "execution_success": true,
        "validation_performed": false, "source_frames": count, "objects": tracker.objects(),
        "precision": "fp16", "requested_provider": args.provider,
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "memory_pool": pool,
        "frame_graph": tracker.runner.has_graph("frame"),
        "graph_seconds": tracker.runner.timing.iter().map(|(k, v)| (k.clone(), v.iter().sum::<f64>())).collect::<BTreeMap<_, _>>(),
        "frames": frames,
        "scope": "SAM 3.1 tracker: objects from the earliest prompted frame (points/box/text), tracked forward then backward, corrections on other prompted frames; binary source-resolution PNG masks",
    });
    std::fs::write(
        args.report.as_ref().context("desktop report missing")?,
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

/// A point prompt in source pixels, optionally carrying text ("the <text> at this point").
#[derive(Clone, Debug)]
pub struct PointPrompt {
    pub xy: [f32; 2],
    pub positive: bool,
    pub text: Option<String>,
}

/// Prompts of one layer on one frame.
#[derive(Clone, Debug, Default)]
pub struct FramePrompts {
    pub points: Vec<PointPrompt>,
    /// Box as [x0, y0, x1, y1] in source pixels.
    pub boxes: Vec<[f32; 4]>,
}

/// Everything one layer asks SAM 3.1 for over a range (frames relative to its start).
#[derive(Clone, Debug, Default)]
pub struct LayerPrompts {
    /// Whole-frame text prompt on the first frame: every detection becomes an object.
    pub text: Option<String>,
    pub frames: BTreeMap<usize, FramePrompts>,
}

/// Resolves text into detector masks with one text encoding per distinct string.
pub struct PromptResolver {
    tokenizer: crate::clip_tokenizer::ClipTokenizer,
    texts: BTreeMap<String, TextFeatures>,
}

impl PromptResolver {
    pub fn new(merges: &str) -> Result<Self> {
        Ok(Self {
            tokenizer: crate::clip_tokenizer::ClipTokenizer::from_merges(merges)?,
            texts: BTreeMap::new(),
        })
    }

    fn detections<R: GraphRunner>(
        &mut self,
        tracker: &mut Tracker<R>,
        frame: usize,
        rgb: &[u8],
        text: &str,
    ) -> Result<Vec<Detection>> {
        if !self.texts.contains_key(text) {
            let tokens = self.tokenizer.tokenize(text, TEXT_TOKENS)?;
            let features = tracker.encode_text(&tokens)?;
            self.texts.insert(text.to_owned(), features);
        }
        tracker.detect(frame, rgb, &self.texts[text])
    }

    /// The point-and-box object of one frame. A point with text picks the detection of that text
    /// under it (best score) as the object's mask, refined by the frame's other points and box;
    /// without a matching detection the text point is an ordinary point.
    pub fn object<R: GraphRunner>(
        &mut self,
        tracker: &mut Tracker<R>,
        frame: usize,
        rgb: &[u8],
        prompts: &FramePrompts,
        first_frame: bool,
    ) -> Result<Option<ObjectPrompt>> {
        let mut object = ObjectPrompt::default();
        let mut picked_text = false;
        for point in &prompts.points {
            let text = point
                .text
                .as_deref()
                .map(str::trim)
                .filter(|t| !t.is_empty());
            if let (Some(text), true, false) = (text, point.positive, picked_text) {
                let detections = self.detections(tracker, frame, rgb, text)?;
                if let Some(hit) = detections
                    .iter()
                    .find(|d| tracker.detection_contains(d, point.xy))
                {
                    object.mask = Some(Tracker::<R>::detection_mask(hit)?);
                    picked_text = true;
                    continue;
                }
            }
            object.points.push((point.xy, i32::from(point.positive)));
        }
        if let Some(b) = prompts.boxes.last() {
            object.points.push(([b[0].min(b[2]), b[1].min(b[3])], 2));
            object.points.push(([b[0].max(b[2]), b[1].max(b[3])], 3));
        }
        let positive =
            object.mask.is_some() || object.points.iter().any(|(_, l)| *l == 1 || *l == 2);
        // A new object needs something positive; a correction may be background points only.
        let usable = positive || (!first_frame && !object.points.is_empty());
        Ok(usable.then_some(object))
    }

    /// Objects of a layer's anchor frame: the point/box object, then one object per text detection.
    pub fn first_frame_objects<R: GraphRunner>(
        &mut self,
        tracker: &mut Tracker<R>,
        anchor: usize,
        rgb: &[u8],
        layer: &LayerPrompts,
    ) -> Result<Vec<ObjectPrompt>> {
        let mut objects = Vec::new();
        if let Some(prompts) = layer.frames.get(&anchor)
            && let Some(object) = self.object(tracker, anchor, rgb, prompts, true)?
        {
            objects.push(object);
        }
        if let Some(text) = layer
            .text
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
        {
            for detection in self.detections(tracker, anchor, rgb, text)? {
                if objects.len() == SLOTS {
                    break;
                }
                objects.push(ObjectPrompt {
                    points: Vec::new(),
                    mask: Some(Tracker::<R>::detection_mask(&detection)?),
                });
            }
        }
        Ok(objects)
    }
}

/// The frame tracking starts from: the earliest frame with a subject point or a box, else the
/// first frame (where a text-only layer is detected).
pub fn anchor_frame(layer: &LayerPrompts) -> usize {
    layer
        .frames
        .iter()
        .find(|(_, p)| !p.boxes.is_empty() || p.points.iter().any(|q| q.positive))
        .map_or(0, |(t, _)| *t)
}

/// Track one layer through `count` frames, upstream's "both" direction: objects from the anchor
/// frame (`anchor_frame`), forward to the end, then backward to the first frame. Prompts on any
/// other frame correct the track when either pass reaches it. `frame(t)` returns RGB24 of
/// relative frame t; `emit` gets every frame's masks (forward frames first, then backward).
pub fn track_layer<R: GraphRunner>(
    tracker: &mut Tracker<R>,
    resolver: &mut PromptResolver,
    layer: &LayerPrompts,
    count: usize,
    mut frame: impl FnMut(usize) -> Result<Vec<u8>>,
    mut emit: impl FnMut(usize, &Tracker<R>, &FrameMasks) -> Result<()>,
) -> Result<()> {
    let anchor = anchor_frame(layer);
    ensure!(
        anchor < count,
        "SAM 3.1 prompt frame {anchor} is outside the range"
    );
    let rgb = frame(anchor)?;
    let objects = resolver.first_frame_objects(tracker, anchor, &rgb, layer)?;
    ensure!(
        !objects.is_empty(),
        "Nothing to track: add a subject point or box, or a text prompt that finds something"
    );
    tracker.plan_backward_from(anchor);
    tracker.set_reverse(false);
    let masks = tracker.add_objects(anchor, &rgb, &objects)?;
    emit(anchor, tracker, &masks)?;
    let mut step = |t: usize, tracker: &mut Tracker<R>| -> Result<()> {
        let clock = std::time::Instant::now();
        let rgb = frame(t)?;
        tracker
            .runner
            .note("host:read_frame", clock.elapsed().as_secs_f64());
        let correction = match layer.frames.get(&t) {
            Some(prompts) => resolver.object(tracker, t, &rgb, prompts, false)?,
            None => None,
        };
        let (masks, _, _) = tracker.propagate_with(t, &rgb, correction.as_ref())?;
        let clock = std::time::Instant::now();
        emit(t, tracker, &masks)?;
        tracker
            .runner
            .note("host:emit", clock.elapsed().as_secs_f64());
        Ok(())
    };
    for t in anchor + 1..count {
        step(t, tracker)?;
    }
    tracker.set_reverse(true);
    for t in (0..anchor).rev() {
        step(t, tracker)?;
    }
    tracker.set_reverse(false);
    Ok(())
}

/// "x,y,label;x,y,label" with relative coordinates.
fn parse_points(value: &str) -> Result<Vec<([f32; 2], i32)>> {
    value
        .split(';')
        .filter(|s| !s.trim().is_empty())
        .map(|point| {
            let parts: Vec<&str> = point.split(',').map(str::trim).collect();
            ensure!(parts.len() == 3, "point must be x,y,label: {point}");
            let (x, y, label): (f32, f32, i32) =
                (parts[0].parse()?, parts[1].parse()?, parts[2].parse()?);
            ensure!(
                (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y) && (label == 0 || label == 1),
                "invalid point {point}"
            );
            Ok(([x, y], label))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (input name, shape) of each input of one recorded call.
    type InputShapes = Vec<(String, Vec<usize>)>;

    /// Records every call and returns zero-filled outputs of the right shapes.
    #[derive(Default)]
    struct FakeRunner {
        calls: Vec<(String, InputShapes)>,
        key_valid: Vec<Vec<bool>>,
    }

    impl GraphRunner for FakeRunner {
        fn run(
            &mut self,
            graph: &str,
            inputs: Vec<(&'static str, Input)>,
        ) -> Result<BTreeMap<String, ArrayD<f32>>> {
            let mut shapes = Vec::new();
            for (name, input) in &inputs {
                let shape = match input {
                    Input::F32(a) => a.shape().to_vec(),
                    Input::I32(a) => a.shape().to_vec(),
                    Input::I64(a) => a.shape().to_vec(),
                    Input::Bool(a) => {
                        if *name == "key_valid" {
                            self.key_valid.push(a.iter().copied().collect());
                        }
                        a.shape().to_vec()
                    }
                };
                shapes.push((name.to_string(), shape));
            }
            self.calls.push((graph.to_owned(), shapes));
            let z = |shape: &[usize]| ArrayD::zeros(IxDyn(shape));
            let mut out = BTreeMap::new();
            match graph {
                "vision" => {
                    for (name, shape) in [
                        ("int_fpn0", [1, 256, 288, 288]),
                        ("int_fpn1", [1, 256, 144, 144]),
                        ("int_fpn2", [1, 256, 72, 72]),
                        ("prop_fpn0", [1, 256, 288, 288]),
                        ("prop_fpn1", [1, 256, 144, 144]),
                        ("prop_fpn2", [1, 256, 72, 72]),
                    ] {
                        out.insert(name.to_owned(), z(&shape));
                    }
                    // Detector features are only carried to the detector; keep them tiny.
                    for name in ["det_fpn0", "det_fpn1", "det_fpn2"] {
                        out.insert(name.to_owned(), z(&[1, 1, 1, 1]));
                    }
                }
                "tracker_features" => {
                    for (name, shape) in [
                        ("int_hr0", [1, 32, 288, 288]),
                        ("int_hr1", [1, 64, 144, 144]),
                        ("int_pix", [1, 256, 72, 72]),
                        ("prop_hr0", [1, 32, 288, 288]),
                        ("prop_hr1", [1, 64, 144, 144]),
                    ] {
                        out.insert(name.to_owned(), z(&shape));
                    }
                }
                "interactive_multi" | "interactive_single" => {
                    out.insert(
                        "low_res_masks".into(),
                        ArrayD::from_elem(IxDyn(&[1, 1, 288, 288]), 5.0),
                    );
                    out.insert(
                        "object_score_logits".into(),
                        ArrayD::from_elem(IxDyn(&[1, 1]), 10.0),
                    );
                    out.insert("obj_ptr".into(), ArrayD::from_elem(IxDyn(&[1, 256]), 1.0));
                }
                "propagation" => {
                    out.insert(
                        "low_res_masks".into(),
                        ArrayD::from_elem(IxDyn(&[16, 1, 288, 288]), 3.0),
                    );
                    out.insert(
                        "object_score_logits".into(),
                        ArrayD::from_elem(IxDyn(&[16, 1]), 8.0),
                    );
                    out.insert("obj_ptr".into(), ArrayD::from_elem(IxDyn(&[16, 256]), 2.0));
                }
                "memory_encoder" => {
                    out.insert("maskmem_features".into(), z(&[1, 256, 72, 72]));
                    out.insert("maskmem_pos".into(), z(&[1, 256, 72, 72]));
                }
                g if g.starts_with("memory_attention") => {
                    out.insert("conditioned".into(), z(&[TOKENS, 1, CHANNELS]));
                }
                other => bail!("unexpected graph {other}"),
            }
            Ok(out)
        }
    }

    fn config() -> Config {
        Config {
            image_size: 1008,
            num_maskmem: 7,
            max_cond_frames_in_attn: 4,
            max_obj_ptrs_in_encoder: 16,
            multimask_max_pt_num: 1,
            sigmoid_scale_for_mem_enc: 2.0,
            sigmoid_bias_for_mem_enc: -1.0,
            condition_as_mask_input_fg: 1.0,
            condition_as_mask_input_bg: 0.0,
            keep_first_cond_frame: false,
            memory_temporal_stride_for_eval: 1,
            use_signed_tpos_enc_to_obj_ptrs: false,
            only_obj_ptrs_in_the_past_for_eval: false,
            use_maskmem_tpos_v2: true,
            use_memory_selection: false,
            multiplex_count: 16,
        }
    }

    fn constants() -> Constants {
        Constants {
            maskmem_tpos_enc: vec![0.0; 7 * CHANNELS],
            obj_ptr_tpos_proj_weight: vec![0.0; CHANNELS * CHANNELS],
            obj_ptr_tpos_proj_bias: vec![0.0; CHANNELS],
            image_pos: vec![0.0; TOKENS * CHANNELS],
        }
    }

    #[test]
    fn memory_selection_matches_upstream_for_forward_tracking() {
        let (h, w) = (36, 64);
        let rgb = vec![128u8; h * w * 3];
        let mut tracker =
            Tracker::new(FakeRunner::default(), config(), constants(), 16, h, w).unwrap();
        tracker
            .add_points(0, &rgb, &[vec![([30.0, 20.0], 1)]])
            .unwrap();
        let mut layouts = Vec::new();
        for t in 1..16 {
            layouts.push(tracker.propagate(t, &rgb).unwrap().1);
        }
        // Frame 1: conditioning frame 0 only, one pointer frame (distance 1).
        assert_eq!(layouts[0].spatial, vec![(1, 0)]);
        assert_eq!(layouts[0].pointers, vec![(1, 0)]);
        assert_eq!(layouts[0].graph, "memory_attention_mem7");
        // Frame 3: cond 0 (t_pos 3), then previous frames 1 (t_pos 5) and 2 (t_pos 6).
        assert_eq!(layouts[2].spatial, vec![(3, 0), (5, 1), (6, 2)]);
        assert_eq!(layouts[2].pointers, vec![(3, 0), (1, 2), (2, 1)]);
        // Frame 15: cond 0 (t_pos 15, out-of-range encoding) + the last six frames; 15 pointer frames.
        let last = &layouts[14];
        assert_eq!(
            last.spatial,
            vec![(15, 0), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14)]
        );
        assert_eq!(last.frames, 7);
        assert_eq!(last.pointer_tokens, 15 * SLOTS);
        assert_eq!(last.pointers.first(), Some(&(15, 0)));
        assert_eq!(last.pointers.last(), Some(&(14, 1)));
        // key_valid marks exactly the used frames and pointer tokens.
        let valid = tracker.runner.key_valid.last().unwrap();
        assert_eq!(valid.len(), 7 * TOKENS + POINTER_TOKENS);
        assert_eq!(
            valid.iter().filter(|v| **v).count(),
            7 * TOKENS + 15 * SLOTS
        );
        assert!(valid[7 * TOKENS + 15 * SLOTS..].iter().all(|v| !v));
        // Old non-conditioning frames drop their spatial memory but keep pointers.
        assert!(tracker.non_cond[&1].spatial.is_none() && tracker.non_cond[&14].spatial.is_some());
    }

    #[test]
    fn backward_pass_selects_following_frames_like_upstream_reverse() {
        let (h, w) = (36, 64);
        let rgb = vec![128u8; h * w * 3];
        let mut tracker =
            Tracker::new(FakeRunner::default(), config(), constants(), 16, h, w).unwrap();
        tracker.plan_backward_from(8);
        tracker
            .add_points(8, &rgb, &[vec![([30.0, 20.0], 1)]])
            .unwrap();
        for t in 9..16 {
            tracker.propagate(t, &rgb).unwrap();
        }
        // The frames the backward pass needs keep their spatial memory; older ones do not.
        assert!((9..14).all(|t| tracker.non_cond[&t].spatial.is_some()));
        tracker.set_reverse(true);
        let layout = tracker.propagate(7, &rgb).unwrap().1;
        // Cond 8 at reversed t_pos (8 - 7) = 1, then frames 13..9 (t_pos 1..5); 8 is selected cond.
        assert_eq!(
            layout.spatial,
            vec![(1, 8), (1, 13), (2, 12), (3, 11), (4, 10), (5, 9)]
        );
        // Pointers: cond 8 (distance 1), then the following frames up to the end of the video.
        assert_eq!(layout.pointers.first(), Some(&(1, 8)));
        assert_eq!(
            &layout.pointers[1..],
            &[(2, 9), (3, 10), (4, 11), (5, 12), (6, 13), (7, 14), (8, 15)]
        );
        let layout = tracker.propagate(6, &rgb).unwrap().1;
        assert_eq!(
            layout.spatial,
            vec![(2, 8), (1, 12), (2, 11), (3, 10), (4, 9), (6, 7)]
        );
        for t in (0..6).rev() {
            tracker.propagate(t, &rgb).unwrap();
        }
        // Reverse tracking drops spatial memory far behind it (here: after the current frame).
        assert!(tracker.non_cond[&7].spatial.is_none() && tracker.non_cond[&1].spatial.is_some());
    }

    #[test]
    fn token_transposes_round_trip_and_match_the_sequential_layout() {
        let chw: Vec<f32> = (0..TOKENS * CHANNELS).map(|i| i as f32).collect();
        let tokens = tokens_from_chw(&chw);
        for (t, c) in [(0, 0), (1, 0), (0, 1), (5183, 255), (64, 7)] {
            assert_eq!(tokens[t * CHANNELS + c], chw[c * TOKENS + t]);
        }
        assert_eq!(chw_from_tokens(&tokens), chw);
        let doubled = tokens_from_chw_map(&chw, |x| x * 2.0);
        assert_eq!(doubled[3 * CHANNELS + 2], 2.0 * chw[2 * TOKENS + 3]);
    }

    #[test]
    fn into_vec_keeps_logical_order() {
        let a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(into_vec(a.clone()), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let t = a.t().to_owned();
        assert_eq!(into_vec(t), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        let sliced = a.slice_move(ndarray::s![1.., ..]).into_dyn();
        assert_eq!(into_vec(sliced), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn memory_pool_pools_memory_attention_by_default() {
        assert!(!MemoryPool::Lean.pooled("memory_attention_mem7"));
        assert!(MemoryPool::Attention.pooled("memory_attention_mem10"));
        assert!(
            !MemoryPool::Attention.pooled("memory_encoder")
                && !MemoryPool::Attention.pooled("frame")
        );
        assert!(MemoryPool::Fast.pooled("memory_encoder") && !MemoryPool::Fast.pooled("frame"));
    }

    #[test]
    fn anchor_is_the_earliest_positive_point_or_box() {
        let point = |positive| PointPrompt {
            xy: [1.0, 1.0],
            positive,
            text: None,
        };
        let mut layer = LayerPrompts::default();
        assert_eq!(anchor_frame(&layer), 0);
        layer.frames.entry(3).or_default().points.push(point(false));
        layer.frames.entry(9).or_default().points.push(point(true));
        assert_eq!(anchor_frame(&layer), 9);
        layer
            .frames
            .entry(5)
            .or_default()
            .boxes
            .push([0.0, 0.0, 4.0, 4.0]);
        assert_eq!(anchor_frame(&layer), 5);
    }

    #[test]
    fn conditioning_selection_keeps_nearest_four() {
        let mut tracker =
            Tracker::new(FakeRunner::default(), config(), constants(), 30, 8, 8).unwrap();
        for t in [0, 5, 10, 20, 25] {
            tracker.cond.push((
                t,
                Stored {
                    spatial: None,
                    obj_ptr: vec![0.0; SLOTS * CHANNELS],
                },
            ));
        }
        let (selected, unselected) = tracker.select_conditioning(12);
        assert_eq!(selected, vec![10, 20, 5, 0]);
        assert_eq!(unselected, vec![25]);
    }

    #[test]
    fn single_point_uses_multimask_and_two_points_do_not() {
        let rgb = vec![0u8; 8 * 8 * 3];
        let mut one = Tracker::new(FakeRunner::default(), config(), constants(), 4, 8, 8).unwrap();
        one.add_points(0, &rgb, &[vec![([1.0, 1.0], 1)]]).unwrap();
        assert!(
            one.runner
                .calls
                .iter()
                .any(|(g, _)| g == "interactive_multi")
        );
        let mut two = Tracker::new(FakeRunner::default(), config(), constants(), 4, 8, 8).unwrap();
        two.add_points(0, &rgb, &[vec![([1.0, 1.0], 1), ([2.0, 2.0], 0)]])
            .unwrap();
        assert!(
            two.runner
                .calls
                .iter()
                .any(|(g, _)| g == "interactive_single")
        );
    }

    #[test]
    fn sine_pe_and_non_overlap() {
        let pe = sine_pe(0.5, 8);
        assert!((pe[0] - 0.5f32.sin()).abs() < 1e-7 && (pe[4] - 0.5f32.cos()).abs() < 1e-7);
        assert!((pe[2] - (0.5f32 / 10000f32.powf(0.5)).sin()).abs() < 1e-7);
        // Object 1 wins both pixels; object 0 is clamped to <= -10 there. Ties go to the lower index.
        let mut masks = vec![1.0, -2.0, 3.0, 0.5];
        non_overlap(&mut masks, 2);
        assert_eq!(masks, vec![-10.0, -10.0, 3.0, 0.5]);
        let mut tie = vec![2.0, 2.0];
        non_overlap(&mut tie, 2);
        assert_eq!(tie, vec![2.0, -10.0]);
    }

    #[test]
    fn model_input_rounds_through_fp16() {
        let rgb = vec![200u8; 4 * 4 * 3];
        let input = model_input(&rgb, 4, 4, 2).unwrap();
        let expected = f16::from_f32(f16::from_f32(200.0 / 255.0).to_f32() - 0.5).to_f32() + 0.5;
        assert!(input.iter().all(|&v| v == expected));
        assert_eq!(input.shape(), [1, 3, 2, 2]);
    }

    #[test]
    fn points_parse_relative() {
        assert_eq!(
            parse_points("0.5,0.25,1;0.1,0.2,0").unwrap(),
            vec![([0.5, 0.25], 1), ([0.1, 0.2], 0)]
        );
        assert!(parse_points("1.5,0,1").is_err());
    }
}
