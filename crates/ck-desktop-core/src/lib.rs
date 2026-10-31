//! Stable native desktop/worker protocol. No interpreter or inference dependency.
mod assets;
mod credentials;
mod devices;
mod media;
mod paths;
mod project;
mod system_runtime;
pub mod updater;
mod worker;

pub use assets::{
    AssetCancellation, AssetFile, AssetGroup, AssetManager, AssetManifest, AssetPhase,
    AssetProgress, AssetStatus,
};
pub use ck_core::mask_layers::{InterpMode, MaskKind, MaskLayer, MaskMode, Tangent};
pub use credentials::read_huggingface_token;
pub use devices::{DEVICE_ID_ENV, GpuDevice, enumerate_gpus, parse_device_id, selected_device_id};
pub use media::{MediaInfo, Rational, SourceIdentity, extract_source_frame};
pub use paths::RuntimePaths;
pub use project::{
    DespillSettings, HintQuality, MaskLayers, MaskStacks, Project, SamLayer, SamLayerMode,
    SceneOverrideValue, SceneOverrides,
};
pub use system_runtime::{SharedOrtRuntime, shared_ort_library, shared_ort_runtime};
pub use worker::Worker;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Desktop job/protocol schema for worker requests and events.
pub const SCHEMA_VERSION: u32 = 1;

/// Native project document schema (`Project`, stored in `.chroma` indexes).
///
/// Version 2 adds per-scene garbage/despill `mask_layers`. The bump is
/// backward-compatible on load: version-1 documents deserialise with empty mask
/// stacks and are rewritten as version 2 on save. The desktop job/protocol
/// [`SCHEMA_VERSION`] is separate because the native worker contract is pinned
/// independently of the project file format. Painted mask rasters are `.chroma`
/// blobs, not JSON, so a version-2 project without mask layers round-trips
/// exactly like a version-1 project.
pub const PROJECT_SCHEMA_VERSION: u32 = 2;

/// All engine ranges are zero-based and half-open. Never convert via seek seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameRange {
    pub start: u64,
    pub end_exclusive: u64,
}
impl FrameRange {
    pub fn new(start: u64, end_exclusive: u64) -> Result<Self> {
        ensure!(
            start < end_exclusive,
            "A frame range must contain at least one frame"
        );
        Ok(Self {
            start,
            end_exclusive,
        })
    }
    pub fn len(self) -> u64 {
        self.end_exclusive.saturating_sub(self.start)
    }
    pub fn is_empty(self) -> bool {
        self.start >= self.end_exclusive
    }
    pub fn contains(self, frame: u64) -> bool {
        frame >= self.start && frame < self.end_exclusive
    }
    pub fn validate(self, frames: u64) -> Result<()> {
        ensure!(
            !self.is_empty() && self.end_exclusive <= frames,
            "Frame range is outside the source footage"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneRange {
    pub index: u32,
    pub start: u64,
    pub end_exclusive: u64,
    pub enabled: bool,
}
impl SceneRange {
    pub fn range(&self) -> FrameRange {
        FrameRange {
            start: self.start,
            end_exclusive: self.end_exclusive,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provider {
    Cpu,
    #[default]
    #[serde(rename = "directml")]
    DirectMl,
    Cuda,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CkWindowMode {
    #[default]
    Balanced,
    #[serde(alias = "auto")]
    Wide,
    Custom,
}

/// Requested render policy. The worker resolves automatic values for each job.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CkSettings {
    pub window_mode: CkWindowMode,
    /// Custom only: zero selects the automatic wide-window policy.
    pub hann_chunk: usize,
    /// Custom only: zero selects half of the resolved chunk, at least one.
    pub hann_stride: usize,
    pub low_vram: bool,
}
impl CkSettings {
    pub fn validate(&self) -> Result<()> {
        if self.window_mode == CkWindowMode::Custom && self.hann_chunk > 0 && self.hann_stride > 0 {
            ensure!(
                self.hann_stride <= self.hann_chunk,
                "Custom Hann stride must not exceed the custom chunk"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamPrompt {
    /// Absolute zero-based source frame, in original image pixel coordinates.
    pub frame: u64,
    pub x: f32,
    pub y: f32,
    pub positive: bool,
    /// SAM layer this point belongs to (absent before layers existed: the first layer).
    #[serde(default)]
    pub layer: u64,
    /// SAM 3.1: "the <text> at this point" (a detection of the text under the point).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// A SAM 3.1 box prompt on one frame, corners in original image pixels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SamBox {
    pub frame: u64,
    #[serde(default)]
    pub layer: u64,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

/// Subject segmentation model. SAM 3.1 is optional: it needs `models/sam3.1` beside `models/sam2`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamModel {
    #[default]
    Sam21,
    Sam31,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum HintInput {
    #[default]
    None,
    FirstFrame {
        mask_file: PathBuf,
        frame: u64,
        quality: f32,
    },
    /// File index = absolute source frame - frame_offset. SAM outputs use offset 0.
    Video {
        mask_directory: PathBuf,
        frame_offset: i64,
        quality: f32,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum JobKind {
    DetectScenes {
        threshold: f64,
        min_scene_frames: u64,
    },
    CorridorKey {
        range: FrameRange,
        provider: Provider,
        hints: HintInput,
        #[serde(default)]
        settings: CkSettings,
    },
    SamVideo {
        range: FrameRange,
        provider: Provider,
        prompts: Vec<SamPrompt>,
        /// Absent in jobs written before SAM 3.1 existed: SAM 2.1.
        #[serde(default)]
        model: SamModel,
        /// SAM 3.1 boxes of this layer within the range.
        #[serde(default)]
        boxes: Vec<SamBox>,
        /// SAM 3.1 whole-frame text prompt of this layer (first frame of the range).
        #[serde(default)]
        text: Option<String>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobRequest {
    pub schema_version: u32,
    pub job_id: String,
    pub media: MediaInfo,
    pub output_dir: PathBuf,
    pub runtime: RuntimePaths,
    pub kind: JobKind,
}
impl JobRequest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == SCHEMA_VERSION,
            "Unsupported desktop job schema"
        );
        ensure!(
            !self.job_id.is_empty()
                && self
                    .job_id
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "Invalid job identifier"
        );
        ensure!(self.media.path.is_file(), "Source video is missing");
        self.media.validate()?;
        self.media.verify_source_unchanged()?;
        match &self.kind {
            JobKind::DetectScenes {
                threshold,
                min_scene_frames,
            } => {
                ensure!(
                    threshold.is_finite() && (0.0..=1.0).contains(threshold),
                    "Scene threshold must be between 0 and 1"
                );
                ensure!(
                    *min_scene_frames > 0,
                    "Minimum scene length must be positive"
                );
            }
            JobKind::CorridorKey {
                range,
                hints,
                settings,
                ..
            } => {
                settings.validate()?;
                range.validate(self.media.frame_count)?;
                match hints {
                    HintInput::None => {}
                    HintInput::FirstFrame {
                        mask_file,
                        frame,
                        quality,
                    } => {
                        ensure!(
                            *frame == range.start && mask_file.is_file(),
                            "First-frame hint must belong to the first selected render frame and its file must exist"
                        );
                        validate_quality(*quality)?;
                    }
                    HintInput::Video {
                        mask_directory,
                        quality,
                        ..
                    } => {
                        ensure!(mask_directory.is_dir(), "Video hint folder is missing");
                        validate_quality(*quality)?;
                    }
                }
            }
            JobKind::SamVideo {
                range,
                prompts,
                model,
                boxes,
                text,
                ..
            } => {
                let has_text = text.as_deref().is_some_and(|t| !t.trim().is_empty())
                    || prompts.iter().any(|p| p.text.is_some());
                ensure!(
                    *model == SamModel::Sam31 || (boxes.is_empty() && !has_text),
                    "Boxes and text prompts need SAM 3.1"
                );
                range.validate(self.media.frame_count)?;
                for b in boxes {
                    ensure!(
                        range.contains(b.frame)
                            && [b.x0, b.y0, b.x1, b.y1].iter().all(|v| v.is_finite())
                            && (b.x1 - b.x0).abs() >= 1.0
                            && (b.y1 - b.y0).abs() >= 1.0,
                        "SAM box is outside the selected range or empty"
                    );
                }
                // SAM 3.1 tracks both ways from the earliest prompted frame; SAM 2.1 only
                // tracks forward from the first selected frame.
                let starts = if *model == SamModel::Sam31 {
                    prompts.iter().any(|p| p.positive)
                        || !boxes.is_empty()
                        || text.as_deref().is_some_and(|t| !t.trim().is_empty())
                } else {
                    prompts.iter().any(|p| p.frame == range.start && p.positive)
                };
                ensure!(
                    starts,
                    if *model == SamModel::Sam31 {
                        "SAM 3.1 needs a subject point, a box or a text prompt in the selected range"
                    } else {
                        "SAM 2.1 needs a subject point on the first selected frame"
                    }
                );
                for point in prompts {
                    ensure!(
                        range.contains(point.frame)
                            && point.x.is_finite()
                            && point.y.is_finite()
                            && point.x >= 0.0
                            && point.y >= 0.0
                            && point.x < self.media.width as f32
                            && point.y < self.media.height as f32,
                        "SAM prompt is outside the source frame or selected range"
                    );
                }
            }
        }
        Ok(())
    }
}
fn validate_quality(quality: f32) -> Result<()> {
    ensure!(
        quality.is_finite() && (0.0..=1.0).contains(&quality),
        "Hint quality must be between 0 and 1"
    );
    Ok(())
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct JobArtifacts {
    pub frames_dir: Option<PathBuf>,
    pub alpha_dir: Option<PathBuf>,
    pub masks_dir: Option<PathBuf>,
    pub checker_dir: Option<PathBuf>,
    #[serde(default)]
    pub scenes: Vec<SceneRange>,
    pub report: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum JobEvent {
    Started {
        job_id: String,
    },
    Progress {
        job_id: String,
        phase: String,
        current: u64,
        total: u64,
        message: String,
    },
    FrameReady {
        job_id: String,
        frame: u64,
        rgba_path: Option<PathBuf>,
        alpha_path: Option<PathBuf>,
        mask_path: Option<PathBuf>,
        checker_path: Option<PathBuf>,
    },
    Completed {
        job_id: String,
        artifacts: JobArtifacts,
    },
    Failed {
        job_id: String,
        message: String,
    },
    Cancelled {
        job_id: String,
    },
}
impl JobEvent {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::Failed { .. } | Self::Cancelled { .. }
        )
    }
    pub fn job_id(&self) -> &str {
        match self {
            Self::Started { job_id }
            | Self::Progress { job_id, .. }
            | Self::FrameReady { job_id, .. }
            | Self::Completed { job_id, .. }
            | Self::Failed { job_id, .. }
            | Self::Cancelled { job_id } => job_id,
        }
    }
}

pub fn frame_file(directory: &std::path::Path, frame: u64) -> PathBuf {
    directory.join(format!("{frame:06}.png"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_frame_ranges_cover_edit_boundaries_once() {
        let a = FrameRange::new(0, 50).unwrap();
        let b = FrameRange::new(50, 119).unwrap();
        assert!(a.contains(49) && !a.contains(50));
        assert!(b.contains(50) && b.contains(118) && !b.contains(119));
        assert_eq!(a.len() + b.len(), 119);
        assert!(FrameRange::new(50, 50).is_err());
    }
    #[test]
    fn progress_and_completion_protocol_preserve_paths_and_frame_indices() {
        let event = JobEvent::FrameReady {
            job_id: "job-1".into(),
            frame: 50,
            rgba_path: Some(PathBuf::from("frames/000050.png")),
            alpha_path: None,
            mask_path: None,
            checker_path: None,
        };
        let text = serde_json::to_string(&event).unwrap();
        let decoded: JobEvent = serde_json::from_str(&text).unwrap();
        assert!(matches!(decoded, JobEvent::FrameReady { frame: 50, .. }));
        assert_eq!(
            frame_file(std::path::Path::new("frames"), 50),
            PathBuf::from("frames/000050.png")
        );
    }
    #[test]
    fn scene_threshold_keeps_the_original_f64_boundary_after_json_round_trip() {
        let job: JobKind = serde_json::from_str(
            r#"{"operation":"detect_scenes","threshold":0.35,"min_scene_frames":6}"#,
        )
        .unwrap();
        let JobKind::DetectScenes { threshold, .. } = job else {
            panic!("expected scene job")
        };
        assert_eq!(threshold, 0.35_f64);
        assert_ne!(threshold, f64::from(0.35_f32));
    }
    #[test]
    fn sam_job_without_model_is_sam21_and_sam31_round_trips() {
        let legacy: JobKind = serde_json::from_str(
            r#"{"operation":"sam_video","range":{"start":4,"end_exclusive":6},"provider":"directml","prompts":[]}"#,
        )
        .unwrap();
        assert!(matches!(
            legacy,
            JobKind::SamVideo {
                model: SamModel::Sam21,
                ..
            }
        ));
        let job = JobKind::SamVideo {
            range: FrameRange::new(4, 6).unwrap(),
            provider: Provider::DirectMl,
            prompts: vec![SamPrompt {
                frame: 4,
                x: 1.0,
                y: 2.0,
                positive: true,
                layer: 0,
                text: Some("hat".into()),
            }],
            model: SamModel::Sam31,
            boxes: vec![SamBox {
                frame: 5,
                layer: 0,
                x0: 1.0,
                y0: 2.0,
                x1: 30.0,
                y1: 40.0,
            }],
            text: Some("person".into()),
        };
        let text = serde_json::to_string(&job).unwrap();
        assert!(text.contains(r#""model":"sam31""#));
        assert!(text.contains(r#""text":"hat""#) && text.contains(r#""x1":30.0"#));
        // A point without text serializes without the field (older readers stay compatible).
        let plain = serde_json::to_string(&SamPrompt {
            frame: 0,
            x: 0.0,
            y: 0.0,
            positive: true,
            layer: 0,
            text: None,
        })
        .unwrap();
        assert!(!plain.contains("text"));
        let decoded: JobKind = serde_json::from_str(&text).unwrap();
        assert!(matches!(
            decoded,
            JobKind::SamVideo {
                model: SamModel::Sam31,
                ..
            }
        ));
    }
    #[test]
    fn legacy_corridorkey_request_defaults_to_original_automatic_settings() {
        let job: JobKind = serde_json::from_str(
            r#"{"operation":"corridor_key","range":{"start":4,"end_exclusive":6},"provider":"directml","hints":{"mode":"none"}}"#,
        )
        .unwrap();
        let JobKind::CorridorKey { settings, .. } = job else {
            panic!("expected CorridorKey job")
        };
        assert_eq!(settings.window_mode, CkWindowMode::Balanced);
        assert!(!settings.low_vram);
        assert_eq!((settings.hann_chunk, settings.hann_stride), (0, 0));
    }
    #[test]
    fn custom_validation_preserves_auto_and_ignores_inactive_custom_values() {
        let mut settings: CkSettings =
            serde_json::from_str(r#"{"window_mode":"custom","hann_chunk":4,"hann_stride":5}"#)
                .unwrap();
        assert!(settings.validate().is_err());
        settings.window_mode = CkWindowMode::Balanced;
        assert!(settings.validate().is_ok());
        settings.window_mode = CkWindowMode::Wide;
        assert!(settings.validate().is_ok());
        settings.window_mode = CkWindowMode::Custom;
        settings.hann_chunk = 0;
        assert!(settings.validate().is_ok());
        settings.hann_chunk = 4;
        settings.hann_stride = 0;
        assert!(settings.validate().is_ok());
        let legacy_auto: CkSettings = serde_json::from_str(r#"{"window_mode":"auto"}"#).unwrap();
        assert_eq!(legacy_auto.window_mode, CkWindowMode::Wide);
    }
    #[test]
    fn one_frame_corridorkey_job_reaches_worker_temporal_preflight() {
        let path = std::env::temp_dir().join(format!(
            "brp-one-frame-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"source identity fixture").unwrap();
        let request = JobRequest {
            schema_version: SCHEMA_VERSION,
            job_id: "one-frame".into(),
            media: MediaInfo {
                path: path.clone(),
                width: 128,
                height: 128,
                frame_count: 8,
                fps: Rational { num: 24, den: 1 },
                time_base: Rational { num: 1, den: 24 },
                first_pts: 0,
                has_b_frames: false,
                source_identity: Some(SourceIdentity::read(&path).unwrap()),
            },
            output_dir: PathBuf::from("renders"),
            runtime: RuntimePaths::default(),
            kind: JobKind::CorridorKey {
                range: FrameRange::new(4, 5).unwrap(),
                provider: Provider::DirectMl,
                hints: HintInput::None,
                settings: CkSettings::default(),
            },
        };
        let result = request.validate();
        std::fs::remove_file(path).unwrap();
        result.unwrap();
    }

    #[test]
    fn sam31_starts_from_a_prompt_anywhere_in_the_range_but_sam21_needs_the_first_frame() {
        let path = std::env::temp_dir().join(format!(
            "brp-sam-start-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"source identity fixture").unwrap();
        let request = |model, prompts: Vec<SamPrompt>, boxes: Vec<SamBox>| JobRequest {
            schema_version: SCHEMA_VERSION,
            job_id: "sam-start".into(),
            media: MediaInfo {
                path: path.clone(),
                width: 128,
                height: 128,
                frame_count: 16,
                fps: Rational { num: 24, den: 1 },
                time_base: Rational { num: 1, den: 24 },
                first_pts: 0,
                has_b_frames: false,
                source_identity: Some(SourceIdentity::read(&path).unwrap()),
            },
            output_dir: PathBuf::from("renders"),
            runtime: RuntimePaths::default(),
            kind: JobKind::SamVideo {
                range: FrameRange::new(2, 12).unwrap(),
                provider: Provider::DirectMl,
                prompts,
                model,
                boxes,
                text: None,
            },
        };
        let point = |frame, positive| SamPrompt {
            frame,
            x: 10.0,
            y: 10.0,
            positive,
            layer: 0,
            text: None,
        };
        let mid_box = SamBox {
            frame: 7,
            layer: 0,
            x0: 1.0,
            y0: 1.0,
            x1: 40.0,
            y1: 40.0,
        };
        let results = [
            request(SamModel::Sam31, vec![point(7, true)], vec![]).validate(),
            request(SamModel::Sam31, vec![], vec![mid_box]).validate(),
            request(SamModel::Sam31, vec![point(7, false)], vec![]).validate(),
            request(SamModel::Sam21, vec![point(7, true)], vec![]).validate(),
            request(SamModel::Sam21, vec![point(2, true)], vec![]).validate(),
        ];
        std::fs::remove_file(&path).unwrap();
        let ok: Vec<bool> = results.iter().map(Result::is_ok).collect();
        assert_eq!(ok, [true, true, false, false, true]);
    }
}
