//! CPU-only media preparation, called from a background thread after an explicit
//! render/export action. Importing footage never creates calculated media.
use crate::{preview::ViewMode, project_store::HintFootage};
use anyhow::{Context, Result, ensure};
use ck_desktop_core::{
    DespillSettings, FrameRange, HintInput, MediaInfo, Rational, RuntimePaths, SourceIdentity,
    frame_file,
};
use std::{
    fs::{self, File, OpenOptions},
    io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

const HINT_QUALITY: f32 = 0.95;
static NONCE: AtomicU64 = AtomicU64::new(0);

fn unique_name(prefix: &str) -> String {
    format!(
        "{prefix}-{}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        NONCE.fetch_add(1, Ordering::Relaxed),
    )
}

fn hidden(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    #[cfg(not(windows))]
    let _ = command;
}

fn run_ffmpeg(command: &mut Command) -> Result<()> {
    hidden(command);
    let output = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .context("Starting CPU hint decoder")?;
    ensure!(
        output.status.success(),
        "Preparing external hints: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

fn ffprobe_beside(ffmpeg: &Path) -> Result<PathBuf> {
    let name = if ffmpeg
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("exe"))
    {
        "ffprobe.exe"
    } else {
        "ffprobe"
    };
    let ffprobe = ffmpeg.with_file_name(name);
    ensure!(
        ffprobe.is_file(),
        "FFprobe must be installed beside FFmpeg: {}",
        ffprobe.display()
    );
    Ok(ffprobe)
}

fn is_still_image(path: &Path) -> bool {
    path.extension().is_some_and(|extension| {
        ["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp", "exr"]
            .iter()
            .any(|candidate| extension.eq_ignore_ascii_case(candidate))
    })
}

fn same_rate(a: Rational, b: Rational) -> bool {
    u128::from(a.num) * u128::from(b.den) == u128::from(b.num) * u128::from(a.den)
}

fn validate_video_hint(hint: &MediaInfo, source: &MediaInfo, range: FrameRange) -> Result<()> {
    hint.validate()?;
    ensure!(
        same_rate(hint.fps, source.fps),
        "Hint video rate {}/{} differs from source {}/{}; re-export it at the source frame rate to preserve exact frame alignment",
        hint.fps.num,
        hint.fps.den,
        source.fps.num,
        source.fps.den
    );
    range.validate(source.frame_count)?;
    Ok(())
}

/// v1 maps each source frame n to min(n, hint_count - 1), repeating the
/// final hint when the mask movie is shorter. There is no FPS retiming.
fn hint_decode_range(hint_frames: u64, source_range: FrameRange) -> FrameRange {
    let last = hint_frames.saturating_sub(1);
    FrameRange {
        start: source_range.start.min(last),
        end_exclusive: source_range.end_exclusive.min(hint_frames),
    }
}

/// Owns only a freshly created, flat directory. Failure cleanup never traverses
/// a project tree or touches a previous hint generation.
struct HintStaging {
    path: PathBuf,
    root: PathBuf,
    published: bool,
}
impl HintStaging {
    fn new(root: &Path) -> Result<Self> {
        ensure!(
            root.is_absolute(),
            "Calculated-media root must be an absolute project or temporary path"
        );
        let app = RuntimePaths::application_directory();
        ensure!(
            !root.starts_with(&app),
            "Calculated hint media belongs beside the project or in its temporary workspace, not the application folder"
        );
        fs::create_dir_all(root).context("Creating calculated-media directory")?;
        let root = fs::canonicalize(root)?;
        if let Ok(app) = fs::canonicalize(app) {
            ensure!(
                !root.starts_with(app),
                "Calculated-media directory resolves inside the application folder"
            );
        }
        let path = root.join(unique_name(".hint-preparing"));
        fs::create_dir(&path)?;
        Ok(Self {
            path,
            root,
            published: false,
        })
    }
    fn publish(&mut self) -> Result<PathBuf> {
        let complete = self.root.join(unique_name("external-hint"));
        fs::rename(&self.path, &complete).context("Publishing prepared hint masks")?;
        self.published = true;
        Ok(complete)
    }
}
impl Drop for HintStaging {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        // FFmpeg image2 writes flat files only. Never recursively follow an
        // unexpected directory/link in a user-controlled project location.
        if let Ok(entries) = fs::read_dir(&self.path) {
            for entry in entries.flatten() {
                if entry.file_type().is_ok_and(|kind| kind.is_file()) {
                    let _ = fs::remove_file(entry.path());
                }
            }
        }
        let _ = fs::remove_dir(&self.path);
    }
}

fn check_gray_mask(path: &Path, source: &MediaInfo) -> Result<()> {
    let image =
        image::open(path).with_context(|| format!("Reading prepared hint {}", path.display()))?;
    ensure!(
        image.width() == source.width && image.height() == source.height,
        "Prepared hint dimensions do not match source footage"
    );
    ensure!(
        image.color() == image::ColorType::L8,
        "Prepared hint must be an 8-bit luminance mask"
    );
    Ok(())
}

/// Builds source-sized luminance PNGs, matching v1's gray-channel/bilinear
/// contract. Video indices are absolute: mask 000123.png means source frame 123.
/// Short hint movies repeat their last frame, matching the original workflow.
/// The caller supplies the project's calculated-media root (or owned Temp root)
/// and runs this blocking CPU helper on its render-preparation thread.
pub fn prepare_external_hint(
    hint: &HintFootage,
    media: &MediaInfo,
    range: FrameRange,
    root: &Path,
    ffmpeg: &Path,
) -> Result<HintInput> {
    media.validate()?;
    range.validate(media.frame_count)?;
    media.verify_source_unchanged()?;
    ensure!(ffmpeg.is_file(), "FFmpeg is unavailable");
    let hint_identity =
        SourceIdentity::read(&hint.path).context("Opening external hint footage")?;
    let first_only = hint.first_frame_only || is_still_image(&hint.path);
    let hint_media = if first_only {
        None
    } else {
        let runtime = RuntimePaths {
            ffmpeg: ffmpeg.to_path_buf(),
            ffprobe: ffprobe_beside(ffmpeg)?,
            ..RuntimePaths::default()
        };
        let probed = MediaInfo::probe(&hint.path, &runtime)
            .context("Checking external hint video timing")?;
        validate_video_hint(&probed, media, range)?;
        Some(probed)
    };
    let mut staging = HintStaging::new(root)?;
    let mut command = Command::new(ffmpeg);
    command
        .args([
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-hwaccel",
            "none",
            "-threads",
            "2",
            "-noautorotate",
            "-i",
        ])
        .arg(&hint.path)
        .args(["-map", "0:v:0", "-an", "-sn", "-dn"]);
    // Convert to gray before resizing, as v1's GrayMaskVideo + resize_mask do.
    // A transparent RGB hint is interpreted by luminance, not by its alpha.
    let decoded_range = hint_media
        .as_ref()
        .map(|probed| hint_decode_range(probed.frame_count, range));
    let filter = if let Some(decoded_range) = decoded_range {
        format!(
            "trim=start_frame={}:end_frame={},format=gray,scale={}:{}:flags=bilinear",
            decoded_range.start, decoded_range.end_exclusive, media.width, media.height
        )
    } else {
        format!(
            "format=gray,scale={}:{}:flags=bilinear",
            media.width, media.height
        )
    };
    let count = if first_only { 1 } else { range.len() };
    let decoded_count = decoded_range.map_or(1, FrameRange::len);
    command
        .args([
            "-vf",
            &filter,
            "-frames:v",
            &decoded_count.to_string(),
            "-fps_mode",
            "passthrough",
            "-c:v",
            "png",
            "-pix_fmt",
            "gray",
            "-threads:v",
            "1",
            "-start_number",
            "0",
            "-f",
            "image2",
        ])
        .arg(staging.path.join("hint_%06d.png"));
    run_ffmpeg(&mut command)?;
    for local in 0..decoded_count {
        let decoded = staging.path.join(format!("hint_{local:06}.png"));
        check_gray_mask(&decoded, media)?;
        fs::rename(&decoded, frame_file(&staging.path, range.start + local))?;
    }
    let last_mask = frame_file(&staging.path, range.start + decoded_count - 1);
    for local in decoded_count..count {
        let destination = frame_file(&staging.path, range.start + local);
        // A hard link avoids duplicate mask bytes where supported. PNG copies
        // are the portable fallback, bounded to the explicitly requested range.
        if fs::hard_link(&last_mask, &destination).is_err() {
            fs::copy(&last_mask, destination)?;
        }
    }
    ensure!(
        u64::try_from(fs::read_dir(&staging.path)?.count())? == count,
        "External hint decoder returned an unexpected frame count"
    );
    ensure!(
        SourceIdentity::read(&hint.path)? == hint_identity,
        "External hint footage changed during preparation"
    );
    if let Some(probed) = hint_media {
        probed.verify_source_unchanged()?;
    }
    media.verify_source_unchanged()?;
    let complete = staging.publish()?;
    Ok(if first_only {
        HintInput::FirstFrame {
            mask_file: frame_file(&complete, range.start),
            frame: range.start,
            quality: HINT_QUALITY,
        }
    } else {
        HintInput::Video {
            mask_directory: complete,
            frame_offset: 0,
            quality: HINT_QUALITY,
        }
    })
}

struct FrameStaging {
    path: PathBuf,
    published: bool,
}
impl FrameStaging {
    fn new(destination: &Path) -> Result<Self> {
        let parent = destination
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        ensure!(
            parent.is_dir(),
            "Export destination directory does not exist"
        );
        let path = parent.join(format!("{}.png", unique_name(".chroma-frame")));
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self {
            path,
            published: false,
        })
    }
    fn publish(&mut self, destination: &Path) -> Result<()> {
        OpenOptions::new()
            .write(true)
            .open(&self.path)?
            .sync_all()?;
        // Same-directory rename publishes only the completed PNG. The caller's
        // native Save dialog supplies consent when destination already exists.
        fs::rename(&self.path, destination).context("Publishing exported frame")?;
        self.published = true;
        Ok(())
    }
}
impl Drop for FrameStaging {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn ensure_different_file(destination: &Path, input: &Path) -> Result<()> {
    if destination.exists() && input.exists() {
        ensure!(
            fs::canonicalize(destination)? != fs::canonicalize(input)?,
            "Export destination is an input file"
        );
    }
    Ok(())
}

/// Result compositing inputs for one exported frame.
pub struct ExportComposition<'a> {
    /// Result despill amount (0 = off).
    pub settings: DespillSettings,
    /// Garbage/despill masks rasterised for this frame.
    pub masks: Option<&'a crate::mask_composite::MaskApplication>,
}

/// Saves a genuine full-size frame independently of preview readiness. Rendered
/// PNGs are copied unchanged; Source/Result fallback uses the verified CPU seek
/// decoder and preserves original RGBA. Alpha/SAM require an actual mask.
pub fn export_single_frame(
    media: &MediaInfo,
    frame: u64,
    viewmode: ViewMode,
    rendered_artifact: Option<PathBuf>,
    destination: &Path,
    ffmpeg: &Path,
) -> Result<()> {
    export_single_frame_impl(
        media,
        frame,
        viewmode,
        rendered_artifact,
        destination,
        ffmpeg,
        ExportComposition {
            settings: DespillSettings::default(),
            masks: None,
        },
    )
}

/// Applies the same result-only colour unmix as the viewer before PNG encoding.
/// Original-footage fallback, alpha/SAM views and the zero setting stay unchanged.
#[cfg(test)]
pub fn export_single_frame_with_despill(
    media: &MediaInfo,
    frame: u64,
    viewmode: ViewMode,
    rendered_artifact: Option<PathBuf>,
    destination: &Path,
    ffmpeg: &Path,
    settings: DespillSettings,
) -> Result<()> {
    export_single_frame_masked(
        media,
        frame,
        viewmode,
        rendered_artifact,
        destination,
        ffmpeg,
        ExportComposition {
            settings,
            masks: None,
        },
    )
}

/// Single-frame export with the frame's garbage/despill masks applied.
pub fn export_single_frame_masked(
    media: &MediaInfo,
    frame: u64,
    viewmode: ViewMode,
    rendered_artifact: Option<PathBuf>,
    destination: &Path,
    ffmpeg: &Path,
    composition: ExportComposition<'_>,
) -> Result<()> {
    let ExportComposition { settings, masks } = composition;
    settings.validate()?;
    let masked =
        masks.is_some_and(|masks| !masks.garbage.is_empty() || masks.despill_weight.is_some());
    if (!settings.is_enabled() && !masked)
        || viewmode != ViewMode::Composite
        || rendered_artifact.is_none()
    {
        return export_single_frame(
            media,
            frame,
            viewmode,
            rendered_artifact,
            destination,
            ffmpeg,
        );
    }
    export_single_frame_impl(
        media,
        frame,
        viewmode,
        rendered_artifact,
        destination,
        ffmpeg,
        ExportComposition { settings, masks },
    )
}

fn export_single_frame_impl(
    media: &MediaInfo,
    frame: u64,
    viewmode: ViewMode,
    rendered_artifact: Option<PathBuf>,
    destination: &Path,
    ffmpeg: &Path,
    composition: ExportComposition<'_>,
) -> Result<()> {
    let ExportComposition { settings, masks } = composition;
    media.validate()?;
    ensure!(
        frame < media.frame_count,
        "Export frame is outside source footage"
    );
    let masked =
        masks.is_some_and(|masks| !masks.garbage.is_empty() || masks.despill_weight.is_some());
    let needs_unmix =
        settings.is_enabled() && rendered_artifact.is_some() && viewmode == ViewMode::Composite;
    let needs_source = rendered_artifact.is_none() || needs_unmix;
    if needs_source {
        media.verify_source_unchanged()?;
    }
    ensure!(
        destination
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("png")),
        "Single-frame export destination must be a PNG file"
    );
    ensure_different_file(destination, &media.path)?;
    if let Some(artifact) = &rendered_artifact {
        ensure_different_file(destination, artifact)?;
        ensure!(
            image::ImageReader::open(artifact)?
                .with_guessed_format()?
                .format()
                == Some(image::ImageFormat::Png),
            "Rendered frame artifact is not a PNG"
        );
        ensure!(
            image::image_dimensions(artifact)? == (media.width, media.height),
            "Rendered frame artifact dimensions differ from source footage"
        );
    } else {
        ensure!(
            matches!(viewmode, ViewMode::Source | ViewMode::Composite),
            "{} is not available for this frame yet",
            viewmode.label()
        );
        ensure!(ffmpeg.is_file(), "FFmpeg is unavailable");
    }
    let mut staging = FrameStaging::new(destination)?;
    if let Some(artifact) = rendered_artifact {
        let identity = SourceIdentity::read(&artifact)?;
        if needs_unmix || masked {
            let plate = if needs_unmix {
                crate::preview::read_source_rgba(media, frame, ffmpeg)
                    .context("Result despill requires the original footage")?
            } else {
                std::sync::Arc::from(Vec::<u8>::new())
            };
            let mut image = image::ImageReader::open(&artifact)?.decode()?.to_rgba8();
            if let Some(masks) = masks {
                crate::mask_composite::apply_garbage_masks(&mut image, masks);
            }
            let weight = masks
                .and_then(|masks| masks.despill_weight.as_ref())
                .map(|gray| crate::despill::MaskWeight::from_gray(gray));
            crate::despill::apply_rgba_weighted(
                &plate,
                image.as_mut(),
                media.width,
                media.height,
                settings,
                weight.as_ref(),
            )?;
            image::save_buffer_with_format(
                &staging.path,
                image.as_raw(),
                media.width,
                media.height,
                image::ColorType::Rgba8,
                image::ImageFormat::Png,
            )?;
        } else {
            let mut source = File::open(&artifact)?;
            let mut output = OpenOptions::new()
                .write(true)
                .truncate(true)
                .open(&staging.path)?;
            io::copy(&mut source, &mut output).context("Copying rendered frame")?;
            output.sync_all()?;
        }
        ensure!(
            SourceIdentity::read(&artifact)? == identity,
            "Rendered frame changed during export"
        );
    } else {
        let rgba = crate::preview::read_source_rgba(media, frame, ffmpeg)?;
        image::save_buffer_with_format(
            &staging.path,
            &rgba,
            media.width,
            media.height,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )
        .context("Encoding original source frame")?;
    }
    if needs_source {
        media.verify_source_unchanged()?;
    }
    staging.publish(destination)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRoot(PathBuf);
    impl TestRoot {
        fn new() -> Self {
            let root = std::env::temp_dir().join(unique_name("brp-media-task-test"));
            fs::create_dir(&root).unwrap();
            Self(root)
        }
    }
    impl Drop for TestRoot {
        fn drop(&mut self) {
            // Every entry is generated by this fixture under a fresh Temp root.
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn media(root: &Path) -> MediaInfo {
        let path = root.join("source.mov");
        fs::write(&path, b"fixture-source-identity").unwrap();
        MediaInfo {
            path: path.clone(),
            width: 2,
            height: 1,
            frame_count: 2,
            fps: Rational { num: 24, den: 1 },
            time_base: Rational { num: 1, den: 24 },
            first_pts: 0,
            has_b_frames: false,
            source_identity: Some(SourceIdentity::read(path).unwrap()),
        }
    }

    #[test]
    fn equivalent_rates_are_accepted_but_hint_frame_drift_is_rejected() {
        let root = TestRoot::new();
        let source = media(&root.0);
        let mut hint = source.clone();
        hint.fps = Rational { num: 48, den: 2 };
        let range = FrameRange {
            start: 1,
            end_exclusive: 2,
        };
        assert!(validate_video_hint(&hint, &source, range).is_ok());
        hint.fps = Rational {
            num: 24000,
            den: 1001,
        };
        assert!(validate_video_hint(&hint, &source, range).is_err());
        hint.fps = source.fps;
        hint.frame_count = 1;
        assert!(validate_video_hint(&hint, &source, range).is_ok());
    }

    #[test]
    fn short_hint_video_clamps_absolute_mapping_to_its_last_frame_like_v1() {
        assert_eq!(
            hint_decode_range(
                3,
                FrameRange {
                    start: 1,
                    end_exclusive: 7
                }
            ),
            FrameRange {
                start: 1,
                end_exclusive: 3
            }
        );
        assert_eq!(
            hint_decode_range(
                3,
                FrameRange {
                    start: 8,
                    end_exclusive: 11
                }
            ),
            FrameRange {
                start: 2,
                end_exclusive: 3
            }
        );
        assert_eq!(
            hint_decode_range(
                100,
                FrameRange {
                    start: 8,
                    end_exclusive: 11
                }
            ),
            FrameRange {
                start: 8,
                end_exclusive: 11
            }
        );
    }

    #[test]
    fn artifact_export_preserves_full_rgba_and_replaces_only_complete_destination() {
        let root = TestRoot::new();
        let source = media(&root.0);
        let artifact = root.0.join("rendered.png");
        let rgba = [12, 24, 48, 0, 100, 150, 200, 173];
        image::save_buffer(&artifact, &rgba, 2, 1, image::ColorType::Rgba8).unwrap();
        let destination = root.0.join("saved.png");
        fs::write(&destination, b"previous-save").unwrap();
        export_single_frame(
            &source,
            1,
            ViewMode::Composite,
            Some(artifact.clone()),
            &destination,
            Path::new("unused-ffmpeg"),
        )
        .unwrap();
        assert_eq!(
            fs::read(&destination).unwrap(),
            fs::read(&artifact).unwrap()
        );
        assert_eq!(
            image::open(destination).unwrap().into_rgba8().as_raw(),
            &rgba
        );
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 3);
    }

    #[test]
    fn unavailable_or_wrong_size_artifact_never_replaces_user_file() {
        let root = TestRoot::new();
        let source = media(&root.0);
        let destination = root.0.join("saved.png");
        fs::write(&destination, b"previous-save").unwrap();
        let artifact = root.0.join("wrong-size.png");
        image::save_buffer(&artifact, &[255], 1, 1, image::ColorType::L8).unwrap();
        assert!(
            export_single_frame(
                &source,
                0,
                ViewMode::Alpha,
                Some(artifact),
                &destination,
                Path::new("unused-ffmpeg")
            )
            .is_err()
        );
        assert!(
            export_single_frame(
                &source,
                0,
                ViewMode::Sam,
                None,
                &destination,
                Path::new("unused-ffmpeg")
            )
            .is_err()
        );
        assert_eq!(fs::read(destination).unwrap(), b"previous-save");
        assert_eq!(fs::read_dir(&root.0).unwrap().count(), 3);
    }

    #[test]
    fn rendered_artifact_exports_when_original_source_is_offline_or_not_materialized() {
        let root = TestRoot::new();
        let source = media(&root.0);
        let artifact = root.0.join("contained-render.png");
        let rgba = [12, 34, 56, 0, 78, 90, 123, 143];
        image::save_buffer(&artifact, &rgba, 2, 1, image::ColorType::Rgba8).unwrap();
        fs::remove_file(&source.path).unwrap();
        let destination = root.0.join("saved.png");
        fs::write(&destination, b"previous-save").unwrap();
        export_single_frame(
            &source,
            1,
            ViewMode::Composite,
            Some(artifact.clone()),
            &destination,
            Path::new("unused-ffmpeg"),
        )
        .unwrap();
        assert_eq!(fs::read(&destination).unwrap(), fs::read(artifact).unwrap());
        assert_eq!(
            image::open(destination).unwrap().into_rgba8().as_raw(),
            &rgba
        );
        assert!(
            !source.path.exists(),
            "rendered export must not extract or require original source"
        );
    }

    #[test]
    fn zero_despill_exports_offline_exactly_and_enabled_unmix_preserves_existing_file_on_error() {
        let root = TestRoot::new();
        let source = media(&root.0);
        let artifact = root.0.join("rgba.png");
        image::save_buffer(
            &artifact,
            &[10, 20, 30, 17, 40, 50, 60, 128],
            2,
            1,
            image::ColorType::Rgba8,
        )
        .unwrap();
        fs::remove_file(&source.path).unwrap();
        let destination = root.0.join("saved.png");
        export_single_frame_with_despill(
            &source,
            0,
            ViewMode::Composite,
            Some(artifact.clone()),
            &destination,
            Path::new("unused"),
            DespillSettings::default(),
        )
        .unwrap();
        let expected = fs::read(&artifact).unwrap();
        assert_eq!(fs::read(&destination).unwrap(), expected);
        assert!(
            export_single_frame_with_despill(
                &source,
                0,
                ViewMode::Composite,
                Some(artifact.clone()),
                &destination,
                Path::new("unused"),
                DespillSettings { amount: 0.5 }
            )
            .is_err()
        );
        assert_eq!(fs::read(&destination).unwrap(), expected);
        export_single_frame_with_despill(
            &source,
            0,
            ViewMode::Alpha,
            Some(artifact),
            &destination,
            Path::new("unused"),
            DespillSettings { amount: 1.0 },
        )
        .unwrap();
        assert_eq!(fs::read(&destination).unwrap(), expected);
    }

    #[test]
    fn hint_failure_cleanup_preserves_other_generations() {
        let root = TestRoot::new();
        let previous = root.0.join("existing-mask.png");
        fs::write(&previous, b"preserve").unwrap();
        let preparing;
        {
            let staging = HintStaging::new(&root.0).unwrap();
            preparing = staging.path.clone();
            fs::write(staging.path.join("hint_000000.png"), b"incomplete").unwrap();
        }
        assert!(!preparing.exists());
        assert_eq!(fs::read(previous).unwrap(), b"preserve");
    }

    #[test]
    fn source_input_cannot_be_replaced_by_frame_export() {
        let root = TestRoot::new();
        let mut source = media(&root.0);
        let png_source = root.0.join("source.png");
        fs::rename(&source.path, &png_source).unwrap();
        source.path = png_source;
        assert!(
            export_single_frame(
                &source,
                0,
                ViewMode::Source,
                None,
                &source.path,
                Path::new("unused-ffmpeg")
            )
            .is_err()
        );
        assert_eq!(fs::read(&source.path).unwrap(), b"fixture-source-identity");
    }

    #[test]
    #[ignore = "Explicit CPU-only FFmpeg fixture; generates two tiny frames, no models or GPU"]
    fn ffmpeg_two_frame_hint_mapping_luminance_and_exact_original_export() {
        let root = TestRoot::new();
        let runtime =
            RuntimePaths::installed(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../app"));
        assert!(runtime.ffmpeg.is_file() && runtime.ffprobe.is_file());
        let first = [255, 255, 255, 0, 255, 255, 255, 0];
        let second = [0, 0, 0, 255, 0, 0, 0, 255];
        for (index, pixels) in [first, second].into_iter().enumerate() {
            image::save_buffer(
                root.0.join(format!("input_{index:06}.png")),
                &pixels,
                2,
                1,
                image::ColorType::Rgba8,
            )
            .unwrap();
        }
        let clip = root.0.join("two-frames.mkv");
        let mut encode = Command::new(&runtime.ffmpeg);
        encode
            .args([
                "-hide_banner",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-framerate",
                "25",
                "-start_number",
                "0",
                "-i",
            ])
            .arg(root.0.join("input_%06d.png"))
            .args([
                "-frames:v",
                "2",
                "-c:v",
                "ffv1",
                "-pix_fmt",
                "bgra",
                "-threads:v",
                "1",
            ])
            .arg(&clip);
        run_ffmpeg(&mut encode).unwrap();
        let source = MediaInfo::probe(&clip, &runtime).unwrap();
        let range = FrameRange {
            start: 1,
            end_exclusive: 2,
        };
        let masks = root.0.join("calculated");
        let result = prepare_external_hint(
            &HintFootage {
                path: clip,
                first_frame_only: false,
            },
            &source,
            range,
            &masks,
            &runtime.ffmpeg,
        )
        .unwrap();
        let HintInput::Video {
            mask_directory,
            frame_offset,
            ..
        } = result
        else {
            panic!("expected video hints");
        };
        assert_eq!(frame_offset, 0);
        assert!(!frame_file(&mask_directory, 0).exists());
        assert_eq!(
            image::open(frame_file(&mask_directory, 1))
                .unwrap()
                .into_luma8()
                .as_raw(),
            &[0, 0]
        );
        // A one-frame mask movie must repeat its last RGB-luminance mask at
        // source frame 1, even though that is beyond the movie's last frame.
        let short_clip = root.0.join("short-hint.mkv");
        let mut encode_short = Command::new(&runtime.ffmpeg);
        encode_short
            .args([
                "-hide_banner",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-framerate",
                "25",
                "-i",
            ])
            .arg(root.0.join("input_000000.png"))
            .args([
                "-frames:v",
                "1",
                "-c:v",
                "ffv1",
                "-pix_fmt",
                "bgra",
                "-threads:v",
                "1",
            ])
            .arg(&short_clip);
        run_ffmpeg(&mut encode_short).unwrap();
        let short = prepare_external_hint(
            &HintFootage {
                path: short_clip,
                first_frame_only: false,
            },
            &source,
            FrameRange {
                start: 0,
                end_exclusive: 2,
            },
            &masks,
            &runtime.ffmpeg,
        )
        .unwrap();
        let HintInput::Video {
            mask_directory: short_masks,
            ..
        } = short
        else {
            panic!("expected video hints");
        };
        for frame in 0..2 {
            assert_eq!(
                image::open(frame_file(&short_masks, frame))
                    .unwrap()
                    .into_luma8()
                    .as_raw(),
                &[255, 255]
            );
        }
        let image_hint = root.0.join("one-pixel-hint.png");
        image::save_buffer(
            &image_hint,
            &[255, 255, 255, 0],
            1,
            1,
            image::ColorType::Rgba8,
        )
        .unwrap();
        let still = prepare_external_hint(
            &HintFootage {
                path: image_hint,
                first_frame_only: true,
            },
            &source,
            range,
            &masks,
            &runtime.ffmpeg,
        )
        .unwrap();
        let HintInput::FirstFrame {
            mask_file, frame, ..
        } = still
        else {
            panic!("expected first-frame hint");
        };
        assert_eq!(frame, 1);
        assert_eq!(
            image::open(mask_file).unwrap().into_luma8().as_raw(),
            &[255, 255]
        );
        let exported = root.0.join("original.png");
        export_single_frame(
            &source,
            1,
            ViewMode::Source,
            None,
            &exported,
            &runtime.ffmpeg,
        )
        .unwrap();
        assert_eq!(
            image::open(exported).unwrap().into_rgba8().as_raw(),
            &second
        );
    }

    #[test]
    fn garbage_mask_changes_exported_alpha_but_not_rgb() {
        use ck_desktop_core::MaskMode;
        let root = TestRoot::new();
        let source = media(&root.0);
        let artifact = root.0.join("rgba.png");
        image::save_buffer(
            &artifact,
            &[10, 20, 30, 255, 40, 50, 60, 255],
            2,
            1,
            image::ColorType::Rgba8,
        )
        .unwrap();
        let matte = image::GrayImage::from_raw(2, 1, vec![255, 0]).unwrap();
        let masks = crate::mask_composite::MaskApplication {
            garbage: vec![(MaskMode::Subtract, std::sync::Arc::new(matte))],
            despill_weight: None,
        };
        let destination = root.0.join("saved.png");
        export_single_frame_masked(
            &source,
            0,
            ViewMode::Composite,
            Some(artifact),
            &destination,
            Path::new("unused"),
            ExportComposition {
                settings: DespillSettings::default(),
                masks: Some(&masks),
            },
        )
        .unwrap();
        let exported = image::open(destination).unwrap().to_rgba8();
        assert_eq!(exported.get_pixel(0, 0).0, [10, 20, 30, 0]);
        assert_eq!(exported.get_pixel(1, 0).0, [40, 50, 60, 255]);
    }
}
