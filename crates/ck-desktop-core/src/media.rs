use crate::{RuntimePaths, frame_file};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rational {
    pub num: u64,
    pub den: u64,
}
impl Default for Rational {
    fn default() -> Self {
        Self { num: 1, den: 1 }
    }
}
impl Rational {
    pub fn validate(self) -> Result<()> {
        ensure!(
            self.num > 0 && self.den > 0,
            "A positive rational rate is required"
        );
        Ok(())
    }
    pub fn parse(value: &str) -> Result<Self> {
        let (num, den) = value
            .split_once('/')
            .context("Invalid rational video rate")?;
        let rate = Self {
            num: num.parse()?,
            den: den.parse()?,
        };
        rate.validate()?;
        Ok(rate)
    }
    pub fn value(self) -> f64 {
        self.num as f64 / self.den as f64
    }
    pub fn frame_seconds(self, frame: u64) -> f64 {
        frame as f64 * self.den as f64 / self.num as f64
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MediaInfo {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub frame_count: u64,
    pub fps: Rational,
    pub time_base: Rational,
    pub first_pts: i64,
    pub has_b_frames: bool,
    #[serde(default)]
    pub source_identity: Option<SourceIdentity>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceIdentity {
    pub bytes: u64,
    pub modified_unix_ns: u64,
}
impl SourceIdentity {
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let metadata = fs::metadata(path.as_ref()).context("Reading source file identity")?;
        ensure!(metadata.is_file(), "Source path is not a video file");
        let modified_unix_ns =
            u64::try_from(metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos())
                .context("Source modification time is out of range")?;
        Ok(Self {
            bytes: metadata.len(),
            modified_unix_ns,
        })
    }
}

pub(crate) fn hidden(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    #[cfg(not(windows))]
    {
        let _ = command;
    }
}
fn checked_output(command: &mut Command, what: &str) -> Result<Output> {
    hidden(command);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    let tree = crate::worker::process_tree::ProcessTree::new()?;
    let child = command
        .spawn()
        .with_context(|| format!("Starting {what}"))?;
    #[cfg(windows)]
    let mut child = child;
    #[cfg(windows)]
    if let Err(error) = tree.assign(&child)
        && child.try_wait()?.is_none()
    {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }
    let output = child.wait_with_output()?;
    ensure!(
        output.status.success(),
        "{what}: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(output)
}
fn number(value: &serde_json::Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_str().and_then(|s| s.parse().ok()))
}

pub fn validate_cfr_pts(pts: &[i64], fps: Rational, time_base: Rational) -> Result<()> {
    fps.validate()?;
    time_base.validate()?;
    ensure!(!pts.is_empty(), "No video presentation timestamps found");
    for pair in pts.windows(2) {
        let delta = pair[1]
            .checked_sub(pair[0])
            .context("Video timestamp overflow")?;
        let actual = u128::try_from(delta)
            .ok()
            .and_then(|d| d.checked_mul(time_base.num as u128))
            .and_then(|d| d.checked_mul(fps.num as u128));
        let expected = (fps.den as u128).checked_mul(time_base.den as u128);
        ensure!(
            delta > 0 && actual.is_some() && actual == expected,
            "Variable-rate or irregular video timing is not supported yet; use a constant-frame-rate source"
        );
    }
    Ok(())
}

impl MediaInfo {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.width > 0 && self.height > 0 && self.frame_count > 0,
            "Source video has no valid frames"
        );
        self.fps.validate()?;
        self.time_base.validate()?;
        Ok(())
    }
    pub fn verify_source_unchanged(&self) -> Result<()> {
        let expected = self.source_identity.context("This project has no recorded source identity; reopen or relink the source video before previewing or processing")?;
        ensure!(
            SourceIdentity::read(&self.path)? == expected,
            "The source video changed after it was imported; reopen or relink it so scenes, prompts and cached results cannot use stale footage"
        );
        Ok(())
    }
    /// Demux presentation timestamps without approximate seek-to-frame conversion.
    /// Call from a background thread, never the GUI painting thread.
    pub fn probe(path: impl AsRef<Path>, runtime: &RuntimePaths) -> Result<Self> {
        let path = fs::canonicalize(path.as_ref()).context("Opening source video")?;
        let source_identity = SourceIdentity::read(&path)?;
        ensure!(
            runtime.ffprobe.is_file(),
            "Choose the native FFprobe executable in runtime settings"
        );
        let output = checked_output(Command::new(&runtime.ffprobe).args(["-v", "error", "-select_streams", "v:0", "-show_packets", "-show_streams", "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,time_base,nb_frames,has_b_frames:packet=pts", "-of", "json"]).arg(&path), "FFprobe")?;
        let data: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        let stream = data["streams"]
            .as_array()
            .and_then(|a| a.first())
            .context("The file has no video stream")?;
        let fps = Rational::parse(stream["avg_frame_rate"].as_str().unwrap_or("0/0"))
            .or_else(|_| Rational::parse(stream["r_frame_rate"].as_str().unwrap_or("0/0")))?;
        let time_base = Rational::parse(
            stream["time_base"]
                .as_str()
                .context("Video time base is missing")?,
        )?;
        let mut pts = data["packets"]
            .as_array()
            .context("Video packets are missing")?
            .iter()
            .map(|packet| {
                number(&packet["pts"]).context("A video packet has no presentation timestamp")
            })
            .collect::<Result<Vec<_>>>()?;
        pts.sort_unstable();
        validate_cfr_pts(&pts, fps, time_base)?;
        if let Some(declared) = number(&stream["nb_frames"]).filter(|n| *n > 0) {
            ensure!(
                declared as usize == pts.len(),
                "Unsupported video packet/frame mapping"
            );
        }
        let width = u32::try_from(number(&stream["width"]).context("Video width missing")?)?;
        let height = u32::try_from(number(&stream["height"]).context("Video height missing")?)?;
        ensure!(width > 0 && height > 0, "Invalid source dimensions");
        ensure!(
            SourceIdentity::read(&path)? == source_identity,
            "The source changed while it was being indexed; import it again"
        );
        Ok(Self {
            path,
            width,
            height,
            frame_count: pts.len() as u64,
            fps,
            time_base,
            first_pts: pts[0],
            has_b_frames: number(&stream["has_b_frames"]).unwrap_or(0) > 0,
            source_identity: Some(source_identity),
        })
    }
    pub fn duration_seconds(&self) -> f64 {
        self.fps.frame_seconds(self.frame_count)
    }
    pub fn source_seconds(&self, frame: u64) -> f64 {
        self.first_pts as f64 * self.time_base.value() + self.fps.frame_seconds(frame)
    }
}

/// Frame extraction selects the decoded display-order frame, not a rounded time.
/// Each cache recipe includes source and decoder identity; output publication is atomic.
pub fn extract_source_frame(
    media: &MediaInfo,
    frame: u64,
    directory: impl AsRef<Path>,
    runtime: &RuntimePaths,
) -> Result<PathBuf> {
    media.verify_source_unchanged()?;
    ensure!(
        frame < media.frame_count,
        "Preview frame is outside the source video"
    );
    let metadata = fs::metadata(&media.path)?;
    let modified = metadata.modified()?.duration_since(UNIX_EPOCH)?.as_nanos();
    let decoder = fs::metadata(&runtime.ffmpeg)?;
    let decoder_modified = decoder.modified()?.duration_since(UNIX_EPOCH)?.as_nanos();
    let recipe = format!(
        "source-rgb-frame-v1|{}|{}|{}|{}|{}|{}|{}",
        media.path.display(),
        metadata.len(),
        modified,
        runtime.ffmpeg.display(),
        decoder.len(),
        decoder_modified,
        frame
    );
    let key = format!("{:x}", Sha256::digest(recipe.as_bytes()));
    let folder = directory.as_ref().join(key);
    fs::create_dir_all(&folder)?;
    let destination = frame_file(&folder, frame);
    if destination.is_file() && fs::metadata(&destination)?.len() > 0 {
        return Ok(destination);
    }
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let temporary = folder.join(format!("preview-{}-{nonce}.png", std::process::id()));
    let filter = format!("select=eq(n\\,{frame})");
    let result = checked_output(
        Command::new(&runtime.ffmpeg)
            .args(["-hide_banner", "-v", "error", "-nostdin", "-y", "-i"])
            .arg(&media.path)
            .args([
                "-map",
                "0:v:0",
                "-vf",
                &filter,
                "-an",
                "-sn",
                "-dn",
                "-frames:v",
                "1",
                "-fps_mode",
                "passthrough",
                "-f",
                "image2",
            ])
            .arg(&temporary),
        "Source frame extraction",
    );
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = media.verify_source_unchanged() {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    ensure!(
        temporary.is_file() && fs::metadata(&temporary)?.len() > 0,
        "FFmpeg did not produce the requested source frame"
    );
    if let Err(error) = fs::rename(&temporary, &destination) {
        let _ = fs::remove_file(&temporary);
        if !destination.is_file() {
            return Err(error.into());
        }
    }
    Ok(destination)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_cfr_validation_preserves_b_frame_display_order_and_source_offset() {
        let fps = Rational {
            num: 30000,
            den: 1001,
        };
        let tb = Rational { num: 1, den: 30000 };
        assert!(validate_cfr_pts(&[30030, 31031, 32032], fps, tb).is_ok());
        assert!(validate_cfr_pts(&[30030, 31031, 32033], fps, tb).is_err());
        assert!(validate_cfr_pts(&[30030, 30030], fps, tb).is_err());
        assert_eq!(fps.frame_seconds(50), 50050.0 / 30000.0);
    }
    #[test]
    fn changed_source_rejects_jobs_and_cached_previews_before_a_decoder_can_start() {
        let folder = std::env::temp_dir().join(format!(
            "brp-identity-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&folder).unwrap();
        let source = folder.join("same-geometry.mp4");
        fs::write(&source, b"first-version").unwrap();
        let media = MediaInfo {
            path: source.clone(),
            width: 1920,
            height: 1080,
            frame_count: 69,
            fps: Rational {
                num: 30000,
                den: 1001,
            },
            time_base: Rational { num: 1, den: 30000 },
            first_pts: 0,
            has_b_frames: false,
            source_identity: Some(SourceIdentity::read(&source).unwrap()),
        };
        assert!(media.verify_source_unchanged().is_ok());
        // Geometry/rate/count metadata stays identical; file identity changes.
        fs::write(&source, b"edited-version-with-different-pixels").unwrap();
        assert!(media.verify_source_unchanged().is_err());
        let request = crate::JobRequest {
            schema_version: crate::SCHEMA_VERSION,
            job_id: "identity-test".into(),
            media: media.clone(),
            output_dir: folder.join("job"),
            runtime: RuntimePaths::default(),
            kind: crate::JobKind::DetectScenes {
                threshold: 0.35,
                min_scene_frames: 6,
            },
        };
        assert!(request.validate().is_err());
        let preview =
            extract_source_frame(&media, 9, folder.join("cache"), &RuntimePaths::default())
                .unwrap_err();
        assert!(preview.to_string().contains("source video changed"));
        fs::remove_file(source).unwrap();
        fs::remove_dir(folder).unwrap();
    }
}
