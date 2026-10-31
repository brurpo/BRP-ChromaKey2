//! CPU-only persistent source decoding. Raw frames carry verified source PTS;
//! previews never create a PNG per seek/playback frame.
use crate::ram_cache::RamCache;
use anyhow::{Context, Result, bail, ensure};
use ck_desktop_core::{MediaInfo, Rational, SourceIdentity};
use std::{
    collections::VecDeque,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::{Child, ChildStdout, Command, Stdio},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

#[cfg(test)]
#[path = "source_decoder_cartolas_tests.rs"]
mod cartolas_tests;

#[cfg(windows)]
#[path = "source_decoder_native.rs"]
mod native;
#[cfg(windows)]
use native::NativeDecoder;

const CACHE_BYTES: usize = 128 * 1024 * 1024;
const CACHE_FRAMES: usize = 24;
const FORWARD_DECODE_LIMIT: u64 = 32;

#[derive(Clone, PartialEq, Eq)]
struct SourceKey {
    path: PathBuf,
    identity: Option<SourceIdentity>,
    width: u32,
    height: u32,
    count: u64,
    fps: Rational,
    time_base: Rational,
    first_pts: i64,
}
impl From<&MediaInfo> for SourceKey {
    fn from(media: &MediaInfo) -> Self {
        Self {
            path: media.path.clone(),
            identity: media.source_identity,
            width: media.width,
            height: media.height,
            count: media.frame_count,
            fps: media.fps,
            time_base: media.time_base,
            first_pts: media.first_pts,
        }
    }
}

struct OwnedProcess {
    child: Child,
    source: SourceKey,
    next: u64,
}

enum DesiredSource {
    Source(SourceKey, u64),
    Cancel,
}

#[derive(Default)]
struct DecodeState {
    process: Option<OwnedProcess>,
    desired: Option<DesiredSource>,
    native_active: bool,
}

#[derive(Default)]
pub(super) struct DecodeControl(Mutex<DecodeState>);
impl DecodeControl {
    fn stop(&self) {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        state.native_active = false;
        if let Some(mut owned) = state.process.take() {
            let _ = owned.child.kill();
            let _ = owned.child.wait();
        }
    }
    pub(super) fn cancel_obsolete(&self, media: &MediaInfo, target: u64) {
        let key = SourceKey::from(media);
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let obsolete = state.process.as_ref().is_some_and(|process| {
            process.source != key
                || target > process.next.saturating_add(FORWARD_DECODE_LIMIT)
                || target.saturating_add(CACHE_FRAMES as u64) < process.next
        });
        state.desired = Some(DesiredSource::Source(key, target));
        if obsolete && let Some(mut process) = state.process.take() {
            let _ = process.child.kill();
            let _ = process.child.wait();
        }
    }
    pub(super) fn shutdown(&self) {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        state.desired = Some(DesiredSource::Cancel);
        state.native_active = false;
        if let Some(mut process) = state.process.take() {
            let _ = process.child.kill();
            let _ = process.child.wait();
        }
    }
    fn install(&self, mut child: Child, source: SourceKey, next: u64) -> Result<()> {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let obsolete = match state.desired.as_ref() {
            Some(DesiredSource::Cancel) => true,
            Some(DesiredSource::Source(key, target)) => {
                key != &source
                    || *target > next.saturating_add(FORWARD_DECODE_LIMIT)
                    || target.saturating_add(CACHE_FRAMES as u64) < next
            }
            None => false,
        };
        if obsolete {
            let _ = child.kill();
            let _ = child.wait();
            bail!("Source preview request was superseded before decoder startup");
        }
        if let Some(mut old) = state.process.take() {
            let _ = old.child.kill();
            let _ = old.child.wait();
        }
        state.process = Some(OwnedProcess {
            child,
            source,
            next,
        });
        state.native_active = false;
        Ok(())
    }
    fn set_next(&self, next: u64) {
        if let Some(process) = self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .process
            .as_mut()
        {
            process.next = next;
        }
    }
    fn is_active(&self) -> bool {
        let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        state.process.is_some() || state.native_active
    }
    #[cfg(any(windows, test))]
    fn allows(&self, media: &MediaInfo, target: u64) -> bool {
        let state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        match state.desired.as_ref() {
            None => true,
            Some(DesiredSource::Cancel) => false,
            Some(DesiredSource::Source(key, desired)) => {
                key == &SourceKey::from(media)
                    && *desired <= target.saturating_add(FORWARD_DECODE_LIMIT)
                    && desired.saturating_add(CACHE_FRAMES as u64) >= target
            }
        }
    }
    #[cfg(windows)]
    fn set_native_active(&self, active: bool) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .native_active = active;
    }
}

/// Source-preview RGBA frame LRU: exact source frame index -> raw RGBA bytes.
///
/// The shared bounded cache keeps the behaviour of the previous private queue:
/// at most [`CACHE_BYTES`] retained bytes and at most [`CACHE_FRAMES`] frames,
/// least-recently-used frames evicted first, and frames larger than the whole
/// byte budget rejected instead of flushing the cache for one frame.
type FrameCache = RamCache<u64, Arc<[u8]>>;

pub(super) struct SourceFrames {
    source: Option<SourceKey>,
    decoder: Option<RawDecoder>,
    cache: FrameCache,
    cache_state: Arc<super::PreviewCacheState>,
    cache_generation: u64,
    control: Arc<DecodeControl>,
    #[cfg(windows)]
    native: Option<NativeDecoder>,
    #[cfg(windows)]
    native_failed: bool,
}
impl SourceFrames {
    pub(super) fn new(control: Arc<DecodeControl>) -> Self {
        Self::with_cache_state(control, Arc::new(super::PreviewCacheState::default()))
    }

    pub(super) fn with_cache_state(
        control: Arc<DecodeControl>,
        cache_state: Arc<super::PreviewCacheState>,
    ) -> Self {
        let byte_limit = cache_state
            .byte_limit
            .load(std::sync::atomic::Ordering::Relaxed);
        let item_limit = cache_state
            .item_limit
            .load(std::sync::atomic::Ordering::Relaxed);
        Self {
            source: None,
            decoder: None,
            cache: FrameCache::new(
                if byte_limit == 0 {
                    CACHE_BYTES
                } else {
                    byte_limit
                },
                if item_limit == 0 {
                    CACHE_FRAMES
                } else {
                    item_limit
                },
            ),
            cache_state,
            cache_generation: 0,
            control,
            #[cfg(windows)]
            native: None,
            #[cfg(windows)]
            native_failed: false,
        }
    }
    pub(super) fn read(
        &mut self,
        media: &MediaInfo,
        frame: u64,
        ffmpeg: &Path,
    ) -> Result<Arc<[u8]>> {
        media.verify_source_unchanged()?;
        ensure!(
            frame < media.frame_count,
            "Preview frame is outside the source video"
        );
        let key = SourceKey::from(media);
        if self.source.as_ref() != Some(&key) {
            self.suspend();
            #[cfg(windows)]
            {
                self.native_failed = false;
            }
            self.cache.clear();
            self.source = Some(key);
        }
        if let Some(pixels) = self.cache.get_cloned(&frame) {
            self.sync_cache_state();
            return Ok(pixels);
        }
        #[cfg(windows)]
        if !self.native_failed {
            let result = self.read_native(media, frame, ffmpeg);
            match result {
                Ok(pixels) => return Ok(pixels),
                Err(error) if !self.control.allows(media, frame) => {
                    self.native = None;
                    return Err(error);
                }
                Err(_) => {
                    // Unsupported ABI, container/side-data or conversion uses
                    // the established exact CLI decoder for this source.
                    self.native = None;
                    self.native_failed = true;
                }
            }
        }
        let reusable = self.control.is_active()
            && self.decoder.as_ref().is_some_and(|decoder| {
                frame >= decoder.next && frame - decoder.next <= FORWARD_DECODE_LIMIT
            });
        if !reusable {
            self.decoder = None;
            self.decoder = Some(RawDecoder::start(
                media,
                frame,
                ffmpeg,
                self.control.clone(),
                true,
            )?);
        }
        let result = self.read_through(media, frame);
        match result {
            Ok(pixels) => Ok(pixels),
            Err(error)
                if self.control.is_active()
                    && self
                        .decoder
                        .as_ref()
                        .is_some_and(|decoder| decoder.used_seek) =>
            {
                // A demuxer that cannot return the requested PTS must never be
                // relabeled as that frame. Retry the exact trim from origin.
                self.decoder = None;
                self.decoder = Some(RawDecoder::start(
                    media,
                    frame,
                    ffmpeg,
                    self.control.clone(),
                    false,
                )?);
                self.read_through(media, frame)
                    .with_context(|| format!("Exact source seek failed: {error:#}"))
            }
            Err(error) => {
                self.decoder = None;
                Err(error)
            }
        }
    }
    /// Apply the shared budget/clear requests and publish residents for the GUI.
    fn sync_cache_state(&mut self) {
        let generation = self
            .cache_state
            .clear_generation
            .load(std::sync::atomic::Ordering::Relaxed);
        if generation != self.cache_generation {
            self.cache.clear();
            self.cache_generation = generation;
        }
        let byte_limit = self
            .cache_state
            .byte_limit
            .load(std::sync::atomic::Ordering::Relaxed);
        let item_limit = self
            .cache_state
            .item_limit
            .load(std::sync::atomic::Ordering::Relaxed);
        if byte_limit > 0 && item_limit > 0 {
            self.cache.retune(byte_limit, item_limit);
        }
        self.cache_state
            .bytes
            .store(self.cache.bytes(), std::sync::atomic::Ordering::Relaxed);
        *self
            .cache_state
            .frames
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = self.cache.keys().into_iter().collect();
    }

    pub(super) fn suspend(&mut self) {
        self.decoder = None;
        #[cfg(windows)]
        {
            self.native = None;
        }
    }
    #[cfg(windows)]
    fn read_native(&mut self, media: &MediaInfo, target: u64, ffmpeg: &Path) -> Result<Arc<[u8]>> {
        if self.native.is_none() {
            self.native = Some(NativeDecoder::open(
                media,
                target,
                ffmpeg,
                self.control.clone(),
            )?);
        }
        let pixels = self
            .native
            .as_mut()
            .expect("existing CPU decoder")
            .read(media, target)?;
        media.verify_source_unchanged()?;
        self.cache.insert(target, pixels.clone());
        self.sync_cache_state();
        Ok(pixels)
    }
    fn read_through(&mut self, media: &MediaInfo, target: u64) -> Result<Arc<[u8]>> {
        loop {
            let decoder = self
                .decoder
                .as_mut()
                .context("Source decoder is unavailable")?;
            let frame = decoder.next;
            let pixels = decoder.read(media)?;
            self.cache.insert(frame, pixels.clone());
            if frame == target {
                self.sync_cache_state();
                media.verify_source_unchanged()?;
                return Ok(pixels);
            }
            ensure!(frame < target, "Source decoder passed the requested frame");
        }
    }
    pub(super) fn prefetch_one(&mut self, media: &MediaInfo, through: u64) -> Result<bool> {
        #[cfg(windows)]
        if let Some(native) = self.native.as_mut() {
            let next = native.next;
            if next >= media.frame_count || next > through {
                return Ok(false);
            }
            match native.read(media, next) {
                Ok(pixels) => {
                    self.cache.insert(next, pixels);
                    self.sync_cache_state();
                }
                Err(error) => {
                    self.native = None;
                    return Err(error);
                }
            }
            return Ok(true);
        }
        let Some(next) = self.decoder.as_ref().map(|decoder| decoder.next) else {
            return Ok(false);
        };
        if next >= media.frame_count || next > through {
            return Ok(false);
        }
        let pixels = match self.decoder.as_mut().expect("existing decoder").read(media) {
            Ok(pixels) => pixels,
            Err(error) => {
                self.decoder = None;
                return Err(error);
            }
        };
        self.cache.insert(next, pixels);
        self.sync_cache_state();
        Ok(true)
    }
}

#[derive(Clone, Copy, Debug)]
struct Stamp {
    pts: i64,
    time_base: Rational,
}
enum Metadata {
    Geometry(u32, u32),
    Frame(Stamp),
}

fn retain_diagnostic(errors: &Mutex<VecDeque<String>>, line: String) {
    let mut errors = errors.lock().unwrap_or_else(|e| e.into_inner());
    if errors.len() >= 12 {
        errors.pop_front();
    }
    errors.push_back(line);
}

/// FFmpeg forwards arbitrary container tags to stderr; those bytes need not be
/// UTF-8. A malformed human-readable tag must not terminate the ASCII frame
/// protocol that follows it on the same pipe.
fn forward_metadata(
    mut stderr: impl BufRead,
    sender: mpsc::Sender<Metadata>,
    errors: &Mutex<VecDeque<String>>,
) -> std::io::Result<()> {
    let mut output_header = false;
    let mut bytes = Vec::new();
    loop {
        bytes.clear();
        if stderr.read_until(b'\n', &mut bytes)? == 0 {
            return Ok(());
        }
        let line = String::from_utf8_lossy(&bytes);
        let line = line.trim_end_matches(['\r', '\n']);
        if line.starts_with("Output #") {
            output_header = true;
        }
        let metadata = if output_header {
            parse_geometry(line).map(|(w, h)| Metadata::Geometry(w, h))
        } else {
            None
        }
        .or_else(|| parse_stamp(line).map(Metadata::Frame));
        if let Some(metadata) = metadata {
            if sender.send(metadata).is_err() {
                return Ok(());
            }
        } else {
            retain_diagnostic(errors, line.to_owned());
        }
    }
}

fn parse_stamp(line: &str) -> Option<Stamp> {
    let fields = line
        .strip_prefix("BRP_FRAME ")?
        .split_whitespace()
        .collect::<Vec<_>>();
    if fields.len() != 4 {
        return None;
    }
    let original = fields[0].parse::<i64>().ok()?;
    let input_tb = Rational::parse(fields[1]);
    if original != i64::MAX
        && let Ok(time_base) = input_tb
    {
        return Some(Stamp {
            pts: original,
            time_base,
        });
    }
    Some(Stamp {
        pts: fields[2].parse().ok()?,
        time_base: Rational::parse(fields[3]).ok()?,
    })
}

fn parse_geometry(line: &str) -> Option<(u32, u32)> {
    if !line.contains("Video: rawvideo") {
        return None;
    }
    line.split(',').find_map(|part| {
        let token = part.split_whitespace().next()?;
        let (w, h) = token.split_once('x')?;
        Some((w.parse().ok()?, h.parse().ok()?))
    })
}

fn source_frame(media: &MediaInfo, stamp: Stamp) -> Result<u64> {
    let a = i128::from(stamp.pts)
        .checked_mul(i128::from(stamp.time_base.num))
        .and_then(|v| v.checked_mul(i128::from(media.time_base.den)))
        .context("Source timestamp overflow")?;
    let b = i128::from(media.first_pts)
        .checked_mul(i128::from(media.time_base.num))
        .and_then(|v| v.checked_mul(i128::from(stamp.time_base.den)))
        .context("Source timestamp overflow")?;
    let numerator = a
        .checked_sub(b)
        .and_then(|v| v.checked_mul(i128::from(media.fps.num)))
        .context("Source timestamp overflow")?;
    let denominator = i128::from(stamp.time_base.den)
        .checked_mul(i128::from(media.time_base.den))
        .and_then(|v| v.checked_mul(i128::from(media.fps.den)))
        .context("Source timestamp overflow")?;
    ensure!(
        numerator >= 0 && denominator > 0 && numerator % denominator == 0,
        "Decoder returned a presentation timestamp outside the exact source frame grid"
    );
    let index = u64::try_from(numerator / denominator)?;
    ensure!(
        index < media.frame_count,
        "Decoder returned a frame outside the imported source"
    );
    Ok(index)
}

fn target_pts(media: &MediaInfo, frame: u64) -> Result<i64> {
    if frame == 0 {
        return Ok(media.first_pts);
    }
    let numerator = u128::from(media.fps.den)
        .checked_mul(u128::from(media.time_base.den))
        .context("Source timestamp overflow")?;
    let denominator = u128::from(media.fps.num)
        .checked_mul(u128::from(media.time_base.num))
        .context("Source timestamp overflow")?;
    ensure!(
        denominator > 0 && numerator % denominator == 0,
        "Source timestamps do not form an exact constant-frame-rate grid"
    );
    let offset = (numerator / denominator)
        .checked_mul(u128::from(frame))
        .context("Source timestamp overflow")?;
    let value = i128::from(media.first_pts)
        .checked_add(i128::try_from(offset)?)
        .context("Source timestamp overflow")?;
    Ok(i64::try_from(value)?)
}

struct RawDecoder {
    stdout: Option<ChildStdout>,
    metadata: mpsc::Receiver<Metadata>,
    stderr_thread: Option<thread::JoinHandle<()>>,
    errors: Arc<Mutex<VecDeque<String>>>,
    geometry: Option<(u32, u32)>,
    next: u64,
    used_seek: bool,
    control: Arc<DecodeControl>,
}
impl RawDecoder {
    fn start(
        media: &MediaInfo,
        frame: u64,
        ffmpeg: &Path,
        control: Arc<DecodeControl>,
        seek: bool,
    ) -> Result<Self> {
        media.validate()?;
        let mut command = Command::new(ffmpeg);
        command.args([
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostats",
            "-nostdin",
            "-hwaccel",
            "none",
            "-threads",
            "2",
            "-copyts",
        ]);
        let used_seek = seek && frame > 2;
        if used_seek {
            command.args([
                "-seek_timestamp",
                "1",
                "-ss",
                &format!("{:.9}", media.source_seconds(frame - 2)),
            ]);
        }
        command
            .arg("-i")
            .arg(&media.path)
            .args([
                "-map",
                "0:v:0",
                "-vf",
                &format!("trim=start_pts={},format=rgba", target_pts(media, frame)?),
                "-an",
                "-sn",
                "-dn",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "rawvideo",
                "-threads:v",
                "1",
                "-enc_time_base:v",
                "demux",
                "-stats_enc_pre:v",
                "pipe:2",
                "-stats_enc_pre_fmt:v",
                "BRP_FRAME {ptsi} {tbi} {pts} {tb}",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000);
        }
        let mut child = command
            .spawn()
            .context("Starting CPU source preview decoder")?;
        let stdout = child
            .stdout
            .take()
            .context("Source decoder stdout missing")?;
        let stderr = child
            .stderr
            .take()
            .context("Source decoder stderr missing")?;
        control.install(child, SourceKey::from(media), frame)?;
        let (sender, metadata) = mpsc::channel();
        let errors = Arc::new(Mutex::new(VecDeque::new()));
        let error_sink = errors.clone();
        let stderr_thread = thread::spawn(move || {
            if let Err(error) = forward_metadata(BufReader::new(stderr), sender, &error_sink) {
                retain_diagnostic(
                    &error_sink,
                    format!("Reading CPU decoder output failed: {error}"),
                );
            }
        });
        Ok(Self {
            stdout: Some(stdout),
            metadata,
            stderr_thread: Some(stderr_thread),
            errors,
            geometry: None,
            next: frame,
            used_seek,
            control,
        })
    }
    fn read(&mut self, media: &MediaInfo) -> Result<Arc<[u8]>> {
        let bytes = usize::try_from(media.width)?
            .checked_mul(usize::try_from(media.height)?)
            .and_then(|v| v.checked_mul(4))
            .context("Source preview dimensions overflow")?;
        ensure!(
            bytes <= 512 * 1024 * 1024,
            "Source preview exceeds the 512MiB single-frame safety limit"
        );
        let mut stamp = None;
        while self.geometry.is_none() || stamp.is_none() {
            match self.metadata.recv_timeout(Duration::from_secs(10)) {
                Ok(Metadata::Geometry(w, h)) => self.geometry = Some((w, h)),
                Ok(Metadata::Frame(value)) => {
                    ensure!(
                        stamp.is_none(),
                        "Duplicate source timestamp before geometry"
                    );
                    stamp = Some(value);
                }
                Err(_) => bail!(
                    "CPU source decoder did not return frame metadata: {}",
                    self.errors
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("\n")
                ),
            }
        }
        ensure!(
            self.geometry == Some((media.width, media.height)),
            "Decoder geometry differs from the imported source; refusing a rotated/resized preview"
        );
        let index = source_frame(media, stamp.expect("existing frame stamp"))?;
        ensure!(
            index == self.next,
            "Decoder returned source frame {index} instead of {}",
            self.next
        );
        let mut pixels = vec![0u8; bytes];
        self.stdout
            .as_mut()
            .context("Source decoder pipe closed")?
            .read_exact(&mut pixels)
            .context("Reading CPU source frame")?;
        self.next += 1;
        self.control.set_next(self.next);
        Ok(pixels.into())
    }
}
impl Drop for RawDecoder {
    fn drop(&mut self) {
        self.control.stop();
        self.stdout.take();
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_encoded_tags_do_not_terminate_later_frame_metadata() {
        let output = b"Input #0, mov, from 'clip.mp4':\r\n    Metadata:\r\n      handler_name: Controlador de M\xeddia de V\xeddeo Apple\r\nOutput #0, rawvideo, to 'pipe:1':\r\n  Stream #0:0: Video: rawvideo, rgba, 1920x1080 [SAR 1:1 DAR 16:9]\r\n      handler_name: M\xeddia\r\nBRP_FRAME 0 1/30000 0 1/30000\nBRP_FRAME 1001 1/30000 1001 1/30000\n";
        let (sender, receiver) = mpsc::channel();
        let errors = Mutex::new(VecDeque::new());
        forward_metadata(std::io::Cursor::new(output), sender, &errors).unwrap();
        let records: Vec<_> = receiver.into_iter().collect();
        assert!(matches!(
            records.as_slice(),
            [
                Metadata::Geometry(1920, 1080),
                Metadata::Frame(Stamp { pts: 0, .. }),
                Metadata::Frame(Stamp { pts: 1001, .. })
            ]
        ));
        let errors = errors.into_inner().unwrap();
        assert!(
            errors
                .iter()
                .any(|line| line.contains("Controlador de M\u{fffd}dia"))
        );
    }

    #[test]
    fn decoder_output_io_errors_are_returned_instead_of_silently_discarded() {
        struct FailedRead;
        impl Read for FailedRead {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("decoder pipe read failed"))
            }
        }
        let (sender, _receiver) = mpsc::channel();
        let errors = Mutex::new(VecDeque::new());
        let error = forward_metadata(BufReader::new(FailedRead), sender, &errors).unwrap_err();
        assert_eq!(error.to_string(), "decoder pipe read failed");
    }
    fn media() -> MediaInfo {
        MediaInfo {
            path: PathBuf::from("fixture.mp4"),
            width: 128,
            height: 72,
            frame_count: 180,
            fps: Rational {
                num: 30000,
                den: 1001,
            },
            time_base: Rational { num: 1, den: 30000 },
            first_pts: 45045,
            has_b_frames: true,
            source_identity: None,
        }
    }
    #[test]
    fn rational_source_grid_preserves_offset_and_rejects_off_by_one_pts() {
        let media = media();
        assert_eq!(target_pts(&media, 89).unwrap(), 134134);
        assert_eq!(
            source_frame(
                &media,
                Stamp {
                    pts: 134134,
                    time_base: media.time_base
                }
            )
            .unwrap(),
            89
        );
        assert!(
            source_frame(
                &media,
                Stamp {
                    pts: 134133,
                    time_base: media.time_base
                }
            )
            .is_err()
        );
        assert!(
            source_frame(
                &media,
                Stamp {
                    pts: 45044,
                    time_base: media.time_base
                }
            )
            .is_err()
        );
    }
    #[test]
    fn encoder_stamp_uses_original_pts_and_handles_documented_fallback() {
        let stamp = parse_stamp("BRP_FRAME 45045 1/30000 45 1001/30000").unwrap();
        assert_eq!(stamp.pts, 45045);
        assert_eq!(stamp.time_base, Rational { num: 1, den: 30000 });
        let fallback = parse_stamp("BRP_FRAME 9223372036854775807 0/1 1001 1/30000").unwrap();
        assert_eq!(fallback.pts, 1001);
        assert!(parse_stamp("frame=5 fps=29.97").is_none());
        assert_eq!(
            parse_geometry(
                "  Stream #0:0: Video: rawvideo (RGBA), rgba(pc), 1920x1080 [SAR 1:1], 30 fps"
            ),
            Some((1920, 1080))
        );
    }
    #[test]
    fn cache_is_bounded_lru_and_oversize_frames_bypass_retention() {
        let mut cache = FrameCache::new(8, 2);
        cache.insert(0, Arc::from([0u8; 4]));
        cache.insert(1, Arc::from([1u8; 4]));
        assert!(cache.get(&0).is_some());
        cache.insert(2, Arc::from([2u8; 4]));
        assert!(cache.get(&1).is_none());
        assert_eq!(cache.bytes, 8);
        cache.insert(3, Arc::from([3u8; 9]));
        assert_eq!(cache.bytes, 8);
        assert!(cache.get(&3).is_none());
        cache.clear();
        assert_eq!(cache.bytes, 0);
    }

    #[test]
    fn quantized_timestamp_grid_is_the_existing_import_limitation() {
        let mut media = media();
        media.time_base = Rational { num: 1, den: 1000 };
        assert!(target_pts(&media, 1).is_err());
        // A single-frame source still needs no representable subsequent step.
        assert_eq!(target_pts(&media, 0).unwrap(), media.first_pts);
    }

    #[test]
    fn desired_target_is_recorded_even_before_decoder_process_exists() {
        let control = DecodeControl::default();
        let media = media();
        control.cancel_obsolete(&media, 77);
        let state = control.0.lock().unwrap();
        assert!(matches!(state.desired, Some(DesiredSource::Source(_, 77))));
        assert!(state.process.is_none());
        drop(state);
        control.shutdown();
        assert!(matches!(
            control.0.lock().unwrap().desired,
            Some(DesiredSource::Cancel)
        ));
    }

    #[test]
    fn native_decode_cancellation_keeps_nearby_requests_and_rejects_obsolete_work() {
        let control = DecodeControl::default();
        let media = media();
        assert!(control.allows(&media, 0));
        control.cancel_obsolete(&media, 20);
        assert!(control.allows(&media, 20));
        assert!(control.allows(&media, 24));
        assert!(!control.allows(&media, 77));
        control.cancel_obsolete(&media, 100);
        assert!(!control.allows(&media, 20));
        let mut other = media.clone();
        other.path = PathBuf::from("other.mp4");
        assert!(!control.allows(&other, 100));
        control.shutdown();
        assert!(!control.allows(&media, 100));
        assert!(!control.is_active());
    }

    #[test]
    fn published_request_exposes_controller_intent_before_worker_wakes() {
        use super::super::{Background, LatestRequests, Request, ViewMode};
        for frame in 0..16 {
            let requests = Arc::new(LatestRequests::default());
            let control = Arc::new(DecodeControl::default());
            control.shutdown();
            let queue = requests.clone();
            let decoder = control.clone();
            let worker = std::thread::spawn(move || {
                let (_, request) = match queue.take_next().unwrap() {
                    super::super::Work::Ui(version, request) => (version, request),
                    super::super::Work::Cache(_) => panic!("unexpected cache request"),
                };
                let state = decoder.0.lock().unwrap();
                assert!(
                    matches!(&state.desired, Some(DesiredSource::Source(key, next))
                    if key == &SourceKey::from(&request.media) && *next == request.frame)
                );
            });
            assert!(requests.publish(
                Request {
                    key: format!("source:{frame}"),
                    media: media(),
                    frame,
                    mode: ViewMode::Source,
                    artifact: None,
                    artifact_is_straight_rgba: false,
                    overlay: None,
                    overlay_remove: None,
                    background: Background::Checker,
                    despill: ck_desktop_core::DespillSettings::default(),
                    masks: None,
                },
                &control
            ));
            worker.join().unwrap();
        }
    }

    fn tools() -> (PathBuf, PathBuf) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let ffmpeg = root.join("app/media/ffmpeg.exe");
        assert!(
            ffmpeg.is_file(),
            "This explicit developer test needs installed CPU FFmpeg"
        );
        (root, ffmpeg)
    }

    fn runtime(root: &Path) -> ck_desktop_core::RuntimePaths {
        ck_desktop_core::RuntimePaths::installed(root.join("app"))
    }

    fn hidden(command: &mut Command) {
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

    fn reference_frame(ffmpeg: &Path, media: &MediaInfo, index: u64) -> Vec<u8> {
        // Independent display-order oracle: decode from file origin and select
        // n, without timestamp seeking, trimming, decoder state or our PTS math.
        let mut command = Command::new(ffmpeg);
        command
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-hwaccel",
                "none",
                "-threads",
                "2",
                "-i",
            ])
            .arg(&media.path)
            .args([
                "-map",
                "0:v:0",
                "-vf",
                &format!("select=eq(n\\,{index}),format=rgba"),
                "-an",
                "-sn",
                "-dn",
                "-frames:v",
                "1",
                "-fps_mode",
                "passthrough",
                "-threads:v",
                "1",
                "-f",
                "rawvideo",
                "pipe:1",
            ]);
        hidden(&mut command);
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            output.stdout.len(),
            media.width as usize * media.height as usize * 4
        );
        output.stdout
    }

    #[test]
    #[ignore = "Developer CPU-only media test; requires installed FFmpeg and private root videos"]
    fn real_video_seeks_match_origin_oracle_and_playback_uses_one_pipe() {
        use std::time::Instant;
        let (root, ffmpeg) = tools();
        for clip in ["shorter.mp4", "longer.mp4"] {
            let media = MediaInfo::probe(root.join(clip), &runtime(&root)).unwrap();
            let control = Arc::new(DecodeControl::default());
            let mut frames = SourceFrames::new(control.clone());
            let start = Instant::now();
            for frame in 0..media.frame_count {
                let pixels = frames.read(&media, frame, &ffmpeg).unwrap();
                assert_eq!(
                    pixels.len(),
                    media.width as usize * media.height as usize * 4
                );
            }
            let sequential = start.elapsed().as_secs_f64();
            let start = Instant::now();
            for _ in 0..20 {
                frames.read(&media, media.frame_count - 1, &ffmpeg).unwrap();
            }
            let cache_ms = start.elapsed().as_secs_f64() * 1000.0 / 20.0;
            assert!(
                frames.cache.bytes <= CACHE_BYTES && frames.cache.entries.len() <= CACHE_FRAMES
            );
            let mut seek_seconds = 0.0;
            let mut old_seconds = 0.0;
            for index in [
                media.frame_count / 2,
                0,
                media.frame_count - 1,
                media.frame_count / 3,
            ] {
                // Fresh reader for each random seek measures an uncached start.
                let mut seeking = SourceFrames::new(Arc::new(DecodeControl::default()));
                let start = Instant::now();
                let actual = seeking.read(&media, index, &ffmpeg).unwrap();
                seek_seconds += start.elapsed().as_secs_f64();
                let start = Instant::now();
                let expected = reference_frame(&ffmpeg, &media, index);
                old_seconds += start.elapsed().as_secs_f64();
                assert_eq!(
                    actual.as_ref(),
                    expected.as_slice(),
                    "{clip} source frame {index}"
                );
            }
            println!(
                "CPU_SOURCE_PREVIEW {{\"clip\":\"{clip}\",\"frames\":{},\"sequential_seconds\":{sequential:.6},\"sequential_fps\":{:.3},\"cache_hit_ms\":{cache_ms:.6},\"four_cold_seeks_seconds\":{seek_seconds:.6},\"four_origin_references_seconds\":{old_seconds:.6},\"cache_bytes\":{},\"cache_frames\":{},\"seek_oracles_exact\":true}}",
                media.frame_count,
                media.frame_count as f64 / sequential,
                frames.cache.bytes,
                frames.cache.entries.len()
            );
            drop(frames);
            assert!(!control.is_active());
        }
    }

    #[test]
    #[ignore = "Developer CPU-only encoder/decoder test; requires installed FFmpeg/FFprobe"]
    fn b_frames_nonzero_pts_and_keyframe_seeks_are_exact() {
        use std::io::Write;
        let (root, ffmpeg) = tools();
        let folder = std::env::temp_dir().join(format!(
            "brp-source-seek-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&folder).unwrap();
        let clip = folder.join("offset-b-frames.mp4");
        let mut command = Command::new(&ffmpeg);
        command
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "128x72",
                "-r",
                "30000/1001",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-g",
                "60",
                "-bf",
                "3",
                "-pix_fmt",
                "yuv420p",
                "-video_track_timescale",
                "30000",
                "-output_ts_offset",
                "1.5",
            ])
            .arg(&clip)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        hidden(&mut command);
        let mut encoder = command.spawn().unwrap();
        let mut input = encoder.stdin.take().unwrap();
        for frame in 0..180u32 {
            let mut rgb = Vec::with_capacity(128 * 72 * 3);
            for y in 0..72 {
                for x in 0..128 {
                    rgb.extend_from_slice(&[
                        (frame % 256) as u8,
                        ((y * 3 + frame * 2) % 256) as u8,
                        ((x + frame * 7) % 256) as u8,
                    ]);
                }
            }
            input.write_all(&rgb).unwrap();
        }
        drop(input);
        let output = encoder.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let media = MediaInfo::probe(&clip, &runtime(&root)).unwrap();
        assert!(media.first_pts != 0 && media.has_b_frames);
        let control = Arc::new(DecodeControl::default());
        let mut frames = SourceFrames::new(control.clone());
        for index in [0, 1, 59, 60, 61, 137, 138, 139, 3, 4, 179, 178] {
            let actual = frames.read(&media, index, &ffmpeg).unwrap();
            let expected = reference_frame(&ffmpeg, &media, index);
            assert_eq!(
                actual.as_ref(),
                expected.as_slice(),
                "Offset/B-frame source index {index}"
            );
        }
        drop(frames);
        assert!(!control.is_active());
        std::fs::remove_file(clip).unwrap();
        std::fs::remove_dir(folder).unwrap();
    }

    #[test]
    #[ignore = "Developer CPU-only process cancellation test; requires local FFmpeg and root shorter.mp4"]
    fn canceled_startup_and_blocked_decoder_release_and_restart_cleanly() {
        use std::time::Instant;
        let (root, ffmpeg) = tools();
        let media = MediaInfo::probe(root.join("shorter.mp4"), &runtime(&root)).unwrap();
        let control = Arc::new(DecodeControl::default());
        // Cancellation arriving before install must prohibit that startup.
        control.shutdown();
        assert!(RawDecoder::start(&media, 0, &ffmpeg, control.clone(), true).is_err());
        assert!(!control.is_active());
        // A live raw pipe can be interrupted while its unread output blocks.
        control.cancel_obsolete(&media, 0);
        let mut decoder = RawDecoder::start(&media, 0, &ffmpeg, control.clone(), true).unwrap();
        let now = Instant::now();
        control.shutdown();
        assert!(decoder.read(&media).is_err());
        drop(decoder);
        assert!(now.elapsed() < Duration::from_secs(2));
        assert!(!control.is_active());
        // A later legitimate target re-enables decoding, with exact identity.
        let target = media.frame_count / 2;
        control.cancel_obsolete(&media, target);
        let mut frames = SourceFrames::new(control.clone());
        let actual = frames.read(&media, target, &ffmpeg).unwrap();
        assert_eq!(actual.as_ref(), reference_frame(&ffmpeg, &media, target));
        drop(frames);
        assert!(!control.is_active());

        // Exercise the actual service/mailbox transition that previously could
        // wake Source while the preceding artifact's Cancel intent was active.
        use super::super::{Background, Message, PreviewService, Request, ViewMode};
        let folder = std::env::temp_dir().join(format!(
            "brp-preview-transition-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&folder).unwrap();
        let alpha = folder.join("alpha.png");
        image::GrayImage::new(media.width, media.height)
            .save(&alpha)
            .unwrap();
        let service = PreviewService::new(
            runtime(&root),
            folder.clone(),
            eframe::egui::Context::default(),
        );
        let request = |key: &str, frame, artifact: Option<PathBuf>| Request {
            key: key.into(),
            media: media.clone(),
            frame,
            mode: if artifact.is_some() {
                ViewMode::Alpha
            } else {
                ViewMode::Source
            },
            artifact,
            artifact_is_straight_rgba: false,
            overlay: None,
            overlay_remove: None,
            background: Background::Checker,
            despill: ck_desktop_core::DespillSettings::default(),
            masks: None,
        };
        for (key, frame, artifact) in [
            ("first", 0, None),
            ("alpha", 0, Some(alpha.clone())),
            ("return", target, None),
        ] {
            assert!(service.request(request(key, frame, artifact)));
            let message = service
                .messages
                .recv_timeout(Duration::from_secs(5))
                .unwrap();
            match message {
                Message::Frame {
                    key: actual_key,
                    frame: actual_frame,
                    result,
                    ..
                } => {
                    assert_eq!(actual_key, key);
                    assert_eq!(actual_frame, frame);
                    let pixels = result.unwrap();
                    if key == "return" {
                        assert_eq!(
                            pixels.source_rgba.unwrap().as_ref(),
                            reference_frame(&ffmpeg, &media, target)
                        );
                        assert!(pixels.path.is_none());
                    }
                }
                Message::Imported(_) => panic!("Unexpected import event"),
                Message::Cached { .. } => panic!("Unexpected cache event"),
            }
        }
        let active = service.decoder.clone();
        drop(service);
        assert!(!active.is_active());
        std::fs::remove_file(alpha).unwrap();
        std::fs::remove_dir(folder).unwrap();
    }
}
