//! CPU playback-range assembly. Rendered straight-RGBA frames replace their
//! corresponding source frames; every gap uses the original plate, opaque.
//! No keying inference, per-frame source subprocess or disk frame staging.

use anyhow::{Context, Result, bail, ensure};
use ck_desktop_core::{DespillSettings, FrameRange, MediaInfo};
use eframe::egui;
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExportFormat {
    #[default]
    ProRes4444,
    PngSequence,
    ExrSequence,
}

impl ExportFormat {
    pub const ALL: [Self; 3] = [Self::ProRes4444, Self::PngSequence, Self::ExrSequence];

    pub fn label(self) -> &'static str {
        match self {
            Self::ProRes4444 => "ProRes 4444 · MOV with alpha",
            Self::PngSequence => "PNG sequence · straight RGBA",
            Self::ExrSequence => "EXR sequence · linear RGBA",
        }
    }

    pub fn is_sequence(self) -> bool {
        self != Self::ProRes4444
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::ProRes4444 => "mov",
            Self::PngSequence => "png",
            Self::ExrSequence => "exr",
        }
    }
}

#[derive(Clone)]
pub struct ExportRequest {
    pub media: MediaInfo,
    pub range: FrameRange,
    /// Absolute source-frame indices of the available straight-RGBA PNGs.
    pub rendered: BTreeMap<u64, PathBuf>,
    pub target: PathBuf,
    pub format: ExportFormat,
    pub despill: DespillSettings,
    /// Effective Result despill per absolute source frame (v1 scene overrides).
    pub despills: BTreeMap<u64, DespillSettings>,
    /// Editable garbage/despill stacks snapshot for masked compositing.
    pub masks: Option<std::sync::Arc<ck_desktop_core::MaskStacks>>,
}

#[derive(Debug)]
pub enum ExportEvent {
    Progress {
        current: u64,
        total: u64,
    },
    Finished {
        target: PathBuf,
        frames: u64,
        rendered_frames: u64,
        source_frames: u64,
    },
    Failed(String),
    Cancelled,
}

pub struct ExportService {
    pub messages: Receiver<ExportEvent>,
    control: Arc<ExportControl>,
    worker: Option<JoinHandle<()>>,
}

impl ExportService {
    pub fn start(request: ExportRequest, ffmpeg: PathBuf, repaint: egui::Context) -> Result<Self> {
        validate_request(&request)?;
        ensure!(ffmpeg.is_file(), "The bundled FFmpeg executable is missing");
        let (sender, messages) = mpsc::channel();
        let control = Arc::new(ExportControl::default());
        let worker_control = control.clone();
        let worker = thread::spawn(move || {
            let result = export(&request, &ffmpeg, &worker_control, |current, total| {
                let _ = sender.send(ExportEvent::Progress { current, total });
                repaint.request_repaint();
            });
            let event = match result {
                Ok(rendered_frames) => ExportEvent::Finished {
                    target: request.target,
                    frames: request.range.len(),
                    rendered_frames,
                    source_frames: request.range.len() - rendered_frames,
                },
                Err(_) if worker_control.cancelled.load(Ordering::Acquire) => {
                    ExportEvent::Cancelled
                }
                Err(error) => ExportEvent::Failed(format!("{error:#}")),
            };
            let _ = sender.send(event);
            repaint.request_repaint();
        });
        Ok(Self {
            messages,
            control,
            worker: Some(worker),
        })
    }

    pub fn cancel(&self) {
        self.control.cancel();
    }
}

impl Drop for ExportService {
    fn drop(&mut self) {
        self.control.cancel();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[derive(Default)]
struct ExportControl {
    cancelled: AtomicBool,
    children: Mutex<Vec<Child>>,
}

impl ExportControl {
    fn check(&self) -> Result<()> {
        ensure!(!self.cancelled.load(Ordering::Acquire), "Export cancelled");
        Ok(())
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        let mut children = self.children.lock().unwrap_or_else(|e| e.into_inner());
        for child in children.iter_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        children.clear();
    }

    fn install(&self, mut child: Child) -> Result<u32> {
        let mut children = self.children.lock().unwrap_or_else(|e| e.into_inner());
        if self.cancelled.load(Ordering::Acquire) {
            let _ = child.kill();
            let _ = child.wait();
            bail!("Export cancelled");
        }
        let id = child.id();
        children.push(child);
        Ok(id)
    }

    fn stop(&self, id: u32) {
        let mut children = self.children.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(index) = children.iter().position(|child| child.id() == id) {
            let mut child = children.swap_remove(index);
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    fn wait(&self, id: u32) -> Result<ExitStatus> {
        loop {
            self.check()?;
            {
                let mut children = self.children.lock().unwrap_or_else(|e| e.into_inner());
                let index = children
                    .iter()
                    .position(|child| child.id() == id)
                    .context("Export process was cancelled")?;
                if let Some(status) = children[index].try_wait()? {
                    let mut finished = children.swap_remove(index);
                    let _ = finished.wait();
                    return Ok(status);
                }
            }
            thread::sleep(Duration::from_millis(10));
        }
    }
}

struct PipeProcess<'a> {
    id: u32,
    stdin: Option<ChildStdin>,
    stdout: Option<ChildStdout>,
    stderr: Option<JoinHandle<Vec<u8>>>,
    control: &'a ExportControl,
}

impl<'a> PipeProcess<'a> {
    fn start(mut command: Command, control: &'a ExportControl) -> Result<Self> {
        control.check()?;
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000);
        }
        command.stderr(Stdio::piped());
        let mut child = command.spawn().context("Starting CPU export process")?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let mut stderr_pipe = child
            .stderr
            .take()
            .context("Export diagnostic pipe missing")?;
        let id = control.install(child)?;
        let stderr = thread::spawn(move || {
            let mut retained = Vec::new();
            let mut buffer = [0; 4096];
            while let Ok(count) = stderr_pipe.read(&mut buffer) {
                if count == 0 {
                    break;
                }
                retained.extend_from_slice(&buffer[..count]);
                if retained.len() > 16384 {
                    retained.drain(..retained.len() - 16384);
                }
            }
            retained
        });
        Ok(Self {
            id,
            stdin,
            stdout,
            stderr: Some(stderr),
            control,
        })
    }

    fn finish(mut self, label: &str) -> Result<()> {
        self.stdin.take();
        let status = self.control.wait(self.id)?;
        self.stdout.take();
        let diagnostics = self
            .stderr
            .take()
            .and_then(|reader| reader.join().ok())
            .unwrap_or_default();
        ensure!(
            status.success(),
            "{label}: {}",
            String::from_utf8_lossy(&diagnostics).trim()
        );
        Ok(())
    }
}

impl Drop for PipeProcess<'_> {
    fn drop(&mut self) {
        self.control.stop(self.id);
        self.stdin.take();
        self.stdout.take();
        if let Some(stderr) = self.stderr.take() {
            let _ = stderr.join();
        }
    }
}

fn validate_request(request: &ExportRequest) -> Result<()> {
    request.despill.validate()?;
    for despill in request.despills.values() {
        despill.validate()?;
    }
    request.media.validate()?;
    request.range.validate(request.media.frame_count)?;
    ensure!(
        !request.target.as_os_str().is_empty(),
        "Choose an export destination"
    );
    if request.target.exists() {
        ensure!(
            request.format.is_sequence() && request.target.is_dir(),
            "The export destination already exists; choose a new filename"
        );
        ensure!(
            fs::read_dir(&request.target)?.next().is_none(),
            "Choose an empty destination folder so existing files are preserved"
        );
    }
    let bytes = u64::from(request.media.width)
        .checked_mul(u64::from(request.media.height))
        .and_then(|value| value.checked_mul(4))
        .context("Export dimensions overflow")?;
    ensure!(
        bytes <= 512 * 1024 * 1024,
        "One export frame exceeds the 512MiB safety limit"
    );
    Ok(())
}

/// Unique sibling staging keeps sequence copies and partially encoded movies
/// away from user destinations. Cleanup only touches individually owned files.
struct Staging {
    path: PathBuf,
    sequence: bool,
    files: Vec<PathBuf>,
    published: bool,
}

impl Staging {
    fn new(target: &Path, sequence: bool) -> Result<Self> {
        let parent = target
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = parent.join(format!(
            ".chroma-export-{}-{nonce}{}",
            std::process::id(),
            if sequence { "" } else { ".mov" }
        ));
        if sequence {
            fs::create_dir(&path)?;
        } else {
            File::options().write(true).create_new(true).open(&path)?;
        }
        Ok(Self {
            path,
            sequence,
            files: Vec::new(),
            published: false,
        })
    }

    fn frame(&mut self, index: u64, extension: &str) -> PathBuf {
        let path = self.path.join(format!("frame_{index:06}.{extension}"));
        self.files.push(path.clone());
        path
    }

    fn publish(&mut self, target: &Path, control: &ExportControl) -> Result<()> {
        control.check()?;
        if self.sequence {
            let empty_destination = target.exists();
            if empty_destination {
                ensure!(
                    target.is_dir() && fs::read_dir(target)?.next().is_none(),
                    "The export destination is no longer empty"
                );
                fs::remove_dir(target).context("Replacing empty export destination")?;
            }
            if let Err(error) = fs::rename(&self.path, target) {
                if empty_destination && !target.exists() {
                    let _ = fs::create_dir(target);
                }
                return Err(error).context("Publishing image sequence");
            }
        } else {
            // A hard link publishes the complete file atomically without ever
            // replacing another file that appeared during a long export.
            if let Err(link_error) = fs::hard_link(&self.path, target) {
                let mut destination = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(target)
                    .with_context(|| format!("Publishing export ({link_error})"))?;
                let copied = (|| -> Result<()> {
                    let mut source = File::open(&self.path)?;
                    let mut buffer = vec![0; 1024 * 1024];
                    loop {
                        control.check()?;
                        let count = source.read(&mut buffer)?;
                        if count == 0 {
                            break;
                        }
                        destination.write_all(&buffer[..count])?;
                    }
                    destination.sync_all()?;
                    Ok(())
                })();
                drop(destination);
                if let Err(error) = copied {
                    let _ = fs::remove_file(target);
                    return Err(error);
                }
            }
            fs::remove_file(&self.path)?;
        }
        self.published = true;
        Ok(())
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        if self.sequence {
            for path in &self.files {
                let _ = fs::remove_file(path);
            }
            let _ = fs::remove_dir(&self.path);
        } else {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn source_command(request: &ExportRequest, ffmpeg: &Path) -> Command {
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
        .arg(&request.media.path)
        .args([
            "-map",
            "0:v:0",
            "-vf",
            &format!(
                "trim=start_frame={}:end_frame={},format=rgba",
                request.range.start, request.range.end_exclusive
            ),
            "-an",
            "-sn",
            "-dn",
            "-frames:v",
            &request.range.len().to_string(),
            "-fps_mode",
            "passthrough",
            "-threads:v",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped());
    command
}

fn prores_command(request: &ExportRequest, ffmpeg: &Path, staging: &Path) -> Command {
    let mut command = Command::new(ffmpeg);
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgba",
            "-video_size",
            &format!("{}x{}", request.media.width, request.media.height),
            "-framerate",
            &format!("{}/{}", request.media.fps.num, request.media.fps.den),
            "-i",
            "pipe:0",
            "-an",
            "-sn",
            "-dn",
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4444",
            "-pix_fmt",
            "yuva444p10le",
            "-alpha_bits",
            "16",
            "-vendor",
            "apl0",
            "-threads",
            "2",
            "-f",
            "mov",
        ])
        .arg(staging)
        .stdin(Stdio::piped())
        .stdout(Stdio::null());
    command
}

fn export(
    request: &ExportRequest,
    ffmpeg: &Path,
    control: &ExportControl,
    mut progress: impl FnMut(u64, u64),
) -> Result<u64> {
    validate_request(request)?;
    let needs_source = request.despill.is_enabled()
        || request
            .despills
            .values()
            .any(|despill| despill.is_enabled())
        || (request.range.start..request.range.end_exclusive).any(|frame| {
            !request
                .rendered
                .get(&frame)
                .is_some_and(|path| path.is_file())
        });
    if needs_source {
        request.media.verify_source_unchanged()?;
    }
    let bytes = request.media.width as usize * request.media.height as usize * 4;
    let mut staging = Staging::new(&request.target, request.format.is_sequence())?;
    let mut source = if needs_source {
        Some(PipeProcess::start(
            source_command(request, ffmpeg),
            control,
        )?)
    } else {
        None
    };
    let mut encoder = if request.format == ExportFormat::ProRes4444 {
        Some(PipeProcess::start(
            prores_command(request, ffmpeg, &staging.path),
            control,
        )?)
    } else {
        None
    };
    let mut rgba = vec![0; bytes];
    let mut rendered_count = 0;
    let mut last_progress = Instant::now();
    progress(0, request.range.len());
    for (output_index, frame) in (request.range.start..request.range.end_exclusive).enumerate() {
        control.check()?;
        if let Some(source) = source.as_mut() {
            source
                .stdout
                .as_mut()
                .context("Source export pipe is missing")?
                .read_exact(&mut rgba)
                .with_context(|| format!("Reading original source frame {frame}"))?;
        }
        if let Some(rendered) = request.rendered.get(&frame).filter(|path| path.is_file()) {
            let mut image = image::ImageReader::open(rendered)?.decode()?.to_rgba8();
            ensure!(
                image.dimensions() == (request.media.width, request.media.height),
                "Rendered frame {frame} does not match the source dimensions"
            );
            let masks = request.masks.as_ref().and_then(|stacks| {
                crate::mask_composite::MaskApplication::build(
                    stacks,
                    frame as i64,
                    request.media.width,
                    request.media.height,
                )
            });
            if let Some(masks) = &masks {
                crate::mask_composite::apply_garbage_masks(&mut image, masks);
            }
            let despill = request
                .despills
                .get(&frame)
                .copied()
                .unwrap_or(request.despill);
            if despill.is_enabled() {
                let weight = masks
                    .as_ref()
                    .and_then(|masks| masks.despill_weight.as_ref())
                    .map(|gray| crate::despill::MaskWeight::from_gray(gray));
                crate::despill::apply_rgba_weighted(
                    &rgba,
                    image.as_mut(),
                    request.media.width,
                    request.media.height,
                    despill,
                    weight.as_ref(),
                )?;
            }
            rgba.copy_from_slice(image.as_raw());
            rendered_count += 1;
        } else {
            ensure!(
                needs_source,
                "Rendered frame {frame} disappeared during export; reconnect the original footage or restore that render"
            );
            for pixel in rgba.as_chunks_mut::<4>().0.iter_mut() {
                pixel[3] = 255;
            }
        }
        match request.format {
            ExportFormat::ProRes4444 => encoder
                .as_mut()
                .and_then(|process| process.stdin.as_mut())
                .context("ProRes encoder input is missing")?
                .write_all(&rgba)
                .context("Writing ProRes RGBA frame")?,
            ExportFormat::PngSequence => {
                let file = staging.frame(output_index as u64, request.format.extension());
                image::save_buffer(
                    &file,
                    &rgba,
                    request.media.width,
                    request.media.height,
                    image::ColorType::Rgba8,
                )?;
            }
            ExportFormat::ExrSequence => {
                let file = staging.frame(output_index as u64, request.format.extension());
                write_linear_exr(
                    &file,
                    request.media.width,
                    request.media.height,
                    &rgba,
                    control,
                )?;
            }
        }
        let current = output_index as u64 + 1;
        if last_progress.elapsed() >= Duration::from_millis(80) || current == request.range.len() {
            progress(current, request.range.len());
            last_progress = Instant::now();
        }
    }
    if let Some(source) = source {
        source.finish("Decoding original export footage")?;
    }
    if let Some(encoder) = encoder {
        encoder.finish("Encoding ProRes 4444")?;
    }
    control.check()?;
    if needs_source {
        request.media.verify_source_unchanged()?;
    }
    staging.publish(&request.target, control)?;
    Ok(rendered_count)
}

fn srgb_to_linear(value: u8) -> f32 {
    let value = f32::from(value) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn exr_attribute(header: &mut Vec<u8>, name: &str, kind: &str, value: &[u8]) {
    header.extend_from_slice(name.as_bytes());
    header.push(0);
    header.extend_from_slice(kind.as_bytes());
    header.push(0);
    header.extend_from_slice(&(value.len() as u32).to_le_bytes());
    header.extend_from_slice(value);
}

/// Minimal standard OpenEXR scanline file, FLOAT straight RGBA, no compression.
/// This avoids adding another codec/runtime just for EXR export. Existing PNG
/// render precision is retained; RGB is converted from display sRGB to linear.
fn write_linear_exr(
    path: &Path,
    width: u32,
    height: u32,
    rgba: &[u8],
    control: &ExportControl,
) -> Result<()> {
    ensure!(
        width > 0 && height > 0 && width <= i32::MAX as u32 && height <= i32::MAX as u32,
        "Invalid EXR dimensions"
    );
    ensure!(
        rgba.len() == width as usize * height as usize * 4,
        "EXR source size mismatch"
    );
    let mut header = Vec::new();
    header.extend_from_slice(&20_000_630_u32.to_le_bytes());
    header.extend_from_slice(&2_u32.to_le_bytes());
    let mut channels = Vec::new();
    for channel in *b"ABGR" {
        channels.extend_from_slice(&[channel, 0]);
        channels.extend_from_slice(&2_i32.to_le_bytes());
        channels.extend_from_slice(&[0; 4]);
        channels.extend_from_slice(&1_i32.to_le_bytes());
        channels.extend_from_slice(&1_i32.to_le_bytes());
    }
    channels.push(0);
    exr_attribute(&mut header, "channels", "chlist", &channels);
    exr_attribute(&mut header, "compression", "compression", &[0]);
    let window: Vec<_> = [0_i32, 0, width as i32 - 1, height as i32 - 1]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect();
    exr_attribute(&mut header, "dataWindow", "box2i", &window);
    exr_attribute(&mut header, "displayWindow", "box2i", &window);
    exr_attribute(&mut header, "lineOrder", "lineOrder", &[0]);
    exr_attribute(
        &mut header,
        "pixelAspectRatio",
        "float",
        &1.0_f32.to_le_bytes(),
    );
    exr_attribute(&mut header, "screenWindowCenter", "v2f", &[0; 8]);
    exr_attribute(
        &mut header,
        "screenWindowWidth",
        "float",
        &1.0_f32.to_le_bytes(),
    );
    let chromaticities: Vec<_> = [0.64_f32, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    exr_attribute(
        &mut header,
        "chromaticities",
        "chromaticities",
        &chromaticities,
    );
    header.push(0);
    let scanline_bytes = width.checked_mul(16).context("EXR scanline is too wide")?;
    let block_size = u64::from(scanline_bytes) + 8;
    let first_block = header.len() as u64 + u64::from(height) * 8;
    let mut file = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
    file.write_all(&header)?;
    for row in 0..height {
        file.write_all(&(first_block + u64::from(row) * block_size).to_le_bytes())?;
    }
    let mut scanline = Vec::with_capacity(scanline_bytes as usize);
    for row in 0..height {
        control.check()?;
        file.write_all(&(row as i32).to_le_bytes())?;
        file.write_all(&scanline_bytes.to_le_bytes())?;
        scanline.clear();
        let start = row as usize * width as usize * 4;
        let pixels = &rgba[start..start + width as usize * 4];
        for channel in [3, 2, 1, 0] {
            for pixel in pixels.as_chunks::<4>().0.iter() {
                let value = if channel == 3 {
                    f32::from(pixel[3]) / 255.0
                } else {
                    srgb_to_linear(pixel[channel])
                };
                scanline.extend_from_slice(&value.to_le_bytes());
            }
        }
        file.write_all(&scanline)?;
    }
    file.flush()?;
    Ok(())
}

#[cfg(test)]
#[path = "range_export_tests.rs"]
mod tests;
