mod source_decoder;

use ck_desktop_core::{DespillSettings, MediaInfo, RuntimePaths, SourceIdentity};
use source_decoder::{DecodeControl, SourceFrames};
use std::{
    path::PathBuf,
    sync::{
        Arc, Condvar, Mutex,
        mpsc::{self, Receiver},
    },
    thread,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewMode {
    Source,
    Composite,
    Alpha,
    Sam,
}

impl ViewMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Source => "Source",
            Self::Composite => "Composite",
            Self::Alpha => "Alpha",
            Self::Sam => "SAM mask",
        }
    }
}

/// Display-space test backgrounds from the original Qt viewer. They are viewer
/// choices, not a change to the stored straight RGBA or the native linear render.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Background {
    #[default]
    Checker,
    Grid,
    Purple,
    Black,
    White,
    Bars,
}

impl Background {
    pub const ALL: [Self; 6] = [
        Self::Checker,
        Self::Grid,
        Self::Purple,
        Self::Black,
        Self::White,
        Self::Bars,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Checker => "Checker",
            Self::Grid => "Colour grid",
            Self::Purple => "Solid purple",
            Self::Black => "Black",
            Self::White => "White",
            Self::Bars => "Colour bars",
        }
    }

    fn pixel(self, x: u32, y: u32) -> [u8; 3] {
        match self {
            Self::Checker => {
                if (x / 64 + y / 64).is_multiple_of(2) {
                    [191, 191, 191]
                } else {
                    [127, 127, 127]
                }
            }
            Self::Grid => {
                const COLOURS: [[u8; 3]; 8] = [
                    [0xe0, 0x53, 0x53],
                    [0xe0, 0xa0, 0x53],
                    [0xd9, 0xe0, 0x53],
                    [0x6f, 0xe0, 0x53],
                    [0x53, 0xd9, 0xe0],
                    [0x5a, 0x7c, 0xe0],
                    [0xa9, 0x53, 0xe0],
                    [0xe0, 0x53, 0xb8],
                ];
                if x.is_multiple_of(30) || y.is_multiple_of(30) {
                    [18, 18, 18]
                } else {
                    COLOURS[((y / 30 % 2) * 3 + x / 30) as usize % COLOURS.len()]
                }
            }
            Self::Purple => [0x6a, 0x34, 0xa8],
            Self::Black => [0, 0, 0],
            Self::White => [255, 255, 255],
            Self::Bars => {
                const COLOURS: [[u8; 3]; 7] = [
                    [255, 255, 255],
                    [0xe0, 0xe0, 0],
                    [0, 0xe0, 0xe0],
                    [0, 0xc0, 0],
                    [0xe0, 0, 0xe0],
                    [0xe0, 0, 0],
                    [0, 0, 0xe0],
                ];
                COLOURS[(x / 32) as usize % COLOURS.len()]
            }
        }
    }
}

pub struct Pixels {
    pub width: usize,
    pub height: usize,
    pub rgba: Arc<[u8]>,
    pub path: Option<PathBuf>,
    /// Untinted source bytes retained in RAM; encode only for an explicit export.
    pub source_rgba: Option<Arc<[u8]>>,
    /// False for source/matte views or a legacy precomposited checker fallback.
    pub background_applied: bool,
    pub despill_applied: bool,
    pub despill_unavailable: bool,
}

pub enum Message {
    Imported(Result<MediaInfo, String>),
    Frame {
        key: String,
        frame: u64,
        mode: ViewMode,
        result: Result<Pixels, String>,
    },
    /// Background auto-cache fill result; inserted into the RAM result cache
    /// without disturbing the currently presented frame.
    Cached {
        key: String,
        frame: u64,
        result: Result<Pixels, String>,
    },
}

#[derive(Clone)]
pub struct Request {
    pub key: String,
    pub media: MediaInfo,
    pub frame: u64,
    pub mode: ViewMode,
    pub artifact: Option<PathBuf>,
    pub artifact_is_straight_rgba: bool,
    pub overlay: Option<PathBuf>,
    /// Union of the enabled Remove SAM layers, tinted red over `overlay`.
    pub overlay_remove: Option<PathBuf>,
    pub background: Background,
    pub despill: DespillSettings,
    /// Garbage/despill masks for a straight-RGBA Result frame, rasterised on the worker.
    pub masks: Option<MaskJob>,
}

/// Masks to apply to one frame. Rasterising and feathering take up to a few hundred
/// milliseconds, so the worker builds them (UI thread never does) and shares the result
/// through the mask RAM cache keyed by content fingerprint.
#[derive(Clone)]
pub struct MaskJob {
    pub key: crate::mask_cache::MaskCacheKey,
    pub stacks: Arc<ck_desktop_core::MaskStacks>,
    pub width: u32,
    pub height: u32,
}

impl MaskJob {
    fn resolve(
        &self,
        cache: &std::sync::Mutex<crate::mask_cache::MaskRasterCache>,
    ) -> Option<Arc<crate::mask_composite::MaskApplication>> {
        if let Some(hit) = cache.lock().ok()?.get(&self.key) {
            return Some(hit);
        }
        let built = Arc::new(crate::mask_composite::MaskApplication::build(
            &self.stacks,
            self.key.frame,
            self.width,
            self.height,
        )?);
        if let Ok(mut cache) = cache.lock() {
            let cost = crate::mask_cache::application_cost(&built);
            cache.insert(self.key, cost, Arc::clone(&built));
        }
        Some(built)
    }
}

impl Request {
    fn needs_despill_plate(&self) -> bool {
        self.mode == ViewMode::Composite
            && self.artifact.is_some()
            && self.artifact_is_straight_rgba
            && self.despill.is_enabled()
    }
}

/// Shared view of the source-frame RAM cache (worker -> GUI feedback + budget).
#[derive(Default)]
pub struct PreviewCacheState {
    pub bytes: std::sync::atomic::AtomicUsize,
    pub frames: std::sync::Mutex<std::collections::BTreeSet<u64>>,
    pub byte_limit: std::sync::atomic::AtomicUsize,
    pub item_limit: std::sync::atomic::AtomicUsize,
    pub clear_generation: std::sync::atomic::AtomicU64,
}

impl PreviewCacheState {
    pub fn new(byte_limit: usize, item_limit: usize) -> Self {
        Self {
            bytes: Default::default(),
            frames: Default::default(),
            byte_limit: std::sync::atomic::AtomicUsize::new(byte_limit),
            item_limit: std::sync::atomic::AtomicUsize::new(item_limit),
            clear_generation: Default::default(),
        }
    }

    pub fn set_limits(&self, byte_limit: usize, item_limit: usize) {
        self.byte_limit
            .store(byte_limit, std::sync::atomic::Ordering::Relaxed);
        self.item_limit
            .store(item_limit, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn request_clear(&self) {
        self.clear_generation
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn bytes(&self) -> usize {
        self.bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn resident_frames(&self) -> std::collections::BTreeSet<u64> {
        self.frames
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }
}

pub struct PreviewService {
    requests: Arc<LatestRequests>,
    mask_cache: Arc<std::sync::Mutex<crate::mask_cache::MaskRasterCache>>,
    cache_state: Arc<PreviewCacheState>,
    decoder: Arc<DecodeControl>,
    pub messages: Receiver<Message>,
    messages_sender: mpsc::Sender<Message>,
    repaint: eframe::egui::Context,
    worker: Option<thread::JoinHandle<()>>,
}

#[derive(Default)]
struct PendingRequests {
    latest: Option<(u64, Request)>,
    version: u64,
    stopped: bool,
    /// Low-priority background requests used to fill the RAM result cache.
    cache: std::collections::VecDeque<Request>,
}

const MAX_CACHE_QUEUE: usize = 256;

pub(super) enum Work {
    Ui(u64, Request),
    Cache(Request),
}

#[derive(Default)]
struct LatestRequests {
    state: Mutex<PendingRequests>,
    changed: Condvar,
}

impl LatestRequests {
    fn take_next(&self) -> Option<Work> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if state.stopped {
                return None;
            }
            if let Some(request) = state.latest.take() {
                return Some(Work::Ui(request.0, request.1));
            }
            if let Some(request) = state.cache.pop_front() {
                return Some(Work::Cache(request));
            }
            state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn push_cache(&self, request: Request) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.stopped || state.cache.len() >= MAX_CACHE_QUEUE {
            return false;
        }
        state.cache.push_back(request);
        self.changed.notify_one();
        true
    }

    /// Drops queued (not yet started) background requests; returns how many were dropped.
    /// Dropped requests never report `Message::Cached`.
    fn clear_cache(&self) -> usize {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let dropped = state.cache.len();
        state.cache.clear();
        dropped
    }

    fn cache_pending(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .cache
            .len()
    }
    fn has_pending(&self) -> bool {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.stopped || state.latest.is_some()
    }
    fn is_newer(&self, version: u64) -> bool {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.stopped || state.version != version
    }
    fn publish(&self, request: Request, decoder: &DecodeControl) -> bool {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.stopped {
            return false;
        }
        state.version = state.version.wrapping_add(1);
        let version = state.version;
        // Publish controller intent under the same queue lock before the worker
        // can consume this version. Old canceled reads also observe the newer
        // version before deciding whether their error is still relevant.
        if request.artifact.is_none() || request.needs_despill_plate() {
            decoder.cancel_obsolete(&request.media, request.frame);
        } else {
            decoder.shutdown();
        }
        state.latest = Some((version, request));
        self.changed.notify_one();
        true
    }
}

impl PreviewService {
    pub fn new(runtime: RuntimePaths, _cache: PathBuf, repaint: eframe::egui::Context) -> Self {
        let requests = Arc::new(LatestRequests::default());
        let input = requests.clone();
        let decoder = Arc::new(DecodeControl::default());
        let worker_decoder = decoder.clone();
        let cache_state = Arc::new(PreviewCacheState::new(
            crate::memory_budget::cache_limits(crate::memory_budget::total_ram_bytes(), 70)
                .source_bytes,
            512,
        ));
        let worker_state = cache_state.clone();
        let limits =
            crate::memory_budget::cache_limits(crate::memory_budget::total_ram_bytes(), 70);
        let mask_cache = Arc::new(std::sync::Mutex::new(
            crate::mask_cache::MaskRasterCache::new(limits.mask_bytes, limits.mask_items),
        ));
        let worker_masks = Arc::clone(&mask_cache);
        let (sender, messages) = mpsc::channel();
        let worker_sender = sender.clone();
        let worker_repaint = repaint.clone();
        let worker = thread::spawn(move || {
            let mut source = SourceFrames::with_cache_state(worker_decoder.clone(), worker_state);
            // Background cache fills read original plates with their own decoder: the
            // interactive decoder follows the playhead (and the shown media) and would reject
            // their frames as superseded, which used to stop the automatic RAM cache.
            let mut background_source = SourceFrames::new(Arc::new(DecodeControl::default()));
            let mut despill = crate::despill::DespillProcessor::default();
            let mut artifact_cache = None;
            while let Some(work) = input.take_next() {
                let (version, from_cache, request) = match work {
                    Work::Ui(version, request) => (Some(version), false, request),
                    Work::Cache(request) => (None, true, request),
                };
                let result = load(
                    &request,
                    &runtime,
                    if from_cache {
                        &mut background_source
                    } else {
                        &mut source
                    },
                    &mut despill,
                    &mut artifact_cache,
                    &worker_masks,
                )
                .map_err(|error| format!("{error:#}"));
                if from_cache {
                    // Cache-fill failures are silent; they must never replace
                    // the presented frame or raise a user-facing error. Always
                    // report completion so the auto-cache cannot stay stuck on
                    // an in-flight request that produced no message.
                    let _ = worker_sender.send(Message::Cached {
                        key: request.key,
                        frame: request.frame,
                        result,
                    });
                    worker_repaint.request_repaint();
                    continue;
                }
                // An obsolete canceled decode is not a user-visible failure.
                // Valid recent frames may still be presented during playback.
                if let (Some(version), true) = (version, result.is_err())
                    && input.is_newer(version)
                {
                    continue;
                }
                let prefetch = request.artifact.is_none() && result.is_ok();
                let media = request.media.clone();
                let prefetch_through = request.frame.saturating_add(4);
                let _ = worker_sender.send(Message::Frame {
                    key: request.key,
                    frame: request.frame,
                    mode: request.mode,
                    result,
                });
                worker_repaint.request_repaint();
                if prefetch {
                    for _ in 0..4 {
                        if input.has_pending() {
                            break;
                        }
                        if !matches!(source.prefetch_one(&media, prefetch_through), Ok(true)) {
                            break;
                        }
                    }
                }
            }
            worker_decoder.shutdown();
        });
        Self {
            requests,
            mask_cache,
            cache_state,
            decoder,
            messages,
            messages_sender: sender,
            repaint,
            worker: Some(worker),
        }
    }

    pub fn import(&self, path: PathBuf, runtime: RuntimePaths) {
        let sender = self.messages_sender.clone();
        let repaint = self.repaint.clone();
        thread::spawn(move || {
            let result = MediaInfo::probe(&path, &runtime).map_err(|error| format!("{error:#}"));
            let _ = sender.send(Message::Imported(result));
            repaint.request_repaint();
        });
    }

    /// Mask raster cache shared with the worker (budget, clearing, RAM readout).
    pub fn mask_cache(&self) -> &std::sync::Mutex<crate::mask_cache::MaskRasterCache> {
        &self.mask_cache
    }

    pub fn request(&self, request: Request) -> bool {
        self.requests.publish(request, &self.decoder)
    }

    /// Shared source-frame RAM-cache state (bytes, residents, limits, clear).
    pub fn cache_state(&self) -> Arc<PreviewCacheState> {
        self.cache_state.clone()
    }

    /// Queue a low-priority background request that fills the RAM result cache
    /// (the worker only processes these when no UI frame request is waiting).
    pub fn cache_request(&self, request: Request) -> bool {
        self.requests.push_cache(request)
    }

    /// Drops queued background requests; returns how many will never report back.
    pub fn clear_cache_queue(&self) -> usize {
        self.requests.clear_cache()
    }

    pub fn cache_queue_len(&self) -> usize {
        self.requests.cache_pending()
    }

    /// Release source file handles before the owner removes temporary media.
    /// Repeated shutdown and the subsequent Drop are safe.
    pub fn shutdown(&mut self) {
        {
            let mut state = self
                .requests
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            state.stopped = true;
            state.latest = None;
            state.cache.clear();
            self.requests.changed.notify_all();
        }
        self.decoder.shutdown();
        if let Some(worker) = self.worker.take() {
            // The worker owns native codec/demuxer objects. A stop signal alone
            // does not prove their file handles have been released on Windows.
            let _ = worker.join();
        }
    }
}

impl Drop for PreviewService {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn load(
    request: &Request,
    runtime: &RuntimePaths,
    source: &mut SourceFrames,
    despill: &mut crate::despill::DespillProcessor,
    artifact_cache: &mut Option<CachedArtifact>,
    mask_cache: &std::sync::Mutex<crate::mask_cache::MaskRasterCache>,
) -> anyhow::Result<Pixels> {
    if request.artifact.is_none() {
        request.media.verify_source_unchanged()?;
    }
    let path = request.artifact.as_ref();
    let raw_source = if path.is_none() {
        Some(source.read(&request.media, request.frame, &runtime.ffmpeg)?)
    } else if !request.needs_despill_plate() {
        source.suspend();
        None
    } else {
        None
    };
    if let Some(pixels) = raw_source.as_ref()
        && request.overlay.is_none()
        && request.overlay_remove.is_none()
    {
        return Ok(Pixels {
            width: request.media.width as usize,
            height: request.media.height as usize,
            rgba: pixels.clone(),
            path: None,
            source_rgba: raw_source,
            background_applied: false,
            despill_applied: false,
            despill_unavailable: false,
        });
    }
    let mut image = if let Some(path) = path {
        anyhow::ensure!(
            path.is_file(),
            "The requested result frame is no longer available; choose Source or render this frame again"
        );
        cached_display_image(
            artifact_cache,
            path,
            (request.media.width, request.media.height),
        )?
    } else {
        image::RgbaImage::from_raw(
            request.media.width,
            request.media.height,
            raw_source.as_ref().expect("source pixels").to_vec(),
        )
        .expect("verified source byte length")
    };
    let mask_composite = request.mode == ViewMode::Composite
        && request.artifact.is_some()
        && request.artifact_is_straight_rgba;
    let masks = if mask_composite {
        request
            .masks
            .as_ref()
            .and_then(|job| job.resolve(mask_cache))
    } else {
        None
    };
    if let Some(masks) = &masks {
        // v1 folds garbage masks into the matte before despill/compositing.
        crate::mask_composite::apply_garbage_masks(&mut image, masks);
    }
    let mut despill_applied = false;
    let mut despill_unavailable = false;
    if request.needs_despill_plate() {
        request.despill.validate()?;
        let weight = if mask_composite {
            masks
                .as_ref()
                .and_then(|masks| masks.despill_weight.as_ref())
                .map(|gray| crate::despill::MaskWeight::from_gray(gray))
        } else {
            None
        };
        match source.read(&request.media, request.frame, &runtime.ffmpeg) {
            Ok(plate) => {
                despill_applied =
                    despill.apply_weighted(plate, &mut image, request.despill, weight.as_ref())?
            }
            Err(_) => {
                // v1 keeps the calculated result visible when its original
                // plate is offline. The UI explicitly reports this fallback.
                despill_unavailable = true;
            }
        }
    }
    let background_applied = request.mode == ViewMode::Composite
        && request.artifact.is_some()
        && request.artifact_is_straight_rgba;
    if background_applied {
        composite_background(&mut image, request.background);
    }
    if let Some(mask) = request.overlay.as_ref().filter(|path| path.is_file()) {
        let mask = image::ImageReader::open(mask)?.decode()?.to_luma8();
        apply_mask_overlay(&mut image, &mask)?;
    }
    if let Some(mask) = request
        .overlay_remove
        .as_ref()
        .filter(|path| path.is_file())
    {
        let mask = image::ImageReader::open(mask)?.decode()?.to_luma8();
        apply_mask_tint(&mut image, &mask, REMOVE_TINT)?;
    }
    Ok(Pixels {
        width: image.width() as usize,
        height: image.height() as usize,
        rgba: image.into_raw().into(),
        path: path.cloned(),
        source_rgba: raw_source,
        background_applied,
        despill_applied,
        despill_unavailable,
    })
}

/// Retain one unmodified calculated image. Interactive strength/background edits
/// clone its pixels instead of repeatedly decoding the same full-size PNG.
struct CachedArtifact {
    path: PathBuf,
    identity: SourceIdentity,
    image: image::RgbaImage,
}
fn cached_display_image(
    cache: &mut Option<CachedArtifact>,
    path: &std::path::Path,
    expected: (u32, u32),
) -> anyhow::Result<image::RgbaImage> {
    let identity = SourceIdentity::read(path)?;
    if let Some(previous) = cache.as_ref()
        && previous.path == path
        && previous.identity == identity
        && previous.image.dimensions() == expected
    {
        return Ok(previous.image.clone());
    }
    let image = open_display_image(path, expected)?;
    anyhow::ensure!(
        SourceIdentity::read(path)? == identity,
        "Rendered frame changed while loading"
    );
    *cache = Some(CachedArtifact {
        path: path.to_path_buf(),
        identity,
        image: image.clone(),
    });
    Ok(image)
}

fn open_display_image(
    path: &std::path::Path,
    expected: (u32, u32),
) -> anyhow::Result<image::RgbaImage> {
    // Source PNGs and stored RGBA contain display sRGB bytes. Decode their
    // channel order/straight alpha without another transfer-function conversion.
    let image = image::ImageReader::open(path)?.decode()?.to_rgba8();
    anyhow::ensure!(
        image.dimensions() == expected,
        "Preview dimensions do not match the imported source; refusing to map prompts onto a resized result"
    );
    Ok(image)
}

fn display_blend(foreground: u8, background: u8, alpha: u16) -> u8 {
    ((u16::from(foreground) * alpha + u16::from(background) * (255 - alpha) + 127) / 255) as u8
}

fn composite_background(image: &mut image::RgbaImage, background: Background) {
    if background == Background::Checker {
        crate::despill::composite_checker(image);
        return;
    }
    // Qt's viewer and egui Color32 composite display-space sRGB. Keep this
    // preview recipe separate from the renderer's linear-light checker PNG.
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if pixel[3] == 255 {
            continue;
        }
        let rgb = background.pixel(x, y);
        let alpha = u16::from(pixel[3]);
        for channel in 0..3 {
            pixel[channel] = display_blend(pixel[channel], rgb[channel], alpha);
        }
        pixel[3] = 255;
    }
}

/// Red for areas a Remove SAM layer takes out of the subject mask.
const REMOVE_TINT: [u8; 3] = [236, 64, 56];

fn apply_mask_overlay(image: &mut image::RgbaImage, mask: &image::GrayImage) -> anyhow::Result<()> {
    // v1 tint_mask_overlay: blue ACCENT and integer mask*135/255 opacity.
    apply_mask_tint(image, mask, [88, 182, 242])
}

fn apply_mask_tint(
    image: &mut image::RgbaImage,
    mask: &image::GrayImage,
    color: [u8; 3],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        image.dimensions() == mask.dimensions(),
        "Mask dimensions do not match the source frame; refusing a misaligned overlay"
    );
    for (pixel, mask) in image.pixels_mut().zip(mask.pixels()) {
        let alpha = u16::from(mask[0]) * 135 / 255;
        for (channel, overlay) in color.into_iter().enumerate() {
            pixel[channel] = display_blend(overlay, pixel[channel], alpha);
        }
    }
    Ok(())
}

/// Full-resolution, exact source pixels for an explicit single-frame export.
pub fn read_source_rgba(
    media: &MediaInfo,
    frame: u64,
    ffmpeg: &std::path::Path,
) -> anyhow::Result<Arc<[u8]>> {
    let mut source = SourceFrames::new(Arc::new(DecodeControl::default()));
    source.read(media, frame, ffmpeg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shutdown_joins_idle_worker_and_is_idempotent() {
        let mut preview = PreviewService::new(
            RuntimePaths::default(),
            PathBuf::new(),
            eframe::egui::Context::default(),
        );
        assert!(preview.worker.is_some());
        let start = std::time::Instant::now();
        preview.shutdown();
        assert!(start.elapsed() < std::time::Duration::from_secs(2));
        assert!(preview.worker.is_none());
        assert!(preview.requests.state.lock().unwrap().stopped);
        assert!(preview.requests.take_next().is_none());
        // The worker and its SourceFrames both retain this controller until
        // native resources are dropped. Joining must release every such owner.
        assert_eq!(Arc::strong_count(&preview.decoder), 1);
        assert_eq!(Arc::strong_count(&preview.requests), 1);
        preview.shutdown();
        assert!(preview.worker.is_none());
    }

    #[test]
    fn offline_result_preserves_default_and_reports_unavailable_unmix() {
        use ck_desktop_core::Rational;
        let folder = std::env::temp_dir().join(format!(
            "brp-offline-despill-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&folder).unwrap();
        let artifact = folder.join("rgba.png");
        image::RgbaImage::from_pixel(2, 1, image::Rgba([200, 100, 50, 128]))
            .save(&artifact)
            .unwrap();
        let media = MediaInfo {
            path: folder.join("offline-source.mp4"),
            width: 2,
            height: 1,
            frame_count: 1,
            fps: Rational { num: 24, den: 1 },
            time_base: Rational { num: 1, den: 24 },
            first_pts: 0,
            has_b_frames: false,
            source_identity: None,
        };
        let mut request = Request {
            key: "offline".into(),
            media,
            frame: 0,
            mode: ViewMode::Composite,
            artifact: Some(artifact.clone()),
            artifact_is_straight_rgba: true,
            overlay: None,
            overlay_remove: None,
            background: Background::White,
            despill: DespillSettings::default(),
            masks: None,
        };
        let mut source = SourceFrames::new(Arc::new(DecodeControl::default()));
        let mut processor = crate::despill::DespillProcessor::default();
        let mut cache = None;
        let masks = std::sync::Mutex::new(crate::mask_cache::MaskRasterCache::new(1 << 20, 4));
        let original = load(
            &request,
            &RuntimePaths::default(),
            &mut source,
            &mut processor,
            &mut cache,
            &masks,
        )
        .unwrap();
        assert!(!original.despill_applied && !original.despill_unavailable);
        request.despill.amount = 0.5;
        let fallback = load(
            &request,
            &RuntimePaths::default(),
            &mut source,
            &mut processor,
            &mut cache,
            &masks,
        )
        .unwrap();
        assert!(!fallback.despill_applied && fallback.despill_unavailable);
        assert_eq!(fallback.rgba, original.rgba);
        assert_eq!(
            cache.as_ref().unwrap().image.get_pixel(0, 0).0,
            [200, 100, 50, 128]
        );
        std::fs::remove_file(artifact).unwrap();
        std::fs::remove_dir(folder).unwrap();
    }

    #[test]
    fn opaque_display_rgb_is_preserved_on_every_background() {
        let source = image::RgbaImage::from_raw(
            3,
            1,
            vec![32, 128, 240, 255, 200, 80, 40, 255, 0, 255, 17, 255],
        )
        .unwrap();
        for background in Background::ALL {
            let mut actual = source.clone();
            composite_background(&mut actual, background);
            assert_eq!(actual, source);
        }
    }

    #[test]
    fn straight_rgba_matches_v1_qpainter_cpu_reference_without_double_gamma() {
        // Independently measured with v1's installed PySide6 QImage/QPainter,
        // using 1px RGBA8888 images only: no app, torch, GPU or models loaded.
        for (rgba, background, expected) in [
            ([200, 100, 50, 135], Background::White, [226, 173, 146, 255]),
            ([255, 0, 0, 128], Background::White, [255, 127, 127, 255]),
            ([90, 140, 201, 85], Background::Purple, [101, 81, 179, 255]),
            ([0, 0, 0, 128], Background::White, [127, 127, 127, 255]),
        ] {
            let mut image = image::RgbaImage::from_pixel(1, 1, image::Rgba(rgba));
            composite_background(&mut image, background);
            assert_eq!(image.get_pixel(0, 0).0, expected);
        }
    }

    #[test]
    fn transparent_pixels_show_original_viewer_background_palette() {
        for background in Background::ALL {
            let mut image = image::RgbaImage::from_pixel(241, 61, image::Rgba([255, 7, 199, 0]));
            composite_background(&mut image, background);
            for (x, y) in [(0, 0), (14, 0), (1, 1), (31, 31), (240, 60)] {
                let rgb = background.pixel(x, y);
                assert_eq!(image.get_pixel(x, y).0, [rgb[0], rgb[1], rgb[2], 255]);
            }
        }
        assert_eq!(Background::Checker.pixel(0, 0), [191, 191, 191]);
        assert_eq!(Background::Checker.pixel(64, 0), [127, 127, 127]);
        assert_eq!(Background::Grid.pixel(1, 31), [111, 224, 83]);
        assert_eq!(Background::Bars.pixel(192, 0), [0, 0, 224]);
    }

    #[test]
    fn mask_overlay_matches_original_blue_and_opacity_without_touching_alpha() {
        let mut image = image::RgbaImage::from_pixel(3, 1, image::Rgba([200, 100, 50, 255]));
        let mask = image::GrayImage::from_raw(3, 1, vec![0, 128, 255]).unwrap();
        apply_mask_overlay(&mut image, &mask).unwrap();
        assert_eq!(image.get_pixel(0, 0).0, [200, 100, 50, 255]);
        assert_eq!(image.get_pixel(1, 0).0, [171, 122, 100, 255]);
        // Also measured independently using the original Qt painter.
        assert_eq!(image.get_pixel(2, 0).0, [141, 143, 152, 255]);
        assert!(apply_mask_overlay(&mut image, &image::GrayImage::new(1, 1)).is_err());
    }

    #[test]
    fn remove_layer_tint_is_red_and_only_where_masked() {
        let mut image = image::RgbaImage::from_pixel(2, 1, image::Rgba([100, 100, 100, 255]));
        let mask = image::GrayImage::from_raw(2, 1, vec![0, 255]).unwrap();
        apply_mask_tint(&mut image, &mask, REMOVE_TINT).unwrap();
        assert_eq!(image.get_pixel(0, 0).0, [100, 100, 100, 255]);
        let red = image.get_pixel(1, 0).0;
        assert!(red[0] > 150 && red[1] < 100 && red[2] < 100, "{red:?}");
        assert_eq!(red[3], 255);
    }

    #[test]
    fn png_decode_preserves_rgb_channels_and_straight_alpha() {
        let folder = std::env::temp_dir().join(format!(
            "brp-preview-colour-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        std::fs::create_dir(&folder).unwrap();
        let path = folder.join("straight-rgba.png");
        let image = image::RgbaImage::from_raw(
            3,
            1,
            vec![32, 128, 240, 255, 200, 80, 40, 128, 255, 20, 30, 0],
        )
        .unwrap();
        image.save(&path).unwrap();
        assert_eq!(open_display_image(&path, (3, 1)).unwrap(), image);
        assert!(open_display_image(&path, (1, 3)).is_err());
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(folder).unwrap();
    }
}
