//! Self-contained portable .chroma media storage, with lazy blob extraction.
use anyhow::{Context, Result, bail, ensure};
use ck_core::mask_layers::Mask;
use ck_desktop_core::{FrameRange, MaskKind, MediaInfo, Project, SourceIdentity};
use image::ImageEncoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, FileTimes, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const MAGIC: &[u8; 16] = b"BRPCHROMA\0\r\n\x1a\0\0\0";
const HEADER_SIZE: u64 = 256;
const SLOT_SIZE: usize = 96;
const SLOT_OFFSETS: [u64; 2] = [64, 160];
const MAX_INDEX_SIZE: u64 = 128 * 1024 * 1024;
const EMBEDDED: &str = "@embedded/";
/// Storage-key segment holding painted mask raster blobs of one document:
/// `documents/<document>/mask-layers/<kind>/<layer-id>/<frame>.png`.
const MASK_LAYERS_SEGMENT: &str = "mask-layers";
static NONCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct FrameFiles {
    pub rgba: Option<PathBuf>,
    pub checker: Option<PathBuf>,
    pub alpha: Option<PathBuf>,
    /// Combined subject mask of the enabled SAM layers (a cache: rebuilt from `sam_layers`).
    pub mask: Option<PathBuf>,
    /// Per-layer SAM results, keyed by `SamLayer::id`.
    #[serde(default)]
    pub sam_layers: BTreeMap<u64, PathBuf>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaskBatch {
    pub range: FrameRange,
    pub directory: PathBuf,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HintFootage {
    pub path: PathBuf,
    pub first_frame_only: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub key: String,
    pub project: Project,
    pub range: FrameRange,
    pub playhead: u64,
    #[serde(default)]
    pub frames: BTreeMap<u64, FrameFiles>,
    #[serde(default)]
    pub masks: Vec<MaskBatch>,
    #[serde(default)]
    pub hint: Option<HintFootage>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChromaProject {
    pub chroma_version: u32,
    pub selected: usize,
    pub documents: Vec<Document>,
    #[serde(skip)]
    pub archive: Option<Arc<EmbeddedArchive>>,
}
pub fn unique_key() -> String {
    format!(
        "media-{}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        NONCE.fetch_add(1, Ordering::Relaxed)
    )
}
pub fn temporary_root() -> Result<PathBuf> {
    let root = std::env::temp_dir().join("BRP-Chroma").join(unique_key());
    fs::create_dir_all(&root)?;
    fs::write(
        root.join(".owned-temporary"),
        b"BRP-Chroma temporary calculations v1",
    )?;
    Ok(root)
}
#[derive(Debug)]
pub struct TemporaryWorkspace {
    pub root: PathBuf,
}
impl TemporaryWorkspace {
    pub fn new() -> Result<Self> {
        Ok(Self {
            root: temporary_root()?,
        })
    }
}
impl Drop for TemporaryWorkspace {
    fn drop(&mut self) {
        let _ = discard_temporary(&self.root);
    }
}
pub fn discard_temporary(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let base = fs::canonicalize(std::env::temp_dir().join("BRP-Chroma"))?;
    let target = fs::canonicalize(path)?;
    ensure!(
        target.starts_with(&base) && target != base,
        "Temporary path escaped the app's scratch directory"
    );
    ensure!(
        fs::read(target.join(".owned-temporary"))? == b"BRP-Chroma temporary calculations v1",
        "Unowned scratch tree"
    );
    fs::remove_dir_all(target)?;
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Blob {
    offset: u64,
    length: u64,
    sha256: [u8; 32],
    modified_unix_ns: u64,
    original_filename: String,
}
#[derive(Serialize, Deserialize)]
struct Index {
    container_version: u32,
    project: ChromaProject,
    blobs: BTreeMap<String, Blob>,
}
#[derive(Clone, Copy)]
struct Commit {
    sequence: u64,
    offset: u64,
    length: u64,
    hash: [u8; 32],
}
fn safe_component(value: &str) -> bool {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.len() > 255
        || value.ends_with(['.', ' '])
        || value.chars().any(|c| {
            c.is_control() || matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
        })
    {
        return false;
    }
    let stem = value
        .split('.')
        .next()
        .unwrap_or_default()
        .to_ascii_uppercase();
    !matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        && !(stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
}
fn safe_key(value: &str) -> bool {
    !value.is_empty()
        && value.is_ascii()
        && value.len() <= 2048
        && value.split('/').all(safe_component)
}
fn key_for_reference(path: &Path) -> Option<String> {
    let text = path.to_str()?.replace('\\', "/");
    text.strip_prefix(EMBEDDED).map(str::to_owned)
}
fn reference(key: &str) -> PathBuf {
    PathBuf::from(format!("{EMBEDDED}{key}"))
}
fn mask_kind_segment(kind: MaskKind) -> &'static str {
    match kind {
        MaskKind::Garbage => "garbage",
        MaskKind::Despill => "despill",
    }
}
/// Deterministic storage-key prefix of every painted raster key of one layer.
fn mask_layer_blob_prefix(document_key: &str, kind: MaskKind, layer_id: u32) -> String {
    format!(
        "documents/{document_key}/{MASK_LAYERS_SEGMENT}/{}/{layer_id:04}/",
        mask_kind_segment(kind)
    )
}
/// Deterministic storage key of one painted mask raster key.
fn mask_blob_key(document_key: &str, kind: MaskKind, layer_id: u32, frame: i64) -> String {
    format!(
        "{}{frame:06}.png",
        mask_layer_blob_prefix(document_key, kind, layer_id)
    )
}
/// Split a painted-mask storage key into `(document key, kind, layer id, frame)`.
fn parse_mask_blob_key(key: &str) -> Option<(&str, MaskKind, u32, i64)> {
    let rest = key.strip_prefix("documents/")?;
    let (document_key, tail) = rest.split_once('/')?;
    let tail = tail.strip_prefix(MASK_LAYERS_SEGMENT)?.strip_prefix('/')?;
    let mut parts = tail.split('/');
    let kind = match parts.next()? {
        "garbage" => MaskKind::Garbage,
        "despill" => MaskKind::Despill,
        _ => return None,
    };
    let layer_id = parts.next()?.parse::<u32>().ok()?;
    let frame = parts.next()?.strip_suffix(".png")?.parse::<i64>().ok()?;
    parts
        .next()
        .is_none()
        .then_some((document_key, kind, layer_id, frame))
}
/// Whether a storage key lies inside a document's painted-mask section, even
/// when it is malformed (used to reject bad keys instead of ignoring them).
fn in_mask_blob_section(key: &str) -> bool {
    key.strip_prefix("documents/")
        .and_then(|rest| rest.split_once('/'))
        .is_some_and(|(_, tail)| {
            tail.strip_prefix(MASK_LAYERS_SEGMENT)
                .is_some_and(|rest| rest.starts_with('/'))
        })
}
fn clean_filename(path: &Path) -> String {
    let mut name: String = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    if !safe_component(&name) {
        name = format!("media_{}", name.trim_matches(['.', ' ']));
    }
    if name.len() > 200 {
        name = format!("media_{}", &name[name.len() - 180..]);
    }
    name
}
fn exact_hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}
fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("fixed header field"))
}
fn decode_commit(bytes: &[u8]) -> Option<Commit> {
    if bytes.len() != SLOT_SIZE || exact_hash(&bytes[..56]) != bytes[56..88] {
        return None;
    }
    let commit = Commit {
        sequence: u64_at(bytes, 0),
        offset: u64_at(bytes, 8),
        length: u64_at(bytes, 16),
        hash: bytes[24..56].try_into().ok()?,
    };
    (commit.sequence > 0
        && commit.offset >= HEADER_SIZE
        && commit.length > 0
        && commit.length <= MAX_INDEX_SIZE)
        .then_some(commit)
}
fn encode_commit(commit: Commit) -> [u8; SLOT_SIZE] {
    let mut bytes = [0; SLOT_SIZE];
    bytes[..8].copy_from_slice(&commit.sequence.to_le_bytes());
    bytes[8..16].copy_from_slice(&commit.offset.to_le_bytes());
    bytes[16..24].copy_from_slice(&commit.length.to_le_bytes());
    bytes[24..56].copy_from_slice(&commit.hash);
    let hash = exact_hash(&bytes[..56]);
    bytes[56..88].copy_from_slice(&hash);
    bytes
}
fn validate_documents(project: &mut ChromaProject) -> Result<()> {
    ensure!(project.chroma_version == 1, "Unsupported .chroma schema");
    let mut keys = BTreeSet::new();
    for document in &mut project.documents {
        ensure!(
            document.key.is_ascii()
                && safe_component(&document.key)
                && keys.insert(document.key.to_ascii_lowercase()),
            "Unsafe or duplicate media key"
        );
        document.project.validate()?;
        let media = document
            .project
            .media
            .as_ref()
            .context("Project entry has no media")?;
        document.range.validate(media.frame_count)?;
        document.playhead = document.playhead.min(media.frame_count - 1);
        ensure!(
            document
                .frames
                .keys()
                .all(|frame| *frame < media.frame_count),
            "Calculated frame outside source range"
        );
        for batch in &document.masks {
            batch.range.validate(media.frame_count)?;
        }
    }
    project.selected = project
        .selected
        .min(project.documents.len().saturating_sub(1));
    Ok(())
}
fn calculated_paths(document: &Document) -> impl Iterator<Item = &PathBuf> {
    document.frames.values().flat_map(|files| {
        [&files.rgba, &files.checker, &files.alpha, &files.mask]
            .into_iter()
            .flatten()
            .chain(files.sam_layers.values())
    })
}
fn validate_index(index: &mut Index, commit: Commit) -> Result<()> {
    ensure!(
        index.container_version == 1,
        "Unsupported .chroma container version"
    );
    validate_documents(&mut index.project)?;
    let mut keys = BTreeSet::new();
    for (key, blob) in &index.blobs {
        ensure!(
            safe_key(key) && keys.insert(key.to_ascii_lowercase()),
            "Unsafe or aliased embedded media path"
        );
        ensure!(
            blob.offset >= HEADER_SIZE
                && blob
                    .offset
                    .checked_add(blob.length)
                    .is_some_and(|end| end <= commit.offset),
            "Embedded blob lies outside committed storage"
        );
    }
    for key in &keys {
        let mut parent = key.as_str();
        while let Some((prefix, _)) = parent.rsplit_once('/') {
            ensure!(
                !keys.contains(prefix),
                "Embedded file also used as a directory"
            );
            parent = prefix;
        }
    }
    for key in index.blobs.keys() {
        if !in_mask_blob_section(key) {
            continue;
        }
        let (document_key, kind, layer_id, _frame) = parse_mask_blob_key(key)
            .with_context(|| format!("Malformed painted-mask blob key {key}"))?;
        let document = index
            .project
            .documents
            .iter()
            .find(|document| document.key.as_str() == document_key)
            .with_context(|| {
                format!("Painted-mask blob {key} references a missing project document")
            })?;
        ensure!(
            document.project.mask_layers.layer(kind, layer_id).is_some(),
            "Painted-mask blob {key} references a missing mask layer"
        );
    }
    let check = |path: &Path, directory: bool, required: bool| -> Result<()> {
        if let Some(key) = key_for_reference(path) {
            ensure!(safe_key(&key), "Unsafe embedded reference");
            ensure!(
                if directory {
                    let prefix = format!("{key}/");
                    index
                        .blobs
                        .range(prefix.clone()..)
                        .next()
                        .is_some_and(|(candidate, _)| candidate.starts_with(&prefix))
                } else {
                    index.blobs.contains_key(&key)
                },
                "Project references missing embedded media"
            );
        } else {
            ensure!(
                !required,
                "Calculated media must be contained inside .chroma"
            );
        }
        Ok(())
    };
    for document in &index.project.documents {
        for path in calculated_paths(document) {
            check(path, false, true)?;
        }
        for batch in &document.masks {
            check(&batch.directory, true, true)?;
            let prefix = format!(
                "{}/",
                key_for_reference(&batch.directory).expect("checked embedded directory")
            );
            ensure!(
                index
                    .blobs
                    .range(prefix.clone()..)
                    .take_while(|(key, _)| key.starts_with(&prefix))
                    .all(|(key, _)| safe_component(
                        key.strip_prefix(&prefix).expect("matching prefix")
                    )),
                "Embedded mask batches must contain flat filenames"
            );
        }
        if let Some(media) = &document.project.media {
            check(&media.path, false, false)?;
        }
        if let Some(hint) = &document.hint {
            check(&hint.path, false, false)?;
        }
    }
    Ok(())
}
/// Encode one 8-bit painted mask raster as a grayscale PNG blob.
///
/// PNG keeps the container small (v1 wrote mask PNGs too) and the encoding is
/// deterministic for identical pixels, which lets [`Plan::add_bytes`] reuse an
/// unchanged blob across repeated saves.
fn encode_mask_png(mask: &Mask) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    image::codecs::png::PngEncoder::new(&mut bytes)
        .write_image(
            mask.as_slice(),
            mask.width(),
            mask.height(),
            image::ExtendedColorType::L8,
        )
        .context("Encoding painted mask raster")?;
    Ok(bytes)
}

/// Decode one painted mask blob, validating its resolution against the layer.
fn decode_mask_png(bytes: &[u8], width: u32, height: u32) -> Result<Mask> {
    let decoded = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)
        .context("Decoding painted mask raster")?;
    ensure!(
        decoded.width() == width && decoded.height() == height,
        "Painted mask raster resolution {}x{} does not match the layer resolution {width}x{height}",
        decoded.width(),
        decoded.height()
    );
    Mask::from_pixels(width, height, decoded.into_luma8().into_raw())
        .context("Painted mask raster has an unexpected pixel length")
}

/// Restore every painted mask blob of an opened container into its layer's
/// runtime `MaskDocument`.
///
/// Painted mask rasters are part of the editable document state, unlike the
/// lazy calculated RGBA/alpha/checker frames, so opening the index restores them
/// eagerly (they are small 8-bit PNGs and the mask timeline needs their
/// keyframes). Every blob is checksum-verified by the archive and its resolution
/// is validated against the layer or source resolution. Containers without mask
/// blobs (old projects) load unchanged.
fn hydrate_mask_rasters(project: &mut ChromaProject, archive: &EmbeddedArchive) -> Result<()> {
    if archive.blobs.is_empty() {
        return Ok(());
    }
    for document in &mut project.documents {
        let source_size = document.project.mask_raster_size();
        for kind in [MaskKind::Garbage, MaskKind::Despill] {
            for layer in document.project.mask_layers.layers_mut(kind) {
                let (width, height) = match (layer.width, layer.height) {
                    (width, height) if width > 0 && height > 0 => (width, height),
                    _ => source_size,
                };
                if width == 0 || height == 0 {
                    continue;
                }
                let prefix = mask_layer_blob_prefix(&document.key, kind, layer.layer_id);
                let mut painted = BTreeMap::new();
                for (key, blob) in archive
                    .blobs
                    .range(prefix.clone()..)
                    .take_while(|(key, _)| key.starts_with(&prefix))
                {
                    let Some(frame) = key
                        .strip_prefix(prefix.as_str())
                        .and_then(|name| name.strip_suffix(".png"))
                        .and_then(|stem| stem.parse::<i64>().ok())
                    else {
                        continue;
                    };
                    let bytes = archive.read_blob(blob)?;
                    let mask = decode_mask_png(&bytes, width, height).with_context(|| {
                        format!(
                            "Reading painted {} mask key {frame} of layer {}",
                            mask_kind_segment(kind),
                            layer.layer_id
                        )
                    })?;
                    painted.insert(frame, mask);
                }
                if !painted.is_empty() {
                    layer.set_raster_size(width, height);
                    layer.doc.frames = painted;
                }
            }
        }
    }
    Ok(())
}

fn read_committed(file: &mut File) -> Result<(Commit, Index)> {
    let length = file.metadata()?.len();
    let mut header = [0; HEADER_SIZE as usize];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut header)?;
    ensure!(
        &header[..16] == MAGIC && u64_at(&header, 16) == 1,
        "Unsupported .chroma binary header"
    );
    let mut candidates: Vec<_> = SLOT_OFFSETS
        .iter()
        .filter_map(|offset| decode_commit(&header[*offset as usize..*offset as usize + SLOT_SIZE]))
        .collect();
    candidates.sort_by_key(|commit| std::cmp::Reverse(commit.sequence));
    for commit in candidates {
        let attempt = (|| -> Result<Index> {
            ensure!(
                commit
                    .offset
                    .checked_add(commit.length)
                    .is_some_and(|end| end <= length),
                "Incomplete index commit"
            );
            let mut bytes = vec![0; usize::try_from(commit.length)?];
            file.seek(SeekFrom::Start(commit.offset))?;
            file.read_exact(&mut bytes)?;
            ensure!(
                exact_hash(&bytes) == commit.hash,
                "Project index checksum mismatch"
            );
            let mut index: Index = serde_json::from_slice(&bytes)?;
            validate_index(&mut index, commit)?;
            Ok(index)
        })();
        if let Ok(index) = attempt {
            return Ok((commit, index));
        }
    }
    bail!("The .chroma file has no complete, valid project commit")
}

#[derive(Debug)]
pub struct EmbeddedArchive {
    path: PathBuf,
    file: Mutex<File>,
    sequence: u64,
    commit_hash: [u8; 32],
    blobs: BTreeMap<String, Blob>,
    workspace: Arc<TemporaryWorkspace>,
    cache_paths: BTreeMap<String, PathBuf>,
    directory_paths: BTreeMap<String, PathBuf>,
    reverse_paths: BTreeMap<PathBuf, String>,
    extracted: Arc<Mutex<BTreeMap<PathBuf, SourceIdentity>>>,
    io_gate: Arc<Mutex<()>>,
}
impl EmbeddedArchive {
    fn cached_path(&self, key: &str) -> Option<&PathBuf> {
        self.cache_paths
            .get(key)
            .or_else(|| self.directory_paths.get(key))
    }
    fn key(&self, path: &Path) -> Option<String> {
        self.reverse_paths.get(path).cloned()
    }
    fn matches(&self, path: &Path) -> Vec<(String, Blob)> {
        let Some(key) = self.key(path) else {
            return Vec::new();
        };
        if let Some(blob) = self.blobs.get(&key) {
            return vec![(key, blob.clone())];
        }
        let prefix = format!("{key}/");
        self.blobs
            .range(prefix.clone()..)
            .take_while(|(candidate, _)| candidate.starts_with(&prefix))
            .map(|(key, blob)| (key.clone(), blob.clone()))
            .collect()
    }
    pub fn is_embedded(&self, path: &Path) -> bool {
        self.key(path).is_some_and(|key| {
            self.blobs.contains_key(&key) || self.directory_paths.contains_key(&key)
        })
    }
    pub fn needs_materialization(&self, path: &Path) -> bool {
        self.matches(path)
            .iter()
            .any(|(key, _)| !self.cache_paths[key].is_file())
    }
    pub fn original_filename(&self, path: &Path) -> Option<&str> {
        self.blobs
            .get(&self.key(path)?)
            .map(|blob| blob.original_filename.as_str())
    }
    /// Identity recorded immediately after checksum-verified extraction. This
    /// accommodates filesystem timestamp rounding without blessing edited data.
    pub fn materialized_identity(&self, path: &Path) -> Result<Option<SourceIdentity>> {
        let Some(key) = self.key(path).filter(|key| self.blobs.contains_key(key)) else {
            return Ok(None);
        };
        let extracted = self
            .extracted
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let identity = *extracted
            .get(&self.cache_paths[&key])
            .context("Embedded source has not been materialized")?;
        ensure!(
            SourceIdentity::read(path)? == identity,
            "Embedded source changed after extraction"
        );
        Ok(Some(identity))
    }
    fn stream_blob(&self, blob: &Blob, destination: &mut File) -> Result<()> {
        let _gate = self
            .io_gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut source = self.file.lock().unwrap_or_else(|error| error.into_inner());
        source.seek(SeekFrom::Start(blob.offset))?;
        let mut remaining = blob.length;
        let mut buffer = vec![0; 1024 * 1024];
        let mut hash = Sha256::new();
        while remaining > 0 {
            let count = usize::try_from(remaining.min(buffer.len() as u64))?;
            source
                .read_exact(&mut buffer[..count])
                .context("Reading embedded media")?;
            destination.write_all(&buffer[..count])?;
            hash.update(&buffer[..count]);
            remaining -= count as u64;
        }
        ensure!(
            <[u8; 32]>::from(hash.finalize()) == blob.sha256,
            "Embedded media checksum mismatch"
        );
        Ok(())
    }
    /// Read one embedded blob into memory, verifying its checksum.
    fn read_blob(&self, blob: &Blob) -> Result<Vec<u8>> {
        let _gate = self
            .io_gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut source = self.file.lock().unwrap_or_else(|error| error.into_inner());
        source.seek(SeekFrom::Start(blob.offset))?;
        let mut bytes = vec![0; usize::try_from(blob.length)?];
        source
            .read_exact(&mut bytes)
            .context("Reading embedded media")?;
        ensure!(
            exact_hash(&bytes) == blob.sha256,
            "Embedded media checksum mismatch"
        );
        Ok(bytes)
    }
    /// CPU I/O only: extracts selected files/directories, verifying each blob.
    pub fn materialize(&self, paths: &[PathBuf]) -> Result<()> {
        let mut selected = BTreeMap::new();
        for path in paths {
            selected.extend(self.matches(path));
        }
        for (key, blob) in selected {
            let mut extracted = self
                .extracted
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let destination = self.cache_paths[&key].clone();
            if extracted.contains_key(&destination) && destination.is_file() {
                continue;
            }
            ensure!(
                !destination.exists(),
                "Unexpected file in owned embedded-media cache"
            );
            let parent = destination
                .parent()
                .context("Embedded path has no parent")?;
            fs::create_dir_all(parent)?;
            ensure!(
                fs::canonicalize(parent)?.starts_with(fs::canonicalize(&self.workspace.root)?),
                "Embedded cache escaped owned Temp root"
            );
            let staging = parent.join(format!(".extract-{}.tmp", unique_key()));
            let result = (|| -> Result<()> {
                let mut output = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&staging)?;
                self.stream_blob(&blob, &mut output)?;
                output.set_times(
                    FileTimes::new()
                        .set_modified(UNIX_EPOCH + Duration::from_nanos(blob.modified_unix_ns)),
                )?;
                output.sync_all()?;
                drop(output);
                if fs::hard_link(&staging, &destination).is_ok() {
                    fs::remove_file(&staging)?;
                } else {
                    // Some portable filesystems do not support hard links. The
                    // archive mutex and private cache still serialize writers.
                    ensure!(
                        !destination.exists(),
                        "Embedded cache destination appeared during extraction"
                    );
                    fs::rename(&staging, &destination).context("Publishing embedded media")?;
                }
                Ok(())
            })();
            if result.is_err() {
                let _ = fs::remove_file(&staging);
            }
            result?;
            extracted.insert(destination.clone(), SourceIdentity::read(&destination)?);
        }
        Ok(())
    }
}
fn visit_paths(document: &mut Document, mut visit: impl FnMut(&mut PathBuf)) {
    if let Some(media) = &mut document.project.media {
        visit(&mut media.path);
    }
    if let Some(hint) = &mut document.hint {
        visit(&mut hint.path);
    }
    for files in document.frames.values_mut() {
        for path in [
            &mut files.rgba,
            &mut files.alpha,
            &mut files.checker,
            &mut files.mask,
        ]
        .into_iter()
        .flatten()
        {
            visit(path);
        }
        for path in files.sam_layers.values_mut() {
            visit(path);
        }
    }
    for batch in &mut document.masks {
        visit(&mut batch.directory);
    }
}
fn resolve_project(
    mut project: ChromaProject,
    archive: Arc<EmbeddedArchive>,
    base: &Path,
) -> Result<ChromaProject> {
    for document in &mut project.documents {
        // Older project documents (schema version 1) have no mask layers and
        // upgrade to the current schema; newer documents are rejected.
        document.project.migrate()?;
        visit_paths(document, |path| {
            if let Some(key) = key_for_reference(path).filter(|_| archive.sequence > 0) {
                *path = archive
                    .cached_path(&key)
                    .expect("validated embedded reference")
                    .clone();
            } else if !path.as_os_str().is_empty() && path.is_relative() {
                *path = base.join(&*path);
            }
        });
        document.project.output_dir = archive
            .workspace
            .root
            .join(&document.key)
            .join("new-results");
    }
    hydrate_mask_rasters(&mut project, &archive)?;
    validate_documents(&mut project)?;
    project.archive = Some(archive);
    Ok(project)
}
fn absolute(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}
fn hex_hash(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn cache_mapping(archive: &mut EmbeddedArchive, project: &ChromaProject) -> Result<()> {
    for document in &project.documents {
        for batch in &document.masks {
            let Some(prefix) = key_for_reference(&batch.directory) else {
                continue;
            };
            if archive.directory_paths.contains_key(&prefix) {
                continue;
            }
            let mut hash = Sha256::new();
            let mut members = Vec::new();
            let member_prefix = format!("{prefix}/");
            for (key, blob) in archive
                .blobs
                .range(member_prefix.clone()..)
                .take_while(|(key, _)| key.starts_with(&member_prefix))
            {
                if let Some(name) = key.strip_prefix(&member_prefix) {
                    ensure!(
                        safe_component(name),
                        "Embedded mask batches must contain flat filenames"
                    );
                    hash.update((name.len() as u64).to_le_bytes());
                    hash.update(name.as_bytes());
                    hash.update(blob.sha256);
                    members.push((key.clone(), name.to_owned()));
                }
            }
            let digest: [u8; 32] = hash.finalize().into();
            let directory = archive
                .workspace
                .root
                .join("embedded/masks")
                .join(hex_hash(&digest));
            archive.directory_paths.insert(prefix, directory.clone());
            for (key, name) in members {
                archive.cache_paths.insert(key, directory.join(name));
            }
        }
    }
    for (key, blob) in &archive.blobs {
        archive.cache_paths.entry(key.clone()).or_insert_with(|| {
            archive
                .workspace
                .root
                .join("embedded/blobs")
                .join(hex_hash(&blob.sha256))
                .join(clean_filename(Path::new(&blob.original_filename)))
        });
    }
    for (key, path) in archive.cache_paths.iter().chain(&archive.directory_paths) {
        archive
            .reverse_paths
            .entry(path.clone())
            .or_insert_with(|| key.clone());
    }
    Ok(())
}
pub fn load(path: &Path) -> Result<ChromaProject> {
    load_reusing(path, None)
}
fn load_reusing(path: &Path, reuse: Option<&Arc<EmbeddedArchive>>) -> Result<ChromaProject> {
    let path = absolute(path)?;
    let mut file = File::open(&path).context("Opening .chroma project")?;
    let mut magic = [0; 16];
    let binary = file.read(&mut magic)? == 16 && &magic == MAGIC;
    let (sequence, commit_hash, blobs, project) = if binary {
        let (commit, index) = read_committed(&mut file)?;
        (commit.sequence, commit.hash, index.blobs, index.project)
    } else {
        file.seek(SeekFrom::Start(0))?;
        ensure!(
            file.metadata()?.len() <= MAX_INDEX_SIZE,
            "Legacy project JSON is too large"
        );
        let value: serde_json::Value =
            serde_json::from_reader(&file).context("Reading legacy native JSON")?;
        let project = if value.get("chroma_version").is_some() {
            serde_json::from_value(value)?
        } else {
            let project = Project::load(&path)?;
            let count = project.media.as_ref().map_or(1, |media| media.frame_count);
            ChromaProject {
                chroma_version: 1,
                selected: 0,
                documents: vec![Document {
                    key: unique_key(),
                    project,
                    range: FrameRange {
                        start: 0,
                        end_exclusive: count,
                    },
                    playhead: 0,
                    frames: BTreeMap::new(),
                    masks: Vec::new(),
                    hint: None,
                }],
                archive: None,
            }
        };
        (0, [0; 32], BTreeMap::new(), project)
    };
    let mut archive = EmbeddedArchive {
        path: path.clone(),
        file: Mutex::new(file),
        sequence,
        commit_hash,
        blobs,
        workspace: if let Some(previous) = reuse {
            previous.workspace.clone()
        } else {
            Arc::new(TemporaryWorkspace::new()?)
        },
        cache_paths: BTreeMap::new(),
        directory_paths: BTreeMap::new(),
        reverse_paths: BTreeMap::new(),
        extracted: reuse.map_or_else(
            || Arc::new(Mutex::new(BTreeMap::new())),
            |previous| previous.extracted.clone(),
        ),
        io_gate: reuse.map_or_else(
            || Arc::new(Mutex::new(())),
            |previous| previous.io_gate.clone(),
        ),
    };
    cache_mapping(&mut archive, &project)?;
    let archive = Arc::new(archive);
    resolve_project(
        project,
        archive,
        path.parent().context("Project path has no parent")?,
    )
}

enum Input {
    Existing {
        archive: Arc<EmbeddedArchive>,
        blob: Blob,
    },
    File {
        path: PathBuf,
        identity: SourceIdentity,
        original_filename: String,
    },
    /// Bytes generated in memory (painted mask PNGs); they have no source file
    /// to reuse, so identical existing blobs are matched by key and hash.
    Bytes {
        bytes: Vec<u8>,
        original_filename: String,
    },
}
struct Plan {
    archive: Option<Arc<EmbeddedArchive>>,
    inputs: BTreeMap<String, Input>,
    dedup: BTreeMap<PathBuf, String>,
}
impl Plan {
    fn add(&mut self, path: &Path, key: String) -> Result<PathBuf> {
        if let Some(existing) = self.dedup.get(path) {
            return Ok(reference(existing));
        }
        ensure!(safe_key(&key), "Unsafe calculated filename");
        let embedded = self.archive.as_ref().and_then(|archive| {
            archive.key(path).and_then(|old| {
                archive
                    .blobs
                    .get(&old)
                    .map(|blob| (archive.clone(), blob.clone()))
            })
        });
        let input = if let Some((archive, blob)) = embedded {
            let unchanged = !path.exists()
                || SourceIdentity::read(path).is_ok_and(|identity| {
                    let extracted = archive
                        .extracted
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    extracted.get(path).map_or(
                        identity.bytes == blob.length
                            && identity.modified_unix_ns == blob.modified_unix_ns,
                        |verified| *verified == identity,
                    )
                });
            if unchanged {
                Input::Existing { archive, blob }
            } else {
                Self::file(path)?
            }
        } else {
            Self::file(path)?
        };
        ensure!(
            !self.inputs.contains_key(&key),
            "Duplicate media storage key"
        );
        self.inputs.insert(key.clone(), input);
        self.dedup.insert(path.to_path_buf(), key.clone());
        Ok(reference(&key))
    }
    fn file(path: &Path) -> Result<Input> {
        Ok(Input::File {
            path: path.to_path_buf(),
            identity: SourceIdentity::read(path)
                .with_context(|| format!("Reading media to embed: {}", path.display()))?,
            original_filename: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
        })
    }
    /// Queue in-memory generated bytes under a deterministic storage key.
    ///
    /// Identical bytes already stored under the same key are reused: the
    /// append-only save path then re-lists the existing blob instead of
    /// appending another copy, so repeated saves of unchanged painted masks do
    /// not grow the container.
    fn add_bytes(&mut self, key: String, bytes: Vec<u8>, original_filename: String) -> Result<()> {
        ensure!(safe_key(&key), "Unsafe generated media storage key");
        ensure!(
            !self.inputs.contains_key(&key),
            "Duplicate media storage key"
        );
        let existing = self.archive.as_ref().and_then(|archive| {
            archive
                .blobs
                .get(&key)
                .filter(|blob| {
                    blob.length == bytes.len() as u64 && blob.sha256 == exact_hash(&bytes)
                })
                .cloned()
                .map(|blob| (archive.clone(), blob))
        });
        let input = match existing {
            Some((archive, blob)) => Input::Existing { archive, blob },
            None => Input::Bytes {
                bytes,
                original_filename,
            },
        };
        self.inputs.insert(key, input);
        Ok(())
    }
    fn mask_files(&self, directory: &Path) -> Result<BTreeMap<String, PathBuf>> {
        let mut files = BTreeMap::new();
        if let Some(archive) = &self.archive {
            for (key, _) in archive.matches(directory) {
                let path = archive.cache_paths[&key].clone();
                let name = path
                    .strip_prefix(directory)?
                    .to_str()
                    .context("Mask filename is not Unicode")?;
                ensure!(
                    safe_component(name),
                    "Mask batches must contain flat filenames"
                );
                files.insert(name.to_owned(), path);
            }
        }
        if directory.is_dir() {
            for entry in fs::read_dir(directory)? {
                let entry = entry?;
                ensure!(
                    entry.file_type()?.is_file(),
                    "Mask batches must contain regular files only"
                );
                let name = entry.file_name().to_string_lossy().into_owned();
                ensure!(safe_component(&name), "Unsafe mask filename");
                files.insert(name, entry.path());
            }
        }
        ensure!(
            !files.is_empty(),
            "Referenced mask batch is missing or empty"
        );
        Ok(files)
    }
}
fn relative_external(path: &Path, base: &Path) -> PathBuf {
    if path.is_relative() {
        return path.to_path_buf();
    }
    let a: Vec<_> = path.components().collect();
    let b: Vec<_> = base.components().collect();
    let common = a.iter().zip(&b).take_while(|(a, b)| a == b).count();
    if common == 0 || matches!(a.first(), Some(Component::Prefix(_))) && a.first() != b.first() {
        return path.to_path_buf();
    }
    let mut result = PathBuf::new();
    for component in &b[common..] {
        if matches!(component, Component::Normal(_)) {
            result.push("..");
        }
    }
    for component in &a[common..] {
        result.push(component.as_os_str());
    }
    result
}
fn build_plan(project: &mut ChromaProject, embed_sources: bool, base: &Path) -> Result<Plan> {
    validate_documents(project)?;
    let mut plan = Plan {
        archive: project.archive.clone(),
        inputs: BTreeMap::new(),
        dedup: BTreeMap::new(),
    };
    for document in &mut project.documents {
        let prefix = format!("documents/{}", document.key);
        let mut directories = BTreeMap::<PathBuf, PathBuf>::new();
        for (index, batch) in document.masks.iter_mut().enumerate() {
            if let Some(stored) = directories.get(&batch.directory) {
                batch.directory = stored.clone();
                continue;
            }
            let key = format!("{prefix}/masks/{index}");
            for (name, file) in plan.mask_files(&batch.directory)? {
                plan.add(&file, format!("{key}/{name}"))?;
            }
            let stored = reference(&key);
            directories.insert(batch.directory.clone(), stored.clone());
            batch.directory = stored;
        }
        for (frame, files) in &mut document.frames {
            for (name, path) in [
                ("rgba", &mut files.rgba),
                ("checker", &mut files.checker),
                ("alpha", &mut files.alpha),
                ("mask", &mut files.mask),
            ] {
                if let Some(source) = path {
                    *source = plan.add(source, format!("{prefix}/frames/{frame}/{name}.png"))?;
                }
            }
            for (layer, source) in &mut files.sam_layers {
                *source = plan.add(
                    source,
                    format!("{prefix}/frames/{frame}/sam-layer-{layer}.png"),
                )?;
            }
        }
        if let Some(media) = &mut document.project.media {
            if embed_sources
                || plan
                    .archive
                    .as_ref()
                    .is_some_and(|archive| archive.is_embedded(&media.path))
            {
                if media.path.is_file() {
                    if let Some(archive) = plan
                        .archive
                        .as_ref()
                        .filter(|archive| archive.is_embedded(&media.path))
                    {
                        media.source_identity = archive.materialized_identity(&media.path)?;
                    } else if media.source_identity.is_some() {
                        media.verify_source_unchanged()?;
                    } else {
                        media.source_identity = Some(SourceIdentity::read(&media.path)?);
                    }
                }
                media.path = plan.add(
                    &media.path,
                    format!("{prefix}/source/{}", clean_filename(&media.path)),
                )?;
            } else {
                media.path = relative_external(&media.path, base);
            }
        }
        if let Some(hint) = &mut document.hint {
            if embed_sources
                || plan
                    .archive
                    .as_ref()
                    .is_some_and(|archive| archive.is_embedded(&hint.path))
            {
                hint.path = plan.add(
                    &hint.path,
                    format!("{prefix}/hint/{}", clean_filename(&hint.path)),
                )?;
            } else {
                hint.path = relative_external(&hint.path, base);
            }
        }
        // Painted mask layers: one deterministic grayscale PNG blob per raster
        // keyframe. Taking the decoded rasters after encoding keeps peak memory
        // bounded; the save reload below restores them from the container.
        {
            let source_size = document.project.mask_raster_size();
            for kind in [MaskKind::Garbage, MaskKind::Despill] {
                for layer in document.project.mask_layers.layers_mut(kind) {
                    if layer.doc.frames.is_empty() {
                        layer.doc.width = layer.width;
                        layer.doc.height = layer.height;
                        continue;
                    }
                    let (width, height) = match (layer.width, layer.height) {
                        (width, height) if width > 0 && height > 0 => (width, height),
                        _ => source_size,
                    };
                    ensure!(
                        width > 0 && height > 0,
                        "Painted mask layer has no raster resolution"
                    );
                    layer.set_raster_size(width, height);
                    let frames = std::mem::take(&mut layer.doc.frames);
                    for (frame, mask) in frames {
                        ensure!(
                            mask.width() == width && mask.height() == height,
                            "Painted mask raster does not match its layer resolution"
                        );
                        let bytes = encode_mask_png(&mask)?;
                        plan.add_bytes(
                            mask_blob_key(&document.key, kind, layer.layer_id, frame),
                            bytes,
                            format!("{frame:06}.png"),
                        )?;
                    }
                }
            }
        }
        document.project.output_dir = PathBuf::new();
    }
    project.archive = None;
    Ok(plan)
}
fn write_blob(output: &mut File, input: &Input, reuse: bool) -> Result<Blob> {
    if let Input::Existing { blob, .. } = input
        && reuse
    {
        return Ok(blob.clone());
    }
    let offset = output.stream_position()?;
    match input {
        Input::Existing { archive, blob } => {
            archive.stream_blob(blob, output)?;
            Ok(Blob {
                offset,
                ..blob.clone()
            })
        }
        Input::File {
            path,
            identity,
            original_filename,
        } => {
            let mut source = File::open(path)?;
            let mut buffer = vec![0; 1024 * 1024];
            let mut hash = Sha256::new();
            let mut length = 0_u64;
            loop {
                let count = source.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                output.write_all(&buffer[..count])?;
                hash.update(&buffer[..count]);
                length = length
                    .checked_add(count as u64)
                    .context("Embedded length overflow")?;
            }
            ensure!(
                length == identity.bytes && SourceIdentity::read(path)? == *identity,
                "Media changed while embedding"
            );
            Ok(Blob {
                offset,
                length,
                sha256: hash.finalize().into(),
                modified_unix_ns: identity.modified_unix_ns,
                original_filename: original_filename.clone(),
            })
        }
        Input::Bytes {
            bytes,
            original_filename,
        } => {
            output.write_all(bytes)?;
            Ok(Blob {
                offset,
                length: bytes.len() as u64,
                sha256: exact_hash(bytes),
                modified_unix_ns: 0,
                original_filename: original_filename.clone(),
            })
        }
    }
}
fn write_container(
    file: &mut File,
    project: ChromaProject,
    plan: &Plan,
    previous: u64,
    reuse: bool,
) -> Result<()> {
    if previous == 0 {
        let mut header = [0; HEADER_SIZE as usize];
        header[..16].copy_from_slice(MAGIC);
        header[16..24].copy_from_slice(&1_u64.to_le_bytes());
        file.write_all(&header)?;
    }
    file.seek(SeekFrom::End(0))?;
    let mut blobs = BTreeMap::new();
    for (key, input) in &plan.inputs {
        blobs.insert(key.clone(), write_blob(file, input, reuse)?);
    }
    let offset = file.stream_position()?;
    let mut index = Index {
        container_version: 1,
        project,
        blobs,
    };
    let sequence = previous.checked_add(1).context("Save sequence overflow")?;
    let mut commit = Commit {
        sequence,
        offset,
        length: 1,
        hash: [0; 32],
    };
    validate_index(&mut index, commit)?;
    let bytes = serde_json::to_vec(&index)?;
    ensure!(
        bytes.len() as u64 <= MAX_INDEX_SIZE,
        "Project index exceeds 128MiB limit"
    );
    commit.length = bytes.len() as u64;
    commit.hash = exact_hash(&bytes);
    file.write_all(&bytes)?;
    file.sync_all()?;
    file.seek(SeekFrom::Start(SLOT_OFFSETS[((sequence - 1) % 2) as usize]))?;
    file.write_all(&encode_commit(commit))?;
    file.sync_all()?;
    Ok(())
}
/// Appends only new media/index data on Save. Save As streams a compact new
/// container directly from old blobs, without extracting the existing project.
pub fn save(path: &Path, mut project: ChromaProject, embed_sources: bool) -> Result<ChromaProject> {
    let path = absolute(path)?;
    let base = path.parent().context("Project destination has no parent")?;
    ensure!(
        base.is_dir(),
        "Project destination directory does not exist"
    );
    let archive = project.archive.clone();
    let append = archive.as_ref().is_some_and(|archive| {
        archive.sequence > 0
            && fs::canonicalize(&path)
                .ok()
                .zip(fs::canonicalize(&archive.path).ok())
                .is_some_and(|(a, b)| a == b)
    });
    let plan = build_plan(&mut project, embed_sources, base)?;
    if append {
        let _gate = archive
            .as_ref()
            .expect("append archive")
            .io_gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        file.lock()
            .context("Locking project for incremental save")?;
        let (current, _) = read_committed(&mut file)?;
        ensure!(
            archive
                .as_ref()
                .is_some_and(|archive| archive.sequence == current.sequence
                    && archive.commit_hash == current.hash),
            "Project changed on disk; reopen before saving"
        );
        let previous_length = file.metadata()?.len();
        if let Err(error) = write_container(&mut file, project, &plan, current.sequence, true) {
            let _ = file.set_len(previous_length);
            let _ = file.sync_all();
            return Err(error);
        }
    } else {
        let staging = base.join(format!(".chroma-save-{}.tmp", unique_key()));
        let result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .write(true)
                .read(true)
                .create_new(true)
                .open(&staging)?;
            write_container(&mut file, project, &plan, 0, false)?;
            drop(file);
            fs::rename(&staging, &path).context("Publishing portable .chroma project")?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&staging);
        }
        result?;
    }
    load_reusing(&path, archive.as_ref())
}
pub fn source_metadata_compatible(old: &MediaInfo, new: &MediaInfo) -> bool {
    old.width == new.width
        && old.height == new.height
        && old.frame_count == new.frame_count
        && old.fps == new.fps
}

#[cfg(test)]
mod tests {
    use super::*;
    use ck_desktop_core::Rational;

    fn fixture(root: &Path) -> ChromaProject {
        let source = root.join("input/source.mov");
        let hint = root.join("input/hint.mov");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        fs::write(&source, b"original source bytes").unwrap();
        fs::write(&hint, b"original hint bytes").unwrap();
        let media = MediaInfo {
            path: source.clone(),
            width: 16,
            height: 16,
            frame_count: 3,
            fps: Rational { num: 30, den: 1 },
            time_base: Rational { num: 1, den: 30 },
            first_pts: 0,
            has_b_frames: false,
            source_identity: Some(SourceIdentity::read(&source).unwrap()),
        };
        let calculations = root.join("calculated");
        let mut files = FrameFiles::default();
        for (name, slot) in [
            ("frames", &mut files.rgba),
            ("checker", &mut files.checker),
            ("alpha", &mut files.alpha),
            ("masks", &mut files.mask),
        ] {
            let path = calculations.join(name).join("000001.png");
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, format!("{name} calculated bytes")).unwrap();
            *slot = Some(path);
        }
        let mask_directory = files.mask.as_ref().unwrap().parent().unwrap().to_path_buf();
        fs::write(mask_directory.join("000000.png"), b"first mask").unwrap();
        fs::write(mask_directory.join("000002.png"), b"last mask").unwrap();
        ChromaProject {
            chroma_version: 1,
            selected: 0,
            archive: None,
            documents: vec![Document {
                key: "media-a".into(),
                project: Project::new(media, calculations),
                range: FrameRange {
                    start: 0,
                    end_exclusive: 3,
                },
                playhead: 0,
                frames: BTreeMap::from([(1, files)]),
                masks: vec![MaskBatch {
                    range: FrameRange {
                        start: 0,
                        end_exclusive: 3,
                    },
                    directory: mask_directory,
                }],
                hint: Some(HintFootage {
                    path: hint,
                    first_frame_only: false,
                }),
            }],
        }
    }
    fn rgba(project: &ChromaProject) -> PathBuf {
        project.documents[0].frames[&1].rgba.clone().unwrap()
    }
    fn remove_calculations(root: &Path) {
        // Exact fixture-owned files, preserving originals/project containers.
        for (directory, names) in [
            ("frames", vec!["000001.png"]),
            ("checker", vec!["000001.png"]),
            ("alpha", vec!["000001.png"]),
            ("masks", vec!["000000.png", "000001.png", "000002.png"]),
        ] {
            for name in names {
                fs::remove_file(root.join("calculated").join(directory).join(name)).unwrap();
            }
        }
    }

    #[test]
    fn sam_layer_masks_are_embedded_and_restored() {
        let workspace = TemporaryWorkspace::new().unwrap();
        let mut project = fixture(&workspace.root);
        let layer_mask = workspace.root.join("calculated/sam-layer/000001.png");
        fs::create_dir_all(layer_mask.parent().unwrap()).unwrap();
        fs::write(&layer_mask, b"hat layer bytes").unwrap();
        project.documents[0]
            .frames
            .get_mut(&1)
            .unwrap()
            .sam_layers
            .insert(3, layer_mask.clone());
        let path = workspace.root.join("layers.chroma");
        drop(save(&path, project, false).unwrap());
        fs::remove_file(&layer_mask).unwrap();
        let loaded = load(&path).unwrap();
        let stored = loaded.documents[0].frames[&1].sam_layers[&3].clone();
        assert_ne!(stored, layer_mask);
        loaded
            .archive
            .as_ref()
            .unwrap()
            .materialize(std::slice::from_ref(&stored))
            .unwrap();
        assert_eq!(fs::read(&stored).unwrap(), b"hat layer bytes");
    }

    #[test]
    fn portable_container_moves_without_calculations_and_opens_without_extracting() {
        let original = TemporaryWorkspace::new().unwrap();
        let destination = TemporaryWorkspace::new().unwrap();
        let path = original.root.join("project.chroma");
        let saved = save(&path, fixture(&original.root), false).unwrap();
        assert_eq!(&fs::read(&path).unwrap()[..16], MAGIC);
        assert!(!rgba(&saved).exists());
        assert!(!saved.documents[0].project.output_dir.exists());
        drop(saved);
        remove_calculations(&original.root);
        let moved = destination.root.join("portable.chroma");
        fs::rename(path, &moved).unwrap();
        let loaded = load(&moved).unwrap();
        let archive = loaded.archive.as_ref().unwrap();
        let frame = rgba(&loaded);
        let batch = loaded.documents[0].masks[0].directory.clone();
        assert!(archive.needs_materialization(&frame));
        assert!(!frame.exists());
        assert!(!batch.exists());
        archive.materialize(std::slice::from_ref(&frame)).unwrap();
        assert_eq!(fs::read(&frame).unwrap(), b"frames calculated bytes");
        assert!(
            !batch.exists(),
            "one selected frame must not extract the mask batch"
        );
        archive.materialize(std::slice::from_ref(&batch)).unwrap();
        assert_eq!(fs::read(batch.join("000002.png")).unwrap(), b"last mask");
        assert!(!archive.needs_materialization(&batch));
        assert!(!original.root.join("project.chroma-data").exists());
    }

    #[test]
    fn embedded_sources_are_portable_keep_identity_and_remain_embedded_when_option_is_off() {
        let root = TemporaryWorkspace::new().unwrap();
        let path = root.root.join("project.chroma");
        let mut project = fixture(&root.root);
        project.documents[0]
            .project
            .media
            .as_mut()
            .unwrap()
            .source_identity = None;
        let saved = save(&path, project, true).unwrap();
        fs::remove_file(root.root.join("input/source.mov")).unwrap();
        fs::remove_file(root.root.join("input/hint.mov")).unwrap();
        let archive = saved.archive.as_ref().unwrap();
        let source = saved.documents[0].project.media.as_ref().unwrap();
        let hint = &saved.documents[0].hint.as_ref().unwrap().path;
        assert!(archive.is_embedded(&source.path));
        assert!(!source.path.exists());
        assert_eq!(archive.original_filename(&source.path), Some("source.mov"));
        archive
            .materialize(&[source.path.clone(), hint.clone()])
            .unwrap();
        source.verify_source_unchanged().unwrap();
        assert_eq!(
            archive.materialized_identity(&source.path).unwrap(),
            source.source_identity
        );
        assert_eq!(fs::read(hint).unwrap(), b"original hint bytes");
        let source_path = source.path.clone();
        let old_size = fs::metadata(&path).unwrap().len();
        let again = save(&path, saved, false).unwrap();
        let current = again.documents[0].project.media.as_ref().unwrap();
        assert_eq!(current.path, source_path);
        assert!(current.path.exists());
        assert!(again.archive.as_ref().unwrap().is_embedded(&current.path));
        assert!(fs::metadata(path).unwrap().len() - old_size < 16 * 1024);
    }

    #[test]
    fn external_sources_remain_relative_and_are_not_read_when_option_is_off() {
        let root = TemporaryWorkspace::new().unwrap();
        let path = root.root.join("project.chroma");
        let project = fixture(&root.root);
        fs::remove_file(root.root.join("input/source.mov")).unwrap();
        let saved = save(&path, project, false).unwrap();
        let media = saved.documents[0].project.media.as_ref().unwrap();
        assert!(!saved.archive.as_ref().unwrap().is_embedded(&media.path));
        assert!(!media.path.exists());
        let (_, index) = read_committed(&mut File::open(path).unwrap()).unwrap();
        assert_eq!(
            index.project.documents[0]
                .project
                .media
                .as_ref()
                .unwrap()
                .path,
            PathBuf::from("input/source.mov")
        );
    }

    #[test]
    fn save_as_streams_unmaterialized_blobs_and_retains_cached_source_paths() {
        let root = TemporaryWorkspace::new().unwrap();
        let other = TemporaryWorkspace::new().unwrap();
        let original_path = root.root.join("one.chroma");
        let copy_path = other.root.join("two.chroma");
        let first = save(&original_path, fixture(&root.root), true).unwrap();
        let frame = rgba(&first);
        assert!(!frame.exists());
        let source = first.documents[0]
            .project
            .media
            .as_ref()
            .unwrap()
            .path
            .clone();
        first
            .archive
            .as_ref()
            .unwrap()
            .materialize(std::slice::from_ref(&source))
            .unwrap();
        remove_calculations(&root.root);
        let second = save(&copy_path, first, false).unwrap();
        assert_eq!(rgba(&second), frame);
        assert!(!frame.exists());
        assert_eq!(
            second.documents[0].project.media.as_ref().unwrap().path,
            source
        );
        assert!(source.exists());
        drop(second);
        fs::remove_file(original_path).unwrap();
        let reopened = load(&copy_path).unwrap();
        let fresh_frame = rgba(&reopened);
        reopened
            .archive
            .as_ref()
            .unwrap()
            .materialize(std::slice::from_ref(&fresh_frame))
            .unwrap();
        assert_eq!(fs::read(fresh_frame).unwrap(), b"frames calculated bytes");
    }

    #[test]
    fn incremental_save_reuses_large_blob_and_recovers_previous_commit_after_torn_slot_or_index() {
        let root = TemporaryWorkspace::new().unwrap();
        let path = root.root.join("project.chroma");
        let project = fixture(&root.root);
        fs::write(
            project.documents[0].frames[&1].rgba.as_ref().unwrap(),
            vec![17; 1024 * 1024],
        )
        .unwrap();
        let mut first = save(&path, project, false).unwrap();
        let first_size = fs::metadata(&path).unwrap().len();
        let frame = rgba(&first);
        first
            .archive
            .as_ref()
            .unwrap()
            .materialize(std::slice::from_ref(&frame))
            .unwrap();
        first.documents[0].playhead = 2;
        let second = save(&path, first, false).unwrap();
        assert_eq!(rgba(&second), frame);
        assert!(frame.exists());
        assert!(fs::metadata(&path).unwrap().len() - first_size < 16 * 1024);
        let (commit, _) = read_committed(&mut File::open(&path).unwrap()).unwrap();
        assert_eq!(commit.sequence, 2);
        for (name, offset) in [("slot", SLOT_OFFSETS[1]), ("index", commit.offset)] {
            let corrupted = root.root.join(format!("{name}.chroma"));
            fs::copy(&path, &corrupted).unwrap();
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&corrupted)
                .unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            let mut byte = [0];
            file.read_exact(&mut byte).unwrap();
            byte[0] ^= 0xff;
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&byte).unwrap();
            drop(file);
            let recovered = load(&corrupted).unwrap();
            assert_eq!(recovered.documents[0].playhead, 0);
        }
    }

    #[test]
    fn lazy_blob_corruption_is_detected_before_publishing_selected_file() {
        let root = TemporaryWorkspace::new().unwrap();
        let path = root.root.join("project.chroma");
        let saved = save(&path, fixture(&root.root), false).unwrap();
        drop(saved);
        let (_, index) = read_committed(&mut File::open(&path).unwrap()).unwrap();
        let blob = index.blobs.values().next().unwrap();
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(blob.offset)).unwrap();
        file.write_all(&[0xff]).unwrap();
        drop(file);
        let loaded = load(&path).unwrap();
        let archive = loaded.archive.as_ref().unwrap();
        let key = index.blobs.keys().next().unwrap();
        let target = archive.cache_paths[key].clone();
        assert!(!target.exists());
        assert!(archive.materialize(std::slice::from_ref(&target)).is_err());
        assert!(!target.exists());
        let copy = root.root.join("corrupt-copy.chroma");
        assert!(save(&copy, loaded, false).is_err());
        assert!(!copy.exists());
    }

    #[test]
    fn repeated_mask_batches_share_contained_media_and_legacy_json_import_still_works() {
        let root = TemporaryWorkspace::new().unwrap();
        let mut project = fixture(&root.root);
        let repeated = project.documents[0].masks[0].clone();
        project.documents[0].masks.push(repeated);
        let legacy = root.root.join("legacy.chroma");
        fs::write(&legacy, serde_json::to_vec(&project).unwrap()).unwrap();
        let imported = load(&legacy).unwrap();
        assert_eq!(imported.documents[0].masks.len(), 2);
        let saved = save(&root.root.join("binary.chroma"), imported, false).unwrap();
        assert_eq!(
            saved.documents[0].masks[0].directory,
            saved.documents[0].masks[1].directory
        );
        let native = root.root.join("legacy.brpnative");
        project.documents[0].project.save(&native).unwrap();
        assert_eq!(load(&native).unwrap().documents.len(), 1);
    }

    #[test]
    fn unsafe_keys_overflow_and_file_directory_aliases_are_rejected() {
        for key in [
            "../escape",
            "a/../b",
            "a\\b",
            "C:/source",
            "a//b",
            "a/CON.txt",
            "a/trailing.",
            "unicode/é",
        ] {
            assert!(!safe_key(key), "{key}");
        }
        let root = TemporaryWorkspace::new().unwrap();
        let path = root.root.join("project.chroma");
        let saved = save(&path, fixture(&root.root), false).unwrap();
        drop(saved);
        let (commit, mut index) = read_committed(&mut File::open(path).unwrap()).unwrap();
        let blob = index.blobs.values().next().unwrap().clone();
        index.blobs.insert("alias".into(), blob.clone());
        index.blobs.insert("alias/child".into(), blob.clone());
        assert!(validate_index(&mut index, commit).is_err());
        index.blobs.remove("alias");
        index.blobs.remove("alias/child");
        index.blobs.insert(
            "overflow".into(),
            Blob {
                offset: u64::MAX,
                length: 2,
                ..blob
            },
        );
        assert!(validate_index(&mut index, commit).is_err());
    }

    #[test]
    fn embedding_rejects_changed_source_and_preserves_previous_project_file() {
        let root = TemporaryWorkspace::new().unwrap();
        let project = fixture(&root.root);
        let destination = root.root.join("existing.chroma");
        fs::write(&destination, b"previous user project").unwrap();
        fs::write(
            &project.documents[0].project.media.as_ref().unwrap().path,
            b"changed",
        )
        .unwrap();
        assert!(save(&destination, project, true).is_err());
        assert_eq!(fs::read(destination).unwrap(), b"previous user project");
    }

    fn painted_mask(width: u32, height: u32, seed: u8) -> Mask {
        let mut mask = Mask::new(width, height);
        for (index, value) in mask.as_mut_slice().iter_mut().enumerate() {
            *value = if (index as u8).wrapping_add(seed).is_multiple_of(7) {
                255
            } else {
                seed
            };
        }
        mask
    }

    /// Add one painted garbage keyframe and one painted despill keyframe.
    fn add_painted_masks(document: &mut Document) {
        let (width, height) = document.project.mask_raster_size();
        let mut stacks = document.project.mask_stacks();
        let garbage = stacks.garbage.add(0, None);
        garbage.set_scene_range(0, 2);
        garbage.set_mask(1, &painted_mask(width, height, 41));
        let despill = stacks.despill.add(0, None);
        despill.set_mask(0, &painted_mask(width, height, 23));
        document.project.replace_mask_stacks(stacks);
    }

    fn mask_blob_keys(index: &Index) -> Vec<String> {
        index
            .blobs
            .keys()
            .filter(|key| key.contains("/mask-layers/"))
            .cloned()
            .collect()
    }

    #[test]
    fn painted_mask_layers_round_trip_through_embedded_blobs() {
        let root = TemporaryWorkspace::new().unwrap();
        let destination = TemporaryWorkspace::new().unwrap();
        let path = root.root.join("masks.chroma");
        let mut project = fixture(&root.root);
        add_painted_masks(&mut project.documents[0]);
        let saved = save(&path, project, false).unwrap();
        assert_eq!(&fs::read(&path).unwrap()[..16], MAGIC);
        // The container holds both painted keyframes as grayscale PNG blobs.
        let (_, index) = read_committed(&mut File::open(&path).unwrap()).unwrap();
        let keys = mask_blob_keys(&index);
        assert_eq!(keys.len(), 2, "{keys:?}");
        assert!(
            keys.iter()
                .any(|key| key.ends_with("/garbage/0001/000001.png")),
            "{keys:?}"
        );
        assert!(
            keys.iter()
                .any(|key| key.ends_with("/despill/0001/000000.png")),
            "{keys:?}"
        );
        // Layer metadata is in the JSON; the pixels are only in the blobs.
        let stored = &index.project.documents[0].project.mask_layers;
        assert_eq!(stored.garbage.len(), 1);
        assert_eq!(stored.garbage[0].layer_id, 1);
        assert!(stored.garbage[0].doc.frames.is_empty());
        // Opening the saved container restores the painted masks eagerly while
        // calculated RGBA frames stay lazy.
        let restored = saved.documents[0].project.mask_stacks();
        assert_eq!(
            restored.garbage.layers[0].doc.get(1).unwrap().as_slice(),
            painted_mask(16, 16, 41).as_slice()
        );
        assert_eq!(
            restored.despill.layers[0].doc.get(0).unwrap().as_slice(),
            painted_mask(16, 16, 23).as_slice()
        );
        assert!(
            !saved.documents[0].frames[&1]
                .rgba
                .as_ref()
                .unwrap()
                .exists()
        );
        // Masks move with the self-contained container.
        drop(saved);
        let moved = destination.root.join("moved.chroma");
        fs::rename(&path, &moved).unwrap();
        let reopened = load(&moved).unwrap();
        let stacks = reopened.documents[0].project.mask_stacks();
        assert_eq!(
            stacks.garbage.layers[0].doc.get(1).unwrap().as_slice(),
            painted_mask(16, 16, 41).as_slice()
        );
        assert_eq!(
            stacks.despill.layers[0].doc.get(0).unwrap().as_slice(),
            painted_mask(16, 16, 23).as_slice()
        );
    }

    #[test]
    fn incremental_save_reuses_unchanged_painted_mask_blobs_and_save_as_keeps_them() {
        let root = TemporaryWorkspace::new().unwrap();
        let other = TemporaryWorkspace::new().unwrap();
        let path = root.root.join("masks.chroma");
        let mut project = fixture(&root.root);
        add_painted_masks(&mut project.documents[0]);
        let mut first = save(&path, project, false).unwrap();
        let first_size = fs::metadata(&path).unwrap().len();
        let (commit_before, index_before) =
            read_committed(&mut File::open(&path).unwrap()).unwrap();
        let keys = mask_blob_keys(&index_before);
        assert_eq!(keys.len(), 2);
        first.documents[0].playhead = 2;
        let second = save(&path, first, false).unwrap();
        assert_eq!(second.documents[0].playhead, 2);
        assert!(
            fs::metadata(&path).unwrap().len() - first_size < 16 * 1024,
            "unchanged painted mask blobs must be reused by the append-only save"
        );
        let (commit_after, index_after) = read_committed(&mut File::open(&path).unwrap()).unwrap();
        assert_eq!(commit_after.sequence, commit_before.sequence + 1);
        for key in &keys {
            assert_eq!(
                index_after.blobs[key].offset,
                index_before.blobs[key].offset
            );
            assert_eq!(
                index_after.blobs[key].sha256,
                index_before.blobs[key].sha256
            );
        }
        // Save As streams the reused painted-mask blobs into a compact copy.
        let copy_path = other.root.join("copy.chroma");
        let copied = save(&copy_path, second, false).unwrap();
        let stacks = copied.documents[0].project.mask_stacks();
        assert_eq!(
            stacks.garbage.layers[0].doc.get(1).unwrap().as_slice(),
            painted_mask(16, 16, 41).as_slice()
        );
        let (_, copy_index) = read_committed(&mut File::open(&copy_path).unwrap()).unwrap();
        assert_eq!(mask_blob_keys(&copy_index).len(), 2);
        // Editing one painted key stores a replacement blob and persists it.
        let mut edited = copied;
        edited.documents[0].project.mask_layers.garbage[0]
            .doc
            .frames
            .insert(1, painted_mask(16, 16, 59));
        let edited_path = other.root.join("edited.chroma");
        let edited = save(&edited_path, edited, false).unwrap();
        let stacks = edited.documents[0].project.mask_stacks();
        assert_eq!(
            stacks.garbage.layers[0].doc.get(1).unwrap().as_slice(),
            painted_mask(16, 16, 59).as_slice()
        );
    }

    #[test]
    fn painted_mask_blobs_are_validated_pngs_with_the_layer_resolution() {
        let encoded = encode_mask_png(&painted_mask(4, 3, 9)).unwrap();
        assert_eq!(&encoded[1..4], b"PNG", "mask blobs are PNG encoded");
        let decoded = decode_mask_png(&encoded, 4, 3).unwrap();
        assert_eq!(decoded.as_slice(), painted_mask(4, 3, 9).as_slice());
        let error = decode_mask_png(&encoded, 8, 6).unwrap_err().to_string();
        assert!(error.contains("resolution"), "{error}");
        assert!(decode_mask_png(b"not a PNG", 4, 3).is_err());
    }

    #[test]
    fn corrupt_painted_mask_blob_fails_the_project_open() {
        let root = TemporaryWorkspace::new().unwrap();
        let path = root.root.join("masks.chroma");
        let mut project = fixture(&root.root);
        add_painted_masks(&mut project.documents[0]);
        let saved = save(&path, project, false).unwrap();
        drop(saved);
        let (_, index) = read_committed(&mut File::open(&path).unwrap()).unwrap();
        let key = mask_blob_keys(&index).into_iter().next().unwrap();
        let blob = &index.blobs[&key];
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(blob.offset + blob.length / 2))
            .unwrap();
        file.write_all(&[0xff]).unwrap();
        drop(file);
        let error = load(&path).unwrap_err().to_string();
        assert!(error.contains("checksum"), "{error}");
    }

    #[test]
    fn older_projects_without_mask_blobs_load_with_empty_stacks() {
        let root = TemporaryWorkspace::new().unwrap();
        let legacy = root.root.join("legacy.brpnative");
        fs::write(
            &legacy,
            r#"{
                "schema_version": 1,
                "media": {
                    "path": "input/source.mov", "width": 16, "height": 16, "frame_count": 3,
                    "fps": {"num": 30, "den": 1}, "time_base": {"num": 1, "den": 30},
                    "first_pts": 0, "has_b_frames": false
                },
                "scenes": [{"index": 0, "start": 0, "end_exclusive": 3, "enabled": true}],
                "prompts": [],
                "output_dir": "renders"
            }"#,
        )
        .unwrap();
        let loaded = load(&legacy).unwrap();
        assert_eq!(loaded.documents.len(), 1);
        let project = &loaded.documents[0].project;
        assert_eq!(
            project.schema_version,
            ck_desktop_core::PROJECT_SCHEMA_VERSION
        );
        assert!(project.mask_layers.is_empty());
        assert!(project.mask_stacks().is_empty());
        // Saving upgrades the document to the current schema without mask blobs.
        let binary = root.root.join("upgraded.chroma");
        let saved = save(&binary, loaded, false).unwrap();
        assert!(saved.documents[0].project.mask_layers.is_empty());
        let (_, index) = read_committed(&mut File::open(&binary).unwrap()).unwrap();
        assert_eq!(
            index.project.documents[0].project.schema_version,
            ck_desktop_core::PROJECT_SCHEMA_VERSION
        );
        assert!(!index.blobs.keys().any(|key| key.contains("/mask-layers/")));
    }

    #[test]
    fn mask_blob_keys_must_reference_an_existing_document_and_layer() {
        let root = TemporaryWorkspace::new().unwrap();
        let path = root.root.join("masks.chroma");
        let mut project = fixture(&root.root);
        add_painted_masks(&mut project.documents[0]);
        let saved = save(&path, project, false).unwrap();
        drop(saved);
        let (commit, index) = read_committed(&mut File::open(&path).unwrap()).unwrap();
        let blob = index.blobs.values().next().unwrap().clone();
        for (key, expected) in [
            (
                "documents/media-z/mask-layers/garbage/0001/000000.png",
                "missing project document",
            ),
            (
                "documents/media-a/mask-layers/despill/0007/000000.png",
                "missing mask layer",
            ),
            (
                "documents/media-a/mask-layers/garbage/bogus/000000.png",
                "Malformed painted-mask blob key",
            ),
        ] {
            let mut mutated = Index {
                container_version: index.container_version,
                project: serde_json::from_slice(&serde_json::to_vec(&index.project).unwrap())
                    .unwrap(),
                blobs: index.blobs.clone(),
            };
            mutated.blobs.insert(key.into(), blob.clone());
            let error = validate_index(&mut mutated, commit)
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{key}: {error}");
        }
    }
}
