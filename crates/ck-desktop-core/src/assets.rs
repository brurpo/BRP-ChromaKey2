//! Verified application-folder assets. The caller owns the background thread and consent UI.
use anyhow::{Context, Result, bail, ensure};
use reqwest::{StatusCode, Url, blocking::Client, header};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, SystemTime},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetGroup {
    Core,
    Runtime,
    Ck1080,
    Sam,
    Transnet,
    Ck4k,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssetFile {
    pub group: AssetGroup,
    pub path: PathBuf,
    #[serde(default)]
    pub url: String,
    pub size: u64,
    pub sha256: String,
    /// Native app files ship in the archive; they are never repaired from a
    /// model host. Missing or modified bytes require re-extracting the app.
    #[serde(default, skip_serializing_if = "is_false")]
    pub bundled_only: bool,
}

fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AssetManifest {
    pub schema_version: u32,
    pub id: String,
    pub version: String,
    pub files: Vec<AssetFile>,
}

#[derive(Clone, Debug)]
pub struct AssetStatus {
    pub ready: bool,
    pub missing: Vec<AssetFile>,
    pub verified_bytes: u64,
    pub total_bytes: u64,
    /// Bytes needed from catalog sources, excluding verified shared files.
    pub download_bytes: u64,
    pub reusable_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetPhase {
    Checking,
    Downloading,
    Verifying,
    Activating,
}

#[derive(Clone, Debug)]
pub struct AssetProgress {
    pub group: AssetGroup,
    pub path: PathBuf,
    pub phase: AssetPhase,
    /// Bytes of the current file, including a valid resumable prefix.
    pub downloaded: u64,
    pub total: u64,
    pub file_index: usize,
    pub file_count: usize,
}

#[derive(Clone, Debug, Default)]
pub struct AssetCancellation(Arc<AtomicBool>);
impl AssetCancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    fn check(&self) -> Result<()> {
        ensure!(!self.is_cancelled(), "Asset installation cancelled");
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Identity {
    size: u64,
    modified: SystemTime,
}
fn identity(path: &Path) -> Result<Identity> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "Asset must be a regular file: {}",
        path.display()
    );
    Ok(Identity {
        size: metadata.len(),
        modified: metadata.modified()?,
    })
}

/// Inspection never contacts a server. Existing files are SHA256-checked once
/// per manager/session, then their length and modification time are rechecked.
/// A new session hashes again, so an untrusted persistent receipt cannot mark a
/// modified model ready. Keep this manager on the UI's asset worker thread.
pub struct AssetManager {
    root: PathBuf,
    manifest: AssetManifest,
    staging: PathBuf,
    verified: HashMap<PathBuf, Identity>,
    client: Option<Client>,
    huggingface_token: Option<String>,
}

impl AssetManager {
    pub fn new(app_dir: PathBuf, manifest: AssetManifest) -> Result<Self> {
        validate_manifest(&manifest)?;
        let root = if app_dir.is_absolute() {
            app_dir
        } else {
            std::env::current_dir()?.join(app_dir)
        };
        fs::create_dir_all(&root).context("Create the application asset folder")?;
        let root = root.canonicalize()?;
        let digest = format!("{:x}", Sha256::digest(serde_json::to_vec(&manifest)?));
        let staging = root.join(".asset-staging").join(digest);
        Ok(Self {
            root,
            manifest,
            staging,
            verified: HashMap::new(),
            client: None,
            huggingface_token: None,
        })
    }

    pub fn from_manifest_file(app_dir: PathBuf, manifest_path: impl AsRef<Path>) -> Result<Self> {
        let manifest =
            serde_json::from_slice(&fs::read(manifest_path.as_ref()).with_context(|| {
                format!("Read asset catalog {}", manifest_path.as_ref().display())
            })?)?;
        Self::new(app_dir, manifest)
    }
    pub fn manifest(&self) -> &AssetManifest {
        &self.manifest
    }
    pub fn app_dir(&self) -> &Path {
        &self.root
    }
    /// Optional existing local credential. Never persisted, formatted, or sent
    /// to a catalog-controlled host or cross-origin download redirect.
    pub fn with_huggingface_token(mut self, token: String) -> Self {
        self.huggingface_token = Some(token);
        self
    }

    fn selected(&self, groups: &[AssetGroup]) -> Result<Vec<AssetFile>> {
        ensure!(!groups.is_empty(), "Select at least one asset group");
        for group in groups {
            ensure!(
                self.manifest.files.iter().any(|file| file.group == *group),
                "The asset catalog does not contain the required {group:?} files"
            );
        }
        Ok(self
            .manifest
            .files
            .iter()
            .filter(|file| groups.contains(&file.group))
            .cloned()
            .collect())
    }

    pub fn inspect(&mut self, groups: &[AssetGroup]) -> Result<AssetStatus> {
        self.inspect_cancellable(groups, &AssetCancellation::default())
    }

    fn inspect_cancellable(
        &mut self,
        groups: &[AssetGroup],
        cancel: &AssetCancellation,
    ) -> Result<AssetStatus> {
        let files = self.selected(groups)?;
        let mut missing = Vec::new();
        let mut verified_bytes = 0;
        let total_bytes = files.iter().try_fold(0_u64, |sum, file| {
            sum.checked_add(file.size)
                .context("Asset catalog byte total overflow")
        })?;
        for file in files {
            cancel.check()?;
            let path = safe_destination(&self.root, &file.path)?;
            if !path.exists() {
                self.verified.remove(&path);
                missing.push(file);
                continue;
            }
            let before = identity(&path)?;
            let valid = before.size == file.size
                && (self.verified.get(&path) == Some(&before)
                    || verify_file(&path, &file, cancel).is_ok());
            cancel.check()?;
            if valid && identity(&path)? == before {
                self.verified.insert(path, before);
                verified_bytes += file.size;
            } else {
                self.verified.remove(&path);
                missing.push(file);
            }
        }
        let mut download_bytes = 0;
        let mut reusable_bytes = 0;
        let mut planned = HashSet::new();
        for file in &missing {
            if file.bundled_only {
                continue;
            }
            let key = (file.sha256.to_ascii_lowercase(), file.size);
            if self.find_reusable(file, cancel)?.is_some() || !planned.insert(key) {
                reusable_bytes += file.size;
            } else {
                download_bytes += file.size;
            }
        }
        Ok(AssetStatus {
            ready: missing.is_empty(),
            missing,
            verified_bytes,
            total_bytes,
            download_bytes,
            reusable_bytes,
        })
    }

    fn find_reusable(
        &mut self,
        wanted: &AssetFile,
        cancel: &AssetCancellation,
    ) -> Result<Option<PathBuf>> {
        let candidates: Vec<_> = self
            .manifest
            .files
            .iter()
            .filter(|file| {
                file.path != wanted.path
                    && file.size == wanted.size
                    && file.sha256.eq_ignore_ascii_case(&wanted.sha256)
            })
            .cloned()
            .collect();
        for file in candidates {
            cancel.check()?;
            let path = safe_destination(&self.root, &file.path)?;
            if !path.is_file() {
                continue;
            }
            let before = identity(&path)?;
            if before.size == wanted.size
                && (self.verified.get(&path) == Some(&before)
                    || verify_file(&path, wanted, cancel).is_ok())
                && identity(&path)? == before
            {
                self.verified.insert(path.clone(), before);
                return Ok(Some(path));
            }
            cancel.check()?;
        }
        Ok(None)
    }

    fn ensure_bundled_assets_present(&mut self, cancel: &AssetCancellation) -> Result<()> {
        let mut groups = Vec::new();
        for file in &self.manifest.files {
            if file.bundled_only && !groups.contains(&file.group) {
                groups.push(file.group);
            }
        }
        if groups.is_empty() {
            return Ok(());
        }
        let status = self.inspect_cancellable(&groups, cancel)?;
        if let Some(file) = status.missing.iter().find(|file| file.bundled_only) {
            bail!(
                "Reinstall app files: bundled native file {} is missing or modified. Re-extract the current application package into this folder; existing model files can be kept.",
                file.path.display()
            );
        }
        Ok(())
    }

    /// Call only after user consent. All replacements are staged and verified
    /// before activation; failed/cancelled downloads leave active packs intact.
    pub fn install(
        &mut self,
        groups: &[AssetGroup],
        cancel: &AssetCancellation,
        mut progress: impl FnMut(AssetProgress),
    ) -> Result<AssetStatus> {
        cancel.check()?;
        self.ensure_bundled_assets_present(cancel)?;
        let lock_path = safe_destination(&self.root, Path::new(".asset-staging/install.lock"))?;
        fs::create_dir_all(lock_path.parent().context("Asset lock has no parent")?)?;
        let install_lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(lock_path)?;
        install_lock
            .try_lock()
            .context("Another application instance is installing assets; retry when it finishes")?;
        self.recover_interrupted_activation(groups, cancel)?;
        let status = self.inspect_cancellable(groups, cancel)?;
        ensure!(
            !status.missing.iter().any(|file| file.bundled_only),
            "Reinstall app files: a bundled native file is missing or modified. Re-extract the current application package; existing model files can be kept."
        );
        if status.ready {
            return Ok(status);
        }
        safe_destination(&self.root, self.staging.strip_prefix(&self.root)?)?;
        fs::create_dir_all(&self.staging)?;
        let count = status.missing.len();
        let mut pending: Vec<(AssetFile, PathBuf)> = Vec::new();
        for (index, file) in status.missing.into_iter().enumerate() {
            cancel.check()?;
            let part_name = file.path.with_extension(format!(
                "{}.part",
                file.path
                    .extension()
                    .and_then(|value| value.to_str())
                    .unwrap_or("asset")
            ));
            let part = safe_destination(&self.staging, &part_name)?;
            if let Some(parent) = part.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut emit = |phase, downloaded| {
                progress(AssetProgress {
                    group: file.group,
                    path: file.path.clone(),
                    phase,
                    downloaded,
                    total: file.size,
                    file_index: index + 1,
                    file_count: count,
                })
            };
            emit(AssetPhase::Checking, 0);
            let complete = part.is_file() && verify_file(&part, &file, cancel).is_ok();
            if !complete {
                let reusable = pending
                    .iter()
                    .find(|(staged, _)| {
                        staged.size == file.size && staged.sha256.eq_ignore_ascii_case(&file.sha256)
                    })
                    .map(|(_, path)| path.clone())
                    .or(self.find_reusable(&file, cancel)?);
                if let Some(source) = reusable {
                    if part.exists() {
                        fs::remove_file(&part)?;
                    }
                    if fs::hard_link(&source, &part).is_err() {
                        let mut input = File::open(source)?;
                        let mut output = File::create(&part)?;
                        let mut offset = 0;
                        copy_body(
                            &mut input,
                            &mut output,
                            &mut offset,
                            file.size,
                            cancel,
                            &mut emit,
                        )?;
                        output.sync_all()?;
                    }
                    emit(AssetPhase::Downloading, file.size);
                } else {
                    if part.exists() && fs::metadata(&part)?.len() >= file.size {
                        fs::remove_file(&part)?;
                    }
                    self.download(&file, &part, cancel, &mut emit)
                        .with_context(|| format!("Obtain {}", file.path.display()))?;
                }
            }
            emit(AssetPhase::Verifying, file.size);
            if let Err(error) = verify_file(&part, &file, cancel) {
                if !cancel.is_cancelled() {
                    let _ = fs::remove_file(&part);
                }
                return Err(error).with_context(|| format!("Verify {}", file.path.display()));
            }
            pending.push((file, part));
        }
        cancel.check()?;
        let mut activated: Vec<(PathBuf, Option<PathBuf>, PathBuf)> = Vec::new();
        let result = (|| -> Result<()> {
            for (index, (file, part)) in pending.iter().enumerate() {
                cancel.check()?;
                let destination = safe_destination(&self.root, &file.path)?;
                if let Some(parent) = destination.parent() {
                    fs::create_dir_all(parent)?;
                }
                progress(AssetProgress {
                    group: file.group,
                    path: file.path.clone(),
                    phase: AssetPhase::Activating,
                    downloaded: file.size,
                    total: file.size,
                    file_index: index + 1,
                    file_count: count,
                });
                cancel.check()?;
                let backup = if destination.exists() {
                    let backup = part.with_extension("previous");
                    ensure!(
                        !backup.exists(),
                        "An interrupted activation backup needs recovery: {}",
                        backup.display()
                    );
                    fs::rename(&destination, &backup)?;
                    Some(backup)
                } else {
                    None
                };
                if let Err(error) = fs::rename(part, &destination) {
                    if let Some(backup) = &backup {
                        fs::rename(backup, &destination)
                            .context("Restore previous asset after activation failure")?;
                    }
                    return Err(error).context("Activate verified asset");
                }
                activated.push((destination, backup, part.clone()));
            }
            Ok(())
        })();
        if let Err(error) = result {
            for (destination, backup, part) in activated.into_iter().rev() {
                fs::rename(&destination, &part)
                    .context("Retain staged file while rolling back activation")?;
                if let Some(backup) = backup {
                    fs::rename(&backup, &destination).context("Restore previous active asset")?;
                }
            }
            return Err(error);
        }
        for (destination, backup, _) in activated {
            if let Some(backup) = backup {
                let _ = fs::remove_file(backup);
            }
            self.verified
                .insert(destination.clone(), identity(&destination)?);
        }
        self.inspect(groups)
    }

    /// Restore an interrupted activation's original baseline before retrying.
    /// Keep any verified replacement staged; never discard the only backup.
    fn recover_interrupted_activation(
        &mut self,
        groups: &[AssetGroup],
        cancel: &AssetCancellation,
    ) -> Result<()> {
        safe_destination(&self.root, self.staging.strip_prefix(&self.root)?)?;
        for file in self.selected(groups)? {
            cancel.check()?;
            let part_name = file.path.with_extension(format!(
                "{}.part",
                file.path
                    .extension()
                    .and_then(|value| value.to_str())
                    .unwrap_or("asset")
            ));
            let part = safe_destination(&self.staging, &part_name)?;
            let backup = safe_destination(&self.staging, &part_name.with_extension("previous"))?;
            if !backup.exists() {
                continue;
            }
            identity(&backup).context("Interrupted asset backup is not a regular file")?;
            let destination = safe_destination(&self.root, &file.path)?;
            if destination.exists() {
                let valid = verify_file(&destination, &file, cancel).is_ok();
                cancel.check()?;
                if valid {
                    let staged_valid = part.exists() && verify_file(&part, &file, cancel).is_ok();
                    cancel.check()?;
                    if staged_valid {
                        fs::remove_file(&destination)?;
                    } else {
                        if part.exists() {
                            fs::remove_file(&part)?;
                        }
                        fs::rename(&destination, &part)
                            .context("Retain verified replacement before recovery")?;
                    }
                } else {
                    // Invalid catalog-managed bytes; the original backup is
                    // untouched until its restoration rename succeeds.
                    fs::remove_file(&destination)?;
                }
            }
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::rename(&backup, &destination).context("Restore interrupted asset activation")?;
            self.verified.remove(&destination);
        }
        Ok(())
    }

    fn download(
        &mut self,
        file: &AssetFile,
        part: &Path,
        cancel: &AssetCancellation,
        emit: &mut impl FnMut(AssetPhase, u64),
    ) -> Result<()> {
        let url = Url::parse(&file.url)?;
        let mut offset = fs::metadata(part)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        if url.scheme() == "file" {
            let source = url
                .to_file_path()
                .map_err(|()| anyhow::anyhow!("Invalid local asset URL"))?;
            let mut input = File::open(&source)
                .with_context(|| format!("Open local asset {}", source.display()))?;
            ensure!(
                input.metadata()?.len() == file.size,
                "Local asset length differs from catalog"
            );
            input.seek(SeekFrom::Start(offset))?;
            let mut output = OpenOptions::new().create(true).append(true).open(part)?;
            copy_body(
                &mut input,
                &mut output,
                &mut offset,
                file.size,
                cancel,
                emit,
            )?;
            output.sync_all()?;
            ensure!(
                offset == file.size,
                "Local asset ended before its declared length"
            );
            return Ok(());
        }
        if self.client.is_none() {
            self.client = Some(
                Client::builder()
                    .redirect(reqwest::redirect::Policy::none())
                    .connect_timeout(Duration::from_secs(10))
                    .timeout(Duration::from_secs(30))
                    .user_agent("BRP-CHROMA-native-assets/1")
                    .build()?,
            );
        }
        if file.size - offset >= PARALLEL_MIN_BYTES
            && self.download_parallel(&url, file, part, &mut offset, cancel, emit)?
        {
            return Ok(());
        }
        while offset < file.size {
            cancel.check()?;
            // Large range requests: each request goes through the model host's
            // redirect, so small ranges spent most of their time on round trips.
            // Cancellation is still checked on every read, the client's timeout
            // applies per read, and an interrupted range resumes from the
            // partial file. Servers ignoring Range are safely reset.
            let end = offset
                .saturating_add(DOWNLOAD_RANGE_BYTES - 1)
                .min(file.size - 1);
            let mut response = self.request(&url, offset, end)?;
            let expected_response = match response.status() {
                StatusCode::PARTIAL_CONTENT => {
                    let value = response
                        .headers()
                        .get(header::CONTENT_RANGE)
                        .context("Partial response lacks Content-Range")?
                        .to_str()?;
                    let (start, stop, total) = parse_content_range(value)?;
                    ensure!(
                        start == offset && total == file.size && stop <= end,
                        "Server returned a mismatched Content-Range"
                    );
                    stop - start + 1
                }
                StatusCode::OK => {
                    offset = 0;
                    File::create(part)?;
                    file.size
                }
                code => bail!("Asset server returned HTTP {code}"),
            };
            if let Some(length) = response.content_length() {
                ensure!(
                    length == expected_response,
                    "HTTP Content-Length differs from the declared response range"
                );
            }
            let response_end = offset
                .checked_add(expected_response)
                .context("HTTP asset length overflow")?;
            let mut output = OpenOptions::new().create(true).append(true).open(part)?;
            copy_body(
                &mut response,
                &mut output,
                &mut offset,
                response_end,
                cancel,
                emit,
            )?;
            output.sync_all()?;
            ensure!(
                offset == response_end,
                "Asset response ended before its declared length"
            );
        }
        Ok(())
    }

    /// Downloads the rest of a large file over several connections. A single
    /// connection to the model host's CDN varies between roughly 5 and 40 MB/s
    /// depending on the byte range, so `PARALLEL_SEGMENTS` workers keep fetching
    /// consecutive segments while finished ones are appended in order: the
    /// partial file stays contiguous and resumable, and at most
    /// `PARALLEL_WINDOW` segments are held in memory. Returns `Ok(false)` when the
    /// server ignores Range before anything was written, so the caller can fall
    /// back to the single-connection path.
    fn download_parallel(
        &self,
        url: &Url,
        file: &AssetFile,
        part: &Path,
        offset: &mut u64,
        cancel: &AssetCancellation,
        emit: &mut impl FnMut(AssetPhase, u64),
    ) -> Result<bool> {
        use std::sync::{Mutex, mpsc};
        let mut output = OpenOptions::new().create(true).append(true).open(part)?;
        let first = *offset;
        let count = (file.size - first).div_ceil(PARALLEL_SEGMENT_BYTES);
        let next = AtomicU64::new(0);
        let written = AtomicU64::new(0);
        let stop = AtomicBool::new(false);
        let received = AtomicU64::new(0);
        // The redirect target (a signed CDN URL) is resolved once and reused.
        let direct: Mutex<Option<Url>> = Mutex::new(None);
        let (sender, receiver) = mpsc::channel::<(u64, Result<(Vec<u8>, Option<Url>)>)>();
        let mut failure: Option<anyhow::Error> = None;
        let mut appended = false;
        emit(AssetPhase::Downloading, *offset);
        std::thread::scope(|scope| {
            for _ in 0..PARALLEL_SEGMENTS {
                let sender = sender.clone();
                let (next, written, stop, received, direct) =
                    (&next, &written, &stop, &received, &direct);
                scope.spawn(move || {
                    loop {
                        let index = next.fetch_add(1, Ordering::SeqCst);
                        if index >= count {
                            return;
                        }
                        while index >= written.load(Ordering::SeqCst) + PARALLEL_WINDOW {
                            if stop.load(Ordering::SeqCst) || cancel.is_cancelled() {
                                return;
                            }
                            std::thread::sleep(Duration::from_millis(20));
                        }
                        let start = first + index * PARALLEL_SEGMENT_BYTES;
                        let end = (start + PARALLEL_SEGMENT_BYTES - 1).min(file.size - 1);
                        let mut result = Err(anyhow::anyhow!("not attempted"));
                        for _ in 0..3 {
                            if stop.load(Ordering::SeqCst) || cancel.is_cancelled() {
                                return;
                            }
                            let target = direct.lock().ok().and_then(|url| url.clone());
                            result = self.fetch_segment(
                                url, target, start, end, file.size, cancel, received,
                            );
                            match &result {
                                Ok(_) => break,
                                Err(error) if error.downcast_ref::<RangeIgnored>().is_some() => {
                                    break;
                                }
                                Err(_) => {}
                            }
                        }
                        let failed = result.is_err();
                        if sender.send((index, result)).is_err() || failed {
                            return;
                        }
                    }
                });
            }
            drop(sender);
            let mut pending = std::collections::BTreeMap::new();
            loop {
                match receiver.recv_timeout(Duration::from_millis(100)) {
                    Ok((index, Ok((bytes, target)))) => {
                        if let Some(target) = target
                            && let Ok(mut slot) = direct.lock()
                        {
                            *slot = Some(target);
                        }
                        pending.insert(index, bytes);
                    }
                    Ok((_, Err(error))) => {
                        failure.get_or_insert(error);
                        stop.store(true, Ordering::SeqCst);
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
                if cancel.is_cancelled() {
                    stop.store(true, Ordering::SeqCst);
                }
                // Append whatever is now contiguous with the partial file.
                while failure.is_none() {
                    let index = written.load(Ordering::SeqCst);
                    let Some(bytes) = pending.remove(&index) else {
                        break;
                    };
                    if let Err(error) = output.write_all(&bytes) {
                        failure = Some(error.into());
                        stop.store(true, Ordering::SeqCst);
                        break;
                    }
                    *offset += bytes.len() as u64;
                    received.fetch_sub(bytes.len() as u64, Ordering::SeqCst);
                    appended = true;
                    written.store(index + 1, Ordering::SeqCst);
                }
                emit(
                    AssetPhase::Downloading,
                    (*offset + received.load(Ordering::SeqCst)).min(file.size),
                );
            }
        });
        output.flush()?;
        output.sync_all()?;
        emit(AssetPhase::Downloading, *offset);
        if let Some(error) = failure {
            if !appended && error.downcast_ref::<RangeIgnored>().is_some() {
                return Ok(false);
            }
            return Err(error);
        }
        cancel.check()?;
        ensure!(
            *offset == file.size,
            "Asset download stopped before its declared length"
        );
        Ok(true)
    }

    /// One `[start, end]` segment, from the resolved CDN URL when known (falling
    /// back to the catalog URL if it was refused, e.g. an expired signature).
    #[allow(clippy::too_many_arguments)]
    fn fetch_segment(
        &self,
        url: &Url,
        direct: Option<Url>,
        start: u64,
        end: u64,
        total: u64,
        cancel: &AssetCancellation,
        received: &AtomicU64,
    ) -> Result<(Vec<u8>, Option<Url>)> {
        let client = self.client.as_ref().expect("client initialized");
        let mut response = None;
        if let Some(direct) = &direct {
            let attempt = client
                .get(direct.clone())
                .header(header::ACCEPT_ENCODING, "identity")
                .header(header::RANGE, format!("bytes={start}-{end}"))
                .send()
                .map_err(|error| error.without_url())?;
            if attempt.status() == StatusCode::PARTIAL_CONTENT {
                response = Some(attempt);
            }
        }
        let mut response = match response {
            Some(response) => response,
            None => self.request(url, start, end)?,
        };
        match response.status() {
            StatusCode::PARTIAL_CONTENT => {}
            StatusCode::OK => return Err(RangeIgnored.into()),
            code => bail!("Asset server returned HTTP {code}"),
        }
        let value = response
            .headers()
            .get(header::CONTENT_RANGE)
            .context("Partial response lacks Content-Range")?
            .to_str()?;
        let (got_start, got_end, got_total) = parse_content_range(value)?;
        ensure!(
            got_start == start && got_end == end && got_total == total,
            "Server returned a mismatched Content-Range"
        );
        let length = end - start + 1;
        if let Some(declared) = response.content_length() {
            ensure!(
                declared == length,
                "HTTP Content-Length differs from the declared response range"
            );
        }
        let target = Some(response.url().clone());
        let mut bytes = Vec::with_capacity(length as usize);
        let mut buffer = vec![0_u8; 256 * 1024];
        loop {
            cancel.check()?;
            let read = response.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            ensure!(
                bytes.len() as u64 + read as u64 <= length,
                "Asset response exceeds the declared byte length"
            );
            bytes.extend_from_slice(&buffer[..read]);
            received.fetch_add(read as u64, Ordering::Relaxed);
        }
        ensure!(
            bytes.len() as u64 == length,
            "Asset response ended before its declared length"
        );
        Ok((bytes, target))
    }

    fn request(
        &self,
        original_url: &Url,
        offset: u64,
        end: u64,
    ) -> Result<reqwest::blocking::Response> {
        let mut url = original_url.clone();
        for _ in 0..10 {
            let mut request = self
                .client
                .as_ref()
                .expect("client initialized")
                .get(url.clone())
                .header(header::ACCEPT_ENCODING, "identity")
                .header(header::RANGE, format!("bytes={offset}-{end}"));
            if accepts_huggingface_credential(&url)
                && let Some(token) = &self.huggingface_token
            {
                let mut value = header::HeaderValue::from_str(&format!("Bearer {token}"))
                    .context("Invalid locally stored model-host credential")?;
                value.set_sensitive(true);
                request = request.header(header::AUTHORIZATION, value);
            }
            // Strip the URL from reqwest errors: signed redirect query strings
            // can carry short-lived server credentials.
            let response = request.send().map_err(|error| error.without_url())?;
            if !response.status().is_redirection() {
                return Ok(response);
            }
            let location = response
                .headers()
                .get(header::LOCATION)
                .context("Asset redirect lacks Location")?
                .to_str()?;
            let next = url.join(location).context("Invalid asset redirect")?;
            ensure!(
                matches!(next.scheme(), "http" | "https")
                    && next.username().is_empty()
                    && next.password().is_none(),
                "Unsafe asset redirect"
            );
            ensure!(
                url.scheme() != "https" || next.scheme() == "https",
                "Refusing an insecure asset redirect"
            );
            url = next;
        }
        bail!("Too many asset download redirects")
    }
}

/// Bytes requested per HTTP range request on the single-connection path.
const DOWNLOAD_RANGE_BYTES: u64 = 256 * 1024 * 1024;
/// Files with at least this much left use parallel connections.
const PARALLEL_MIN_BYTES: u64 = 32 * 1024 * 1024;
/// Parallel connections, bytes per request, and how many segments may be held
/// in memory ahead of the partial file (12 x 16 MiB = 192 MiB at most).
const PARALLEL_SEGMENTS: u64 = 6;
const PARALLEL_SEGMENT_BYTES: u64 = 16 * 1024 * 1024;
const PARALLEL_WINDOW: u64 = 12;

/// The server answered a range request with the whole file.
#[derive(Debug)]
struct RangeIgnored;
impl std::fmt::Display for RangeIgnored {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Asset server ignored the byte range")
    }
}
impl std::error::Error for RangeIgnored {}

fn accepts_huggingface_credential(url: &Url) -> bool {
    url.scheme() == "https"
        && url.host_str() == Some("huggingface.co")
        && url.port_or_known_default() == Some(443)
}

fn copy_body(
    input: &mut impl Read,
    output: &mut File,
    offset: &mut u64,
    end: u64,
    cancel: &AssetCancellation,
    emit: &mut impl FnMut(AssetPhase, u64),
) -> Result<()> {
    let mut buffer = vec![0_u8; 1024 * 1024];
    emit(AssetPhase::Downloading, *offset);
    loop {
        cancel.check()?;
        let read = input.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        ensure!(
            *offset + read as u64 <= end,
            "Asset response exceeds the declared byte length"
        );
        output.write_all(&buffer[..read])?;
        *offset += read as u64;
        emit(AssetPhase::Downloading, *offset);
    }
    output.flush()?;
    Ok(())
}

fn verify_file(path: &Path, expected: &AssetFile, cancel: &AssetCancellation) -> Result<()> {
    let before = identity(path)?;
    ensure!(
        before.size == expected.size,
        "Asset length differs from catalog"
    );
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        cancel.check()?;
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    ensure!(
        format!("{:x}", hash.finalize()).eq_ignore_ascii_case(&expected.sha256),
        "Asset SHA256 differs from catalog"
    );
    ensure!(
        identity(path)? == before,
        "Asset changed while checking its SHA256"
    );
    Ok(())
}

fn validate_manifest(manifest: &AssetManifest) -> Result<()> {
    ensure!(
        manifest.schema_version == 1,
        "Unsupported asset catalog version"
    );
    ensure!(
        !manifest.id.trim().is_empty() && !manifest.version.trim().is_empty(),
        "Asset catalog identity/version is required"
    );
    ensure!(
        !manifest.files.is_empty(),
        "Asset catalog contains no files"
    );
    let mut paths = HashSet::new();
    for file in &manifest.files {
        validate_relative_path(&file.path)?;
        let key = file.path.to_string_lossy().to_lowercase();
        ensure!(
            paths.insert(key),
            "Asset catalog contains duplicate destination paths"
        );
        ensure!(file.size > 0, "Asset byte length must be positive");
        ensure!(
            file.sha256.len() == 64 && file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "Asset SHA256 must contain 64 hexadecimal characters"
        );
        if file.bundled_only {
            ensure!(
                matches!(file.group, AssetGroup::Core | AssetGroup::Runtime),
                "Only native Core/Runtime files may be marked bundled-only; model downloads require a source URL"
            );
            if file.url.is_empty() {
                continue;
            }
        }
        let url =
            Url::parse(&file.url).context("Asset requires a real HTTP(S) or local file URL")?;
        ensure!(
            matches!(url.scheme(), "http" | "https" | "file"),
            "Unsupported asset source URL scheme"
        );
        ensure!(
            url.username().is_empty() && url.password().is_none(),
            "Credentials must not be embedded in the asset catalog"
        );
        if url.scheme() == "file" {
            ensure!(url.to_file_path().is_ok(), "Invalid local asset URL");
        }
    }
    Ok(())
}

fn validate_relative_path(path: &Path) -> Result<()> {
    let raw = path.to_str().context("Asset path must be UTF-8")?;
    ensure!(
        !raw.is_empty() && !raw.contains(['\\', ':']) && !raw.ends_with('/') && !raw.contains("//"),
        "Asset paths must use safe relative forward-slash names"
    );
    ensure!(
        raw.split('/').all(|part| !part.is_empty()
            && part != "."
            && part != ".."
            && !part.ends_with([' ', '.'])
            && !part
                .chars()
                .any(|character| character.is_control() || "<>\"|?*".contains(character))
            && !is_windows_device(part)),
        "Unsafe asset path component"
    );
    ensure!(
        path.components()
            .all(|part| matches!(part, Component::Normal(_))),
        "Absolute/traversing asset paths are forbidden"
    );
    ensure!(
        path.components()
            .next()
            .is_some_and(|part| part.as_os_str() != ".asset-staging"),
        "Asset path collides with installer staging"
    );
    Ok(())
}

fn is_windows_device(part: &str) -> bool {
    let base = part.split('.').next().unwrap_or(part).to_ascii_uppercase();
    matches!(base.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (base.len() == 4
            && (base.starts_with("COM") || base.starts_with("LPT"))
            && matches!(base.as_bytes()[3], b'1'..=b'9'))
}

fn is_link(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}

fn safe_destination(root: &Path, relative: &Path) -> Result<PathBuf> {
    // Callers separately validate catalog names. The staging directory uses the
    // same containment check while retaining its reserved prefix.
    let mut path = root.to_path_buf();
    for part in relative.components() {
        ensure!(
            matches!(part, Component::Normal(_)),
            "Unsafe asset destination"
        );
        path.push(part.as_os_str());
        match fs::symlink_metadata(&path) {
            Ok(metadata) => ensure!(
                !is_link(&metadata),
                "Asset destination contains a symbolic link or reparse point: {}",
                path.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    ensure!(
        path.starts_with(root),
        "Asset destination escapes the application folder"
    );
    Ok(path)
}

fn parse_content_range(value: &str) -> Result<(u64, u64, u64)> {
    let value = value
        .strip_prefix("bytes ")
        .context("Unsupported Content-Range unit")?;
    let (range, total) = value.split_once('/').context("Invalid Content-Range")?;
    let (start, end) = range
        .split_once('-')
        .context("Invalid Content-Range bounds")?;
    let (start, end, total) = (
        start.parse::<u64>()?,
        end.parse::<u64>()?,
        total.parse::<u64>()?,
    );
    ensure!(start <= end && end < total, "Invalid Content-Range bounds");
    Ok((start, end, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{net::TcpListener, thread};

    fn scratch(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "brp-assets-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        let resolved = path.canonicalize().unwrap();
        assert_eq!(
            resolved.parent(),
            Some(std::env::temp_dir().canonicalize().unwrap().as_path())
        );
        assert!(
            resolved
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("brp-assets-")
        );
        path
    }
    fn manifest(url: String, content: &[u8]) -> AssetManifest {
        AssetManifest {
            schema_version: 1,
            id: "test".into(),
            version: "1".into(),
            files: vec![AssetFile {
                group: AssetGroup::Ck1080,
                path: "models/corridorkey/1080/model.onnx.data".into(),
                url,
                size: content.len() as u64,
                sha256: format!("{:x}", Sha256::digest(content)),
                bundled_only: false,
            }],
        }
    }

    #[test]
    fn bundled_native_inspection_hashes_files_but_never_plans_a_download() {
        let root = scratch("bundled-inspect");
        let app = root.join("app");
        let mut data = manifest(String::new(), b"native");
        data.files[0].group = AssetGroup::Core;
        data.files[0].path = "ck-native.exe".into();
        data.files[0].bundled_only = true;
        let mut manager = AssetManager::new(app.clone(), data).unwrap();
        let missing = manager.inspect(&[AssetGroup::Core]).unwrap();
        assert!(!missing.ready && missing.missing[0].bundled_only);
        assert_eq!(missing.download_bytes, 0);
        fs::write(app.join("ck-native.exe"), b"native").unwrap();
        assert!(manager.inspect(&[AssetGroup::Core]).unwrap().ready);
        fs::write(app.join("ck-native.exe"), b"broken").unwrap();
        let corrupt = manager.inspect(&[AssetGroup::Core]).unwrap();
        assert!(!corrupt.ready && corrupt.missing[0].bundled_only);
        assert_eq!(corrupt.download_bytes, 0);
        assert!(manager.client.is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn missing_bundled_native_blocks_model_install_before_staging_or_network() {
        let root = scratch("bundled-install");
        let app = root.join("app");
        let mut data = manifest(
            "https://127.0.0.1:1/no-network-must-happen".into(),
            b"model",
        );
        let mut native = data.files[0].clone();
        native.group = AssetGroup::Core;
        native.path = "ck-native.exe".into();
        native.url.clear();
        native.size = 6;
        native.sha256 = format!("{:x}", Sha256::digest(b"native"));
        native.bundled_only = true;
        data.files.push(native);
        let mut manager = AssetManager::new(app.clone(), data).unwrap();
        let mut events = Vec::new();
        let error = manager
            .install(
                &[AssetGroup::Ck1080],
                &AssetCancellation::default(),
                |event| events.push(event),
            )
            .unwrap_err();
        assert!(error.to_string().contains("Reinstall app files"));
        assert!(manager.client.is_none());
        assert!(events.is_empty());
        assert!(!app.join(".asset-staging").exists());
        assert!(!app.join("models").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bundled_flag_is_backward_compatible_and_cannot_disable_model_sources() {
        let old = manifest("https://example.org/model".into(), b"x");
        let encoded = serde_json::to_value(&old).unwrap();
        assert!(encoded["files"][0].get("bundled_only").is_none());
        let decoded: AssetManifest = serde_json::from_value(encoded).unwrap();
        assert!(!decoded.files[0].bundled_only);
        let root = scratch("bundled-policy");
        let mut model = old.clone();
        model.files[0].bundled_only = true;
        model.files[0].url.clear();
        assert!(AssetManager::new(root.clone(), model).is_err());
        let mut no_source = old;
        no_source.files[0].url.clear();
        assert!(AssetManager::new(root.clone(), no_source).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    fn server(
        body: Vec<u8>,
        expected_range: &'static str,
        status: &'static str,
        content_range: Option<String>,
    ) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut byte = [0_u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            let request = String::from_utf8(request).unwrap().to_lowercase();
            assert!(request.contains(expected_range));
            assert!(
                !request.contains("authorization:"),
                "credential leaked to a non-Hugging-Face origin"
            );
            let range_header = content_range
                .map(|value| format!("Content-Range: {value}\r\n"))
                .unwrap_or_default();
            write!(stream, "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{range_header}Connection: close\r\n\r\n", body.len()).unwrap();
            stream.write_all(&body).unwrap();
        });
        (format!("http://{address}/asset"), handle)
    }
    #[test]
    fn rejects_traversal_duplicate_and_absent_group() {
        let root = scratch("paths");
        for path in [
            "../escape",
            "/absolute",
            "C:/escape",
            "models\\escape",
            "models/../escape",
            ".asset-staging/bad",
            "models/file.",
        ] {
            let mut data = manifest("https://example.org/model".into(), b"x");
            data.files[0].path = path.into();
            assert!(AssetManager::new(root.clone(), data).is_err(), "{path}");
        }
        let data = manifest("https://example.org/model".into(), b"x");
        let mut manager = AssetManager::new(root.clone(), data.clone()).unwrap();
        assert!(manager.inspect(&[AssetGroup::Sam]).is_err());
        let mut duplicate = data;
        duplicate.files.push(duplicate.files[0].clone());
        assert!(AssetManager::new(root.clone(), duplicate).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn local_adoption_restart_tamper_and_missing_are_checked() {
        let root = scratch("adopt");
        let source = root.join("source");
        fs::write(&source, b"model-data").unwrap();
        let data = manifest(
            Url::from_file_path(&source).unwrap().to_string(),
            b"model-data",
        );
        let app = root.join("app");
        let mut manager = AssetManager::new(app.clone(), data.clone()).unwrap();
        assert!(!manager.inspect(&[AssetGroup::Ck1080]).unwrap().ready);
        assert!(
            manager
                .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                .unwrap()
                .ready
        );
        let mut restarted = AssetManager::new(app.clone(), data.clone()).unwrap();
        assert!(restarted.inspect(&[AssetGroup::Ck1080]).unwrap().ready);
        let destination = app.join(&data.files[0].path);
        fs::write(&destination, b"wrong-data").unwrap();
        assert!(!restarted.inspect(&[AssetGroup::Ck1080]).unwrap().ready);
        fs::remove_file(&destination).unwrap();
        assert!(!restarted.inspect(&[AssetGroup::Ck1080]).unwrap().ready);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn http_resumes_valid_content_range() {
        let body = b"native-model-external-data";
        let (url, server) = server(
            body[7..].to_vec(),
            "range: bytes=7-25",
            "206 Partial Content",
            Some("bytes 7-25/26".into()),
        );
        let root = scratch("resume");
        let data = manifest(url, body);
        let mut manager = AssetManager::new(root.clone(), data.clone()).unwrap();
        let part = manager
            .staging
            .join(&data.files[0].path)
            .with_extension("data.part");
        fs::create_dir_all(part.parent().unwrap()).unwrap();
        fs::write(&part, &body[..7]).unwrap();
        assert!(
            manager
                .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                .unwrap()
                .ready
        );
        assert_eq!(fs::read(root.join(&data.files[0].path)).unwrap(), body);
        server.join().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
    /// A range server that accepts any number of connections.
    fn range_server(body: Arc<Vec<u8>>) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = requests.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { break };
                let body = body.clone();
                let counter = counter.clone();
                thread::spawn(move || {
                    let mut request = Vec::new();
                    let mut byte = [0_u8; 1];
                    while !request.ends_with(b"\r\n\r\n") {
                        if stream.read_exact(&mut byte).is_err() {
                            return;
                        }
                        request.push(byte[0]);
                    }
                    counter.fetch_add(1, Ordering::SeqCst);
                    let request = String::from_utf8(request).unwrap().to_lowercase();
                    let range = request
                        .lines()
                        .find_map(|line| line.strip_prefix("range: bytes="))
                        .unwrap();
                    let (start, end) = range.trim().split_once('-').unwrap();
                    let (start, end): (usize, usize) =
                        (start.parse().unwrap(), end.parse().unwrap());
                    let slice = &body[start..=end];
                    let _ = write!(
                        stream,
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{}\r\nConnection: close\r\n\r\n",
                        slice.len(),
                        body.len()
                    );
                    let _ = stream.write_all(slice);
                });
            }
        });
        (format!("http://{address}/asset"), requests)
    }
    #[test]
    fn large_files_download_in_parallel_and_resume_from_the_partial_file() {
        let body: Vec<u8> = (0..(PARALLEL_MIN_BYTES as usize + 7_000_003))
            .map(|index| (index.wrapping_mul(2_654_435_761) >> 13) as u8)
            .collect();
        let body = Arc::new(body);
        let (url, requests) = range_server(body.clone());
        let root = scratch("parallel");
        let data = manifest(url, &body);
        let mut manager = AssetManager::new(root.clone(), data.clone()).unwrap();
        // An earlier interrupted download left the first 5 MB.
        let part = manager
            .staging
            .join(&data.files[0].path)
            .with_extension("data.part");
        fs::create_dir_all(part.parent().unwrap()).unwrap();
        fs::write(&part, &body[..5_000_000]).unwrap();
        let mut last = 0;
        let status = manager
            .install(
                &[AssetGroup::Ck1080],
                &AssetCancellation::default(),
                |progress| {
                    if progress.phase == AssetPhase::Downloading {
                        assert!(progress.downloaded >= 5_000_000);
                        last = last.max(progress.downloaded);
                    }
                },
            )
            .unwrap();
        assert!(status.ready);
        assert_eq!(last, body.len() as u64);
        assert_eq!(fs::read(root.join(&data.files[0].path)).unwrap(), *body);
        // 32 MB + 2 MB remaining in 16 MB segments.
        assert_eq!(requests.load(Ordering::SeqCst), 3);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn ignored_range_resets_and_bad_range_preserves_active_file() {
        let body = b"replacement";
        let (url, server) = server(body.to_vec(), "range: bytes=3-10", "200 OK", None);
        let root = scratch("reset");
        let data = manifest(url, body);
        let mut manager = AssetManager::new(root.clone(), data.clone()).unwrap();
        let part = manager
            .staging
            .join(&data.files[0].path)
            .with_extension("data.part");
        fs::create_dir_all(part.parent().unwrap()).unwrap();
        fs::write(&part, b"bad").unwrap();
        assert!(
            manager
                .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                .unwrap()
                .ready
        );
        server.join().unwrap();
        let (url, server) = self::server(
            body[3..].to_vec(),
            "range: bytes=3-10",
            "206 Partial Content",
            Some("bytes 2-9/11".into()),
        );
        let data = manifest(url, body);
        let mut manager = AssetManager::new(root.clone(), data.clone()).unwrap();
        let destination = root.join(&data.files[0].path);
        fs::write(&destination, b"old-working").unwrap();
        let part = manager
            .staging
            .join(&data.files[0].path)
            .with_extension("data.part");
        fs::create_dir_all(part.parent().unwrap()).unwrap();
        fs::write(&part, b"rep").unwrap();
        assert!(
            manager
                .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                .is_err()
        );
        assert_eq!(fs::read(destination).unwrap(), b"old-working");
        server.join().unwrap();
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn checksum_failure_or_cancellation_never_replaces_active_assets() {
        let root = scratch("cancel");
        let source = root.join("source");
        fs::write(&source, vec![5_u8; 512 * 1024]).unwrap();
        let body = vec![5_u8; 512 * 1024];
        let data = manifest(Url::from_file_path(&source).unwrap().to_string(), &body);
        let app = root.join("app");
        let mut manager = AssetManager::new(app.clone(), data.clone()).unwrap();
        let destination = app.join(&data.files[0].path);
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        fs::write(&destination, b"old-working").unwrap();
        let cancel = AssetCancellation::default();
        let control = cancel.clone();
        assert!(
            manager
                .install(&[AssetGroup::Ck1080], &cancel, |event| {
                    if event.phase == AssetPhase::Downloading && event.downloaded > 0 {
                        control.cancel();
                    }
                })
                .is_err()
        );
        assert_eq!(fs::read(&destination).unwrap(), b"old-working");
        assert!(
            manager
                .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                .unwrap()
                .ready
        );
        fs::write(&source, vec![6_u8; 512 * 1024]).unwrap();
        fs::write(&destination, b"old-working").unwrap();
        assert!(
            manager
                .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                .is_err()
        );
        assert_eq!(fs::read(destination).unwrap(), b"old-working");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atomic_activation_rolls_back_all_files_and_serializes_installers() {
        let root = scratch("transaction");
        let source = root.join("source");
        fs::write(&source, b"replacement").unwrap();
        let mut data = manifest(
            Url::from_file_path(&source).unwrap().to_string(),
            b"replacement",
        );
        let mut second = data.files[0].clone();
        second.path = "models/corridorkey/1080/other.onnx".into();
        data.files.push(second);
        let app = root.join("app");
        let mut manager = AssetManager::new(app.clone(), data.clone()).unwrap();
        for file in &data.files {
            let path = app.join(&file.path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, b"old-working").unwrap();
        }
        let mut competing = AssetManager::new(app.clone(), data.clone()).unwrap();
        let mut checked_lock = false;
        let cancel = AssetCancellation::default();
        let control = cancel.clone();
        assert!(
            manager
                .install(&[AssetGroup::Ck1080], &cancel, |event| {
                    if !checked_lock {
                        checked_lock = true;
                        assert!(
                            competing
                                .install(
                                    &[AssetGroup::Ck1080],
                                    &AssetCancellation::default(),
                                    |_| {}
                                )
                                .is_err()
                        );
                    }
                    if event.phase == AssetPhase::Activating && event.file_index == 2 {
                        control.cancel();
                    }
                })
                .is_err()
        );
        assert!(checked_lock);
        for file in &data.files {
            assert_eq!(fs::read(app.join(&file.path)).unwrap(), b"old-working");
        }
        assert!(
            manager
                .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                .unwrap()
                .ready
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn credentials_are_origin_scoped_and_never_follow_cross_host_redirects() {
        for (url, expected) in [
            ("https://huggingface.co/a", true),
            ("https://huggingface.co:443/a", true),
            ("http://huggingface.co/a", false),
            ("https://huggingface.co.evil.test/a", false),
            ("https://cdn.huggingface.co/a", false),
            ("https://huggingface.co:444/a", false),
        ] {
            assert_eq!(
                accepts_huggingface_credential(&Url::parse(url).unwrap()),
                expected
            );
        }
        let body = b"native-redirect";
        let (target, server) = server(body.to_vec(), "range: bytes=0-14", "200 OK", None);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let source = format!("http://{}/asset", listener.local_addr().unwrap());
        // Different URL host as well as origin. Both must be unauthenticated.
        let target = target.replace("127.0.0.1", "localhost");
        let redirect = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut byte = [0_u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            assert!(
                !String::from_utf8(request)
                    .unwrap()
                    .to_lowercase()
                    .contains("authorization:")
            );
            write!(stream, "HTTP/1.1 302 Found\r\nLocation: {target}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap();
        });
        let root = scratch("auth-redirect");
        let data = manifest(source, body);
        let mut manager = AssetManager::new(root.clone(), data)
            .unwrap()
            .with_huggingface_token("private-test-credential".into());
        assert!(
            manager
                .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                .unwrap()
                .ready
        );
        redirect.join().unwrap();
        server.join().unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn optional_profile_reuses_verified_external_data_without_contacting_its_url() {
        let root = scratch("dedup");
        let source = root.join("source");
        fs::write(&source, b"shared-native-weights").unwrap();
        let mut data = manifest(
            Url::from_file_path(&source).unwrap().to_string(),
            b"shared-native-weights",
        );
        let mut optional = data.files[0].clone();
        optional.group = AssetGroup::Ck4k;
        optional.path = "models/corridorkey/4k/model.onnx.data".into();
        optional.url = "http://127.0.0.1:1/must-not-contact".into();
        data.files.push(optional);
        let app = root.join("app");
        let mut manager = AssetManager::new(app.clone(), data.clone()).unwrap();
        let initial = manager
            .inspect(&[AssetGroup::Ck1080, AssetGroup::Ck4k])
            .unwrap();
        assert_eq!(initial.download_bytes, data.files[0].size);
        assert_eq!(initial.reusable_bytes, data.files[0].size);
        manager
            .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
            .unwrap();
        // New process/session rehashes the existing1080 file before reuse.
        let mut manager = AssetManager::new(app.clone(), data.clone()).unwrap();
        let optional = manager.inspect(&[AssetGroup::Ck4k]).unwrap();
        assert_eq!(optional.download_bytes, 0);
        assert_eq!(optional.reusable_bytes, data.files[0].size);
        assert!(
            manager
                .install(&[AssetGroup::Ck4k], &AssetCancellation::default(), |_| {})
                .unwrap()
                .ready
        );
        assert_eq!(
            fs::read(app.join(&data.files[1].path)).unwrap(),
            b"shared-native-weights"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn interrupted_activation_recovers_before_inspection_and_retry() {
        for state in ["before-new-rename", "after-new-rename", "later-corruption"] {
            let root = scratch(state);
            let source = root.join("source");
            fs::write(&source, b"replacement").unwrap();
            let data = manifest(
                Url::from_file_path(&source).unwrap().to_string(),
                b"replacement",
            );
            let app = root.join("app");
            let manager = AssetManager::new(app.clone(), data.clone()).unwrap();
            let destination = app.join(&data.files[0].path);
            fs::create_dir_all(destination.parent().unwrap()).unwrap();
            let part = manager
                .staging
                .join(&data.files[0].path)
                .with_extension("data.part");
            fs::create_dir_all(part.parent().unwrap()).unwrap();
            let backup = part.with_extension("previous");
            fs::write(&backup, b"old-working").unwrap();
            match state {
                "before-new-rename" => fs::write(&part, b"replacement").unwrap(),
                "after-new-rename" => fs::write(&destination, b"replacement").unwrap(),
                _ => fs::write(&destination, b"corrupt-data").unwrap(),
            }
            drop(manager);
            let mut restarted = AssetManager::new(app.clone(), data.clone()).unwrap();
            assert!(
                restarted
                    .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                    .unwrap()
                    .ready
            );
            assert_eq!(fs::read(&destination).unwrap(), b"replacement");
            assert!(!backup.exists());
            let mut restarted = AssetManager::new(app, data).unwrap();
            assert!(
                restarted
                    .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                    .unwrap()
                    .ready
            );
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn large_assets_install_and_recheck_on_a_small_native_worker_stack() {
        let root = scratch("small-stack");
        let source = root.join("source");
        let content = vec![43_u8; 3 * 1024 * 1024 + 17];
        fs::write(&source, &content).unwrap();
        let data = manifest(Url::from_file_path(&source).unwrap().to_string(), &content);
        let app = root.join("app");
        thread::Builder::new()
            .name("asset-small-stack-proof".into())
            .stack_size(128 * 1024)
            .spawn(move || {
                let mut manager = AssetManager::new(app.clone(), data.clone()).unwrap();
                assert!(
                    manager
                        .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                        .unwrap()
                        .ready
                );
                let mut restarted = AssetManager::new(app, data).unwrap();
                assert!(restarted.inspect(&[AssetGroup::Ck1080]).unwrap().ready);
            })
            .unwrap()
            .join()
            .unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn interrupted_multi_file_transaction_restores_baseline_before_failed_acquisition() {
        let root = scratch("multi-recovery");
        let first = root.join("first");
        let second = root.join("second");
        fs::write(&first, b"new-first").unwrap();
        fs::write(&second, b"bad-second").unwrap();
        let mut data = manifest(
            Url::from_file_path(&first).unwrap().to_string(),
            b"new-first",
        );
        let mut other = manifest(
            Url::from_file_path(&second).unwrap().to_string(),
            b"new-second",
        )
        .files
        .remove(0);
        other.path = "models/corridorkey/1080/other.onnx".into();
        data.files.push(other);
        let app = root.join("app");
        let manager = AssetManager::new(app.clone(), data.clone()).unwrap();
        let first_dest = app.join(&data.files[0].path);
        let second_dest = app.join(&data.files[1].path);
        fs::create_dir_all(first_dest.parent().unwrap()).unwrap();
        fs::write(&first_dest, b"new-first").unwrap();
        fs::write(&second_dest, b"old-second").unwrap();
        let part = manager
            .staging
            .join(&data.files[0].path)
            .with_extension("data.part");
        fs::create_dir_all(part.parent().unwrap()).unwrap();
        fs::write(part.with_extension("previous"), b"old-first").unwrap();
        drop(manager);
        let mut restarted = AssetManager::new(app.clone(), data.clone()).unwrap();
        assert!(
            restarted
                .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                .is_err()
        );
        assert_eq!(fs::read(&first_dest).unwrap(), b"old-first");
        assert_eq!(fs::read(&second_dest).unwrap(), b"old-second");
        assert_eq!(fs::read(&part).unwrap(), b"new-first");
        fs::write(second, b"new-second").unwrap();
        let mut restarted = AssetManager::new(app, data).unwrap();
        assert!(
            restarted
                .install(&[AssetGroup::Ck1080], &AssetCancellation::default(), |_| {})
                .unwrap()
                .ready
        );
        assert_eq!(fs::read(first_dest).unwrap(), b"new-first");
        assert_eq!(fs::read(second_dest).unwrap(), b"new-second");
        fs::remove_dir_all(root).unwrap();
    }
}
