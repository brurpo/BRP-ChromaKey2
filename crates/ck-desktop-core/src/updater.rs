//! In-app updates from GitHub releases.
//!
//! Each release of `UPDATE_REPOSITORY` has one asset: the install zip (`*-windows-x64.zip`,
//! about 35 MB; models are downloaded separately). When the latest release's version is newer than
//! `release.json`, the app downloads the whole zip, checks it against the SHA-256 digest GitHub
//! publishes for the asset, unpacks it into `.update-staging/<version>` and swaps every file in.
//! Any installed version therefore updates straight to the newest release.
//!
//! Flow: [`check_latest`] (GitHub API, anonymous) -> [`download`] (zip + unpack) -> [`apply`] (each
//! file is swapped in by renaming the current one to `*.brp-old`, with rollback on failure; Windows
//! allows renaming running executables and loaded DLLs) -> restart. [`cleanup_previous`] removes
//! `*.brp-old` files at the next start.

use anyhow::{Context, Result, bail, ensure};
use reqwest::{StatusCode, blocking::Client, header};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::time::Duration;

/// Public repository whose releases the app updates from.
pub const UPDATE_REPOSITORY: &str = "brurpo/BRP-ChromaKey2";
/// The release asset the updater downloads: the install zip.
pub const RELEASE_ZIP_SUFFIX: &str = "-windows-x64.zip";
/// Installed release, shipped in the zip and rewritten by [`apply`].
pub const RELEASE_FILE: &str = "release.json";
const STAGING: &str = ".update-staging";
const OLD_SUFFIX: &str = "brp-old";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstalledRelease {
    pub version: String,
}

/// A newer release.
#[derive(Clone, Debug)]
pub struct UpdatePlan {
    pub version: String,
    pub notes: String,
    pub page: String,
    /// Size of the zip to download.
    pub bytes: u64,
    zip_url: String,
    /// SHA-256 of the zip as published by GitHub, when available.
    sha256: Option<String>,
}

#[derive(Clone, Default)]
pub struct UpdateCancel(Arc<AtomicBool>);

impl UpdateCancel {
    pub fn cancel(&self) {
        self.0.store(true, AtomicOrdering::Relaxed);
    }
    fn check(&self) -> Result<()> {
        ensure!(!self.0.load(AtomicOrdering::Relaxed), "Update cancelled");
        Ok(())
    }
}

/// Installed release version (`release.json`), `"0"` when unknown.
pub fn installed_version(app_dir: &Path) -> String {
    fs::read(app_dir.join(RELEASE_FILE))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<InstalledRelease>(&bytes).ok())
        .map_or_else(|| "0".to_owned(), |release| release.version)
}

/// Compares dotted versions numerically (`2026.11.01.2` > `2026.10.31.9`); a leading `v` and any
/// non-numeric suffix of a part (`3-native-test`) are ignored.
pub fn compare_versions(a: &str, b: &str) -> Ordering {
    let parts = |v: &str| -> Vec<u64> {
        v.trim()
            .trim_start_matches(['v', 'V'])
            .split('.')
            .map(|part| {
                let digits: String = part.chars().take_while(char::is_ascii_digit).collect();
                digits.parse().unwrap_or(0)
            })
            .collect()
    };
    let (a, b) = (parts(a), parts(b));
    for i in 0..a.len().max(b.len()) {
        match a.get(i).unwrap_or(&0).cmp(b.get(i).unwrap_or(&0)) {
            Ordering::Equal => {}
            other => return other,
        }
    }
    Ordering::Equal
}

fn client() -> Result<Client> {
    // The timeout applies to each read, not the whole download.
    Ok(Client::builder()
        .user_agent(concat!("BRP-Chroma-updater/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(60))
        .build()?)
}

#[derive(Deserialize)]
struct GithubRelease {
    tag_name: String,
    #[serde(default)]
    html_url: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    #[serde(default)]
    assets: Vec<GithubAsset>,
}

#[derive(Deserialize)]
struct GithubAsset {
    name: String,
    browser_download_url: String,
    size: u64,
    /// `sha256:<hex>`, published by GitHub for release assets.
    #[serde(default)]
    digest: Option<String>,
}

/// The latest release when its version is newer than `current`.
pub fn check_latest(repository: &str, current: &str) -> Result<Option<UpdatePlan>> {
    let response = client()?
        .get(format!(
            "https://api.github.com/repos/{repository}/releases/latest"
        ))
        .header(header::ACCEPT, "application/vnd.github+json")
        .send()
        .context("contacting GitHub")?;
    if response.status() == StatusCode::NOT_FOUND {
        // No release yet (or the repository is not public).
        return Ok(None);
    }
    let release: GithubRelease = serde_json::from_slice(
        &response
            .error_for_status()
            .context("reading the latest release")?
            .bytes()?,
    )
    .context("reading the latest release")?;
    if release.draft
        || release.prerelease
        || compare_versions(&release.tag_name, current) != Ordering::Greater
    {
        return Ok(None);
    }
    let Some(asset) = release
        .assets
        .iter()
        .find(|asset| asset.name.ends_with(RELEASE_ZIP_SUFFIX))
    else {
        return Ok(None);
    };
    ensure!(
        asset.browser_download_url.starts_with("https://"),
        "the release download has no HTTPS URL"
    );
    Ok(Some(UpdatePlan {
        version: release.tag_name.trim_start_matches(['v', 'V']).to_owned(),
        notes: release.body.unwrap_or_default(),
        page: release.html_url,
        bytes: asset.size,
        zip_url: asset.browser_download_url.clone(),
        sha256: asset
            .digest
            .as_deref()
            .and_then(|digest| digest.strip_prefix("sha256:"))
            .map(str::to_owned),
    }))
}

/// Validates a path from the zip: relative, no `..`, not inside the staging folder.
fn relative_path(path: &str) -> Result<PathBuf> {
    ensure!(!path.is_empty(), "empty path in the update");
    let mut out = PathBuf::new();
    for part in Path::new(path).components() {
        match part {
            Component::Normal(name) => out.push(name),
            _ => bail!("unsafe path in the update: {path}"),
        }
    }
    ensure!(
        !out.starts_with(STAGING),
        "update path inside the staging folder: {path}"
    );
    Ok(out)
}

fn u16_at(data: &[u8], at: usize) -> Result<u16> {
    let bytes = data.get(at..at + 2).context("truncated zip")?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn u32_at(data: &[u8], at: usize) -> Result<usize> {
    let bytes = data.get(at..at + 4).context("truncated zip")?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize)
}

/// Unpacks a zip (stored or deflated entries) into `out`, dropping the zip's single top-level
/// folder (`BRP-ChromaKey2/BRP-Chroma.exe` -> `BRP-Chroma.exe`). Returns the unpacked files.
fn unpack_zip(zip: &[u8], out: &Path) -> Result<Vec<PathBuf>> {
    ensure!(zip.len() >= 22, "the update archive is too small");
    let eocd = (0..=zip.len() - 22)
        .rev()
        .find(|&at| zip[at..at + 4] == [0x50, 0x4b, 0x05, 0x06])
        .context("the update archive has no zip directory")?;
    let count = usize::from(u16_at(zip, eocd + 10)?);
    let mut at = u32_at(zip, eocd + 16)?;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        ensure!(u32_at(zip, at)? == 0x0201_4b50, "bad zip directory entry");
        let method = u16_at(zip, at + 10)?;
        let compressed = u32_at(zip, at + 20)?;
        let size = u32_at(zip, at + 24)?;
        let name_len = usize::from(u16_at(zip, at + 28)?);
        let skip = usize::from(u16_at(zip, at + 30)?) + usize::from(u16_at(zip, at + 32)?);
        let header = u32_at(zip, at + 42)?;
        let name = zip
            .get(at + 46..at + 46 + name_len)
            .context("truncated zip directory")?;
        let name = String::from_utf8_lossy(name).replace('\\', "/");
        entries.push((name, method, compressed, size, header));
        at += 46 + name_len + skip;
    }
    let prefix = entries
        .first()
        .and_then(|(name, ..)| name.split_once('/'))
        .map(|(top, _)| format!("{top}/"))
        .filter(|prefix| {
            entries
                .iter()
                .all(|(name, ..)| name.starts_with(prefix.as_str()))
        })
        .unwrap_or_default();
    let mut files = Vec::new();
    for (name, method, compressed, size, header) in entries {
        let relative = &name[prefix.len()..];
        if relative.is_empty() || relative.ends_with('/') {
            continue;
        }
        let relative = relative_path(relative)?;
        ensure!(u32_at(zip, header)? == 0x0403_4b50, "bad zip file header");
        let start = header
            + 30
            + usize::from(u16_at(zip, header + 26)?)
            + usize::from(u16_at(zip, header + 28)?);
        let data = zip
            .get(start..start + compressed)
            .context("truncated zip data")?;
        let bytes = match method {
            0 => data.to_vec(),
            8 => miniz_oxide::inflate::decompress_to_vec_with_limit(data, size)
                .map_err(|error| anyhow::anyhow!("corrupt update archive: {error:?}"))?,
            method => bail!("unsupported zip compression method {method}"),
        };
        ensure!(bytes.len() == size, "corrupt update archive");
        let target = out.join(&relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&target, bytes)?;
        files.push(relative);
    }
    ensure!(!files.is_empty(), "the update archive is empty");
    Ok(files)
}

fn staging_dir(app_dir: &Path, version: &str) -> PathBuf {
    let safe: String = version
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    app_dir.join(STAGING).join(safe)
}

/// Downloads the release zip, verifies it and unpacks it into the staging folder, which is
/// returned. `progress(done_bytes, total_bytes)`.
pub fn download(
    app_dir: &Path,
    plan: &UpdatePlan,
    cancel: &UpdateCancel,
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf> {
    let staging = staging_dir(app_dir, &plan.version);
    if staging.exists() {
        fs::remove_dir_all(&staging).context("clearing an earlier update attempt")?;
    }
    fs::create_dir_all(&staging)?;
    let mut response = client()?
        .get(&plan.zip_url)
        .send()
        .map_err(|error| error.without_url())
        .context("downloading the update")?
        .error_for_status()
        .map_err(|error| error.without_url())
        .context("downloading the update")?;
    let mut zip = Vec::with_capacity(usize::try_from(plan.bytes).unwrap_or(0));
    let mut buffer = vec![0_u8; 1 << 20];
    progress(0, plan.bytes);
    loop {
        cancel.check()?;
        let count = response.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        zip.extend_from_slice(&buffer[..count]);
        ensure!(
            zip.len() as u64 <= plan.bytes,
            "the update is larger than announced"
        );
        progress(zip.len() as u64, plan.bytes);
    }
    ensure!(
        zip.len() as u64 == plan.bytes,
        "the update download was incomplete"
    );
    if let Some(expected) = &plan.sha256 {
        let digest: String = Sha256::digest(&zip)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        ensure!(
            digest.eq_ignore_ascii_case(expected),
            "the update failed its checksum"
        );
    }
    cancel.check()?;
    unpack_zip(&zip, &staging)?;
    Ok(staging)
}

fn old_name(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{OLD_SUFFIX}"));
    path.with_file_name(name)
}

fn staged_files(staging: &Path) -> Result<Vec<PathBuf>> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                walk(root, &path, out)?;
            } else {
                out.push(path.strip_prefix(root)?.to_path_buf());
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(staging, staging, &mut files)?;
    files.sort();
    Ok(files)
}

/// Swaps every staged file in. Each current file is renamed to `*.brp-old` first (possible while
/// it runs); on any failure every swapped file is restored. Writes `release.json` last.
pub fn apply(app_dir: &Path, plan: &UpdatePlan, staging: &Path) -> Result<()> {
    let files = staged_files(staging)?;
    ensure!(!files.is_empty(), "nothing to install");
    let mut swapped: Vec<(PathBuf, Option<PathBuf>)> = Vec::new();
    let result = (|| -> Result<()> {
        for relative in &files {
            let source = staging.join(relative);
            let destination = app_dir.join(relative);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            let backup = if destination.exists() {
                let backup = old_name(&destination);
                if backup.exists() {
                    fs::remove_file(&backup).with_context(|| {
                        format!("removing an old copy of {}", relative.display())
                    })?;
                }
                fs::rename(&destination, &backup)
                    .with_context(|| format!("moving the current {} aside", relative.display()))?;
                Some(backup)
            } else {
                None
            };
            swapped.push((destination.clone(), backup));
            fs::rename(&source, &destination)
                .with_context(|| format!("installing {}", relative.display()))?;
        }
        let release = serde_json::to_vec_pretty(&InstalledRelease {
            version: plan.version.clone(),
        })?;
        fs::write(app_dir.join(RELEASE_FILE), release)?;
        Ok(())
    })();
    if let Err(error) = result {
        for (destination, backup) in swapped.into_iter().rev() {
            let _ = fs::remove_file(&destination);
            if let Some(backup) = backup {
                let _ = fs::rename(&backup, &destination);
            }
        }
        return Err(error);
    }
    let _ = fs::remove_dir_all(app_dir.join(STAGING));
    Ok(())
}

/// Removes `*.brp-old` files left by the previous update and stale staging folders. Files still
/// in use (the previous process has not exited yet) are skipped.
pub fn cleanup_previous(app_dir: &Path) {
    fn walk(dir: &Path, depth: usize) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() && depth < 6 {
                walk(&path, depth + 1);
            } else if kind.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case(OLD_SUFFIX))
            {
                let _ = fs::remove_file(&path);
            }
        }
    }
    walk(app_dir, 0);
    let _ = fs::remove_dir_all(app_dir.join(STAGING));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn versions_compare_numerically() {
        assert_eq!(
            compare_versions("2026.11.01.1", "2026.10.31.9"),
            Ordering::Greater
        );
        assert_eq!(
            compare_versions("v2026.11.01", "2026.11.01.0"),
            Ordering::Equal
        );
        assert_eq!(
            compare_versions("2026.10.25.3-native-test", "2026.10.25.3"),
            Ordering::Equal
        );
        assert_eq!(compare_versions("0", "2026.1.1"), Ordering::Less);
        assert_eq!(
            compare_versions("2026.10.10", "2026.10.9"),
            Ordering::Greater
        );
    }

    #[test]
    fn unsafe_paths_are_rejected() {
        assert!(relative_path("models/sam3.1/frame/fp16/model.onnx").is_ok());
        assert!(relative_path("../evil.exe").is_err());
        assert!(relative_path("C:/Windows/evil.exe").is_err());
        assert!(relative_path(".update-staging/x").is_err());
        assert!(relative_path("").is_err());
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "brp-updater-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A minimal zip writer (stored and deflated entries) under a top-level folder.
    fn make_zip(files: &[(&str, &[u8], bool)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut directory = Vec::new();
        let push16 = |v: &mut Vec<u8>, x: u16| v.extend_from_slice(&x.to_le_bytes());
        let push32 = |v: &mut Vec<u8>, x: u32| v.extend_from_slice(&x.to_le_bytes());
        for (name, bytes, deflate) in files {
            let name = format!("BRP-ChromaKey2/{name}");
            let data = if *deflate {
                miniz_oxide::deflate::compress_to_vec(bytes, 6)
            } else {
                bytes.to_vec()
            };
            let method: u16 = if *deflate { 8 } else { 0 };
            let offset = out.len() as u32;
            push32(&mut out, 0x0403_4b50);
            for value in [20, 0, method, 0, 0] {
                push16(&mut out, value);
            }
            push32(&mut out, 0);
            push32(&mut out, data.len() as u32);
            push32(&mut out, bytes.len() as u32);
            push16(&mut out, name.len() as u16);
            push16(&mut out, 3); // local extra differs from the directory on purpose
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&[0, 0, 0]);
            out.extend_from_slice(&data);
            push32(&mut directory, 0x0201_4b50);
            for value in [20, 20, 0, method, 0, 0] {
                push16(&mut directory, value);
            }
            push32(&mut directory, 0);
            push32(&mut directory, data.len() as u32);
            push32(&mut directory, bytes.len() as u32);
            push16(&mut directory, name.len() as u16);
            for value in [0, 0, 0, 0] {
                push16(&mut directory, value);
            }
            push32(&mut directory, 0);
            push32(&mut directory, offset);
            directory.extend_from_slice(name.as_bytes());
        }
        let directory_offset = out.len() as u32;
        out.extend_from_slice(&directory);
        push32(&mut out, 0x0605_4b50);
        for value in [0, 0, files.len() as u16, files.len() as u16] {
            push16(&mut out, value);
        }
        push32(&mut out, directory.len() as u32);
        push32(&mut out, directory_offset);
        push16(&mut out, 0);
        out
    }

    /// Serves `body` once over plain HTTP on localhost; returns the URL.
    fn serve(body: Vec<u8>) -> String {
        use std::io::{BufRead, BufReader};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap() > 2 {
                line.clear();
            }
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(header.as_bytes()).unwrap();
            stream.write_all(&body).unwrap();
        });
        format!("http://127.0.0.1:{port}/BRP-ChromaKey2-9-windows-x64.zip")
    }

    fn plan_for(zip: &[u8], url: String, sha256: Option<String>) -> UpdatePlan {
        UpdatePlan {
            version: "2026.11.01.1".into(),
            notes: String::new(),
            page: String::new(),
            bytes: zip.len() as u64,
            zip_url: url,
            sha256,
        }
    }

    #[test]
    fn downloads_unpacks_and_replaces_every_file() {
        let app = temp_dir("zip");
        fs::write(app.join("app.exe"), b"old exe").unwrap();
        fs::write(app.join("kept-local.txt"), b"user file").unwrap();
        let big: Vec<u8> = (0..300_000).map(|i| (i % 251) as u8).collect();
        let zip = make_zip(&[
            ("app.exe", b"new exe", true),
            ("same.dll", b"same", false),
            ("models/new/model.bin", &big, true),
        ]);
        let digest: String = Sha256::digest(&zip)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let plan = plan_for(&zip, serve(zip.clone()), Some(digest));
        let mut last = (0, 1);
        let staging =
            download(&app, &plan, &UpdateCancel::default(), |d, t| last = (d, t)).unwrap();
        assert_eq!(last, (zip.len() as u64, zip.len() as u64));
        apply(&app, &plan, &staging).unwrap();
        assert_eq!(fs::read(app.join("app.exe")).unwrap(), b"new exe");
        assert_eq!(fs::read(app.join("same.dll")).unwrap(), b"same");
        assert_eq!(fs::read(app.join("models/new/model.bin")).unwrap(), big);
        assert_eq!(fs::read(app.join("kept-local.txt")).unwrap(), b"user file");
        assert_eq!(fs::read(app.join("app.exe.brp-old")).unwrap(), b"old exe");
        assert_eq!(installed_version(&app), "2026.11.01.1");
        assert!(!app.join(STAGING).exists());
        cleanup_previous(&app);
        assert!(!app.join("app.exe.brp-old").exists());
        fs::remove_dir_all(&app).unwrap();
    }

    #[test]
    fn a_zip_that_fails_its_checksum_is_not_installed() {
        let app = temp_dir("checksum");
        let zip = make_zip(&[("x.dll", b"tampered", false)]);
        let plan = plan_for(&zip, serve(zip.clone()), Some("00".repeat(32)));
        assert!(download(&app, &plan, &UpdateCancel::default(), |_, _| {}).is_err());
        assert!(!app.join("x.dll").exists());
        fs::remove_dir_all(&app).unwrap();
    }

    #[test]
    fn failed_apply_restores_every_swapped_file() {
        let app = temp_dir("rollback");
        fs::write(app.join("a.dll"), b"old a").unwrap();
        fs::write(app.join("b.dll"), b"old b").unwrap();
        // b.dll is in use as a folder name, so installing it fails after a.dll was swapped.
        let staging = staging_dir(&app, "2");
        fs::create_dir_all(staging.join("b.dll")).unwrap();
        fs::write(staging.join("a.dll"), b"new a").unwrap();
        fs::write(staging.join("b.dll").join("inner"), b"x").unwrap();
        fs::remove_file(app.join("b.dll")).unwrap();
        fs::write(app.join("b.dll"), b"old b").unwrap();
        let plan = plan_for(&[], String::new(), None);
        assert!(apply(&app, &plan, &staging).is_err());
        assert_eq!(fs::read(app.join("a.dll")).unwrap(), b"old a");
        assert_eq!(fs::read(app.join("b.dll")).unwrap(), b"old b");
        assert_eq!(installed_version(&app), "0");
        fs::remove_dir_all(&app).unwrap();
    }
}
