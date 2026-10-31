//! In-app updates from GitHub releases.
//!
//! Every release of `UPDATE_REPOSITORY` carries `UPDATE_MANIFEST_ASSET`: the complete list of app
//! files (relative path, size, sha256, download URL). Files that did not change keep pointing at
//! the older release asset that already holds them, so a release uploads only what changed and
//! the app downloads only files whose local copy differs.
//!
//! Flow: [`check_latest`] (GitHub API, anonymous) -> [`plan`] (hash local files) -> [`download`]
//! (staged under `.update-staging/<version>`, sha256 verified) -> [`apply`] (each file is swapped
//! in by renaming the current one to `*.brp-old`, with rollback on failure; Windows allows renaming
//! running executables and loaded DLLs) -> restart. [`cleanup_previous`] removes `*.brp-old` files
//! at the next start. `release.json` records the installed version.

use anyhow::{Context, Result, bail, ensure};
use reqwest::{blocking::Client, header};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::time::Duration;

/// Public repository whose releases the app updates from.
pub const UPDATE_REPOSITORY: &str = "brurpo/BRP-ChromaKey2";
/// Release asset holding the [`UpdateManifest`].
pub const UPDATE_MANIFEST_ASSET: &str = "brp-chroma-update.json";
/// Installed release, written by the packager and by [`apply`].
pub const RELEASE_FILE: &str = "release.json";
const STAGING: &str = ".update-staging";
const OLD_SUFFIX: &str = "brp-old";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct UpdateFile {
    /// Relative path inside the app folder, `/`-separated.
    pub path: String,
    pub size: u64,
    pub sha256: String,
    pub url: String,
}

/// Every app version that has this updater reads this manifest from the latest release, so any
/// installed version updates straight to the newest one (files are compared by sha256, not by
/// version history). Schema 1 is frozen: later releases may only add optional fields (unknown
/// fields are ignored here); a breaking change needs a new asset name next to this one.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct UpdateManifest {
    pub schema_version: u32,
    pub version: String,
    #[serde(default)]
    pub notes: String,
    pub files: Vec<UpdateFile>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InstalledRelease {
    pub version: String,
}

/// A newer release and what it would change here.
#[derive(Clone, Debug)]
pub struct UpdatePlan {
    pub version: String,
    pub notes: String,
    pub page: String,
    /// Files whose local copy is missing or different.
    pub files: Vec<UpdateFile>,
    pub bytes: u64,
    pub total_files: usize,
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
    Ok(Client::builder()
        .user_agent(concat!("BRP-Chroma-updater/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(10))
        .timeout(None)
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
}

/// The latest release's manifest when it is newer than `current`.
pub fn check_latest(repository: &str, current: &str) -> Result<Option<(UpdateManifest, String)>> {
    let client = client()?;
    let response = client
        .get(format!(
            "https://api.github.com/repos/{repository}/releases/latest"
        ))
        .header(header::ACCEPT, "application/vnd.github+json")
        .send()
        .context("contacting GitHub")?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
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
    if release.draft || release.prerelease {
        return Ok(None);
    }
    let Some(asset) = release
        .assets
        .iter()
        .find(|asset| asset.name == UPDATE_MANIFEST_ASSET)
    else {
        return Ok(None);
    };
    let mut manifest: UpdateManifest = serde_json::from_slice(
        &client
            .get(&asset.browser_download_url)
            .send()?
            .error_for_status()
            .context("downloading the update manifest")?
            .bytes()?,
    )
    .context("reading the update manifest")?;
    ensure!(
        manifest.schema_version == 1,
        "unsupported update manifest {}",
        manifest.schema_version
    );
    if manifest.version.trim().is_empty() {
        manifest.version = release.tag_name.clone();
    }
    if manifest.notes.is_empty() {
        manifest.notes = release.body.unwrap_or_default();
    }
    for file in &manifest.files {
        relative_path(&file.path)?;
        ensure!(
            file.url.starts_with("https://"),
            "update file {} has no HTTPS URL",
            file.path
        );
    }
    if compare_versions(&manifest.version, current) != Ordering::Greater {
        return Ok(None);
    }
    Ok(Some((manifest, release.html_url)))
}

/// Validates a manifest path: relative, `/`-separated, no `..`, no reserved staging folder.
fn relative_path(path: &str) -> Result<PathBuf> {
    let candidate = Path::new(path);
    ensure!(
        !path.is_empty() && !path.contains('\\'),
        "invalid update path {path}"
    );
    let mut out = PathBuf::new();
    for part in candidate.components() {
        match part {
            Component::Normal(name) => out.push(name),
            _ => bail!("unsafe update path {path}"),
        }
    }
    ensure!(
        !out.starts_with(STAGING),
        "update path inside the staging folder: {path}"
    );
    Ok(out)
}

fn sha256_file(path: &Path, cancel: &UpdateCancel) -> Result<String> {
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        cancel.check()?;
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// True when the local copy of `file` already matches the manifest.
fn up_to_date(app_dir: &Path, file: &UpdateFile, cancel: &UpdateCancel) -> Result<bool> {
    let path = app_dir.join(relative_path(&file.path)?);
    let Ok(metadata) = fs::metadata(&path) else {
        return Ok(false);
    };
    if !metadata.is_file() || metadata.len() != file.size {
        return Ok(false);
    }
    Ok(sha256_file(&path, cancel)?.eq_ignore_ascii_case(&file.sha256))
}

/// Which files of `manifest` differ from the local app folder.
pub fn plan(
    app_dir: &Path,
    manifest: &UpdateManifest,
    page: String,
    cancel: &UpdateCancel,
) -> Result<UpdatePlan> {
    let mut files = Vec::new();
    for file in &manifest.files {
        if !up_to_date(app_dir, file, cancel)? {
            files.push(file.clone());
        }
    }
    Ok(UpdatePlan {
        version: manifest.version.clone(),
        notes: manifest.notes.clone(),
        page,
        bytes: files.iter().map(|f| f.size).sum(),
        total_files: manifest.files.len(),
        files,
    })
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

/// Downloads every planned file into the staging folder and verifies it. Already staged and
/// verified files (from an interrupted attempt) are kept. `progress(done_bytes, total_bytes)`.
pub fn download(
    app_dir: &Path,
    plan: &UpdatePlan,
    cancel: &UpdateCancel,
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf> {
    let staging = staging_dir(app_dir, &plan.version);
    let client = client()?;
    let mut done = 0;
    progress(0, plan.bytes);
    for file in &plan.files {
        cancel.check()?;
        let target = staging.join(relative_path(&file.path)?);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        let staged_ok = fs::metadata(&target).is_ok_and(|m| m.len() == file.size)
            && sha256_file(&target, cancel)?.eq_ignore_ascii_case(&file.sha256);
        if !staged_ok {
            let part = target.with_extension("part");
            let mut response = client
                .get(&file.url)
                .send()
                .with_context(|| format!("downloading {}", file.path))?
                .error_for_status()
                .with_context(|| format!("downloading {}", file.path))?;
            let mut output = File::create(&part)?;
            let mut hash = Sha256::new();
            let mut buffer = vec![0_u8; 1 << 20];
            let mut received = 0_u64;
            loop {
                cancel.check()?;
                let count = response.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                output.write_all(&buffer[..count])?;
                hash.update(&buffer[..count]);
                received += count as u64;
                ensure!(
                    received <= file.size,
                    "{} is larger than expected",
                    file.path
                );
                progress(done + received, plan.bytes);
            }
            output.sync_all()?;
            drop(output);
            let digest: String = hash
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            if received != file.size || !digest.eq_ignore_ascii_case(&file.sha256) {
                let _ = fs::remove_file(&part);
                bail!("{} failed verification", file.path);
            }
            if target.exists() {
                fs::remove_file(&target)?;
            }
            fs::rename(&part, &target)?;
        }
        done += file.size;
        progress(done, plan.bytes);
    }
    Ok(staging)
}

fn old_name(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{OLD_SUFFIX}"));
    path.with_file_name(name)
}

/// Swaps the staged files in. Each current file is renamed to `*.brp-old` first (possible while
/// it runs); on any failure every swapped file is restored. Writes `release.json` last.
pub fn apply(app_dir: &Path, plan: &UpdatePlan, staging: &Path) -> Result<()> {
    let mut swapped: Vec<(PathBuf, Option<PathBuf>)> = Vec::new();
    let result = (|| -> Result<()> {
        for file in &plan.files {
            let relative = relative_path(&file.path)?;
            let source = staging.join(&relative);
            let destination = app_dir.join(&relative);
            ensure!(source.is_file(), "{} was not downloaded", file.path);
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            let backup = if destination.exists() {
                let backup = old_name(&destination);
                if backup.exists() {
                    fs::remove_file(&backup)
                        .with_context(|| format!("removing an old copy of {}", file.path))?;
                }
                fs::rename(&destination, &backup)
                    .with_context(|| format!("moving the current {} aside", file.path))?;
                Some(backup)
            } else {
                None
            };
            swapped.push((destination.clone(), backup));
            fs::rename(&source, &destination)
                .with_context(|| format!("installing {}", file.path))?;
        }
        let release = serde_json::to_vec_pretty(&InstalledRelease {
            version: plan.version.clone(),
        })?;
        fs::write(app_dir.join(RELEASE_FILE), release)?;
        Ok(())
    })();
    if let Err(error) = result {
        for (destination, backup) in swapped.into_iter().rev() {
            if let Some(backup) = backup {
                let _ = fs::remove_file(&destination);
                let _ = fs::rename(&backup, &destination);
            }
        }
        return Err(error);
    }
    let _ = fs::remove_dir_all(staging);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_accepts_future_optional_fields() {
        let manifest: UpdateManifest = serde_json::from_str(
            r#"{"schema_version":1,"version":"2027.1.1","min_app":"x","files":[
                {"path":"a.dll","size":1,"sha256":"00","url":"https://x/a","compressed":true}]}"#,
        )
        .unwrap();
        assert_eq!(manifest.files.len(), 1);
        assert!(manifest.notes.is_empty());
    }

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
        assert!(relative_path("a\\b").is_err());
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

    fn entry(path: &str, bytes: &[u8]) -> UpdateFile {
        let mut hash = Sha256::new();
        hash.update(bytes);
        UpdateFile {
            path: path.into(),
            size: bytes.len() as u64,
            sha256: hash.finalize().iter().map(|b| format!("{b:02x}")).collect(),
            url: "https://example.invalid/x".into(),
        }
    }

    #[test]
    fn plan_lists_only_changed_files_and_apply_swaps_them() {
        let app = temp_dir("apply");
        fs::write(app.join("same.dll"), b"same").unwrap();
        fs::write(app.join("app.exe"), b"old exe").unwrap();
        let manifest = UpdateManifest {
            schema_version: 1,
            version: "2026.11.01.1".into(),
            notes: String::new(),
            files: vec![
                entry("same.dll", b"same"),
                entry("app.exe", b"new exe"),
                entry("models/new/model.onnx", b"model"),
            ],
        };
        let cancel = UpdateCancel::default();
        let plan = plan(&app, &manifest, String::new(), &cancel).unwrap();
        let changed: Vec<&str> = plan.files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(changed, ["app.exe", "models/new/model.onnx"]);
        assert_eq!(plan.bytes, 7 + 5);
        // Stage as download() would.
        let staging = staging_dir(&app, &plan.version);
        fs::create_dir_all(staging.join("models/new")).unwrap();
        fs::write(staging.join("app.exe"), b"new exe").unwrap();
        fs::write(staging.join("models/new/model.onnx"), b"model").unwrap();
        apply(&app, &plan, &staging).unwrap();
        assert_eq!(fs::read(app.join("app.exe")).unwrap(), b"new exe");
        assert_eq!(
            fs::read(app.join("models/new/model.onnx")).unwrap(),
            b"model"
        );
        assert_eq!(fs::read(app.join("app.exe.brp-old")).unwrap(), b"old exe");
        assert_eq!(installed_version(&app), "2026.11.01.1");
        assert!(!staging.exists());
        cleanup_previous(&app);
        assert!(!app.join("app.exe.brp-old").exists());
        assert!(plan_is_empty(&app, &manifest));
        fs::remove_dir_all(&app).unwrap();
    }

    fn plan_is_empty(app: &Path, manifest: &UpdateManifest) -> bool {
        plan(app, manifest, String::new(), &UpdateCancel::default())
            .unwrap()
            .files
            .is_empty()
    }

    /// Serves each body once over plain HTTP on localhost; returns the base URL.
    fn serve(bodies: Vec<Vec<u8>>) -> String {
        use std::io::{BufRead, BufReader};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for body in bodies {
                let (mut stream, _) = listener.accept().unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap() > 2 {
                    line.clear();
                }
                let header = format!(
                    "HTTP/1.1 200 OK
Content-Length: {}
Connection: close

",
                    body.len()
                );
                stream.write_all(header.as_bytes()).unwrap();
                stream.write_all(&body).unwrap();
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    #[test]
    fn download_streams_verifies_and_rejects_corrupt_files() {
        let app = temp_dir("download");
        let good = vec![7_u8; 3 * 1024 * 1024 + 5];
        let base = serve(vec![good.clone(), b"tampered".to_vec()]);
        let mut file = entry("bin/big.dll", &good);
        file.url = format!("{base}/big");
        let plan = UpdatePlan {
            version: "3".into(),
            notes: String::new(),
            page: String::new(),
            bytes: file.size,
            files: vec![file],
            total_files: 1,
        };
        let mut last = (0, 0);
        let staging =
            download(&app, &plan, &UpdateCancel::default(), |d, t| last = (d, t)).unwrap();
        assert_eq!(fs::read(staging.join("bin/big.dll")).unwrap(), good);
        assert_eq!(last, (good.len() as u64, good.len() as u64));
        // A file whose bytes do not match the manifest is rejected and not staged.
        let mut bad = entry("x.dll", b"original");
        bad.url = format!("{base}/bad");
        let bad_plan = UpdatePlan {
            version: "4".into(),
            notes: String::new(),
            page: String::new(),
            bytes: bad.size,
            files: vec![bad],
            total_files: 1,
        };
        assert!(download(&app, &bad_plan, &UpdateCancel::default(), |_, _| {}).is_err());
        assert!(!staging_dir(&app, "4").join("x.dll").exists());
        fs::remove_dir_all(&app).unwrap();
    }

    #[test]
    fn failed_apply_restores_every_swapped_file() {
        let app = temp_dir("rollback");
        fs::write(app.join("a.dll"), b"old a").unwrap();
        fs::write(app.join("b.dll"), b"old b").unwrap();
        let plan = UpdatePlan {
            version: "2".into(),
            notes: String::new(),
            page: String::new(),
            files: vec![entry("a.dll", b"new a"), entry("b.dll", b"new b")],
            bytes: 10,
            total_files: 2,
        };
        let staging = staging_dir(&app, "2");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("a.dll"), b"new a").unwrap(); // b.dll missing -> failure
        assert!(apply(&app, &plan, &staging).is_err());
        assert_eq!(fs::read(app.join("a.dll")).unwrap(), b"old a");
        assert_eq!(fs::read(app.join("b.dll")).unwrap(), b"old b");
        assert_eq!(installed_version(&app), "0");
        fs::remove_dir_all(&app).unwrap();
    }
}
