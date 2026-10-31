//! Development validation of the same installer used by the desktop consent UI.
use anyhow::{Context, Result, bail};
use ck_desktop_core::{
    AssetCancellation, AssetGroup, AssetManager, AssetPhase, read_huggingface_token,
};
use std::{path::PathBuf, time::Instant};

fn main() -> Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag| -> Result<String> {
        arguments
            .windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].clone())
            .with_context(|| format!("Missing {flag}"))
    };
    let root = PathBuf::from(value("--app-dir")?);
    let catalog = PathBuf::from(value("--catalog")?);
    let groups: Vec<AssetGroup> = value("--groups")?
        .split(',')
        .map(|group| match group {
            "core" => Ok(AssetGroup::Core),
            "runtime" => Ok(AssetGroup::Runtime),
            "ck1080" => Ok(AssetGroup::Ck1080),
            "sam" => Ok(AssetGroup::Sam),
            "transnet" => Ok(AssetGroup::Transnet),
            "ck4k" => Ok(AssetGroup::Ck4k),
            _ => bail!("Unknown asset group"),
        })
        .collect::<Result<_>>()?;
    let consent = arguments.iter().any(|argument| argument == "--consent");
    let anonymous = arguments.iter().any(|argument| argument == "--anonymous");
    let report = arguments
        .iter()
        .position(|argument| argument == "--report")
        .and_then(|position| arguments.get(position + 1))
        .map(PathBuf::from);
    let cancel_after = arguments
        .iter()
        .position(|argument| argument == "--cancel-after-bytes")
        .and_then(|position| arguments.get(position + 1))
        .map(|text| text.parse::<u64>())
        .transpose()?;
    let started = Instant::now();
    let mut manager = AssetManager::from_manifest_file(root, catalog)?;
    let before = manager.inspect(&groups)?;
    println!(
        "{}",
        serde_json::json!({"phase":"inspection","ready":before.ready,
            "missing_files":before.missing.len(),"download_bytes":before.download_bytes,
            "reusable_bytes":before.reusable_bytes,"consent":consent})
    );
    let mut install_error = None;
    let status = if consent && !before.ready {
        if !anonymous && let Some(token) = read_huggingface_token()? {
            manager = manager.with_huggingface_token(token);
        }
        let cancel = AssetCancellation::default();
        let mut last_phase = None;
        let mut last_emit = Instant::now();
        match manager.install(&groups, &cancel, |progress| {
            if cancel_after.is_some_and(|limit| {
                progress.phase == AssetPhase::Downloading && progress.downloaded >= limit
            }) {
                cancel.cancel();
            }
            if last_phase != Some((progress.file_index, progress.phase))
                || last_emit.elapsed().as_secs() >= 5
            {
                println!(
                    "{}",
                    serde_json::json!({"phase":format!("{:?}",progress.phase),
                        "file":progress.path,"downloaded":progress.downloaded,
                        "total":progress.total,"index":progress.file_index,
                        "files":progress.file_count})
                );
                last_phase = Some((progress.file_index, progress.phase));
                last_emit = Instant::now();
            }
        }) {
            Ok(status) => status,
            Err(error) => {
                install_error = Some(format!("{error:#}"));
                manager.inspect(&groups)?
            }
        }
    } else {
        before
    };
    let result = serde_json::json!({"schema_version":1,"ready":status.ready,
        "consent":consent,"anonymous":anonymous,"missing_files":status.missing.len(),
        "download_bytes":status.download_bytes,"reusable_bytes":status.reusable_bytes,
        "total_bytes":status.total_bytes,"verified_bytes":status.verified_bytes,
        "elapsed_seconds":started.elapsed().as_secs_f64(),"error":install_error,
        "catalog_id":manager.manifest().id,"catalog_version":manager.manifest().version});
    if let Some(path) = report {
        std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
    }
    println!("{result}");
    if consent && !status.ready {
        bail!("Asset installation did not complete; see the result report");
    }
    Ok(())
}
