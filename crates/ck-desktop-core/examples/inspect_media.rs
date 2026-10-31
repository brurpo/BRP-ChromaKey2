//! Native developer smoke for the same media/project APIs used by the editor.
use anyhow::{Context, Result};
use ck_desktop_core::{MediaInfo, Project, RuntimePaths, extract_source_frame};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let video = PathBuf::from(
        arguments
            .next()
            .context("usage: inspect_media VIDEO OUTPUT [FRAME]")?,
    );
    let output = PathBuf::from(arguments.next().context("output folder is required")?);
    let frame: u64 = arguments
        .next()
        .map(|s| s.to_string_lossy().parse())
        .transpose()?
        .unwrap_or(9);
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .context("workspace path")?
        .to_path_buf();
    let runtime = RuntimePaths::discover(workspace);
    let media = MediaInfo::probe(video, &runtime)?;
    let image = extract_source_frame(&media, frame, &output, &runtime)?;
    let project = Project::new(media.clone(), output.join("renders"));
    let project_path = output.join("smoke.brp.json");
    project.save(&project_path)?;
    let loaded = Project::load(&project_path)?;
    anyhow::ensure!(
        loaded.media.as_ref().map(|m| &m.path) == Some(&media.path),
        "Project source changed during persistence"
    );
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"media": media, "frame":frame,"frame_seconds":media.fps.frame_seconds(frame),"preview":image,"project":project_path,"source_decode":"native ffmpeg exact display-order frame select"})
        )?
    );
    Ok(())
}
