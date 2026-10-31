//! Explicit CPU integration: move only the .chroma container, open its index
//! lazily, materialize selected blobs, then export mixed original/rendered frames.
use crate::{
    project_store::{self, ChromaProject, Document, FrameFiles, TemporaryWorkspace},
    range_export::{ExportEvent, ExportFormat, ExportRequest, ExportService},
};
use ck_desktop_core::{DespillSettings, FrameRange, MediaInfo, Project, RuntimePaths};
use eframe::egui;
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

const WIDTH: u32 = 64;
const HEIGHT: u32 = 32;
const FRAME_BYTES: usize = WIDTH as usize * HEIGHT as usize * 4;

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

fn runtime() -> RuntimePaths {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let runtime = RuntimePaths::installed(root.join("app"));
    assert!(runtime.ffmpeg.is_file() && runtime.ffprobe.is_file());
    runtime
}

fn create_source(ffmpeg: &Path, source: &Path) {
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
            "rgb24",
            "-video_size",
            "64x32",
            "-framerate",
            "24/1",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            "-g",
            "12",
            "-bf",
            "2",
            "-threads",
            "2",
        ])
        .arg(source)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    hidden(&mut command);
    let mut child = command.spawn().unwrap();
    let mut input = child.stdin.take().unwrap();
    for frame in 0..3_u8 {
        let mut rgb = Vec::with_capacity(WIDTH as usize * HEIGHT as usize * 3);
        for y in 0..HEIGHT as u8 {
            for x in 0..WIDTH as u8 {
                rgb.extend_from_slice(&[
                    30 + frame * 40 + x / 4,
                    40 + frame * 30,
                    70 + frame * 20 + y / 4,
                ]);
            }
        }
        input.write_all(&rgb).unwrap();
    }
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn source_oracle(ffmpeg: &Path, source: &Path) -> Vec<u8> {
    // The expected plate pixels come from independent full-origin decoding,
    // before the original file is removed in the portable case.
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
        .arg(source)
        .args([
            "-an",
            "-frames:v",
            "3",
            "-pix_fmt",
            "rgba",
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
    assert_eq!(output.stdout.len(), FRAME_BYTES * 3);
    output.stdout
}

fn remove_owned_file(workspace: &TemporaryWorkspace, path: &Path) {
    let root = workspace.root.canonicalize().unwrap();
    let target = path.canonicalize().unwrap();
    assert!(target.starts_with(&root) && target != root);
    assert!(target.is_file());
    fs::remove_file(target).unwrap();
}

fn wait_for_export(service: &ExportService) -> (u64, u64, u64) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        assert!(
            !remaining.is_zero(),
            "CPU playback-range export did not complete"
        );
        match service
            .messages
            .recv_timeout(remaining)
            .expect("Export progress service disconnected")
        {
            ExportEvent::Progress { current, total } => assert!(current <= total),
            ExportEvent::Finished {
                frames,
                rendered_frames,
                source_frames,
                ..
            } => return (frames, rendered_frames, source_frames),
            ExportEvent::Failed(error) => panic!("Portable project export failed: {error}"),
            ExportEvent::Cancelled => panic!("Portable project export was unexpectedly cancelled"),
        }
    }
}

fn workflow(embed_sources: bool) {
    let runtime = runtime();
    let workspace = TemporaryWorkspace::new().unwrap();
    let originals = workspace.root.join("originals");
    let project_directory = workspace.root.join("saved");
    let relocated = workspace.root.join("relocated");
    fs::create_dir(&originals).unwrap();
    fs::create_dir(&project_directory).unwrap();
    fs::create_dir(&relocated).unwrap();
    let source = originals.join("plate.mp4");
    let rendered = originals.join("keyed.png");
    create_source(&runtime.ffmpeg, &source);
    let expected_original = source_oracle(&runtime.ffmpeg, &source);
    let media = MediaInfo::probe(&source, &runtime).unwrap();
    assert_eq!(media.frame_count, 3);
    let mut keyed_pixels = [200, 60, 20, 128].repeat(WIDTH as usize * HEIGHT as usize);
    keyed_pixels[..4].copy_from_slice(&[100, 50, 25, 0]);
    image::save_buffer(
        &rendered,
        &keyed_pixels,
        WIDTH,
        HEIGHT,
        image::ColorType::Rgba8,
    )
    .unwrap();
    let range = FrameRange {
        start: 0,
        end_exclusive: 3,
    };
    let project_path = project_directory.join("portable.chroma");
    let mut clip_project = Project::new(media.clone(), originals.join("calculations"));
    clip_project.despill_settings = DespillSettings { amount: 0.37 };
    let document = ChromaProject {
        chroma_version: 1,
        selected: 0,
        documents: vec![Document {
            key: "clip".into(),
            project: clip_project,
            range,
            playhead: 1,
            frames: BTreeMap::from([(
                1,
                FrameFiles {
                    rgba: Some(rendered.clone()),
                    ..Default::default()
                },
            )]),
            masks: Vec::new(),
            hint: None,
        }],
        archive: None,
    };
    let saved = project_store::save(&project_path, document, embed_sources).unwrap();
    assert!(saved.archive.is_some());
    drop(saved); // Release the .chroma file handle before moving it on Windows.

    let moved_project = relocated.join("moved.chroma");
    assert!(
        project_path
            .canonicalize()
            .unwrap()
            .starts_with(workspace.root.canonicalize().unwrap())
    );
    fs::rename(&project_path, &moved_project).unwrap(); // Move only the container.
    remove_owned_file(&workspace, &rendered);
    if embed_sources {
        remove_owned_file(&workspace, &source);
    }
    assert!(!rendered.exists());
    assert_eq!(source.exists(), !embed_sources);
    assert_eq!(fs::read_dir(&relocated).unwrap().count(), 1);

    let start = Instant::now();
    let loaded = project_store::load(&moved_project).unwrap();
    let index_load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let archive = loaded.archive.clone().unwrap();
    let document = &loaded.documents[0];
    assert_eq!(
        document.project.despill_settings,
        DespillSettings { amount: 0.37 }
    );
    let mut export_media = document.project.media.clone().unwrap();
    let rendered_reference = document.frames[&1].rgba.clone().unwrap();
    assert_eq!(document.range, range);
    assert_eq!(document.playhead, 1);
    assert!(archive.is_embedded(&rendered_reference));
    assert!(archive.needs_materialization(&rendered_reference));
    assert!(
        !rendered_reference.is_file(),
        "Loading the index must not extract calculations"
    );
    if embed_sources {
        assert!(archive.is_embedded(&export_media.path));
        assert!(archive.needs_materialization(&export_media.path));
        assert!(
            !export_media.path.is_file(),
            "Loading the index must not extract the embedded plate"
        );
    } else {
        assert!(!archive.is_embedded(&export_media.path));
        assert_eq!(
            export_media.path.canonicalize().unwrap(),
            media.path.canonicalize().unwrap()
        );
        assert!(export_media.path.is_file());
        assert!(
            archive
                .materialized_identity(&export_media.path)
                .unwrap()
                .is_none()
        );
    }
    assert_eq!(
        fs::read_dir(&relocated).unwrap().count(),
        1,
        "No external project sidecar is required"
    );

    archive
        .materialize(&[export_media.path.clone(), rendered_reference.clone()])
        .unwrap();
    if let Some(identity) = archive.materialized_identity(&export_media.path).unwrap() {
        export_media.source_identity = Some(identity);
    }
    export_media.verify_source_unchanged().unwrap();
    assert!(rendered_reference.is_file());
    assert!(!archive.needs_materialization(&rendered_reference));
    assert!(!archive.needs_materialization(&export_media.path));
    let target = workspace.root.join("exported");
    let service = ExportService::start(
        ExportRequest {
            media: export_media.clone(),
            range: document.range,
            rendered: BTreeMap::from([(1, rendered_reference.clone())]),
            target: target.clone(),
            format: ExportFormat::PngSequence,
            despill: DespillSettings::default(),
            despills: BTreeMap::new(),
            masks: None,
        },
        runtime.ffmpeg.clone(),
        egui::Context::default(),
    )
    .unwrap();
    assert_eq!(wait_for_export(&service), (3, 1, 2));
    drop(service);
    assert_eq!(fs::read_dir(&target).unwrap().count(), 3);
    for index in [0, 2] {
        let exported = image::open(target.join(format!("frame_{index:06}.png")))
            .unwrap()
            .to_rgba8();
        assert_eq!(
            exported.as_raw(),
            &expected_original[index * FRAME_BYTES..(index + 1) * FRAME_BYTES],
            "Original fallback source frame {index}"
        );
        assert!(exported.pixels().all(|pixel| pixel[3] == 255));
    }
    let exported = image::open(target.join("frame_000001.png"))
        .unwrap()
        .to_rgba8();
    assert_eq!(exported.as_raw(), &keyed_pixels);
    assert_eq!(exported.get_pixel(0, 0).0, [100, 50, 25, 0]);
    assert_eq!(exported.get_pixel(1, 0).0, [200, 60, 20, 128]);
    let adjusted_target = workspace.root.join("despilled-export");
    let service = ExportService::start(
        ExportRequest {
            media: export_media.clone(),
            range: document.range,
            rendered: BTreeMap::from([(1, rendered_reference.clone())]),
            target: adjusted_target.clone(),
            format: ExportFormat::PngSequence,
            despill: document.project.despill_settings,
            despills: BTreeMap::new(),
            masks: None,
        },
        runtime.ffmpeg.clone(),
        egui::Context::default(),
    )
    .unwrap();
    assert_eq!(wait_for_export(&service), (3, 1, 2));
    drop(service);
    let mut expected_adjusted = keyed_pixels.clone();
    crate::despill::apply_rgba(
        &expected_original[FRAME_BYTES..2 * FRAME_BYTES],
        &mut expected_adjusted,
        WIDTH,
        HEIGHT,
        document.project.despill_settings,
    )
    .unwrap();
    assert_ne!(expected_adjusted, keyed_pixels);
    let adjusted = image::open(adjusted_target.join("frame_000001.png"))
        .unwrap()
        .to_rgba8();
    assert_eq!(adjusted.as_raw(), &expected_adjusted);
    assert!(
        adjusted
            .pixels()
            .zip(keyed_pixels.as_chunks::<4>().0.iter())
            .all(|(actual, before)| actual[3] == before[3])
    );
    for index in [0, 2] {
        assert_eq!(
            image::open(adjusted_target.join(format!("frame_{index:06}.png")))
                .unwrap()
                .to_rgba8()
                .as_raw(),
            &expected_original[index * FRAME_BYTES..(index + 1) * FRAME_BYTES],
        );
    }
    let single = workspace.root.join("despilled-single.png");
    crate::media_tasks::export_single_frame_with_despill(
        &export_media,
        1,
        crate::preview::ViewMode::Composite,
        Some(rendered_reference),
        &single,
        &runtime.ffmpeg,
        document.project.despill_settings,
    )
    .unwrap();
    assert_eq!(
        image::open(single).unwrap().to_rgba8().as_raw(),
        &expected_adjusted
    );
    assert_eq!(fs::read_dir(&relocated).unwrap().count(), 1);
    println!(
        "CPU_PORTABLE_PROJECT_EXPORT {}",
        serde_json::json!({
            "embed_sources":embed_sources,
            "index_load_ms":index_load_ms,
            "original_plate_available":source.exists(),
            "original_render_available":rendered.exists(),
            "index_load_extracted_blobs":0,
            "exported_frames":3,
            "rendered_frames":1,
            "source_fallback_frames":2,
            "source_and_rgba_byte_exact":true,
            "transparent_and_half_alpha_preserved":true,
            "external_project_sidecars":0,
            "inference_executed":false,
            "despill_settings_roundtrip":true,
            "live_despill_frame_and_range_agree":true,
            "despill_preserves_alpha_and_original_fallbacks":true,
        })
    );
    drop(archive);
    drop(loaded);
}

#[test]
#[ignore = "Explicit CPU portable-project workflow; requires installed FFmpeg/FFprobe"]
fn moved_chroma_projects_load_lazily_and_export_original_plus_calculated_rgba() {
    workflow(true);
    workflow(false);
}

/// Development-only: writes `.dev/interface-qa/ui-review.chroma` (CARTOLAS, 16 calculated
/// frames from a native run, three scenes, a scene despill override and a soft garbage mask)
/// for CPU screenshot review with `--software-renderer --ui-only --screenshot`.
#[test]
#[ignore = "Development UI review fixture; needs CARTOLAS and a completed 16-frame run"]
fn build_ui_review_project() {
    use ck_core::mask_layers::{BezierNode, BezierShape, MaskKind, MaskLayer, MaskMode};
    let root = PathBuf::from("D:/BRP_CHROMA_2");
    let run = root
        .join(".dev/experiments/native-integration-20261028/runs/nv-9-selftest/attempt-000-000");
    let source = PathBuf::from("C:/Users/brurp/Downloads/CARTOLAS FINAL_export.mp4");
    let media = MediaInfo::probe(
        &source,
        &RuntimePaths::installed(root.join("app-native-test")),
    )
    .unwrap();
    let qa = root.join(".dev/interface-qa");
    let mut project = Project::new(media.clone(), qa.join("ui-review-calculations"));
    project.scenes = vec![
        ck_desktop_core::SceneRange {
            index: 0,
            start: 0,
            end_exclusive: 9,
            enabled: true,
        },
        ck_desktop_core::SceneRange {
            index: 1,
            start: 9,
            end_exclusive: 16,
            enabled: true,
        },
        ck_desktop_core::SceneRange {
            index: 2,
            start: 16,
            end_exclusive: media.frame_count,
            enabled: false,
        },
    ];
    project.despill_settings = DespillSettings { amount: 0.5 };
    project.scene_overrides.insert(
        1,
        ck_desktop_core::SceneOverrides {
            despill: Some(ck_desktop_core::SceneOverrideValue {
                enabled: true,
                amount: 0.8,
            }),
        },
    );
    let mut layer = MaskLayer::new(1, "Garbage 1", MaskKind::Garbage, media.width, media.height);
    layer.mode = MaskMode::Subtract;
    layer.softness = 30;
    layer.scene_index = 0;
    layer.set_scene_range(0, 8);
    layer.set_shape(
        0,
        BezierShape::with_nodes(
            vec![
                BezierNode::new(1350.0, 120.0),
                BezierNode::new(1850.0, 160.0),
                BezierNode::new(1800.0, 620.0),
                BezierNode::new(1420.0, 560.0),
            ],
            true,
        ),
    );
    project.mask_layers.garbage.push(layer);
    let frames = (0..16u64)
        .map(|frame| {
            let name = format!("{frame:06}.png");
            (
                frame,
                FrameFiles {
                    rgba: Some(run.join("frames").join(&name)),
                    checker: Some(run.join("checker").join(&name)),
                    alpha: Some(run.join("alpha").join(&name)),
                    mask: None,
                    sam_layers: Default::default(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert!(
        frames
            .values()
            .all(|files| files.rgba.as_ref().unwrap().is_file())
    );
    let document = ChromaProject {
        chroma_version: 1,
        selected: 0,
        documents: vec![Document {
            key: "cartolas".into(),
            project,
            range: FrameRange {
                start: 0,
                end_exclusive: 16,
            },
            playhead: 4,
            frames,
            masks: Vec::new(),
            hint: None,
        }],
        archive: None,
    };
    let path = qa.join("ui-review.chroma");
    let _ = fs::remove_file(&path);
    project_store::save(&path, document, false).unwrap();
    println!("wrote {}", path.display());
}
