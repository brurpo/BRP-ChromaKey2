use super::*;
use ck_desktop_core::{Rational, RuntimePaths};

fn media() -> MediaInfo {
    MediaInfo {
        path: PathBuf::from("source.mov"),
        width: 64,
        height: 32,
        frame_count: 5,
        fps: Rational { num: 24, den: 1 },
        time_base: Rational { num: 1, den: 12288 },
        first_pts: 0,
        has_b_frames: false,
        source_identity: None,
    }
}

fn request(target: PathBuf, format: ExportFormat) -> ExportRequest {
    ExportRequest {
        media: media(),
        range: FrameRange {
            start: 1,
            end_exclusive: 4,
        },
        rendered: BTreeMap::new(),
        target,
        format,
        despill: DespillSettings::default(),
        despills: BTreeMap::new(),
        masks: None,
    }
}

struct TestFolder(PathBuf);
impl TestFolder {
    fn new() -> Self {
        let temporary = std::env::temp_dir();
        let path = temporary.join(format!(
            "brp-range-export-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for TestFolder {
    fn drop(&mut self) {
        // This test creates the unique directory itself; never recursively
        // remove a path supplied by a user or outside the actual temp root.
        if let (Ok(path), Ok(temporary)) =
            (self.0.canonicalize(), std::env::temp_dir().canonicalize())
            && path.parent() == Some(temporary.as_path())
            && path
                .file_name()
                .is_some_and(|name| name.to_string_lossy().starts_with("brp-range-export-test-"))
        {
            let _ = fs::remove_dir_all(path);
        }
    }
}

#[test]
fn linear_export_transfer_preserves_black_white_and_straight_alpha() {
    assert_eq!(srgb_to_linear(0), 0.0);
    assert_eq!(srgb_to_linear(255), 1.0);
    assert!((srgb_to_linear(128) - 0.215_860_53).abs() < 1e-7);
    let folder = TestFolder::new();
    let path = folder.0.join("one.exr");
    write_linear_exr(&path, 1, 1, &[128, 255, 0, 64], &ExportControl::default()).unwrap();
    let bytes = fs::read(path).unwrap();
    assert_eq!(&bytes[..8], &[0x76, 0x2f, 0x31, 0x01, 2, 0, 0, 0]);
    // The last scanline stores straight A, B, G, R floats. RGB does not get
    // multiplied by alpha while transferring display values to scene linear.
    let values = bytes[bytes.len() - 16..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|value| f32::from_le_bytes(*value))
        .collect::<Vec<_>>();
    assert!((values[0] - 64.0 / 255.0).abs() < 1e-7);
    assert_eq!(values[1], 0.0);
    assert_eq!(values[2], 1.0);
    assert!((values[3] - srgb_to_linear(128)).abs() < 1e-7);
}

#[test]
fn export_refuses_occupied_destinations_without_modifying_them() {
    let folder = TestFolder::new();
    let file = folder.0.join("existing.mov");
    fs::write(&file, b"user file").unwrap();
    assert!(validate_request(&request(file.clone(), ExportFormat::ProRes4444)).is_err());
    assert_eq!(fs::read(file).unwrap(), b"user file");
    let sequence = folder.0.join("sequence");
    fs::create_dir(&sequence).unwrap();
    fs::write(sequence.join("unrelated.txt"), b"keep").unwrap();
    assert!(validate_request(&request(sequence.clone(), ExportFormat::PngSequence)).is_err());
    assert_eq!(fs::read(sequence.join("unrelated.txt")).unwrap(), b"keep");
}

#[test]
fn failed_publication_preserves_a_file_that_appeared_during_export() {
    let folder = TestFolder::new();
    let target = folder.0.join("collision.mov");
    let staged_path;
    {
        let mut staging = Staging::new(&target, false).unwrap();
        staged_path = staging.path.clone();
        fs::write(&staging.path, b"new export").unwrap();
        fs::write(&target, b"another user file").unwrap();
        assert!(staging.publish(&target, &ExportControl::default()).is_err());
    }
    assert!(!staged_path.exists());
    assert_eq!(fs::read(target).unwrap(), b"another user file");
}

#[test]
fn fully_rendered_png_range_exports_without_opening_missing_original_media() {
    let folder = TestFolder::new();
    let rendered = folder.0.join("rendered.png");
    let pixels = [90, 60, 30, 128].repeat(64 * 32);
    image::save_buffer(&rendered, &pixels, 64, 32, image::ColorType::Rgba8).unwrap();
    let mut request = request(folder.0.join("complete"), ExportFormat::PngSequence);
    request.media.path = folder.0.join("offline-original.mov");
    request.rendered = (1..4).map(|frame| (frame, rendered.clone())).collect();
    let control = ExportControl::default();
    assert_eq!(
        export(&request, Path::new("not-used-ffmpeg"), &control, |_, _| {}).unwrap(),
        3
    );
    assert!(control.children.lock().unwrap().is_empty());
    assert_eq!(fs::read_dir(&request.target).unwrap().count(), 3);
    assert_eq!(
        image::open(request.target.join("frame_000002.png"))
            .unwrap()
            .to_rgba8()
            .as_raw(),
        &pixels
    );
}

#[test]
fn rendered_range_applies_garbage_masks_to_the_exported_alpha() {
    let folder = TestFolder::new();
    let rendered = folder.0.join("rendered.png");
    let pixels = [90, 60, 30, 255].repeat(64 * 32);
    image::save_buffer(&rendered, &pixels, 64, 32, image::ColorType::Rgba8).unwrap();
    let mut stacks = ck_desktop_core::MaskStacks::new(64, 32);
    let layer = stacks
        .garbage
        .add(0, Some(ck_desktop_core::MaskMode::Subtract));
    layer.set_shape(
        0,
        ck_core::mask_layers::BezierShape::with_nodes(
            vec![
                ck_core::mask_layers::BezierNode::new(0.0, 0.0),
                ck_core::mask_layers::BezierNode::new(32.0, 0.0),
                ck_core::mask_layers::BezierNode::new(32.0, 32.0),
                ck_core::mask_layers::BezierNode::new(0.0, 32.0),
            ],
            true,
        ),
    );
    let mut request = request(folder.0.join("masked"), ExportFormat::PngSequence);
    request.media.path = folder.0.join("offline-original.mov");
    request.rendered = (1..4).map(|frame| (frame, rendered.clone())).collect();
    request.masks = Some(std::sync::Arc::new(stacks));
    let control = ExportControl::default();
    assert_eq!(
        export(&request, Path::new("not-used-ffmpeg"), &control, |_, _| {}).unwrap(),
        3
    );
    let exported = image::open(request.target.join("frame_000002.png"))
        .unwrap()
        .to_rgba8();
    assert_eq!(exported.get_pixel(0, 0).0, [90, 60, 30, 0]);
    assert_eq!(exported.get_pixel(63, 0).0, [90, 60, 30, 255]);
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

fn ffmpeg_output(ffmpeg: &Path, input: &Path, pixel_format: &str, frames: usize) -> Vec<u8> {
    let mut command = Command::new(ffmpeg);
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-hwaccel",
            "none",
            "-i",
        ])
        .arg(input)
        .args([
            "-an",
            "-frames:v",
            &frames.to_string(),
            "-pix_fmt",
            pixel_format,
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
    output.stdout
}

#[test]
#[ignore = "Explicit CPU-only export fixture; requires installed native FFmpeg/FFprobe"]
fn mixed_range_exports_png_linear_exr_and_prores_with_alpha() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let runtime = RuntimePaths::installed(root.join("app"));
    assert!(runtime.ffmpeg.is_file());
    let folder = TestFolder::new();
    let source = folder.0.join("source.mov");
    let mut command = Command::new(&runtime.ffmpeg);
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
            "64x32",
            "-framerate",
            "24/1",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "png",
            "-threads:v",
            "1",
        ])
        .arg(&source)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    hidden(&mut command);
    let mut child = command.spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for frame in 0..5u8 {
        let rgba = [frame * 30, 40, 80, 64].repeat(64 * 32);
        stdin.write_all(&rgba).unwrap();
    }
    drop(stdin);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let media = MediaInfo::probe(&source, &runtime).unwrap();
    let rendered = folder.0.join("keyed.png");
    let mut keyed = [200, 80, 40, 128].repeat(64 * 32);
    keyed[..4].copy_from_slice(&[100, 60, 20, 0]);
    image::save_buffer(&rendered, &keyed, 64, 32, image::ColorType::Rgba8).unwrap();
    let mut rendered_frames = BTreeMap::from([(2, rendered)]);
    rendered_frames.insert(3, folder.0.join("not-rendered.png"));
    let mut request = ExportRequest {
        media,
        range: FrameRange {
            start: 1,
            end_exclusive: 4,
        },
        rendered: rendered_frames,
        target: folder.0.join("png"),
        format: ExportFormat::PngSequence,
        despill: DespillSettings::default(),
        despills: BTreeMap::new(),
        masks: None,
    };
    fs::create_dir(&request.target).unwrap(); // An existing empty user folder is valid.
    let mut progress = Vec::new();
    assert_eq!(
        export(
            &request,
            &runtime.ffmpeg,
            &ExportControl::default(),
            |current, total| progress.push((current, total))
        )
        .unwrap(),
        1
    );
    assert_eq!(progress.first(), Some(&(0, 3)));
    assert_eq!(progress.last(), Some(&(3, 3)));
    assert_eq!(fs::read_dir(&request.target).unwrap().count(), 3);
    for (output_frame, source_frame) in [(0, 1u8), (2, 3u8)] {
        let rgba = image::open(request.target.join(format!("frame_{output_frame:06}.png")))
            .unwrap()
            .to_rgba8();
        assert_eq!(
            rgba.as_raw(),
            &[source_frame * 30, 40, 80, 255].repeat(64 * 32)
        );
    }
    assert_eq!(
        image::open(request.target.join("frame_000001.png"))
            .unwrap()
            .to_rgba8()
            .as_raw(),
        &keyed
    );

    request.despill = DespillSettings { amount: 0.5 };
    request.target = folder.0.join("despilled-png");
    assert_eq!(
        export(
            &request,
            &runtime.ffmpeg,
            &ExportControl::default(),
            |_, _| {}
        )
        .unwrap(),
        1
    );
    let adjusted = image::open(request.target.join("frame_000001.png"))
        .unwrap()
        .to_rgba8();
    // Independent unchanged v1 NumPy/OpenCV oracle for the uniform source plate.
    assert_eq!(adjusted.get_pixel(0, 0).0, [30, 20, 40, 0]);
    assert_eq!(adjusted.get_pixel(10, 10).0, [60, 40, 65, 128]);
    for (output_frame, source_frame) in [(0, 1u8), (2, 3u8)] {
        assert_eq!(
            image::open(request.target.join(format!("frame_{output_frame:06}.png")))
                .unwrap()
                .to_rgba8()
                .as_raw(),
            &[source_frame * 30, 40, 80, 255].repeat(64 * 32)
        );
    }
    let single = folder.0.join("despilled-single.png");
    crate::media_tasks::export_single_frame_with_despill(
        &request.media,
        2,
        crate::preview::ViewMode::Composite,
        request.rendered.get(&2).cloned(),
        &single,
        &runtime.ffmpeg,
        request.despill,
    )
    .unwrap();
    assert_eq!(image::open(single).unwrap().to_rgba8(), adjusted);
    request.despill = DespillSettings::default();

    request.format = ExportFormat::ExrSequence;
    request.target = folder.0.join("exr");
    assert_eq!(
        export(
            &request,
            &runtime.ffmpeg,
            &ExportControl::default(),
            |_, _| {}
        )
        .unwrap(),
        1
    );
    assert_eq!(fs::read_dir(&request.target).unwrap().count(), 3);
    let decoded = ffmpeg_output(
        &runtime.ffmpeg,
        &request.target.join("frame_000001.exr"),
        "gbrapf32le",
        1,
    );
    assert_eq!(decoded.len(), 64 * 32 * 4 * 4);
    let floats = decoded
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect::<Vec<_>>();
    for (plane, value) in [
        (0, srgb_to_linear(80)),
        (1, srgb_to_linear(40)),
        (2, srgb_to_linear(200)),
        (3, 128.0 / 255.0),
    ] {
        assert!((floats[plane * 64 * 32 + 1] - value).abs() < 1e-6);
    }
    assert_eq!(floats[3 * 64 * 32], 0.0);
    assert!((floats[2 * 64 * 32] - srgb_to_linear(100)).abs() < 1e-6); // Unassociated transparent RGB retained.

    request.format = ExportFormat::ProRes4444;
    request.target = folder.0.join("export.mov");
    assert_eq!(
        export(
            &request,
            &runtime.ffmpeg,
            &ExportControl::default(),
            |_, _| {}
        )
        .unwrap(),
        1
    );
    let mut probe = Command::new(&runtime.ffprobe);
    probe
        .args(["-v", "error", "-show_streams", "-of", "json"])
        .arg(&request.target);
    hidden(&mut probe);
    let output = probe.output().unwrap();
    assert!(output.status.success());
    let metadata: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let streams = metadata["streams"].as_array().unwrap();
    assert_eq!(streams.len(), 1); // V1 playback export is video-only.
    assert_eq!(streams[0]["codec_name"], "prores");
    assert!(matches!(streams[0]["profile"].as_str(), Some("4444" | "4")));
    assert_eq!(streams[0]["codec_tag_string"], "ap4h");
    assert_eq!(streams[0]["width"], 64);
    assert_eq!(streams[0]["height"], 32);
    assert_eq!(streams[0]["nb_frames"], "3");
    assert_eq!(streams[0]["r_frame_rate"], "24/1");
    assert!(
        streams[0]["pix_fmt"]
            .as_str()
            .unwrap()
            .starts_with("yuva444")
    );
    let decoded = ffmpeg_output(&runtime.ffmpeg, &request.target, "rgba", 3);
    assert_eq!(decoded.len(), 64 * 32 * 4 * 3);
    assert!(
        decoded[..64 * 32 * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| pixel[3] >= 254)
    );
    assert!((i16::from(decoded[64 * 32 * 4 + 7]) - 128).abs() <= 2);
    assert!(
        decoded[64 * 32 * 4 * 2..]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| pixel[3] >= 254)
    );
    request.media.verify_source_unchanged().unwrap();
    println!(
        "CPU_RANGE_EXPORT {}",
        serde_json::json!({ "frames":3, "rendered_frames":1, "source_frames":2, "png_byte_exact":true, "linear_exr_independent_decode":true, "prores_4444_alpha":true, "audio":"video_only_matching_v1", "new_runtime_dependencies":0 })
    );
}
