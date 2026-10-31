//! Explicit CPU regression for the private clip whose legacy-encoded metadata
//! previously disconnected the stderr protocol reader. No footage is copied.
use super::*;
use std::time::Instant;

fn regression_video() -> PathBuf {
    if let Some(path) = std::env::var_os("BRP_DECODER_REGRESSION_VIDEO") {
        return PathBuf::from(path);
    }
    PathBuf::from(
        std::env::var_os("USERPROFILE")
            .expect("Set BRP_DECODER_REGRESSION_VIDEO to an existing CPU test clip"),
    )
    .join("Downloads/CARTOLAS FINAL_export.mp4")
}

pub(super) fn origin_reference(ffmpeg: &Path, media: &MediaInfo, index: u64) -> Vec<u8> {
    // Independent display-order oracle: no input seek, trim, PTS calculation,
    // persistent decoder state or source-preview metadata parser.
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
        .arg(&media.path)
        .args([
            "-map",
            "0:v:0",
            "-vf",
            &format!("select=eq(n\\,{index}),format=rgba"),
            "-an",
            "-sn",
            "-dn",
            "-frames:v",
            "1",
            "-fps_mode",
            "passthrough",
            "-threads:v",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
    let output = command.output().expect("Starting CPU origin oracle");
    assert!(
        output.status.success(),
        "CPU origin oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.stdout.len(),
        media.width as usize * media.height as usize * 4,
        "CPU origin oracle must return exactly one full RGBA frame"
    );
    output.stdout
}

fn decoder_identity(frames: &SourceFrames, control: &DecodeControl) -> u64 {
    #[cfg(windows)]
    if let Some(decoder) = frames.native.as_ref() {
        return decoder.identity();
    }
    #[cfg(not(windows))]
    let _ = frames;
    control
        .0
        .lock()
        .unwrap()
        .process
        .as_ref()
        .expect("Source decoder must remain active until dropped")
        .child
        .id() as u64
}

#[test]
#[ignore = "Explicit CPU-only private-clip regression; installed FFmpeg/FFprobe and BRP_DECODER_REGRESSION_VIDEO or CARTOLAS download required"]
fn real_cartolas_metadata_playback_and_seeks_match_origin() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let runtime = ck_desktop_core::RuntimePaths::installed(root.join("app"));
    assert!(runtime.ffmpeg.is_file() && runtime.ffprobe.is_file());
    let video = regression_video();
    assert!(video.is_file(), "The explicit regression clip is missing");
    let media = MediaInfo::probe(video, &runtime).expect("Indexing CPU regression clip");
    assert!(media.frame_count > 137, "Regression needs source frame 137");

    let control = Arc::new(DecodeControl::default());
    let mut frames = SourceFrames::new(control.clone());
    let mut pipe_id = None;
    let mut first_frame_ms = 0.0;
    let mut sequential_ms = 0.0;
    let mut oracle_ms = 0.0;
    let mut peak_cache_bytes = 0;
    for index in 0..12 {
        let start = Instant::now();
        let actual = frames
            .read(&media, index, &runtime.ffmpeg)
            .unwrap_or_else(|error| panic!("CPU sequential frame {index}: {error:#}"));
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        sequential_ms += elapsed;
        if index == 0 {
            first_frame_ms = elapsed;
            pipe_id = Some(decoder_identity(&frames, &control));
        }
        assert_eq!(
            Some(decoder_identity(&frames, &control)),
            pipe_id,
            "Sequential playback must keep one FFmpeg pipe"
        );
        assert!(frames.cache.bytes <= CACHE_BYTES);
        assert!(frames.cache.entries.len() <= CACHE_FRAMES);
        peak_cache_bytes = peak_cache_bytes.max(frames.cache.bytes);

        let start = Instant::now();
        let expected = origin_reference(&runtime.ffmpeg, &media, index);
        oracle_ms += start.elapsed().as_secs_f64() * 1000.0;
        assert!(
            actual.as_ref() == expected.as_slice(),
            "Sequential source frame {index} differs from origin decoding"
        );
    }
    let sequential_cache_frames = frames.cache.entries.len();
    #[cfg(windows)]
    assert!(
        frames.native.is_some(),
        "Installed Windows FFmpeg must exercise the in-process decoder"
    );
    let start = Instant::now();
    drop(frames);
    let mut drop_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert!(!control.is_active(), "Drop must kill and reap the decoder");

    let mut seek_metrics = Vec::new();
    for index in [0, 137, media.frame_count - 1] {
        // A fresh bounded reader forces real uncached source seeks, including
        // frame zero; the sequential LRU cannot accidentally satisfy this test.
        let control = Arc::new(DecodeControl::default());
        let mut seeking = SourceFrames::new(control.clone());
        let start = Instant::now();
        let actual = seeking
            .read(&media, index, &runtime.ffmpeg)
            .unwrap_or_else(|error| panic!("CPU cold seek frame {index}: {error:#}"));
        let seek_ms = start.elapsed().as_secs_f64() * 1000.0;
        #[cfg(windows)]
        assert!(
            seeking.native.is_some(),
            "Cold seek must exercise the in-process decoder"
        );
        assert!(seeking.cache.bytes <= CACHE_BYTES);
        assert!(seeking.cache.entries.len() <= CACHE_FRAMES);
        peak_cache_bytes = peak_cache_bytes.max(seeking.cache.bytes);

        let start = Instant::now();
        let expected = origin_reference(&runtime.ffmpeg, &media, index);
        oracle_ms += start.elapsed().as_secs_f64() * 1000.0;
        assert!(
            actual.as_ref() == expected.as_slice(),
            "Cold-seek source frame {index} differs from origin decoding"
        );
        let old_control = Arc::new(DecodeControl::default());
        let start = Instant::now();
        let mut old = RawDecoder::start(&media, index, &runtime.ffmpeg, old_control.clone(), true)
            .expect("Starting established CLI seek baseline");
        let baseline = old.read(&media).expect("Reading established CLI baseline");
        let baseline_ms = start.elapsed().as_secs_f64() * 1000.0;
        assert!(
            baseline.as_ref() == expected.as_slice(),
            "CLI baseline frame {index} differs from origin decoding"
        );
        drop(old);
        assert!(!old_control.is_active());
        seek_metrics.push(serde_json::json!({
            "frame": index,
            "decode_ms": seek_ms,
            "baseline_cli_ms": baseline_ms,
            "baseline_over_native": baseline_ms / seek_ms,
        }));
        let start = Instant::now();
        drop(seeking);
        drop_ms += start.elapsed().as_secs_f64() * 1000.0;
        assert!(!control.is_active(), "Cold-seek drop must reap its decoder");
    }
    media.verify_source_unchanged().unwrap();
    println!(
        "CPU_CARTOLAS_REGRESSION {}",
        serde_json::json!({
            "width": media.width,
            "height": media.height,
            "source_frames": media.frame_count,
            "sequential_frames": 12,
            "first_frame_ms": first_frame_ms,
            "sequential_decode_ms": sequential_ms,
            "sequential_decoder_count": 1,
            "cold_seeks": seek_metrics,
            "origin_oracle_ms": oracle_ms,
            "peak_retained_cache_bytes": peak_cache_bytes,
            "sequential_cache_frames": sequential_cache_frames,
            "decoder_drop_ms": drop_ms,
            "all_15_comparisons_byte_exact": true,
            "all_decoders_reaped": true,
            "disk_frame_cache_files": 0,
        })
    );
}
