//! Small, version-checked FFmpeg 9 CPU adapter. Uses the shared libraries already
//! packaged beside ffmpeg.exe; it adds no media runtime or Python dependency.
//!
//! Prefix layouts below follow the public n9.0.2 headers. FFmpeg permits appending
//! fields within an ABI major, but not removing/reordering them. We never allocate
//! these structs ourselves and reject other majors before dereferencing them.
//! https://github.com/FFmpeg/FFmpeg/tree/n9.0.2/{libavformat,libavcodec,libavutil}
use super::*;
use libloading::{Library, os::windows::Library as WindowsLibrary};
use std::ffi::{CString, c_char, c_int, c_uint, c_void};

type Ptr = *mut c_void;
const EAGAIN: c_int = -11;
const AVERROR_EOF: c_int = -541_478_725;

#[repr(C)]
#[derive(Clone, Copy)]
struct AvRational {
    num: c_int,
    den: c_int,
}
#[repr(C)]
struct FormatPrefix {
    class: Ptr,
    input: Ptr,
    output: Ptr,
    private: Ptr,
    io: Ptr,
    flags: c_int,
    stream_count: c_uint,
    streams: *mut *mut StreamPrefix,
}
#[repr(C)]
struct StreamPrefix {
    class: Ptr,
    index: c_int,
    id: c_int,
    parameters: *mut ParametersPrefix,
    private: Ptr,
    time_base: AvRational,
}
#[repr(C)]
struct ParametersPrefix {
    media_type: c_int,
    codec_id: c_int,
    codec_tag: c_uint,
    extra_data: Ptr,
    extra_size: c_int,
    side_data: Ptr,
    side_count: c_int,
}
#[repr(C)]
struct PacketPrefix {
    buffer: Ptr,
    pts: i64,
    dts: i64,
    data: Ptr,
    size: c_int,
    stream: c_int,
}
#[repr(C)]
struct FramePrefix {
    data: [*mut u8; 8],
    stride: [c_int; 8],
    extended_data: Ptr,
    width: c_int,
    height: c_int,
    samples: c_int,
    format: c_int,
    picture_type: c_int,
    aspect: AvRational,
    pts: i64,
    dts: i64,
    time_base: AvRational,
}

struct Api {
    open_input: unsafe extern "C" fn(*mut *mut FormatPrefix, *const c_char, Ptr, Ptr) -> c_int,
    close_input: unsafe extern "C" fn(*mut *mut FormatPrefix),
    read_packet: unsafe extern "C" fn(*mut FormatPrefix, *mut PacketPrefix) -> c_int,
    seek: unsafe extern "C" fn(*mut FormatPrefix, c_int, i64, c_int) -> c_int,
    find_codec: unsafe extern "C" fn(c_int) -> Ptr,
    alloc_codec: unsafe extern "C" fn(Ptr) -> Ptr,
    copy_parameters: unsafe extern "C" fn(Ptr, *const ParametersPrefix) -> c_int,
    open_codec: unsafe extern "C" fn(Ptr, Ptr, Ptr) -> c_int,
    free_codec: unsafe extern "C" fn(*mut Ptr),
    send_packet: unsafe extern "C" fn(Ptr, *const PacketPrefix) -> c_int,
    receive_frame: unsafe extern "C" fn(Ptr, *mut FramePrefix) -> c_int,
    flush_codec: unsafe extern "C" fn(Ptr),
    alloc_packet: unsafe extern "C" fn() -> *mut PacketPrefix,
    unref_packet: unsafe extern "C" fn(*mut PacketPrefix),
    free_packet: unsafe extern "C" fn(*mut *mut PacketPrefix),
    alloc_frame: unsafe extern "C" fn() -> *mut FramePrefix,
    unref_frame: unsafe extern "C" fn(*mut FramePrefix),
    free_frame: unsafe extern "C" fn(*mut *mut FramePrefix),
    copy_properties: unsafe extern "C" fn(*mut FramePrefix, *const FramePrefix) -> c_int,
    pixel_format: unsafe extern "C" fn(*const c_char) -> c_int,
    set_integer: unsafe extern "C" fn(Ptr, *const c_char, i64, c_int) -> c_int,
    error_text: unsafe extern "C" fn(c_int, *mut c_char, usize) -> c_int,
    alloc_scaler: unsafe extern "C" fn() -> Ptr,
    free_scaler: unsafe extern "C" fn(Ptr),
    scale: unsafe extern "C" fn(Ptr, *mut FramePrefix, *const FramePrefix) -> c_int,
    _libraries: [Library; 4],
}
impl Api {
    fn load(ffmpeg: &Path) -> Result<Self> {
        let folder = ffmpeg.parent().context("FFmpeg directory is missing")?;
        let load = |name: &str| -> Result<Library> {
            // Resolve dependencies only beside the selected FFmpeg and in the
            // standard safe search directories. Do not modify the process PATH.
            Ok(unsafe {
                WindowsLibrary::load_with_flags(folder.join(name), 0x100 | 0x1000)?.into()
            })
        };
        let util = load("avutil-61.dll")?;
        let codec = load("avcodec-63.dll")?;
        let format = load("avformat-63.dll")?;
        let scaler = load("swscale-10.dll")?;
        unsafe fn version(library: &Library, symbol: &[u8]) -> Result<c_uint> {
            let function = unsafe { library.get::<unsafe extern "C" fn() -> c_uint>(symbol)? };
            Ok(unsafe { function() } >> 16)
        }
        ensure!(
            unsafe { version(&util, b"avutil_version\0")? } == 61
                && unsafe { version(&codec, b"avcodec_version\0")? } == 63
                && unsafe { version(&format, b"avformat_version\0")? } == 63
                && unsafe { version(&scaler, b"swscale_version\0")? } == 10,
            "Installed FFmpeg ABI does not match the native preview adapter"
        );
        macro_rules! symbol {
            ($library:expr, $name:literal) => {
                unsafe { *$library.get(concat!($name, "\0").as_bytes())? }
            };
        }
        Ok(Self {
            open_input: symbol!(format, "avformat_open_input"),
            close_input: symbol!(format, "avformat_close_input"),
            read_packet: symbol!(format, "av_read_frame"),
            seek: symbol!(format, "av_seek_frame"),
            find_codec: symbol!(codec, "avcodec_find_decoder"),
            alloc_codec: symbol!(codec, "avcodec_alloc_context3"),
            copy_parameters: symbol!(codec, "avcodec_parameters_to_context"),
            open_codec: symbol!(codec, "avcodec_open2"),
            free_codec: symbol!(codec, "avcodec_free_context"),
            send_packet: symbol!(codec, "avcodec_send_packet"),
            receive_frame: symbol!(codec, "avcodec_receive_frame"),
            flush_codec: symbol!(codec, "avcodec_flush_buffers"),
            alloc_packet: symbol!(codec, "av_packet_alloc"),
            unref_packet: symbol!(codec, "av_packet_unref"),
            free_packet: symbol!(codec, "av_packet_free"),
            alloc_frame: symbol!(util, "av_frame_alloc"),
            unref_frame: symbol!(util, "av_frame_unref"),
            free_frame: symbol!(util, "av_frame_free"),
            copy_properties: symbol!(util, "av_frame_copy_props"),
            pixel_format: symbol!(util, "av_get_pix_fmt"),
            set_integer: symbol!(util, "av_opt_set_int"),
            error_text: symbol!(util, "av_strerror"),
            alloc_scaler: symbol!(scaler, "sws_alloc_context"),
            free_scaler: symbol!(scaler, "sws_freeContext"),
            scale: symbol!(scaler, "sws_scale_frame"),
            _libraries: [format, codec, scaler, util],
        })
    }
    fn check(&self, code: c_int, operation: &str) -> Result<()> {
        if code >= 0 {
            return Ok(());
        }
        let mut message = [0u8; 256];
        unsafe { (self.error_text)(code, message.as_mut_ptr().cast(), message.len()) };
        let length = message.iter().position(|byte| *byte == 0).unwrap_or(256);
        bail!(
            "{operation}: {} ({code})",
            String::from_utf8_lossy(&message[..length])
        )
    }
}

pub(super) struct NativeDecoder {
    api: Api,
    format: *mut FormatPrefix,
    codec: Ptr,
    packet: *mut PacketPrefix,
    frame: *mut FramePrefix,
    output: *mut FramePrefix,
    scaler: Ptr,
    stream: c_int,
    time_base: Rational,
    rgba_format: c_int,
    draining: bool,
    pub(super) next: u64,
    control: Arc<DecodeControl>,
}
impl NativeDecoder {
    pub(super) fn open(
        media: &MediaInfo,
        target: u64,
        ffmpeg: &Path,
        control: Arc<DecodeControl>,
    ) -> Result<Self> {
        media.validate()?;
        ensure!(
            control.allows(media, target),
            "Source preview was superseded"
        );
        let api = Api::load(ffmpeg)?;
        let mut decoder = Self {
            api,
            format: std::ptr::null_mut(),
            codec: std::ptr::null_mut(),
            packet: std::ptr::null_mut(),
            frame: std::ptr::null_mut(),
            output: std::ptr::null_mut(),
            scaler: std::ptr::null_mut(),
            stream: -1,
            time_base: media.time_base,
            rgba_format: -1,
            draining: false,
            next: 0,
            control,
        };
        let path = CString::new(media.path.to_str().context("Source path is not Unicode")?)?;
        unsafe {
            decoder.api.check(
                (decoder.api.open_input)(
                    &mut decoder.format,
                    path.as_ptr(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ),
                "Opening CPU source",
            )?;
            let format = &*decoder.format;
            ensure!(
                format.stream_count <= 1024 && !format.streams.is_null(),
                "Invalid source stream table"
            );
            let mut parameters = std::ptr::null_mut();
            for index in 0..format.stream_count {
                let stream = *format.streams.add(index as usize);
                if !stream.is_null()
                    && !(*stream).parameters.is_null()
                    && (*(*stream).parameters).media_type == 0
                {
                    decoder.stream = (*stream).index;
                    let time_base = (*stream).time_base;
                    ensure!(
                        time_base.num > 0 && time_base.den > 0,
                        "Invalid stream time base"
                    );
                    decoder.time_base = Rational {
                        num: time_base.num as u64,
                        den: time_base.den as u64,
                    };
                    parameters = (*stream).parameters;
                    break;
                }
            }
            ensure!(!parameters.is_null(), "Source has no video stream");
            // A display matrix or other codec side data needs the CLI's filter
            // handling. Do not silently ignore rotation or HDR metadata.
            ensure!(
                (*parameters).side_count == 0,
                "Source requires FFmpeg side-data processing"
            );
            ensure!(
                decoder.time_base == media.time_base,
                "Source time base changed"
            );
            let codec = (decoder.api.find_codec)((*parameters).codec_id);
            ensure!(!codec.is_null(), "Software video decoder is unavailable");
            decoder.codec = (decoder.api.alloc_codec)(codec);
            ensure!(
                !decoder.codec.is_null(),
                "Allocating software video decoder failed"
            );
            decoder.api.check(
                (decoder.api.copy_parameters)(decoder.codec, parameters),
                "Copying source codec parameters",
            )?;
            let threads = thread::available_parallelism()
                .map_or(2, |count| count.get().div_ceil(2).clamp(1, 12));
            decoder.api.check(
                (decoder.api.set_integer)(decoder.codec, c"threads".as_ptr(), threads as i64, 0),
                "Configuring CPU decode threads",
            )?;
            decoder.api.check(
                (decoder.api.open_codec)(decoder.codec, codec, std::ptr::null_mut()),
                "Opening software video decoder",
            )?;
            decoder.packet = (decoder.api.alloc_packet)();
            decoder.frame = (decoder.api.alloc_frame)();
            decoder.output = (decoder.api.alloc_frame)();
            decoder.scaler = (decoder.api.alloc_scaler)();
            decoder.rgba_format = (decoder.api.pixel_format)(c"rgba".as_ptr());
        }
        ensure!(
            !decoder.packet.is_null()
                && !decoder.frame.is_null()
                && !decoder.output.is_null()
                && !decoder.scaler.is_null()
                && decoder.rgba_format >= 0,
            "Allocating CPU preview buffers failed"
        );
        ensure!(
            decoder.control.allows(media, target),
            "Source preview was superseded"
        );
        decoder.control.set_native_active(true);
        Ok(decoder)
    }
    #[cfg(test)]
    pub(super) fn identity(&self) -> u64 {
        self.format as usize as u64
    }
    pub(super) fn read(&mut self, media: &MediaInfo, target: u64) -> Result<Arc<[u8]>> {
        ensure!(
            self.control.allows(media, target),
            "Source preview was superseded"
        );
        if target < self.next || target - self.next > FORWARD_DECODE_LIMIT {
            unsafe {
                self.api.check(
                    (self.api.seek)(self.format, self.stream, target_pts(media, target)?, 1),
                    "Seeking CPU source",
                )?;
                (self.api.flush_codec)(self.codec);
                (self.api.unref_packet)(self.packet);
                (self.api.unref_frame)(self.frame);
            }
            self.draining = false;
        }
        let pts = target_pts(media, target)?;
        loop {
            ensure!(
                self.control.allows(media, target),
                "Source preview was superseded"
            );
            let status = unsafe { (self.api.receive_frame)(self.codec, self.frame) };
            if status >= 0 {
                let actual_pts = unsafe { (*self.frame).pts };
                if actual_pts < pts {
                    unsafe { (self.api.unref_frame)(self.frame) };
                    continue;
                }
                let index = source_frame(
                    media,
                    Stamp {
                        pts: actual_pts,
                        time_base: self.time_base,
                    },
                )?;
                ensure!(
                    index == target,
                    "CPU decoder returned source frame {index} instead of {target}"
                );
                let pixels = self.convert(media)?;
                unsafe { (self.api.unref_frame)(self.frame) };
                self.next = target + 1;
                ensure!(
                    self.control.allows(media, target),
                    "Source preview was superseded"
                );
                return Ok(pixels);
            }
            ensure!(
                status != AVERROR_EOF,
                "CPU source ended before requested frame"
            );
            self.api.check(
                if status == EAGAIN { 0 } else { status },
                "Receiving CPU source frame",
            )?;
            ensure!(!self.draining, "CPU source did not produce the final frame");
            loop {
                ensure!(
                    self.control.allows(media, target),
                    "Source preview was superseded"
                );
                let status = unsafe { (self.api.read_packet)(self.format, self.packet) };
                if status == AVERROR_EOF {
                    self.api.check(
                        unsafe { (self.api.send_packet)(self.codec, std::ptr::null()) },
                        "Draining CPU source decoder",
                    )?;
                    self.draining = true;
                    break;
                }
                self.api.check(status, "Reading CPU source packet")?;
                if unsafe { (*self.packet).stream } != self.stream {
                    unsafe { (self.api.unref_packet)(self.packet) };
                    continue;
                }
                let status = unsafe { (self.api.send_packet)(self.codec, self.packet) };
                unsafe { (self.api.unref_packet)(self.packet) };
                self.api.check(status, "Decoding CPU source packet")?;
                break;
            }
        }
    }
    fn convert(&mut self, media: &MediaInfo) -> Result<Arc<[u8]>> {
        let width = usize::try_from(media.width)?;
        let height = usize::try_from(media.height)?;
        let row = width.checked_mul(4).context("Preview row size overflow")?;
        let length = row
            .checked_mul(height)
            .context("Preview dimensions overflow")?;
        ensure!(
            length <= 512 * 1024 * 1024,
            "Source preview exceeds the 512MiB single-frame safety limit"
        );
        unsafe {
            ensure!(
                (*self.frame).width == media.width as c_int
                    && (*self.frame).height == media.height as c_int,
                "Decoded dimensions differ from imported source"
            );
            (self.api.unref_frame)(self.output);
            self.api.check(
                (self.api.copy_properties)(self.output, self.frame),
                "Copying source color properties",
            )?;
            (*self.output).width = media.width as c_int;
            (*self.output).height = media.height as c_int;
            (*self.output).format = self.rgba_format;
            self.api.check(
                (self.api.scale)(self.scaler, self.output, self.frame),
                "Converting CPU source to RGBA",
            )?;
            let output = &*self.output;
            ensure!(
                !output.data[0].is_null() && output.stride[0].unsigned_abs() as usize >= row,
                "Invalid CPU RGBA output stride"
            );
            let mut bytes = vec![0; length];
            for y in 0..height {
                let source = output.data[0].offset(y as isize * output.stride[0] as isize);
                std::ptr::copy_nonoverlapping(source, bytes.as_mut_ptr().add(y * row), row);
            }
            (self.api.unref_frame)(self.output);
            Ok(bytes.into())
        }
    }
}
impl Drop for NativeDecoder {
    fn drop(&mut self) {
        // All codec/demuxer/scaler objects belong to this worker thread. Nothing
        // frees them from the UI thread while a decode call is using them.
        unsafe {
            (self.api.free_frame)(&mut self.output);
            (self.api.free_frame)(&mut self.frame);
            (self.api.free_packet)(&mut self.packet);
            (self.api.free_codec)(&mut self.codec);
            (self.api.close_input)(&mut self.format);
            (self.api.free_scaler)(self.scaler);
        }
        self.control.set_native_active(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ffmpeg9_public_prefix_offsets_match_x64_c_abi() {
        assert_eq!(std::mem::offset_of!(FormatPrefix, streams), 48);
        assert_eq!(std::mem::offset_of!(StreamPrefix, parameters), 16);
        assert_eq!(std::mem::offset_of!(StreamPrefix, time_base), 32);
        assert_eq!(std::mem::offset_of!(ParametersPrefix, side_count), 40);
        assert_eq!(std::mem::offset_of!(PacketPrefix, stream), 36);
        assert_eq!(std::mem::offset_of!(FramePrefix, width), 104);
        assert_eq!(std::mem::offset_of!(FramePrefix, pts), 136);
    }

    #[test]
    #[ignore = "Explicit CPU-only FFmpeg fixture; creates and removes two tiny owned FFV1 clips"]
    fn native_rgb_alpha_and_bt709_yuva_match_cli_and_cancel_cleanly() {
        use std::io::Write;
        struct FixtureFolder {
            directory: PathBuf,
            files: Vec<PathBuf>,
        }
        impl Drop for FixtureFolder {
            fn drop(&mut self) {
                for path in &self.files {
                    let _ = std::fs::remove_file(path);
                }
                let _ = std::fs::remove_dir(&self.directory);
            }
        }
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let runtime = ck_desktop_core::RuntimePaths::installed(root.join("app"));
        let directory = std::env::temp_dir().join(format!(
            "brp-native-color-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let mut folder = FixtureFolder {
            directory,
            files: Vec::new(),
        };
        let mut rgba = Vec::with_capacity(8 * 64 * 48 * 4);
        for frame in 0..8 {
            for y in 0..48 {
                for x in 0..64 {
                    rgba.extend([
                        ((x * 4 + frame * 9) % 256) as u8,
                        ((y * 5 + frame * 11) % 256) as u8,
                        ((x * 7 + y * 3 + frame * 17) % 256) as u8,
                        ((x * 3 + y * 5 + frame * 19) % 256) as u8,
                    ]);
                }
            }
        }
        for pixel_format in ["bgra", "yuva444p10le"] {
            let path = folder.directory.join(format!("{pixel_format}.mkv"));
            folder.files.push(path.clone());
            let mut command = Command::new(&runtime.ffmpeg);
            command
                .args([
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-threads",
                    "2",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgba",
                    "-video_size",
                    "64x48",
                    "-framerate",
                    "25",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "ffv1",
                    "-threads:v",
                    "2",
                    "-pix_fmt",
                    pixel_format,
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-colorspace",
                    "bt709",
                    "-color_range",
                    "tv",
                ])
                .arg(&path)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000);
            let mut child = command.spawn().unwrap();
            child.stdin.take().unwrap().write_all(&rgba).unwrap();
            let encoded = child.wait_with_output().unwrap();
            assert!(
                encoded.status.success(),
                "{}",
                String::from_utf8_lossy(&encoded.stderr)
            );
            let media = MediaInfo::probe(&path, &runtime).unwrap();
            let control = Arc::new(DecodeControl::default());
            let mut decoder =
                NativeDecoder::open(&media, 0, &runtime.ffmpeg, control.clone()).unwrap();
            let identity = decoder.identity();
            for frame in [0, 1, 7, 3, 0] {
                let actual = decoder.read(&media, frame).unwrap();
                let expected =
                    super::super::cartolas_tests::origin_reference(&runtime.ffmpeg, &media, frame);
                assert!(
                    actual.as_ref() == expected.as_slice(),
                    "{pixel_format} frame {frame}: RGBA bytes changed"
                );
                assert_eq!(
                    decoder.identity(),
                    identity,
                    "Seeks must retain the demuxer/codec"
                );
                assert!(actual.as_chunks::<4>().0.iter().any(|pixel| pixel[3] < 255));
                assert!(actual.as_chunks::<4>().0.iter().any(|pixel| pixel[3] > 0));
            }
            control.shutdown();
            assert!(decoder.read(&media, 2).is_err());
            drop(decoder);
            assert!(!control.is_active());
        }
    }
}
