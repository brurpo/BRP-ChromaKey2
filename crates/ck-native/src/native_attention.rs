//! Development-only exact native attention-map boundary.
//!
//! The CPU operator deliberately uses ordinary ORT host transfers. A private
//! DirectML command queue is entered only after its CPU inputs are ready and
//! returns only after the complete scan is back on the CPU. It never submits
//! work inside ORT's unflushed DML queue. No DLL or GPU is loaded by `domain`.
mod encoder;
mod masked;
mod strip;

use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char, c_void},
    path::PathBuf,
    ptr::NonNull,
    sync::{Arc, Mutex, Weak},
};

use half::f16;
use libloading::Library;
use ort::{
    Error, Result,
    memory::MemoryType,
    operator::{
        Operator, OperatorDomain, ShapeInferenceContext,
        io::{OperatorInput, OperatorOutput},
        kernel::{Kernel, KernelAttributes, KernelContext},
    },
    tensor::{Shape, SymbolicDimensions, TensorElementType},
    value::ValueType,
};

const DOMAIN: &str = "com.brp_chroma.experimental";
const OPERATOR: &str = "NativeAttentionMap1080";
const BODY_OPERATOR: &str = "NativeBodyAttentionMap1080";
#[cfg(test)]
const Q_SHAPE: [i64; 4] = [1, 8, 8640, 64];
#[cfg(test)]
const K_SHAPE: [i64; 4] = [1, 8, 64, 8640];
const OUTPUT_SHAPE: [i64; 5] = [68, 1, 8, 128, 64];
const SCALE: f32 = f32::from_bits(0x3eb5_0000); // Exactly FP16 0x35a8.
const BODY_SCALE: f32 = f32::from_bits(0x3ea3_a000); // Exactly FP16 0x351d.

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Geometry {
    heads: u32,
    queries: u32,
    head_dim: u32,
    query_chunk: u32,
    scale_bits: u32,
}

impl Geometry {
    fn q_shape(self) -> [i64; 4] {
        [
            1,
            self.heads.into(),
            self.queries.into(),
            self.head_dim.into(),
        ]
    }

    fn k_shape(self) -> [i64; 4] {
        [
            1,
            self.heads.into(),
            self.head_dim.into(),
            self.queries.into(),
        ]
    }

    fn output_shape(self) -> [i64; 5] {
        [
            self.queries.div_ceil(self.query_chunk).into(),
            1,
            self.heads.into(),
            self.query_chunk.into(),
            self.head_dim.into(),
        ]
    }
}

#[derive(Clone, Copy, Debug)]
enum MapKind {
    Encoder,
    Body,
}

impl MapKind {
    fn name(self) -> &'static str {
        match self {
            Self::Encoder => OPERATOR,
            Self::Body => BODY_OPERATOR,
        }
    }

    fn scale(self) -> f32 {
        match self {
            Self::Encoder => SCALE,
            Self::Body => BODY_SCALE,
        }
    }

    fn geometry(self, queries: i64) -> Result<Geometry> {
        let (heads, head_dim) = match self {
            Self::Encoder if queries == 8640 => (8, 64),
            Self::Body if (540..=4320).contains(&queries) && queries % 540 == 0 => (16, 96),
            _ => {
                return Err(Error::new(format!(
                    "{} query count {queries} is outside the experimental geometry contract",
                    self.name()
                )));
            }
        };
        Ok(Geometry {
            heads,
            queries: queries as u32,
            head_dim,
            query_chunk: 128,
            scale_bits: self.scale().to_bits(),
        })
    }

    fn unknown_output_shape(self) -> [i64; 5] {
        match self {
            Self::Encoder => OUTPUT_SHAPE,
            Self::Body => [-1, 1, 16, 128, 96],
        }
    }
}

// The cache owns no engines. Kernels retain only the engine matching their
// latest runtime shape; obsolete entries disappear when their last kernel
// releases them, and all resources disappear with the Stage1 session.
struct EngineCache<T> {
    entries: HashMap<Geometry, Weak<Mutex<T>>>,
}

impl<T> EngineCache<T> {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        geometry: Geometry,
        create: impl FnOnce() -> Result<T>,
    ) -> Result<Arc<Mutex<T>>> {
        self.entries.retain(|_, engine| engine.strong_count() > 0);
        if let Some(engine) = self.entries.get(&geometry).and_then(Weak::upgrade) {
            return Ok(engine);
        }
        let engine = Arc::new(Mutex::new(create()?));
        self.entries.insert(geometry, Arc::downgrade(&engine));
        Ok(engine)
    }
}

#[repr(C)]
struct FfiConfig {
    abi_version: u32,
    adapter_id: i32,
    heads: u32,
    queries: u32,
    head_dim: u32,
    query_chunk: u32,
    query_scale: f32,
    directml_path_utf8: *const c_char,
}

type Create = unsafe extern "C" fn(*const FfiConfig, *mut *mut c_void, *mut c_char, usize) -> i32;
type Run = unsafe extern "C" fn(
    *mut c_void,
    *const u16,
    usize,
    *const u16,
    usize,
    *const u16,
    usize,
    *mut u16,
    usize,
    *mut c_char,
    usize,
) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);
type AbiVersion = unsafe extern "C" fn() -> u32;

fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| Error::new(format!("native attention ABI symbol: {error}")))
}

struct Engine {
    pointer: NonNull<c_void>,
    run: Run,
    destroy: Destroy,
    // Must remain loaded through Drop, which invokes its destroy entrypoint.
    _library: Library,
}

// The opaque native object contains free-threaded D3D12/DML objects and its own
// mutex. Rust also serializes every call/destruction through Arc<Mutex<Engine>>.
// There is no borrowed Rust data retained after a synchronous FFI run returns.
unsafe impl Send for Engine {}

fn ffi_error(message: &[c_char], fallback: &str) -> Error {
    // Every ABI entrypoint receives this zeroed bounded array and guarantees a
    // terminating zero. Never trust an unterminated third-party error buffer.
    let bytes: Vec<u8> = message.iter().map(|&byte| byte as u8).collect();
    let detail = CStr::from_bytes_until_nul(&bytes)
        .map(|text| text.to_string_lossy().into_owned())
        .unwrap_or_else(|_| fallback.to_owned());
    Error::new(format!("native attention {fallback}: {detail}"))
}

impl Engine {
    fn new(config: &Config, geometry: Geometry) -> Result<Self> {
        // Explicit existing absolute paths are required by domain(). There is
        // no download, executable search, CPU fallback, or vendor renumbering.
        let library = unsafe { Library::new(&config.helper_dll) }
            .map_err(|error| Error::new(format!("native attention helper load failed: {error}")))?;
        let version: AbiVersion = symbol(&library, b"brp_map_abi_version\0")?;
        if unsafe { version() } != 1 {
            return Err(Error::new("native attention helper ABI must be 1"));
        }
        let create: Create = symbol(&library, b"brp_map_create\0")?;
        let run: Run = symbol(&library, b"brp_map_run\0")?;
        let destroy: Destroy = symbol(&library, b"brp_map_destroy\0")?;
        let path = config.directml_dll.to_str().ok_or_else(|| {
            Error::new("native attention experiment requires a Unicode DirectML path")
        })?;
        let path = CString::new(path)
            .map_err(|_| Error::new("native attention DirectML path contains a zero byte"))?;
        let abi = FfiConfig {
            abi_version: 1,
            adapter_id: config.adapter_id,
            heads: geometry.heads,
            queries: geometry.queries,
            head_dim: geometry.head_dim,
            query_chunk: geometry.query_chunk,
            query_scale: f32::from_bits(geometry.scale_bits),
            directml_path_utf8: path.as_ptr(),
        };
        let mut pointer = std::ptr::null_mut();
        let mut error = [0 as c_char; 1024];
        if unsafe { create(&abi, &mut pointer, error.as_mut_ptr(), error.len()) } != 0 {
            return Err(ffi_error(&error, "engine creation failed"));
        }
        let pointer = NonNull::new(pointer)
            .ok_or_else(|| Error::new("native attention create returned a null engine"))?;
        Ok(Self {
            pointer,
            run,
            destroy,
            _library: library,
        })
    }

    fn run(&mut self, q: &[f16], k: &[f16], v: &[f16], output: &mut [f16]) -> Result<()> {
        let mut error = [0 as c_char; 1024];
        // half::f16 is repr(transparent) over u16; slices remain borrowed and
        // valid until this synchronous call has completed its GPU fence/copy.
        let status = unsafe {
            (self.run)(
                self.pointer.as_ptr(),
                q.as_ptr().cast(),
                q.len(),
                k.as_ptr().cast(),
                k.len(),
                v.as_ptr().cast(),
                v.len(),
                output.as_mut_ptr().cast(),
                output.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(ffi_error(&error, "execution failed"))
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Native destruction completes pending work before releasing every
        // resource/queue/device. It runs while the helper library is alive.
        unsafe { (self.destroy)(self.pointer.as_ptr()) };
    }
}

struct Config {
    helper_dll: PathBuf,
    directml_dll: PathBuf,
    adapter_id: i32,
    // Weak ownership is essential: cached scratch must disappear with the
    // final Stage1 kernel/session, before the staged Low-VRAM Stage2 pass.
    cached: Mutex<EngineCache<Engine>>,
    cached_masked: Mutex<EngineCache<masked::Engine>>,
    cached_strip: Mutex<EngineCache<strip::Engine>>,
}

impl Config {
    fn engine(&self, geometry: Geometry) -> Result<Arc<Mutex<Engine>>> {
        let mut cached = self
            .cached
            .lock()
            .map_err(|_| Error::new("native attention engine cache was poisoned"))?;
        cached.get_or_create(geometry, || Engine::new(self, geometry))
    }
}

fn check_attributes(kind: MapKind, chunk: i64, scale: f32) -> Result<()> {
    if chunk != 128 || scale.to_bits() != kind.scale().to_bits() {
        return Err(Error::new(format!(
            "{} requires query_chunk128 and query_scale{}",
            kind.name(),
            kind.scale()
        )));
    }
    Ok(())
}

fn tensor_type(shape: &[i64]) -> ValueType {
    ValueType::Tensor {
        ty: TensorElementType::Float16,
        shape: Shape::new(shape.iter().copied()),
        // ORT 1.23 ShapeInferContext_SetOutputTypeShape serializes a negative
        // dimension as a symbolic dim_param only when this string is nonempty.
        // An empty string would write the illegal TensorProto dim_value=-1 and
        // can fail DML initialization with narrowing_error before Compute.
        dimension_symbols: SymbolicDimensions::new(shape.iter().enumerate().map(
            |(axis, &dimension)| {
                if dimension < 0 {
                    format!("native_attention_unknown_axis_{axis}")
                } else {
                    String::new()
                }
            },
        )),
    }
}

fn check_shapes(kind: MapKind, shapes: &[&[i64]]) -> Result<Option<Geometry>> {
    if shapes.len() != 3 {
        return Err(Error::new("native attention requires Q/K/V"));
    }
    let baseline = kind.geometry(match kind {
        MapKind::Encoder => 8640,
        MapKind::Body => 540,
    })?;
    let mut known_queries = None;
    for (index, shape) in shapes.iter().enumerate() {
        let query_axis = if index == 1 { 3 } else { 2 };
        let expected = if index == 1 {
            baseline.k_shape()
        } else {
            baseline.q_shape()
        };
        if shape.len() != 4
            || shape.iter().enumerate().any(|(axis, &actual)| {
                axis != query_axis && actual >= 0 && actual != expected[axis]
            })
        {
            return Err(Error::new(format!(
                "{} requires FP16 B1/H{}/D{} inputs",
                kind.name(),
                baseline.heads,
                baseline.head_dim
            )));
        }
        let queries = shape[query_axis];
        if queries >= 0 {
            kind.geometry(queries)?;
            if known_queries.is_some_and(|previous| previous != queries) {
                return Err(Error::new("native attention Q/K/V query counts differ"));
            }
            known_queries = Some(queries);
        }
    }
    match kind {
        MapKind::Encoder => Ok(Some(baseline)),
        MapKind::Body => known_queries
            .map(|queries| kind.geometry(queries))
            .transpose(),
    }
}

fn check_input_types(kind: MapKind, inputs: &[ValueType]) -> Result<Option<Geometry>> {
    if inputs.len() != 3 {
        return Err(Error::new("native attention requires Q/K/V"));
    }
    let mut shapes = Vec::with_capacity(3);
    for input in inputs {
        match input {
            ValueType::Tensor {
                ty: TensorElementType::Float16,
                shape,
                ..
            } => shapes.push(&**shape),
            _ => return Err(Error::new("native attention inputs must be FP16 tensors")),
        }
    }
    check_shapes(kind, &shapes)
}

struct MapKernel {
    config: Arc<Config>,
    kind: MapKind,
    engine: Option<(Geometry, Arc<Mutex<Engine>>)>,
}

impl Kernel for MapKernel {
    fn compute(&mut self, context: &KernelContext) -> Result<()> {
        let q = context
            .input(0)?
            .ok_or_else(|| Error::new("native attention missing Q"))?;
        let k = context
            .input(1)?
            .ok_or_else(|| Error::new("native attention missing K"))?;
        let v = context
            .input(2)?
            .ok_or_else(|| Error::new("native attention missing V"))?;
        // Extraction checks actual CPU accessibility; never dereference a DML
        // allocation as a host pointer or silently alter provider placement.
        let (q_shape, q_data) = q.try_extract_tensor::<f16>()?;
        let (k_shape, k_data) = k.try_extract_tensor::<f16>()?;
        let (v_shape, v_data) = v.try_extract_tensor::<f16>()?;
        let geometry = check_shapes(self.kind, &[q_shape, k_shape, v_shape])?
            .ok_or_else(|| Error::new("native attention runtime query count must be known"))?;
        // Runtime dimensions cannot be symbolic even though metadata can be.
        if **q_shape != geometry.q_shape()
            || **k_shape != geometry.k_shape()
            || **v_shape != geometry.q_shape()
        {
            return Err(Error::new(
                "native attention runtime dimensions are invalid",
            ));
        }
        if self
            .engine
            .as_ref()
            .is_none_or(|(previous, _)| *previous != geometry)
        {
            self.engine = None;
            self.engine = Some((geometry, self.config.engine(geometry)?));
        }
        let mut output = context
            .output(0, geometry.output_shape())?
            .ok_or_else(|| Error::new("native attention missing output allocation"))?;
        let output_data = output.try_extract_tensor_mut::<f16>()?.1;
        self.engine
            .as_ref()
            .ok_or_else(|| Error::new("native attention engine not initialized"))?
            .1
            .lock()
            .map_err(|_| Error::new("native attention engine mutex was poisoned"))?
            .run(q_data, k_data, v_data, output_data)
    }
}

struct NativeAttentionMap1080 {
    config: Arc<Config>,
    kind: MapKind,
}

impl Operator for NativeAttentionMap1080 {
    fn name(&self) -> &str {
        self.kind.name()
    }
    fn execution_provider_type(&self) -> Option<&str> {
        Some("CPUExecutionProvider")
    }
    fn inputs(&self) -> Vec<OperatorInput> {
        (0..3)
            .map(|_| {
                OperatorInput::required(TensorElementType::Float16)
                    .memory_type(MemoryType::CPUInput)
            })
            .collect()
    }
    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float16)]
    }
    fn max_version(&self) -> i32 {
        1
    }
    fn infer_shape(&self, context: &mut ShapeInferenceContext) -> Result<()> {
        check_attributes(
            self.kind,
            context.attr::<i64>("query_chunk")?,
            context.attr::<f32>("query_scale")?,
        )?;
        let geometry = check_input_types(self.kind, &context.inputs())?;
        let shape =
            geometry.map_or_else(|| self.kind.unknown_output_shape(), Geometry::output_shape);
        context.set_output(0, &tensor_type(&shape))
    }
    fn create_kernel(&self, attributes: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        check_attributes(
            self.kind,
            attributes
                .get::<i64>("query_chunk")
                .ok_or_else(|| Error::new("native attention query_chunk missing"))?,
            attributes
                .get::<f32>("query_scale")
                .ok_or_else(|| Error::new("native attention query_scale missing"))?,
        )?;
        Ok(Box::new(MapKernel {
            config: Arc::clone(&self.config),
            kind: self.kind,
            engine: None,
        }))
    }
}

/// Register only when the explicitly opted-in graph references this domain.
/// This validates paths/metadata but does not load any DLL or initialize a GPU.
pub fn domain(
    helper_dll: PathBuf,
    directml_dll: PathBuf,
    adapter_id: i32,
) -> Result<OperatorDomain> {
    if !cfg!(windows) || adapter_id < 0 {
        return Err(Error::new(
            "native attention experiment requires Windows and an explicit DXGI adapter",
        ));
    }
    for path in [&helper_dll, &directml_dll] {
        if !path.is_absolute() || !path.is_file() {
            return Err(Error::new(format!(
                "native attention requires an existing absolute DLL path: {}",
                path.display()
            )));
        }
    }
    let config = Arc::new(Config {
        helper_dll,
        directml_dll,
        adapter_id,
        cached: Mutex::new(EngineCache::new()),
        cached_masked: Mutex::new(EngineCache::new()),
        cached_strip: Mutex::new(EngineCache::new()),
    });
    let domain = OperatorDomain::new(DOMAIN)?
        .add(NativeAttentionMap1080 {
            config: Arc::clone(&config),
            kind: MapKind::Encoder,
        })?
        .add(NativeAttentionMap1080 {
            config: Arc::clone(&config),
            kind: MapKind::Body,
        })?;
    let domain = masked::register(domain, Arc::clone(&config))?;
    let domain = strip::register(domain, Arc::clone(&config))?;
    encoder::register(domain, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_schema_and_scale_are_exact() {
        check_input_types(
            MapKind::Encoder,
            &[
                tensor_type(&Q_SHAPE),
                tensor_type(&K_SHAPE),
                tensor_type(&Q_SHAPE),
            ],
        )
        .unwrap();
        check_attributes(MapKind::Encoder, 128, f16::from_bits(0x35a8).to_f32()).unwrap();
        for (chunk, scale) in [(64, SCALE), (128, 0.3535), (128, f32::NAN)] {
            assert!(check_attributes(MapKind::Encoder, chunk, scale).is_err());
        }
        for shape in [[1, 8, 8704, 64], [1, 16, 8640, 64], [2, 8, 8640, 64]] {
            assert!(
                check_input_types(
                    MapKind::Encoder,
                    &[
                        tensor_type(&shape),
                        tensor_type(&K_SHAPE),
                        tensor_type(&Q_SHAPE)
                    ]
                )
                .is_err()
            );
        }
        assert_eq!(OUTPUT_SHAPE.iter().product::<i64>(), 4_456_448);
    }

    #[test]
    fn body_geometries_preserve_full_scan_and_exact_scale() {
        check_attributes(MapKind::Body, 128, f16::from_bits(0x351d).to_f32()).unwrap();
        assert!(check_attributes(MapKind::Body, 128, SCALE).is_err());
        for frames in 1..=8 {
            let geometry = MapKind::Body.geometry(540 * frames).unwrap();
            let inferred = check_input_types(
                MapKind::Body,
                &[
                    tensor_type(&geometry.q_shape()),
                    tensor_type(&geometry.k_shape()),
                    tensor_type(&geometry.q_shape()),
                ],
            )
            .unwrap()
            .unwrap();
            assert_eq!(inferred, geometry);
            let shape = geometry.output_shape();
            assert_eq!(shape, [(540 * frames + 127) / 128, 1, 16, 128, 96]);
            assert_eq!(geometry.scale_bits, BODY_SCALE.to_bits());
            assert!(shape[0] * 128 >= 540 * frames);
            assert!(shape[0] * 128 - 540 * frames < 128);
        }
        assert_eq!(
            MapKind::Body.geometry(2160).unwrap().output_shape(),
            [17, 1, 16, 128, 96]
        );
        assert_eq!(
            MapKind::Body.geometry(4320).unwrap().output_shape(),
            [34, 1, 16, 128, 96]
        );
        for queries in [-1, 0, 539, 541, 4319, 4321, 4860, i64::MAX] {
            assert!(MapKind::Body.geometry(queries).is_err());
        }
    }

    #[test]
    fn body_metadata_checks_consistency_and_defers_symbolic_queries() {
        let symbolic_q = [1, 16, -1, 96];
        let symbolic_k = [1, 16, 96, -1];
        assert_eq!(
            check_shapes(MapKind::Body, &[&symbolic_q, &symbolic_k, &symbolic_q]).unwrap(),
            None
        );
        let body4 = MapKind::Body.geometry(2160).unwrap();
        assert_eq!(
            check_shapes(MapKind::Body, &[&symbolic_q, &body4.k_shape(), &symbolic_q]).unwrap(),
            Some(body4)
        );
        assert!(
            check_shapes(
                MapKind::Body,
                &[&body4.q_shape(), &body4.k_shape(), &[1, 16, 4320, 96]]
            )
            .is_err()
        );
        for shape in [
            [2, 16, 2160, 96],
            [1, 8, 2160, 96],
            [1, 16, 2160, 80],
            [1, 16, 0, 96],
            [1, 16, 8640, 96],
        ] {
            assert!(
                check_shapes(MapKind::Body, &[&shape, &body4.k_shape(), &body4.q_shape()]).is_err()
            );
        }
        assert!(
            check_shapes(
                MapKind::Body,
                &[&[1, 16, 2160], &body4.k_shape(), &body4.q_shape()]
            )
            .is_err()
        );
        let mut wrong_type = tensor_type(&body4.q_shape());
        if let ValueType::Tensor { ty, .. } = &mut wrong_type {
            *ty = TensorElementType::Float32;
        }
        assert!(
            check_input_types(
                MapKind::Body,
                &[
                    wrong_type,
                    tensor_type(&body4.k_shape()),
                    tensor_type(&body4.q_shape())
                ]
            )
            .is_err()
        );
    }

    #[test]
    fn unknown_shape_axes_have_symbols_so_ort_never_serializes_negative_dim_values() {
        let ValueType::Tensor {
            shape,
            dimension_symbols,
            ..
        } = tensor_type(&[-1, 1, 16, 128, 96])
        else {
            panic!("expected tensor type");
        };
        assert_eq!(&*shape, &[-1, 1, 16, 128, 96]);
        assert_eq!(dimension_symbols[0], "native_attention_unknown_axis_0");
        assert!(dimension_symbols.iter().skip(1).all(String::is_empty));
        let ValueType::Tensor {
            dimension_symbols, ..
        } = tensor_type(&[17, 1, 16, 128, 96])
        else {
            panic!("expected static tensor type");
        };
        assert!(dimension_symbols.iter().all(String::is_empty));
        let ValueType::Tensor {
            dimension_symbols, ..
        } = tensor_type(&[-1, 1, -1])
        else {
            panic!("expected symbolic tensor type");
        };
        assert!(!dimension_symbols[0].is_empty());
        assert!(!dimension_symbols[2].is_empty());
        assert_ne!(dimension_symbols[0], dimension_symbols[2]);
    }

    #[test]
    fn geometry_cache_shares_ten_kernels_and_drops_without_owning_engines() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct FakeEngine(Arc<AtomicUsize>);
        impl Drop for FakeEngine {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let mut cache = EngineCache::new();
        let body4 = MapKind::Body.geometry(2160).unwrap();
        let body8 = MapKind::Body.geometry(4320).unwrap();
        let encoder = MapKind::Encoder.geometry(8640).unwrap();
        let mut kernels = Vec::new();
        for _ in 0..10 {
            kernels.push(
                cache
                    .get_or_create(body4, || Ok(FakeEngine(Arc::clone(&dropped))))
                    .unwrap(),
            );
        }
        assert!(
            kernels
                .iter()
                .all(|engine| Arc::ptr_eq(engine, &kernels[0]))
        );
        let other = cache
            .get_or_create(body8, || Ok(FakeEngine(Arc::clone(&dropped))))
            .unwrap();
        let encoder_engine = cache
            .get_or_create(encoder, || Ok(FakeEngine(Arc::clone(&dropped))))
            .unwrap();
        assert!(!Arc::ptr_eq(&other, &kernels[0]));
        assert_eq!(cache.entries.len(), 3);
        kernels.clear();
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        drop(other);
        drop(encoder_engine);
        assert_eq!(dropped.load(Ordering::SeqCst), 3);
        assert!(
            cache
                .entries
                .values()
                .all(|entry| entry.upgrade().is_none())
        );
        let revived = cache
            .get_or_create(body4, || Ok(FakeEngine(Arc::clone(&dropped))))
            .unwrap();
        assert_eq!(cache.entries.len(), 1);
        drop(cache);
        assert_eq!(dropped.load(Ordering::SeqCst), 3);
        drop(revived);
        assert_eq!(dropped.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn failed_geometry_creation_leaves_no_retained_cache_entry() {
        let mut cache = EngineCache::<()>::new();
        let geometry = MapKind::Body.geometry(2160).unwrap();
        assert!(
            cache
                .get_or_create(geometry, || Err(Error::new("test creation failed")))
                .is_err()
        );
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn metadata_validation_does_not_load_a_runtime() {
        assert!(domain("missing-helper.dll".into(), "missing-dml.dll".into(), 0).is_err());
        assert!(domain("missing-helper.dll".into(), "missing-dml.dll".into(), -1).is_err());
        assert_eq!(std::mem::size_of::<f16>(), std::mem::size_of::<u16>());
        assert_eq!(std::mem::align_of::<f16>(), std::mem::align_of::<u16>());
    }
}
