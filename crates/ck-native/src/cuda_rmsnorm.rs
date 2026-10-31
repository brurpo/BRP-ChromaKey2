//! CUDA-only source-faithful FP32 RMSNorm/LayerNorm for selected model packs.
//! Embedded PTX is compiled by a developer using NVRTC; runtime uses only the
//! installed CUDA driver, ORT-owned device tensors and ORT's compute stream.
use libloading::Library;
use ort::{
    Error, Result,
    memory::AllocationDevice,
    operator::{
        Operator, OperatorDomain, ShapeInferenceContext,
        io::{OperatorInput, OperatorOutput},
        kernel::{Kernel, KernelAttributes, KernelContext},
    },
    tensor::{Shape, SymbolicDimensions, TensorElementType},
    value::{TensorValueType, ValueType},
};
use std::{
    ffi::{CStr, c_char, c_void},
    ptr,
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, OnceLock},
};

const PTX: &[u8] = include_bytes!("../kernels/rmsnorm_fp32.ptx");
const LAYER_PTX: &[u8] = include_bytes!("../kernels/layernorm_fp32.ptx");

#[derive(Clone, Copy, PartialEq, Eq)]
enum NormKind {
    Rms,
    Layer,
}
impl NormKind {
    fn name(self) -> &'static str {
        match self {
            Self::Rms => "RmsNorm",
            Self::Layer => "LayerNorm",
        }
    }
    fn binary(self) -> (&'static [u8], &'static CStr) {
        match self {
            Self::Rms => (PTX, c"brp_rmsnorm_fp32"),
            Self::Layer => (LAYER_PTX, c"brp_layernorm_fp32"),
        }
    }
}
type Handle = *mut c_void;
type Init = unsafe extern "system" fn(u32) -> i32;
type PrimaryRetain = unsafe extern "system" fn(*mut Handle, i32) -> i32;
type PrimaryRelease = unsafe extern "system" fn(i32) -> i32;
type ContextGet = unsafe extern "system" fn(*mut Handle) -> i32;
type ContextPush = unsafe extern "system" fn(Handle) -> i32;
type ContextPop = unsafe extern "system" fn(*mut Handle) -> i32;
type ModuleLoad = unsafe extern "system" fn(*mut Handle, *const c_void) -> i32;
type ModuleUnload = unsafe extern "system" fn(Handle) -> i32;
type FunctionGet = unsafe extern "system" fn(*mut Handle, Handle, *const c_char) -> i32;
type Launch = unsafe extern "system" fn(
    Handle,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    Handle,
    *mut *mut c_void,
    *mut *mut c_void,
) -> i32;
type ErrorText = unsafe extern "system" fn(i32, *mut *const c_char) -> i32;

struct Driver {
    _library: Library,
    retain: PrimaryRetain,
    release: PrimaryRelease,
    context_get: ContextGet,
    context_push: ContextPush,
    context_pop: ContextPop,
    module_load: ModuleLoad,
    module_unload: ModuleUnload,
    function_get: FunctionGet,
    launch: Launch,
    error_text: ErrorText,
}

impl Driver {
    fn load() -> Result<Arc<Self>> {
        static DRIVER: OnceLock<std::result::Result<Arc<Driver>, String>> = OnceLock::new();
        DRIVER
            .get_or_init(|| {
                unsafe { Self::load_inner() }
                    .map(Arc::new)
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(|error| Error::new(error.clone()))
    }

    unsafe fn load_inner() -> Result<Self> {
        // Search the OS driver directory on Windows, never the app/work folder.
        #[cfg(windows)]
        let library: Library =
            unsafe { libloading::os::windows::Library::load_with_flags("nvcuda.dll", 0x0000_0800) }
                .map_err(|e| Error::new(format!("loading system CUDA driver: {e}")))?
                .into();
        #[cfg(not(windows))]
        let library = unsafe { Library::new("libcuda.so.1") }
            .map_err(|e| Error::new(format!("loading CUDA driver: {e}")))?;
        macro_rules! symbol {
            ($name:literal, $ty:ty) => {
                *unsafe { library.get::<$ty>(concat!($name, "\0").as_bytes()) }
                    .map_err(|e| Error::new(format!("CUDA driver symbol {}: {e}", $name)))?
            };
        }
        let init = symbol!("cuInit", Init);
        let driver = Self {
            retain: symbol!("cuDevicePrimaryCtxRetain", PrimaryRetain),
            release: symbol!("cuDevicePrimaryCtxRelease_v2", PrimaryRelease),
            context_get: symbol!("cuCtxGetCurrent", ContextGet),
            context_push: symbol!("cuCtxPushCurrent_v2", ContextPush),
            context_pop: symbol!("cuCtxPopCurrent_v2", ContextPop),
            module_load: symbol!("cuModuleLoadData", ModuleLoad),
            module_unload: symbol!("cuModuleUnload", ModuleUnload),
            function_get: symbol!("cuModuleGetFunction", FunctionGet),
            launch: symbol!("cuLaunchKernel", Launch),
            error_text: symbol!("cuGetErrorString", ErrorText),
            _library: library,
        };
        driver.check(unsafe { init(0) }, "cuInit")?;
        Ok(driver)
    }

    fn check(&self, result: i32, operation: &str) -> Result<()> {
        if result == 0 {
            return Ok(());
        }
        let mut message = ptr::null();
        let status = unsafe { (self.error_text)(result, &mut message) };
        let text = if status == 0 && !message.is_null() {
            unsafe { CStr::from_ptr(message) }
                .to_string_lossy()
                .into_owned()
        } else {
            "CUDA driver did not provide error text".into()
        };
        Err(Error::new(format!(
            "{operation}: CUDA error {result}: {text}"
        )))
    }
}

struct ContextGuard<'a> {
    driver: &'a Driver,
}
impl<'a> ContextGuard<'a> {
    fn enter(driver: &'a Driver, context: Handle) -> Result<Self> {
        let mut current = ptr::null_mut();
        driver.check(
            unsafe { (driver.context_get)(&mut current) },
            "cuCtxGetCurrent",
        )?;
        if !current.is_null() && current != context {
            return Err(Error::new(
                "RmsNorm supports ORT's CUDA primary context; a different active context was found",
            ));
        }
        driver.check(
            unsafe { (driver.context_push)(context) },
            "cuCtxPushCurrent",
        )?;
        Ok(Self { driver })
    }
}
impl Drop for ContextGuard<'_> {
    fn drop(&mut self) {
        let mut prior = ptr::null_mut();
        let _ = unsafe { (self.driver.context_pop)(&mut prior) };
    }
}

struct Module {
    driver: Arc<Driver>,
    context: Handle,
    device: i32,
    module: Handle,
    function: Handle,
}
impl Module {
    fn load(driver: Arc<Driver>, device: i32, kind: NormKind) -> Result<Self> {
        let (ptx, entry) = kind.binary();
        let mut context = ptr::null_mut();
        driver.check(
            unsafe { (driver.retain)(&mut context, device) },
            "cuDevicePrimaryCtxRetain",
        )?;
        let mut module = Self {
            driver,
            context,
            device,
            module: ptr::null_mut(),
            function: ptr::null_mut(),
        };
        {
            let _context = ContextGuard::enter(&module.driver, context)?;
            module.driver.check(
                unsafe { (module.driver.module_load)(&mut module.module, ptx.as_ptr().cast()) },
                "loading embedded RMSNorm PTX",
            )?;
            module.driver.check(
                unsafe {
                    (module.driver.function_get)(
                        &mut module.function,
                        module.module,
                        entry.as_ptr(),
                    )
                },
                "finding RMSNorm PTX entry",
            )?;
        }
        Ok(module)
    }
}
impl Drop for Module {
    fn drop(&mut self) {
        // ORT owns the stream and allocations. We only own this module/retain.
        // cuModuleUnload completes outstanding uses before releasing code.
        if !self.module.is_null() {
            // Cleanup may run on a different worker; preserve its context stack.
            let pushed = unsafe { (self.driver.context_push)(self.context) } == 0;
            if pushed {
                let _ = unsafe { (self.driver.module_unload)(self.module) };
                let mut prior = ptr::null_mut();
                let _ = unsafe { (self.driver.context_pop)(&mut prior) };
            }
        }
        let _ = unsafe { (self.driver.release)(self.device) };
    }
}

fn dimensions(shape: &[i64], gamma: &[i64]) -> Result<(i32, u32)> {
    let columns = *shape
        .last()
        .ok_or_else(|| Error::new("RmsNorm requires a tensor rank >=1"))?;
    if columns <= 0 || columns > (1 << 24) || columns % 4 != 0 || gamma != [columns] {
        return Err(Error::new(
            "RmsNorm requires aligned vectorized FP32 width N%4==0, N<=2^24, and gamma[N]",
        ));
    }
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1u32, |rows, &dimension| {
            u32::try_from(dimension)
                .ok()
                .filter(|&d| d > 0)
                .and_then(|d| rows.checked_mul(d))
        })
        .filter(|&rows| rows <= i32::MAX as u32)
        .ok_or_else(|| Error::new("RmsNorm row count is empty or exceeds CUDA grid capacity"))?;
    Ok((columns as i32, rows))
}

fn inferred_output(input: &ValueType, gamma: &ValueType) -> Result<ValueType> {
    static UNKNOWN: AtomicU64 = AtomicU64::new(0);
    let ValueType::Tensor {
        ty,
        shape,
        dimension_symbols,
    } = input
    else {
        return Err(Error::new("RmsNorm input must be a tensor"));
    };
    let width = gamma
        .tensor_shape()
        .filter(|shape| shape.len() == 1 && shape[0] > 0)
        .map(|shape| shape[0])
        .ok_or_else(|| Error::new("RmsNorm needs a known rank-one Gamma width"))?;
    let mut dimensions = shape.to_vec();
    let last = dimensions
        .last_mut()
        .ok_or_else(|| Error::new("RmsNorm input rank must be known and positive"))?;
    if *last > 0 && *last != width {
        return Err(Error::new("RmsNorm input/Gamma inference width mismatch"));
    }
    *last = width;
    let unique = UNKNOWN.fetch_add(1, Ordering::Relaxed);
    let names = dimensions.iter().enumerate().map(|(axis, &size)| {
        if size >= 0 {
            String::new()
        } else {
            let existing = dimension_symbols
                .get(axis)
                .map(String::as_str)
                .unwrap_or("");
            if existing.is_empty() {
                format!("brp_rms_unknown_{unique}_{axis}")
            } else {
                existing.to_owned()
            }
        }
    });
    // ORT 1.23's custom shape bridge serializes an anonymous -1 as a literal
    // negative dim_value. A nonempty symbol preserves its actual unknown status.
    Ok(ValueType::Tensor {
        ty: *ty,
        shape: Shape::new(dimensions.iter().copied()),
        dimension_symbols: SymbolicDimensions::new(names),
    })
}

struct RmsKernel {
    driver: Arc<Driver>,
    module: Option<Module>,
    epsilon: f32,
    kind: NormKind,
}
impl Kernel for RmsKernel {
    fn compute(&mut self, context: &KernelContext) -> Result<()> {
        let input = context
            .input(0)?
            .ok_or_else(|| Error::new("RmsNorm missing X"))?
            .downcast::<TensorValueType<f32>>()?;
        let gamma = context
            .input(1)?
            .ok_or_else(|| Error::new("RmsNorm missing Gamma"))?
            .downcast::<TensorValueType<f32>>()?;
        let beta = if self.kind == NormKind::Layer {
            Some(
                context
                    .input(2)?
                    .ok_or_else(|| Error::new("LayerNorm missing Beta"))?
                    .downcast::<TensorValueType<f32>>()?,
            )
        } else {
            None
        };
        let shape = input.shape().to_vec();
        let (mut columns, rows) = dimensions(&shape, gamma.shape())?;
        if input.memory_info().allocation_device() != AllocationDevice::CUDA
            || gamma.memory_info().allocation_device() != AllocationDevice::CUDA
        {
            return Err(Error::new(
                "RmsNorm requires actual CUDA device tensors; host pointers are not accepted",
            ));
        }
        let device = input.memory_info().device_id();
        if gamma.memory_info().device_id() != device {
            return Err(Error::new("RmsNorm X/Gamma device mismatch"));
        }
        if beta.as_ref().is_some_and(|beta| {
            beta.shape() != gamma.shape()
                || beta.memory_info().allocation_device() != AllocationDevice::CUDA
                || beta.memory_info().device_id() != device
        }) {
            return Err(Error::new(
                "LayerNorm Beta must match Gamma shape and CUDA device",
            ));
        }
        let mut output = context
            .output(0, shape)?
            .ok_or_else(|| Error::new("RmsNorm missing Y allocation"))?
            .downcast::<TensorValueType<f32>>()?;
        if output.memory_info().allocation_device() != AllocationDevice::CUDA
            || output.memory_info().device_id() != device
        {
            return Err(Error::new("RmsNorm output is not on the input CUDA device"));
        }
        let mut x = input.data_ptr() as u64;
        let mut g = gamma.data_ptr() as u64;
        let mut y = output.data_ptr_mut() as u64;
        let mut b = beta.as_ref().map_or(0, |beta| beta.data_ptr() as u64);
        if [x, g, y]
            .iter()
            .any(|&address| address == 0 || address % 16 != 0)
        {
            return Err(Error::new(
                "RmsNorm needs 16-byte aligned CUDA X/Gamma/Y pointers; no alternate reduction is silently substituted",
            ));
        }
        if beta.is_some() && (b == 0 || b % 16 != 0) {
            return Err(Error::new("LayerNorm Beta requires 16-byte CUDA alignment"));
        }
        if self.module.is_none() {
            self.module = Some(Module::load(Arc::clone(&self.driver), device, self.kind)?);
        }
        let module = self.module.as_ref().expect("module initialized above");
        if module.device != device {
            return Err(Error::new("RmsNorm kernel cannot change CUDA devices"));
        }
        let _context = ContextGuard::enter(&self.driver, module.context)?;
        // A null CUDA stream denotes the valid CUDA default stream. It is not
        // dereferenced; all device accesses happen inside cuLaunchKernel.
        let stream = context
            .compute_stream()?
            .map_or(ptr::null_mut(), |stream| stream.as_ptr());
        let mut epsilon = self.epsilon;
        let mut arguments = vec![
            (&mut x as *mut u64).cast::<c_void>(),
            (&mut g as *mut u64).cast(),
        ];
        if beta.is_some() {
            arguments.push((&mut b as *mut u64).cast());
        }
        arguments.extend([
            (&mut y as *mut u64).cast(),
            (&mut columns as *mut i32).cast(),
            (&mut epsilon as *mut f32).cast(),
        ]);
        self.driver.check(
            unsafe {
                (self.driver.launch)(
                    module.function,
                    rows,
                    1,
                    1,
                    32,
                    4,
                    1,
                    24,
                    stream,
                    arguments.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "launching RMSNorm",
        )
    }
}

struct RmsNorm {
    kind: NormKind,
}
impl Operator for RmsNorm {
    fn name(&self) -> &str {
        self.kind.name()
    }
    fn execution_provider_type(&self) -> Option<&str> {
        Some("CUDAExecutionProvider")
    }
    fn inputs(&self) -> Vec<OperatorInput> {
        let mut inputs = vec![
            OperatorInput::required(TensorElementType::Float32),
            OperatorInput::required(TensorElementType::Float32),
        ];
        if self.kind == NormKind::Layer {
            inputs.push(OperatorInput::required(TensorElementType::Float32));
        }
        inputs
    }
    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![OperatorOutput::required(TensorElementType::Float32)]
    }
    fn max_version(&self) -> i32 {
        1
    }
    fn infer_shape(&self, context: &mut ShapeInferenceContext) -> Result<()> {
        let inputs = context.inputs();
        let input = inputs
            .first()
            .ok_or_else(|| Error::new("RmsNorm shape inference missing X"))?;
        let gamma = inputs
            .get(1)
            .ok_or_else(|| Error::new("RmsNorm shape inference missing Gamma"))?;
        context.set_output(0, &inferred_output(input, gamma)?)
    }
    fn create_kernel(&self, attributes: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        let epsilon = attributes
            .get::<f32>("epsilon")
            .ok_or_else(|| Error::new("RmsNorm requires float epsilon attribute"))?;
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Error::new("RmsNorm epsilon must be positive and finite"));
        }
        Ok(Box::new(RmsKernel {
            driver: Driver::load()?,
            module: None,
            epsilon,
            kind: self.kind,
        }))
    }
}

pub fn operator_domain() -> Result<OperatorDomain> {
    OperatorDomain::new("brp.chroma")?
        .add(RmsNorm {
            kind: NormKind::Rms,
        })?
        .add(RmsNorm {
            kind: NormKind::Layer,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_vectorized_source_dispatch_and_all_ck_widths() {
        for width in [64, 96, 384, 512, 768, 1152, 1536, 2816] {
            assert_eq!(
                dimensions(&[2, 3, width], &[width]).unwrap(),
                (width as i32, 6)
            );
        }
        for (shape, gamma) in [
            (vec![], vec![64]),
            (vec![2, 63], vec![63]),
            (vec![2, 64], vec![32]),
            (vec![0, 64], vec![64]),
            (vec![i64::MAX, 64], vec![64]),
        ] {
            assert!(dimensions(&shape, &gamma).is_err());
        }
    }
    #[test]
    fn embedded_kernel_contains_source_fma_and_fixed_entry() {
        assert_eq!(PTX.last(), Some(&0));
        let source = std::str::from_utf8(&PTX[..PTX.len() - 1]).unwrap();
        assert!(source.contains(".entry brp_rmsnorm_fp32"));
        assert!(source.contains("fma.rn.f32"));
        assert!(source.contains("shfl.sync.bfly") || source.contains("shfl.sync.down"));
    }
    #[test]
    fn layernorm_registration_uses_separate_affine_kernel_and_bias_input() {
        let layer = RmsNorm {
            kind: NormKind::Layer,
        };
        let rms = RmsNorm {
            kind: NormKind::Rms,
        };
        assert_eq!(layer.name(), "LayerNorm");
        assert_eq!(layer.inputs().len(), 3);
        assert_eq!(rms.inputs().len(), 2);
        assert_ne!(layer.kind.binary().0, rms.kind.binary().0);
        let source = std::str::from_utf8(&LAYER_PTX[..LAYER_PTX.len() - 1]).unwrap();
        assert_eq!(LAYER_PTX.last(), Some(&0));
        assert!(source.contains(".entry brp_layernorm_fp32"));
        assert!(source.contains("fma.rn.f32"));
    }
    #[test]
    fn unknown_shapes_never_round_trip_as_literal_negative_dimensions() {
        let input = ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape: Shape::new([-1, 2, -1, 512]),
            dimension_symbols: SymbolicDimensions::new([
                "batch".into(),
                "".into(),
                "".into(),
                "".into(),
            ]),
        };
        let gamma = ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape: Shape::new([512]),
            dimension_symbols: SymbolicDimensions::new([String::new()]),
        };
        let ValueType::Tensor {
            shape,
            dimension_symbols,
            ..
        } = inferred_output(&input, &gamma).unwrap()
        else {
            panic!("expected tensor")
        };
        assert_eq!(&shape[..], &[-1, 2, -1, 512]);
        assert_eq!(dimension_symbols[0], "batch");
        assert!(!dimension_symbols[2].is_empty());
        assert!(dimension_symbols[1].is_empty() && dimension_symbols[3].is_empty());
    }
}
