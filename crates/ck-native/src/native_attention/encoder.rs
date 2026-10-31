//! Opt-in full encoder block. CPU boundaries surround GPU-resident children.
//!
//! The parent ORT graph must finish its host transfer before this kernel runs.
//! The helper owns its DML1 queue and returns only after the final CPU copy.
//! Every child session and GPU buffer dies with this Stage1 kernel.
use super::{Config, ffi_error, symbol, tensor_type};
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
    tensor::TensorElementType,
    value::ValueType,
};
use std::{
    ffi::{CString, c_char, c_void},
    path::{Path, PathBuf},
    ptr::NonNull,
    sync::Arc,
};

const NAME: &str = "NativeEncoderBlock1080";
const FRAME: [i64; 5] = [1, 1, 72, 120, 512];
const CONDITION: [i64; 2] = [1, 1];
const ELEMENTS: usize = 72 * 120 * 512;

#[repr(C)]
struct FfiConfig {
    abi_version: u32,
    adapter_id: i32,
    ort_path_utf8: *const c_char,
    directml_path_utf8: *const c_char,
    pre_model_utf8: *const c_char,
    post_model_utf8: *const c_char,
}
type Version = unsafe extern "C" fn() -> u32;
type Create = unsafe extern "C" fn(*const FfiConfig, *mut *mut c_void, *mut c_char, usize) -> i32;
type Run = unsafe extern "C" fn(
    *mut c_void,
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

fn attributes(index: i64, version: i64) -> Result<usize> {
    if !(0..=1).contains(&index) || version != 1 {
        return Err(Error::new(
            "resident encoder requires index 0/1 and layout_version 1",
        ));
    }
    Ok(index as usize)
}

fn check_types(inputs: &[ValueType]) -> Result<()> {
    if inputs.len() != 2 {
        return Err(Error::new("resident encoder requires frame and condition"));
    }
    for (input, expected) in inputs.iter().zip([FRAME.as_slice(), CONDITION.as_slice()]) {
        match input {
            ValueType::Tensor {
                ty: TensorElementType::Float16,
                shape,
                ..
            } if shape.len() == expected.len()
                && shape
                    .iter()
                    .zip(expected)
                    .all(|(&actual, &want)| actual < 0 || actual == want) => {}
            _ => {
                return Err(Error::new(
                    "resident encoder requires FP16 B1/T1/72x120x512 and B1/T1 condition",
                ));
            }
        }
    }
    Ok(())
}

fn path_string(path: &Path) -> Result<CString> {
    let text = path
        .to_str()
        .ok_or_else(|| Error::new("resident encoder path must be Unicode"))?;
    CString::new(text).map_err(|_| Error::new("resident encoder path contains a zero byte"))
}

fn explicit_file(variable: &str) -> Result<PathBuf> {
    let path = std::env::var_os(variable)
        .map(PathBuf::from)
        .ok_or_else(|| Error::new(format!("resident encoder requires {variable}")))?;
    if !path.is_absolute() || !path.is_file() {
        return Err(Error::new(format!(
            "{variable} must name an existing absolute file"
        )));
    }
    Ok(path)
}

fn model_paths(root: &Path, index: usize) -> Result<[PathBuf; 2]> {
    if index > 1 || !root.is_absolute() {
        return Err(Error::new(
            "resident encoder requires an absolute model root and index 0/1",
        ));
    }
    let directory = root.join(format!("enc{index}"));
    Ok([
        directory.join("pre-model/model.onnx"),
        directory.join("post-model/model.onnx"),
    ])
}

struct Engine {
    pointer: NonNull<c_void>,
    run: Run,
    destroy: Destroy,
    failed: bool,
    // Drop invokes the native destructor while this handle is still alive.
    _library: Library,
}

// Compute has exclusive &mut ownership. The native object contains free-threaded
// D3D12/DML objects; no Rust input pointer is retained across its synchronous call.
unsafe impl Send for Engine {}

impl Engine {
    fn new(common: &Config, index: usize) -> Result<Self> {
        let helper = explicit_file("BRP_CHROMA_RESIDENT_ENCODER_DLL")?;
        let resident_identity = helper
            .canonicalize()
            .map_err(|error| Error::new(format!("resident encoder helper path: {error}")))?;
        let attention_identity = common
            .helper_dll
            .canonicalize()
            .map_err(|error| Error::new(format!("native attention helper path: {error}")))?;
        if resident_identity != attention_identity {
            return Err(Error::new(
                "resident encoder and attention maps must use the same helper DLL to share score scratch",
            ));
        }
        let runtime = explicit_file("BRP_CHROMA_RESIDENT_ENCODER_ORT_DLL")?;
        let root = std::env::var_os("BRP_CHROMA_RESIDENT_ENCODER_MODELS")
            .map(PathBuf::from)
            .ok_or_else(|| {
                Error::new("resident encoder requires BRP_CHROMA_RESIDENT_ENCODER_MODELS")
            })?;
        let [pre, post] = model_paths(&root, index)?;
        for path in [&pre, &post] {
            if !path.is_file() {
                return Err(Error::new(format!(
                    "resident encoder model is missing: {}",
                    path.display()
                )));
            }
        }
        let runtime = path_string(&runtime)?;
        let directml = path_string(&common.directml_dll)?;
        let pre = path_string(&pre)?;
        let post = path_string(&post)?;
        let library = unsafe { Library::new(helper) }
            .map_err(|error| Error::new(format!("resident encoder helper load: {error}")))?;
        let version: Version = symbol(&library, b"brp_encoder_abi_version\0")?;
        if unsafe { version() } != 1 {
            return Err(Error::new("resident encoder helper ABI must be 1"));
        }
        let create: Create = symbol(&library, b"brp_encoder_create\0")?;
        let run: Run = symbol(&library, b"brp_encoder_run\0")?;
        let destroy: Destroy = symbol(&library, b"brp_encoder_destroy\0")?;
        let config = FfiConfig {
            abi_version: 1,
            adapter_id: common.adapter_id,
            ort_path_utf8: runtime.as_ptr(),
            directml_path_utf8: directml.as_ptr(),
            pre_model_utf8: pre.as_ptr(),
            post_model_utf8: post.as_ptr(),
        };
        let mut raw = std::ptr::null_mut();
        let mut error = [0 as c_char; 1024];
        // All four strings are copied by create; the helper never borrows them.
        let status = unsafe { create(&config, &mut raw, error.as_mut_ptr(), error.len()) };
        if status != 0 {
            if !raw.is_null() {
                unsafe { destroy(raw) };
            }
            return Err(ffi_error(&error, "resident encoder creation failed"));
        }
        let pointer =
            NonNull::new(raw).ok_or_else(|| Error::new("resident encoder returned null"))?;
        Ok(Self {
            pointer,
            run,
            destroy,
            failed: false,
            _library: library,
        })
    }

    fn run(&mut self, frame: &[f16], condition: &[f16], output: &mut [f16]) -> Result<()> {
        if self.failed
            || frame.len() != ELEMENTS
            || condition.len() != 1
            || output.len() != ELEMENTS
        {
            return Err(Error::new(
                "resident encoder is failed or tensor lengths are invalid",
            ));
        }
        let mut error = [0 as c_char; 1024];
        let status = unsafe {
            (self.run)(
                self.pointer.as_ptr(),
                frame.as_ptr().cast(),
                frame.len(),
                condition.as_ptr().cast(),
                condition.len(),
                output.as_mut_ptr().cast(),
                output.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            self.failed = true;
            return Err(ffi_error(&error, "resident encoder execution failed"));
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.pointer.as_ptr()) };
    }
}

struct EncoderKernel {
    common: Arc<Config>,
    index: usize,
    engine: Option<Engine>,
}

impl Kernel for EncoderKernel {
    fn compute(&mut self, context: &KernelContext) -> Result<()> {
        let frame = context
            .input(0)?
            .ok_or_else(|| Error::new("resident encoder frame missing"))?;
        let condition = context
            .input(1)?
            .ok_or_else(|| Error::new("resident encoder condition missing"))?;
        let (frame_shape, frame_data) = frame.try_extract_tensor::<f16>()?;
        let (condition_shape, condition_data) = condition.try_extract_tensor::<f16>()?;
        if **frame_shape != FRAME || **condition_shape != CONDITION {
            return Err(Error::new(
                "resident encoder runtime shape outside qualified B1/T1/1080p geometry",
            ));
        }
        if self.engine.is_none() {
            self.engine = Some(Engine::new(&self.common, self.index)?);
        }
        let mut output = context
            .output(0, FRAME)?
            .ok_or_else(|| Error::new("resident encoder output missing"))?;
        self.engine
            .as_mut()
            .ok_or_else(|| Error::new("resident encoder was not initialized"))?
            .run(
                frame_data,
                condition_data,
                output.try_extract_tensor_mut::<f16>()?.1,
            )
    }
}

struct EncoderOperator {
    common: Arc<Config>,
}

impl Operator for EncoderOperator {
    fn name(&self) -> &str {
        NAME
    }
    fn execution_provider_type(&self) -> Option<&str> {
        Some("CPUExecutionProvider")
    }
    fn inputs(&self) -> Vec<OperatorInput> {
        (0..2)
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
        attributes(
            context.attr::<i64>("encoder_index")?,
            context.attr::<i64>("layout_version")?,
        )?;
        check_types(&context.inputs())?;
        context.set_output(0, &tensor_type(&FRAME))
    }
    fn create_kernel(&self, attrs: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        let index = attributes(
            attrs
                .get::<i64>("encoder_index")
                .ok_or_else(|| Error::new("resident encoder index missing"))?,
            attrs
                .get::<i64>("layout_version")
                .ok_or_else(|| Error::new("resident encoder layout missing"))?,
        )?;
        Ok(Box::new(EncoderKernel {
            common: Arc::clone(&self.common),
            index,
            engine: None,
        }))
    }
}

pub(super) fn register(domain: OperatorDomain, common: Arc<Config>) -> Result<OperatorDomain> {
    domain.add(EncoderOperator { common })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_block_metadata_rejects_other_geometry_and_dtype() {
        check_types(&[tensor_type(&FRAME), tensor_type(&CONDITION)]).unwrap();
        check_types(&[tensor_type(&[1, -1, 72, 120, 512]), tensor_type(&[-1, 1])]).unwrap();
        for bad in [
            [1, 8, 72, 120, 512],
            [1, 1, 144, 240, 512],
            [1, 1, 120, 72, 512],
        ] {
            assert!(check_types(&[tensor_type(&bad), tensor_type(&CONDITION)]).is_err());
        }
        assert!(check_types(&[tensor_type(&FRAME), tensor_type(&[1])]).is_err());
        let mut wrong = tensor_type(&FRAME);
        if let ValueType::Tensor { ty, .. } = &mut wrong {
            *ty = TensorElementType::Float32;
        }
        assert!(check_types(&[wrong, tensor_type(&CONDITION)]).is_err());
    }

    #[test]
    fn layout_version_and_index_are_explicit() {
        assert_eq!(attributes(0, 1).unwrap(), 0);
        assert_eq!(attributes(1, 1).unwrap(), 1);
        for (index, version) in [(-1, 1), (2, 1), (0, 0), (1, 2)] {
            assert!(attributes(index, version).is_err());
        }
    }

    #[test]
    fn paths_cannot_escape_the_selected_encoder_bank() {
        let root = std::env::current_dir()
            .unwrap()
            .join("unused-resident-fixture");
        let paths = model_paths(&root, 1).unwrap();
        assert_eq!(paths[0], root.join("enc1/pre-model/model.onnx"));
        assert_eq!(paths[1], root.join("enc1/post-model/model.onnx"));
        assert!(model_paths(Path::new("relative"), 0).is_err());
        assert!(model_paths(&root, 2).is_err());
    }

    #[test]
    fn ffi_layout_matches_native_win64_config() {
        if cfg!(target_pointer_width = "64") {
            assert_eq!(std::mem::size_of::<FfiConfig>(), 40);
            assert_eq!(std::mem::offset_of!(FfiConfig, ort_path_utf8), 8);
            assert_eq!(std::mem::offset_of!(FfiConfig, post_model_utf8), 32);
        }
        assert_eq!(ELEMENTS, 4_423_680);
    }
}
