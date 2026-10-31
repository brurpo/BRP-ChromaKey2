//! Fixed-T4 CSwin vertical map with the original B/H axes. The native helper
//! retains at most B8 and B16 programs; no batch/head folding is performed.
use super::*;

const OPERATOR: &str = "NativeStripAttentionMap1080";

fn geometry() -> Geometry {
    Geometry {
        heads: 4,
        queries: 288,
        head_dim: 64,
        query_chunk: 128,
        scale_bits: SCALE.to_bits(),
    }
}

fn output_shape(batch: i64) -> [i64; 5] {
    [3, batch, 4, 128, 64]
}

fn output_type(batch: i64) -> ValueType {
    let mut value = tensor_type(&output_shape(batch));
    if let ValueType::Tensor {
        shape,
        dimension_symbols,
        ..
    } = &mut value
    {
        *dimension_symbols =
            SymbolicDimensions::new(shape.iter().enumerate().map(|(axis, &size)| {
                if size < 0 && axis == 1 {
                    "strip_scan_batch".to_owned()
                } else {
                    String::new()
                }
            }));
    }
    value
}

fn attributes(chunk: i64, scale: f32) -> Result<()> {
    if chunk != 128 || scale.to_bits() != SCALE.to_bits() {
        return Err(Error::new(
            "strip attention requires chunk128 and exact scale0.353515625",
        ));
    }
    Ok(())
}

fn shapes(inputs: &[&[i64]]) -> Result<Option<i64>> {
    if inputs.len() != 3 {
        return Err(Error::new("strip attention requires Q/K/V"));
    }
    let mut known_batch = None;
    for (index, shape) in inputs.iter().enumerate() {
        let expected = if index == 1 {
            [-1, 4, 64, 288]
        } else {
            [-1, 4, 288, 64]
        };
        if shape.len() != 4
            || shape
                .iter()
                .zip(expected)
                .any(|(&actual, required)| actual >= 0 && required >= 0 && actual != required)
        {
            return Err(Error::new(
                "strip attention requires Q/V[B,4,288,64], K[B,4,64,288]",
            ));
        }
        let batch = shape[0];
        if batch >= 0 {
            if ![8, 16].contains(&batch) || known_batch.is_some_and(|previous| previous != batch) {
                return Err(Error::new(
                    "strip attention actual batch must agree and be8 or16",
                ));
            }
            known_batch = Some(batch);
        }
    }
    Ok(known_batch)
}

fn input_types(inputs: &[ValueType]) -> Result<Option<i64>> {
    let mut input_shapes = Vec::with_capacity(inputs.len());
    for input in inputs {
        match input {
            ValueType::Tensor {
                ty: TensorElementType::Float16,
                shape,
                ..
            } => input_shapes.push(&**shape),
            _ => return Err(Error::new("strip attention requires FP16 Q/K/V")),
        }
    }
    shapes(&input_shapes)
}

pub(super) struct Engine {
    pointer: NonNull<c_void>,
    run: Run,
    destroy: Destroy,
    _library: Library,
}

// Every call and Drop is serialized by Arc<Mutex<_>>. The synchronous native
// ABI copies all borrowed inputs and completes its fence before returning.
unsafe impl Send for Engine {}

impl Engine {
    fn new(config: &Config) -> Result<Self> {
        let library = unsafe { Library::new(&config.helper_dll) }
            .map_err(|error| Error::new(format!("strip helper load failed: {error}")))?;
        let version: AbiVersion = symbol(&library, b"brp_strip_map_abi_version\0")?;
        if unsafe { version() } != 1 {
            return Err(Error::new("strip attention helper ABI must be1"));
        }
        let create: Create = symbol(&library, b"brp_strip_map_create\0")?;
        let run: Run = symbol(&library, b"brp_strip_map_run\0")?;
        let destroy: Destroy = symbol(&library, b"brp_strip_map_destroy\0")?;
        let path = config
            .directml_dll
            .to_str()
            .ok_or_else(|| Error::new("strip attention requires a Unicode DirectML path"))?;
        let path = CString::new(path)
            .map_err(|_| Error::new("strip DirectML path contains a zero byte"))?;
        let g = geometry();
        let abi = FfiConfig {
            abi_version: 1,
            adapter_id: config.adapter_id,
            heads: g.heads,
            queries: g.queries,
            head_dim: g.head_dim,
            query_chunk: g.query_chunk,
            query_scale: f32::from_bits(g.scale_bits),
            directml_path_utf8: path.as_ptr(),
        };
        let mut pointer = std::ptr::null_mut();
        let mut error = [0 as c_char; 1024];
        if unsafe { create(&abi, &mut pointer, error.as_mut_ptr(), error.len()) } != 0 {
            return Err(ffi_error(&error, "strip engine creation failed"));
        }
        Ok(Self {
            pointer: NonNull::new(pointer)
                .ok_or_else(|| Error::new("strip create returned null"))?,
            run,
            destroy,
            _library: library,
        })
    }

    fn run(&mut self, q: &[f16], k: &[f16], v: &[f16], output: &mut [f16]) -> Result<()> {
        let mut error = [0 as c_char; 1024];
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
            Err(ffi_error(&error, "strip execution failed"))
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.pointer.as_ptr()) };
    }
}

struct StripKernel {
    config: Arc<Config>,
    engine: Option<Arc<Mutex<Engine>>>,
}

impl Kernel for StripKernel {
    fn compute(&mut self, context: &KernelContext) -> Result<()> {
        let q = context
            .input(0)?
            .ok_or_else(|| Error::new("strip missing Q"))?;
        let k = context
            .input(1)?
            .ok_or_else(|| Error::new("strip missing K"))?;
        let v = context
            .input(2)?
            .ok_or_else(|| Error::new("strip missing V"))?;
        let (qs, q_data) = q.try_extract_tensor::<f16>()?;
        let (ks, k_data) = k.try_extract_tensor::<f16>()?;
        let (vs, v_data) = v.try_extract_tensor::<f16>()?;
        let actual = [&**qs, &**ks, &**vs];
        if actual.iter().any(|s| s.iter().any(|&d| d < 0)) {
            return Err(Error::new("strip runtime dimensions must be concrete"));
        }
        let batch = shapes(&actual)?.ok_or_else(|| Error::new("strip runtime batch missing"))?;
        if self.engine.is_none() {
            let engine = self
                .config
                .cached_strip
                .lock()
                .map_err(|_| Error::new("strip cache was poisoned"))?
                .get_or_create(geometry(), || Engine::new(&self.config))?;
            self.engine = Some(engine);
        }
        let mut output = context
            .output(0, output_shape(batch))?
            .ok_or_else(|| Error::new("strip output missing"))?;
        self.engine
            .as_ref()
            .ok_or_else(|| Error::new("strip engine missing"))?
            .lock()
            .map_err(|_| Error::new("strip engine mutex was poisoned"))?
            .run(
                q_data,
                k_data,
                v_data,
                output.try_extract_tensor_mut::<f16>()?.1,
            )
    }
}

struct StripOperator {
    config: Arc<Config>,
}

impl Operator for StripOperator {
    fn name(&self) -> &str {
        OPERATOR
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
        attributes(
            context.attr::<i64>("query_chunk")?,
            context.attr::<f32>("query_scale")?,
        )?;
        let batch = input_types(&context.inputs())?;
        context.set_output(0, &output_type(batch.unwrap_or(-1)))
    }
    fn create_kernel(&self, attrs: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        attributes(
            attrs
                .get::<i64>("query_chunk")
                .ok_or_else(|| Error::new("strip query_chunk missing"))?,
            attrs
                .get::<f32>("query_scale")
                .ok_or_else(|| Error::new("strip query_scale missing"))?,
        )?;
        Ok(Box::new(StripKernel {
            config: Arc::clone(&self.config),
            engine: None,
        }))
    }
}

pub(super) fn register(domain: OperatorDomain, config: Arc<Config>) -> Result<OperatorDomain> {
    domain.add(StripOperator { config })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(batch: i64) -> Vec<ValueType> {
        vec![
            tensor_type(&[batch, 4, 288, 64]),
            tensor_type(&[batch, 4, 64, 288]),
            tensor_type(&[batch, 4, 288, 64]),
        ]
    }

    #[test]
    fn actual_batches_keep_original_axes_full_scan_and_exact_attributes() {
        for batch in [8, 16] {
            assert_eq!(input_types(&values(batch)).unwrap(), Some(batch));
            assert_eq!(output_shape(batch), [3, batch, 4, 128, 64]);
            assert_eq!(
                output_shape(batch).iter().product::<i64>(),
                batch * 4 * 384 * 64
            );
        }
        attributes(128, f16::from_bits(0x35a8).to_f32()).unwrap();
        for (chunk, scale) in [(64, SCALE), (128, BODY_SCALE), (128, f32::NAN)] {
            assert!(attributes(chunk, scale).is_err());
        }
    }

    #[test]
    fn wrong_batches_geometry_rank_dtype_or_count_fail_closed() {
        for batch in [0, 1, 2, 7, 9, 15, 17, i64::MAX] {
            assert!(input_types(&values(batch)).is_err());
        }
        for (index, shape) in [
            (0, vec![16, 4, 289, 64]),
            (1, vec![8, 4, 64, 288]),
            (2, vec![16, 8, 288, 64]),
            (0, vec![16, 4, 288, 64, 1]),
        ] {
            let mut wrong = values(16);
            wrong[index] = tensor_type(&shape);
            assert!(input_types(&wrong).is_err());
        }
        let mut wrong = values(16);
        if let ValueType::Tensor { ty, .. } = &mut wrong[0] {
            *ty = TensorElementType::Float32;
        }
        assert!(input_types(&wrong).is_err());
        assert!(input_types(&values(16)[..2]).is_err());
    }

    #[test]
    fn unknown_batch_uses_unique_legal_symbol_and_runtime_contract_stays_fixed() {
        assert_eq!(input_types(&values(-1)).unwrap(), None);
        let ValueType::Tensor {
            shape,
            dimension_symbols,
            ..
        } = output_type(-1)
        else {
            panic!("tensor");
        };
        assert_eq!(&*shape, &[3, -1, 4, 128, 64]);
        assert_eq!(dimension_symbols[1], "strip_scan_batch");
        for axis in [0, 2, 3, 4] {
            assert!(dimension_symbols[axis].is_empty());
        }
    }

    #[test]
    fn batch_changes_reuse_one_weak_engine_and_last_kernel_releases_it() {
        let mut cache = EngineCache::<()>::new();
        let first = cache.get_or_create(geometry(), || Ok(())).unwrap();
        for batch in [16, 16, 8, 16] {
            input_types(&values(batch)).unwrap();
            let next = cache
                .get_or_create(geometry(), || panic!("recompiled on batch change"))
                .unwrap();
            assert!(Arc::ptr_eq(&first, &next));
        }
        let weak = Arc::downgrade(&first);
        drop(first);
        assert!(weak.upgrade().is_none());
    }
}
