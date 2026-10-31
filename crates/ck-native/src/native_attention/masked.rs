//! Exact masked query-map bridge. The native executor owns bounded GPU scratch;
//! the complete scan retains the original query-tile, batch, head, row, dim order.
use super::*;

const OPERATOR: &str = "NativeMaskedAttentionMap1080";

fn geometry(queries: i64) -> Result<Geometry> {
    if !(256..=2048).contains(&queries) || queries % 256 != 0 {
        return Err(Error::new("masked attention requires N=256*T, T1..8"));
    }
    Ok(Geometry {
        heads: 8,
        queries: queries as u32,
        head_dim: 64,
        query_chunk: 128,
        scale_bits: SCALE.to_bits(),
    })
}

fn output_shape(batch: i64, geometry: Option<Geometry>) -> [i64; 5] {
    [
        geometry.map_or(-1, |g| i64::from(g.queries / 128)),
        batch,
        8,
        128,
        64,
    ]
}

fn output_type(batch: i64, geometry: Option<Geometry>) -> ValueType {
    let mut value = tensor_type(&output_shape(batch, geometry));
    if let ValueType::Tensor {
        shape,
        dimension_symbols,
        ..
    } = &mut value
    {
        *dimension_symbols =
            SymbolicDimensions::new(shape.iter().enumerate().map(|(axis, &size)| {
                if size < 0 {
                    match axis {
                        0 => "masked_scan_query_tiles".to_owned(),
                        1 => "masked_scan_batch".to_owned(),
                        _ => unreachable!("only tile and batch axes can be symbolic"),
                    }
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
            "masked attention requires chunk128 and exact scale0.353515625",
        ));
    }
    Ok(())
}

fn shapes(inputs: &[&[i64]]) -> Result<(Option<i64>, Option<Geometry>)> {
    if inputs.len() != 4 {
        return Err(Error::new(
            "masked attention requires Q/K/V and INT64 region IDs",
        ));
    }
    let mut known_batch = None;
    let mut known_queries = None;
    for (index, shape) in inputs.iter().enumerate() {
        let expected: &[i64] = match index {
            1 => &[-1, 8, 64, -1],
            3 => &[-1, -1],
            _ => &[-1, 8, -1, 64],
        };
        if shape.len() != expected.len()
            || shape
                .iter()
                .zip(expected)
                .any(|(&actual, &required)| actual >= 0 && required >= 0 && actual != required)
        {
            return Err(Error::new(
                "masked attention requires Q/V[B,8,N,64], K[B,8,64,N], IDs[B,N]",
            ));
        }
        let batch = shape[0];
        if batch >= 0 {
            if !(1..=16).contains(&batch) || known_batch.is_some_and(|previous| previous != batch) {
                return Err(Error::new(
                    "masked attention batch must agree and be in1..16",
                ));
            }
            known_batch = Some(batch);
        }
        let queries = shape[match index {
            1 => 3,
            3 => 1,
            _ => 2,
        }];
        if queries >= 0 {
            let candidate = geometry(queries)?;
            if known_queries.is_some_and(|previous| previous != candidate) {
                return Err(Error::new("masked attention Q/K/V/ID token counts differ"));
            }
            known_queries = Some(candidate);
        }
    }
    Ok((known_batch, known_queries))
}

fn input_types(inputs: &[ValueType]) -> Result<(Option<i64>, Option<Geometry>)> {
    if inputs.len() != 4 {
        return Err(Error::new("masked attention requires four inputs"));
    }
    let mut input_shapes = Vec::with_capacity(4);
    for (index, input) in inputs.iter().enumerate() {
        let expected = if index == 3 {
            TensorElementType::Int64
        } else {
            TensorElementType::Float16
        };
        match input {
            ValueType::Tensor { ty, shape, .. } if *ty == expected => input_shapes.push(&**shape),
            _ => {
                return Err(Error::new(
                    "masked attention requires FP16 Q/K/V and INT64 IDs",
                ));
            }
        }
    }
    shapes(&input_shapes)
}

type MaskedRun = unsafe extern "C" fn(
    *mut c_void,
    *const u16,
    usize,
    *const u16,
    usize,
    *const u16,
    usize,
    *const i64,
    usize,
    *mut u16,
    usize,
    *mut c_char,
    usize,
) -> i32;

pub(super) struct Engine {
    pointer: NonNull<c_void>,
    run: MaskedRun,
    destroy: Destroy,
    _library: Library,
}

// Calls and Drop are serialized by Arc<Mutex<_>>; the synchronous C ABI owns
// all GPU storage and never retains a borrowed Rust tensor after returning.
unsafe impl Send for Engine {}

impl Engine {
    fn new(config: &Config, geometry: Geometry) -> Result<Self> {
        let library = unsafe { Library::new(&config.helper_dll) }
            .map_err(|error| Error::new(format!("masked attention helper load failed: {error}")))?;
        let version: AbiVersion = symbol(&library, b"brp_masked_map_abi_version\0")?;
        if unsafe { version() } != 1 {
            return Err(Error::new("masked attention helper ABI must be1"));
        }
        let create: Create = symbol(&library, b"brp_masked_map_create\0")?;
        let run: MaskedRun = symbol(&library, b"brp_masked_map_run\0")?;
        let destroy: Destroy = symbol(&library, b"brp_masked_map_destroy\0")?;
        let path = config
            .directml_dll
            .to_str()
            .ok_or_else(|| Error::new("masked attention requires a Unicode DirectML path"))?;
        let path = CString::new(path)
            .map_err(|_| Error::new("masked attention DirectML path contains a zero byte"))?;
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
            return Err(ffi_error(&error, "masked engine creation failed"));
        }
        Ok(Self {
            pointer: NonNull::new(pointer)
                .ok_or_else(|| Error::new("masked create returned null"))?,
            run,
            destroy,
            _library: library,
        })
    }

    fn run(
        &mut self,
        q: &[f16],
        k: &[f16],
        v: &[f16],
        ids: &[i64],
        output: &mut [f16],
    ) -> Result<()> {
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
                ids.as_ptr(),
                ids.len(),
                output.as_mut_ptr().cast(),
                output.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(ffi_error(&error, "masked execution failed"))
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.pointer.as_ptr()) };
    }
}

struct MaskedKernel {
    config: Arc<Config>,
    engine: Option<(Geometry, Arc<Mutex<Engine>>)>,
}

impl Kernel for MaskedKernel {
    fn compute(&mut self, context: &KernelContext) -> Result<()> {
        let q = context
            .input(0)?
            .ok_or_else(|| Error::new("masked attention missing Q"))?;
        let k = context
            .input(1)?
            .ok_or_else(|| Error::new("masked attention missing K"))?;
        let v = context
            .input(2)?
            .ok_or_else(|| Error::new("masked attention missing V"))?;
        let ids = context
            .input(3)?
            .ok_or_else(|| Error::new("masked attention missing IDs"))?;
        let (qs, q_data) = q.try_extract_tensor::<f16>()?;
        let (ks, k_data) = k.try_extract_tensor::<f16>()?;
        let (vs, v_data) = v.try_extract_tensor::<f16>()?;
        let (rs, id_data) = ids.try_extract_tensor::<i64>()?;
        let actual = [&**qs, &**ks, &**vs, &**rs];
        if actual
            .iter()
            .any(|shape| shape.iter().any(|&dimension| dimension < 0))
        {
            return Err(Error::new("masked runtime dimensions must be concrete"));
        }
        let (batch, geometry) = shapes(&actual)?;
        let batch = batch.ok_or_else(|| Error::new("masked runtime batch missing"))?;
        let geometry = geometry.ok_or_else(|| Error::new("masked runtime token count missing"))?;
        if self
            .engine
            .as_ref()
            .is_none_or(|(previous, _)| *previous != geometry)
        {
            self.engine = None;
            let engine = self
                .config
                .cached_masked
                .lock()
                .map_err(|_| Error::new("masked attention cache was poisoned"))?
                .get_or_create(geometry, || Engine::new(&self.config, geometry))?;
            self.engine = Some((geometry, engine));
        }
        let mut output = context
            .output(0, output_shape(batch, Some(geometry)))?
            .ok_or_else(|| Error::new("masked attention output missing"))?;
        self.engine
            .as_ref()
            .ok_or_else(|| Error::new("masked engine missing"))?
            .1
            .lock()
            .map_err(|_| Error::new("masked attention engine mutex was poisoned"))?
            .run(
                q_data,
                k_data,
                v_data,
                id_data,
                output.try_extract_tensor_mut::<f16>()?.1,
            )
    }
}

struct MaskedOperator {
    config: Arc<Config>,
}

impl Operator for MaskedOperator {
    fn name(&self) -> &str {
        OPERATOR
    }
    fn execution_provider_type(&self) -> Option<&str> {
        Some("CPUExecutionProvider")
    }
    fn inputs(&self) -> Vec<OperatorInput> {
        [
            TensorElementType::Float16,
            TensorElementType::Float16,
            TensorElementType::Float16,
            TensorElementType::Int64,
        ]
        .into_iter()
        .map(|ty| OperatorInput::required(ty).memory_type(MemoryType::CPUInput))
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
        let (batch, geometry) = input_types(&context.inputs())?;
        context.set_output(0, &output_type(batch.unwrap_or(-1), geometry))
    }
    fn create_kernel(&self, attrs: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        attributes(
            attrs
                .get::<i64>("query_chunk")
                .ok_or_else(|| Error::new("masked query_chunk missing"))?,
            attrs
                .get::<f32>("query_scale")
                .ok_or_else(|| Error::new("masked query_scale missing"))?,
        )?;
        Ok(Box::new(MaskedKernel {
            config: Arc::clone(&self.config),
            engine: None,
        }))
    }
}

pub(super) fn register(domain: OperatorDomain, config: Arc<Config>) -> Result<OperatorDomain> {
    domain.add(MaskedOperator { config })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(batch: i64, queries: i64) -> Vec<ValueType> {
        let mut ids = tensor_type(&[batch, queries]);
        if let ValueType::Tensor { ty, .. } = &mut ids {
            *ty = TensorElementType::Int64;
        }
        vec![
            tensor_type(&[batch, 8, queries, 64]),
            tensor_type(&[batch, 8, 64, queries]),
            tensor_type(&[batch, 8, queries, 64]),
            ids,
        ]
    }

    #[test]
    fn every_temporal_shape_and_batch_preserves_original_scan_order_and_count() {
        for frames in 1..=8 {
            for batch in 1..=16 {
                let (b, g) = input_types(&values(batch, 256 * frames)).unwrap();
                let output = output_shape(b.unwrap(), g);
                assert_eq!(output, [2 * frames, batch, 8, 128, 64]);
                assert_eq!(
                    output.iter().product::<i64>(),
                    batch * 8 * 256 * frames * 64
                );
            }
        }
        attributes(128, f16::from_bits(0x35a8).to_f32()).unwrap();
        for (chunk, scale) in [(64, SCALE), (128, BODY_SCALE), (128, f32::NAN)] {
            assert!(attributes(chunk, scale).is_err());
        }
    }

    #[test]
    fn invalid_geometry_types_and_inconsistent_inputs_fail_closed() {
        for batch in [0, 17, i64::MAX] {
            assert!(input_types(&values(batch, 2048)).is_err());
        }
        for n in [0, 255, 257, 2049, 2304, i64::MAX] {
            assert!(input_types(&values(16, n)).is_err());
        }
        let mut wrong = values(16, 2048);
        wrong[3] = tensor_type(&[16, 2048]);
        assert!(input_types(&wrong).is_err());
        for (index, shape) in [
            (0, vec![16, 8, 1024, 64]),
            (1, vec![8, 8, 64, 2048]),
            (2, vec![16, 16, 2048, 64]),
            (3, vec![16, 2048, 1]),
        ] {
            let mut wrong = values(16, 2048);
            let mut value = tensor_type(&shape);
            if index == 3
                && let ValueType::Tensor { ty, .. } = &mut value
            {
                *ty = TensorElementType::Int64;
            }
            wrong[index] = value;
            assert!(input_types(&wrong).is_err());
        }
    }

    #[test]
    fn symbolic_batch_and_tokens_stay_symbolic_until_compute() {
        let (b, g) = input_types(&values(-1, -1)).unwrap();
        assert!(b.is_none() && g.is_none());
        let ValueType::Tensor {
            shape,
            dimension_symbols,
            ..
        } = output_type(-1, None)
        else {
            panic!("tensor");
        };
        assert_eq!(&*shape, &[-1, -1, 8, 128, 64]);
        assert!(!dimension_symbols[0].is_empty() && !dimension_symbols[1].is_empty());
        assert_ne!(dimension_symbols[0], dimension_symbols[1]);
        assert_eq!(dimension_symbols[0], "masked_scan_query_tiles");
        assert_eq!(dimension_symbols[1], "masked_scan_batch");
        let ValueType::Tensor {
            dimension_symbols: body_symbols,
            ..
        } = tensor_type(&[-1, 8, 128, 64])
        else {
            panic!("tensor");
        };
        assert_ne!(dimension_symbols[0], body_symbols[0]);
        let mut partial = values(-1, -1);
        partial[0] = tensor_type(&[8, 8, 1024, 64]);
        let (b, g) = input_types(&partial).unwrap();
        assert_eq!(b, Some(8));
        assert_eq!(g.unwrap().queries, 1024);
    }

    #[test]
    fn eight_masked_kernels_share_an_engine_across_batch_changes_and_release_it() {
        let mut cache = EngineCache::<()>::new();
        let g = geometry(2048).unwrap();
        let mut kernels = Vec::new();
        for _ in 0..8 {
            kernels.push(cache.get_or_create(g, || Ok(())).unwrap());
        }
        assert!(
            kernels
                .iter()
                .all(|engine| Arc::ptr_eq(engine, &kernels[0]))
        );
        let batch8 = input_types(&values(8, 2048)).unwrap().1.unwrap();
        assert!(Arc::ptr_eq(
            &cache
                .get_or_create(batch8, || panic!("rebuild on batch change"))
                .unwrap(),
            &kernels[0]
        ));
        let weak = Arc::downgrade(&kernels[0]);
        kernels.clear();
        assert!(weak.upgrade().is_none());
        assert_eq!(cache.entries.len(), 1);
        cache
            .get_or_create(geometry(1024).unwrap(), || Ok(()))
            .unwrap();
        assert_eq!(cache.entries.len(), 1);
    }
}
