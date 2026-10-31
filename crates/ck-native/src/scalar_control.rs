//! CPU-only metadata for the bounded Stage1 query-attention experiment.
//!
//! This operator changes no neural arithmetic. It keeps query-tile Slice and
//! Pad/If controls on the CPU instead of dispatching tiny integer operations to
//! an accelerator and copying their results back for every tile. A candidate
//! graph must also carry `total_queries` through the Loop in CPU memory.
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

const DOMAIN: &str = "com.brp_chroma.control";
const OPERATOR: &str = "QueryTileControl";

#[derive(Debug, PartialEq, Eq)]
struct TileControl {
    starts: [i64; 1],
    ends: [i64; 1],
    pads: [i64; 8],
    needs_padding: bool,
}

fn checked_chunk(chunk: i64) -> Result<i64> {
    if chunk <= 0 {
        return Err(Error::new("QueryTileControl chunk must be positive"));
    }
    Ok(chunk)
}

fn tile_control(iteration: i64, total_queries: i64, chunk: i64) -> Result<TileControl> {
    let chunk = checked_chunk(chunk)?;
    if iteration < 0 || total_queries <= 0 {
        return Err(Error::new(
            "QueryTileControl requires iteration >= 0 and total_queries > 0",
        ));
    }
    let start = iteration
        .checked_mul(chunk)
        .ok_or_else(|| Error::new("QueryTileControl start overflow"))?;
    if start >= total_queries {
        return Err(Error::new(
            "QueryTileControl iteration starts outside total_queries",
        ));
    }
    let end = start
        .checked_add(chunk)
        .ok_or_else(|| Error::new("QueryTileControl end overflow"))?;
    let extra = end - end.min(total_queries);
    Ok(TileControl {
        starts: [start],
        // ONNX Slice clamps this endpoint to the available query rows. The
        // unclamped endpoint is also what determines padding of the last tile.
        ends: [end],
        pads: [0, 0, 0, 0, 0, 0, extra, 0],
        needs_padding: extra > 0,
    })
}

fn tensor_type(ty: TensorElementType, shape: &[i64]) -> ValueType {
    ValueType::Tensor {
        ty,
        shape: Shape::new(shape.iter().copied()),
        dimension_symbols: SymbolicDimensions::new(shape.iter().map(|_| String::new())),
    }
}

fn output_types(inputs: &[ValueType]) -> Result<[ValueType; 4]> {
    if inputs.len() != 2
        || inputs.iter().any(|input| {
            !matches!(input, ValueType::Tensor { ty: TensorElementType::Int64, shape, .. } if shape.is_empty())
        })
    {
        return Err(Error::new(
            "QueryTileControl requires two scalar INT64 inputs",
        ));
    }
    Ok([
        tensor_type(TensorElementType::Int64, &[1]),
        tensor_type(TensorElementType::Int64, &[1]),
        tensor_type(TensorElementType::Int64, &[8]),
        tensor_type(TensorElementType::Bool, &[]),
    ])
}

fn write_i64(context: &KernelContext, index: usize, data: &[i64]) -> Result<()> {
    let mut output = context
        .output(index, [data.len()])?
        .ok_or_else(|| Error::new("QueryTileControl missing INT64 output"))?;
    // Extraction validates CPU accessibility before exposing the host slice.
    // The kernel is CPU-only, so a device allocation is an error, never a raw
    // device pointer dereference or an implicit fallback.
    output
        .try_extract_tensor_mut::<i64>()?
        .1
        .copy_from_slice(data);
    Ok(())
}

struct QueryTileKernel {
    chunk: i64,
}

impl Kernel for QueryTileKernel {
    fn compute(&mut self, context: &KernelContext) -> Result<()> {
        let iteration = context
            .input(0)?
            .ok_or_else(|| Error::new("QueryTileControl missing iteration"))?
            .try_extract_scalar::<i64>()?;
        let total_queries = context
            .input(1)?
            .ok_or_else(|| Error::new("QueryTileControl missing total_queries"))?
            .try_extract_scalar::<i64>()?;
        let control = tile_control(iteration, total_queries, self.chunk)?;
        write_i64(context, 0, &control.starts)?;
        write_i64(context, 1, &control.ends)?;
        write_i64(context, 2, &control.pads)?;
        let mut output = context
            .output(3, Vec::<i64>::new())?
            .ok_or_else(|| Error::new("QueryTileControl missing BOOL output"))?;
        output.try_extract_tensor_mut::<bool>()?.1[0] = control.needs_padding;
        Ok(())
    }
}

struct QueryTileControl;

impl Operator for QueryTileControl {
    fn name(&self) -> &str {
        OPERATOR
    }

    fn execution_provider_type(&self) -> Option<&str> {
        Some("CPUExecutionProvider")
    }

    fn inputs(&self) -> Vec<OperatorInput> {
        vec![
            OperatorInput::required(TensorElementType::Int64).memory_type(MemoryType::CPUInput),
            OperatorInput::required(TensorElementType::Int64).memory_type(MemoryType::CPUInput),
        ]
    }

    fn outputs(&self) -> Vec<OperatorOutput> {
        vec![
            OperatorOutput::required(TensorElementType::Int64),
            OperatorOutput::required(TensorElementType::Int64),
            OperatorOutput::required(TensorElementType::Int64),
            OperatorOutput::required(TensorElementType::Bool),
        ]
    }

    fn max_version(&self) -> i32 {
        1
    }

    fn infer_shape(&self, context: &mut ShapeInferenceContext) -> Result<()> {
        checked_chunk(context.attr::<i64>("chunk")?)?;
        for (index, output) in output_types(&context.inputs())?.iter().enumerate() {
            context.set_output(index, output)?;
        }
        Ok(())
    }

    fn create_kernel(&self, attributes: &KernelAttributes) -> Result<Box<dyn Kernel>> {
        let chunk = attributes
            .get::<i64>("chunk")
            .ok_or_else(|| Error::new("QueryTileControl requires INT64 chunk attribute"))?;
        Ok(Box::new(QueryTileKernel {
            chunk: checked_chunk(chunk)?,
        }))
    }
}

/// Registers only the CPU metadata operator. Unmodified standard ONNX models
/// do not use it, and registering the domain does not create a GPU device.
pub fn domain() -> Result<OperatorDomain> {
    OperatorDomain::new(DOMAIN)?.add(QueryTileControl)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 348-byte, weights-free ONNX graph containing only QueryTileControl,
    // two scalar inputs and the four fixed-shape outputs. The .bin suffix
    // distinguishes this checked-in test fixture from excluded model weights.
    #[cfg(windows)]
    const ABI_FIXTURE: &[u8] = include_bytes!("../fixtures/query_tile_control.onnx.bin");
    #[cfg(windows)]
    const ATTENTION_ORIGINAL_FIXTURE: &[u8] =
        include_bytes!("../fixtures/query_tile_attention_original.onnx.bin");
    #[cfg(windows)]
    const ATTENTION_CONTROL_FIXTURE: &[u8] =
        include_bytes!("../fixtures/query_tile_attention_cpu_control.onnx.bin");

    #[cfg(windows)]
    fn cpu_fixture_session(fixture: &[u8]) -> Result<ort::session::Session> {
        use ort::{
            execution_providers::CPUExecutionProvider,
            session::{Session, builder::GraphOptimizationLevel},
        };
        let runtime =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../app/onnxruntime.dll");
        if !runtime.is_file() {
            return Err(Error::new(format!(
                "CPU ABI test requires the existing runtime at {}; no fallback or download is allowed",
                runtime.display()
            )));
        }
        ort::init_from(runtime.to_string_lossy())
            .with_name("BRP CHROMA CPU scalar-control ABI test")
            .with_execution_providers([CPUExecutionProvider::default().build().error_on_failure()])
            .commit()
            .map_err(|error| {
                Error::new(format!(
                    "CPU ABI test could not initialize {}: {error}",
                    runtime.display()
                ))
            })?;
        // Only the CPU provider is registered. Loading the DirectML-capable
        // runtime DLL does not create a DirectML session or a GPU device.
        Session::builder()?
            .with_intra_threads(1)?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .with_execution_providers([CPUExecutionProvider::default().build().error_on_failure()])?
            .with_operators(domain()?)?
            .commit_from_memory(fixture)
    }

    #[test]
    fn query_controls_match_independent_integer_reference_at_all_small_boundaries() {
        for total in 1_i64..=521 {
            for chunk in 1_i64..=137 {
                let tile_count = 1 + (total - 1) / chunk;
                for iteration in 0..tile_count {
                    let control = tile_control(iteration, total, chunk).unwrap();
                    let start = i128::from(iteration) * i128::from(chunk);
                    let end = start + i128::from(chunk);
                    let available = (i128::from(total) - start).min(i128::from(chunk));
                    let extra = i128::from(chunk) - available;
                    assert_eq!(i128::from(control.starts[0]), start);
                    assert_eq!(i128::from(control.ends[0]), end);
                    assert_eq!(i128::from(control.pads[6]), extra);
                    assert_eq!(control.needs_padding, extra != 0);
                    assert!(
                        control
                            .pads
                            .iter()
                            .enumerate()
                            .all(|(axis, &padding)| { axis == 6 || padding == 0 })
                    );
                }
                assert!(tile_control(tile_count, total, chunk).is_err());
            }
        }
    }

    #[test]
    fn query_controls_preserve_partial_and_full_128_row_tiles() {
        assert_eq!(
            tile_control(1, 129, 128).unwrap(),
            TileControl {
                starts: [128],
                ends: [256],
                pads: [0, 0, 0, 0, 0, 0, 127, 0],
                needs_padding: true,
            }
        );
        for total in [128, 256, 384, 4_096, 16_384] {
            let last = tile_control(total / 128 - 1, total, 128).unwrap();
            assert_eq!(last.ends, [total]);
            assert_eq!(last.pads, [0; 8]);
            assert!(!last.needs_padding);
        }
    }

    #[test]
    fn query_controls_reject_invalid_domain_and_integer_overflow() {
        for (iteration, total, chunk) in [
            (-1, 128, 128),
            (i64::MIN, 128, 128),
            (0, 0, 128),
            (0, -1, 128),
            (0, 128, 0),
            (0, 128, -1),
            (0, 128, i64::MIN),
            (1, 128, 128),
            // start multiplication overflows despite a positive domain.
            (i64::MAX, i64::MAX, 2),
            // start fits and is in range; only the end addition overflows.
            (i64::MAX / 128, i64::MAX, 128),
        ] {
            assert!(tile_control(iteration, total, chunk).is_err());
        }
        // No floating-point conversion: integers beyond 2^53 stay exact.
        for total in [(1_i64 << 53) + 1, i64::MAX] {
            let control = tile_control(total - 1, total, 1).unwrap();
            assert_eq!(control.starts, [total - 1]);
            assert_eq!(control.ends, [total]);
            assert_eq!(control.pads, [0; 8]);
            assert!(!control.needs_padding);
        }
        assert_eq!(tile_control(0, 1, i64::MAX).unwrap().pads[6], i64::MAX - 1);
    }

    #[test]
    fn query_control_schema_is_cpu_only_with_exact_output_types_and_shapes() {
        let operator = QueryTileControl;
        assert_eq!(operator.name(), "QueryTileControl");
        assert_eq!(
            operator.execution_provider_type(),
            Some("CPUExecutionProvider")
        );
        assert_eq!(operator.min_version(), 1);
        assert_eq!(operator.max_version(), 1);
        assert_eq!(operator.inputs().len(), 2);
        assert_eq!(operator.outputs().len(), 4);
        let inputs = [
            tensor_type(TensorElementType::Int64, &[]),
            tensor_type(TensorElementType::Int64, &[]),
        ];
        assert_eq!(
            output_types(&inputs).unwrap(),
            [
                tensor_type(TensorElementType::Int64, &[1]),
                tensor_type(TensorElementType::Int64, &[1]),
                tensor_type(TensorElementType::Int64, &[8]),
                tensor_type(TensorElementType::Bool, &[]),
            ]
        );
        assert!(output_types(&[]).is_err());
        assert!(output_types(&[tensor_type(TensorElementType::Int64, &[])]).is_err());
        for invalid in [
            tensor_type(TensorElementType::Int64, &[1]),
            tensor_type(TensorElementType::Float32, &[]),
            tensor_type(TensorElementType::Bool, &[]),
        ] {
            assert!(output_types(&[invalid.clone(), inputs[1].clone()]).is_err());
            assert!(output_types(&[inputs[0].clone(), invalid]).is_err());
        }
    }

    #[test]
    #[cfg(windows)]
    #[ignore = "Explicit CPU-only custom-op ABI test; requires existing app/onnxruntime.dll, no models/GPU"]
    fn query_control_ort_cpu_abi_preserves_shapes_values_and_errors() -> Result<()> {
        use ort::{
            memory::AllocationDevice,
            value::{Tensor, TensorValueType},
        };
        let mut session = cpu_fixture_session(ABI_FIXTURE)?;
        let inputs = [
            tensor_type(TensorElementType::Int64, &[]),
            tensor_type(TensorElementType::Int64, &[]),
        ];
        for (output, expected) in session.outputs.iter().zip(output_types(&inputs)?) {
            assert_eq!(output.output_type, expected);
        }
        for total in [1_i64, 127, 128, 129, 255, 256, 257, 384, 1_025] {
            for iteration in 0..(1 + (total - 1) / 128) {
                let expected = tile_control(iteration, total, 128)?;
                let iteration = Tensor::from_array(((), vec![iteration]))?;
                let total = Tensor::from_array(((), vec![total]))?;
                let outputs = session.run(ort::inputs![iteration, total])?;
                for (index, expected) in [
                    expected.starts.as_slice(),
                    expected.ends.as_slice(),
                    expected.pads.as_slice(),
                ]
                .into_iter()
                .enumerate()
                {
                    let tensor = outputs[index].downcast_ref::<TensorValueType<i64>>()?;
                    assert_eq!(
                        tensor.memory_info().allocation_device(),
                        AllocationDevice::CPU
                    );
                    assert!(tensor.memory_info().is_cpu_accessible());
                    let (shape, values) = tensor.try_extract_tensor::<i64>()?;
                    assert_eq!(&shape[..], &[expected.len() as i64]);
                    assert_eq!(values, expected);
                }
                let condition = outputs[3].downcast_ref::<TensorValueType<bool>>()?;
                assert_eq!(
                    condition.memory_info().allocation_device(),
                    AllocationDevice::CPU
                );
                assert!(condition.memory_info().is_cpu_accessible());
                assert!(condition.shape().is_empty());
                assert_eq!(
                    condition.try_extract_scalar::<bool>()?,
                    expected.needs_padding
                );
            }
        }
        for (iteration, total) in [
            (-1_i64, 128_i64),
            (0, 0),
            (0, -1),
            (1, 128),
            (i64::MAX, i64::MAX),
            (i64::MAX / 128, i64::MAX),
        ] {
            let iteration = Tensor::from_array(((), vec![iteration]))?;
            let total = Tensor::from_array(((), vec![total]))?;
            let error = session
                .run(ort::inputs![iteration, total])
                .expect_err("invalid query controls must fail inside the CPU kernel");
            assert!(error.to_string().contains("QueryTileControl"));
        }
        Ok(())
    }

    #[test]
    #[cfg(windows)]
    #[ignore = "Explicit CPU-only native custom-op Loop ABI test; requires existing app/onnxruntime.dll, no models/GPU"]
    fn query_control_ort_cpu_loop_matches_complete_original_attention_bytes() -> Result<()> {
        use ort::{
            memory::AllocationDevice,
            value::{Tensor, TensorValueType},
        };
        let mut original = cpu_fixture_session(ATTENTION_ORIGINAL_FIXTURE)?;
        let mut candidate = cpu_fixture_session(ATTENTION_CONTROL_FIXTURE)?;
        fn values(count: usize, salt: usize) -> Vec<f32> {
            (0..count)
                .map(|index| ((index * 37 + salt) % 997) as f32 * 0.0002 - 0.1)
                .collect()
        }
        let mut cases = 0;
        for (batch, heads) in [(1_usize, 1_usize), (1, 2), (2, 2)] {
            for queries in [1_usize, 7, 127, 128, 129, 255, 256, 257] {
                for keys in [1_usize, 13] {
                    let q = Tensor::from_array((
                        [batch, heads, queries, 4],
                        values(batch * heads * queries * 4, 17),
                    ))?;
                    let k = Tensor::from_array((
                        [batch, heads, 4, keys],
                        values(batch * heads * 4 * keys, 43),
                    ))?;
                    let v = Tensor::from_array((
                        [batch, heads, keys, 4],
                        values(batch * heads * keys * 4, 71),
                    ))?;
                    let old = original.run(ort::inputs![&q, &k, &v])?;
                    let new = candidate.run(ort::inputs![&q, &k, &v])?;
                    let tensor = new[0].downcast_ref::<TensorValueType<f32>>()?;
                    assert_eq!(
                        tensor.memory_info().allocation_device(),
                        AllocationDevice::CPU
                    );
                    assert!(tensor.memory_info().is_cpu_accessible());
                    let (old_shape, expected) = old[0].try_extract_tensor::<f32>()?;
                    let (new_shape, actual) = tensor.try_extract_tensor::<f32>()?;
                    let expected_shape = [batch as i64, heads as i64, queries as i64, 4];
                    assert_eq!(&old_shape[..], &expected_shape);
                    assert_eq!(&new_shape[..], &expected_shape);
                    assert_eq!(actual.len(), expected.len());
                    assert!(
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(left, right)| { left.to_bits() == right.to_bits() })
                    );
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 48);
        println!(
            "CPU_QUERY_CONTROL_LOOP_ABI {{\"cases\":{cases},\"all_output_bytes_equal\":true,\"provider\":\"CPUExecutionProvider\",\"gpu_used\":false}}"
        );
        Ok(())
    }
}
