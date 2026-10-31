//! Minimal ONNX protobuf messages (field numbers from onnx.proto3) for reading and
//! lightly rewriting the loop-free stage graphs. No protoc or build script: the
//! messages are declared with `prost` derive. Only fields these graphs use are
//! modelled (subgraph attributes included); the graphs contain no functions or sparse
//! tensors. Unmodelled fields are dropped on save; ONNX Runtime re-validates the result.

use anyhow::{Context, Result, bail, ensure};
use prost::Message;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(int64, tag = "1")]
    pub ir_version: i64,
    #[prost(string, tag = "2")]
    pub producer_name: String,
    #[prost(string, tag = "3")]
    pub producer_version: String,
    #[prost(string, tag = "4")]
    pub domain: String,
    #[prost(int64, tag = "5")]
    pub model_version: i64,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
    #[prost(message, repeated, tag = "8")]
    pub opset_import: Vec<OperatorSetIdProto>,
    #[prost(message, repeated, tag = "14")]
    pub metadata_props: Vec<StringStringEntryProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct OperatorSetIdProto {
    #[prost(string, tag = "1")]
    pub domain: String,
    #[prost(int64, tag = "2")]
    pub version: i64,
}

#[derive(Clone, PartialEq, Message)]
pub struct StringStringEntryProto {
    #[prost(string, tag = "1")]
    pub key: String,
    #[prost(string, tag = "2")]
    pub value: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "1")]
    pub node: Vec<NodeProto>,
    #[prost(string, tag = "2")]
    pub name: String,
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
    #[prost(string, tag = "10")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "11")]
    pub input: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "12")]
    pub output: Vec<ValueInfoProto>,
    #[prost(message, repeated, tag = "13")]
    pub value_info: Vec<ValueInfoProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct NodeProto {
    #[prost(string, repeated, tag = "1")]
    pub input: Vec<String>,
    #[prost(string, repeated, tag = "2")]
    pub output: Vec<String>,
    #[prost(string, tag = "3")]
    pub name: String,
    #[prost(string, tag = "4")]
    pub op_type: String,
    #[prost(message, repeated, tag = "5")]
    pub attribute: Vec<AttributeProto>,
    #[prost(string, tag = "6")]
    pub doc_string: String,
    #[prost(string, tag = "7")]
    pub domain: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct AttributeProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(float, tag = "2")]
    pub f: f32,
    #[prost(int64, tag = "3")]
    pub i: i64,
    #[prost(bytes = "vec", tag = "4")]
    pub s: Vec<u8>,
    #[prost(message, optional, tag = "5")]
    pub t: Option<TensorProto>,
    /// If/Loop bodies: the symbolic graphs still carry a few before ORT folds them.
    #[prost(message, optional, boxed, tag = "6")]
    pub g: Option<Box<GraphProto>>,
    #[prost(float, repeated, tag = "7")]
    pub floats: Vec<f32>,
    #[prost(int64, repeated, tag = "8")]
    pub ints: Vec<i64>,
    #[prost(bytes = "vec", repeated, tag = "9")]
    pub strings: Vec<Vec<u8>>,
    #[prost(message, repeated, tag = "10")]
    pub tensors: Vec<TensorProto>,
    #[prost(message, repeated, tag = "11")]
    pub graphs: Vec<GraphProto>,
    #[prost(string, tag = "13")]
    pub doc_string: String,
    #[prost(int32, tag = "20")]
    pub r#type: i32,
    #[prost(string, tag = "21")]
    pub ref_attr_name: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(int32, repeated, tag = "5")]
    pub int32_data: Vec<i32>,
    #[prost(int64, repeated, tag = "7")]
    pub int64_data: Vec<i64>,
    #[prost(string, tag = "8")]
    pub name: String,
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
    #[prost(double, repeated, tag = "10")]
    pub double_data: Vec<f64>,
    #[prost(string, tag = "12")]
    pub doc_string: String,
    #[prost(message, repeated, tag = "13")]
    pub external_data: Vec<StringStringEntryProto>,
    #[prost(int32, tag = "14")]
    pub data_location: i32,
}

#[derive(Clone, PartialEq, Message)]
pub struct ValueInfoProto {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(message, optional, tag = "2")]
    pub r#type: Option<TypeProto>,
    #[prost(string, tag = "3")]
    pub doc_string: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct TypeProto {
    #[prost(message, optional, tag = "1")]
    pub tensor_type: Option<TypeTensor>,
    #[prost(string, tag = "6")]
    pub denotation: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct TypeTensor {
    #[prost(int32, tag = "1")]
    pub elem_type: i32,
    #[prost(message, optional, tag = "2")]
    pub shape: Option<TensorShapeProto>,
}

#[derive(Clone, PartialEq, Message)]
pub struct TensorShapeProto {
    #[prost(message, repeated, tag = "1")]
    pub dim: Vec<Dimension>,
}

#[derive(Clone, PartialEq, Message)]
pub struct Dimension {
    #[prost(int64, optional, tag = "1")]
    pub dim_value: Option<i64>,
    #[prost(string, optional, tag = "2")]
    pub dim_param: Option<String>,
    #[prost(string, tag = "3")]
    pub denotation: String,
}

pub const FLOAT: i32 = 1;
pub const INT32: i32 = 6;
pub const INT64: i32 = 7;
pub const BOOL: i32 = 9;
pub const FLOAT16: i32 = 10;
pub const DOUBLE: i32 = 11;
pub const EXTERNAL: i32 = 1;

pub fn load_model(path: &Path) -> Result<ModelProto> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ModelProto::decode(bytes.as_slice()).with_context(|| format!("decode ONNX {}", path.display()))
}

pub fn save_model(model: &ModelProto, path: &Path) -> Result<()> {
    fs::write(path, model.encode_to_vec()).with_context(|| format!("write {}", path.display()))
}

/// Host copy of one initializer's bytes, resolved from inline fields or external data.
pub struct TensorData {
    pub data_type: i32,
    pub dims: Vec<i64>,
    pub bytes: Vec<u8>,
}

impl TensorData {
    pub fn elements(&self) -> usize {
        self.dims.iter().product::<i64>().max(0) as usize
    }
    /// Values as i64 (integer, bool or float tensors).
    pub fn to_i64(&self) -> Result<Vec<i64>> {
        Ok(match self.data_type {
            INT64 => self
                .bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect(),
            INT32 => self
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c) as i64)
                .collect(),
            BOOL => self.bytes.iter().map(|&b| (b != 0) as i64).collect(),
            _ => self.to_f64()?.into_iter().map(|v| v as i64).collect(),
        })
    }
    pub fn to_f64(&self) -> Result<Vec<f64>> {
        Ok(match self.data_type {
            FLOAT => self
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c) as f64)
                .collect(),
            FLOAT16 => self
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f64())
                .collect(),
            DOUBLE => self
                .bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect(),
            INT64 | INT32 | BOOL => self.to_i64()?.into_iter().map(|v| v as f64).collect(),
            t => bail!("unsupported tensor type {t}"),
        })
    }
    pub fn to_bool(&self) -> Result<Vec<bool>> {
        ensure!(self.data_type == BOOL, "expected bool tensor");
        Ok(self.bytes.iter().map(|&b| b != 0).collect())
    }
}

pub fn element_size(data_type: i32) -> Result<usize> {
    Ok(match data_type {
        FLOAT | INT32 => 4,
        FLOAT16 => 2,
        BOOL => 1,
        INT64 | DOUBLE => 8,
        t => bail!("unsupported element type {t}"),
    })
}

/// Resolves initializer bytes; external files are opened once and cached.
pub struct ExternalData {
    dir: PathBuf,
    files: HashMap<String, Vec<u8>>,
}

impl ExternalData {
    pub fn new(model_dir: &Path) -> Self {
        Self {
            dir: model_dir.to_path_buf(),
            files: HashMap::new(),
        }
    }

    pub fn tensor(&mut self, t: &TensorProto) -> Result<TensorData> {
        let n = t.dims.iter().product::<i64>().max(0) as usize;
        let bytes = if t.data_location == EXTERNAL {
            let entry = |key: &str| {
                t.external_data
                    .iter()
                    .find(|e| e.key == key)
                    .map(|e| e.value.clone())
            };
            let location = entry("location").context("external tensor without location")?;
            ensure!(
                !location.contains(".."),
                "external data path must stay in the model folder"
            );
            let offset: usize = entry("offset").map(|v| v.parse()).transpose()?.unwrap_or(0);
            let length: usize = match entry("length") {
                Some(v) => v.parse()?,
                None => n * element_size(t.data_type)?,
            };
            if !self.files.contains_key(&location) {
                let data = fs::read(self.dir.join(&location))
                    .with_context(|| format!("read external data {location}"))?;
                self.files.insert(location.clone(), data);
            }
            let file = &self.files[&location];
            ensure!(
                offset + length <= file.len(),
                "external tensor {} outside its file",
                t.name
            );
            file[offset..offset + length].to_vec()
        } else if !t.raw_data.is_empty() {
            t.raw_data.clone()
        } else {
            match t.data_type {
                FLOAT => t.float_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                INT64 => t.int64_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                INT32 => t.int32_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                // FLOAT16 / BOOL values live in int32_data per the ONNX spec.
                FLOAT16 => t
                    .int32_data
                    .iter()
                    .flat_map(|v| (*v as u16).to_le_bytes())
                    .collect(),
                BOOL => t.int32_data.iter().map(|v| (*v != 0) as u8).collect(),
                DOUBLE => t.double_data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ty => bail!("tensor {} has no data (type {ty})", t.name),
            }
        };
        ensure!(
            bytes.len() == n * element_size(t.data_type)?,
            "tensor {} byte size mismatch",
            t.name
        );
        Ok(TensorData {
            data_type: t.data_type,
            dims: t.dims.clone(),
            bytes,
        })
    }
}
