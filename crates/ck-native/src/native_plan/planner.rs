//! Lowers a specialised (static-shape, constant-folded) loop-free stage graph to the
//! native executor's plan: DirectML-shaped ops over explicit strided views, every
//! intermediate placed in ONE heap by liveness first-fit, weights packed in one blob.
//!
//! This is a direct port of the development `tools/model_export/dense_loopfree/plan.py`
//! and keeps its iteration orders, so the two produce the same plan for the same graph.

use super::onnx::{self, AttributeProto, ExternalData, ModelProto, NodeProto, TensorData};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::Path;

const ALIGN: i64 = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dt {
    F32,
    F16,
    U8,
    I64,
    I32,
}

impl Dt {
    fn from_onnx(t: i32) -> Result<Self> {
        Ok(match t {
            onnx::FLOAT => Dt::F32,
            onnx::FLOAT16 => Dt::F16,
            onnx::BOOL => Dt::U8,
            onnx::INT64 => Dt::I64,
            onnx::INT32 => Dt::I32,
            t => bail!("unsupported dtype {t}"),
        })
    }
    fn size(self) -> i64 {
        match self {
            Dt::F32 | Dt::I32 => 4,
            Dt::F16 => 2,
            Dt::U8 => 1,
            Dt::I64 => 8,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Dt::F32 => "f32",
            Dt::F16 => "f16",
            Dt::U8 => "u8",
            Dt::I64 => "i64",
            Dt::I32 => "i32",
        }
    }
}

#[derive(Clone, Debug)]
pub struct View {
    pub v: String,
    pub dtype: Dt,
    pub sizes: Vec<i64>,
    pub strides: Vec<i64>,
    pub offset: i64,
    pub weights: bool,
    pub base: i64,
}

#[derive(Clone, Copy, Debug)]
pub enum Num {
    I(i64),
    F(f64),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Unary = 0,
    Binary = 1,
    Where = 2,
    Reduce = 3,
    Gemm = 4,
    Softmax = 5,
    RmsNorm = 6,
    Slice = 7,
    Join = 8,
    Pad = 9,
    Conv = 10,
    DepthToSpace = 11,
    Fill = 12,
    Mha = 13,
}

#[derive(Clone, Debug)]
pub struct PlanOp {
    pub kind: Kind,
    pub ins: Vec<View>,
    pub outs: Vec<View>,
    pub attrs: Vec<(&'static str, Vec<Num>)>,
}

pub struct IoSpec {
    pub name: String,
    pub dtype: Dt,
    pub shape: Vec<i64>,
    pub base: i64,
}

pub struct Plan {
    pub ops: Vec<PlanOp>,
    pub heap_bytes: i64,
    pub naive_bytes: i64,
    pub weights: Vec<u8>,
    pub inputs: Vec<IoSpec>,
    pub outputs: Vec<IoSpec>,
}

fn fn_code(name: &str) -> i64 {
    match name {
        "identity" => 0,
        "cast" => 1,
        "sqrt" => 2,
        "sigmoid" => 3,
        "exp" => 4,
        "sin" => 5,
        "cos" => 6,
        "neg" => 7,
        "clip" => 8,
        "erf" => 9,
        "add" => 20,
        "sub" => 21,
        "mul" => 22,
        "div" => 23,
        "equal" => 24,
        "max" => 30,
        "sum" => 31,
        "mean" => 32,
        other => panic!("unknown fn {other}"),
    }
}

fn contig(shape: &[i64]) -> Vec<i64> {
    let mut st = vec![0; shape.len()];
    let mut acc = 1;
    for i in (0..shape.len()).rev() {
        st[i] = acc;
        acc *= shape[i];
    }
    st
}

/// out[idx] = data[src(idx)], where `map(o, axis)` maps an output coordinate to the input one.
fn gather_broadcast(
    data: &[i32],
    in_shape: &[i64],
    out_shape: &[i64],
    map: impl Fn(i64, usize) -> i64,
) -> Vec<i32> {
    let in_st = contig(in_shape);
    let total = nelem(out_shape) as usize;
    let mut out = Vec::with_capacity(total);
    let mut idx = vec![0i64; out_shape.len()];
    for _ in 0..total {
        let src: i64 = idx
            .iter()
            .enumerate()
            .map(|(k, &o)| map(o, k) * in_st[k])
            .sum();
        out.push(data[src as usize]);
        for k in (0..idx.len()).rev() {
            idx[k] += 1;
            if idx[k] < out_shape[k] {
                break;
            }
            idx[k] = 0;
        }
    }
    out
}

fn nelem(shape: &[i64]) -> i64 {
    shape.iter().product()
}

fn pad_left(s: &[i64], r: usize) -> Vec<i64> {
    let mut v = vec![1; r - s.len()];
    v.extend_from_slice(s);
    v
}

fn attr<'a>(n: &'a NodeProto, name: &str) -> Option<&'a AttributeProto> {
    n.attribute.iter().find(|a| a.name == name)
}
fn attr_i(n: &NodeProto, name: &str, default: i64) -> i64 {
    attr(n, name).map(|a| a.i).unwrap_or(default)
}
fn attr_f(n: &NodeProto, name: &str, default: f64) -> f64 {
    attr(n, name).map(|a| a.f as f64).unwrap_or(default)
}
fn attr_ints(n: &NodeProto, name: &str, default: &[i64]) -> Vec<i64> {
    attr(n, name)
        .map(|a| a.ints.clone())
        .unwrap_or_else(|| default.to_vec())
}
fn attr_s(n: &NodeProto, name: &str, default: &str) -> String {
    attr(n, name)
        .map(|a| String::from_utf8_lossy(&a.s).into_owned())
        .unwrap_or_else(|| default.to_owned())
}

pub struct Planner {
    nodes: Vec<NodeProto>,
    graph_in: Vec<String>,
    graph_out: Vec<String>,
    inits: HashMap<String, TensorData>,
    init_order: Vec<String>,
    dtype: HashMap<String, Dt>,
    shape: HashMap<String, Vec<i64>>,
    order: Vec<String>,
    alias: HashMap<String, String>,
    ops: Vec<PlanOp>,
    pending: HashMap<usize, (String, String, String, String, f64, String)>,
    score_budget: i64,
    mask_cache: HashMap<String, std::rc::Rc<Vec<bool>>>,
    /// Attention masks of the form `X[b,i] == X[b,j]`, kept as the small id table X [B, L]
    /// instead of a materialised [B,1,L,L] boolean tensor.
    region_masks: HashMap<String, std::rc::Rc<Vec<i32>>>,
}

impl Planner {
    /// `dims` resolves symbolic graph-input dimensions (empty for fully specialised graphs).
    pub fn new(model: &ModelProto, model_dir: &Path, dims: &HashMap<String, i64>) -> Result<Self> {
        let g = model.graph.as_ref().context("model without graph")?;
        let mut ext = ExternalData::new(model_dir);
        let mut p = Planner {
            nodes: g.node.clone(),
            graph_in: g.input.iter().map(|i| i.name.clone()).collect(),
            graph_out: g.output.iter().map(|o| o.name.clone()).collect(),
            inits: HashMap::new(),
            init_order: Vec::new(),
            dtype: HashMap::new(),
            shape: HashMap::new(),
            order: Vec::new(),
            alias: HashMap::new(),
            ops: Vec::new(),
            pending: HashMap::new(),
            mask_cache: HashMap::new(),
            region_masks: HashMap::new(),
            score_budget: std::env::var("CK_PLAN_SCORE_BUDGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(160_000_000),
        };
        for i in &g.input {
            let tt = i
                .r#type
                .as_ref()
                .and_then(|t| t.tensor_type.as_ref())
                .context("non-tensor input")?;
            let mut shape = Vec::new();
            for d in tt.shape.as_ref().map(|s| s.dim.as_slice()).unwrap_or(&[]) {
                shape.push(match (&d.dim_value, &d.dim_param) {
                    (Some(v), _) => *v,
                    (None, Some(p)) => *dims
                        .get(p)
                        .with_context(|| format!("unbound dimension {p}"))?,
                    _ => bail!("input {} has an unknown dimension", i.name),
                });
            }
            p.define(&i.name, Dt::from_onnx(tt.elem_type)?, shape);
        }
        for t in &g.initializer {
            let data = ext.tensor(t)?;
            p.define(&t.name, Dt::from_onnx(t.data_type)?, t.dims.clone());
            p.init_order.push(t.name.clone());
            p.inits.insert(t.name.clone(), data);
        }
        Ok(p)
    }

    // ---------- value helpers ----------
    fn root<'a>(&'a self, mut x: &'a str) -> &'a str {
        while let Some(next) = self.alias.get(x) {
            x = next;
        }
        x
    }
    fn define(&mut self, name: &str, dtype: Dt, shape: Vec<i64>) {
        if !self.dtype.contains_key(name) {
            self.order.push(name.to_owned());
        }
        self.dtype.insert(name.to_owned(), dtype);
        self.shape.insert(name.to_owned(), shape);
    }
    fn sh(&self, x: &str) -> Result<Vec<i64>> {
        self.shape
            .get(x)
            .cloned()
            .with_context(|| format!("unknown value {x}"))
    }
    fn dt(&self, x: &str) -> Result<Dt> {
        self.dtype
            .get(x)
            .copied()
            .with_context(|| format!("unknown value {x}"))
    }
    fn konst(&self, x: &str) -> Option<&TensorData> {
        self.inits.get(x)
    }
    fn ints(&self, x: &str) -> Result<Vec<i64>> {
        self.konst(x)
            .with_context(|| format!("{x} is not constant"))?
            .to_i64()
    }
    fn scalar(&self, x: &str) -> Result<f64> {
        let v = self
            .konst(x)
            .with_context(|| format!("{x} is not constant"))?
            .to_f64()?;
        ensure!(v.len() == 1, "{x} is not a scalar");
        Ok(v[0])
    }
    fn view(
        &self,
        x: &str,
        shape: Option<Vec<i64>>,
        strides: Option<Vec<i64>>,
        offset: i64,
    ) -> Result<View> {
        let shape = match shape {
            Some(s) => s,
            None => self.sh(x)?,
        };
        let strides = strides.unwrap_or_else(|| contig(&shape));
        Ok(View {
            v: self.root(x).to_owned(),
            dtype: self.dt(x)?,
            sizes: shape,
            strides,
            offset,
            weights: false,
            base: 0,
        })
    }
    fn full(&self, x: &str) -> Result<View> {
        self.view(x, None, None, 0)
    }
    fn bcast(&self, x: &str, out_shape: &[i64]) -> Result<View> {
        let s = self.sh(x)?;
        let r = out_shape.len();
        let pad = pad_left(&s, r);
        let mut cs = vec![0; r - s.len()];
        cs.extend(contig(&s));
        let st = (0..r)
            .map(|i| {
                if pad[i] == 1 && out_shape[i] != 1 {
                    0
                } else {
                    cs[i]
                }
            })
            .collect();
        self.view(x, Some(out_shape.to_vec()), Some(st), 0)
    }
    fn emit(
        &mut self,
        kind: Kind,
        mut ins: Vec<View>,
        outs: Vec<View>,
        attrs: Vec<(&'static str, Vec<Num>)>,
    ) {
        if outs.iter().any(|o| nelem(&o.sizes) == 0) {
            return;
        }
        if kind == Kind::Join {
            ins.retain(|v| nelem(&v.sizes) > 0);
        }
        self.ops.push(PlanOp {
            kind,
            ins,
            outs,
            attrs,
        });
    }

    // ---------- lowering ----------
    pub fn lower(&mut self) -> Result<()> {
        let nodes = self.nodes.clone();
        let mut consumers: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            for x in &n.input {
                consumers.entry(x.clone()).or_default().push(i);
            }
        }
        let producer: HashMap<&str, usize> = nodes
            .iter()
            .enumerate()
            .flat_map(|(i, n)| n.output.iter().map(move |o| (o.as_str(), i)))
            .collect();
        let mut skip = HashSet::new();
        self.detect_region_masks(&nodes, &consumers, &producer, &mut skip)?;
        for i in 0..nodes.len() {
            if skip.contains(&i) {
                continue;
            }
            if let Some(p) = self.pending.remove(&i) {
                self.emit_masked_mha(&p.0, &p.1, &p.2, &p.3, p.4, &p.5)?;
                continue;
            }
            if self.try_softmax(i, &nodes, &consumers, &mut skip)?
                || self.try_masked_mha(i, &nodes, &consumers, &producer, &mut skip)?
            {
                continue;
            }
            self.lower_node(&nodes[i])
                .with_context(|| format!("lowering {} ({})", nodes[i].name, nodes[i].op_type))?;
        }
        ensure!(
            self.pending.is_empty(),
            "masked attention without its final MatMul"
        );
        Ok(())
    }

    fn try_softmax(
        &mut self,
        i: usize,
        nodes: &[NodeProto],
        consumers: &HashMap<String, Vec<usize>>,
        skip: &mut HashSet<usize>,
    ) -> Result<bool> {
        let n = &nodes[i];
        if n.op_type != "ReduceMax" || n.input.len() < 2 {
            return Ok(false);
        }
        let x = &n.input[0];
        let Some(axes) = self.konst(&n.input[1]) else {
            return Ok(false);
        };
        let axes = axes.to_i64()?;
        let r = self.sh(x)?.len() as i64;
        if axes != [-1] && axes != [r - 1] {
            return Ok(false);
        }
        let m = &n.output[0];
        let cm = consumers.get(m).cloned().unwrap_or_default();
        let subs: Vec<usize> = cm
            .iter()
            .copied()
            .filter(|&j| nodes[j].op_type == "Sub" && nodes[j].input == [x.clone(), m.clone()])
            .collect();
        if subs.len() != 1 || cm.len() != 1 {
            return Ok(false);
        }
        let sub = subs[0];
        let exps = consumers
            .get(&nodes[sub].output[0])
            .cloned()
            .unwrap_or_default();
        if exps.len() != 1 || nodes[exps[0]].op_type != "Exp" {
            return Ok(false);
        }
        let e = &nodes[exps[0]].output[0];
        let cs = consumers.get(e).cloned().unwrap_or_default();
        let rs: Vec<usize> = cs
            .iter()
            .copied()
            .filter(|&j| nodes[j].op_type == "ReduceSum")
            .collect();
        let dv: Vec<usize> = cs
            .iter()
            .copied()
            .filter(|&j| nodes[j].op_type == "Div")
            .collect();
        if cs.len() != 2
            || rs.len() != 1
            || dv.len() != 1
            || nodes[dv[0]].input != [e.clone(), nodes[rs[0]].output[0].clone()]
        {
            return Ok(false);
        }
        if consumers
            .get(&nodes[rs[0]].output[0])
            .map(|v| v.len())
            .unwrap_or(0)
            != 1
        {
            return Ok(false);
        }
        let out = nodes[dv[0]].output[0].clone();
        let (dt, s) = (self.dt(x)?, self.sh(x)?);
        self.define(&out, dt, s);
        let v = self.full(x)?;
        let o = self.full(&out)?;
        self.emit(
            Kind::Softmax,
            vec![v],
            vec![o],
            vec![("axis", vec![Num::I(r - 1)])],
        );
        for j in [sub, exps[0], rs[0], dv[0]] {
            skip.insert(j);
        }
        Ok(true)
    }

    fn try_masked_mha(
        &mut self,
        i: usize,
        nodes: &[NodeProto],
        consumers: &HashMap<String, Vec<usize>>,
        producer: &HashMap<&str, usize>,
        skip: &mut HashSet<usize>,
    ) -> Result<bool> {
        let n = &nodes[i];
        if n.op_type != "MatMul" {
            return Ok(false);
        }
        let (q, kt) = (&n.input[0], &n.input[1]);
        let Some(&tr) = producer.get(kt.as_str()) else {
            return Ok(false);
        };
        let trn = &nodes[tr];
        if trn.op_type != "Transpose"
            || attr_ints(trn, "perm", &[]) != [0, 1, 3, 2]
            || self.shape.get(q).map(|s| s.len()) != Some(4)
        {
            return Ok(false);
        }
        let only = |x: &str, op: &str| -> Option<usize> {
            let c = consumers.get(x)?;
            (c.len() == 1 && nodes[c[0]].op_type == op).then_some(c[0])
        };
        let Some(mul) = only(&n.output[0], "Mul") else {
            return Ok(false);
        };
        if nodes[mul].input[0] != n.output[0]
            || self.konst(&nodes[mul].input[1]).map(|t| t.elements()) != Some(1)
        {
            return Ok(false);
        }
        let Some(wh) = only(&nodes[mul].output[0], "Where") else {
            return Ok(false);
        };
        if nodes[wh].input[1] != nodes[mul].output[0] || !self.is_constant_mask(&nodes[wh].input[0])
        {
            return Ok(false);
        }
        match self.konst(&nodes[wh].input[2]) {
            Some(t) if t.elements() == 1 => {
                let f = t.to_f64()?[0];
                if !(f.is_infinite() && f < 0.0) {
                    return Ok(false);
                }
            }
            _ => return Ok(false),
        }
        let Some(sm) = only(&nodes[wh].output[0], "Softmax") else {
            return Ok(false);
        };
        if !matches!(attr_i(&nodes[sm], "axis", -1), -1 | 3) {
            return Ok(false);
        }
        let Some(mm) = only(&nodes[sm].output[0], "MatMul") else {
            return Ok(false);
        };
        if nodes[mm].input[0] != nodes[sm].output[0]
            || consumers.get(kt).map(|v| v.len()) != Some(1)
        {
            return Ok(false);
        }
        let scale = self.scalar(&nodes[mul].input[1])?;
        self.pending.insert(
            mm,
            (
                q.clone(),
                trn.input[0].clone(),
                nodes[mm].input[1].clone(),
                nodes[wh].input[0].clone(),
                scale,
                nodes[mm].output[0].clone(),
            ),
        );
        for j in [tr, mul, wh, sm] {
            skip.insert(j);
        }
        Ok(true)
    }

    fn emit_masked_mha(
        &mut self,
        q: &str,
        k: &str,
        v: &str,
        mask: &str,
        scale: f64,
        out: &str,
    ) -> Result<()> {
        let qs = self.sh(q)?;
        let (b, h, s, d) = (qs[0], qs[1], qs[2], qs[3]);
        let l = self.sh(k)?[2];
        let dv = self.sh(v)?[3];
        ensure!(
            self.sh(k)? == [b, h, l, d] && self.sh(v)? == [b, h, l, dv],
            "masked mha shapes"
        );
        let ms = self.sh(mask)?;
        ensure!(
            ms.len() == 4 && ms[1] == 1 && ms[3] == l,
            "masked mha mask shape"
        );
        let dt = self.dt(q)?;
        self.define(out, dt, vec![b, h, s, dv]);
        let flat = format!("{out}/bsd");
        self.define(&flat, dt, vec![b, s, h * dv]);
        let srcs = [
            (
                "q",
                self.view(
                    q,
                    Some(vec![b, s, h, d]),
                    Some(vec![h * s * d, d, s * d, 1]),
                    0,
                )?,
                s,
                d,
            ),
            (
                "k",
                self.view(
                    k,
                    Some(vec![b, l, h, d]),
                    Some(vec![h * l * d, d, l * d, 1]),
                    0,
                )?,
                l,
                d,
            ),
            (
                "v",
                self.view(
                    v,
                    Some(vec![b, l, h, dv]),
                    Some(vec![h * l * dv, dv, l * dv, 1]),
                    0,
                )?,
                l,
                dv,
            ),
        ];
        let mut packed = Vec::new();
        for (name, src, n_, d_) in srcs {
            let t = format!("{out}/{name}3");
            self.define(&t, dt, vec![b, n_, h * d_]);
            let o = self.view(&t, Some(vec![b, n_, h, d_]), None, 0)?;
            self.emit(
                Kind::Unary,
                vec![src],
                vec![o],
                vec![("fn", vec![Num::I(0)])],
            );
            packed.push(self.full(&t)?);
        }
        // windows whose mask is all-true run unmasked; others keep the mask
        let full: Vec<bool> = if let Some(rid) = self.region_masks.get(mask).cloned() {
            let n = ms[3] as usize;
            (0..b as usize)
                .map(|w| {
                    let wi = if ms[0] == b { w } else { 0 };
                    let row = &rid[wi * n..(wi + 1) * n];
                    row.iter().all(|&x| x == row[0])
                })
                .collect()
        } else {
            let m = self.mask_bools(mask)?;
            let per = (ms[2] * ms[3]) as usize;
            (0..b as usize)
                .map(|w| {
                    let wi = if ms[0] == b { w } else { 0 };
                    m[wi * per..(wi + 1) * per].iter().all(|&x| x)
                })
                .collect()
        };
        let mut runs = Vec::new();
        let mut w = 0usize;
        while w < b as usize {
            let mut e = w;
            while e < b as usize && full[e] == full[w] {
                e += 1;
            }
            runs.push((w as i64, e as i64, full[w]));
            w = e;
        }
        let bstep = (self.score_budget / (h * s * l)).max(1);
        let runs: Vec<(i64, i64, bool)> = runs
            .into_iter()
            .flat_map(|(a, e, f)| {
                (a..e)
                    .step_by(bstep as usize)
                    .map(move |x| (x, (x + bstep).min(e), f))
            })
            .collect();
        for (r0, r1, is_full) in runs {
            let nb = r1 - r0;
            let sub = |v: &View| {
                let mut v = v.clone();
                let per = v.strides[0];
                v.sizes[0] = nb;
                v.offset += r0 * per;
                v
            };
            let mut ins = vec![sub(&packed[0]), sub(&packed[1]), sub(&packed[2])];
            if !is_full {
                ins.push(self.compact_mask(
                    mask,
                    &ms,
                    b,
                    h,
                    s,
                    l,
                    &format!("{out}/w{r0}"),
                    r0,
                    r1,
                )?);
            }
            let o = sub(&self.full(&flat)?);
            self.emit(
                Kind::Mha,
                ins,
                vec![o],
                vec![("heads", vec![Num::I(h)]), ("scale", vec![Num::F(scale)])],
            );
        }
        let src = self.view(
            &flat,
            Some(vec![b, h, s, dv]),
            Some(vec![s * h * dv, dv, h * dv, 1]),
            0,
        )?;
        let o = self.full(out)?;
        self.emit(
            Kind::Unary,
            vec![src],
            vec![o],
            vec![("fn", vec![Num::I(0)])],
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn compact_mask(
        &mut self,
        mask: &str,
        ms: &[i64],
        b: i64,
        h: i64,
        s: i64,
        l: i64,
        out: &str,
        r0: i64,
        r1: i64,
    ) -> Result<View> {
        let name = format!("{mask}/region_i32");
        if let Some(rid) = self.region_masks.get(mask).cloned()
            && !self.inits.contains_key(&name)
        {
            let bytes = rid.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.inits.insert(
                name.clone(),
                TensorData {
                    data_type: onnx::INT32,
                    dims: vec![ms[0], l],
                    bytes,
                },
            );
            self.define(&name, Dt::I32, vec![ms[0], l]);
        }
        if ms[2] == 1 && !self.region_masks.contains_key(mask) {
            let m = self.mask_bools(mask)?;
            let name = format!("{mask}/kpm_i32");
            if !self.inits.contains_key(&name) {
                let bytes = m.iter().flat_map(|&x| (x as i32).to_le_bytes()).collect();
                self.inits.insert(
                    name.clone(),
                    TensorData {
                        data_type: onnx::INT32,
                        dims: ms.to_vec(),
                        bytes,
                    },
                );
                self.define(&name, Dt::I32, ms.to_vec());
            }
            let mut v = self.view(
                &name,
                Some(vec![r1 - r0, h, s, l]),
                Some(vec![if ms[0] == b { l } else { 0 }, 0, 0, 1]),
                0,
            )?;
            v.offset = if ms[0] == b { r0 * l } else { 0 };
            return Ok(v);
        }
        ensure!(
            ms[2] == l && (ms[0] == 1 || ms[0] == b),
            "unsupported mask layout"
        );
        if !self.inits.contains_key(&name) {
            let m = self.mask_bools(mask)?;
            let (bm, n) = (ms[0] as usize, l as usize);
            let mut rid = vec![0i32; bm * n];
            for w in 0..bm {
                for i in 0..n {
                    let row = &m[(w * n + i) * n..(w * n + i + 1) * n];
                    rid[w * n + i] = row.iter().position(|&x| x).unwrap_or(0) as i32;
                }
                for i in 0..n {
                    for j in 0..n {
                        ensure!(
                            (rid[w * n + i] == rid[w * n + j]) == m[(w * n + i) * n + j],
                            "mask is not a region-equality mask"
                        );
                    }
                }
            }
            let bytes = rid.iter().flat_map(|x| x.to_le_bytes()).collect();
            self.inits.insert(
                name.clone(),
                TensorData {
                    data_type: onnx::INT32,
                    dims: vec![ms[0], l],
                    bytes,
                },
            );
            self.define(&name, Dt::I32, vec![ms[0], l]);
        }
        let rb = if ms[0] == b { l } else { 0 };
        let nb = r1 - r0;
        let eq = format!("{out}/mask_u8");
        self.define(&eq, Dt::U8, vec![nb, 1, s, l]);
        let mut va = self.view(&name, Some(vec![nb, 1, s, l]), Some(vec![rb, 0, 1, 0]), 0)?;
        va.offset = r0 * rb;
        let mut vb = self.view(&name, Some(vec![nb, 1, s, l]), Some(vec![rb, 0, 0, 1]), 0)?;
        vb.offset = r0 * rb;
        let eqv = self.full(&eq)?;
        self.emit(
            Kind::Binary,
            vec![va, vb],
            vec![eqv],
            vec![("fn", vec![Num::I(fn_code("equal"))])],
        );
        let m32 = format!("{out}/mask_i32");
        self.define(&m32, Dt::I32, vec![nb, 1, s, l]);
        let (src, dst) = (self.full(&eq)?, self.full(&m32)?);
        self.emit(
            Kind::Unary,
            vec![src],
            vec![dst],
            vec![("fn", vec![Num::I(fn_code("cast"))])],
        );
        self.view(&m32, Some(vec![nb, h, s, l]), Some(vec![s * l, 0, l, 1]), 0)
    }

    fn is_constant_mask(&self, x: &str) -> bool {
        self.konst(x).is_some() || self.region_masks.contains_key(x)
    }

    /// Evaluates a small int constant expression (Expand/Tile/Reshape/Unsqueeze/Cast of an
    /// initializer). Returns `None` when `x` is not such an expression.
    fn eval_int(
        &self,
        x: &str,
        nodes: &[NodeProto],
        producer: &HashMap<&str, usize>,
        depth: usize,
    ) -> Result<Option<(Vec<i64>, Vec<i32>)>> {
        if let Some(t) = self.inits.get(x) {
            if !matches!(t.data_type, onnx::INT32 | onnx::INT64) {
                return Ok(None);
            }
            return Ok(Some((
                t.dims.clone(),
                t.to_i64()?.into_iter().map(|v| v as i32).collect(),
            )));
        }
        let Some(&i) = producer.get(x) else {
            return Ok(None);
        };
        if depth > 8 {
            return Ok(None);
        }
        let n = &nodes[i];
        let Some((sh, d)) = (match n.op_type.as_str() {
            "Identity" | "Unsqueeze" | "Reshape" | "Expand" | "Tile" => {
                self.eval_int(&n.input[0], nodes, producer, depth + 1)?
            }
            "Cast" if matches!(attr_i(n, "to", 0) as i32, onnx::INT32 | onnx::INT64) => {
                self.eval_int(&n.input[0], nodes, producer, depth + 1)?
            }
            _ => None,
        }) else {
            return Ok(None);
        };
        Ok(Some(match n.op_type.as_str() {
            "Unsqueeze" => {
                let axes = self.ints(&n.input[1])?;
                let r = (sh.len() + axes.len()) as i64;
                let mut axes: Vec<i64> = axes.into_iter().map(|a| a.rem_euclid(r)).collect();
                axes.sort();
                let mut sh = sh;
                for a in axes {
                    sh.insert(a as usize, 1);
                }
                (sh, d)
            }
            "Reshape" => {
                let mut out: Vec<i64> = self
                    .ints(&n.input[1])?
                    .iter()
                    .enumerate()
                    .map(|(k, &v)| if v == 0 { sh[k] } else { v })
                    .collect();
                if let Some(k) = out.iter().position(|&v| v == -1) {
                    let rest: i64 = out.iter().filter(|&&v| v != -1).product();
                    out[k] = d.len() as i64 / rest;
                }
                (out, d)
            }
            "Expand" => {
                let tgt = self.ints(&n.input[1])?;
                let r = sh.len().max(tgt.len());
                let (ps, pt) = (pad_left(&sh, r), pad_left(&tgt, r));
                let out: Vec<i64> = ps.iter().zip(&pt).map(|(a, b)| *a.max(b)).collect();
                let data = gather_broadcast(&d, &ps, &out, |o, k| if ps[k] == 1 { 0 } else { o });
                (out, data)
            }
            "Tile" => {
                let reps = self.ints(&n.input[1])?;
                let out: Vec<i64> = sh.iter().zip(&reps).map(|(a, b)| a * b).collect();
                let data = gather_broadcast(&d, &sh, &out, |o, k| o % sh[k]);
                (out, data)
            }
            _ => (sh, d),
        }))
    }

    /// Finds `Where(Unsqueeze(Equal(Unsqueeze(X,-1), Unsqueeze(X,-2)), 1), ...)` attention masks
    /// with constant id tables X and records X instead of building the mask at full size.
    fn detect_region_masks(
        &mut self,
        nodes: &[NodeProto],
        consumers: &HashMap<String, Vec<usize>>,
        producer: &HashMap<&str, usize>,
        skip: &mut HashSet<usize>,
    ) -> Result<()> {
        let single = |x: &str, op: &str| -> Option<usize> {
            let &i = producer.get(x)?;
            (nodes[i].op_type == op).then_some(i)
        };
        let column = |a: &[i64]| a == [-1] || a == [2];
        let row = |a: &[i64]| a == [-2] || a == [1];
        let mut chain = HashSet::new();
        for n in nodes.iter().filter(|n| n.op_type == "Where") {
            let c = &n.input[0];
            if self.region_masks.contains_key(c) || self.konst(c).is_some() {
                continue;
            }
            let Some(pc) = single(c, "Unsqueeze") else {
                continue;
            };
            if self.ints(&nodes[pc].input[1]).ok() != Some(vec![1]) {
                continue;
            }
            let Some(pe) = single(&nodes[pc].input[0], "Equal") else {
                continue;
            };
            let (Some(pa), Some(pb)) = (
                single(&nodes[pe].input[0], "Unsqueeze"),
                single(&nodes[pe].input[1], "Unsqueeze"),
            ) else {
                continue;
            };
            if nodes[pa].input[0] != nodes[pb].input[0] {
                continue;
            }
            let (Ok(a), Ok(b)) = (
                self.ints(&nodes[pa].input[1]),
                self.ints(&nodes[pb].input[1]),
            ) else {
                continue;
            };
            if !(column(&a) && row(&b) || column(&b) && row(&a)) {
                continue;
            }
            if !consumers
                .get(c)
                .is_some_and(|v| v.iter().all(|&j| nodes[j].op_type == "Where"))
            {
                continue;
            }
            let Some((sh, data)) = self.eval_int(&nodes[pa].input[0], nodes, producer, 0)? else {
                continue;
            };
            if sh.len() != 2 {
                continue;
            }
            self.define(c, Dt::U8, vec![sh[0], 1, sh[1], sh[1]]);
            self.region_masks.insert(c.clone(), std::rc::Rc::new(data));
            chain.extend([pc, pe, pa, pb]);
        }
        // upstream id-table nodes (Expand/Tile/...) are dead once every consumer is skipped
        loop {
            let mut grew = false;
            for (i, n) in nodes.iter().enumerate() {
                if chain.contains(&i)
                    || n.output.is_empty()
                    || n.output.iter().any(|o| self.graph_out.contains(o))
                {
                    continue;
                }
                let users: Vec<usize> = n
                    .output
                    .iter()
                    .flat_map(|o| consumers.get(o).cloned().unwrap_or_default())
                    .collect();
                if !users.is_empty() && users.iter().all(|u| chain.contains(u)) {
                    chain.insert(i);
                    grew = true;
                }
            }
            if !grew {
                break;
            }
        }
        skip.extend(chain);
        Ok(())
    }

    /// Boolean attention masks are decoded once; every window chunk reuses them.
    fn mask_bools(&mut self, mask: &str) -> Result<std::rc::Rc<Vec<bool>>> {
        if let Some(m) = self.mask_cache.get(mask) {
            return Ok(m.clone());
        }
        let m = std::rc::Rc::new(self.konst(mask).context("mask not constant")?.to_bool()?);
        self.mask_cache.insert(mask.to_owned(), m.clone());
        Ok(m)
    }

    fn alias_to(&mut self, n: &NodeProto, shape: Vec<i64>) -> Result<()> {
        let x = &n.input[0];
        ensure!(
            nelem(&shape) == nelem(&self.sh(x)?),
            "alias size mismatch {}",
            n.name
        );
        let dt = self.dt(x)?;
        self.define(&n.output[0], dt, shape);
        self.alias.insert(n.output[0].clone(), x.clone());
        Ok(())
    }

    fn unary(&mut self, n: &NodeProto, fname: &str, out_dt: Option<Dt>) -> Result<()> {
        let x = &n.input[0];
        let dt = out_dt.unwrap_or(self.dt(x)?);
        let s = self.sh(x)?;
        self.define(&n.output[0], dt, s);
        let (i, o) = (self.full(x)?, self.full(&n.output[0])?);
        self.emit(
            Kind::Unary,
            vec![i],
            vec![o],
            vec![("fn", vec![Num::I(fn_code(fname))])],
        );
        Ok(())
    }

    fn binary(&mut self, n: &NodeProto, fname: &str) -> Result<()> {
        let (a, b) = (&n.input[0], &n.input[1]);
        let (sa, sb) = (self.sh(a)?, self.sh(b)?);
        let r = sa.len().max(sb.len());
        let (pa, pb) = (pad_left(&sa, r), pad_left(&sb, r));
        let out: Vec<i64> = pa.iter().zip(&pb).map(|(p, q)| *p.max(q)).collect();
        let dt = self.dt(a)?;
        ensure!(self.dt(b)? == dt, "binary dtype mismatch {}", n.name);
        self.define(&n.output[0], dt, out.clone());
        let (va, vb, vo) = (
            self.bcast(a, &out)?,
            self.bcast(b, &out)?,
            self.full(&n.output[0])?,
        );
        self.emit(
            Kind::Binary,
            vec![va, vb],
            vec![vo],
            vec![("fn", vec![Num::I(fn_code(fname))])],
        );
        Ok(())
    }

    fn slice_op(
        &mut self,
        x: &str,
        out: &str,
        off: Vec<i64>,
        size: Vec<i64>,
        step: Vec<i64>,
    ) -> Result<()> {
        let dt = self.dt(x)?;
        self.define(out, dt, size);
        let (i, o) = (self.full(x)?, self.full(out)?);
        self.emit(
            Kind::Slice,
            vec![i],
            vec![o],
            vec![
                ("offsets", off.into_iter().map(Num::I).collect()),
                ("steps", step.into_iter().map(Num::I).collect()),
            ],
        );
        Ok(())
    }

    fn lower_node(&mut self, n: &NodeProto) -> Result<()> {
        let inp = |k: usize| n.input.get(k).filter(|s| !s.is_empty()).cloned();
        match n.op_type.as_str() {
            "Reshape" => {
                let tgt = self.ints(&n.input[1])?;
                let s = self.sh(&n.input[0])?;
                let mut out: Vec<i64> = tgt
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| if d == 0 { s[i] } else { d })
                    .collect();
                if let Some(k) = out.iter().position(|&d| d == -1) {
                    let rest: i64 = out
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != k)
                        .map(|(_, d)| *d)
                        .product();
                    out[k] = nelem(&s) / rest;
                }
                self.alias_to(n, out)
            }
            "Unsqueeze" => {
                let mut axes = self.ints(&n.input[1])?;
                axes.sort();
                let mut s = self.sh(&n.input[0])?;
                let r = (s.len() + axes.len()) as i64;
                let mut axes: Vec<i64> = axes
                    .into_iter()
                    .map(|a| if a < 0 { a + r } else { a })
                    .collect();
                axes.sort();
                for a in axes {
                    s.insert(a as usize, 1);
                }
                self.alias_to(n, s)
            }
            "Squeeze" => {
                let s = self.sh(&n.input[0])?;
                let axes: Vec<usize> = match inp(1) {
                    Some(a) => self
                        .ints(&a)?
                        .into_iter()
                        .map(|a| a.rem_euclid(s.len() as i64) as usize)
                        .collect(),
                    None => (0..s.len()).filter(|&i| s[i] == 1).collect(),
                };
                let out = s
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !axes.contains(i))
                    .map(|(_, d)| *d)
                    .collect();
                self.alias_to(n, out)
            }
            "Identity" => {
                let s = self.sh(&n.input[0])?;
                self.alias_to(n, s)
            }
            "Transpose" => {
                let x = &n.input[0];
                let perm = attr_ints(n, "perm", &[]);
                let (s, cs) = (self.sh(x)?, contig(&self.sh(x)?));
                let out: Vec<i64> = perm.iter().map(|&p| s[p as usize]).collect();
                let st: Vec<i64> = perm.iter().map(|&p| cs[p as usize]).collect();
                let dt = self.dt(x)?;
                self.define(&n.output[0], dt, out.clone());
                let (i, o) = (
                    self.view(x, Some(out), Some(st), 0)?,
                    self.full(&n.output[0])?,
                );
                self.emit(Kind::Unary, vec![i], vec![o], vec![("fn", vec![Num::I(0)])]);
                Ok(())
            }
            "Expand" => {
                let x = &n.input[0];
                let tgt = self.ints(&n.input[1])?;
                let s = self.sh(x)?;
                let r = s.len().max(tgt.len());
                let out: Vec<i64> = pad_left(&s, r)
                    .iter()
                    .zip(pad_left(&tgt, r))
                    .map(|(p, q)| *p.max(&q))
                    .collect();
                let dt = self.dt(x)?;
                self.define(&n.output[0], dt, out.clone());
                let (i, o) = (self.bcast(x, &out)?, self.full(&n.output[0])?);
                self.emit(Kind::Unary, vec![i], vec![o], vec![("fn", vec![Num::I(0)])]);
                Ok(())
            }
            "Add" => self.binary(n, "add"),
            "Sub" => self.binary(n, "sub"),
            "Mul" => self.binary(n, "mul"),
            "Div" => self.binary(n, "div"),
            "Where" => {
                let (c, a, b) = (&n.input[0], &n.input[1], &n.input[2]);
                ensure!(
                    !self.region_masks.contains_key(c),
                    "region mask {c} used outside masked attention"
                );
                let shapes = [self.sh(c)?, self.sh(a)?, self.sh(b)?];
                let r = shapes.iter().map(|s| s.len()).max().unwrap();
                let pads: Vec<Vec<i64>> = shapes.iter().map(|s| pad_left(s, r)).collect();
                let out: Vec<i64> = (0..r)
                    .map(|i| pads.iter().map(|p| p[i]).max().unwrap())
                    .collect();
                let dt = self.dt(a)?;
                self.define(&n.output[0], dt, out.clone());
                let ins = vec![
                    self.bcast(c, &out)?,
                    self.bcast(a, &out)?,
                    self.bcast(b, &out)?,
                ];
                let o = self.full(&n.output[0])?;
                self.emit(Kind::Where, ins, vec![o], vec![]);
                Ok(())
            }
            "Cast" => {
                let to = Dt::from_onnx(attr_i(n, "to", 0) as i32)?;
                if to == self.dt(&n.input[0])? {
                    let s = self.sh(&n.input[0])?;
                    return self.alias_to(n, s);
                }
                self.unary(n, "cast", Some(to))
            }
            "Sqrt" => self.unary(n, "sqrt", None),
            "Erf" => self.unary(n, "erf", None),
            "Sigmoid" => self.unary(n, "sigmoid", None),
            "Exp" => self.unary(n, "exp", None),
            "Sin" => self.unary(n, "sin", None),
            "Cos" => self.unary(n, "cos", None),
            "Neg" => self.unary(n, "neg", None),
            "Clip" => {
                let lo = match inp(1) {
                    Some(x) => self.scalar(&x)?,
                    None => -3.4e38,
                };
                let hi = match inp(2) {
                    Some(x) => self.scalar(&x)?,
                    None => 3.4e38,
                };
                let x = &n.input[0];
                let (dt, s) = (self.dt(x)?, self.sh(x)?);
                self.define(&n.output[0], dt, s);
                let (i, o) = (self.full(x)?, self.full(&n.output[0])?);
                self.emit(
                    Kind::Unary,
                    vec![i],
                    vec![o],
                    vec![
                        ("fn", vec![Num::I(fn_code("clip"))]),
                        ("lo", vec![Num::F(lo)]),
                        ("hi", vec![Num::F(hi)]),
                    ],
                );
                Ok(())
            }
            "ReduceMax" | "ReduceSum" | "ReduceMean" => {
                let fname = match n.op_type.as_str() {
                    "ReduceMax" => "max",
                    "ReduceSum" => "sum",
                    _ => "mean",
                };
                let x = &n.input[0];
                let s = self.sh(x)?;
                let mut axes: Vec<i64> = self
                    .ints(&n.input[1])?
                    .into_iter()
                    .map(|a| a.rem_euclid(s.len() as i64))
                    .collect();
                axes.sort();
                ensure!(attr_i(n, "keepdims", 1) == 1, "keepdims=0 unsupported");
                let out = s
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| if axes.contains(&(i as i64)) { 1 } else { d })
                    .collect();
                let dt = self.dt(x)?;
                self.define(&n.output[0], dt, out);
                let (i, o) = (self.full(x)?, self.full(&n.output[0])?);
                self.emit(
                    Kind::Reduce,
                    vec![i],
                    vec![o],
                    vec![
                        ("fn", vec![Num::I(fn_code(fname))]),
                        ("axes", axes.into_iter().map(Num::I).collect()),
                    ],
                );
                Ok(())
            }
            "Softmax" => {
                let x = &n.input[0];
                let r = self.sh(x)?.len() as i64;
                let axis = attr_i(n, "axis", -1).rem_euclid(r);
                ensure!(axis == r - 1, "softmax on non-last axis");
                let (dt, s) = (self.dt(x)?, self.sh(x)?);
                self.define(&n.output[0], dt, s);
                let (i, o) = (self.full(x)?, self.full(&n.output[0])?);
                self.emit(
                    Kind::Softmax,
                    vec![i],
                    vec![o],
                    vec![("axis", vec![Num::I(axis)])],
                );
                Ok(())
            }
            "MatMul" => self.matmul(n),
            "Gemm" => self.gemm(n),
            "MultiHeadAttention" => self.mha(n),
            "SimplifiedLayerNormalization" => {
                let (x, w) = (&n.input[0], &n.input[1]);
                let s = self.sh(x)?;
                let ax = attr_i(n, "axis", -1);
                ensure!(ax == -1 || ax == s.len() as i64 - 1, "rmsnorm axis");
                let dt = self.dt(x)?;
                self.define(&n.output[0], dt, s.clone());
                let c = *s.last().unwrap();
                let rows = nelem(&s) / c;
                let ins = vec![
                    self.view(x, Some(vec![rows, c]), None, 0)?,
                    self.view(w, Some(vec![c]), None, 0)?,
                ];
                let o = self.view(&n.output[0], Some(vec![rows, c]), None, 0)?;
                self.emit(
                    Kind::RmsNorm,
                    ins,
                    vec![o],
                    vec![("eps", vec![Num::F(attr_f(n, "epsilon", 1e-5))])],
                );
                Ok(())
            }
            "Slice" => {
                let x = &n.input[0];
                let s = self.sh(x)?;
                let starts = self.ints(&n.input[1])?;
                let ends = self.ints(&n.input[2])?;
                let axes = match inp(3) {
                    Some(a) => self.ints(&a)?,
                    None => (0..starts.len() as i64).collect(),
                };
                let steps = match inp(4) {
                    Some(a) => self.ints(&a)?,
                    None => vec![1; starts.len()],
                };
                let (mut off, mut size, mut step) = (vec![0; s.len()], s.clone(), vec![1; s.len()]);
                for (((st, en), ax), sp) in starts.into_iter().zip(ends).zip(axes).zip(steps) {
                    let ax = ax.rem_euclid(s.len() as i64) as usize;
                    let d = s[ax];
                    ensure!(sp > 0, "negative slice step unsupported");
                    let st = (if st < 0 { st + d } else { st }).clamp(0, d);
                    let en = (if en < 0 { en.saturating_add(d) } else { en }).clamp(0, d);
                    off[ax] = st;
                    size[ax] = ((en - st + sp - 1) / sp).max(0);
                    step[ax] = sp;
                }
                self.slice_op(x, &n.output[0], off, size, step)
            }
            "Split" => {
                let x = &n.input[0];
                let s = self.sh(x)?;
                let ax = attr_i(n, "axis", 0).rem_euclid(s.len() as i64) as usize;
                let sizes = match inp(1) {
                    Some(a) => self.ints(&a)?,
                    None => vec![s[ax] / n.output.len() as i64; n.output.len()],
                };
                let mut pos = 0;
                for (o, sz) in n.output.iter().zip(sizes) {
                    let mut off = vec![0; s.len()];
                    off[ax] = pos;
                    let mut size = s.clone();
                    size[ax] = sz;
                    self.slice_op(x, o, off, size, vec![1; s.len()])?;
                    pos += sz;
                }
                Ok(())
            }
            "Concat" => {
                let ax =
                    attr_i(n, "axis", 0).rem_euclid(self.sh(&n.input[0])?.len() as i64) as usize;
                let mut ins = Vec::new();
                for x in &n.input {
                    if nelem(&self.sh(x)?) > 0 {
                        ins.push(x.clone());
                    }
                }
                let mut out = self.sh(&ins[0])?;
                out[ax] = ins
                    .iter()
                    .map(|x| self.sh(x).map(|s| s[ax]))
                    .sum::<Result<i64>>()?;
                let dt = self.dt(&ins[0])?;
                self.define(&n.output[0], dt, out);
                let views = ins
                    .iter()
                    .map(|x| self.full(x))
                    .collect::<Result<Vec<_>>>()?;
                let o = self.full(&n.output[0])?;
                self.emit(
                    Kind::Join,
                    views,
                    vec![o],
                    vec![("axis", vec![Num::I(ax as i64)])],
                );
                Ok(())
            }
            "Gather" => {
                let x = &n.input[0];
                let s = self.sh(x)?;
                let ax = attr_i(n, "axis", 0).rem_euclid(s.len() as i64) as usize;
                let ind = self
                    .konst(&n.input[1])
                    .context("only constant Gather supported")?;
                ensure!(ind.dims.is_empty(), "only scalar Gather supported");
                let i = ind.to_i64()?[0].rem_euclid(s[ax]);
                let mut off = vec![0; s.len()];
                off[ax] = i;
                let mut size = s.clone();
                size[ax] = 1;
                let tmp = format!("{}/keep", n.output[0]);
                self.slice_op(x, &tmp, off, size, vec![1; s.len()])?;
                let dt = self.dt(x)?;
                self.define(
                    &n.output[0],
                    dt,
                    s.iter()
                        .enumerate()
                        .filter(|(j, _)| *j != ax)
                        .map(|(_, d)| *d)
                        .collect(),
                );
                self.alias.insert(n.output[0].clone(), tmp);
                Ok(())
            }
            "Pad" => {
                let x = &n.input[0];
                let s = self.sh(x)?;
                let pads = self.ints(&n.input[1])?;
                let value = match inp(2) {
                    Some(v) => self.scalar(&v)?,
                    None => 0.0,
                };
                ensure!(inp(3).is_none(), "Pad axes unsupported");
                let r = s.len();
                let (start, end) = (pads[..r].to_vec(), pads[r..].to_vec());
                let out = (0..r).map(|i| s[i] + start[i] + end[i]).collect();
                let mode = match attr_s(n, "mode", "constant").as_str() {
                    "constant" => 0,
                    "reflect" => 1,
                    "edge" => 2,
                    m => bail!("pad mode {m}"),
                };
                let dt = self.dt(x)?;
                self.define(&n.output[0], dt, out);
                let (i, o) = (self.full(x)?, self.full(&n.output[0])?);
                self.emit(
                    Kind::Pad,
                    vec![i],
                    vec![o],
                    vec![
                        ("start", start.into_iter().map(Num::I).collect()),
                        ("end", end.into_iter().map(Num::I).collect()),
                        ("mode", vec![Num::I(mode)]),
                        ("value", vec![Num::F(value)]),
                    ],
                );
                Ok(())
            }
            "Conv" => self.conv(n, false),
            "ConvTranspose" => self.conv_transpose(n),
            "DepthToSpace" => {
                let x = &n.input[0];
                let bs = attr_i(n, "blocksize", 0);
                let mode = match attr_s(n, "mode", "DCR").as_str() {
                    "DCR" => 0,
                    "CRD" => 1,
                    m => bail!("depth-to-space mode {m}"),
                };
                let s = self.sh(x)?;
                let dt = self.dt(x)?;
                self.define(
                    &n.output[0],
                    dt,
                    vec![s[0], s[1] / (bs * bs), s[2] * bs, s[3] * bs],
                );
                let (i, o) = (self.full(x)?, self.full(&n.output[0])?);
                self.emit(
                    Kind::DepthToSpace,
                    vec![i],
                    vec![o],
                    vec![("block", vec![Num::I(bs)]), ("mode", vec![Num::I(mode)])],
                );
                Ok(())
            }
            other => bail!("unsupported op {other}"),
        }
    }

    fn matmul(&mut self, n: &NodeProto) -> Result<()> {
        let (a, b) = (&n.input[0], &n.input[1]);
        let (sa, sb) = (self.sh(a)?, self.sh(b)?);
        ensure!(sa.len() >= 2 && sb.len() >= 2, "vector matmul unsupported");
        let (m, k) = (sa[sa.len() - 2], sa[sa.len() - 1]);
        let (k2, nn) = (sb[sb.len() - 2], sb[sb.len() - 1]);
        ensure!(k == k2, "matmul K mismatch {}", n.name);
        let (ba, bb) = (&sa[..sa.len() - 2], &sb[..sb.len() - 2]);
        let r = ba.len().max(bb.len());
        let batch: Vec<i64> = pad_left(ba, r)
            .iter()
            .zip(pad_left(bb, r))
            .map(|(p, q)| *p.max(&q))
            .collect();
        let mut out = batch.clone();
        out.extend([m, nn]);
        let dt = self.dt(a)?;
        self.define(&n.output[0], dt, out);
        let vo = self.full(&n.output[0])?;
        let with = |x: &[i64], y: &[i64]| -> Vec<i64> { [x, y].concat() };
        if k == 1 {
            let mut va = self.bcast(a, &with(&batch, &[m, 1]))?;
            let mut vb = self.bcast(b, &with(&batch, &[1, nn]))?;
            let (la, lb) = (va.strides.len(), vb.strides.len());
            va.strides[la - 1] = 0;
            vb.strides[lb - 2] = 0;
            va.sizes = with(&batch, &[m, nn]);
            vb.sizes = with(&batch, &[m, nn]);
            self.emit(
                Kind::Binary,
                vec![va, vb],
                vec![vo],
                vec![("fn", vec![Num::I(fn_code("mul"))])],
            );
            return Ok(());
        }
        let va = self.bcast(a, &with(&batch, &[m, k]))?;
        let vb = self.bcast(b, &with(&batch, &[k, nn]))?;
        self.emit(
            Kind::Gemm,
            vec![va, vb],
            vec![vo],
            vec![("alpha", vec![Num::F(1.0)])],
        );
        Ok(())
    }

    fn gemm(&mut self, n: &NodeProto) -> Result<()> {
        let (a, b) = (&n.input[0], &n.input[1]);
        let c = n.input.get(2).filter(|s| !s.is_empty()).cloned();
        let (ta, tb) = (attr_i(n, "transA", 0), attr_i(n, "transB", 0));
        let (alpha, beta) = (attr_f(n, "alpha", 1.0), attr_f(n, "beta", 1.0));
        let (sa, sb) = (self.sh(a)?, self.sh(b)?);
        let (m, k) = if ta != 0 {
            (sa[1], sa[0])
        } else {
            (sa[0], sa[1])
        };
        let nn = if tb != 0 { sb[0] } else { sb[1] };
        let dt = self.dt(a)?;
        self.define(&n.output[0], dt, vec![m, nn]);
        let va = self.view(
            a,
            Some(vec![1, 1, m, k]),
            Some(if ta != 0 {
                vec![0, 0, 1, m]
            } else {
                vec![0, 0, k, 1]
            }),
            0,
        )?;
        let vb = self.view(
            b,
            Some(vec![1, 1, k, nn]),
            Some(if tb != 0 {
                vec![0, 0, 1, k]
            } else {
                vec![0, 0, nn, 1]
            }),
            0,
        )?;
        let mut ins = vec![va, vb];
        if let Some(c) = &c {
            let cs = self.sh(c)?;
            if cs == [nn] || cs == [1, nn] {
                ins.push(self.view(c, Some(vec![1, 1, m, nn]), Some(vec![0, 0, 0, 1]), 0)?);
            } else if cs == [m, nn] {
                ins.push(self.view(c, Some(vec![1, 1, m, nn]), Some(vec![0, 0, nn, 1]), 0)?);
            } else {
                bail!("Gemm bias shape {cs:?}");
            }
        }
        let o = self.view(&n.output[0], Some(vec![1, 1, m, nn]), None, 0)?;
        let beta = if c.is_some() { beta } else { 0.0 };
        self.emit(
            Kind::Gemm,
            ins,
            vec![o],
            vec![("alpha", vec![Num::F(alpha)]), ("beta", vec![Num::F(beta)])],
        );
        Ok(())
    }

    fn mha(&mut self, n: &NodeProto) -> Result<()> {
        let (q, k, v) = (&n.input[0], &n.input[1], &n.input[2]);
        let h = attr_i(n, "num_heads", 0);
        let scale = attr_f(n, "scale", 0.0);
        let qs = self.sh(q)?;
        let (b, s, hd) = (qs[0], qs[1], qs[2]);
        let l = self.sh(k)?[1];
        let hdv = self.sh(v)?[2];
        let out = &n.output[0];
        let dt = self.dt(q)?;
        self.define(out, dt, vec![b, s, hdv]);
        // Optional runtime key-padding mask (input 4, int32 [b or 1, l], nonzero = valid), broadcast
        // over heads and queries for DirectML's boolean MHA mask.
        let mask = match n.input.get(4).filter(|m| !m.is_empty()) {
            Some(m) => {
                let ms = self.sh(m)?;
                ensure!(
                    ms.len() == 2 && ms[1] == l && (ms[0] == 1 || ms[0] == b),
                    "MHA key_padding_mask must be [1 or batch, kv_length]"
                );
                ensure!(self.dt(m)? == Dt::I32, "MHA key_padding_mask must be int32");
                Some((m.clone(), ms[0]))
            }
            None => None,
        };
        let per_b = h * s * l;
        let bstep = if per_b <= self.score_budget {
            (self.score_budget / per_b).max(1)
        } else {
            1
        };
        let sstep = if per_b <= self.score_budget {
            s
        } else {
            (self.score_budget / (h * l) / 64 * 64).max(64)
        };
        let mut b0 = 0;
        while b0 < b {
            let nb = bstep.min(b - b0);
            let mut s0 = 0;
            while s0 < s {
                let ns = sstep.min(s - s0);
                let ins = vec![
                    self.view(
                        q,
                        Some(vec![nb, ns, hd]),
                        Some(vec![s * hd, hd, 1]),
                        b0 * s * hd + s0 * hd,
                    )?,
                    self.view(
                        k,
                        Some(vec![nb, l, hd]),
                        Some(vec![l * hd, hd, 1]),
                        b0 * l * hd,
                    )?,
                    self.view(
                        v,
                        Some(vec![nb, l, hdv]),
                        Some(vec![l * hdv, hdv, 1]),
                        b0 * l * hdv,
                    )?,
                ];
                let mut ins = ins;
                if let Some((m, mb)) = &mask {
                    let per = if *mb == b { l } else { 0 };
                    ins.push(self.view(
                        m,
                        Some(vec![nb, h, ns, l]),
                        Some(vec![per, 0, 0, 1]),
                        b0 * per,
                    )?);
                }
                let o = self.view(
                    out,
                    Some(vec![nb, ns, hdv]),
                    Some(vec![s * hdv, hdv, 1]),
                    b0 * s * hdv + s0 * hdv,
                )?;
                self.emit(
                    Kind::Mha,
                    ins,
                    vec![o],
                    vec![("heads", vec![Num::I(h)]), ("scale", vec![Num::F(scale)])],
                );
                s0 += sstep;
            }
            b0 += bstep;
        }
        Ok(())
    }

    fn conv(&mut self, n: &NodeProto, transpose: bool) -> Result<()> {
        let (x, w) = (&n.input[0], &n.input[1]);
        let bias = n.input.get(2).filter(|s| !s.is_empty()).cloned();
        let xs = self.sh(x)?;
        let (nb, c, h, wd) = (xs[0], xs[1], xs[2], xs[3]);
        let ws = self.sh(w)?;
        let (kh, kw) = (ws[2], ws[3]);
        let st = attr_ints(n, "strides", &[1, 1]);
        let dl = attr_ints(n, "dilations", &[1, 1]);
        let pd = attr_ints(n, "pads", &[0, 0, 0, 0]);
        let mut grp = attr_i(n, "group", 1);
        let (m, oh, ow, op_) = if transpose {
            let op_ = attr_ints(n, "output_padding", &[0, 0]);
            let m = ws[1] * grp;
            (
                m,
                (h - 1) * st[0] - pd[0] - pd[2] + dl[0] * (kh - 1) + 1 + op_[0],
                (wd - 1) * st[1] - pd[1] - pd[3] + dl[1] * (kw - 1) + 1 + op_[1],
                op_,
            )
        } else {
            (
                ws[0],
                (h + pd[0] + pd[2] - dl[0] * (kh - 1) - 1) / st[0] + 1,
                (wd + pd[1] + pd[3] - dl[1] * (kw - 1) - 1) / st[1] + 1,
                vec![0, 0],
            )
        };
        let dt = self.dt(x)?;
        self.define(&n.output[0], dt, vec![nb, m, oh, ow]);
        let mut xin = self.full(x)?;
        if !transpose && grp > 1 && grp == c && m % c == 0 && m > c && ws[1] == 1 {
            // depthwise with channel multiplier r: duplicate input channels, run multiplier-1 depthwise
            let r = m / c;
            let dup = format!("{}/dup", n.output[0]);
            self.define(&dup, dt, vec![nb, m, h, wd]);
            let src = self.view(
                x,
                Some(vec![nb, c, r, h, wd]),
                Some(vec![c * h * wd, h * wd, 0, wd, 1]),
                0,
            )?;
            let dst = self.view(&dup, Some(vec![nb, c, r, h, wd]), None, 0)?;
            self.emit(
                Kind::Unary,
                vec![src],
                vec![dst],
                vec![("fn", vec![Num::I(0)])],
            );
            xin = self.full(&dup)?;
            grp = m;
        }
        let mut ins = vec![xin, self.full(w)?];
        if let Some(b) = &bias {
            ins.push(self.view(b, Some(vec![1, m, 1, 1]), None, 0)?);
        }
        let o = self.full(&n.output[0])?;
        self.emit(
            Kind::Conv,
            ins,
            vec![o],
            vec![
                ("strides", st.iter().map(|&v| Num::I(v)).collect()),
                ("dilations", dl.iter().map(|&v| Num::I(v)).collect()),
                ("start", vec![Num::I(pd[0]), Num::I(pd[1])]),
                ("end", vec![Num::I(pd[2]), Num::I(pd[3])]),
                ("out_pad", op_.iter().map(|&v| Num::I(v)).collect()),
                ("groups", vec![Num::I(grp)]),
                ("transpose", vec![Num::I(transpose as i64)]),
            ],
        );
        Ok(())
    }

    fn conv_transpose(&mut self, n: &NodeProto) -> Result<()> {
        let (x, w) = (&n.input[0], &n.input[1]);
        let bias = n.input.get(2).filter(|s| !s.is_empty()).cloned();
        let xs = self.sh(x)?;
        let (nb, c, h, wd) = (xs[0], xs[1], xs[2], xs[3]);
        let ws = self.sh(w)?;
        let (m, kh, kw) = (ws[1], ws[2], ws[3]);
        let st = attr_ints(n, "strides", &[1, 1]);
        let pd = attr_ints(n, "pads", &[0, 0, 0, 0]);
        let grp = attr_i(n, "group", 1);
        let dl = attr_ints(n, "dilations", &[1, 1]);
        let opad = attr_ints(n, "output_padding", &[0, 0]);
        if !(grp == 1
            && dl == [1, 1]
            && opad == [0, 0]
            && kh == 2 * st[0]
            && kw == 2 * st[1]
            && st[0] == st[1]
            && pd[0] == pd[2]
            && pd[1] == pd[3])
        {
            return self.conv(n, true);
        }
        let (s_, k) = (st[0], kh);
        let dt = self.dt(x)?;
        let out = n.output[0].clone();
        let oh = (h - 1) * s_ - pd[0] - pd[2] + k;
        let ow = (wd - 1) * s_ - pd[1] - pd[3] + k;
        self.define(&out, dt, vec![nb, m, oh, ow]);
        let xt = format!("{out}/xt");
        self.define(&xt, dt, vec![nb, h * wd, c]);
        let src = self.view(
            x,
            Some(vec![nb, h * wd, c]),
            Some(vec![c * h * wd, 1, h * wd]),
            0,
        )?;
        let dst = self.full(&xt)?;
        self.emit(
            Kind::Unary,
            vec![src],
            vec![dst],
            vec![("fn", vec![Num::I(0)])],
        );
        let pp = format!("{out}/taps");
        self.define(&pp, dt, vec![nb, h * wd, m * k * k]);
        let ga = self.view(&xt, Some(vec![1, nb, h * wd, c]), None, 0)?;
        let gb = self.view(
            w,
            Some(vec![1, nb, c, m * k * k]),
            Some(vec![0, 0, m * k * k, 1]),
            0,
        )?;
        let go = self.view(&pp, Some(vec![1, nb, h * wd, m * k * k]), None, 0)?;
        self.emit(
            Kind::Gemm,
            vec![ga, gb],
            vec![go],
            vec![("alpha", vec![Num::F(1.0)])],
        );
        let acc = format!("{out}/acc");
        let (hb, wb) = (h + 1, wd + 1);
        self.define(&acc, dt, vec![nb, m, hb * s_, wb * s_]);
        let accv = self.full(&acc)?;
        self.emit(
            Kind::Fill,
            vec![],
            vec![accv],
            vec![("value", vec![Num::F(0.0)])],
        );
        let accst = vec![
            m * hb * s_ * wb * s_,
            hb * s_ * wb * s_,
            s_ * wb * s_,
            wb * s_,
            s_,
            1,
        ];
        let sizes = vec![nb, m, h, s_, wd, s_];
        let tapst = vec![h * wd * m * k * k, k * k, wd * m * k * k, k, m * k * k, 1];
        for a in 0..2 {
            for bb in 0..2 {
                let region = self.view(
                    &acc,
                    Some(sizes.clone()),
                    Some(accst.clone()),
                    a * s_ * wb * s_ + bb * s_,
                )?;
                let taps = self.view(
                    &pp,
                    Some(sizes.clone()),
                    Some(tapst.clone()),
                    a * s_ * k + bb * s_,
                )?;
                self.emit(
                    Kind::Binary,
                    vec![region.clone(), taps],
                    vec![region],
                    vec![("fn", vec![Num::I(fn_code("add"))])],
                );
            }
        }
        let (i, o) = (self.full(&acc)?, self.full(&out)?);
        self.emit(
            Kind::Slice,
            vec![i],
            vec![o],
            vec![
                (
                    "offsets",
                    vec![Num::I(0), Num::I(0), Num::I(pd[0]), Num::I(pd[1])],
                ),
                ("steps", vec![Num::I(1); 4]),
            ],
        );
        if let Some(b) = &bias {
            let ov = self.full(&out)?;
            let bv = self.view(b, Some(vec![nb, m, oh, ow]), Some(vec![0, 1, 0, 0]), 0)?;
            self.emit(
                Kind::Binary,
                vec![ov.clone(), bv],
                vec![ov],
                vec![("fn", vec![Num::I(fn_code("add"))])],
            );
        }
        Ok(())
    }

    // ---------- memory planning ----------
    pub fn plan(mut self) -> Result<Plan> {
        // byte size per root, in first-definition order (matches the Python dict order)
        let mut roots: Vec<String> = Vec::new();
        let mut seen = HashSet::new();
        for x in &self.order {
            let r = self.root(x).to_owned();
            if self.inits.contains_key(&r) {
                continue;
            }
            if seen.insert(r.clone()) {
                roots.push(r);
            }
        }
        let bytes_of: HashMap<String, i64> = roots
            .iter()
            .map(|r| (r.clone(), nelem(&self.shape[r]) * self.dtype[r].size()))
            .collect();
        let mut first: HashMap<String, i64> = HashMap::new();
        let mut last: HashMap<String, i64> = HashMap::new();
        for (i, op) in self.ops.iter().enumerate() {
            let i = i as i64;
            for o in &op.outs {
                first.entry(o.v.clone()).or_insert(i);
                let e = last.entry(o.v.clone()).or_insert(i);
                *e = (*e).max(i);
            }
            for v in &op.ins {
                let e = last.entry(v.v.clone()).or_insert(-1);
                *e = (*e).max(i);
            }
        }
        let n_ops = self.ops.len() as i64;
        for x in &self.graph_in {
            first.insert(self.root(x).to_owned(), -1);
        }
        for x in &self.graph_out {
            last.insert(self.root(x).to_owned(), n_ops);
        }
        let mut used_weights: Vec<String> = self
            .ops
            .iter()
            .flat_map(|op| op.ins.iter())
            .filter(|v| self.inits.contains_key(&v.v))
            .map(|v| v.v.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        used_weights.sort();
        let mut live: Vec<String> = roots
            .iter()
            .filter(|r| first.contains_key(*r))
            .cloned()
            .collect();
        live.sort_by_key(|r| -bytes_of[r]); // stable: ties keep definition order
        let mut placed: Vec<(i64, i64, i64, i64)> = Vec::new();
        let mut offset: HashMap<String, i64> = HashMap::new();
        for r in &live {
            let size = (bytes_of[r] + ALIGN - 1) / ALIGN * ALIGN;
            let a0 = first[r];
            let a1 = *last.get(r).unwrap_or(&a0);
            let mut busy: Vec<(i64, i64)> = placed
                .iter()
                .filter(|(_, _, b0, b1)| !(*b1 < a0 || a1 < *b0))
                .map(|(o, s, _, _)| (*o, o + s))
                .collect();
            busy.sort();
            let mut cand = 0;
            for (lo, hi) in busy {
                if lo - cand >= size {
                    break;
                }
                cand = cand.max(hi);
            }
            offset.insert(r.clone(), cand);
            placed.push((cand, size, a0, a1));
        }
        let heap = placed.iter().map(|(o, s, _, _)| o + s).max().unwrap_or(0);
        let mut woff: HashMap<String, i64> = HashMap::new();
        let mut blob: Vec<u8> = Vec::new();
        for name in &used_weights {
            let t = &self.inits[name];
            ensure!(
                t.data_type != onnx::INT64,
                "int64 weight used as data: {name}"
            );
            woff.insert(name.clone(), blob.len() as i64);
            blob.extend_from_slice(&t.bytes);
            let pad = (ALIGN as usize - blob.len() % ALIGN as usize) % ALIGN as usize;
            blob.extend(std::iter::repeat_n(0u8, pad));
        }
        for op in &mut self.ops {
            for v in op.ins.iter_mut().chain(op.outs.iter_mut()) {
                if let Some(&b) = woff.get(&v.v) {
                    v.weights = true;
                    v.base = b;
                } else {
                    v.weights = false;
                    v.base = *offset
                        .get(&v.v)
                        .with_context(|| format!("value {} has no heap slot", v.v))?;
                }
            }
        }
        let naive = live.iter().map(|r| bytes_of[r]).sum();
        let io = |names: &[String], required: bool| -> Result<Vec<IoSpec>> {
            let mut out = Vec::new();
            for x in names {
                match offset.get(self.root(x)) {
                    Some(&base) => out.push(IoSpec {
                        name: x.clone(),
                        dtype: self.dtype[x],
                        shape: self.shape[x].clone(),
                        base,
                    }),
                    None if !required => {}
                    None => bail!("output {x} has no heap slot"),
                }
            }
            Ok(out)
        };
        let inputs = io(&self.graph_in, false)?;
        let outputs = io(&self.graph_out, true)?;
        Ok(Plan {
            ops: self.ops,
            heap_bytes: heap,
            naive_bytes: naive,
            weights: blob,
            inputs,
            outputs,
        })
    }
}

impl Plan {
    /// Token stream read by the native executor (same format as plan.py's plan.txt).
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "PLAN 1 HEAP {} WEIGHTS {}",
            self.heap_bytes,
            self.weights.len()
        );
        for (tag, list) in [("INPUTS", &self.inputs), ("OUTPUTS", &self.outputs)] {
            let _ = writeln!(s, "{tag} {}", list.len());
            for x in list {
                let dims: Vec<String> = x.shape.iter().map(|d| d.to_string()).collect();
                let _ = writeln!(
                    s,
                    "IO {} {} {} {} {}",
                    x.name,
                    x.dtype.name(),
                    x.base,
                    x.shape.len(),
                    dims.join(" ")
                );
            }
        }
        let _ = writeln!(s, "OPS {}", self.ops.len());
        for op in &self.ops {
            let mut attrs = Vec::new();
            for (k, vals) in &op.attrs {
                let v: Vec<String> = vals
                    .iter()
                    .map(|n| match n {
                        Num::I(i) => i.to_string(),
                        Num::F(f) => format!("{f:?}"),
                    })
                    .collect();
                attrs.push(format!("{k} {} {}", vals.len(), v.join(" ")));
            }
            let _ = writeln!(
                s,
                "OP {} {} {} {} {}",
                op.kind as i32,
                attrs.len(),
                attrs.join(" "),
                op.ins.len(),
                op.outs.len()
            );
            for v in op.ins.iter().chain(&op.outs) {
                let sz: Vec<String> = v.sizes.iter().map(|d| d.to_string()).collect();
                let st: Vec<String> = v.strides.iter().map(|d| d.to_string()).collect();
                let _ = writeln!(
                    s,
                    "V {} {} {} {} {} {} {}",
                    if v.weights { 'w' } else { 'h' },
                    v.base,
                    v.offset,
                    v.dtype.name(),
                    v.sizes.len(),
                    sz.join(" "),
                    st.join(" ")
                );
            }
        }
        s
    }
}
