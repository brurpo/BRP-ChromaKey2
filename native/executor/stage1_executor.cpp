// Development-only native Stage1 executor: runs a static plan (plan.txt + weights.bin)
// produced by tools/model_export/dense_loopfree/plan.py with raw DirectML operators.
// No ONNX Runtime, no Python. Every intermediate lives at a planned offset inside ONE
// device heap; one shared scratch buffer serves all operators (they run in order).
//
// Build with -DCK_EXECUTOR_DLL -shared for the C ABI (ck_plan_executor.dll).
// Usage: stage1_executor --plan DIR --data DIR --directml PATH [--adapter N] [--reps N]
//        [--out DIR]   (writes out_<name>.f32 raw outputs and a JSON report)
#include "native_dml_support.h"
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <limits>

// Vendor metacommands off: DirectML uses its own vendor-neutral shaders. Some drivers (AMD
// integrated Radeon, 32.0.21030) return wrong normalization results through metacommands.
static bool g_no_metacommands = std::getenv("CK_NO_METACOMMANDS") != nullptr;

static const GUID IID_Exec_Initializer = {0x427c1113, 0x435c, 0x469c, {0x86, 0x76, 0x4d, 0x5d, 0xd0, 0x72, 0xf8, 0x13}};
static void ck(HRESULT hr, const char* where) {
    if (FAILED(hr)) { char s[160]; std::snprintf(s, sizeof s, "%s: HRESULT 0x%08lx", where, (unsigned long)hr); throw std::runtime_error(s); }
}
// MSVC member functions returning structs: this in RCX, result pointer in RDX.
template <class T> static T member_struct(void* self, size_t index) {
    T result{};
    using Fn = void(__attribute__((ms_abi)) *)(void*, T*);
    auto table = *reinterpret_cast<void***>(self);
    reinterpret_cast<Fn>(table[index])(self, &result);
    return result;
}
static DML_BINDING_PROPERTIES properties(IDMLDispatchable* d) { return member_struct<DML_BINDING_PROPERTIES>(d, 8); }

// ---------------------------------------------------------------- plan
struct View {
    char buf = 'h'; uint64_t base = 0, offset = 0; std::string dtype; std::vector<int64_t> sizes, strides;
};
struct Op {
    int kind = 0; std::map<std::string, std::vector<double>> attr; std::vector<View> in, out;
    double a(const char* k, double d = 0) const { auto it = attr.find(k); return it == attr.end() || it->second.empty() ? d : it->second[0]; }
    std::vector<int64_t> v(const char* k) const { std::vector<int64_t> r; auto it = attr.find(k); if (it != attr.end()) for (double x : it->second) r.push_back(int64_t(x)); return r; }
};
struct IoSpec { std::string name, dtype; uint64_t base = 0; std::vector<int64_t> shape; };
struct Plan { uint64_t heap = 0, weights = 0; std::vector<IoSpec> inputs, outputs; std::vector<Op> ops; };
enum Kind { UNARY, BINARY, WHERE, REDUCE, GEMM, SOFTMAX, RMSNORM, SLICE, JOIN, PAD, CONV, D2S, FILL, MHA };

static Plan load_plan(const std::string& path) {
    std::ifstream f(path); if (!f) throw std::runtime_error("cannot read plan " + path);
    Plan p; std::string tok; int ver;
    f >> tok >> ver; if (tok != "PLAN" || ver != 1) throw std::runtime_error("bad plan header");
    f >> tok >> p.heap >> tok >> p.weights;
    auto io = [&](std::vector<IoSpec>& dst) {
        size_t n; f >> tok >> n;
        for (size_t i = 0; i < n; i++) { IoSpec s; size_t r; f >> tok >> s.name >> s.dtype >> s.base >> r; s.shape.resize(r); for (auto& d : s.shape) f >> d; dst.push_back(s); }
    };
    io(p.inputs); io(p.outputs);
    size_t nops; f >> tok >> nops; p.ops.resize(nops);
    auto read_view = [&](View& v) {
        size_t r; f >> tok >> v.buf >> v.base >> v.offset >> v.dtype >> r; if (tok != "V") throw std::runtime_error("bad view");
        v.sizes.resize(r); v.strides.resize(r); for (auto& d : v.sizes) f >> d; for (auto& d : v.strides) f >> d;
    };
    for (auto& op : p.ops) {
        size_t na, ni, no; f >> tok >> op.kind >> na; if (tok != "OP") throw std::runtime_error("bad op");
        for (size_t i = 0; i < na; i++) { std::string k; size_t c; f >> k >> c; std::vector<double> vals(c); for (auto& x : vals) f >> x; op.attr[k] = vals; }
        f >> ni >> no; op.in.resize(ni); op.out.resize(no);
        for (auto& v : op.in) read_view(v);
        for (auto& v : op.out) read_view(v);
    }
    if (!f) throw std::runtime_error("plan truncated");
    return p;
}

// ---------------------------------------------------------------- heavy-operator splitting
// Windows resets a GPU whose single dispatch runs for ~2 s (TDR). On slow GPUs (small
// integrated ones reach ~100 GFLOP/s) one 400-GFLOP convolution exceeds that, so heavy
// operators are split into independent pieces over sub-views of the same buffers: conv by
// frame then output channel, GEMM by rows, attention by query rows. Every output element is
// still computed by the same reduction, so results do not depend on the split.
static double g_max_op_gflop = std::getenv("CK_MAX_OP_GFLOP") ? std::atof(std::getenv("CK_MAX_OP_GFLOP")) : 32.0;

static int64_t numel(const std::vector<int64_t>& s) { int64_t n = 1; for (auto d : s) n *= d; return n; }
static double op_flops(const Op& op) {
    if (op.kind == GEMM) return 2.0 * double(numel(op.out[0].sizes)) * double(op.in[0].sizes.back());
    if (op.kind == CONV) {
        auto& w = op.in[1].sizes; double k = double(numel(w)) / double(w[0]);
        return 2.0 * double(numel(op.a("transpose") != 0 ? op.in[0].sizes : op.out[0].sizes)) * k;
    }
    if (op.kind == MHA) return 4.0 * double(numel(op.in[0].sizes)) * double(op.in[1].sizes[1]);
    return 0;
}
struct Cut { bool out; size_t operand; int dim; };  // dim < 0 counts from the end
static View& cut_view(Op& op, const Cut& c) { return c.out ? op.out[c.operand] : op.in[c.operand]; }
static const View& cut_view(const Op& op, const Cut& c) { return c.out ? op.out[c.operand] : op.in[c.operand]; }
static size_t cut_dim(const View& v, int dim) { return dim < 0 ? v.sizes.size() - size_t(-dim) : size_t(dim); }
// Splittable axes for `op`, in preference order; each axis lists every view it cuts.
static std::vector<std::vector<Cut>> split_axes(const Op& op) {
    std::vector<std::vector<Cut>> axes;
    auto add_if = [&](std::vector<Cut> cuts, size_t n) {  // keep optional cuts only where the view spans n
        std::vector<Cut> ok;
        for (auto& c : cuts) {
            if (c.out ? c.operand >= op.out.size() : c.operand >= op.in.size()) continue;
            const View& v = c.out ? op.out[c.operand] : op.in[c.operand];
            if (v.sizes.size() >= size_t(c.dim < 0 ? -c.dim : c.dim + 1) && v.sizes[cut_dim(v, c.dim)] == int64_t(n)) ok.push_back(c);
        }
        axes.push_back(ok);
    };
    if (op.kind == CONV) {
        add_if({{false, 0, 0}, {true, 0, 0}}, size_t(op.out[0].sizes[0]));
        if (op.a("groups", 1) == 1) {
            size_t ch = size_t(op.out[0].sizes[1]);
            add_if({{false, 1, op.a("transpose") != 0 ? 1 : 0}, {false, 2, 1}, {true, 0, 1}}, ch);
        }
    } else if (op.kind == GEMM) {
        add_if({{false, 0, -2}, {false, 2, -2}, {true, 0, -2}}, size_t(op.out[0].sizes[op.out[0].sizes.size() - 2]));
    } else if (op.kind == MHA) {
        add_if({{false, 0, 1}, {false, 3, -2}, {true, 0, 1}}, size_t(op.out[0].sizes[1]));
    }
    return axes;
}
static void split_op(const Op& op, double limit, size_t axis, std::vector<Op>& dst) {
    double f = op_flops(op);
    auto axes = split_axes(op);
    if (f <= limit || axis >= axes.size()) { dst.push_back(op); return; }
    auto& cuts = axes[axis];
    // The first cut must be the output-defining operand; skip axes the views cannot express.
    bool has_out = false; for (auto& c : cuts) has_out |= c.out;
    if (!has_out) { split_op(op, limit, axis + 1, dst); return; }
    const View& ref = cut_view(op, cuts.back());
    int64_t n = ref.sizes[cut_dim(ref, cuts.back().dim)];
    int64_t pieces = std::min<int64_t>(n, int64_t(std::ceil(f / limit)));
    if (pieces <= 1) { split_op(op, limit, axis + 1, dst); return; }
    for (int64_t p = 0; p < pieces; p++) {
        int64_t lo = n * p / pieces, hi = n * (p + 1) / pieces;
        Op piece = op;
        for (auto& c : cuts) {
            View& v = cut_view(piece, c); size_t d = cut_dim(v, c.dim);
            v.offset += uint64_t(lo * v.strides[d]); v.sizes[d] = hi - lo;
        }
        split_op(piece, limit, axis + 1, dst);
    }
}
static void split_heavy_ops(Plan& p) {
    if (g_max_op_gflop <= 0) return;
    std::vector<Op> ops; ops.reserve(p.ops.size());
    for (auto& op : p.ops) split_op(op, g_max_op_gflop * 1e9, 0, ops);
    p.ops = std::move(ops);
}

// ---------------------------------------------------------------- tensor descs
static uint32_t esize(const std::string& t) { return t == "f32" || t == "i32" ? 4 : t == "f16" ? 2 : 1; }
static DML_TENSOR_DATA_TYPE dmltype(const std::string& t) {
    if (t == "f32") return DML_TENSOR_DATA_TYPE_FLOAT32; if (t == "f16") return DML_TENSOR_DATA_TYPE_FLOAT16;
    if (t == "u8") return DML_TENSOR_DATA_TYPE_UINT8; if (t == "i32") return DML_TENSOR_DATA_TYPE_INT32; throw std::runtime_error("dtype " + t);
}
struct TD {  // owns the arrays a DML_TENSOR_DESC points to
    std::vector<uint32_t> sizes, strides; DML_BUFFER_TENSOR_DESC buf{}; DML_TENSOR_DESC desc{};
    void set(const View& v, const std::vector<int64_t>& s, const std::vector<int64_t>& st) {
        sizes.clear(); strides.clear();
        uint64_t span = 1; bool empty = false;
        for (size_t i = 0; i < s.size(); i++) {
            if (s[i] <= 0) empty = true;
            if (s[i] > 0xffffffffll || st[i] > 0xffffffffll || st[i] < 0) throw std::runtime_error("dimension out of DML range");
            sizes.push_back(uint32_t(s[i])); strides.push_back(uint32_t(st[i])); span += uint64_t(s[i] - 1) * uint64_t(st[i]);
        }
        if (empty) throw std::runtime_error("empty tensor");
        buf.DataType = dmltype(v.dtype); buf.Flags = DML_TENSOR_FLAG_NONE; buf.DimensionCount = UINT(sizes.size());
        buf.Sizes = sizes.data(); buf.Strides = strides.data(); buf.GuaranteedBaseOffsetAlignment = 0;
        buf.TotalTensorSizeInBytes = align_up(span * esize(v.dtype), 4);
        desc = {DML_TENSOR_TYPE_BUFFER, &buf};
    }
};
// pad rank to `rank` with leading 1s (stride 0)
static void pad_to(std::vector<int64_t>& s, std::vector<int64_t>& st, size_t rank) {
    while (s.size() < rank) { s.insert(s.begin(), 1); st.insert(st.begin(), 0); }
}
// merge adjacent dims that are contiguous-compatible for every operand (elementwise ops)
static void coalesce(std::vector<int64_t>& shape, std::vector<std::vector<int64_t>*> strides) {
    for (size_t i = shape.size(); i-- > 0;)  // size-1 dims carry no addressing
        if (shape[i] == 1 && shape.size() > 1) { shape.erase(shape.begin() + long(i)); for (auto* st : strides) st->erase(st->begin() + long(i)); }
    for (size_t i = 0; i + 1 < shape.size();) {
        bool ok = true;
        for (auto* st : strides) if ((*st)[i] != (*st)[i + 1] * shape[i + 1]) { ok = false; break; }
        if (ok) {
            for (auto* st : strides) { (*st)[i] = (*st)[i + 1]; st->erase(st->begin() + long(i + 1)); }
            shape[i] *= shape[i + 1]; shape.erase(shape.begin() + long(i + 1));
        } else i++;
    }
}

// ---------------------------------------------------------------- executor
struct Compiled {
    IDMLCompiledOperator* op = nullptr; DML_BINDING_PROPERTIES props{}; GpuBuffer persistent;
    std::vector<DML_BUFFER_BINDING> in, out; IDMLBindingTable* table = nullptr; size_t input_slots = 0, output_slots = 0;
    ~Compiled() { com_release(op); com_release(table); }
};
struct Exec {
    Harness h; Plan plan; GpuBuffer heap, weights, scratch; uint64_t scratch_bytes = 0;
    std::vector<std::unique_ptr<Compiled>> ops; ID3D12DescriptorHeap* dheap = nullptr; uint32_t descriptors = 0;
    size_t chunk = 1;  // operators per submission; run() grows it (x4 max per step) toward ~0.2 s
    ID3D12Resource* res(const View& v) { return v.buf == 'w' ? weights.res : heap.res; }
    DML_BUFFER_BINDING binding(const View& v, const TD& t) {
        uint64_t off = v.base + v.offset * esize(v.dtype);
        auto& b = v.buf == 'w' ? weights : heap;
        if (off + t.buf.TotalTensorSizeInBytes > b.size) throw std::runtime_error("view outside buffer");
        return {res(v), off, t.buf.TotalTensorSizeInBytes};
    }
    void compile(const Op& op, size_t index);
};

static DML_OPERATOR_TYPE unary_type(int fn) {
    switch (fn) { case 0: return DML_OPERATOR_ELEMENT_WISE_IDENTITY; case 2: return DML_OPERATOR_ELEMENT_WISE_SQRT;
    case 4: return DML_OPERATOR_ELEMENT_WISE_EXP; case 5: return DML_OPERATOR_ELEMENT_WISE_SIN; case 6: return DML_OPERATOR_ELEMENT_WISE_COS;
    case 9: return DML_OPERATOR_ELEMENT_WISE_ERF;
    default: throw std::runtime_error("unary fn"); }
}

void Exec::compile(const Op& op, size_t index) {
    auto c = std::make_unique<Compiled>();
    std::vector<TD> ins(op.in.size()), outs(op.out.size());
    DML_EXECUTION_FLAGS flags = DML_EXECUTION_FLAG_NONE;
    IDMLOperator* dop = nullptr;
    auto create = [&](DML_OPERATOR_TYPE type, const void* desc) {
        DML_OPERATOR_DESC d{type, desc};
        HRESULT hr = h.dml->CreateOperator(&d, IID_DML_Operator, reinterpret_cast<void**>(&dop));
        if (FAILED(hr)) { char s[200]; std::snprintf(s, sizeof s, "CreateOperator op %zu kind %d type %d: 0x%08lx", index, op.kind, int(type), (unsigned long)hr); throw std::runtime_error(s); }
    };
    auto plain = [&](TD& t, const View& v, size_t minrank) { auto s = v.sizes, st = v.strides; pad_to(s, st, minrank); t.set(v, s, st); };
    switch (op.kind) {
    case UNARY: case BINARY: case WHERE: {
        auto shape = op.out[0].sizes;
        std::vector<std::vector<int64_t>> st; for (auto& v : op.in) st.push_back(v.strides); st.push_back(op.out[0].strides);
        std::vector<std::vector<int64_t>*> ptrs; for (auto& s : st) ptrs.push_back(&s);
        coalesce(shape, ptrs);
        size_t rank = std::max<size_t>(4, shape.size());
        for (size_t i = 0; i < op.in.size(); i++) { auto s = shape, t = st[i]; pad_to(s, t, rank); ins[i].set(op.in[i], s, t); }
        { auto s = shape, t = st.back(); pad_to(s, t, rank); outs[0].set(op.out[0], s, t); }
        if (op.kind == BINARY) {
            DML_ELEMENT_WISE_ADD_OPERATOR_DESC d{&ins[0].desc, &ins[1].desc, &outs[0].desc};
            int fn = int(op.a("fn"));
            create(fn == 20 ? DML_OPERATOR_ELEMENT_WISE_ADD : fn == 21 ? DML_OPERATOR_ELEMENT_WISE_SUBTRACT : fn == 22 ? DML_OPERATOR_ELEMENT_WISE_MULTIPLY : fn == 24 ? DML_OPERATOR_ELEMENT_WISE_LOGICAL_EQUALS : DML_OPERATOR_ELEMENT_WISE_DIVIDE, &d);
        } else if (op.kind == WHERE) {
            DML_ELEMENT_WISE_IF_OPERATOR_DESC d{&ins[0].desc, &ins[1].desc, &ins[2].desc, &outs[0].desc}; create(DML_OPERATOR_ELEMENT_WISE_IF, &d);
        } else {
            int fn = int(op.a("fn"));
            if (fn == 1) { DML_CAST_OPERATOR_DESC d{&ins[0].desc, &outs[0].desc}; create(DML_OPERATOR_CAST, &d); }
            else if (fn == 3) { DML_ACTIVATION_SIGMOID_OPERATOR_DESC d{&ins[0].desc, &outs[0].desc}; create(DML_OPERATOR_ACTIVATION_SIGMOID, &d); }
            else if (fn == 7) { DML_SCALE_BIAS sb{-1.f, 0.f}; DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC d{&ins[0].desc, &outs[0].desc, &sb}; create(DML_OPERATOR_ELEMENT_WISE_IDENTITY, &d); }
            else if (fn == 8) { DML_ELEMENT_WISE_CLIP_OPERATOR_DESC d{&ins[0].desc, &outs[0].desc, nullptr, float(op.a("lo")), float(op.a("hi"))}; create(DML_OPERATOR_ELEMENT_WISE_CLIP, &d); }
            else { DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC d{&ins[0].desc, &outs[0].desc, nullptr}; create(unary_type(fn), &d); }
        }
        break; }
    case REDUCE: {
        plain(ins[0], op.in[0], 4); plain(outs[0], op.out[0], 4);
        size_t shift = ins[0].sizes.size() - op.in[0].sizes.size();
        std::vector<UINT> axes; for (auto a : op.v("axes")) axes.push_back(UINT(a + int64_t(shift)));
        int fn = int(op.a("fn"));
        DML_REDUCE_OPERATOR_DESC d{fn == 30 ? DML_REDUCE_FUNCTION_MAX : fn == 31 ? DML_REDUCE_FUNCTION_SUM : DML_REDUCE_FUNCTION_AVERAGE,
                                   &ins[0].desc, &outs[0].desc, UINT(axes.size()), axes.data()};
        create(DML_OPERATOR_REDUCE, &d); break; }
    case SOFTMAX: {
        plain(ins[0], op.in[0], 4); plain(outs[0], op.out[0], 4);
        UINT axis = UINT(op.a("axis") + double(ins[0].sizes.size() - op.in[0].sizes.size()));
        DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC d{&ins[0].desc, &outs[0].desc, 1, &axis};
        create(DML_OPERATOR_ACTIVATION_SOFTMAX1, &d); break; }
    case GEMM: {
        for (size_t i = 0; i < op.in.size(); i++) { if (op.in[i].sizes.size() > 4) throw std::runtime_error("gemm rank>4"); plain(ins[i], op.in[i], 4); }
        if (op.out[0].sizes.size() > 4) throw std::runtime_error("gemm out rank>4");
        plain(outs[0], op.out[0], 4);
        DML_GEMM_OPERATOR_DESC d{&ins[0].desc, &ins[1].desc, op.in.size() > 2 ? &ins[2].desc : nullptr, &outs[0].desc, DML_MATRIX_TRANSFORM_NONE, DML_MATRIX_TRANSFORM_NONE, float(op.a("alpha", 1)), float(op.a("beta", 0)), nullptr};
        flags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
        if (g_no_metacommands) flags = DML_EXECUTION_FLAGS(flags | DML_EXECUTION_FLAG_DISABLE_META_COMMANDS);
        create(DML_OPERATOR_GEMM, &d); break; }
    case RMSNORM: {
        // input [rows,C] -> [1,1,rows,C]; scale [C] broadcast with stride 0
        auto s = op.in[0].sizes, st = op.in[0].strides; pad_to(s, st, 4); ins[0].set(op.in[0], s, st);
        auto o = op.out[0].sizes, ost = op.out[0].strides; pad_to(o, ost, 4); outs[0].set(op.out[0], o, ost);
        std::vector<int64_t> ss = s, sst = {0, 0, 0, 1}; ins[1].set(op.in[1], ss, sst);
        UINT axis = 3;
        DML_MEAN_VARIANCE_NORMALIZATION2_OPERATOR_DESC d{&ins[0].desc, &ins[1].desc, nullptr, &outs[0].desc, 1, &axis, FALSE, TRUE, float(op.a("eps")), nullptr};
        create(DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2, &d); break; }
    case SLICE: {
        plain(ins[0], op.in[0], 4); plain(outs[0], op.out[0], 4);
        size_t shift = ins[0].sizes.size() - op.in[0].sizes.size();
        auto offs = op.v("offsets"), steps = op.v("steps");
        std::vector<UINT> o(ins[0].sizes.size(), 0), w(ins[0].sizes.size(), 1); std::vector<INT> sp(ins[0].sizes.size(), 1);
        for (size_t i = 0; i < op.in[0].sizes.size(); i++) {
            o[i + shift] = UINT(offs[i]); sp[i + shift] = INT(steps[i]);
            w[i + shift] = UINT((op.out[0].sizes[i] - 1) * steps[i] + 1);
        }
        DML_SLICE1_OPERATOR_DESC d{&ins[0].desc, &outs[0].desc, UINT(o.size()), o.data(), w.data(), sp.data()};
        create(DML_OPERATOR_SLICE1, &d); break; }
    case JOIN: {
        size_t rank = std::max<size_t>(4, op.out[0].sizes.size());
        for (size_t i = 0; i < op.in.size(); i++) plain(ins[i], op.in[i], rank);
        plain(outs[0], op.out[0], rank);
        std::vector<DML_TENSOR_DESC> descs; for (auto& t : ins) descs.push_back(t.desc);
        DML_JOIN_OPERATOR_DESC d{UINT(descs.size()), descs.data(), &outs[0].desc, UINT(op.a("axis") + double(rank - op.out[0].sizes.size()))};
        create(DML_OPERATOR_JOIN, &d); break; }
    case PAD: {
        plain(ins[0], op.in[0], 4); plain(outs[0], op.out[0], 4);
        size_t shift = ins[0].sizes.size() - op.in[0].sizes.size();
        std::vector<UINT> a(ins[0].sizes.size(), 0), b(ins[0].sizes.size(), 0);
        auto st = op.v("start"), en = op.v("end");
        for (size_t i = 0; i < st.size(); i++) { a[i + shift] = UINT(st[i]); b[i + shift] = UINT(en[i]); }
        int mode = int(op.a("mode"));
        DML_PADDING_OPERATOR_DESC d{&ins[0].desc, &outs[0].desc, mode == 0 ? DML_PADDING_MODE_CONSTANT : mode == 1 ? DML_PADDING_MODE_REFLECTION : DML_PADDING_MODE_EDGE,
                                    float(op.a("value")), UINT(a.size()), a.data(), b.data()};
        create(DML_OPERATOR_PADDING, &d); break; }
    case CONV: {
        for (size_t i = 0; i < op.in.size(); i++) plain(ins[i], op.in[i], 4);
        plain(outs[0], op.out[0], 4);
        auto s = op.v("strides"), dl = op.v("dilations"), a = op.v("start"), b = op.v("end"), op_ = op.v("out_pad");
        std::array<UINT, 2> S{UINT(s[0]), UINT(s[1])}, D{UINT(dl[0]), UINT(dl[1])}, A{UINT(a[0]), UINT(a[1])}, B{UINT(b[0]), UINT(b[1])}, O{UINT(op_[0]), UINT(op_[1])};
        bool tr = op.a("transpose") != 0;
        DML_CONVOLUTION_OPERATOR_DESC d{&ins[0].desc, &ins[1].desc, op.in.size() > 2 ? &ins[2].desc : nullptr, &outs[0].desc,
                                        DML_CONVOLUTION_MODE_CROSS_CORRELATION, tr ? DML_CONVOLUTION_DIRECTION_BACKWARD : DML_CONVOLUTION_DIRECTION_FORWARD,
                                        2, S.data(), D.data(), A.data(), B.data(), O.data(), UINT(op.a("groups", 1)), nullptr};
        flags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
        if (g_no_metacommands) flags = DML_EXECUTION_FLAGS(flags | DML_EXECUTION_FLAG_DISABLE_META_COMMANDS);
        create(DML_OPERATOR_CONVOLUTION, &d); break; }
    case D2S: {
        plain(ins[0], op.in[0], 4); plain(outs[0], op.out[0], 4);
        DML_DEPTH_TO_SPACE1_OPERATOR_DESC d{&ins[0].desc, &outs[0].desc, UINT(op.a("block")),
                                            int(op.a("mode")) == 1 ? DML_DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH : DML_DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW};
        create(DML_OPERATOR_DEPTH_TO_SPACE1, &d); break; }
    case FILL: {
        plain(outs[0], op.out[0], 4);
        DML_SCALAR_UNION v{}; DML_TENSOR_DATA_TYPE t = dmltype(op.out[0].dtype);
        if (t == DML_TENSOR_DATA_TYPE_FLOAT32) v.Float32 = float(op.a("value"));
        else if (t == DML_TENSOR_DATA_TYPE_FLOAT16) { if (op.a("value") != 0) throw std::runtime_error("only zero FP16 fill"); v.UInt16 = 0; }
        else throw std::runtime_error("fill dtype");
        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC d{&outs[0].desc, t, v};
        create(DML_OPERATOR_FILL_VALUE_CONSTANT, &d); break; }
    case MHA: {  // DirectML fused attention, Q/K/V/out [B, seq, heads*dim]
        for (size_t i = 0; i < 3; i++) plain(ins[i], op.in[i], 3);
        plain(outs[0], op.out[0], 3);
        DML_MULTIHEAD_ATTENTION_OPERATOR_DESC d{};
        d.QueryTensor = &ins[0].desc; d.KeyTensor = &ins[1].desc; d.ValueTensor = &ins[2].desc; d.OutputTensor = &outs[0].desc;
        d.Scale = float(op.a("scale")); d.MaskFilterValue = -10000.f; d.HeadCount = UINT(op.a("heads")); d.MaskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
        if (op.in.size() > 3) { plain(ins[3], op.in[3], 4); d.MaskTensor = &ins[3].desc; d.MaskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_BOOLEAN; }
        flags = DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION;
        create(DML_OPERATOR_MULTIHEAD_ATTENTION, &d); break; }
    default: throw std::runtime_error("unknown op kind");
    }
    if (g_no_metacommands) flags = DML_EXECUTION_FLAGS(flags | DML_EXECUTION_FLAG_DISABLE_META_COMMANDS);  // vendor-neutral shaders only
    HRESULT hr = h.dml->CompileOperator(dop, flags, IID_DML_CompiledOperator, reinterpret_cast<void**>(&c->op));
    dop->Release();
    if (FAILED(hr)) { char s[160]; std::snprintf(s, sizeof s, "CompileOperator op %zu kind %d: 0x%08lx", index, op.kind, (unsigned long)hr); throw std::runtime_error(s); }
    c->props = properties(c->op);
    if (std::getenv("CK_DEBUG")) std::fprintf(stderr, "op %zu kind %d desc %u temp %llu persist %llu dml_removed 0x%08lx\n", index, op.kind, c->props.RequiredDescriptorCount, (unsigned long long)c->props.TemporaryResourceSize, (unsigned long long)c->props.PersistentResourceSize, (unsigned long)h.dml->GetDeviceRemovedReason());
    scratch_bytes = std::max(scratch_bytes, c->props.TemporaryResourceSize);
    descriptors += c->props.RequiredDescriptorCount;
    if (c->props.PersistentResourceSize) {
        create_buffer(h.device, c->props.PersistentResourceSize, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_HEAP_TYPE_DEFAULT, &c->persistent);
        h.buffer_bytes_total += c->persistent.size;
    }
    for (size_t i = 0; i < op.in.size(); i++) c->in.push_back(binding(op.in[i], ins[i]));
    for (size_t i = 0; i < op.out.size(); i++) c->out.push_back(binding(op.out[i], outs[i]));
    c->input_slots = op.kind == MHA ? 11 : (op.kind == GEMM || op.kind == CONV || op.kind == RMSNORM) ? 3 : op.in.size();  // optional inputs bound as NONE
    c->output_slots = op.kind == MHA ? 3 : op.out.size();
    ops.push_back(std::move(c));
}

static void init_gpu_exec(Harness& h, int adapter_id, const std::string& dll, std::string& name) {
    h.dxgi_library = LoadLibraryW(L"dxgi.dll"); h.d3d12_library = LoadLibraryW(L"d3d12.dll"); h.dml_library = load_library_utf8(dll);
    if (!h.dxgi_library || !h.d3d12_library || !h.dml_library) throw std::runtime_error("runtime DLL load failed");
    using Factory = HRESULT(WINAPI*)(UINT, REFIID, void**); using Device = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
    using ML = HRESULT(STDAPICALLTYPE*)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, REFIID, void**);
    auto pf = reinterpret_cast<Factory>(GetProcAddress(h.dxgi_library, "CreateDXGIFactory2"));
    auto pd = reinterpret_cast<Device>(GetProcAddress(h.d3d12_library, "D3D12CreateDevice"));
    auto pm = reinterpret_cast<ML>(GetProcAddress(h.dml_library, "DMLCreateDevice"));
    if (!pf || !pd || !pm) throw std::runtime_error("missing device exports");
    ScopedCom<IDXGIFactory2> factory; ck(pf(0, IID_DXGI_Factory2, reinterpret_cast<void**>(&factory.value)), "CreateDXGIFactory2");
    ScopedCom<IDXGIAdapter1> adapter;
    if (adapter_id < 0) {  // WARP: Microsoft software D3D12 device (vendor-neutral correctness check)
        static const GUID IID_DXGI_Factory4 = {0x1bc6ea02, 0xef36, 0x464f, {0xbf, 0x0c, 0x21, 0xca, 0x39, 0xe5, 0x16, 0x8a}};
        ScopedCom<IDXGIFactory4> f4; ck(factory.value->QueryInterface(IID_DXGI_Factory4, reinterpret_cast<void**>(&f4.value)), "QueryInterface Factory4");
        static const GUID IID_DXGI_Adapter1 = {0x29038f61, 0x3839, 0x4626, {0x91, 0xfd, 0x08, 0x68, 0x79, 0x01, 0x1a, 0x05}};
        ck(f4.value->EnumWarpAdapter(IID_DXGI_Adapter1, reinterpret_cast<void**>(&adapter.value)), "EnumWarpAdapter");
    } else ck(factory.value->EnumAdapters1(UINT(adapter_id), &adapter.value), "EnumAdapters1");
    DXGI_ADAPTER_DESC1 d{}; ck(adapter.value->GetDesc1(&d), "GetDesc1");
    if ((d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && adapter_id >= 0) throw std::runtime_error("software adapter prohibited");
    name = narrow(d.Description);
    ck(pd(adapter.value, D3D_FEATURE_LEVEL_11_0, IID_D3D12_Device, reinterpret_cast<void**>(&h.device)), "D3D12CreateDevice");
    ck(pm(h.device, DML_CREATE_DEVICE_FLAG_NONE, IID_DML_Device, reinterpret_cast<void**>(&h.dml)), "DMLCreateDevice");
    D3D12_COMMAND_QUEUE_DESC q{}; q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ck(h.device->CreateCommandQueue(&q, IID_D3D12_Queue, reinterpret_cast<void**>(&h.queue)), "CreateCommandQueue");
    ck(h.device->CreateCommandAllocator(q.Type, IID_D3D12_Allocator, reinterpret_cast<void**>(&h.allocator)), "CreateCommandAllocator");
    ck(h.device->CreateCommandList(0, q.Type, h.allocator, nullptr, IID_D3D12_CommandList, reinterpret_cast<void**>(&h.cmdlist)), "CreateCommandList");
    ck(h.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_D3D12_Fence, reinterpret_cast<void**>(&h.fence)), "CreateFence");
    h.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    D3D12_QUERY_HEAP_DESC qh{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 2, 0};
    ck(h.device->CreateQueryHeap(&qh, IID_D3D12_QueryHeap, reinterpret_cast<void**>(&h.qheap)), "CreateQueryHeap");
    ck(h.queue->GetTimestampFrequency(&h.ts_freq), "GetTimestampFrequency");
    ck(h.dml->CreateCommandRecorder(IID_DML_CommandRecorder, reinterpret_cast<void**>(&h.recorder)), "CreateCommandRecorder");
}
static void submit_wait(Harness& h) {
    ck(h.cmdlist->Close(), "Close");
    ID3D12CommandList* lists[] = {h.cmdlist}; h.queue->ExecuteCommandLists(1, lists);
    ck(h.queue->Signal(h.fence, ++h.fence_val), "Signal");
    if (h.fence->GetCompletedValue() < h.fence_val) {
        ck(h.fence->SetEventOnCompletion(h.fence_val, h.event), "SetEventOnCompletion");
        if (WaitForSingleObject(h.event, 120000) != WAIT_OBJECT_0) throw std::runtime_error("GPU fence timeout");
    }
    ck(h.device->GetDeviceRemovedReason(), "device status");
}
static void reset_list(Harness& h) { ck(h.allocator->Reset(), "allocator Reset"); ck(h.cmdlist->Reset(h.allocator, nullptr), "list Reset"); }
static void uav_all(Harness& h) { D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV; h.cmdlist->ResourceBarrier(1, &b); }
static void barrier(Harness& h, ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x{}; x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition = {r, 0, a, b}; h.cmdlist->ResourceBarrier(1, &x);
}
// copy host bytes into a DEFAULT buffer region through a bounded upload ring
static void upload(Harness& h, GpuBuffer& dst, uint64_t offset, const uint8_t* data, uint64_t bytes) {
    const uint64_t chunk = 256ull << 20;
    GpuBuffer staging; create_buffer(h.device, std::min(chunk, bytes), D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_HEAP_TYPE_UPLOAD, &staging);
    void* p = nullptr; D3D12_RANGE rr{0, 0}; ck(staging.res->Map(0, &rr, &p), "Map upload");
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = std::min(chunk, bytes - done);
        std::memcpy(p, data + done, n);
        barrier(h, dst.res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        h.cmdlist->CopyBufferRegion(dst.res, offset + done, staging.res, 0, n);
        barrier(h, dst.res, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        submit_wait(h); reset_list(h);
    }
    staging.res->Unmap(0, nullptr);
}
static std::vector<uint8_t> download(Harness& h, GpuBuffer& src, uint64_t offset, uint64_t bytes) {
    GpuBuffer rb; create_buffer(h.device, bytes, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK, &rb);
    barrier(h, src.res, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    h.cmdlist->CopyBufferRegion(rb.res, 0, src.res, offset, bytes);
    barrier(h, src.res, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    submit_wait(h); reset_list(h);
    void* p = nullptr; D3D12_RANGE rr{0, size_t(bytes)}; ck(rb.res->Map(0, &rr, &p), "Map readback");
    std::vector<uint8_t> out(bytes); std::memcpy(out.data(), p, bytes); D3D12_RANGE none{0, 0}; rb.res->Unmap(0, &none);
    return out;
}
// minimal .npy reader (float32 little-endian, C order)
static std::vector<uint8_t> read_npy_f32(const std::string& path, uint64_t expect_bytes) {
    auto bytes = read_file(path);
    if (bytes.size() < 10 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0) throw std::runtime_error("not npy: " + path);
    uint32_t hlen = bytes[6] == 1 ? (bytes[8] | (bytes[9] << 8)) : (bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (uint32_t(bytes[11]) << 24));
    size_t off = bytes[6] == 1 ? 10 + hlen : 12 + hlen;
    std::string header(reinterpret_cast<char*>(bytes.data()) + (bytes[6] == 1 ? 10 : 12), hlen);
    if (header.find("<f4") == std::string::npos || header.find("'fortran_order': False") == std::string::npos) throw std::runtime_error("npy must be <f4 C-order: " + path);
    if (bytes.size() - off != expect_bytes) throw std::runtime_error("npy size mismatch: " + path);
    return std::vector<uint8_t>(bytes.begin() + long(off), bytes.end());
}
static uint64_t process_dedicated_bytes() { return 0; }

// ---------------------------------------------------------------- session (shared by CLI and DLL)
struct Session {
    Exec e; std::string adapter_name; uint32_t total_desc = 0; double compile_ms = 0, bind_ms = 0;
    GpuBuffer ts;
    ~Session() {
        e.ops.clear(); com_release(e.dheap);
        Harness& h = e.h;
        com_release(h.recorder); com_release(h.qheap); com_release(h.fence); com_release(h.cmdlist);
        com_release(h.allocator); com_release(h.queue);
        // buffers must go before the device
        com_release(ts.res); com_release(e.scratch.res); com_release(e.weights.res); com_release(e.heap.res);
        com_release(h.dml); com_release(h.device);
        if (h.event) CloseHandle(h.event);
    }
};

static void setup(Session& s, const std::string& plan_dir, const std::string& dll, int adapter, size_t limit) {
    Exec& e = s.e; e.plan = load_plan(plan_dir + "/plan.txt"); split_heavy_ops(e.plan);
    init_gpu_exec(e.h, adapter, dll, s.adapter_name);
    auto t0 = std::chrono::steady_clock::now();
    create_buffer(e.h.device, e.plan.heap + 4096, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_HEAP_TYPE_DEFAULT, &e.heap);
    create_buffer(e.h.device, e.plan.weights, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_HEAP_TYPE_DEFAULT, &e.weights);
    {   // weights, streamed in chunks
        std::ifstream wf(plan_dir + "/weights.bin", std::ios::binary); if (!wf) throw std::runtime_error("weights.bin");
        std::vector<uint8_t> buf(64ull << 20); uint64_t done = 0;
        while (done < e.plan.weights) {
            uint64_t n = std::min<uint64_t>(buf.size(), e.plan.weights - done);
            wf.read(reinterpret_cast<char*>(buf.data()), std::streamsize(n)); if (!wf) throw std::runtime_error("weights short");
            upload(e.h, e.weights, done, buf.data(), n); done += n;
        }
    }
    if (limit < e.plan.ops.size()) e.plan.ops.resize(limit);
    for (size_t i = 0; i < e.plan.ops.size(); i++) e.compile(e.plan.ops[i], i);
    auto t_compiled = std::chrono::steady_clock::now();
    // Operator initialization (persistent resources). One initializer for all ops can ask for a
    // multi-GiB temporary (Stage1 1080p T8: 6.7 GB), so ops are initialized in batches whose
    // temporary stays under a cap, reusing one buffer and one descriptor region.
    const uint64_t cap = std::getenv("CK_INIT_TEMP_CAP") ? std::stoull(std::getenv("CK_INIT_TEMP_CAP")) : (512ull << 20);
    IDMLOperatorInitializer* init = nullptr;
    ck(e.h.dml->CreateOperatorInitializer(0, nullptr, IID_Exec_Initializer, reinterpret_cast<void**>(&init)), "CreateOperatorInitializer");
    struct Batch { size_t begin, end; DML_BINDING_PROPERTIES props; };
    std::vector<Batch> batches;
    {
        std::vector<uint64_t> temp(e.ops.size());
        for (size_t i = 0; i < e.ops.size(); i++) {
            IDMLCompiledOperator* one = e.ops[i]->op;
            ck(init->Reset(1, &one), "initializer Reset (probe)");
            temp[i] = properties(init).TemporaryResourceSize;
        }
        size_t i = 0;
        while (i < e.ops.size()) {
            size_t j = i; uint64_t sum = 0;
            while (j < e.ops.size() && (j == i || sum + temp[j] <= cap)) sum += temp[j++];
            std::vector<IDMLCompiledOperator*> ops; for (size_t k = i; k < j; k++) ops.push_back(e.ops[k]->op);
            ck(init->Reset(UINT(ops.size()), ops.data()), "initializer Reset");
            batches.push_back({i, j, properties(init)});
            i = j;
        }
    }
    uint64_t init_temp_bytes = 0; uint32_t init_desc = 0;
    for (auto& bt : batches) { init_temp_bytes = std::max<uint64_t>(init_temp_bytes, bt.props.TemporaryResourceSize); init_desc = std::max(init_desc, bt.props.RequiredDescriptorCount); }
    s.total_desc = e.descriptors + init_desc;
    create_buffer(e.h.device, std::max<uint64_t>(e.scratch_bytes, 256), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_HEAP_TYPE_DEFAULT, &e.scratch);
    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, std::max(1u, s.total_desc), D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ck(e.h.device->CreateDescriptorHeap(&hd, IID_D3D12_DescriptorHeap, reinterpret_cast<void**>(&e.dheap)), "CreateDescriptorHeap");
    auto cpu0 = member_struct<D3D12_CPU_DESCRIPTOR_HANDLE>(e.dheap, 9); auto gpu0 = member_struct<D3D12_GPU_DESCRIPTOR_HANDLE>(e.dheap, 10);
    UINT inc = e.h.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV), cursor = 0;
    auto make_table = [&](IDMLDispatchable* d, UINT count) {
        DML_BINDING_TABLE_DESC td{d, {cpu0.ptr + uint64_t(cursor) * inc}, {gpu0.ptr + uint64_t(cursor) * inc}, count};
        IDMLBindingTable* t = nullptr; HRESULT hr = e.h.dml->CreateBindingTable(&td, IID_DML_BindingTable, reinterpret_cast<void**>(&t));
        if (FAILED(hr)) { char m[200]; std::snprintf(m, sizeof m, "CreateBindingTable at cursor %u count %u: 0x%08lx d3d 0x%08lx dml 0x%08lx", cursor, count, (unsigned long)hr, (unsigned long)e.h.device->GetDeviceRemovedReason(), (unsigned long)e.h.dml->GetDeviceRemovedReason()); throw std::runtime_error(m); }
        cursor += count; return t;
    };
    DML_BUFFER_BINDING scratch_b{e.scratch.res, 0, e.scratch.size};
    DML_BINDING_DESC scratch_d{DML_BINDING_TYPE_BUFFER, &scratch_b};
    {   // initialize batch by batch; the shared temporary is released right after
        GpuBuffer init_temp;
        if (init_temp_bytes) create_buffer(e.h.device, init_temp_bytes, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_HEAP_TYPE_DEFAULT, &init_temp);
        for (auto& bt : batches) {
            std::vector<IDMLCompiledOperator*> ops; for (size_t k = bt.begin; k < bt.end; k++) ops.push_back(e.ops[k]->op);
            ck(init->Reset(UINT(ops.size()), ops.data()), "initializer Reset");
            cursor = 0;  // batches reuse the initializer descriptor region at the heap start
            IDMLBindingTable* it = make_table(init, std::max(1u, bt.props.RequiredDescriptorCount));
            std::vector<DML_BUFFER_BINDING> pb(ops.size()); std::vector<DML_BINDING_DESC> pd(ops.size());
            for (size_t k = 0; k < ops.size(); k++) {
                auto& c = e.ops[bt.begin + k];
                if (c->props.PersistentResourceSize) { pb[k] = {c->persistent.res, 0, c->persistent.size}; pd[k] = {DML_BINDING_TYPE_BUFFER, &pb[k]}; }
                else pd[k] = {DML_BINDING_TYPE_NONE, nullptr};
            }
            it->BindOutputs(UINT(pd.size()), pd.data());
            DML_BUFFER_BINDING itb{}; DML_BINDING_DESC itd{};
            if (bt.props.TemporaryResourceSize) { itb = {init_temp.res, 0, init_temp.size}; itd = {DML_BINDING_TYPE_BUFFER, &itb}; it->BindTemporaryResource(&itd); }
            e.h.cmdlist->SetDescriptorHeaps(1, &e.dheap);
            e.h.recorder->RecordDispatch(e.h.cmdlist, init, it); uav_all(e.h);
            submit_wait(e.h); reset_list(e.h); it->Release();
        }
        init->Release();
        if (std::getenv("CK_DEBUG")) { std::fprintf(stderr, "initializer batches %zu, temp %llu (cap %llu), op scratch %llu, descriptors %u\n", batches.size(), (unsigned long long)init_temp_bytes, (unsigned long long)cap, (unsigned long long)e.scratch_bytes, s.total_desc); Sleep(1500); ck(e.h.device->GetDeviceRemovedReason(), "device after initializer"); }
    }
    cursor = init_desc;  // op tables follow the initializer region
    for (auto& c : e.ops) {
        c->table = make_table(c->op, c->props.RequiredDescriptorCount);
        std::vector<DML_BINDING_DESC> in; for (auto& b : c->in) in.push_back({DML_BINDING_TYPE_BUFFER, &b});
        while (in.size() < c->input_slots) in.push_back({DML_BINDING_TYPE_NONE, nullptr});
        if (c->input_slots == 11 && c->in.size() > 3) { in[7] = in[3]; in[3] = {DML_BINDING_TYPE_NONE, nullptr}; }  // MHA mask slot
        std::vector<DML_BINDING_DESC> out; for (auto& b : c->out) out.push_back({DML_BINDING_TYPE_BUFFER, &b});
        while (out.size() < c->output_slots) out.push_back({DML_BINDING_TYPE_NONE, nullptr});
        c->table->BindInputs(UINT(in.size()), in.data()); c->table->BindOutputs(UINT(out.size()), out.data());
        if (c->props.TemporaryResourceSize) c->table->BindTemporaryResource(&scratch_d);
        if (c->props.PersistentResourceSize) { DML_BUFFER_BINDING pb{c->persistent.res, 0, c->persistent.size}; DML_BINDING_DESC pd{DML_BINDING_TYPE_BUFFER, &pb}; c->table->BindPersistentResource(&pd); }
    }
    for (auto& in : e.plan.inputs) if (in.dtype != "f32") throw std::runtime_error("only f32 graph inputs");
    for (auto& o : e.plan.outputs) if (o.dtype != "f32") throw std::runtime_error("only f32 graph outputs");
    create_buffer(e.h.device, 16, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK, &s.ts);
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    s.compile_ms = ms(t0, t_compiled); s.bind_ms = ms(t_compiled, std::chrono::steady_clock::now());
}

static uint64_t io_elements(const IoSpec& io) { uint64_t n = 1; for (auto d : io.shape) n *= uint64_t(d); return n; }

// One forward pass: host f32 inputs (plan order) -> host f32 outputs (plan order). Returns GPU ms.
static double run(Session& s, const float* const* inputs, float* const* outputs) {
    Exec& e = s.e;
    for (size_t i = 0; i < e.plan.inputs.size(); i++)
        upload(e.h, e.heap, e.plan.inputs[i].base, reinterpret_cast<const uint8_t*>(inputs[i]), io_elements(e.plan.inputs[i]) * 4);
    // Windows resets the GPU when one submission runs for ~2 s (TDR). Submit the operators in
    // chunks of ~0.2 s, sized from the previous chunk's time: slow (integrated) GPUs get many
    // short submissions, fast GPUs grow to one submission per pass within the first run.
    const size_t n = e.ops.size();
    for (size_t i = 0; i < n;) {
        size_t end = std::min(n, i + std::max<size_t>(1, e.chunk));
        e.h.cmdlist->SetDescriptorHeaps(1, &e.dheap);
        if (i == 0) e.h.cmdlist->EndQuery(e.h.qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
        for (size_t k = i; k < end; k++) { e.h.recorder->RecordDispatch(e.h.cmdlist, e.ops[k]->op, e.ops[k]->table); uav_all(e.h); }
        if (end == n) {
            e.h.cmdlist->EndQuery(e.h.qheap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            e.h.cmdlist->ResolveQueryData(e.h.qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, s.ts.res, 0);
        }
        auto t0 = std::chrono::steady_clock::now();
        submit_wait(e.h); reset_list(e.h);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double scale = std::clamp(0.2 / std::max(sec, 1e-4), 0.25, 4.0);
        e.chunk = std::max<size_t>(1, size_t(double(end - i) * scale));
        if (std::getenv("CK_DEBUG")) std::fprintf(stderr, "chunk %zu..%zu %.3f s next %zu\n", i, end, sec, e.chunk);
        i = end;
    }
    for (size_t i = 0; i < e.plan.outputs.size(); i++) {
        auto& o = e.plan.outputs[i]; auto bytes = download(e.h, e.heap, o.base, io_elements(o) * 4);
        std::memcpy(outputs[i], bytes.data(), bytes.size());
    }
    void* p = nullptr; D3D12_RANGE rr{0, 16}; ck(s.ts.res->Map(0, &rr, &p), "Map ts"); uint64_t t[2]; std::memcpy(t, p, 16); D3D12_RANGE none{0, 0}; s.ts.res->Unmap(0, &none);
    return double(t[1] - t[0]) * 1000.0 / double(e.h.ts_freq);
}

static void profile_ops(Session& s, const std::string& out_dir) {  // one extra pass with a timestamp after every operator
    Exec& e = s.e; UINT n = UINT(e.ops.size() + 1);
    ID3D12QueryHeap* qh = nullptr; D3D12_QUERY_HEAP_DESC qd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, n, 0};
    ck(e.h.device->CreateQueryHeap(&qd, IID_D3D12_QueryHeap, reinterpret_cast<void**>(&qh)), "CreateQueryHeap profile");
    GpuBuffer pts; create_buffer(e.h.device, uint64_t(n) * 8, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_HEAP_TYPE_READBACK, &pts);
    e.h.cmdlist->SetDescriptorHeaps(1, &e.dheap);
    e.h.cmdlist->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0);
    for (size_t i = 0; i < e.ops.size(); i++) { e.h.recorder->RecordDispatch(e.h.cmdlist, e.ops[i]->op, e.ops[i]->table); uav_all(e.h); e.h.cmdlist->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, UINT(i + 1)); }
    e.h.cmdlist->ResolveQueryData(qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, n, pts.res, 0);
    submit_wait(e.h); reset_list(e.h);
    void* p = nullptr; D3D12_RANGE rr{0, size_t(n) * 8}; ck(pts.res->Map(0, &rr, &p), "Map profile");
    std::vector<uint64_t> t(n); std::memcpy(t.data(), p, size_t(n) * 8); D3D12_RANGE none{0, 0}; pts.res->Unmap(0, &none);
    std::ofstream f(out_dir + "/op_times.txt");
    for (size_t i = 0; i < e.ops.size(); i++) f << i << ' ' << e.plan.ops[i].kind << ' ' << double(t[i + 1] - t[i]) * 1000.0 / double(e.h.ts_freq) << '\n';
    qh->Release();
}

#ifdef CK_EXECUTOR_DLL
// ---------------------------------------------------------------- C ABI (ck_plan_executor.dll)
// All calls are serialised by the caller (one GPU owner). Errors return nonzero and copy a
// message into err. Tensors are host f32 in plan order; shapes come from ck_exec_io.
#define CK_API extern "C" __declspec(dllexport)
static void copy_err(const char* m, char* err, size_t n) { if (err && n) { std::snprintf(err, n, "%s", m); } }
CK_API int ck_exec_abi_version() { return 1; }
// Optional (added after ABI 1 shipped): applies to sessions created after the call.
CK_API void ck_exec_set_disable_metacommands(int disable) { g_no_metacommands = disable != 0; }
CK_API int ck_exec_create(const char* plan_dir, const char* directml_dll, int adapter, void** out, char* err, size_t err_len) {
    *out = nullptr;
    try { auto s = std::make_unique<Session>(); setup(*s, plan_dir, directml_dll, adapter, SIZE_MAX); *out = s.release(); return 0; }
    catch (const std::exception& ex) { copy_err(ex.what(), err, err_len); return 1; }
}
CK_API int ck_exec_io_count(void* h, int is_output) {
    auto* s = static_cast<Session*>(h); return int(is_output ? s->e.plan.outputs.size() : s->e.plan.inputs.size());
}
// name copied (NUL-terminated) into name/name_len; up to 8 dims into shape; returns rank or -1.
CK_API int ck_exec_io(void* h, int is_output, int index, char* name, size_t name_len, int64_t* shape) {
    auto* s = static_cast<Session*>(h); auto& list = is_output ? s->e.plan.outputs : s->e.plan.inputs;
    if (index < 0 || size_t(index) >= list.size() || list[size_t(index)].shape.size() > 8) return -1;
    auto& io = list[size_t(index)]; copy_err(io.name.c_str(), name, name_len);
    for (size_t i = 0; i < io.shape.size(); i++) shape[i] = io.shape[i];
    return int(io.shape.size());
}
CK_API int ck_exec_stats(void* h, uint64_t* device_bytes, double* compile_ms, double* bind_ms) {
    auto* s = static_cast<Session*>(h); uint64_t persistent = 0; for (auto& c : s->e.ops) persistent += c->persistent.size;
    *device_bytes = s->e.heap.size + s->e.weights.size + s->e.scratch.size + persistent; *compile_ms = s->compile_ms; *bind_ms = s->bind_ms; return 0;
}
CK_API int ck_exec_run(void* h, const float* const* inputs, float* const* outputs, double* gpu_ms, char* err, size_t err_len) {
    try { double g = run(*static_cast<Session*>(h), inputs, outputs); if (gpu_ms) *gpu_ms = g; return 0; }
    catch (const std::exception& ex) { copy_err(ex.what(), err, err_len); return 1; }
}
CK_API void ck_exec_destroy(void* h) { delete static_cast<Session*>(h); }
#else
// ---------------------------------------------------------------- CLI
int main(int argc, char** argv) {
    std::string plan_dir, data_dir, dll, out_dir = "."; int adapter = 0, reps = 3; size_t limit = SIZE_MAX;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--plan") plan_dir = v; else if (k == "--data") data_dir = v; else if (k == "--directml") dll = v;
        else if (k == "--adapter") adapter = std::stoi(v); else if (k == "--reps") reps = std::stoi(v); else if (k == "--out") out_dir = v;
        else if (k == "--limit") limit = std::stoul(v);
        else { std::fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    g_out_json = out_dir + "/executor_result.json";
    try {
        auto s = std::make_unique<Session>(); setup(*s, plan_dir, dll, adapter, limit);
        Exec& e = s->e;
        std::fprintf(stderr, "compiled %zu ops, scratch %llu, descriptors %u\n", e.ops.size(), (unsigned long long)e.scratch_bytes, e.descriptors);
        std::vector<std::vector<uint8_t>> inputs; std::vector<const float*> in_ptr;
        for (auto& in : e.plan.inputs) inputs.push_back(read_npy_f32(data_dir + "/inputs/" + in.name + ".npy", io_elements(in) * 4));
        for (auto& v : inputs) in_ptr.push_back(reinterpret_cast<const float*>(v.data()));
        std::vector<std::vector<float>> outputs; std::vector<float*> out_ptr;
        for (auto& o : e.plan.outputs) outputs.emplace_back(io_elements(o));
        for (auto& v : outputs) out_ptr.push_back(v.data());
        std::vector<double> wall_ms, gpu_ms;
        for (int r = 0; r < reps; r++) {
            auto a = std::chrono::steady_clock::now();
            gpu_ms.push_back(run(*s, in_ptr.data(), out_ptr.data()));
            wall_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
        }
        if (std::getenv("CK_PROFILE")) profile_ops(*s, out_dir);
        for (size_t i = 0; i < outputs.size(); i++) {
            std::vector<uint8_t> b(outputs[i].size() * 4); std::memcpy(b.data(), outputs[i].data(), b.size());
            write_file(out_dir + "/out_" + e.plan.outputs[i].name + ".f32", b);
        }
        uint64_t persistent = 0; for (auto& c : e.ops) persistent += c->persistent.size;
        std::ostringstream j; j.precision(10);
        j << "{\"status\":\"ok\",\"adapter\":\"" << s->adapter_name << "\",\"ops\":" << e.ops.size() << ",\"heap_bytes\":" << e.heap.size
          << ",\"weights_bytes\":" << e.weights.size << ",\"scratch_bytes\":" << e.scratch.size << ",\"persistent_bytes\":" << persistent
          << ",\"descriptors\":" << s->total_desc << ",\"setup_compile_ms\":" << s->compile_ms << ",\"setup_bind_ms\":" << s->bind_ms
          << ",\"gpu_ms\":["; for (size_t i = 0; i < gpu_ms.size(); i++) j << (i ? "," : "") << gpu_ms[i];
        j << "],\"wall_ms_incl_io\":["; for (size_t i = 0; i < wall_ms.size(); i++) j << (i ? "," : "") << wall_ms[i]; j << "]}";
        emit_json(j.str());
        return 0;
    } catch (const std::exception& ex) { fail("executor", ex.what(), 1); }
}
#endif
