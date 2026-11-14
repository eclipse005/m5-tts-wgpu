//! WGSL kernels. Each entry point is one dispatch; host wrappers live in
//! `crate::ops`.

/// `y += x`, in place (`y` is read_write, `x` read-only).
pub const RESIDUAL_F32: &str = r#"
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32, 1>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= p[0]) { return; }
    y[i] = y[i] + x[i];
}
"#;

/// `y[m,n] = sum_k x[m,k] * w[n,k] + bias[n]` — one workgroup per 16x16 tile.
///
/// Weight layout is the checkpoint's own `[out_features, in_features]`, so no
/// transpose is needed at load time; both the 1-token decode GEMV and the
/// 100-token prefills go through this kernel.
pub const LINEAR_F32: &str = r#"
// params[0]=m params[1]=n params[2]=k params[3]=has_bias
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 4>;

const T: u32 = 16u;
var<workgroup> xs: array<array<f32, 16>, 16>;
var<workgroup> ws: array<array<f32, 16>, 16>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let row0 = wid.x * T;
    let col0 = wid.y * T;
    let ty = li / T;
    let tx = li % T;
    var acc = 0.0;
    let steps = (p[2] + T - 1u) / T;
    for (var t = 0u; t < steps; t = t + 1u) {
        let kk = t * T + tx;
        let xr = row0 + ty;
        let wr = col0 + ty;
        xs[ty][tx] = 0.0;
        if (xr < p[0] && kk < p[2]) { xs[ty][tx] = x[xr * p[2] + kk]; }
        ws[ty][tx] = 0.0;
        if (wr < p[1] && kk < p[2]) { ws[ty][tx] = w[wr * p[2] + kk]; }
        workgroupBarrier();
        for (var i = 0u; i < T; i = i + 1u) {
            acc = acc + xs[ty][i] * ws[tx][i];
        }
        workgroupBarrier();
    }
    let out_r = row0 + ty;
    let out_c = col0 + tx;
    if (out_r < p[0] && out_c < p[1]) {
        var v = acc;
        if (p[3] == 1u) { v = v + b[out_c]; }
        y[out_r * p[1] + out_c] = v;
    }
}
"#;

/// Causal conv1d in `[channels, time]` layout, one thread per output element.
///
/// `x[ci, t*stride + k*dilation - pad]`, out-of-range taps read 0. This is the
/// straightforward form: correct first, then tile it once the pipeline is whole.
pub const CONV1D_F32: &str = r#"
// 0=cin 1=cout 2=ks 3=dilation 4=stride 5=pad 6=t_in 7=t_out
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 8>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 128u;
    if (idx >= p[1] * p[7]) { return; }
    let co = idx / p[7];
    let t = idx % p[7];
    var acc = 0.0;
    for (var ci = 0u; ci < p[0]; ci = ci + 1u) {
        for (var k = 0u; k < p[2]; k = k + 1u) {
            let src = i32(t * p[4] + k * p[3]) - i32(p[5]);
            if (src >= 0 && src < i32(p[6])) {
                let wv = w[(co * p[0] + ci) * p[2] + k];
                acc = acc + x[ci * p[6] + u32(src)] * wv;
            }
        }
    }
    y[co * p[7] + t] = acc + b[co];
}
"#;

/// `x + sin(exp(alpha) * x)^2 / (exp(beta) + 1e-9)` — SnakeBeta, in place.
pub const SNAKE_BETA_F32: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32, 2>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 128u;
    if (idx >= p[0] * p[1]) { return; }
    let c = idx / p[1];
    let v = x[idx];
    let a = exp(alpha[c]);
    let b = exp(beta[c]);
    x[idx] = v + (1.0 / (b + 0.000000001)) * pow(sin(v * a), 2.0);
}
"#;

/// `y += x` in place. A separate kernel because wgpu forbids binding one buffer
/// as both `read` and `read_write` inside a single dispatch, which the general
/// `ADD_F32` would need for an in-place residual.
pub const ADD_F32: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32, 1>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= p[0]) { return; }
    y[i] = a[i] + b[i];
}
"#;

/// RVQ dequantise-and-sum: `out[d,t] = sum_k table[k][code[t,k]][d] / clamp(usage, eps)`.
///
/// One thread per (dim, t). The embedding tables are pre-divided on load? No —
/// the reference divides at decode time, so we do too, bit for bit.
pub const GATHER_SUM_F32: &str = r#"
// 0=dim 1=T 2=k_start 3=k_count 4=codebook_size 5=num_codebooks
@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> tables: array<f32>;
@group(0) @binding(2) var<storage, read> usage: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 8>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 128u;
    let dim = p[0];
    let frames = p[1];
    if (idx >= dim * frames) { return; }
    let d = idx % dim;
    let t = idx / dim;
    var acc = 0.0;
    for (var k = 0u; k < p[3]; k = k + 1u) {
        let cb = p[2] + k;
        let code = codes[t * p[5] + cb];
        let slot = (cb - p[2]) * p[4] + code;
        acc = acc + tables[slot * dim + d] / max(usage[slot], 0.00001);
    }
    out[d * frames + t] = acc;
}
"#;

/// `[rows, cols] -> [cols, rows]`. Cheap here (T is small before the SEANet
/// stack) and keeps the graph honest about layouts.
pub const TRANSPOSE_F32: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32, 4>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let r = gid.x;
    let c = gid.y;
    if (r >= p[0] || c >= p[1]) { return; }
    y[c * p[0] + r] = x[r * p[1] + c];
}
"#;

/// Row-wise RMSNorm: `y = x / sqrt(mean(x^2) + eps) * w`.
pub const RMS_NORM_F32: &str = r#"
// 0=rows 1=cols 2=eps(bits) 3=0
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32, 4>;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = p[1];
    let row = wid.x;
    let lane = lid.x;
    let base = row * cols;
    var acc = 0.0;
    var c = lane;
    loop {
        if (c >= cols) { break; }
        let v = x[base + c];
        acc = acc + v * v;
        c = c + 64u;
    }
    partial[lane] = acc;
    workgroupBarrier();
    if (lane == 0u) {
        var total = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) { total = total + partial[i]; }
        let eps = bitcast<f32>(p[2]);
        partial[0] = inverseSqrt(total / f32(cols) + eps);
    }
    workgroupBarrier();
    let inv = partial[0];
    var c2 = lane;
    loop {
        if (c2 >= cols) { break; }
        y[base + c2] = x[base + c2] * inv * w[c2];
        c2 = c2 + 64u;
    }
}
"#;

/// Row-wise LayerNorm with affine: `y = (x - mean) / sqrt(var + eps) * w + b`.
pub const LAYER_NORM_F32: &str = r#"
// 0=rows 1=cols 2=eps(bits) 3=0
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 4>;
var<workgroup> partial: array<f32, 64>;
var<workgroup> stats: array<f32, 2>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = p[1];
    let base = wid.x * cols;
    let lane = lid.x;
    var sum = 0.0;
    var c = lane;
    loop {
        if (c >= cols) { break; }
        sum = sum + x[base + c];
        c = c + 64u;
    }
    partial[lane] = sum;
    workgroupBarrier();
    if (lane == 0u) {
        var total = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) { total = total + partial[i]; }
        stats[0] = total / f32(cols);
    }
    workgroupBarrier();
    let mean = stats[0];
    var sq = 0.0;
    var c2 = lane;
    loop {
        if (c2 >= cols) { break; }
        let d = x[base + c2] - mean;
        sq = sq + d * d;
        c2 = c2 + 64u;
    }
    partial[lane] = sq;
    workgroupBarrier();
    if (lane == 0u) {
        var total = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) { total = total + partial[i]; }
        let eps = bitcast<f32>(p[2]);
        stats[1] = inverseSqrt(total / f32(cols) + eps);
    }
    workgroupBarrier();
    let inv = stats[1];
    var c3 = lane;
    loop {
        if (c3 >= cols) { break; }
        y[base + c3] = (x[base + c3] - mean) * inv * w[c3] + b[c3];
        c3 = c3 + 64u;
    }
}
"#;

/// `y = silu(gate) * up`.
pub const SILU_MUL_F32: &str = r#"
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32, 1>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= p[0]) { return; }
    let g = gate[i];
    y[i] = (g / (1.0 + exp(-g))) * up[i];
}
"#;

/// `x[.., c] *= scale[c]` on a `[rows, cols]` tensor.
pub const SCALE_COLS_F32: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32, 2>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= p[0] * p[1]) { return; }
    x[i] = x[i] * scale[i % p[1]];
}
"#;

/// `y = a + scale * b` (residual with a learned per-channel gate).
pub const ADD_SCALED_F32: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 2>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= p[0] * p[1]) { return; }
    let c = i % p[1];
    y[i] = a[i] + scale[c] * b[i];
}
"#;

/// Rotary position embedding, rotate-half form, applied in place to `[T, H*D]`.
pub const ROPE_F32: &str = r#"
// 0=T 1=heads 2=head_dim 3=theta(bits)
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> p: array<u32, 4>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = p[2];
    let h = p[1];
    let half = d / 2u;
    let total = p[0] * h * half;
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= total) { return; }
    let j = i % half;              // rotary pair index
    let t = i / (h * half);        // position
    let theta = bitcast<f32>(p[3]);
    let freq = pow(theta, -2.0 * f32(j) / f32(d));
    let angle = f32(t) * freq;
    let c = cos(angle);
    let s = sin(angle);
    let base = t * h * d + (i / half % h) * d;
    let a = x[base + j];
    let b = x[base + j + half];
    x[base + j] = a * c - b * s;
    x[base + j + half] = b * c + a * s;
}
"#;

/// Causal attention with a sliding window, `[T, H*D]` in and out.
///
/// One workgroup per (query, head). Lanes split the score pass across keys and
/// each lane writes one output element. Nothing is written back to workgroup
/// memory after the barrier: every lane recomputes exp() locally, which is cheap
/// at window <= 256 and removes a genuine write/read race.
pub const ATTENTION_F32: &str = r#"
// 0=T 1=heads 2=head_dim 3=window
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 4>;
var<workgroup> scores: array<f32, 256>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let heads = p[1];
    let d = p[2];
    let window = p[3];
    let row = wid.x;
    let qpos = row / heads;
    let head = row % heads;
    let lane = lid.x;
    let start = select(0u, qpos + 1u - window, qpos + 1u > window);
    let span = qpos + 1u - start;
    let stride = heads * d;
    let qbase = qpos * stride + head * d;
    let scale = inverseSqrt(f32(d));

    var kk = lane;
    loop {
        if (kk >= span) { break; }
        let kpos = start + kk;
        let kbase = kpos * stride + head * d;
        var acc = 0.0;
        for (var i = 0u; i < d; i = i + 1u) {
            acc = acc + q[qbase + i] * k[kbase + i];
        }
        scores[kk] = acc * scale;
        kk = kk + 64u;
    }
    workgroupBarrier();

    var maxs = -1e30;
    for (var i = 0u; i < span; i = i + 1u) { maxs = max(maxs, scores[i]); }
    var sum = 0.0;
    for (var i = 0u; i < span; i = i + 1u) { sum = sum + exp(scores[i] - maxs); }
    let inv_sum = 1.0 / sum;

    if (lane < d) {
        var acc = 0.0;
        for (var j = 0u; j < span; j = j + 1u) {
            acc = acc + exp(scores[j] - maxs) * v[(start + j) * stride + head * d + lane];
        }
        out[qbase + lane] = acc * inv_sum;
    }
}
"#;

/// Depthwise causal conv1d (`groups == channels`): `y[c,t] = sum_k x[c,t+k-pad] w[c,k] + b[c]`.
pub const CONV1D_DW_F32: &str = r#"
// 0=channels 1=ks 2=pad 3=t_in 4=t_out
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 8>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 128u;
    if (idx >= p[0] * p[4]) { return; }
    let c = idx / p[4];
    let t = idx % p[4];
    var acc = 0.0;
    for (var k = 0u; k < p[1]; k = k + 1u) {
        let src = i32(t) + i32(k) - i32(p[2]);
        if (src >= 0 && src < i32(p[3])) {
            acc = acc + x[c * p[3] + u32(src)] * w[c * p[1] + k];
        }
    }
    y[c * p[4] + t] = acc + b[c];
}
"#;

/// Transposed conv1d, causal form: `y[co, t*S + k] += x[ci, t] * w[ci, co, k]`,
/// then the reference trims `kernel - stride` samples from the right.
///
/// Weight layout is PyTorch's `ConvTranspose1d`: `[in, out, kernel]`.
pub const CONV_TRANSPOSE_F32: &str = r#"
// 0=cin 1=cout 2=ks 3=stride 4=t_in 5=t_out
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 8>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 128u;
    if (idx >= p[1] * p[5]) { return; }
    let co = idx / p[5];
    let j = idx % p[5];
    var acc = 0.0;
    for (var k = 0u; k < p[2]; k = k + 1u) {
        let num = i32(j) - i32(k);
        if (num < 0) { continue; }
        if (num % i32(p[3]) != 0) { continue; }
        let t = u32(num) / p[3];
        if (t >= p[4]) { continue; }
        for (var ci = 0u; ci < p[0]; ci = ci + 1u) {
            acc = acc + x[ci * p[4] + t] * w[(ci * p[1] + co) * p[2] + k];
        }
    }
    y[co * p[5] + j] = acc + b[co];
}
"#;

/// Phased transposed conv1d for the codec's upsampling shapes (`ks == taps*stride`).
///
/// Output `t = m*stride + q` reads only input samples `m-tap` with kernel tap
/// `q + tap*stride`, so each `(co, q)` pair is an independent contiguous stream:
/// no modulo, no gather, and 32 m-accumulators per thread give the FMA pipeline
/// something to chew on (the naive kernel above is one accumulator + modulo per
/// output, measured 16-25 GFLOP/s on the P104). Weights are pre-packed on the
/// host as `[(co*stride + q)*taps*cin + tap*cin + ci]` - one cache line's worth
/// of broadcast reads per input channel per thread.
pub const CONV_TRANSPOSE_PHASED_F32: &str = r#"
// 0=cin 1=rows_total(cout*stride) 2=stride 3=t_in 4=cout 5=t_out 6=taps
@group(0) @binding(0) var<storage, read> x: array<f32>;    // [cin, t_in]
@group(0) @binding(1) var<storage, read> w2: array<f32>;   // [(tap*cin+ci)*rows + cq]
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>; // [cout, t_out]
@group(0) @binding(4) var<storage, read> p: array<u32, 7>;

const LANES: u32 = 8u;   // m per workgroup: consecutive lanes read x coalesced
const SLOTS: u32 = 32u;  // (co,q) slots per workgroup: ci-major w2 reads coalesce

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let cq = wid.x * SLOTS + lid.x / LANES;   // co*stride + q
    let m = wid.y * LANES + lid.x % LANES;
    // one accumulator per thread: a runtime-indexed local array spills to local
    // memory on this backend and cost ~3x (measured).
    var acc = 0.0;
    if (cq < p[1] && m < p[3]) {
        let co = cq / p[2];
        let q = cq % p[2];
        for (var tap = 0u; tap < p[6]; tap = tap + 1u) {
            // the m >= tap edge is thread-uniform, so the branch leaves the ci loop
            if (m >= tap) {
                for (var ci = 0u; ci < p[0]; ci = ci + 1u) {
                    acc = acc + x[ci * p[3] + m - tap] * w2[(tap * p[0] + ci) * p[1] + cq];
                }
            }
        }
        let t = m * p[2] + q;
        y[co * p[5] + t] = acc + b[co];
    }
}
"#;

/// Exact GELU (`erf` form; A&S 7.1.26 for erf, max error ~1.5e-7).
pub const GELU_F32: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> p: array<u32, 1>;

fn erf_approx(v: f32) -> f32 {
    let s = select(-1.0, 1.0, v >= 0.0);
    let ax = abs(v);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return s * y;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= p[0]) { return; }
    let v = x[i];
    x[i] = 0.5 * v * (1.0 + erf_approx(v * 0.70710678));
}
"#;

/// `y = x`, used by [`crate::device::Device::copy`].
pub const COPY_F32: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32, 1>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= p[0]) { return; }
    y[i] = x[i];
}
"#;

/// Tiled conv1d: implicit GEMM over `K = cin * ks`.
///
/// One workgroup computes an 8-channel x 32-time tile; the K loop stages an 8x8
/// weight block and an 8x32 input window in workgroup memory. That is ~6x less
/// global traffic than the per-output form and keeps each dispatch in the
/// microsecond range, which is what stops the Windows TDR watchdog on the last
/// SEANet block.
pub const CONV1D_TILED_F32: &str = r#"
// 0=cin 1=cout 2=ks 3=dilation 4=stride 5=pad 6=t_in 7=t_out
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 8>;

const BM: u32 = 8u;
const BN: u32 = 32u;
const BK: u32 = 8u;
var<workgroup> As: array<array<f32, 8>, 8>;
var<workgroup> Bs: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let co0 = wid.x * BM;
    let t0 = wid.y * BN;
    let kdim = p[0] * p[2];
    let row = li / BN;
    let col = li % BN;
    var acc = 0.0;
    let chunks = (kdim + BK - 1u) / BK;
    for (var c0 = 0u; c0 < chunks; c0 = c0 + 1u) {
        if (li < BM * BK) {
            let r = li / BK;
            let kk = c0 * BK + (li % BK);
            var v = 0.0;
            if (co0 + r < p[1] && kk < kdim) { v = w[(co0 + r) * kdim + kk]; }
            As[r][li % BK] = v;
        }
        {
            let kk = c0 * BK + row;
            var v = 0.0;
            if (kk < kdim) {
                let ci = kk / p[2];
                let k = kk % p[2];
                let src = i32(t0 + col) * i32(p[4]) + i32(k * p[3]) - i32(p[5]);
                if (src >= 0 && src < i32(p[6])) { v = x[ci * p[6] + u32(src)]; }
            }
            Bs[row][col] = v;
        }
        workgroupBarrier();
        for (var i = 0u; i < BK; i = i + 1u) {
            acc = acc + As[row][i] * Bs[i][col];
        }
        workgroupBarrier();
    }
    let co = co0 + row;
    let t = t0 + col;
    if (co < p[1] && t < p[7]) {
        y[co * p[7] + t] = acc + b[co];
    }
}
"#;

/// Streaming conv1d: 8 output channels per thread, one timestep per lane.
///
/// The codec's shapes are narrow in channels (96..1536) and very long in time
/// (up to 182 400), which makes the implicit-GEMM tiling barrier-bound: 84 K
/// chunks x 2 barriers per workgroup cost more than the arithmetic. This form
/// keeps 8 accumulators in registers, streams `x` coalesced across lanes and
/// lets L1/L2 absorb the reuse across the 8 channels, which measured 15x faster
/// than the tiled version on the same layer.
pub const CONV1D_STREAM_F32: &str = r#"
// 0=cin 1=cout 2=ks 3=dilation 4=stride 5=pad 6=t_in 7=t_out
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 8>;

const BC: u32 = 8u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    // x-major over time so consecutive workgroups keep the same 21 KB weight
    // block hot and stream a new input window (the swapped order thrashing both).
    let co0 = wid.y * BC;
    let t = wid.x * 256u + lid.x;
    if (t >= p[7]) { return; }
    var acc: array<f32, 8>;
    for (var c = 0u; c < BC; c = c + 1u) { acc[c] = 0.0; }
    for (var ci = 0u; ci < p[0]; ci = ci + 1u) {
        for (var k = 0u; k < p[2]; k = k + 1u) {
            let src = i32(t) * i32(p[4]) + i32(k * p[3]) - i32(p[5]);
            if (src < 0 || src >= i32(p[6])) { continue; }
            let xv = x[ci * p[6] + u32(src)];
            for (var c = 0u; c < BC; c = c + 1u) {
                if (co0 + c < p[1]) {
                    acc[c] = acc[c] + xv * w[(co0 + c) * p[0] * p[2] + ci * p[2] + k];
                }
            }
        }
    }
    for (var c = 0u; c < BC; c = c + 1u) {
        if (co0 + c < p[1]) { y[(co0 + c) * p[7] + t] = acc[c] + b[co0 + c]; }
    }
}
"#;

/// Copy `rows` rows into `dst` starting at row `offset` (KV-cache append).
pub const WRITE_ROWS_F32: &str = r#"
// 0=cols 1=offset 2=rows
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32, 4>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= p[0] * p[2]) { return; }
    dst[p[1] * p[0] + i] = src[i];
}
"#;

/// Attention for a single query row against a KV cache of `len` rows.
///
/// No mask: the query is the newest position, so every cached key is in the past.
/// One workgroup per head; lanes split the keys for the score pass and one lane
/// writes one output dimension, exactly like [`ATTENTION_F32`].
pub const ATTENTION_DECODE_F32: &str = r#"
// 0=len 1=heads 2=head_dim 3=0
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 4>;
var<workgroup> scores: array<f32, 256>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let heads = p[1];
    let d = p[2];
    let len = p[0];
    let head = wid.x;
    let lane = lid.x;
    let stride = heads * d;
    let qbase = head * d;
    let scale = inverseSqrt(f32(d));

    var kk = lane;
    loop {
        if (kk >= len) { break; }
        let kbase = kk * stride + head * d;
        var acc = 0.0;
        for (var i = 0u; i < d; i = i + 1u) {
            acc = acc + q[qbase + i] * k[kbase + i];
        }
        scores[kk] = acc * scale;
        kk = kk + 64u;
    }
    workgroupBarrier();

    var maxs = -1e30;
    for (var i = 0u; i < len; i = i + 1u) { maxs = max(maxs, scores[i]); }
    var sum = 0.0;
    for (var i = 0u; i < len; i = i + 1u) { sum = sum + exp(scores[i] - maxs); }
    let inv_sum = 1.0 / sum;

    if (lane < d) {
        var acc = 0.0;
        for (var j = 0u; j < len; j = j + 1u) {
            acc = acc + exp(scores[j] - maxs) * v[j * stride + head * d + lane];
        }
        out[qbase + lane] = acc * inv_sum;
    }
}
"#;

/// Rotary embedding driven by a precomputed cos/sin table (llama-3 scaled freqs
/// for the LM, plain freqs for the codec). `cos`/`sin` are `[T, head_dim]`.
pub const ROPE_TABLED_F32: &str = r#"
// 0=T 1=heads 2=head_dim 3=0
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_t: array<f32>;
@group(0) @binding(2) var<storage, read> sin_t: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32, 4>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = p[2];
    let h = p[1];
    let half = d / 2u;
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= p[0] * h * half) { return; }
    let j = i % half;
    let t = i / (h * half);
    let head = (i / half) % h;
    let base = t * h * d + head * d;
    let c = cos_t[t * d + j];
    let s = sin_t[t * d + j];
    let a = x[base + j];
    let b = x[base + j + half];
    x[base + j] = a * c - b * s;
    x[base + j + half] = b * c + a * s;
}
"#;

/// Repeat KV heads for grouped-query attention: `[T, kv*D] -> [T, heads*D]`.
pub const REPEAT_KV_F32: &str = r#"
// 0=T 1=kv_heads 2=heads 3=head_dim
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32, 4>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let d = p[3];
    let heads = p[2];
    let kv = p[1];
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= p[0] * heads * d) { return; }
    let t = i / (heads * d);
    let h = (i / d) % heads;
    let dd = i % d;
    let src_head = h / (heads / kv);
    y[i] = x[t * kv * d + src_head * d + dd];
}
"#;

/// ELU with alpha = 1: `x > 0 ? x : exp(x) - 1`.
pub const ELU_F32: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> p: array<u32, 1>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= p[0]) { return; }
    let v = x[i];
    x[i] = select(exp(v) - 1.0, v, v > 0.0);
}
"#;

// A register-blocked 64x64 tile (16x16 workgroup, 4x4 per thread, bank-padded
// shared tiles) was tried here and measured slower on the P104-100: 44 GFLOP/s
// against 86 for the simple kernel below, and a single 512x512 GEMV already costs
// 88 us on this card. Kept the simple kernel; revisit on real hardware.


/// Single-token GEMV: one warp per output row, lanes stride over K.
///
/// The 16x16 tiled kernel computes sixteen rows per tile and the LM's decode step
/// only ever needs one, so it did 16x the arithmetic in the hot loop. A warp per
/// row removes the waste: 512 rows become 64 workgroups of 8 warps.
pub const GEMV_F32: &str = r#"
// Shared with LINEAR_F32: 0=m 1=n 2=k 3=has_bias
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 4>;
var<workgroup> red: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wid.x * 8u + warp;
    let kdim = p[2];
    var acc = 0.0;
    if (row < p[1]) {
        let base = row * kdim;
        var i = lane;
        loop {
            if (i >= kdim) { break; }
            acc = acc + x[i] * w[base + i];
            i = i + 32u;
        }
    }
    red[warp][lane] = acc;
    workgroupBarrier();
    if (lane == 0u && row < p[1]) {
        var sum = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + red[warp][i]; }
        if (p[3] == 1u) { sum = sum + b[row]; }
        y[row] = sum;
    }
}
"#;

/// Sum of the sixteen audio embedding rows for one frame.
///
/// `out[d] = sum_cb table[cb * vocab + token[cb], d]`. Keeping this on the device
/// is what lets the generation loop sample without a host round-trip.
pub const EMBED_SUM_F32: &str = r#"
// 0=hidden 1=cb_offset 2=vocab 3=cb_count
@group(0) @binding(0) var<storage, read> table: array<f32>;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32, 4>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let hidden = p[0];
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= hidden) { return; }
    var acc = 0.0;
    for (var j = 0u; j < p[3]; j = j + 1u) {
        let cb = p[1] + j;
        let row = cb * p[2] + tokens[cb];
        acc = acc + table[row * hidden + i];
    }
    out[i] = acc;
}
"#;

/// top-k sampler, port of `sample_topk` + `_multinomial_sample_one_no_sync`.
///
/// Mask EOS, divide by temperature, keep the top-k, then `argmax(probs / noise)`.
/// Parallel max/sum over 256 lanes; the final scans are serial in lane 0, which is
/// a few microseconds and still far cheaper than a host round-trip.
pub const SAMPLE_TOPK_F32: &str = r#"
// 0=vocab 1=topk 2=temperature(bits) 3=allow_eos*2 + out_offset 4=noise row offset
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> noise: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<storage, read> p: array<u32, 5>;
var<workgroup> part: array<f32, 256>;

fn score(i: u32) -> f32 {
    var v = logits[i] / bitcast<f32>(p[2]);
    if (i == 0u && (p[3] & 1u) == 0u) { v = -1e5; }
    return v;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let vocab = p[0];
    let topk = min(p[1], vocab);

    // Cutoff = k-th largest. Each lane ranks its own strided subset by counting how
    // many scores beat it (O(vocab) per element, /256 lanes), then the kept scores'
    // minimum is reduced across the workgroup.
    var local_min = 1e30;
    var i = lane;
    loop {
        if (i >= vocab) { break; }
        let vi = score(i);
        var greater = 0u;
        for (var j = 0u; j < vocab; j = j + 1u) {
            let vj = score(j);
            if (vj > vi || (vj == vi && j < i)) { greater = greater + 1u; }
        }
        if (greater < topk) { local_min = min(local_min, vi); }
        i = i + 256u;
    }
    part[lane] = local_min;
    workgroupBarrier();
    if (lane == 0u) {
        var m = 1e30;
        for (var j = 0u; j < 256u; j = j + 1u) { m = min(m, part[j]); }
        part[0] = m;
    }
    workgroupBarrier();
    let cutoff = part[0];

    // Parallel max over surviving scores.
    var local_max = -1e30;
    i = lane;
    loop {
        if (i >= vocab) { break; }
        let v = score(i);
        if (v >= cutoff && v > local_max) { local_max = v; }
        i = i + 256u;
    }
    part[lane] = local_max;
    workgroupBarrier();
    if (lane == 0u) {
        var m = -1e30;
        for (var j = 0u; j < 256u; j = j + 1u) { m = max(m, part[j]); }
        part[0] = m;
    }
    workgroupBarrier();
    let maxv = part[0];

    // Parallel sum of exp(score - max) over survivors.
    var local_sum = 0.0;
    i = lane;
    loop {
        if (i >= vocab) { break; }
        let v = score(i);
        if (v >= cutoff) { local_sum = local_sum + exp(v - maxv); }
        i = i + 256u;
    }
    part[lane] = local_sum;
    workgroupBarrier();
    if (lane == 0u) {
        var sum = 0.0;
        for (var j = 0u; j < 256u; j = j + 1u) { sum = sum + part[j]; }
        part[0] = sum;
    }
    workgroupBarrier();
    let total = part[0];

    // argmax(probs / noise) over survivors.
    if (lane == 0u) {
        var best_i = 0u;
        var best_ratio = -1e30;
        for (var j = 0u; j < vocab; j = j + 1u) {
            let v = score(j);
            if (v >= cutoff) {
                let ratio = (exp(v - maxv) / total) / noise[p[4] + j];
                if (ratio > best_ratio) { best_ratio = ratio; best_i = j; }
            }
        }
        out[p[3] >> 1u] = best_i;
    }
}
"#;

/// Tiled causal conv1d for the GPU.
///
/// One workgroup owns 64 output channels x 64 timesteps. The input window is
/// staged in workgroup memory per 32-channel chunk, so each staged value feeds 16
/// FMAs (4 timesteps x 4 channels per thread) instead of one; the naive kernel let
/// every output re-read its window from global memory, which is why it ran at
/// 28 GFLOP/s while cuDNN reaches ~800 GMAC/s on the same card for this shape.
pub const CONV1D_TILED_GPU: &str = r#"
// 0=cin 1=cout 2=ks 3=dilation 4=stride 5=pad 6=t_in 7=t_out
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 8>;

const CT: u32 = 64u;    // output channels per workgroup
const TT: u32 = 64u;    // timesteps per workgroup
const CI: u32 = 32u;    // input channels staged at once
const WIN: u32 = TT + 8u * 9u;   // window: TT + (ks-1)*dilation, dilation <= 9

var<workgroup> stage: array<f32, 32 * 640>;   // [ci][window]

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let co0 = wid.x * CT;
    let t0 = wid.y * TT;
    // 16 threads in y own 4 timesteps each; 16 in x own 4 channels each.
    let t_base = t0 + lid.y * 4u;
    let c_base = co0 + lid.x * 4u;
    let span = (p[2] - 1u) * p[3] + 1u;

    var acc: array<array<f32, 4>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) { acc[i][j] = 0.0; }
    }

    let chunks = (p[0] + CI - 1u) / CI;
    for (var c0 = 0u; c0 < chunks; c0 = c0 + 1u) {
        // Stage [ci0..ci0+CI) x [t0*stride - pad .. + TT*stride + span)
        let win_len = TT * p[4] + span;
        for (var i = tid; i < CI * win_len; i = i + 256u) {
            let ci_local = i / win_len;
            let j = i % win_len;
            let ci = c0 * CI + ci_local;
            var value = 0.0;
            if (ci < p[0]) {
                let src = i32(t0 * p[4]) + i32(j) - i32(p[5]);
                if (src >= 0 && src < i32(p[6])) {
                    value = x[ci * p[6] + u32(src)];
                }
            }
            stage[ci_local * win_len + j] = value;
        }
        workgroupBarrier();

        for (var ci_local = 0u; ci_local < CI; ci_local = ci_local + 1u) {
            let ci = c0 * CI + ci_local;
            if (ci >= p[0]) { continue; }
            for (var tap = 0u; tap < p[2]; tap = tap + 1u) {
                var xv: array<f32, 4>;
                for (var i = 0u; i < 4u; i = i + 1u) {
                    xv[i] = stage[ci_local * win_len + (lid.y * 4u + i) * p[4] + tap * p[3]];
                }
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let wv = w[((c_base + j) * p[0] + ci) * p[2] + tap];
                    for (var i = 0u; i < 4u; i = i + 1u) {
                        acc[i][j] = acc[i][j] + xv[i] * wv;
                    }
                }
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let t = t_base + i;
        if (t >= p[7]) { continue; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let co = c_base + j;
            if (co >= p[1]) { continue; }
            y[co * p[7] + t] = acc[i][j] + b[co];
        }
    }
}
"#;

/// Pure FMA loop: measures the card's achievable fp32 throughput with no memory
/// traffic, i.e. the ceiling every other kernel is judged against.
pub const FMA_PROBE: &str = r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> p: array<u32, 2>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var a = f32(gid.x) * 1e-6 + 1.0;
    var b = 1.0000001;
    var c = 0.9999999;
    var d = 1.0000002;
    for (var i = 0u; i < p[0]; i = i + 1u) {
        a = a * b + 0.0000001;
        b = b * c + 0.0000001;
        c = c * d + 0.0000001;
        d = d * a + 0.0000001;
    }
    out[gid.x] = a + b + c + d;
}
"#;

/// Tiled causal conv1d, channels-last output, **packed weights** `[tap*cin+ci][cout]`.
///
/// The weight layout is what makes this fast: with lanes walking `co`, one
/// vector load per lane covers four consecutive output channels, so a whole warp
/// touches one contiguous run instead of sixteen cache lines `cin*ks` apart (the
/// reason the checkpoint-layout kernel measured 27 GFLOP/s while the card does
/// 2.1 TFLOP/s in a pure FMA loop).
pub const CONV1D_CL_GPU: &str = r#"
// 0=cin 1=cout 2=ks 3=dilation 4=stride 5=pad 6=t_in 7=t_out
@group(0) @binding(0) var<storage, read> x: array<f32>;          // [t_in, cin]
@group(0) @binding(1) var<storage, read> w: array<f32>;          // [(tap*cin+ci)*cout + co]
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;    // [t_out, cout]
@group(0) @binding(4) var<storage, read> p: array<u32, 8>;

const TT: u32 = 64u;   // timesteps per workgroup
const CT: u32 = 64u;   // channels per workgroup
const CI: u32 = 16u;   // input channels staged at once

var<workgroup> stage: array<f32, 16 * 640>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let t0 = wid.x * TT;
    let co0 = wid.y * CT;
    let t_base = t0 + lid.y * 4u;
    let c_base = co0 + lid.x * 4u;
    let span = (p[2] - 1u) * p[3] + 1u;
    let win_len = TT * p[4] + span;

    var acc: array<array<f32, 4>, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        for (var j = 0u; j < 4u; j = j + 1u) { acc[i][j] = 0.0; }
    }

    let chunks = (p[0] + CI - 1u) / CI;
    for (var c0 = 0u; c0 < chunks; c0 = c0 + 1u) {
        for (var i = tid; i < CI * win_len; i = i + 256u) {
            let ci_local = i / win_len;
            let j = i % win_len;
            let ci = c0 * CI + ci_local;
            var value = 0.0;
            if (ci < p[0]) {
                let src = i32(t0 * p[4]) + i32(j) - i32(p[5]);
                if (src >= 0 && src < i32(p[6])) {
                    value = x[u32(src) * p[0] + ci];
                }
            }
            stage[ci_local * win_len + j] = value;
        }
        workgroupBarrier();

        for (var ci_local = 0u; ci_local < CI; ci_local = ci_local + 1u) {
            let ci = c0 * CI + ci_local;
            if (ci >= p[0]) { continue; }
            for (var tap = 0u; tap < p[2]; tap = tap + 1u) {
                var xv: array<f32, 4>;
                for (var i = 0u; i < 4u; i = i + 1u) {
                    xv[i] = stage[ci_local * win_len + (lid.y * 4u + i) * p[4] + tap * p[3]];
                }
                let wbase = (tap * p[0] + ci) * p[1] + c_base;
                for (var j = 0u; j < 4u; j = j + 1u) {
                    let wv = w[wbase + j];   // consecutive lanes read consecutive co
                    for (var i = 0u; i < 4u; i = i + 1u) {
                        acc[i][j] = acc[i][j] + xv[i] * wv;
                    }
                }
            }
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let t = t_base + i;
        if (t >= p[7]) { continue; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let co = c_base + j;
            if (co >= p[1]) { continue; }
            y[t * p[1] + co] = acc[i][j] + b[co];
        }
    }
}
"#;

/// Register-blocked channels-last conv: 8 timesteps x 4 channels per thread.
///
/// Weights are read once per eight FMAs (vs once per FMA in the naive kernel), so
/// the kernel needs ~500 GB/s of cache bandwidth at the card's 2.1 TFLOP/s instead
/// of the ~2 TB/s that made the simple version stall at 49 GFLOP/s. The 32
/// accumulators are named locals: an `array<f32, N>` spills out of registers on
/// this backend (see ctc-forced-aligner-wgpu's tile notes).
pub const CONV1D_REG_GPU: &str = r#"
// 0=cin 1=cout 2=ks 3=dilation 4=stride 5=pad 6=t_in 7=t_out
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 8>;

const TT: u32 = 128u;
const CT: u32 = 64u;
// Staged input channels (stride <= 2: the window fits 16 channels in 20 KB). The window length is `TT * stride + span`, so with the
// worst stride the codec uses (8) a chunk of 4 keeps the staging inside 18 KB of
// workgroup memory; sizing it for the largest case is what a constant `WIN`
// silently broke before (shared-memory overrun on strided layers).
const CI: u32 = 16u;
const WIN_MAX: u32 = 128u * 8u + 64u;

var<workgroup> stage: array<f32, 16 * 320>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let t0 = wid.x * TT;
    let co0 = wid.y * CT;
    let t_base = t0 + lid.y * 8u;
    let c_base = co0 + lid.x * 4u;
    let span = (p[2] - 1u) * p[3] + 1u;
    let win_len = TT * p[4] + span;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a02: f32 = 0.0;
    var a03: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;
    var a12: f32 = 0.0;
    var a13: f32 = 0.0;
    var a20: f32 = 0.0;
    var a21: f32 = 0.0;
    var a22: f32 = 0.0;
    var a23: f32 = 0.0;
    var a30: f32 = 0.0;
    var a31: f32 = 0.0;
    var a32: f32 = 0.0;
    var a33: f32 = 0.0;
    var a40: f32 = 0.0;
    var a41: f32 = 0.0;
    var a42: f32 = 0.0;
    var a43: f32 = 0.0;
    var a50: f32 = 0.0;
    var a51: f32 = 0.0;
    var a52: f32 = 0.0;
    var a53: f32 = 0.0;
    var a60: f32 = 0.0;
    var a61: f32 = 0.0;
    var a62: f32 = 0.0;
    var a63: f32 = 0.0;
    var a70: f32 = 0.0;
    var a71: f32 = 0.0;
    var a72: f32 = 0.0;
    var a73: f32 = 0.0;

    let chunks = (p[0] + CI - 1u) / CI;
    for (var c0 = 0u; c0 < chunks; c0 = c0 + 1u) {
        for (var i = tid; i < CI * win_len; i = i + 256u) {
            let ci_local = i / win_len;
            let j = i % win_len;
            let ci = c0 * CI + ci_local;
            var value = 0.0;
            if (ci < p[0]) {
                let src = i32(t0 * p[4]) + i32(j) - i32(p[5]);
                if (src >= 0 && src < i32(p[6])) {
                    value = x[u32(src) * p[0] + ci];
                }
            }
            stage[ci_local * win_len + j] = value;
        }
        workgroupBarrier();

        for (var ci_local = 0u; ci_local < CI; ci_local = ci_local + 1u) {
            let ci = c0 * CI + ci_local;
            if (ci >= p[0]) { continue; }
            for (var tap = 0u; tap < p[2]; tap = tap + 1u) {
                var xv: array<f32, 8>;
                for (var i = 0u; i < 8u; i = i + 1u) {
                    xv[i] = stage[ci_local * win_len + (lid.y * 8u + i) * p[4] + tap * p[3]];
                }
                let wbase = (tap * p[0] + ci) * p[1] + c_base;
                let wv0 = w[wbase];
                let wv1 = w[wbase + 1u];
                let wv2 = w[wbase + 2u];
                let wv3 = w[wbase + 3u];
                a00 = a00 + xv[0] * wv0;
                a01 = a01 + xv[0] * wv1;
                a02 = a02 + xv[0] * wv2;
                a03 = a03 + xv[0] * wv3;
                a10 = a10 + xv[1] * wv0;
                a11 = a11 + xv[1] * wv1;
                a12 = a12 + xv[1] * wv2;
                a13 = a13 + xv[1] * wv3;
                a20 = a20 + xv[2] * wv0;
                a21 = a21 + xv[2] * wv1;
                a22 = a22 + xv[2] * wv2;
                a23 = a23 + xv[2] * wv3;
                a30 = a30 + xv[3] * wv0;
                a31 = a31 + xv[3] * wv1;
                a32 = a32 + xv[3] * wv2;
                a33 = a33 + xv[3] * wv3;
                a40 = a40 + xv[4] * wv0;
                a41 = a41 + xv[4] * wv1;
                a42 = a42 + xv[4] * wv2;
                a43 = a43 + xv[4] * wv3;
                a50 = a50 + xv[5] * wv0;
                a51 = a51 + xv[5] * wv1;
                a52 = a52 + xv[5] * wv2;
                a53 = a53 + xv[5] * wv3;
                a60 = a60 + xv[6] * wv0;
                a61 = a61 + xv[6] * wv1;
                a62 = a62 + xv[6] * wv2;
                a63 = a63 + xv[6] * wv3;
                a70 = a70 + xv[7] * wv0;
                a71 = a71 + xv[7] * wv1;
                a72 = a72 + xv[7] * wv2;
                a73 = a73 + xv[7] * wv3;
            }
        }
        workgroupBarrier();
    }

        let cc0 = c_base + 0u;
        let cc1 = c_base + 1u;
        let cc2 = c_base + 2u;
        let cc3 = c_base + 3u;
        if (t_base + 0u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 0u) * p[1] + cc0] = a00 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 0u) * p[1] + cc1] = a01 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 0u) * p[1] + cc2] = a02 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 0u) * p[1] + cc3] = a03 + b[cc3]; }
        }
        if (t_base + 1u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 1u) * p[1] + cc0] = a10 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 1u) * p[1] + cc1] = a11 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 1u) * p[1] + cc2] = a12 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 1u) * p[1] + cc3] = a13 + b[cc3]; }
        }
        if (t_base + 2u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 2u) * p[1] + cc0] = a20 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 2u) * p[1] + cc1] = a21 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 2u) * p[1] + cc2] = a22 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 2u) * p[1] + cc3] = a23 + b[cc3]; }
        }
        if (t_base + 3u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 3u) * p[1] + cc0] = a30 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 3u) * p[1] + cc1] = a31 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 3u) * p[1] + cc2] = a32 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 3u) * p[1] + cc3] = a33 + b[cc3]; }
        }
        if (t_base + 4u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 4u) * p[1] + cc0] = a40 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 4u) * p[1] + cc1] = a41 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 4u) * p[1] + cc2] = a42 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 4u) * p[1] + cc3] = a43 + b[cc3]; }
        }
        if (t_base + 5u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 5u) * p[1] + cc0] = a50 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 5u) * p[1] + cc1] = a51 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 5u) * p[1] + cc2] = a52 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 5u) * p[1] + cc3] = a53 + b[cc3]; }
        }
        if (t_base + 6u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 6u) * p[1] + cc0] = a60 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 6u) * p[1] + cc1] = a61 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 6u) * p[1] + cc2] = a62 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 6u) * p[1] + cc3] = a63 + b[cc3]; }
        }
        if (t_base + 7u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 7u) * p[1] + cc0] = a70 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 7u) * p[1] + cc1] = a71 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 7u) * p[1] + cc2] = a72 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 7u) * p[1] + cc3] = a73 + b[cc3]; }
        }
}
"#;
pub const CONV1D_REG_GPU_STRIDED: &str = r#"
// 0=cin 1=cout 2=ks 3=dilation 4=stride 5=pad 6=t_in 7=t_out
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 8>;

const TT: u32 = 128u;
const CT: u32 = 64u;
// Staged input channels. The window length is `TT * stride + span`, so with the
// worst stride the codec uses (8) a chunk of 4 keeps the staging inside 18 KB of
// workgroup memory; sizing it for the largest case is what a constant `WIN`
// silently broke before (shared-memory overrun on strided layers).
const CI: u32 = 4u;
const WIN_MAX: u32 = 128u * 8u + 64u;

var<workgroup> stage: array<f32, 4 * 1152>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let t0 = wid.x * TT;
    let co0 = wid.y * CT;
    let t_base = t0 + lid.y * 8u;
    let c_base = co0 + lid.x * 4u;
    let span = (p[2] - 1u) * p[3] + 1u;
    let win_len = TT * p[4] + span;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a02: f32 = 0.0;
    var a03: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;
    var a12: f32 = 0.0;
    var a13: f32 = 0.0;
    var a20: f32 = 0.0;
    var a21: f32 = 0.0;
    var a22: f32 = 0.0;
    var a23: f32 = 0.0;
    var a30: f32 = 0.0;
    var a31: f32 = 0.0;
    var a32: f32 = 0.0;
    var a33: f32 = 0.0;
    var a40: f32 = 0.0;
    var a41: f32 = 0.0;
    var a42: f32 = 0.0;
    var a43: f32 = 0.0;
    var a50: f32 = 0.0;
    var a51: f32 = 0.0;
    var a52: f32 = 0.0;
    var a53: f32 = 0.0;
    var a60: f32 = 0.0;
    var a61: f32 = 0.0;
    var a62: f32 = 0.0;
    var a63: f32 = 0.0;
    var a70: f32 = 0.0;
    var a71: f32 = 0.0;
    var a72: f32 = 0.0;
    var a73: f32 = 0.0;

    let chunks = (p[0] + CI - 1u) / CI;
    for (var c0 = 0u; c0 < chunks; c0 = c0 + 1u) {
        for (var i = tid; i < CI * win_len; i = i + 256u) {
            let ci_local = i / win_len;
            let j = i % win_len;
            let ci = c0 * CI + ci_local;
            var value = 0.0;
            if (ci < p[0]) {
                let src = i32(t0 * p[4]) + i32(j) - i32(p[5]);
                if (src >= 0 && src < i32(p[6])) {
                    value = x[u32(src) * p[0] + ci];
                }
            }
            stage[ci_local * win_len + j] = value;
        }
        workgroupBarrier();

        for (var ci_local = 0u; ci_local < CI; ci_local = ci_local + 1u) {
            let ci = c0 * CI + ci_local;
            if (ci >= p[0]) { continue; }
            for (var tap = 0u; tap < p[2]; tap = tap + 1u) {
                var xv: array<f32, 8>;
                for (var i = 0u; i < 8u; i = i + 1u) {
                    xv[i] = stage[ci_local * win_len + (lid.y * 8u + i) * p[4] + tap * p[3]];
                }
                let wbase = (tap * p[0] + ci) * p[1] + c_base;
                let wv0 = w[wbase];
                let wv1 = w[wbase + 1u];
                let wv2 = w[wbase + 2u];
                let wv3 = w[wbase + 3u];
                a00 = a00 + xv[0] * wv0;
                a01 = a01 + xv[0] * wv1;
                a02 = a02 + xv[0] * wv2;
                a03 = a03 + xv[0] * wv3;
                a10 = a10 + xv[1] * wv0;
                a11 = a11 + xv[1] * wv1;
                a12 = a12 + xv[1] * wv2;
                a13 = a13 + xv[1] * wv3;
                a20 = a20 + xv[2] * wv0;
                a21 = a21 + xv[2] * wv1;
                a22 = a22 + xv[2] * wv2;
                a23 = a23 + xv[2] * wv3;
                a30 = a30 + xv[3] * wv0;
                a31 = a31 + xv[3] * wv1;
                a32 = a32 + xv[3] * wv2;
                a33 = a33 + xv[3] * wv3;
                a40 = a40 + xv[4] * wv0;
                a41 = a41 + xv[4] * wv1;
                a42 = a42 + xv[4] * wv2;
                a43 = a43 + xv[4] * wv3;
                a50 = a50 + xv[5] * wv0;
                a51 = a51 + xv[5] * wv1;
                a52 = a52 + xv[5] * wv2;
                a53 = a53 + xv[5] * wv3;
                a60 = a60 + xv[6] * wv0;
                a61 = a61 + xv[6] * wv1;
                a62 = a62 + xv[6] * wv2;
                a63 = a63 + xv[6] * wv3;
                a70 = a70 + xv[7] * wv0;
                a71 = a71 + xv[7] * wv1;
                a72 = a72 + xv[7] * wv2;
                a73 = a73 + xv[7] * wv3;
            }
        }
        workgroupBarrier();
    }

        let cc0 = c_base + 0u;
        let cc1 = c_base + 1u;
        let cc2 = c_base + 2u;
        let cc3 = c_base + 3u;
        if (t_base + 0u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 0u) * p[1] + cc0] = a00 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 0u) * p[1] + cc1] = a01 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 0u) * p[1] + cc2] = a02 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 0u) * p[1] + cc3] = a03 + b[cc3]; }
        }
        if (t_base + 1u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 1u) * p[1] + cc0] = a10 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 1u) * p[1] + cc1] = a11 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 1u) * p[1] + cc2] = a12 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 1u) * p[1] + cc3] = a13 + b[cc3]; }
        }
        if (t_base + 2u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 2u) * p[1] + cc0] = a20 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 2u) * p[1] + cc1] = a21 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 2u) * p[1] + cc2] = a22 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 2u) * p[1] + cc3] = a23 + b[cc3]; }
        }
        if (t_base + 3u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 3u) * p[1] + cc0] = a30 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 3u) * p[1] + cc1] = a31 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 3u) * p[1] + cc2] = a32 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 3u) * p[1] + cc3] = a33 + b[cc3]; }
        }
        if (t_base + 4u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 4u) * p[1] + cc0] = a40 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 4u) * p[1] + cc1] = a41 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 4u) * p[1] + cc2] = a42 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 4u) * p[1] + cc3] = a43 + b[cc3]; }
        }
        if (t_base + 5u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 5u) * p[1] + cc0] = a50 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 5u) * p[1] + cc1] = a51 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 5u) * p[1] + cc2] = a52 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 5u) * p[1] + cc3] = a53 + b[cc3]; }
        }
        if (t_base + 6u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 6u) * p[1] + cc0] = a60 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 6u) * p[1] + cc1] = a61 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 6u) * p[1] + cc2] = a62 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 6u) * p[1] + cc3] = a63 + b[cc3]; }
        }
        if (t_base + 7u < p[7]) {
            if (cc0 < p[1]) { y[(t_base + 7u) * p[1] + cc0] = a70 + b[cc0]; }
            if (cc1 < p[1]) { y[(t_base + 7u) * p[1] + cc1] = a71 + b[cc1]; }
            if (cc2 < p[1]) { y[(t_base + 7u) * p[1] + cc2] = a72 + b[cc2]; }
            if (cc3 < p[1]) { y[(t_base + 7u) * p[1] + cc3] = a73 + b[cc3]; }
        }
}
"#;
