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
