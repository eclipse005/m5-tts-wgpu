//! WGSL kernels. Each entry point is one dispatch; host wrappers live in
//! `crate::ops`.

/// `y += x`, in place (`y` is read_write, `x` read-only).
pub const RESIDUAL_F32: &str = r#"
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32, 1>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let c = gid.y;
    if (r >= p[0] || c >= p[1]) { return; }
    y[c * p[0] + r] = x[r * p[1] + c];
}
"#;
