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
    // alpha/beta arrive pre-exp'ed (they are per-channel constants) and the
    // square is a multiply: pow(sin(x), 2.0) compiles to the exp2/log2 slow
    // path, and exp() per element was 18 M exps for 96 channels' worth.
    let s = sin(v * alpha[c]);
    x[idx] = v + s * s / beta[c];
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

/// Single-token GEMV with bf16 weights packed 2-per-u32 (resident, decoded
/// in-shader): warp-per-row, 8 rows per workgroup, f32 accumulate. Halves the
/// weight traffic for the LM decode GEMVs. Port of ming-3.0-tts GEMV_BF16W.
pub const GEMV_BF16W_M5: &str = r#"
// 0=rows 1=in_words (in_features / 2)
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32, 2>;
var<workgroup> red: array<array<f32, 32>, 8>;

fn bf_lo(u: u32) -> f32 { return bitcast<f32>((u & 0xFFFFu) << 16u); }
fn bf_hi(u: u32) -> f32 { return bitcast<f32>((u & 0xFFFF0000u)); }

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wg.x * 8u + warp;
    var acc = 0.0;
    if (row < p[0]) {
        let wbase = row * p[1];
        var wi = lane;
        loop {
            if (wi >= p[1]) { break; }
            let u = w[wbase + wi];
            acc = acc + bf_lo(u) * x[wi * 2u] + bf_hi(u) * x[wi * 2u + 1u];
            wi = wi + 32u;
        }
    }
    red[warp][lane] = acc;
    workgroupBarrier();
    if (lane == 0u) {
        var sum = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + red[warp][i]; }
        y[row] = sum;
    }
}
"#;

/// Tiled 2-D transpose: both sides coalesced via a padded shared tile.
///
/// The naive one-element-per-thread kernel read x rows 752 KB apart - every
/// warp transaction fetched a full line for 4 used bytes (128x read
/// amplification), which cost ~2 s across the codec's 48 transposes.
pub const TRANSPOSE_TILED_F32: &str = r#"
// 0=rows 1=cols
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32, 2>;

var<workgroup> tile: array<f32, 17 * 16>;  // +1 padding: no bank conflicts

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r0 = wid.x * 16u;
    let c0 = wid.y * 16u;
    let r = r0 + lid.x;
    let c = c0 + lid.y;
    if (r < p[0] && c < p[1]) {
        tile[lid.x * 17u + lid.y] = x[r * p[1] + c];
    }
    workgroupBarrier();
    // write phase: consecutive threads take consecutive r (coalesced y writes)
    let i = lid.x;
    let j = lid.y;
    let rr = r0 + i;
    let cc = c0 + j;
    if (rr < p[0] && cc < p[1]) {
        y[cc * p[0] + rr] = tile[i * 17u + j];
    }
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
/// memory after the barrier: every lane recomputes exp() locally, which removes
/// a genuine write/read race.
///
/// `scores` holds one span, so it must cover the longest span any caller asks
/// for. The codec passes its 72-frame sliding window, but the backbone prefill
/// passes the whole prompt (`window = frames`), i.e. up to the model's context
/// limit - the array used to be 256 entries and silently corrupted workgroup
/// memory for any prompt longer than that (a 500-token text already did).
pub const ATTENTION_F32: &str = r#"
// 0=T 1=heads 2=head_dim 3=window
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 4>;
// MAX_CONTEXT = 2048 (crate::lm::MAX_CONTEXT); the host rejects longer requests.
var<workgroup> scores: array<f32, 2048>;

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
/// Scatter for the phased transposed conv: `out[co][m*stride+q] = y[m][co*stride+q]`.
///
/// The convolution itself is the register-blocked causal one (`CONV1D_REG8_F32`
/// with `cout := cout*stride`, tap order reversed in the packed weight): a
/// transposed conv with `ks == stride * taps` is `taps`-tap causal conv on the
/// phase rows, and reusing that kernel is what took the codec's upsampling from
/// ~50 GFLOP/s to the same rate as its forward convolutions.
pub const PHASED_INTERLEAVE_F32: &str = r#"
// 0=rows(cout*stride) 1=stride 2=t_in
@group(0) @binding(0) var<storage, read> y: array<f32>;        // [t_in, rows]
@group(0) @binding(1) var<storage, read_write> out: array<f32>; // [rows/stride, t_in*stride]
@group(0) @binding(2) var<storage, read> p: array<u32, 4>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 128u;
    let cout = p[0] / p[1];
    if (i >= cout * p[2]) { return; }
    let co = i / p[2];
    let m = i % p[2];
    let src = m * p[0] + co * p[1];
    let dst = co * p[2] * p[1] + m * p[1];
    for (var q = 0u; q < p[1]; q = q + 1u) {
        out[dst + q] = y[src + q];
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
/// `y[dst_off .. dst_off+len] = x[src_off .. src_off+len]` (chunk splicing).
pub const COPY_RANGE_F32: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> p: array<u32, 4>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 128u;
    if (i >= p[0]) { return; }
    y[p[2] + i] = x[p[1] + i];
}
"#;

/// `y[0..len] = value`, 4 stores per thread.
///
/// `Device::zeros` used to reach the device by uploading a host-side zero vec
/// through `queue.write_buffer`: at codec block-4 sizes that is a 72 MB copy over
/// a x1 PCIe link (~2 GB/s measured) for data the GPU could write itself in a
/// fraction of a millisecond.
pub const FILL_F32: &str = r#"
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@group(0) @binding(1) var<storage, read> p: array<u32, 2>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let base = (gid.x + gid.y * nwg.x * 128u) * 4u;
    let v = bitcast<f32>(p[1]);
    if (base + 3u < p[0]) {
        y[base] = v;
        y[base + 1u] = v;
        y[base + 2u] = v;
        y[base + 3u] = v;
    } else {
        for (var i = base; i < p[0]; i = i + 1u) { y[i] = v; }
    }
}
"#;

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

/// Fused decode-step attention block: RMSNorm -> qkv projection -> rope ->
/// KV-cache append -> attention, one workgroup per query head (the GPU twin of
/// `cpu_ops::qkv_attend_bf16`). The heads sharing a KV head re-project its rows
/// (128 dots duplicated - cheaper than a barrier). Cache layout, rope and score
/// accumulation order match the unfused kernels, so results are bit-identical.
pub const QKV_ATTEND_F32: &str = r#"
// 0=q_rows 1=kv_rows 2=head_dim 3=position 4=heads
//
// `scores` is one span over the KV cache: it used to be 1024 entries while
// `len = position + 1` runs to the model's 2048-position context, so every
// generation past frame ~1024 wrote out of bounds and the output stopped being
// reproducible (the same command produced 467 or 1168 frames on this card).
@group(0) @binding(0) var<storage, read> qkv: array<f32>;      // [q_rows + 2*kv_rows]
@group(0) @binding(1) var<storage, read_write> cache_k: array<f32>; // [cap, q_rows]
@group(0) @binding(2) var<storage, read_write> cache_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> attended: array<f32>; // [q_rows]
@group(0) @binding(4) var<storage, read> cos_t: array<f32>;    // [cap, head_dim]
@group(0) @binding(5) var<storage, read> sin_t: array<f32>;
@group(0) @binding(6) var<storage, read> p: array<u32, 4>;

var<workgroup> qs: array<f32, 64>;
var<workgroup> ks: array<f32, 64>;
// MAX_CONTEXT wide (crate::lm::MAX_CONTEXT); the host rejects longer requests.
var<workgroup> scores: array<f32, 2048>;
var<workgroup> red123: array<f32, 1>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wid.x;
    let hd = p[2];   // head_dim (p[1] is kv_rows)
    let t = lid.x;
    let qo = head * hd;
    let kvh = head / (p[0] / p[1]);  // heads / kv_heads

    // Load this head's projected q and its kv head's k/v rows.
    qs[t] = qkv[qo + t];
    ks[t] = qkv[p[0] + kvh * hd + t];
    let vv = qkv[p[0] + p[1] + kvh * hd + t];

    // Rotate-half rope inside the 64-dim head (t < 32 owns the (j, j+32) pair).
    // The cos/sin table covers every position; this layer reads row `p[2]`.
    workgroupBarrier();
    if (t < hd / 2u) {
        let c = cos_t[p[3] * hd + t];
        let sn = sin_t[p[3] * hd + t];
        let ha = qs[t]; let hb = qs[t + hd / 2u];
        qs[t] = ha * c - hb * sn;
        qs[t + hd / 2u] = hb * c + ha * sn;
        let ka = ks[t]; let kb = ks[t + hd / 2u];
        ks[t] = ka * c - kb * sn;
        ks[t + hd / 2u] = kb * c + ka * sn;
    }
    workgroupBarrier();

    // Append to the cache at this position (expanded q-head layout, disjoint).
    let off = p[3] * p[0] + qo;
    cache_k[off + t] = ks[t];
    cache_v[off + t] = vv;
    // The rows just written are read back below (key `position` included), so the
    // barrier has to order *storage* accesses, not only workgroup memory.
    workgroupBarrier();
    storageBarrier();

    // Attention over rows 0..=position: scores in shared, softmax in thread 0,
    // weighted sum per thread over this head's value slice.
    let len = p[3] + 1u;
    let scale = 1.0 / sqrt(f32(hd));
    for (var j = t; j < len; j = j + 64u) {
        var acc = 0.0;
        for (var i = 0u; i < hd; i = i + 1u) { acc = acc + qs[i] * cache_k[j * p[0] + qo + i]; }
        scores[j] = acc * scale;
    }
    workgroupBarrier();
    if (t == 0u) {
        var maxs = -1e30;
        for (var j = 0u; j < len; j = j + 1u) { maxs = max(maxs, scores[j]); }
        var sum = 0.0;
        for (var j = 0u; j < len; j = j + 1u) {
            let e = exp(scores[j] - maxs);
            scores[j] = e;
            sum = sum + e;
        }
        red123[0u] = 1.0 / sum;
    }
    workgroupBarrier();
    let inv_sum = red123[0u];
    if (t < hd) {
        var acc = 0.0;
        for (var j = 0u; j < len; j = j + 1u) {
            acc = acc + scores[j] * cache_v[j * p[0] + qo + t];
        }
        attended[qo + t] = acc * inv_sum;
    }
}
"#;

/// Register-blocked channels-last conv: 8 timesteps x 8 output channels per
/// thread (64 named accumulators), 16 staged input channels per chunk. The
/// previous staged kernel read its staged window through a runtime-indexed
/// local array, which spilled to local memory; with named registers the inner
/// loop is 64 FMAs against 8 shared + 8 coalesced global loads. Same bindings
/// and packed weights as CONV1D_REG_GPU (channels-last x, [(tap*cin+ci)*cout+co]).
pub const CONV1D_REG8_F32: &str = r#"
// 0=cin 1=cout 2=ks 3=dilation 4=stride 5=pad 6=t_in 7=t_out
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 8>;

const TT: u32 = 64u;
const CT: u32 = 64u;
const CI: u32 = 16u;
/// Input window per staged channel: `TT * stride + (ks - 1) * dilation + 1`.
/// The only caller (`codec::seanet::causal_conv`) runs stride 1 with ks in
/// {1, 3, 7} and dilation in {1, 3, 9}, so the widest window is 64 + 54 + 1 =
/// 119. The array used to be sized 576 per channel (36 KB of workgroup memory),
/// which capped occupancy at two workgroups per SM on the P104 and ran these
/// convolutions at ~500 GFLOP/s instead of the kernel's ~1 TFLOP/s ceiling.
const WIN_MAX: u32 = 128u;

var<workgroup> stage: array<f32, 16 * 128>;

@compute @workgroup_size(8, 8)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) tid: u32,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let t0 = wid.x * TT;
    let co0 = wid.y * CT;
    let t_base = t0 + lid.y * 8u;
    let c_base = co0 + lid.x * 8u;
    let span = (p[2] - 1u) * p[3] + 1u;
    let win_len = TT * p[4] + span;

    var a00: f32 = 0.0;
    var a01: f32 = 0.0;
    var a02: f32 = 0.0;
    var a03: f32 = 0.0;
    var a04: f32 = 0.0;
    var a05: f32 = 0.0;
    var a06: f32 = 0.0;
    var a07: f32 = 0.0;
    var a10: f32 = 0.0;
    var a11: f32 = 0.0;
    var a12: f32 = 0.0;
    var a13: f32 = 0.0;
    var a14: f32 = 0.0;
    var a15: f32 = 0.0;
    var a16: f32 = 0.0;
    var a17: f32 = 0.0;
    var a20: f32 = 0.0;
    var a21: f32 = 0.0;
    var a22: f32 = 0.0;
    var a23: f32 = 0.0;
    var a24: f32 = 0.0;
    var a25: f32 = 0.0;
    var a26: f32 = 0.0;
    var a27: f32 = 0.0;
    var a30: f32 = 0.0;
    var a31: f32 = 0.0;
    var a32: f32 = 0.0;
    var a33: f32 = 0.0;
    var a34: f32 = 0.0;
    var a35: f32 = 0.0;
    var a36: f32 = 0.0;
    var a37: f32 = 0.0;
    var a40: f32 = 0.0;
    var a41: f32 = 0.0;
    var a42: f32 = 0.0;
    var a43: f32 = 0.0;
    var a44: f32 = 0.0;
    var a45: f32 = 0.0;
    var a46: f32 = 0.0;
    var a47: f32 = 0.0;
    var a50: f32 = 0.0;
    var a51: f32 = 0.0;
    var a52: f32 = 0.0;
    var a53: f32 = 0.0;
    var a54: f32 = 0.0;
    var a55: f32 = 0.0;
    var a56: f32 = 0.0;
    var a57: f32 = 0.0;
    var a60: f32 = 0.0;
    var a61: f32 = 0.0;
    var a62: f32 = 0.0;
    var a63: f32 = 0.0;
    var a64: f32 = 0.0;
    var a65: f32 = 0.0;
    var a66: f32 = 0.0;
    var a67: f32 = 0.0;
    var a70: f32 = 0.0;
    var a71: f32 = 0.0;
    var a72: f32 = 0.0;
    var a73: f32 = 0.0;
    var a74: f32 = 0.0;
    var a75: f32 = 0.0;
    var a76: f32 = 0.0;
    var a77: f32 = 0.0;

    let chunks = (p[0] + CI - 1u) / CI;
    for (var c0 = 0u; c0 < chunks; c0 = c0 + 1u) {
        for (var i = tid; i < CI * win_len; i = i + 64u) {
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
                let x0 = stage[ci_local * win_len + (lid.y * 8u + 0u) * p[4] + tap * p[3]];
                let x1 = stage[ci_local * win_len + (lid.y * 8u + 1u) * p[4] + tap * p[3]];
                let x2 = stage[ci_local * win_len + (lid.y * 8u + 2u) * p[4] + tap * p[3]];
                let x3 = stage[ci_local * win_len + (lid.y * 8u + 3u) * p[4] + tap * p[3]];
                let x4 = stage[ci_local * win_len + (lid.y * 8u + 4u) * p[4] + tap * p[3]];
                let x5 = stage[ci_local * win_len + (lid.y * 8u + 5u) * p[4] + tap * p[3]];
                let x6 = stage[ci_local * win_len + (lid.y * 8u + 6u) * p[4] + tap * p[3]];
                let x7 = stage[ci_local * win_len + (lid.y * 8u + 7u) * p[4] + tap * p[3]];
                let wbase = (tap * p[0] + ci) * p[1] + c_base;
                let w0 = w[wbase];
                let w1 = w[wbase + 1u];
                let w2 = w[wbase + 2u];
                let w3 = w[wbase + 3u];
                let w4 = w[wbase + 4u];
                let w5 = w[wbase + 5u];
                let w6 = w[wbase + 6u];
                let w7 = w[wbase + 7u];
                a00 = a00 + x0 * w0;
                a01 = a01 + x0 * w1;
                a02 = a02 + x0 * w2;
                a03 = a03 + x0 * w3;
                a04 = a04 + x0 * w4;
                a05 = a05 + x0 * w5;
                a06 = a06 + x0 * w6;
                a07 = a07 + x0 * w7;
                a10 = a10 + x1 * w0;
                a11 = a11 + x1 * w1;
                a12 = a12 + x1 * w2;
                a13 = a13 + x1 * w3;
                a14 = a14 + x1 * w4;
                a15 = a15 + x1 * w5;
                a16 = a16 + x1 * w6;
                a17 = a17 + x1 * w7;
                a20 = a20 + x2 * w0;
                a21 = a21 + x2 * w1;
                a22 = a22 + x2 * w2;
                a23 = a23 + x2 * w3;
                a24 = a24 + x2 * w4;
                a25 = a25 + x2 * w5;
                a26 = a26 + x2 * w6;
                a27 = a27 + x2 * w7;
                a30 = a30 + x3 * w0;
                a31 = a31 + x3 * w1;
                a32 = a32 + x3 * w2;
                a33 = a33 + x3 * w3;
                a34 = a34 + x3 * w4;
                a35 = a35 + x3 * w5;
                a36 = a36 + x3 * w6;
                a37 = a37 + x3 * w7;
                a40 = a40 + x4 * w0;
                a41 = a41 + x4 * w1;
                a42 = a42 + x4 * w2;
                a43 = a43 + x4 * w3;
                a44 = a44 + x4 * w4;
                a45 = a45 + x4 * w5;
                a46 = a46 + x4 * w6;
                a47 = a47 + x4 * w7;
                a50 = a50 + x5 * w0;
                a51 = a51 + x5 * w1;
                a52 = a52 + x5 * w2;
                a53 = a53 + x5 * w3;
                a54 = a54 + x5 * w4;
                a55 = a55 + x5 * w5;
                a56 = a56 + x5 * w6;
                a57 = a57 + x5 * w7;
                a60 = a60 + x6 * w0;
                a61 = a61 + x6 * w1;
                a62 = a62 + x6 * w2;
                a63 = a63 + x6 * w3;
                a64 = a64 + x6 * w4;
                a65 = a65 + x6 * w5;
                a66 = a66 + x6 * w6;
                a67 = a67 + x6 * w7;
                a70 = a70 + x7 * w0;
                a71 = a71 + x7 * w1;
                a72 = a72 + x7 * w2;
                a73 = a73 + x7 * w3;
                a74 = a74 + x7 * w4;
                a75 = a75 + x7 * w5;
                a76 = a76 + x7 * w6;
                a77 = a77 + x7 * w7;
            }
        }
        workgroupBarrier();
    }

    // Epilogue: bias + bounds-checked store, 8x8 per thread (channels-last y).
        {
            let t = t_base + 0u;
            let co = c_base + 0u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a00 + b[co];
            }
        }
        {
            let t = t_base + 0u;
            let co = c_base + 1u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a01 + b[co];
            }
        }
        {
            let t = t_base + 0u;
            let co = c_base + 2u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a02 + b[co];
            }
        }
        {
            let t = t_base + 0u;
            let co = c_base + 3u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a03 + b[co];
            }
        }
        {
            let t = t_base + 0u;
            let co = c_base + 4u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a04 + b[co];
            }
        }
        {
            let t = t_base + 0u;
            let co = c_base + 5u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a05 + b[co];
            }
        }
        {
            let t = t_base + 0u;
            let co = c_base + 6u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a06 + b[co];
            }
        }
        {
            let t = t_base + 0u;
            let co = c_base + 7u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a07 + b[co];
            }
        }
        {
            let t = t_base + 1u;
            let co = c_base + 0u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a10 + b[co];
            }
        }
        {
            let t = t_base + 1u;
            let co = c_base + 1u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a11 + b[co];
            }
        }
        {
            let t = t_base + 1u;
            let co = c_base + 2u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a12 + b[co];
            }
        }
        {
            let t = t_base + 1u;
            let co = c_base + 3u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a13 + b[co];
            }
        }
        {
            let t = t_base + 1u;
            let co = c_base + 4u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a14 + b[co];
            }
        }
        {
            let t = t_base + 1u;
            let co = c_base + 5u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a15 + b[co];
            }
        }
        {
            let t = t_base + 1u;
            let co = c_base + 6u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a16 + b[co];
            }
        }
        {
            let t = t_base + 1u;
            let co = c_base + 7u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a17 + b[co];
            }
        }
        {
            let t = t_base + 2u;
            let co = c_base + 0u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a20 + b[co];
            }
        }
        {
            let t = t_base + 2u;
            let co = c_base + 1u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a21 + b[co];
            }
        }
        {
            let t = t_base + 2u;
            let co = c_base + 2u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a22 + b[co];
            }
        }
        {
            let t = t_base + 2u;
            let co = c_base + 3u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a23 + b[co];
            }
        }
        {
            let t = t_base + 2u;
            let co = c_base + 4u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a24 + b[co];
            }
        }
        {
            let t = t_base + 2u;
            let co = c_base + 5u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a25 + b[co];
            }
        }
        {
            let t = t_base + 2u;
            let co = c_base + 6u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a26 + b[co];
            }
        }
        {
            let t = t_base + 2u;
            let co = c_base + 7u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a27 + b[co];
            }
        }
        {
            let t = t_base + 3u;
            let co = c_base + 0u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a30 + b[co];
            }
        }
        {
            let t = t_base + 3u;
            let co = c_base + 1u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a31 + b[co];
            }
        }
        {
            let t = t_base + 3u;
            let co = c_base + 2u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a32 + b[co];
            }
        }
        {
            let t = t_base + 3u;
            let co = c_base + 3u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a33 + b[co];
            }
        }
        {
            let t = t_base + 3u;
            let co = c_base + 4u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a34 + b[co];
            }
        }
        {
            let t = t_base + 3u;
            let co = c_base + 5u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a35 + b[co];
            }
        }
        {
            let t = t_base + 3u;
            let co = c_base + 6u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a36 + b[co];
            }
        }
        {
            let t = t_base + 3u;
            let co = c_base + 7u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a37 + b[co];
            }
        }
        {
            let t = t_base + 4u;
            let co = c_base + 0u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a40 + b[co];
            }
        }
        {
            let t = t_base + 4u;
            let co = c_base + 1u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a41 + b[co];
            }
        }
        {
            let t = t_base + 4u;
            let co = c_base + 2u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a42 + b[co];
            }
        }
        {
            let t = t_base + 4u;
            let co = c_base + 3u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a43 + b[co];
            }
        }
        {
            let t = t_base + 4u;
            let co = c_base + 4u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a44 + b[co];
            }
        }
        {
            let t = t_base + 4u;
            let co = c_base + 5u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a45 + b[co];
            }
        }
        {
            let t = t_base + 4u;
            let co = c_base + 6u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a46 + b[co];
            }
        }
        {
            let t = t_base + 4u;
            let co = c_base + 7u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a47 + b[co];
            }
        }
        {
            let t = t_base + 5u;
            let co = c_base + 0u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a50 + b[co];
            }
        }
        {
            let t = t_base + 5u;
            let co = c_base + 1u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a51 + b[co];
            }
        }
        {
            let t = t_base + 5u;
            let co = c_base + 2u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a52 + b[co];
            }
        }
        {
            let t = t_base + 5u;
            let co = c_base + 3u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a53 + b[co];
            }
        }
        {
            let t = t_base + 5u;
            let co = c_base + 4u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a54 + b[co];
            }
        }
        {
            let t = t_base + 5u;
            let co = c_base + 5u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a55 + b[co];
            }
        }
        {
            let t = t_base + 5u;
            let co = c_base + 6u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a56 + b[co];
            }
        }
        {
            let t = t_base + 5u;
            let co = c_base + 7u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a57 + b[co];
            }
        }
        {
            let t = t_base + 6u;
            let co = c_base + 0u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a60 + b[co];
            }
        }
        {
            let t = t_base + 6u;
            let co = c_base + 1u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a61 + b[co];
            }
        }
        {
            let t = t_base + 6u;
            let co = c_base + 2u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a62 + b[co];
            }
        }
        {
            let t = t_base + 6u;
            let co = c_base + 3u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a63 + b[co];
            }
        }
        {
            let t = t_base + 6u;
            let co = c_base + 4u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a64 + b[co];
            }
        }
        {
            let t = t_base + 6u;
            let co = c_base + 5u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a65 + b[co];
            }
        }
        {
            let t = t_base + 6u;
            let co = c_base + 6u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a66 + b[co];
            }
        }
        {
            let t = t_base + 6u;
            let co = c_base + 7u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a67 + b[co];
            }
        }
        {
            let t = t_base + 7u;
            let co = c_base + 0u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a70 + b[co];
            }
        }
        {
            let t = t_base + 7u;
            let co = c_base + 1u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a71 + b[co];
            }
        }
        {
            let t = t_base + 7u;
            let co = c_base + 2u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a72 + b[co];
            }
        }
        {
            let t = t_base + 7u;
            let co = c_base + 3u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a73 + b[co];
            }
        }
        {
            let t = t_base + 7u;
            let co = c_base + 4u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a74 + b[co];
            }
        }
        {
            let t = t_base + 7u;
            let co = c_base + 5u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a75 + b[co];
            }
        }
        {
            let t = t_base + 7u;
            let co = c_base + 6u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a76 + b[co];
            }
        }
        {
            let t = t_base + 7u;
            let co = c_base + 7u;
            if (t < p[7] && co < p[1]) {
                y[t * p[1] + co] = a77 + b[co];
            }
        }
}
"#;

/// Arena variant of RMS_NORM_F32: params at binding 0 (uniform, dynamic offset).
pub const FUSED_RMS: &str = r#"
// 0=rows 1=cols 2=eps(bits) 3=0
struct P { a: u32, b: u32, c: u32, d: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = p.b;
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
        let eps = bitcast<f32>(p.c);
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

/// Arena variant of GEMV_F32: params at binding 0 (uniform, dynamic offset).
pub const FUSED_GEMV: &str = r#"
// Shared with LINEAR_F32: 0=m 1=n 2=k 3=has_bias
struct P { a: u32, b: u32, c: u32, d: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wid.x * 8u + warp;
    let kdim = p.c;
    var acc = 0.0;
    if (row < p.b) {
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
    if (lane == 0u && row < p.b) {
        var sum = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + red[warp][i]; }
        if (p.d == 1u) { sum = sum + b[row]; }
        y[row] = sum;
    }
}
"#;

/// Arena variant of QKV_ATTEND_F32: params at binding 0 (uniform, dynamic offset).
pub const FUSED_QKV_ATTEND: &str = r#"
// 0=q_rows 1=kv_rows 2=head_dim 3=position 4=heads
struct P { a: u32, b: u32, c: u32, d: u32, e: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache_k: array<f32>;
@group(0) @binding(3) var<storage, read_write> cache_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> attended: array<f32>;
@group(0) @binding(5) var<storage, read> cos_t: array<f32>;
@group(0) @binding(6) var<storage, read> sin_t: array<f32>;
var<workgroup> qs: array<f32, 64>;
var<workgroup> ks: array<f32, 64>;
var<workgroup> scores: array<f32, 1024>;
var<workgroup> red123: array<f32, 1>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wid.x;
    let hd = p.c;   // head_dim (p.b is kv_rows)
    let t = lid.x;
    let qo = head * hd;
    let kvh = head / (p.a / p.b);  // heads / kv_heads

    // Load this head's projected q and its kv head's k/v rows.
    qs[t] = qkv[qo + t];
    ks[t] = qkv[p.a + kvh * hd + t];
    let vv = qkv[p.a + p.b + kvh * hd + t];

    // Rotate-half rope inside the 64-dim head (t < 32 owns the (j, j+32) pair).
    // The cos/sin table covers every position; this layer reads row `p.c`.
    workgroupBarrier();
    if (t < hd / 2u) {
        let c = cos_t[p.d * hd + t];
        let sn = sin_t[p.d * hd + t];
        let ha = qs[t]; let hb = qs[t + hd / 2u];
        qs[t] = ha * c - hb * sn;
        qs[t + hd / 2u] = hb * c + ha * sn;
        let ka = ks[t]; let kb = ks[t + hd / 2u];
        ks[t] = ka * c - kb * sn;
        ks[t + hd / 2u] = kb * c + ka * sn;
    }
    workgroupBarrier();

    // Append to the cache at this position (expanded q-head layout, disjoint).
    let off = p.d * p.a + qo;
    cache_k[off + t] = ks[t];
    cache_v[off + t] = vv;
    workgroupBarrier();

    // Attention over rows 0..=position: scores in shared, softmax in thread 0,
    // weighted sum per thread over this head's value slice.
    let len = p.d + 1u;
    let scale = 1.0 / sqrt(f32(hd));
    for (var j = t; j < len; j = j + 64u) {
        var acc = 0.0;
        for (var i = 0u; i < hd; i = i + 1u) { acc = acc + qs[i] * cache_k[j * p.a + qo + i]; }
        scores[j] = acc * scale;
    }
    workgroupBarrier();
    if (t == 0u) {
        var maxs = -1e30;
        for (var j = 0u; j < len; j = j + 1u) { maxs = max(maxs, scores[j]); }
        var sum = 0.0;
        for (var j = 0u; j < len; j = j + 1u) {
            let e = exp(scores[j] - maxs);
            scores[j] = e;
            sum = sum + e;
        }
        red123[0u] = 1.0 / sum;
    }
    workgroupBarrier();
    let inv_sum = red123[0u];
    if (t < hd) {
        var acc = 0.0;
        for (var j = 0u; j < len; j = j + 1u) {
            acc = acc + scores[j] * cache_v[j * p.a + qo + t];
        }
        attended[qo + t] = acc * inv_sum;
    }
}
"#;

/// Arena variant of GEMV_ADD_F32: params at binding 0 (uniform, dynamic offset).
pub const FUSED_GEMV_ADD: &str = r#"
// 0=n 1=k
struct P { a: u32, b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> res: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wid.x * 8u + warp;
    var acc = 0.0;
    if (row < p.a) {
        let base = row * p.b;
        var i = lane;
        loop {
            if (i >= p.b) { break; }
            acc = acc + x[i] * w[base + i];
            i = i + 32u;
        }
    }
    red[warp][lane] = acc;
    workgroupBarrier();
    if (lane == 0u && row < p.a) {
        var sum = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + red[warp][i]; }
        y[row] = sum + res[row];
    }
}
"#;

/// Arena variant of NORM_GATEUP_F32: params at binding 0 (uniform, dynamic offset).
pub const FUSED_NORM_GATEUP: &str = r#"
// 0=inter 1=hidden 2=eps_bits
struct P { a: u32, b: u32, c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> norm_w: array<f32>;
@group(0) @binding(3) var<storage, read> gate_w: array<f32>;
@group(0) @binding(4) var<storage, read> up_w: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
var<workgroup> normed: array<f32, 512>;
var<workgroup> nred: array<f32, 256>;
var<workgroup> wred: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let hidden = p.b;
    let inter = p.a;
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;

    // One norm per 64-row workgroup: 512 loads amortised over 8 warps x 8 rows.
    var sq = 0.0;
    for (var i = lid.x; i < hidden; i = i + 256u) { sq = sq + x[i] * x[i]; }
    nred[lid.x] = sq;
    workgroupBarrier();
    if (lid.x == 0u) {
        var sum = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) { sum = sum + nred[i]; }
        nred[0u] = 1.0 / sqrt(sum / f32(hidden) + bitcast<f32>(p.c));
    }
    workgroupBarrier();
    let inv = nred[0u];
    for (var i = lid.x; i < hidden; i = i + 256u) {
        normed[i] = x[i] * inv * norm_w[i];
    }
    workgroupBarrier();

    // Eight warps x eight rows: warp `w` handles rows wg*64 + w + 8*r.
    for (var r = 0u; r < 8u; r = r + 1u) {
        let row = wid.x * 64u + warp + 8u * r;
        if (row < 2u * inter) {
            var acc = 0.0;
            if (row < inter) {
                let wrow = row * hidden;
                for (var i = lane; i < hidden; i = i + 32u) {
                    acc = acc + normed[i] * gate_w[wrow + i];
                }
            } else {
                let wrow = (row - inter) * hidden;
                for (var i = lane; i < hidden; i = i + 32u) {
                    acc = acc + normed[i] * up_w[wrow + i];
                }
            }
            wred[warp][lane] = acc;
            workgroupBarrier();
            if (lane == 0u) {
                var sum = 0.0;
                for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + wred[warp][i]; }
                y[row] = sum;
            }
            workgroupBarrier();
        }
    }
}
"#;

/// Arena variant of SILU_DOWN_F32: params at binding 0 (uniform, dynamic offset).
pub const FUSED_SILU_DOWN: &str = r#"
// 0=inter 1=hidden
struct P { a: u32, b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> gate_up: array<f32>;
@group(0) @binding(2) var<storage, read> down_w: array<f32>;
@group(0) @binding(3) var<storage, read> x1: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let inter = p.a;
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wid.x * 8u + warp;
    var acc = 0.0;
    if (row < p.b) {
        let base = row * inter;
        var i = lane;
        loop {
            if (i >= inter) { break; }
            let g = gate_up[i];
            let u = gate_up[inter + i];
            let act = g / (1.0 + exp(-g)) * u;
            acc = acc + act * down_w[base + i];
            i = i + 32u;
        }
    }
    red[warp][lane] = acc;
    workgroupBarrier();
    if (lane == 0u && row < p.b) {
        var sum = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + red[warp][i]; }
        out[row] = sum + x1[row];
    }
}
"#;

/// GEMV with a residual add in the epilogue: `y[r] = dot(w[r], x) + res[r]`.
pub const GEMV_ADD_F32: &str = r#"
// 0=n 1=k
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;   // [n, k]
@group(0) @binding(2) var<storage, read> res: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 2>;
var<workgroup> red: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wid.x * 8u + warp;
    var acc = 0.0;
    if (row < p[0]) {
        let base = row * p[1];
        var i = lane;
        loop {
            if (i >= p[1]) { break; }
            acc = acc + x[i] * w[base + i];
            i = i + 32u;
        }
    }
    red[warp][lane] = acc;
    workgroupBarrier();
    if (lane == 0u && row < p[0]) {
        var sum = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + red[warp][i]; }
        y[row] = sum + res[row];
    }
}
"#;

/// Fused RMSNorm -> gate+up projection: one dispatch for what was three.
/// Output rows `[0, inter)` are the gate projection, `[inter, 2*inter)` the up
/// projection; the norm is recomputed per workgroup (512 loads) instead of
/// paying a barrier round-trip between two kernels.
pub const NORM_GATEUP_F32: &str = r#"
// 0=inter 1=hidden 2=eps_bits
@group(0) @binding(0) var<storage, read> x: array<f32>;      // [hidden]
@group(0) @binding(1) var<storage, read> norm_w: array<f32>;
@group(0) @binding(2) var<storage, read> gate_w: array<f32>; // [inter, hidden]
@group(0) @binding(3) var<storage, read> up_w: array<f32>;   // [inter, hidden]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [2*inter]
@group(0) @binding(5) var<storage, read> p: array<u32, 3>;
var<workgroup> normed: array<f32, 512>;
var<workgroup> nred: array<f32, 256>;
var<workgroup> wred: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let hidden = p[1];
    let inter = p[0];
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;

    // One norm per 64-row workgroup: 512 loads amortised over 8 warps x 8 rows.
    var sq = 0.0;
    for (var i = lid.x; i < hidden; i = i + 256u) { sq = sq + x[i] * x[i]; }
    nred[lid.x] = sq;
    workgroupBarrier();
    if (lid.x == 0u) {
        var sum = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) { sum = sum + nred[i]; }
        nred[0u] = 1.0 / sqrt(sum / f32(hidden) + bitcast<f32>(p[2]));
    }
    workgroupBarrier();
    let inv = nred[0u];
    for (var i = lid.x; i < hidden; i = i + 256u) {
        normed[i] = x[i] * inv * norm_w[i];
    }
    workgroupBarrier();

    // Eight warps x eight rows: warp `w` handles rows wg*64 + w + 8*r.
    for (var r = 0u; r < 8u; r = r + 1u) {
        let row = wid.x * 64u + warp + 8u * r;
        if (row < 2u * inter) {
            var acc = 0.0;
            if (row < inter) {
                let wrow = row * hidden;
                for (var i = lane; i < hidden; i = i + 32u) {
                    acc = acc + normed[i] * gate_w[wrow + i];
                }
            } else {
                let wrow = (row - inter) * hidden;
                for (var i = lane; i < hidden; i = i + 32u) {
                    acc = acc + normed[i] * up_w[wrow + i];
                }
            }
            wred[warp][lane] = acc;
            workgroupBarrier();
            if (lane == 0u) {
                var sum = 0.0;
                for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + wred[warp][i]; }
                y[row] = sum;
            }
            workgroupBarrier();
        }
    }
}
"#;

/// Fused silu-mul -> down projection -> residual: `out[r] = x1[r] +
/// dot(silu(g)*u, down_w[r])`. The silu is recomputed per workgroup from the
/// gate/up rows (2048 ops, trivial) so the activation never hits memory.
pub const SILU_DOWN_F32: &str = r#"
// 0=inter 1=hidden
@group(0) @binding(0) var<storage, read> gate_up: array<f32>; // [2*inter]
@group(0) @binding(1) var<storage, read> down_w: array<f32>;  // [hidden, inter]
@group(0) @binding(2) var<storage, read> x1: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>; // [hidden]
@group(0) @binding(4) var<storage, read> p: array<u32, 2>;
var<workgroup> red: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let inter = p[0];
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wid.x * 8u + warp;
    var acc = 0.0;
    if (row < p[1]) {
        let base = row * inter;
        var i = lane;
        loop {
            if (i >= inter) { break; }
            let g = gate_up[i];
            let u = gate_up[inter + i];
            let act = g / (1.0 + exp(-g)) * u;
            acc = acc + act * down_w[base + i];
            i = i + 32u;
        }
    }
    red[warp][lane] = acc;
    workgroupBarrier();
    if (lane == 0u && row < p[1]) {
        var sum = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + red[warp][i]; }
        out[row] = sum + x1[row];
    }
}
"#;

// ---- uniform-arena fused decode kernels (ming nn.rs pattern) ----
// Same math as the storage-params fused kernels above, but the parameter block
// is a uniform at binding 0 selected by a dynamic offset into the shared arena.
// That lets a cached bind group be replayed with a fresh params slot per
// dispatch (no per-dispatch buffer write, no params overwrite when a layer's
// key repeats inside one batch). Storage bindings shift to 1..

/// RMSNorm, uniform params {rows, cols, eps_bits, pad}.
pub const AF_RMS: &str = r#"
struct P { rows: u32, cols: u32, eps_bits: u32, pad: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let cols = p.cols;
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
        let eps = bitcast<f32>(p.eps_bits);
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

/// Single-token GEMV, uniform params {m, n, k, has_bias}.
pub const AF_GEMV: &str = r#"
struct P { m: u32, n: u32, k: u32, has_bias: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wid.x * 8u + warp;
    let kdim = p.k;
    var acc = 0.0;
    if (row < p.n) {
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
    if (lane == 0u && row < p.n) {
        var sum = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + red[warp][i]; }
        if (p.has_bias == 1u) { sum = sum + b[row]; }
        y[row] = sum;
    }
}
"#;

/// Fused rope + KV append + decode attention, uniform params
/// {q_rows, kv_rows, head_dim, position, heads, pad0, pad1, pad2}.
pub const AF_QKV_ATTEND: &str = r#"
struct P { q_rows: u32, kv_rows: u32, head_dim: u32, position: u32,
           heads: u32, pad0: u32, pad1: u32, pad2: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache_k: array<f32>;
@group(0) @binding(3) var<storage, read_write> cache_v: array<f32>;
@group(0) @binding(4) var<storage, read_write> attended: array<f32>;
@group(0) @binding(5) var<storage, read> cos_t: array<f32>;
@group(0) @binding(6) var<storage, read> sin_t: array<f32>;

var<workgroup> qs: array<f32, 64>;
var<workgroup> ks: array<f32, 64>;
var<workgroup> scores: array<f32, 1024>;
var<workgroup> red123: array<f32, 1>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let head = wid.x;
    let hd = p.head_dim;
    let t = lid.x;
    let qo = head * hd;
    let kvh = head / (p.q_rows / p.kv_rows);

    qs[t] = qkv[qo + t];
    ks[t] = qkv[p.q_rows + kvh * hd + t];
    let vv = qkv[p.q_rows + p.kv_rows + kvh * hd + t];

    workgroupBarrier();
    if (t < hd / 2u) {
        let c = cos_t[p.position * hd + t];
        let sn = sin_t[p.position * hd + t];
        let ha = qs[t]; let hb = qs[t + hd / 2u];
        qs[t] = ha * c - hb * sn;
        qs[t + hd / 2u] = hb * c + ha * sn;
        let ka = ks[t]; let kb = ks[t + hd / 2u];
        ks[t] = ka * c - kb * sn;
        ks[t + hd / 2u] = kb * c + ka * sn;
    }
    workgroupBarrier();

    let off = p.position * p.q_rows + qo;
    cache_k[off + t] = ks[t];
    cache_v[off + t] = vv;
    workgroupBarrier();

    let len = p.position + 1u;
    let scale = 1.0 / sqrt(f32(hd));
    for (var j = t; j < len; j = j + 64u) {
        var acc = 0.0;
        for (var i = 0u; i < hd; i = i + 1u) { acc = acc + qs[i] * cache_k[j * p.q_rows + qo + i]; }
        scores[j] = acc * scale;
    }
    workgroupBarrier();
    if (t == 0u) {
        var maxs = -1e30;
        for (var j = 0u; j < len; j = j + 1u) { maxs = max(maxs, scores[j]); }
        var sum = 0.0;
        for (var j = 0u; j < len; j = j + 1u) {
            let e = exp(scores[j] - maxs);
            scores[j] = e;
            sum = sum + e;
        }
        red123[0u] = 1.0 / sum;
    }
    workgroupBarrier();
    let inv_sum = red123[0u];
    if (t < hd) {
        var acc = 0.0;
        for (var j = 0u; j < len; j = j + 1u) {
            acc = acc + scores[j] * cache_v[j * p.q_rows + qo + t];
        }
        attended[qo + t] = acc * inv_sum;
    }
}
"#;

/// GEMV + residual, uniform params {n, k, pad0, pad1}.
pub const AF_GEMV_ADD: &str = r#"
struct P { n: u32, k: u32, pad0: u32, pad1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> res: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wid.x * 8u + warp;
    var acc = 0.0;
    if (row < p.n) {
        let base = row * p.k;
        var i = lane;
        loop {
            if (i >= p.k) { break; }
            acc = acc + x[i] * w[base + i];
            i = i + 32u;
        }
    }
    red[warp][lane] = acc;
    workgroupBarrier();
    if (lane == 0u && row < p.n) {
        var sum = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + red[warp][i]; }
        y[row] = sum + res[row];
    }
}
"#;

/// Fused RMSNorm -> gate+up, uniform params {inter, hidden, eps_bits, pad}.
pub const AF_NORM_GATEUP: &str = r#"
struct P { inter: u32, hidden: u32, eps_bits: u32, pad: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> norm_w: array<f32>;
@group(0) @binding(3) var<storage, read> gate_w: array<f32>;
@group(0) @binding(4) var<storage, read> up_w: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
var<workgroup> normed: array<f32, 512>;
var<workgroup> nred: array<f32, 256>;
var<workgroup> wred: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let hidden = p.hidden;
    let inter = p.inter;
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;

    var sq = 0.0;
    for (var i = lid.x; i < hidden; i = i + 256u) { sq = sq + x[i] * x[i]; }
    nred[lid.x] = sq;
    workgroupBarrier();
    if (lid.x == 0u) {
        var sum = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) { sum = sum + nred[i]; }
        nred[0u] = 1.0 / sqrt(sum / f32(hidden) + bitcast<f32>(p.eps_bits));
    }
    workgroupBarrier();
    let inv = nred[0u];
    for (var i = lid.x; i < hidden; i = i + 256u) {
        normed[i] = x[i] * inv * norm_w[i];
    }
    workgroupBarrier();

    for (var r = 0u; r < 8u; r = r + 1u) {
        let row = wid.x * 64u + warp + 8u * r;
        if (row < 2u * inter) {
            var acc = 0.0;
            if (row < inter) {
                let wrow = row * hidden;
                for (var i = lane; i < hidden; i = i + 32u) {
                    acc = acc + normed[i] * gate_w[wrow + i];
                }
            } else {
                let wrow = (row - inter) * hidden;
                for (var i = lane; i < hidden; i = i + 32u) {
                    acc = acc + normed[i] * up_w[wrow + i];
                }
            }
            wred[warp][lane] = acc;
            workgroupBarrier();
            if (lane == 0u) {
                var sum = 0.0;
                for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + wred[warp][i]; }
                y[row] = sum;
            }
            workgroupBarrier();
        }
    }
}
"#;

/// Fused silu-mul -> down -> residual, uniform params {inter, hidden, pad0, pad1}.
pub const AF_SILU_DOWN: &str = r#"
struct P { inter: u32, hidden: u32, pad0: u32, pad1: u32, }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> gate_up: array<f32>;
@group(0) @binding(2) var<storage, read> down_w: array<f32>;
@group(0) @binding(3) var<storage, read> x1: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let inter = p.inter;
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wid.x * 8u + warp;
    var acc = 0.0;
    if (row < p.hidden) {
        let base = row * inter;
        var i = lane;
        loop {
            if (i >= inter) { break; }
            let g = gate_up[i];
            let u = gate_up[inter + i];
            let act = g / (1.0 + exp(-g)) * u;
            acc = acc + act * down_w[base + i];
            i = i + 32u;
        }
    }
    red[warp][lane] = acc;
    workgroupBarrier();
    if (lane == 0u && row < p.hidden) {
        var sum = 0.0;
        for (var i = 0u; i < 32u; i = i + 1u) { sum = sum + red[warp][i]; }
        out[row] = sum + x1[row];
    }
}
"#;

// ---- bf16-weight fused decode kernels (storage params, cached bind groups) ----
// Weights are resident bf16 packed 2-per-u32 (little-endian pairs); activations
// stay f32. The inner loop walks elements in the SAME order as the f32 kernels
// (lane, lane+32, ...) and expands each bf16 exactly, so the result is
// bit-identical to the f32 path while reading half the weight bytes.

/// GEMV with bf16 weights, params {m, n, k, has_bias}.
pub const GEMV_BF16_F32: &str = r#"
// 0=m 1=n 2=k 3=has_bias
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
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
        let base = row * (kdim / 2u);
        var i = lane;
        loop {
            if (i >= kdim) { break; }
            let u = w[base + (i >> 1u)];
            let v = select(bitcast<f32>(u & 0xFFFF0000u), bitcast<f32>((u & 0xFFFFu) << 16u), (i & 1u) == 0u);
            acc = acc + v * x[i];
            i = i + 32u;
        }
    }
    red[warp][lane] = acc;
    workgroupBarrier();
    if (lane == 0u && row < p[1]) {
        var sum = 0.0;
        for (var j = 0u; j < 32u; j = j + 1u) { sum = sum + red[warp][j]; }
        if (p[3] == 1u) { sum = sum + b[row]; }
        y[row] = sum;
    }
}
"#;

/// GEMV + residual with bf16 weights, params {n, k}.
pub const GEMV_ADD_BF16_F32: &str = r#"
// 0=n 1=k
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> res: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 2>;
var<workgroup> red: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wid.x * 8u + warp;
    var acc = 0.0;
    if (row < p[0]) {
        let base = row * (p[1] / 2u);
        var i = lane;
        loop {
            if (i >= p[1]) { break; }
            let u = w[base + (i >> 1u)];
            let v = select(bitcast<f32>(u & 0xFFFF0000u), bitcast<f32>((u & 0xFFFFu) << 16u), (i & 1u) == 0u);
            acc = acc + v * x[i];
            i = i + 32u;
        }
    }
    red[warp][lane] = acc;
    workgroupBarrier();
    if (lane == 0u && row < p[0]) {
        var sum = 0.0;
        for (var j = 0u; j < 32u; j = j + 1u) { sum = sum + red[warp][j]; }
        y[row] = sum + res[row];
    }
}
"#;

/// Fused RMSNorm -> gate+up with bf16 weights, params {inter, hidden, eps_bits}.
pub const NORM_GATEUP_BF16_F32: &str = r#"
// 0=inter 1=hidden 2=eps_bits
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> norm_w: array<f32>;
@group(0) @binding(2) var<storage, read> gate_w: array<u32>;
@group(0) @binding(3) var<storage, read> up_w: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> p: array<u32, 3>;
var<workgroup> normed: array<f32, 512>;
var<workgroup> nred: array<f32, 256>;
var<workgroup> wred: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let hidden = p[1];
    let inter = p[0];
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;

    var sq = 0.0;
    for (var i = lid.x; i < hidden; i = i + 256u) { sq = sq + x[i] * x[i]; }
    nred[lid.x] = sq;
    workgroupBarrier();
    if (lid.x == 0u) {
        var sum = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) { sum = sum + nred[i]; }
        nred[0u] = 1.0 / sqrt(sum / f32(hidden) + bitcast<f32>(p[2]));
    }
    workgroupBarrier();
    let inv = nred[0u];
    for (var i = lid.x; i < hidden; i = i + 256u) {
        normed[i] = x[i] * inv * norm_w[i];
    }
    workgroupBarrier();

    let half = hidden / 2u;
    for (var r = 0u; r < 8u; r = r + 1u) {
        let row = wid.x * 64u + warp + 8u * r;
        if (row < 2u * inter) {
            var acc = 0.0;
            if (row < inter) {
                let wrow = row * half;
                for (var i = lane; i < hidden; i = i + 32u) {
                    let u = gate_w[wrow + (i >> 1u)];
                    let v = select(bitcast<f32>(u & 0xFFFF0000u), bitcast<f32>((u & 0xFFFFu) << 16u), (i & 1u) == 0u);
                    acc = acc + normed[i] * v;
                }
            } else {
                let wrow = (row - inter) * half;
                for (var i = lane; i < hidden; i = i + 32u) {
                    let u = up_w[wrow + (i >> 1u)];
                    let v = select(bitcast<f32>(u & 0xFFFF0000u), bitcast<f32>((u & 0xFFFFu) << 16u), (i & 1u) == 0u);
                    acc = acc + normed[i] * v;
                }
            }
            wred[warp][lane] = acc;
            workgroupBarrier();
            if (lane == 0u) {
                var sum = 0.0;
                for (var j = 0u; j < 32u; j = j + 1u) { sum = sum + wred[warp][j]; }
                y[row] = sum;
            }
            workgroupBarrier();
        }
    }
}
"#;

/// Fused silu-mul -> down -> residual with bf16 weights, params {inter, hidden}.
pub const SILU_DOWN_BF16_F32: &str = r#"
// 0=inter 1=hidden
@group(0) @binding(0) var<storage, read> gate_up: array<f32>;
@group(0) @binding(1) var<storage, read> down_w: array<u32>;
@group(0) @binding(2) var<storage, read> x1: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> p: array<u32, 2>;
var<workgroup> red: array<array<f32, 32>, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let inter = p[0];
    let warp = lid.x / 32u;
    let lane = lid.x % 32u;
    let row = wid.x * 8u + warp;
    var acc = 0.0;
    if (row < p[1]) {
        let base = row * (inter / 2u);
        var i = lane;
        loop {
            if (i >= inter) { break; }
            let g = gate_up[i];
            let u = gate_up[inter + i];
            let act = g / (1.0 + exp(-g)) * u;
            let wu = down_w[base + (i >> 1u)];
            let v = select(bitcast<f32>(wu & 0xFFFF0000u), bitcast<f32>((wu & 0xFFFFu) << 16u), (i & 1u) == 0u);
            acc = acc + act * v;
            i = i + 32u;
        }
    }
    red[warp][lane] = acc;
    workgroupBarrier();
    if (lane == 0u && row < p[1]) {
        var sum = 0.0;
        for (var j = 0u; j < 32u; j = j + 1u) { sum = sum + red[warp][j]; }
        out[row] = sum + x1[row];
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
// MAX_CONTEXT = 2048 (crate::lm::MAX_CONTEXT); the host rejects longer requests.
var<workgroup> scores: array<f32, 2048>;

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
/// The cutoff is the exact k-th largest score, found by radix select on the
/// order-preserving u32 image of the f32 bits: four 8-bit passes, each one
/// O(vocab/256) scan plus a 256-bin walk, so ~50 element visits per lane
/// instead of the O(vocab^2) ranking this kernel used to do (2.8 ms a call,
/// 45 ms of a 59 ms depth frame; the whole decode loop samples 16 times).
/// max / sum / argmax then stay single strided passes; their reduction order is
/// unchanged from the parity-verified version, so results are bit-identical.
pub const SAMPLE_TOPK_F32: &str = r#"
// 0=vocab 1=topk 2=temperature(bits) 3=allow_eos*2 + out_offset 4=noise row offset
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> noise: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<storage, read> p: array<u32, 5>;
var<workgroup> bins: array<atomic<u32>, 256>;
var<workgroup> part: array<f32, 256>;
var<workgroup> pidx: array<u32, 256>;
// 0 = key prefix chosen so far, 1 = rank still to skip inside that prefix
var<workgroup> state: array<u32, 2>;

fn score(i: u32) -> f32 {
    if (i == 0u && (p[3] & 1u) == 0u) { return -1e5; }
    return logits[i] / bitcast<f32>(p[2]);
}

// Monotone map f32 -> u32: comparing keys orders the floats, ties and all.
fn okey(v: f32) -> u32 {
    let b = bitcast<u32>(v);
    if ((b & 0x80000000u) != 0u) { return ~b; }
    return b | 0x80000000u;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let vocab = p[0];
    let topk = min(p[1], vocab);
    if (lane == 0u) {
        state[0] = 0u;
        state[1] = topk - 1u;
    }
    atomicStore(&bins[lane], 0u);
    workgroupBarrier();

    // Radix select, most significant digit first. `prefix` is the high digits
    // already fixed, so digit_pass p only counts keys inside that prefix; `rank` is
    // the number of keys still to skip inside it.
    for (var digit_pass = 0u; digit_pass < 4u; digit_pass = digit_pass + 1u) {
        let shift = 24u - 8u * digit_pass;
        let prefix = state[0];
        var i = lane;
        loop {
            if (i >= vocab) { break; }
            let key = okey(score(i));
            if (((key >> shift) >> 8u) == prefix) {
                atomicAdd(&bins[(key >> shift) & 255u], 1u);
            }
            i = i + 256u;
        }
        workgroupBarrier();
        if (lane == 0u) {
            var acc = 0u;
            let rank = state[1];
            for (var d = 255u; ; d = d - 1u) {
                let c = atomicLoad(&bins[d]);
                if (acc + c > rank) {
                    state[0] = (prefix << 8u) | d;
                    state[1] = rank - acc;
                    break;
                }
                acc = acc + c;
            }
        }
        workgroupBarrier();
        atomicStore(&bins[lane], 0u);
        workgroupBarrier();
    }
    let cutoff = state[0];

    // Parallel max over surviving scores.
    var local_max = -1e30;
    var i = lane;
    loop {
        if (i >= vocab) { break; }
        let v = score(i);
        if (okey(v) >= cutoff && v > local_max) { local_max = v; }
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
        if (okey(v) >= cutoff) { local_sum = local_sum + exp(v - maxv); }
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

    // argmax(probs / noise) over survivors: per-lane best, then a reduce that
    // breaks exact ties towards the lower index, i.e. the same token the old
    // serial scan produced.
    var best_ratio = -1e30;
    var best = 0u;
    i = lane;
    loop {
        if (i >= vocab) { break; }
        let v = score(i);
        if (okey(v) >= cutoff) {
            let ratio = (exp(v - maxv) / total) / noise[p[4] + i];
            if (ratio > best_ratio) { best_ratio = ratio; best = i; }
        }
        i = i + 256u;
    }
    part[lane] = best_ratio;
    pidx[lane] = best;
    workgroupBarrier();
    if (lane == 0u) {
        var pick_ratio = -1e30;
        var pick = 0u;
        for (var j = 0u; j < 256u; j = j + 1u) {
            let r = part[j];
            let index = pidx[j];
            if (r > pick_ratio || (r == pick_ratio && index < pick)) {
                pick_ratio = r;
                pick = index;
            }
        }
        out[p[3] >> 1u] = pick;
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
