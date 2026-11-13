//! wgpu launchers for the kernels in [`crate::shaders`].
//!
//! Nothing here knows about model structure: it takes buffers and shapes. The
//! device-independent API lives in [`crate::ops`].

use crate::ops::Conv1d;
use crate::{gpu::Gpu, shaders};
use wgpu::Buffer;

/// Workgroup grid for a flat kernel: x is capped at 65535, y carries the rest
/// (a single x dimension overflows on the last SEANet block, 18M elements).
fn grid(total: usize) -> (u32, u32, u32) {
    let groups = total.div_ceil(128).max(1);
    ((groups.min(65535)) as u32, groups.div_ceil(65535) as u32, 1)
}


/// `y[m,n] = x[m,k] · w[n,k]^T + bias[n]`.
///
/// Small products (the per-frame GEMVs) keep the simple kernel: a 64×64 tile for
/// m = 1 wastes 63/64 of its rows. Anything with a few dozen rows goes through the
/// register-blocked tile, which is where the prefill and the codec's projections sit.
pub fn linear(gpu: &Gpu, x: &Buffer, w: &Buffer, bias: Option<&Buffer>, y: &Buffer,
              m: usize, n: usize, k: usize) {
    let params = gpu.upload_u32("linear.params", &[m as u32, n as u32, k as u32, bias.is_some() as u32]);
    let zero = gpu.empty("linear.no_bias", 1);
    let b = bias.unwrap_or(&zero);
    // A register-blocked 64x64 tile was implemented and measured *slower* here
    // (44 vs 86 GFLOP/s at 2048^3; see examples/gpu_probe.rs for the card's
    // dispatch and GEMV cost). On this P104-100 a dispatch costs ~20 us and a
    // 512x512 GEMV 88 us, so the kernel schedule is not what limits us.
    if m == 1 {
        // One warp per output row: the tiled kernel would compute 16 rows and use 1.
        gpu.dispatch("gemv_f32", shaders::GEMV_F32, "main",
                     &[(0, x), (1, w), (2, b), (3, y), (4, &params)],
                     (n.div_ceil(8) as u32, 1, 1));
        return;
    }
    gpu.dispatch("linear_f32", shaders::LINEAR_F32, "main",
                 &[(0, x), (1, w), (2, b), (3, y), (4, &params)],
                 (m.div_ceil(16) as u32, n.div_ceil(16) as u32, 1));
}

/// Causal conv1d over `[channels, time]`.
pub fn conv1d(gpu: &Gpu, x: &Buffer, w: &Buffer, bias: &Buffer, y: &Buffer, p: &Conv1d) {
    let params = gpu.upload_u32("conv1d.params", &[
        p.cin as u32, p.cout as u32, p.ks as u32, p.dilation as u32,
        p.stride as u32, p.pad as u32, p.t_in as u32, p.t_out as u32,
    ]);
    let total = (p.cout * p.t_out) as u32;
    gpu.dispatch("conv1d_f32", shaders::CONV1D_F32, "main",
                 &[(0, x), (1, w), (2, bias), (3, y), (4, &params)],
                 grid(total as usize));
}

/// SnakeBeta, in place, `[channels, time]`.
pub fn snake_beta(gpu: &Gpu, x: &Buffer, alpha: &Buffer, beta: &Buffer, channels: usize, time: usize) {
    let params = gpu.upload_u32("snake.params", &[channels as u32, time as u32]);
    gpu.dispatch("snake_beta_f32", shaders::SNAKE_BETA_F32, "main",
                 &[(0, x), (1, alpha), (2, beta), (3, &params)],
                 grid(channels * time));
}

/// `y = a + b`.
pub fn add(gpu: &Gpu, a: &Buffer, b: &Buffer, y: &Buffer, len: usize) {
    let params = gpu.upload_u32("add.params", &[len as u32]);
    gpu.dispatch("add_f32", shaders::ADD_F32, "main",
                 &[(0, a), (1, b), (2, y), (3, &params)],
                 grid(len));
}

/// `y += x`, in place. Uses its own kernel: binding `y` as both read and
/// read_write in one dispatch is a wgpu validation error.
pub fn residual(gpu: &Gpu, y: &Buffer, x: &Buffer, len: usize) {
    let params = gpu.upload_u32("residual.params", &[len as u32]);
    gpu.dispatch("residual_f32", crate::shaders::RESIDUAL_F32, "main",
                 &[(0, y), (1, x), (2, &params)],
                 grid(len));
}

/// RVQ dequantise-and-sum into a `[dim, frames]` tensor.
pub fn gather_sum(gpu: &Gpu, codes: &Buffer, tables: &Buffer, usage: &Buffer, out: &Buffer,
                  dim: usize, frames: usize, k_start: usize, k_count: usize,
                  codebook_size: usize, num_codebooks: usize) {
    let params = gpu.upload_u32("gather_sum.params", &[
        dim as u32, frames as u32, k_start as u32, k_count as u32,
        codebook_size as u32, num_codebooks as u32, 0, 0,
    ]);
    gpu.dispatch("gather_sum_f32", shaders::GATHER_SUM_F32, "main",
                 &[(0, codes), (1, tables), (2, usage), (3, out), (4, &params)],
                 grid(dim * frames));
}

/// `[rows, cols] -> [cols, rows]`.
pub fn transpose(gpu: &Gpu, x: &Buffer, y: &Buffer, rows: usize, cols: usize) {
    let params = gpu.upload_u32("transpose.params", &[rows as u32, cols as u32, 0, 0]);
    gpu.dispatch("transpose_f32", shaders::TRANSPOSE_F32, "main",
                 &[(0, x), (1, y), (2, &params)],
                 (rows.div_ceil(16) as u32, cols.div_ceil(16) as u32, 1));
}

/// Row-wise RMSNorm.
pub fn rms_norm(gpu: &Gpu, x: &Buffer, w: &Buffer, y: &Buffer, rows: usize, cols: usize, eps: f32) {
    let params = gpu.upload_u32("rms_norm.params", &[rows as u32, cols as u32, eps.to_bits(), 0]);
    gpu.dispatch("rms_norm_f32", shaders::RMS_NORM_F32, "main",
                 &[(0, x), (1, w), (2, y), (3, &params)], (rows as u32, 1, 1));
}

/// Row-wise LayerNorm with affine parameters.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm(gpu: &Gpu, x: &Buffer, w: &Buffer, b: &Buffer, y: &Buffer,
                  rows: usize, cols: usize, eps: f32) {
    let params = gpu.upload_u32("layer_norm.params", &[rows as u32, cols as u32, eps.to_bits(), 0]);
    gpu.dispatch("layer_norm_f32", shaders::LAYER_NORM_F32, "main",
                 &[(0, x), (1, w), (2, b), (3, y), (4, &params)], (rows as u32, 1, 1));
}

/// `y = silu(gate) * up`.
pub fn silu_mul(gpu: &Gpu, gate: &Buffer, up: &Buffer, y: &Buffer, len: usize) {
    let params = gpu.upload_u32("silu_mul.params", &[len as u32]);
    gpu.dispatch("silu_mul_f32", shaders::SILU_MUL_F32, "main",
                 &[(0, gate), (1, up), (2, y), (3, &params)], grid(len));
}

/// In-place per-channel scale on `[rows, cols]`.
pub fn scale_cols(gpu: &Gpu, x: &Buffer, scale: &Buffer, rows: usize, cols: usize) {
    let params = gpu.upload_u32("scale_cols.params", &[rows as u32, cols as u32]);
    gpu.dispatch("scale_cols_f32", shaders::SCALE_COLS_F32, "main",
                 &[(0, x), (1, scale), (2, &params)],
                 grid(rows * cols));
}

/// `y = a + scale[c] * b`.
pub fn add_scaled(gpu: &Gpu, a: &Buffer, b: &Buffer, scale: &Buffer, y: &Buffer, rows: usize, cols: usize) {
    let params = gpu.upload_u32("add_scaled.params", &[rows as u32, cols as u32]);
    gpu.dispatch("add_scaled_f32", shaders::ADD_SCALED_F32, "main",
                 &[(0, a), (1, b), (2, scale), (3, y), (4, &params)],
                 grid(rows * cols));
}

/// Rotary embedding, in place.
pub fn rope(gpu: &Gpu, x: &Buffer, frames: usize, heads: usize, head_dim: usize, theta: f32) {
    let params = gpu.upload_u32("rope.params", &[frames as u32, heads as u32, head_dim as u32, theta.to_bits()]);
    let total = (frames * heads * (head_dim / 2)) as u32;
    gpu.dispatch("rope_f32", shaders::ROPE_F32, "main",
                 &[(0, x), (1, &params)], grid(total as usize));
}

/// Causal sliding-window attention over `[T, heads*head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn attention(gpu: &Gpu, q: &Buffer, k: &Buffer, v: &Buffer, out: &Buffer,
                 frames: usize, heads: usize, head_dim: usize, window: usize) {
    let params = gpu.upload_u32("attention.params", &[frames as u32, heads as u32, head_dim as u32, window as u32]);
    gpu.dispatch("attention_f32", shaders::ATTENTION_F32, "main",
                 &[(0, q), (1, k), (2, v), (3, out), (4, &params)],
                 ((frames * heads) as u32, 1, 1));
}

/// Depthwise causal conv1d.
pub fn conv1d_depthwise(gpu: &Gpu, x: &Buffer, w: &Buffer, b: &Buffer, y: &Buffer,
                        channels: usize, ks: usize, pad: usize, t_in: usize, t_out: usize) {
    // The shader declares `array<u32, 8>`; the binding must be that large.
    let params = gpu.upload_u32("conv1d_dw.params",
                               &[channels as u32, ks as u32, pad as u32, t_in as u32, t_out as u32, 0, 0, 0]);
    let total = (channels * t_out) as u32;
    gpu.dispatch("conv1d_dw_f32", shaders::CONV1D_DW_F32, "main",
                 &[(0, x), (1, w), (2, b), (3, y), (4, &params)], grid(total as usize));
}

/// Transposed conv1d (upsampling).
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(gpu: &Gpu, x: &Buffer, w: &Buffer, b: &Buffer, y: &Buffer,
                        cin: usize, cout: usize, ks: usize, stride: usize, t_in: usize) {
    let t_out = t_in * stride;
    let params = gpu.upload_u32("conv_transpose.params",
                               &[cin as u32, cout as u32, ks as u32, stride as u32, t_in as u32, t_out as u32,
                                 0, 0]);
    let total = (cout * t_out) as u32;
    gpu.dispatch("conv_transpose_f32", shaders::CONV_TRANSPOSE_F32, "main",
                 &[(0, x), (1, w), (2, b), (3, y), (4, &params)], grid(total as usize));
}

/// Exact GELU, in place.
pub fn gelu(gpu: &Gpu, x: &Buffer, len: usize) {
    let params = gpu.upload_u32("gelu.params", &[len as u32]);
    gpu.dispatch("gelu_f32", shaders::GELU_F32, "main",
                 &[(0, x), (1, &params)], grid(len));
}

/// `y = x`.
pub fn copy(gpu: &Gpu, x: &Buffer, y: &Buffer, len: usize) {
    let params = gpu.upload_u32("copy.params", &[len as u32]);
    gpu.dispatch("copy_f32", shaders::COPY_F32, "main",
                 &[(0, x), (1, y), (2, &params)], grid(len));
}

/// Copy `rows` rows of `src` into `dst` at row `offset`.
pub fn write_rows(gpu: &Gpu, dst: &Buffer, src: &Buffer, cols: usize, offset: usize, rows: usize) {
    let params = gpu.upload_u32("write_rows.params", &[cols as u32, offset as u32, rows as u32, 0]);
    gpu.dispatch("write_rows_f32", shaders::WRITE_ROWS_F32, "main",
                 &[(0, dst), (1, src), (2, &params)], grid(cols * rows));
}

/// Single-query attention over a KV cache.
pub fn attention_decode(gpu: &Gpu, q: &Buffer, k: &Buffer, v: &Buffer, out: &Buffer,
                        len: usize, heads: usize, head_dim: usize) {
    let params = gpu.upload_u32("attention_decode.params",
                               &[len as u32, heads as u32, head_dim as u32, 0]);
    gpu.dispatch("attention_decode_f32", shaders::ATTENTION_DECODE_F32, "main",
                 &[(0, q), (1, k), (2, v), (3, out), (4, &params)], (heads as u32, 1, 1));
}

/// Rotary embedding from a cos/sin table.
pub fn rope_tabled(gpu: &Gpu, x: &Buffer, cos: &Buffer, sin: &Buffer,
                   frames: usize, heads: usize, head_dim: usize) {
    let params = gpu.upload_u32("rope_tabled.params", &[frames as u32, heads as u32, head_dim as u32, 0]);
    let total = frames * heads * (head_dim / 2);
    gpu.dispatch("rope_tabled_f32", shaders::ROPE_TABLED_F32, "main",
                 &[(0, x), (1, cos), (2, sin), (3, &params)], grid(total));
}

/// Grouped-query KV repeat.
pub fn repeat_kv(gpu: &Gpu, x: &Buffer, y: &Buffer, frames: usize, kv_heads: usize, heads: usize, head_dim: usize) {
    let params = gpu.upload_u32("repeat_kv.params", &[frames as u32, kv_heads as u32, heads as u32, head_dim as u32]);
    gpu.dispatch("repeat_kv_f32", shaders::REPEAT_KV_F32, "main",
                 &[(0, x), (1, y), (2, &params)], grid(frames * heads * head_dim));
}

/// ELU (alpha 1), in place.
pub fn elu(gpu: &Gpu, x: &Buffer, len: usize) {
    let params = gpu.upload_u32("elu.params", &[len as u32]);
    gpu.dispatch("elu_f32", shaders::ELU_F32, "main",
                 &[(0, x), (1, &params)], grid(len));
}
