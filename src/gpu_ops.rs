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
    let params = gpu.params_u32(&[m as u32, n as u32, k as u32, bias.is_some() as u32]);
    let zero = gpu.empty("linear.no_bias", 1);
    let b = bias.unwrap_or(&zero);
    // A register-blocked 64x64 tile was implemented and measured *slower* here
    // (44 vs 86 GFLOP/s at 2048^3; see examples/gpu_probe.rs for the card's
    // dispatch and GEMV cost). On this P104-100 a dispatch costs ~20 us and a
    // 512x512 GEMV 88 us, so the kernel schedule is not what limits us.
    if m == 1 {
        // One warp per output row: the tiled kernel would compute 16 rows and use 1.
        gpu.dispatch_params("gemv_f32", shaders::GEMV_F32, "main", &[(0, x), (1, w), (2, b), (3, y)], &params, (n.div_ceil(8) as u32, 1, 1));
        return;
    }
    gpu.dispatch_params("linear_f32", shaders::LINEAR_F32, "main", &[(0, x), (1, w), (2, b), (3, y)], &params, (m.div_ceil(16) as u32, n.div_ceil(16) as u32, 1));
}

/// Causal conv1d over `[channels, time]`.
pub fn conv1d(gpu: &Gpu, x: &Buffer, w: &Buffer, bias: &Buffer, y: &Buffer, p: &Conv1d) {
    let params = gpu.params_u32(&[
        p.cin as u32, p.cout as u32, p.ks as u32, p.dilation as u32,
        p.stride as u32, p.pad as u32, p.t_in as u32, p.t_out as u32,
    ]);
    let total = (p.cout * p.t_out) as u32;
    gpu.dispatch_params("conv1d_f32", shaders::CONV1D_F32, "main", &[(0, x), (1, w), (2, bias), (3, y)], &params, grid(total as usize));
}

/// SnakeBeta, in place, `[channels, time]`.
pub fn snake_beta(gpu: &Gpu, x: &Buffer, alpha: &Buffer, beta: &Buffer, channels: usize, time: usize) {
    let params = gpu.params_u32(&[channels as u32, time as u32]);
    gpu.dispatch_params("snake_beta_f32", shaders::SNAKE_BETA_F32, "main", &[(0, x), (1, alpha), (2, beta)], &params, grid(channels * time));
}

/// `y = a + b`.
pub fn add(gpu: &Gpu, a: &Buffer, b: &Buffer, y: &Buffer, len: usize) {
    let params = gpu.params_u32(&[len as u32]);
    gpu.dispatch_params("add_f32", shaders::ADD_F32, "main", &[(0, a), (1, b), (2, y)], &params, grid(len));
}

/// `y += x`, in place. Uses its own kernel: binding `y` as both read and
/// read_write in one dispatch is a wgpu validation error.
pub fn residual(gpu: &Gpu, y: &Buffer, x: &Buffer, len: usize) {
    let params = gpu.params_u32(&[len as u32]);
    gpu.dispatch_params("residual_f32", crate::shaders::RESIDUAL_F32, "main", &[(0, y), (1, x)], &params, grid(len));
}

/// RVQ dequantise-and-sum into a `[dim, frames]` tensor.
pub fn gather_sum(gpu: &Gpu, codes: &Buffer, tables: &Buffer, usage: &Buffer, out: &Buffer,
                  dim: usize, frames: usize, k_start: usize, k_count: usize,
                  codebook_size: usize, num_codebooks: usize) {
    let params = gpu.params_u32(&[
        dim as u32, frames as u32, k_start as u32, k_count as u32,
        codebook_size as u32, num_codebooks as u32, 0, 0,
    ]);
    gpu.dispatch_params("gather_sum_f32", shaders::GATHER_SUM_F32, "main", &[(0, codes), (1, tables), (2, usage), (3, out)], &params, grid(dim * frames));
}

/// `[rows, cols] -> [cols, rows]`.
pub fn transpose(gpu: &Gpu, x: &Buffer, y: &Buffer, rows: usize, cols: usize) {
    let params = gpu.params_u32(&[rows as u32, cols as u32, 0, 0]);
    gpu.dispatch_params("transpose_f32", shaders::TRANSPOSE_F32, "main", &[(0, x), (1, y)], &params, (rows.div_ceil(16) as u32, cols.div_ceil(16) as u32, 1));
}

/// Row-wise RMSNorm.
pub fn rms_norm(gpu: &Gpu, x: &Buffer, w: &Buffer, y: &Buffer, rows: usize, cols: usize, eps: f32) {
    let params = gpu.params_u32(&[rows as u32, cols as u32, eps.to_bits(), 0]);
    gpu.dispatch_params("rms_norm_f32", shaders::RMS_NORM_F32, "main", &[(0, x), (1, w), (2, y)], &params, (rows as u32, 1, 1));
}

/// Row-wise LayerNorm with affine parameters.
#[allow(clippy::too_many_arguments)]
pub fn layer_norm(gpu: &Gpu, x: &Buffer, w: &Buffer, b: &Buffer, y: &Buffer,
                  rows: usize, cols: usize, eps: f32) {
    let params = gpu.params_u32(&[rows as u32, cols as u32, eps.to_bits(), 0]);
    gpu.dispatch_params("layer_norm_f32", shaders::LAYER_NORM_F32, "main", &[(0, x), (1, w), (2, b), (3, y)], &params, (rows as u32, 1, 1));
}

/// `y = silu(gate) * up`.
pub fn silu_mul(gpu: &Gpu, gate: &Buffer, up: &Buffer, y: &Buffer, len: usize) {
    let params = gpu.params_u32(&[len as u32]);
    gpu.dispatch_params("silu_mul_f32", shaders::SILU_MUL_F32, "main", &[(0, gate), (1, up), (2, y)], &params, grid(len));
}

/// In-place per-channel scale on `[rows, cols]`.
pub fn scale_cols(gpu: &Gpu, x: &Buffer, scale: &Buffer, rows: usize, cols: usize) {
    let params = gpu.params_u32(&[rows as u32, cols as u32]);
    gpu.dispatch_params("scale_cols_f32", shaders::SCALE_COLS_F32, "main", &[(0, x), (1, scale)], &params, grid(rows * cols));
}

/// `y = a + scale[c] * b`.
pub fn add_scaled(gpu: &Gpu, a: &Buffer, b: &Buffer, scale: &Buffer, y: &Buffer, rows: usize, cols: usize) {
    let params = gpu.params_u32(&[rows as u32, cols as u32]);
    gpu.dispatch_params("add_scaled_f32", shaders::ADD_SCALED_F32, "main", &[(0, a), (1, b), (2, scale), (3, y)], &params, grid(rows * cols));
}

/// Rotary embedding, in place.
pub fn rope(gpu: &Gpu, x: &Buffer, frames: usize, heads: usize, head_dim: usize, theta: f32) {
    let params = gpu.params_u32(&[frames as u32, heads as u32, head_dim as u32, theta.to_bits()]);
    let total = (frames * heads * (head_dim / 2)) as u32;
    gpu.dispatch_params("rope_f32", shaders::ROPE_F32, "main", &[(0, x)], &params, grid(total as usize));
}

/// Causal sliding-window attention over `[T, heads*head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn attention(gpu: &Gpu, q: &Buffer, k: &Buffer, v: &Buffer, out: &Buffer,
                 frames: usize, heads: usize, head_dim: usize, window: usize) {
    let params = gpu.params_u32(&[frames as u32, heads as u32, head_dim as u32, window as u32]);
    gpu.dispatch_params("attention_f32", shaders::ATTENTION_F32, "main", &[(0, q), (1, k), (2, v), (3, out)], &params, ((frames * heads) as u32, 1, 1));
}

/// Depthwise causal conv1d.
pub fn conv1d_depthwise(gpu: &Gpu, x: &Buffer, w: &Buffer, b: &Buffer, y: &Buffer,
                        channels: usize, ks: usize, pad: usize, t_in: usize, t_out: usize) {
    // The shader declares `array<u32, 8>`; the binding must be that large.
    let params = gpu.params_u32(&[channels as u32, ks as u32, pad as u32, t_in as u32, t_out as u32, 0, 0, 0]);
    let total = (channels * t_out) as u32;
    gpu.dispatch_params("conv1d_dw_f32", shaders::CONV1D_DW_F32, "main", &[(0, x), (1, w), (2, b), (3, y)], &params, grid(total as usize));
}

/// Transposed conv1d (upsampling).
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(gpu: &Gpu, x: &Buffer, w: &Buffer, b: &Buffer, y: &Buffer,
                        cin: usize, cout: usize, ks: usize, stride: usize, t_in: usize) {
    let t_out = t_in * stride;
    let params = gpu.params_u32(&[cin as u32, cout as u32, ks as u32, stride as u32, t_in as u32, t_out as u32,
                                 0, 0]);
    let total = (cout * t_out) as u32;
    gpu.dispatch_params("conv_transpose_f32", shaders::CONV_TRANSPOSE_F32, "main", &[(0, x), (1, w), (2, b), (3, y)], &params, grid(total as usize));
}

/// Exact GELU, in place.
pub fn gelu(gpu: &Gpu, x: &Buffer, len: usize) {
    let params = gpu.params_u32(&[len as u32]);
    gpu.dispatch_params("gelu_f32", shaders::GELU_F32, "main", &[(0, x)], &params, grid(len));
}

/// `y = x`.
pub fn copy(gpu: &Gpu, x: &Buffer, y: &Buffer, len: usize) {
    let params = gpu.params_u32(&[len as u32]);
    gpu.dispatch_params("copy_f32", shaders::COPY_F32, "main", &[(0, x), (1, y)], &params, grid(len));
}

/// Tiled 2-D transpose: y[c, r] = x[r, c], both sides coalesced.
pub fn transpose_tiled(gpu: &Gpu, x: &Buffer, y: &Buffer, rows: usize, cols: usize) {
    let params = gpu.params_u32(&[rows as u32, cols as u32]);
    gpu.dispatch_params("transpose_tiled", shaders::TRANSPOSE_TILED_F32, "main",
                        &[(0, x), (1, y)], &params,
                        ((rows as u32).div_ceil(16).max(1), (cols as u32).div_ceil(16).max(1), 1u32));
}

/// Fused decode-step attention block (one workgroup per query head).
#[allow(clippy::too_many_arguments)]
pub fn qkv_attend_f32(gpu: &Gpu, qkv: &Buffer, cache_k: &Buffer, cache_v: &Buffer,
                      attended: &Buffer, cos: &Buffer, sin: &Buffer, position: usize,
                      heads: usize, q_rows: usize) {
    let kv_rows = q_rows / 4;  // heads / kv_heads == 4 in the LM geometry
    let params = gpu.params_u32(&[
        q_rows as u32, kv_rows as u32, 64u32, position as u32, heads as u32,
    ]);
    gpu.dispatch_params("qkv_attend_f32", shaders::QKV_ATTEND_F32, "main",
                        &[(0, qkv), (1, cache_k), (2, cache_v), (3, attended),
                          (4, cos), (5, sin)],
                        &params, (heads as u32, 1, 1));
}

/// GEMV with the residual add fused into the epilogue.
pub fn gemv_add_f32(gpu: &Gpu, x: &Buffer, w: &Buffer, res: &Buffer, y: &Buffer,
                    n: usize, k: usize) {
    let params = gpu.params_u32(&[n as u32, k as u32]);
    gpu.dispatch_params("gemv_add_f32", shaders::GEMV_ADD_F32, "main",
                        &[(0, x), (1, w), (2, res), (3, y)],
                        &params, (n.div_ceil(8) as u32, 1, 1));
}

/// RMSNorm fused into the gate+up projection (one dispatch).
pub fn norm_gateup_f32(gpu: &Gpu, x: &Buffer, norm_w: &Buffer, gate_w: &Buffer,
                       up_w: &Buffer, y: &Buffer, inter: usize, hidden: usize, eps: f32) {
    let params = gpu.params_u32(&[inter as u32, hidden as u32, eps.to_bits()]);
    gpu.dispatch_params("norm_gateup_f32", shaders::NORM_GATEUP_F32, "main",
                        &[(0, x), (1, norm_w), (2, gate_w), (3, up_w), (4, y)],
                        &params, ((2 * inter as u32).div_ceil(64).max(1), 1u32, 1u32));
}

/// silu-mul fused into the down projection with the residual add.
pub fn silu_down_f32(gpu: &Gpu, gate_up: &Buffer, down_w: &Buffer, x1: &Buffer, out: &Buffer,
                     inter: usize, hidden: usize) {
    let params = gpu.params_u32(&[inter as u32, hidden as u32]);
    gpu.dispatch_params("silu_down_f32", shaders::SILU_DOWN_F32, "main",
                        &[(0, gate_up), (1, down_w), (2, x1), (3, out)],
                        &params, ((hidden as u32).div_ceil(8).max(1), 1u32, 1u32));
}

/// Copy `rows` rows of `src` into `dst` at row `offset`.
pub fn write_rows(gpu: &Gpu, dst: &Buffer, src: &Buffer, cols: usize, offset: usize, rows: usize) {
    let params = gpu.params_u32(&[cols as u32, offset as u32, rows as u32, 0]);
    gpu.dispatch_params("write_rows_f32", shaders::WRITE_ROWS_F32, "main", &[(0, dst), (1, src)], &params, grid(cols * rows));
}

/// Single-query attention over a KV cache.
pub fn attention_decode(gpu: &Gpu, q: &Buffer, k: &Buffer, v: &Buffer, out: &Buffer,
                        len: usize, heads: usize, head_dim: usize) {
    let params = gpu.params_u32(&[len as u32, heads as u32, head_dim as u32, 0]);
    gpu.dispatch_params("attention_decode_f32", shaders::ATTENTION_DECODE_F32, "main", &[(0, q), (1, k), (2, v), (3, out)], &params, (heads as u32, 1, 1));
}

/// Rotary embedding from a cos/sin table.
pub fn rope_tabled(gpu: &Gpu, x: &Buffer, cos: &Buffer, sin: &Buffer,
                   frames: usize, heads: usize, head_dim: usize) {
    let params = gpu.params_u32(&[frames as u32, heads as u32, head_dim as u32, 0]);
    let total = frames * heads * (head_dim / 2);
    gpu.dispatch_params("rope_tabled_f32", shaders::ROPE_TABLED_F32, "main", &[(0, x), (1, cos), (2, sin)], &params, grid(total));
}

/// Grouped-query KV repeat.
pub fn repeat_kv(gpu: &Gpu, x: &Buffer, y: &Buffer, frames: usize, kv_heads: usize, heads: usize, head_dim: usize) {
    let params = gpu.params_u32(&[frames as u32, kv_heads as u32, heads as u32, head_dim as u32]);
    gpu.dispatch_params("repeat_kv_f32", shaders::REPEAT_KV_F32, "main", &[(0, x), (1, y)], &params, grid(frames * heads * head_dim));
}

/// ELU (alpha 1), in place.
pub fn elu(gpu: &Gpu, x: &Buffer, len: usize) {
    let params = gpu.params_u32(&[len as u32]);
    gpu.dispatch_params("elu_f32", shaders::ELU_F32, "main", &[(0, x)], &params, grid(len));
}

/// Sum of the sixteen audio embedding rows for one frame.
pub fn embed_sum(gpu: &Gpu, table: &Buffer, tokens: &Buffer, out: &Buffer,
                 hidden: usize, cb_offset: usize, cb_count: usize, vocab: usize) {
    let params = gpu.params_u32(&[hidden as u32, cb_offset as u32, vocab as u32, cb_count as u32]);
    gpu.dispatch_params("embed_sum_f32", shaders::EMBED_SUM_F32, "main", &[(0, table), (1, tokens), (2, out)], &params, (hidden.div_ceil(128) as u32, 1, 1));
}

/// top-k sample into `out[0]`.
#[allow(clippy::too_many_arguments)]
pub fn sample_topk(gpu: &Gpu, logits: &Buffer, noise: &Buffer, out: &Buffer,
                   vocab: usize, topk: usize, temperature: f32, allow_eos: bool,
                   out_offset: usize) {
    sample_topk_offset(gpu, logits, noise, out, vocab, topk, temperature, allow_eos,
                       out_offset, 0);
}

/// Same, reading the exponential draws from `noise[noise_offset..]` so one
/// `[CODEBOOKS, vocab]` buffer can stay bound for a whole frame.
#[allow(clippy::too_many_arguments)]
pub fn sample_topk_offset(gpu: &Gpu, logits: &Buffer, noise: &Buffer, out: &Buffer,
                          vocab: usize, topk: usize, temperature: f32, allow_eos: bool,
                          out_offset: usize, noise_offset: usize) {
    let params = gpu.params_u32(&[vocab as u32, topk as u32, temperature.to_bits(),
                                 ((out_offset as u32) << 1) | allow_eos as u32,
                                 noise_offset as u32]);
    gpu.dispatch_params("sample_topk_f32", shaders::SAMPLE_TOPK_F32, "main", &[(0, logits), (1, noise), (2, out)], &params, (1, 1, 1));
}

/// Register-blocked channels-last conv (8x8 per thread). Same contract as
/// `conv1d_cl`; dispatched by the codec for stride <= 2 shapes.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_reg8(gpu: &Gpu, x: &Buffer, w: &Buffer, bias: &Buffer, y: &Buffer,
                   cin: usize, cout: usize, ks: usize, dilation: usize, stride: usize,
                   pad: usize, t_in: usize, t_out: usize) {
    let params = gpu.params_u32(&[
        cin as u32, cout as u32, ks as u32, dilation as u32,
        stride as u32, pad as u32, t_in as u32, t_out as u32,
    ]);
    let groups = (t_out.div_ceil(64) as u32, cout.div_ceil(64) as u32, 1);
    gpu.dispatch_params("conv1d_reg8", shaders::CONV1D_REG8_F32, "main",
                        &[(0, x), (1, w), (2, bias), (3, y)], &params, groups);
}

/// Phased transposed conv (weights pre-packed as `[(co*stride+q)*taps*cin + tap*cin + ci]`).
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose_phased(gpu: &Gpu, x: &Buffer, w2: &Buffer, bias: &Buffer, y: &Buffer,
                             cin: usize, cout: usize, ks: usize, stride: usize, t_in: usize) {
    let taps = ks / stride;
    let rows_total = cout * stride;
    let params = gpu.params_u32(&[
        cin as u32, rows_total as u32, stride as u32, t_in as u32,
        cout as u32, (t_in * stride) as u32, taps as u32,
    ]);
    let groups = (rows_total.div_ceil(32) as u32, t_in.div_ceil(8) as u32, 1);
    gpu.dispatch_params("ct_phased", shaders::CONV_TRANSPOSE_PHASED_F32, "main", &[(0, x), (1, w2), (2, bias), (3, y)], &params, groups);
}

/// Channels-last conv with packed weights `[tap*cin+ci][cout]`.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_cl(gpu: &Gpu, x: &Buffer, w: &Buffer, bias: &Buffer, y: &Buffer,
                 cin: usize, cout: usize, ks: usize, dilation: usize, stride: usize,
                 pad: usize, t_in: usize, t_out: usize) {
    let params = gpu.params_u32(&[
        cin as u32, cout as u32, ks as u32, dilation as u32,
        stride as u32, pad as u32, t_in as u32, t_out as u32,
    ]);
    let groups = (t_out.div_ceil(128) as u32, cout.div_ceil(64) as u32, 1);
    // Short windows stage 16 channels; strided layers need 4 to stay in 18 KB.
    if stride <= 2 {
        gpu.dispatch_params("conv1d_reg_gpu", shaders::CONV1D_REG_GPU, "main", &[(0, x), (1, w), (2, bias), (3, y)], &params, groups);
    } else {
        gpu.dispatch_params("conv1d_reg_gpu_strided", shaders::CONV1D_REG_GPU_STRIDED, "main", &[(0, x), (1, w), (2, bias), (3, y)], &params, groups);
    }
}
