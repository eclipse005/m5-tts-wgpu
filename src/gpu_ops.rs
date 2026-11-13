//! wgpu launchers for the kernels in [`crate::shaders`].
//!
//! Nothing here knows about model structure: it takes buffers and shapes. The
//! device-independent API lives in [`crate::ops`].

use crate::ops::Conv1d;
use crate::{gpu::Gpu, shaders};
use wgpu::Buffer;

/// `y[m,n] = x[m,k] · w[n,k]^T + bias[n]`.
pub fn linear(gpu: &Gpu, x: &Buffer, w: &Buffer, bias: Option<&Buffer>, y: &Buffer,
              m: usize, n: usize, k: usize) {
    let params = gpu.upload_u32("linear.params", &[m as u32, n as u32, k as u32, bias.is_some() as u32]);
    let zero = gpu.empty("linear.no_bias", 1);
    let b = bias.unwrap_or(&zero);
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
                 (total.div_ceil(128), 1, 1));
}

/// SnakeBeta, in place, `[channels, time]`.
pub fn snake_beta(gpu: &Gpu, x: &Buffer, alpha: &Buffer, beta: &Buffer, channels: usize, time: usize) {
    let params = gpu.upload_u32("snake.params", &[channels as u32, time as u32]);
    gpu.dispatch("snake_beta_f32", shaders::SNAKE_BETA_F32, "main",
                 &[(0, x), (1, alpha), (2, beta), (3, &params)],
                 (((channels * time) as u32).div_ceil(128), 1, 1));
}

/// `y = a + b`.
pub fn add(gpu: &Gpu, a: &Buffer, b: &Buffer, y: &Buffer, len: usize) {
    let params = gpu.upload_u32("add.params", &[len as u32]);
    gpu.dispatch("add_f32", shaders::ADD_F32, "main",
                 &[(0, a), (1, b), (2, y), (3, &params)],
                 ((len as u32).div_ceil(128), 1, 1));
}

/// `y += x`, in place. Uses its own kernel: binding `y` as both read and
/// read_write in one dispatch is a wgpu validation error.
pub fn residual(gpu: &Gpu, y: &Buffer, x: &Buffer, len: usize) {
    let params = gpu.upload_u32("residual.params", &[len as u32]);
    gpu.dispatch("residual_f32", crate::shaders::RESIDUAL_F32, "main",
                 &[(0, y), (1, x), (2, &params)],
                 ((len as u32).div_ceil(128), 1, 1));
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
                 (((dim * frames) as u32).div_ceil(128), 1, 1));
}

/// `[rows, cols] -> [cols, rows]`.
pub fn transpose(gpu: &Gpu, x: &Buffer, y: &Buffer, rows: usize, cols: usize) {
    let params = gpu.upload_u32("transpose.params", &[rows as u32, cols as u32, 0, 0]);
    gpu.dispatch("transpose_f32", shaders::TRANSPOSE_F32, "main",
                 &[(0, x), (1, y), (2, &params)],
                 (rows.div_ceil(16) as u32, cols.div_ceil(16) as u32, 1));
}
