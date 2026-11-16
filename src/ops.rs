//! Device-independent tensor operations — the seam between the model graphs and
//! the two backends.
//!
//! `codec` and `lm` only call these functions, so a new stage is written once and
//! runs on wgpu and on the hand-written CPU path. Shapes are validated here, and
//! the parity tests drive exactly this API on both devices.

use anyhow::{bail, Result};

use crate::device::{Device, Tensor};
use crate::{cpu_ops, gpu_ops};

/// Copy host data into a tensor of the same shape.
pub fn write(tensor: &mut Tensor, data: &[f32]) -> Result<()> {
    if data.len() != tensor.numel() {
        bail!("write: {} values into a {:?} tensor", data.len(), tensor.shape());
    }
    match tensor {
        Tensor::Cpu(host, _) => {
            host.copy_from_slice(data);
            Ok(())
        }
        Tensor::Gpu(..) => bail!("GPU tensors are written through Device::write"),
    }
}

/// Causal conv1d geometry. `t_out` is the caller's business because the codec
/// pads causally (`pad = kernel_size - 1`) before growing the time axis.
#[derive(Clone, Copy, Debug)]
pub struct Conv1d {
    pub cin: usize,
    pub cout: usize,
    pub ks: usize,
    pub dilation: usize,
    pub stride: usize,
    pub pad: usize,
    pub t_in: usize,
    pub t_out: usize,
}

impl Conv1d {
    /// Causal convolution: `ks - 1` samples of left padding, `pad` may be larger
    /// for the transposed-conv emulation (`stride > 1`).
    pub fn causal(cin: usize, cout: usize, ks: usize, t_in: usize) -> Self {
        Self { cin, cout, ks, dilation: 1, stride: 1, pad: ks - 1, t_in, t_out: t_in }
    }

    pub fn dilated(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self.pad = (self.ks - 1) * dilation;
        self
    }
}

fn expect(name: &str, got: [usize; 2], want: [usize; 2]) -> Result<()> {
    if got != want {
        bail!("{name}: expected shape {want:?}, got {got:?}");
    }
    Ok(())
}

/// `y[m,n] = x[m,k] · w[n,k]^T + bias[n]`.
pub fn linear(dev: &Device, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, y: &mut Tensor) -> Result<()> {
    let [m, k] = x.shape();
    let [n, k2] = w.shape();
    if k != k2 {
        bail!("linear: x {m}x{k} and w {n}x{k2} disagree on in_features");
    }
    expect("linear output", y.shape(), [m, n])?;
    match dev {
        Device::Gpu(gpu) => {
            let b = bias.map(|b| b.gpu_buffer()).transpose()?;
            gpu_ops::linear(gpu, x.gpu_buffer()?, w.gpu_buffer()?, b, y.gpu_buffer()?, m, n, k);
        }
        Device::Cpu => {
            cpu_ops::linear(x.cpu_slice()?, w.cpu_slice()?, bias.map(|b| b.cpu_slice()).transpose()?,
                            y.cpu_slice_mut()?, (m, n, k));
        }
    }
    Ok(())
}

/// `y = conv1d(x) + bias`, `[channels, time]` throughout.
pub fn conv1d(dev: &Device, x: &Tensor, w: &Tensor, bias: &Tensor, y: &mut Tensor, p: &Conv1d) -> Result<()> {
    expect("conv1d input", x.shape(), [p.cin, p.t_in])?;
    expect("conv1d weight", w.shape(), [p.cout, p.cin * p.ks])?;
    expect("conv1d bias", bias.shape(), [p.cout, 1])?;
    expect("conv1d output", y.shape(), [p.cout, p.t_out])?;
    match dev {
        Device::Gpu(gpu) => {
            gpu_ops::conv1d(gpu, x.gpu_buffer()?, w.gpu_buffer()?, bias.gpu_buffer()?, y.gpu_buffer()?, p);
        }
        Device::Cpu => {
            cpu_ops::conv1d(x.cpu_slice()?, w.cpu_slice()?, bias.cpu_slice()?, y.cpu_slice_mut()?,
                            (p.cin, p.cout, p.ks, p.dilation, p.stride, p.pad, p.t_in, p.t_out));
        }
    }
    Ok(())
}

/// SnakeBeta, out of place: the activation runs on an owned copy so callers keep
/// their input intact (in-place on a shared GPU buffer is a silent aliasing bug).
pub fn snake_beta(dev: &Device, x: &Tensor, alpha: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let [channels, time] = x.shape();
    expect("snake alpha", alpha.shape(), [channels, 1])?;
    expect("snake beta", beta.shape(), [channels, 1])?;
    match dev {
        Device::Gpu(gpu) => {
            let out = dev.copy(x)?;
            gpu_ops::snake_beta(gpu, out.gpu_buffer()?, alpha.gpu_buffer()?, beta.gpu_buffer()?, channels, time);
            Ok(out)
        }
        // Fused read/write with the polynomial sine: one 72 MB pass instead of
        // copy + a second in-place pass (4 ms vs 40 ms at block-4 size — the old
        // "copy is faster" note predates the lazy-page zeros fix, when the copy
        // was amortised against a freshly materialised output either way).
        Device::Cpu => {
            let mut out = dev.zeros(x.shape())?;
            cpu_ops::snake_beta_into(x.cpu_slice()?, out.cpu_slice_mut()?, alpha.cpu_slice()?,
                                     beta.cpu_slice()?, channels, time);
            Ok(out)
        }
    }
}

/// `y = a + b`.
pub fn add(dev: &Device, a: &Tensor, b: &Tensor, y: &mut Tensor) -> Result<()> {
    expect("add a", a.shape(), y.shape())?;
    expect("add b", b.shape(), y.shape())?;
    let len = y.numel();
    match dev {
        Device::Gpu(gpu) => {
            gpu_ops::add(gpu, a.gpu_buffer()?, b.gpu_buffer()?, y.gpu_buffer()?, len);
        }
        Device::Cpu => cpu_ops::add(a.cpu_slice()?, b.cpu_slice()?, y.cpu_slice_mut()?),
    }
    Ok(())
}

/// `y += x` (residual branch).
pub fn residual(dev: &Device, y: &mut Tensor, x: &Tensor) -> Result<()> {
    expect("residual x", x.shape(), y.shape())?;
    let len = y.numel();
    match dev {
        Device::Gpu(gpu) => {
            gpu_ops::residual(gpu, y.gpu_buffer()?, x.gpu_buffer()?, len);
        }
        Device::Cpu => cpu_ops::residual(y.cpu_slice_mut()?, x.cpu_slice()?),
    }
    Ok(())
}

/// RVQ dequantise-and-sum: `out[d, t] = sum_k table[k][code[t, k]][d] / clamp(usage, 1e-5)`.
#[allow(clippy::too_many_arguments)]
pub fn gather_sum(dev: &Device, codes: &[u32], tables: &Tensor, usage: &Tensor, out: &mut Tensor,
                  dim: usize, frames: usize, k_start: usize, k_count: usize,
                  codebook_size: usize, num_codebooks: usize) -> Result<()> {
    if codes.len() != frames * num_codebooks {
        bail!("gather_sum: {} codes for {frames} frames x {num_codebooks} codebooks", codes.len());
    }
    expect("gather_sum tables", tables.shape(), [k_count * codebook_size, dim])?;
    expect("gather_sum usage", usage.shape(), [k_count * codebook_size, 1])?;
    expect("gather_sum output", out.shape(), [dim, frames])?;
    match dev {
        Device::Gpu(gpu) => {
            let codes = gpu.upload_u32("gather_sum.codes", codes);
            gpu_ops::gather_sum(gpu, &codes, tables.gpu_buffer()?, usage.gpu_buffer()?, out.gpu_buffer()?,
                                dim, frames, k_start, k_count, codebook_size, num_codebooks);
        }
        Device::Cpu => {
            cpu_ops::gather_sum(codes, tables.cpu_slice()?, usage.cpu_slice()?, out.cpu_slice_mut()?,
                                (dim, frames, k_start, k_count, codebook_size, num_codebooks));
        }
    }
    Ok(())
}

/// Transpose a 2-D tensor.
pub fn transpose(dev: &Device, x: &Tensor, y: &mut Tensor) -> Result<()> {
    let [rows, cols] = x.shape();
    expect("transpose output", y.shape(), [cols, rows])?;
    match dev {
        Device::Gpu(gpu) => gpu_ops::transpose_tiled(gpu, x.gpu_buffer()?, y.gpu_buffer()?, rows, cols),
        Device::Cpu => cpu_ops::transpose(x.cpu_slice()?, y.cpu_slice_mut()?, rows, cols),
    }
    Ok(())
}

/// Row-wise RMSNorm over the last axis.
pub fn rms_norm(dev: &Device, x: &Tensor, w: &Tensor, y: &mut Tensor, eps: f64) -> Result<()> {
    let [rows, cols] = x.shape();
    expect("rms_norm output", y.shape(), [rows, cols])?;
    expect("rms_norm weight", w.shape(), [cols, 1])?;
    match dev {
        Device::Gpu(gpu) => gpu_ops::rms_norm(gpu, x.gpu_buffer()?, w.gpu_buffer()?, y.gpu_buffer()?, rows, cols, eps as f32),
        Device::Cpu => cpu_ops::rms_norm(x.cpu_slice()?, w.cpu_slice()?, y.cpu_slice_mut()?, rows, cols, eps as f32),
    }
    Ok(())
}

/// Row-wise LayerNorm with affine parameters.
pub fn layer_norm(dev: &Device, x: &Tensor, w: &Tensor, b: &Tensor, y: &mut Tensor, eps: f64) -> Result<()> {
    let [rows, cols] = x.shape();
    expect("layer_norm output", y.shape(), [rows, cols])?;
    expect("layer_norm weight", w.shape(), [cols, 1])?;
    expect("layer_norm bias", b.shape(), [cols, 1])?;
    match dev {
        Device::Gpu(gpu) => gpu_ops::layer_norm(gpu, x.gpu_buffer()?, w.gpu_buffer()?, b.gpu_buffer()?,
                                                y.gpu_buffer()?, rows, cols, eps as f32),
        Device::Cpu => cpu_ops::layer_norm(x.cpu_slice()?, w.cpu_slice()?, b.cpu_slice()?, y.cpu_slice_mut()?,
                                           rows, cols, eps as f32),
    }
    Ok(())
}

/// `y = silu(gate) * up`.
pub fn silu_mul(dev: &Device, gate: &Tensor, up: &Tensor, y: &mut Tensor) -> Result<()> {
    expect("silu_mul gate", gate.shape(), y.shape())?;
    expect("silu_mul up", up.shape(), y.shape())?;
    let len = y.numel();
    match dev {
        Device::Gpu(gpu) => gpu_ops::silu_mul(gpu, gate.gpu_buffer()?, up.gpu_buffer()?, y.gpu_buffer()?, len),
        Device::Cpu => cpu_ops::silu_mul(gate.cpu_slice()?, up.cpu_slice()?, y.cpu_slice_mut()?),
    }
    Ok(())
}

/// In-place per-column scale (layer scale, ConvNeXt gamma).
pub fn scale_cols(dev: &Device, x: &mut Tensor, scale: &Tensor) -> Result<()> {
    let [rows, cols] = x.shape();
    expect("scale_cols scale", scale.shape(), [cols, 1])?;
    match dev {
        Device::Gpu(gpu) => gpu_ops::scale_cols(gpu, x.gpu_buffer()?, scale.gpu_buffer()?, rows, cols),
        Device::Cpu => cpu_ops::scale_cols(x.cpu_slice_mut()?, scale.cpu_slice()?, cols),
    }
    Ok(())
}

/// `y = a + scale[c] * b`.
pub fn add_scaled(dev: &Device, a: &Tensor, b: &Tensor, scale: &Tensor, y: &mut Tensor) -> Result<()> {
    let [rows, cols] = y.shape();
    expect("add_scaled a", a.shape(), [rows, cols])?;
    expect("add_scaled b", b.shape(), [rows, cols])?;
    expect("add_scaled scale", scale.shape(), [cols, 1])?;
    match dev {
        Device::Gpu(gpu) => gpu_ops::add_scaled(gpu, a.gpu_buffer()?, b.gpu_buffer()?, scale.gpu_buffer()?,
                                                y.gpu_buffer()?, rows, cols),
        Device::Cpu => cpu_ops::add_scaled(a.cpu_slice()?, b.cpu_slice()?, scale.cpu_slice()?,
                                           y.cpu_slice_mut()?, cols),
    }
    Ok(())
}

/// Rotary embedding applied in place to `[frames, heads * head_dim]`.
pub fn rope(dev: &Device, x: &mut Tensor, heads: usize, head_dim: usize, theta: f64) -> Result<()> {
    let [frames, cols] = x.shape();
    if cols != heads * head_dim {
        bail!("rope: {cols} columns is not {heads} heads x {head_dim}");
    }
    match dev {
        Device::Gpu(gpu) => gpu_ops::rope(gpu, x.gpu_buffer()?, frames, heads, head_dim, theta as f32),
        Device::Cpu => cpu_ops::rope(x.cpu_slice_mut()?, frames, heads, head_dim, theta as f32),
    }
    Ok(())
}

/// Causal sliding-window attention.
#[allow(clippy::too_many_arguments)]
pub fn attention(dev: &Device, q: &Tensor, k: &Tensor, v: &Tensor, out: &mut Tensor,
                 heads: usize, head_dim: usize, window: usize) -> Result<()> {
    let [frames, cols] = q.shape();
    if cols != heads * head_dim {
        bail!("attention: {cols} columns is not {heads} heads x {head_dim}");
    }
    expect("attention k", k.shape(), [frames, cols])?;
    expect("attention v", v.shape(), [frames, cols])?;
    expect("attention output", out.shape(), [frames, cols])?;
    match dev {
        Device::Gpu(gpu) => gpu_ops::attention(gpu, q.gpu_buffer()?, k.gpu_buffer()?, v.gpu_buffer()?,
                                               out.gpu_buffer()?, frames, heads, head_dim, window),
        Device::Cpu => cpu_ops::attention(q.cpu_slice()?, k.cpu_slice()?, v.cpu_slice()?, out.cpu_slice_mut()?,
                                          frames, heads, head_dim, window),
    }
    Ok(())
}

/// Depthwise causal conv1d, `[channels, time]` in and out.
pub fn conv1d_depthwise(dev: &Device, x: &Tensor, w: &Tensor, bias: &Tensor, y: &mut Tensor,
                        ks: usize, pad: usize) -> Result<()> {
    let [channels, t_in] = x.shape();
    let [_, t_out] = y.shape();
    expect("depthwise weight", w.shape(), [channels, ks])?;
    expect("depthwise bias", bias.shape(), [channels, 1])?;
    match dev {
        Device::Gpu(gpu) => gpu_ops::conv1d_depthwise(gpu, x.gpu_buffer()?, w.gpu_buffer()?, bias.gpu_buffer()?,
                                                      y.gpu_buffer()?, channels, ks, pad, t_in, t_out),
        Device::Cpu => cpu_ops::conv1d_depthwise(x.cpu_slice()?, w.cpu_slice()?, bias.cpu_slice()?,
                                                 y.cpu_slice_mut()?, channels, ks, pad, t_in, t_out),
    }
    Ok(())
}

/// Transposed conv1d upsampling; output length is `t_in * stride`.
pub fn conv_transpose1d(dev: &Device, x: &Tensor, w: &Tensor, bias: &Tensor, y: &mut Tensor,
                        ks: usize, stride: usize) -> Result<()> {
    let [cin, t_in] = x.shape();
    let [cout, t_out] = y.shape();
    expect("transposed weight", w.shape(), [cin, cout * ks])?;
    expect("transposed bias", bias.shape(), [cout, 1])?;
    if t_out != t_in * stride {
        bail!("conv_transpose1d: output {t_out} != {t_in} x {stride}");
    }
    match dev {
        Device::Gpu(gpu) => gpu_ops::conv_transpose1d(gpu, x.gpu_buffer()?, w.gpu_buffer()?, bias.gpu_buffer()?,
                                                      y.gpu_buffer()?, cin, cout, ks, stride, t_in),
        Device::Cpu => {
            if ks == 2 * stride {
                cpu_ops::conv_transpose1d_phased(x.cpu_slice()?, w.cpu_slice()?, bias.cpu_slice()?,
                                                 y.cpu_slice_mut()?, cin, cout, ks, stride, t_in);
            } else {
                cpu_ops::conv_transpose1d(x.cpu_slice()?, w.cpu_slice()?, bias.cpu_slice()?,
                                          y.cpu_slice_mut()?, cin, cout, ks, stride, t_in);
            }
        }
    }
    Ok(())
}

/// Exact GELU, in place.
pub fn gelu(dev: &Device, x: &mut Tensor) -> Result<()> {
    let len = x.numel();
    match dev {
        Device::Gpu(gpu) => gpu_ops::gelu(gpu, x.gpu_buffer()?, len),
        Device::Cpu => cpu_ops::gelu(x.cpu_slice_mut()?),
    }
    Ok(())
}

/// Append `rows` rows of `src` into `dst` at row `offset` (KV cache).
pub fn write_rows(dev: &Device, dst: &mut Tensor, offset: usize, src: &Tensor) -> Result<()> {
    let [dst_rows, cols] = dst.shape();
    let [src_rows, src_cols] = src.shape();
    if src_cols != cols {
        bail!("write_rows: {src_cols} columns into a {cols}-column cache");
    }
    if offset + src_rows > dst_rows {
        bail!("write_rows: {src_rows} rows at {offset} overflow a {dst_rows}-row cache");
    }
    match dev {
        Device::Gpu(gpu) => gpu_ops::write_rows(gpu, dst.gpu_buffer()?, src.gpu_buffer()?, cols, offset, src_rows),
        Device::Cpu => cpu_ops::write_rows(dst.cpu_slice_mut()?, src.cpu_slice()?, cols, offset),
    }
    Ok(())
}

/// Attention of one query row over the first `len` rows of a KV cache.
pub fn attention_decode(dev: &Device, q: &Tensor, k: &Tensor, v: &Tensor, out: &mut Tensor,
                        len: usize, heads: usize, head_dim: usize) -> Result<()> {
    let [q_rows, cols] = q.shape();
    if q_rows != 1 || cols != heads * head_dim {
        bail!("attention_decode: query must be 1 x {}, got {q_rows} x {cols}", heads * head_dim);
    }
    expect("attention_decode k", k.shape(), [k.shape()[0], cols])?;
    expect("attention_decode v", v.shape(), [v.shape()[0], cols])?;
    expect("attention_decode output", out.shape(), [1, cols])?;
    match dev {
        Device::Gpu(gpu) => gpu_ops::attention_decode(gpu, q.gpu_buffer()?, k.gpu_buffer()?, v.gpu_buffer()?,
                                                      out.gpu_buffer()?, len, heads, head_dim),
        Device::Cpu => cpu_ops::attention_decode(q.cpu_slice()?, k.cpu_slice()?, v.cpu_slice()?,
                                                 out.cpu_slice_mut()?, len, heads, head_dim),
    }
    Ok(())
}

/// Rotary embedding from a cos/sin table (`[T, head_dim]` each), in place.
pub fn rope_tabled(dev: &Device, x: &mut Tensor, cos: &Tensor, sin: &Tensor,
                   heads: usize, head_dim: usize) -> Result<()> {
    let [frames, cols] = x.shape();
    if cols != heads * head_dim {
        bail!("rope_tabled: {cols} columns is not {heads} heads x {head_dim}");
    }
    expect("rope cos", cos.shape(), [frames, head_dim])?;
    expect("rope sin", sin.shape(), [frames, head_dim])?;
    match dev {
        Device::Gpu(gpu) => gpu_ops::rope_tabled(gpu, x.gpu_buffer()?, cos.gpu_buffer()?, sin.gpu_buffer()?,
                                                 frames, heads, head_dim),
        Device::Cpu => cpu_ops::rope_tabled(x.cpu_slice_mut()?, cos.cpu_slice()?, sin.cpu_slice()?,
                                            frames, heads, head_dim),
    }
    Ok(())
}

/// Grouped-query KV repeat: `[T, kv*D] -> [T, heads*D]`.
pub fn repeat_kv(dev: &Device, x: &Tensor, y: &mut Tensor, kv_heads: usize, heads: usize,
                 head_dim: usize) -> Result<()> {
    let [frames, cols] = x.shape();
    if cols != kv_heads * head_dim {
        bail!("repeat_kv: {cols} columns is not {kv_heads} kv heads x {head_dim}");
    }
    expect("repeat_kv output", y.shape(), [frames, heads * head_dim])?;
    match dev {
        Device::Gpu(gpu) => gpu_ops::repeat_kv(gpu, x.gpu_buffer()?, y.gpu_buffer()?, frames, kv_heads, heads, head_dim),
        Device::Cpu => cpu_ops::repeat_kv(x.cpu_slice()?, y.cpu_slice_mut()?, frames, kv_heads, heads, head_dim),
    }
    Ok(())
}

/// ELU (alpha 1), in place. Callers own the tensor (see [`Device::copy`]).
pub fn elu(dev: &Device, x: &mut Tensor) -> Result<()> {
    let len = x.numel();
    match dev {
        Device::Gpu(gpu) => gpu_ops::elu(gpu, x.gpu_buffer()?, len),
        Device::Cpu => cpu_ops::elu(x.cpu_slice_mut()?),
    }
    Ok(())
}
