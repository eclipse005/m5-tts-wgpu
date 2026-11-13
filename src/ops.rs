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

/// SnakeBeta in place.
pub fn snake_beta(dev: &Device, x: &mut Tensor, alpha: &Tensor, beta: &Tensor) -> Result<()> {
    let [channels, time] = x.shape();
    expect("snake alpha", alpha.shape(), [channels, 1])?;
    expect("snake beta", beta.shape(), [channels, 1])?;
    match dev {
        Device::Gpu(gpu) => {
            gpu_ops::snake_beta(gpu, x.gpu_buffer()?, alpha.gpu_buffer()?, beta.gpu_buffer()?, channels, time);
        }
        Device::Cpu => {
            cpu_ops::snake_beta(x.cpu_slice_mut()?, alpha.cpu_slice()?, beta.cpu_slice()?, channels, time);
        }
    }
    Ok(())
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
        Device::Gpu(gpu) => gpu_ops::transpose(gpu, x.gpu_buffer()?, y.gpu_buffer()?, rows, cols),
        Device::Cpu => cpu_ops::transpose(x.cpu_slice()?, y.cpu_slice_mut()?, rows, cols),
    }
    Ok(())
}
