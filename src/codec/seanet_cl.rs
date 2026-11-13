//! SEANet decoder tail, channels-last.
//!
//! The whole codec now runs on `[time, channels]` activations: the transformer
//! already wanted that layout, and every convolution becomes an implicit GEMM
//! whose reduction axis `(tap, ci)` is contiguous — the same reason oneDNN works
//! channels-last. Weights are packed once per layer (`pack_conv_weights`) instead
//! of being re-walked strided for every output channel.

use anyhow::Result;

use super::Codec;
use crate::cpu_ops;
use crate::device::Tensor;
use std::rc::Rc;

/// Geometry of one convolution.
pub struct ConvShape {
    pub cin: usize,
    pub cout: usize,
    pub ks: usize,
    pub dilation: usize,
    pub stride: usize,
}

impl ConvShape {
    /// Causal padding for a stride-1 convolution.
    pub fn pad(&self) -> usize {
        (self.ks - 1) * self.dilation
    }
    pub fn t_out(&self, t_in: usize) -> usize {
        if self.stride == 1 {
            t_in
        } else {
            t_in.div_ceil(self.stride)
        }
    }
}

impl Codec {
    /// Packed channels-last weight (cached). `name` is the checkpoint tensor.
    fn packed_conv(&self, name: &str, shape: &ConvShape, transposed: bool) -> Result<Rc<Tensor>> {
        let key = format!("packed.{name}");
        if let Some(tensor) = self.cache_ref().borrow().get(&key) {
            return Ok(tensor.clone());
        }
        let raw = self.weights.get_f32(name)?;
        // ConvTranspose1d stores `[cin, cout, ks]`, Conv1d stores `[cout, cin, ks]`.
        let packed = if transposed {
            cpu_ops::pack_conv_transpose_weights(&raw, shape.cin, shape.cout, shape.ks)
        } else {
            cpu_ops::pack_conv_weights(&raw, shape.cout, shape.cin, shape.ks)
        };
        let tensor = Rc::new(self.dev.tensor([shape.ks * shape.cin, shape.cout], &packed)?);
        self.cache_ref().borrow_mut().insert(key, tensor.clone());
        Ok(tensor)
    }

    /// Causal convolution, channels-last.
    pub fn conv_cl(&self, x: &Tensor, name: &str, shape: ConvShape) -> Result<Tensor> {
        let [t_in, cin] = x.shape();
        if cin != shape.cin {
            anyhow::bail!("{name}: expected {} channels, got {cin}", shape.cin);
        }
        let weight = self.packed_conv(name, &shape, false)?;
        let bias = self.vector(&name.replace(".weight", ".bias"), shape.cout)?;
        let t_out = shape.t_out(t_in);
        let mut out = self.dev.zeros([t_out, shape.cout])?;
        match self.device() {
            crate::device::Device::Cpu => cpu_ops::conv1d_channels_last(
                x.cpu_slice()?, weight.cpu_slice()?, bias.cpu_slice()?, out.cpu_slice_mut()?,
                shape.cin, shape.cout, shape.ks, shape.dilation, shape.stride, shape.pad(),
                t_in, t_out),
            _ => anyhow::bail!("channels-last conv is CPU-only for now"),
        }
        Ok(out)
    }

    /// Transposed (upsampling) convolution, channels-last.
    pub fn conv_transpose_cl(&self, x: &Tensor, name: &str, shape: ConvShape) -> Result<Tensor> {
        let [t_in, cin] = x.shape();
        if cin != shape.cin {
            anyhow::bail!("{name}: expected {} channels, got {cin}", shape.cin);
        }
        let weight = self.packed_conv(name, &shape, true)?;
        let bias = self.vector(&name.replace(".weight", ".bias"), shape.cout)?;
        let t_out = t_in * shape.stride;
        let mut out = self.dev.zeros([t_out, shape.cout])?;
        match self.device() {
            crate::device::Device::Cpu => cpu_ops::conv_transpose1d_channels_last(
                x.cpu_slice()?, weight.cpu_slice()?, bias.cpu_slice()?, out.cpu_slice_mut()?,
                shape.cin, shape.cout, shape.ks, shape.stride, t_in),
            _ => anyhow::bail!("channels-last transposed conv is CPU-only for now"),
        }
        Ok(out)
    }

    /// SnakeBeta, channels-last.
    pub fn snake_cl(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let channels = x.shape()[1];
        let alpha = self.vector(&format!("{prefix}.alpha"), channels)?;
        let beta = self.vector(&format!("{prefix}.beta"), channels)?;
        let mut out = self.device().copy(x)?;
        match self.device() {
            crate::device::Device::Cpu => cpu_ops::snake_beta_channels_last(
                out.cpu_slice_mut()?, alpha.cpu_slice()?, beta.cpu_slice()?, channels),
            _ => anyhow::bail!("channels-last snake is CPU-only for now"),
        }
        Ok(out)
    }

    /// `DecoderResidualUnit`: snake -> conv7(dilated) -> snake -> conv1 -> residual.
    fn residual_unit_cl(&self, x: &Tensor, prefix: &str, dilation: usize) -> Result<Tensor> {
        let channels = x.shape()[1];
        let act1 = self.snake_cl(x, &format!("{prefix}.act1"))?;
        let h = self.conv_cl(&act1, &format!("{prefix}.conv1.conv.weight"), ConvShape {
            cin: channels, cout: channels, ks: 7, dilation, stride: 1,
        })?;
        let act2 = self.snake_cl(&h, &format!("{prefix}.act2"))?;
        let out = self.conv_cl(&act2, &format!("{prefix}.conv2.conv.weight"), ConvShape {
            cin: channels, cout: channels, ks: 1, dilation: 1, stride: 1,
        })?;
        let dev = self.device();
        let mut sum = dev.zeros(x.shape())?;
        crate::ops::add(dev, &out, x, &mut sum)?;
        Ok(sum)
    }

    /// `DecoderBlock`: snake -> transposed conv -> three residual units.
    fn decoder_block_cl(&self, x: &Tensor, prefix: &str, upsample_rate: usize) -> Result<Tensor> {
        let [_, cin] = x.shape();
        let cout = cin / 2;
        let act = self.snake_cl(x, &format!("{prefix}.block.0"))?;
        let ks = 2 * upsample_rate;
        let mut h = self.conv_transpose_cl(&act, &format!("{prefix}.block.1.conv.weight"), ConvShape {
            cin, cout, ks, dilation: 1, stride: upsample_rate,
        })?;
        for (unit, dilation) in [(2usize, 1usize), (3, 3), (4, 9)] {
            h = self.residual_unit_cl(&h, &format!("{prefix}.block.{unit}"), dilation)?;
        }
        Ok(h)
    }

    /// Full SEANet tail on channels-last activations, `[t, latent]` -> `[samples, 1]`.
    pub fn seanet_cl(&self, upsampled: &Tensor) -> Result<Tensor> {
        let cfg = &self.cfg;
        let mut h = self.conv_cl(upsampled, "decoder.decoder.0.conv.weight", ConvShape {
            cin: cfg.latent_dim, cout: cfg.decoder_dim, ks: 7, dilation: 1, stride: 1,
        })?;
        for (index, rate) in cfg.upsample_rates.iter().enumerate() {
            let prefix = format!("decoder.decoder.{}", index + 1);
            h = self.decoder_block_cl(&h, &prefix, *rate)?;
        }
        let channels = h.shape()[1];
        let h = self.snake_cl(&h, "decoder.decoder.5")?;
        let _ = channels;
        self.conv_cl(&h, "decoder.decoder.6.conv.weight", ConvShape {
            cin: h.shape()[1], cout: 1, ks: 7, dilation: 1, stride: 1,
        })
    }
}
