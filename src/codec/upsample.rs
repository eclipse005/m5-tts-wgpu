//! The two x2 upsampling stages between the transformer and the SEANet decoder.
//!
//! Each stage is a causal transposed convolution followed by a ConvNeXt block
//! (depthwise causal conv -> LayerNorm -> 4x pointwise -> GELU -> pointwise ->
//! learned gamma -> residual). Activations stay `[channels, time]`.

use anyhow::Result;

use super::Codec;
use crate::device::Tensor;
use crate::ops;

impl Codec {
    /// `[latent, frames]` -> `[latent, frames * product(upsampling_ratios)]`.
    pub fn upsample(&self, x: &Tensor) -> Result<Tensor> {
        let dev = self.device();
        let mut hidden = dev.copy(x)?;
        for (stage, factor) in self.cfg.upsampling_ratios.iter().enumerate() {
            let prefix = format!("decoder.upsample.{stage}");
            let [channels, frames] = hidden.shape();

            // CausalTransConvNet(latent, latent, kernel=factor, stride=factor):
            // kernel == stride, so nothing is trimmed.
            let weight = self.tensor(&format!("{prefix}.0.conv.weight"), [channels, channels * factor])?;
            let bias = self.vector(&format!("{prefix}.0.conv.bias"), channels)?;
            let mut upsampled = dev.zeros([channels, frames * factor])?;
            if let crate::device::Device::Gpu(_) = dev {
                upsampled = self.conv_transpose_phased(&hidden,
                                                       &format!("{prefix}.0.conv.weight"),
                                                       &format!("{prefix}.0.conv.bias"),
                                                       channels, *factor, *factor)?;
            } else {
                ops::conv_transpose1d(dev, &hidden, &weight, &bias, &mut upsampled, *factor, *factor)?;
            }

            hidden = self.convnext(&upsampled, &prefix)?;
        }
        Ok(hidden)
    }

    /// One ConvNeXt block, `[channels, time]` in and out.
    fn convnext(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let dev = self.device();
        let [channels, frames] = x.shape();
        let width = 4 * channels;

        let dw_weight = self.tensor(&format!("{prefix}.1.dwconv.conv.weight"), [channels, 7])?;
        let dw_bias = self.vector(&format!("{prefix}.1.dwconv.conv.bias"), channels)?;
        let mut mixed = dev.zeros([channels, frames])?;
        ops::conv1d_depthwise(dev, x, &dw_weight, &dw_bias, &mut mixed, 7, 6)?;

        let mut time_major = dev.zeros([frames, channels])?;
        ops::transpose(dev, &mixed, &mut time_major)?;

        let norm_w = self.vector(&format!("{prefix}.1.norm.weight"), channels)?;
        let norm_b = self.vector(&format!("{prefix}.1.norm.bias"), channels)?;
        let mut normed = dev.zeros([frames, channels])?;
        ops::layer_norm(dev, &time_major, &norm_w, &norm_b, &mut normed, 1e-6)?;

        let pw1_w = self.tensor(&format!("{prefix}.1.pwconv1.weight"), [width, channels])?;
        let pw1_b = self.vector(&format!("{prefix}.1.pwconv1.bias"), width)?;
        let mut expanded = dev.zeros([frames, width])?;
        ops::linear(dev, &normed, &pw1_w, Some(&pw1_b), &mut expanded)?;
        ops::gelu(dev, &mut expanded)?;

        let pw2_w = self.tensor(&format!("{prefix}.1.pwconv2.weight"), [channels, width])?;
        let pw2_b = self.vector(&format!("{prefix}.1.pwconv2.bias"), channels)?;
        let mut reduced = dev.zeros([frames, channels])?;
        ops::linear(dev, &expanded, &pw2_w, Some(&pw2_b), &mut reduced)?;

        let gamma = self.vector(&format!("{prefix}.1.gamma"), channels)?;
        ops::scale_cols(dev, &mut reduced, &gamma)?;

        let mut branch = dev.zeros([channels, frames])?;
        ops::transpose(dev, &reduced, &mut branch)?;
        let mut out = dev.zeros([channels, frames])?;
        ops::add(dev, x, &branch, &mut out)?;
        Ok(out)
    }
}
