//! SEANet decoder tail: `decoder.0` conv, four upsampling blocks, final SnakeBeta
//! and the mono output conv. This is where codes become a waveform.

use anyhow::Result;

use super::Codec;
use crate::device::{Device, Tensor};
use crate::ops::{self, Conv1d};

/// `CausalConvNet(in, out, kernel, dilation)`: `pad = (kernel - 1) * dilation`,
/// time preserved. The reference computes the same padding and adds no extra tail
/// for stride 1.
fn causal_conv(codec: &Codec, x: &Tensor, weight_name: &str, bias_name: &str,
               cout: usize, ks: usize, dilation: usize) -> Result<Tensor> {
    let dev = codec.device();
    let [cin, frames] = x.shape();
    let weight = codec.tensor(weight_name, [cout, cin * ks])?;
    let bias = codec.vector(bias_name, cout)?;
    let mut out = dev.zeros([cout, frames])?;
    let conv = Conv1d::causal(cin, cout, ks, frames).dilated(dilation);
    ops::conv1d(dev, x, &weight, &bias, &mut out, &conv)?;
    Ok(out)
}

/// One `DecoderResidualUnit`: snake -> conv7(dilated) -> snake -> conv1 -> add.
fn residual_unit(codec: &Codec, x: &Tensor, prefix: &str, dilation: usize) -> Result<Tensor> {
    let dev: &Device = codec.device();
    let channels = x.shape()[0];

    let mut h = x.clone();
    let alpha = codec.vector(&format!("{prefix}.act1.alpha"), channels)?;
    let beta = codec.vector(&format!("{prefix}.act1.beta"), channels)?;
    ops::snake_beta(dev, &mut h, &alpha, &beta)?;
    let mut h3 = causal_conv(codec, &h, &format!("{prefix}.conv1.conv.weight"),
                             &format!("{prefix}.conv1.conv.bias"), channels, 7, dilation)?;
    let alpha2 = codec.vector(&format!("{prefix}.act2.alpha"), channels)?;
    let beta2 = codec.vector(&format!("{prefix}.act2.beta"), channels)?;
    ops::snake_beta(dev, &mut h3, &alpha2, &beta2)?;
    let out = causal_conv(codec, &h3, &format!("{prefix}.conv2.conv.weight"),
                          &format!("{prefix}.conv2.conv.bias"), channels, 1, 1)?;

    let mut sum = dev.zeros(x.shape())?;
    ops::add(dev, &out, x, &mut sum)?;
    Ok(sum)
}

/// One `DecoderBlock`: snake -> transposed conv -> three residual units.
fn decoder_block(codec: &Codec, x: &Tensor, prefix: &str, upsample_rate: usize) -> Result<Tensor> {
    let dev = codec.device();
    let [cin, frames] = x.shape();
    let cout = cin / 2;

    let mut h = x.clone();
    let alpha = codec.vector(&format!("{prefix}.block.0.alpha"), cin)?;
    let beta = codec.vector(&format!("{prefix}.block.0.beta"), cin)?;
    ops::snake_beta(dev, &mut h, &alpha, &beta)?;

    // CausalTransConvNet(cin, cout, kernel=2*rate, stride=rate), trimmed to frames*rate.
    let ks = 2 * upsample_rate;
    let weight = codec.tensor(&format!("{prefix}.block.1.conv.weight"), [cin, cout * ks])?;
    let bias = codec.vector(&format!("{prefix}.block.1.conv.bias"), cout)?;
    let mut up = dev.zeros([cout, frames * upsample_rate])?;
    ops::conv_transpose1d(dev, &h, &weight, &bias, &mut up, ks, upsample_rate)?;

    for (unit, dilation) in [(2usize, 1usize), (3, 3), (4, 9)] {
        up = residual_unit(codec, &up, &format!("{prefix}.block.{unit}"), dilation)?;
    }
    Ok(up)
}

impl Codec {
    /// SEANet tail only: `[latent, frames]` (upsampler output) -> `[1, samples]`.
    ///
    /// Split out so the tests can drive it with a fixed input and bisect GPU/CPU
    /// differences without re-running the whole decode.
    pub fn seanet(&self, upsampled: &Tensor) -> Result<Tensor> {
        let cfg = &self.cfg;
        let mut h = causal_conv(self, upsampled, "decoder.decoder.0.conv.weight",
                                "decoder.decoder.0.conv.bias", cfg.decoder_dim, 7, 1)?;
        for (index, rate) in cfg.upsample_rates.iter().enumerate() {
            let prefix = format!("decoder.decoder.{}", index + 1);
            h = decoder_block(self, &h, &prefix, *rate)?;
        }
        let channels = h.shape()[0];
        let alpha = self.vector("decoder.decoder.5.alpha", channels)?;
        let beta = self.vector("decoder.decoder.5.beta", channels)?;
        ops::snake_beta(self.device(), &mut h, &alpha, &beta)?;
        let mut pcm = causal_conv(self, &h, "decoder.decoder.6.conv.weight",
                                  "decoder.decoder.6.conv.bias", 1, 7, 1)?;

        // `wav.clamp(min=-1, max=1)`: the reference's own saturation, which is also
        // where its occasional click comes from.
        let mut host = self.device().read(&pcm)?;
        for v in host.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        self.device().write(&mut pcm, &host)?;
        Ok(pcm)
    }

    /// Codes -> PCM in `[-1, 1]`, `[1, frames * 1920]`.
    pub fn decoder(&self, codes: &[u32], frames: usize) -> Result<Tensor> {
        let dev = self.device();
        let cfg = &self.cfg;

        let quantized = self.dequantize(codes, frames)?; // [codebook_dim, frames]
        let mut h = causal_conv(self, &quantized, "decoder.pre_conv.conv.weight",
                                "decoder.pre_conv.conv.bias", cfg.latent_dim, 3, 1)?;

        let mut time_major = dev.zeros([frames, cfg.latent_dim])?;
        ops::transpose(dev, &h, &mut time_major)?;
        let transformed = self.transformer(&time_major)?;
        let mut channels_major = dev.zeros([cfg.latent_dim, frames])?;
        ops::transpose(dev, &transformed, &mut channels_major)?;
        let upsampled = self.upsample(&channels_major)?;
        self.seanet(&upsampled)
    }
}
