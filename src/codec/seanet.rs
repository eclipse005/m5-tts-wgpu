//! SEANet decoder tail: `decoder.0` conv, four upsampling blocks, final SnakeBeta
//! and the mono output conv. This is where codes become a waveform.

use anyhow::Result;

use super::Codec;
use crate::cpu_ops;
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
    if let Device::Gpu(gpu) = dev {
        // Channels-last detour: the staged CL kernel is 3-5x faster than the
        // naive tiled one at these shapes, and on the P104 the two transposes
        // cost ~1 ms per 72 MB pass - noise against the kernel time.
        let packed = codec.packed_conv1d(weight_name, cout, cin, ks)?;
        let x_cl = gpu.empty("conv1d.x_cl", frames * cin);
        crate::gpu_ops::transpose(gpu, x.gpu_buffer()?, &x_cl, cin, frames);
        let y_cl = gpu.empty("conv1d.y_cl", frames * cout);
        crate::gpu_ops::conv1d_cl(gpu, &x_cl, packed.gpu_buffer()?, bias.gpu_buffer()?, &y_cl,
                                  cin, cout, ks, dilation, 1, (ks - 1) * dilation, frames, frames);
        crate::gpu_ops::transpose(gpu, &y_cl, out.gpu_buffer()?, frames, cout);
        return Ok(out);
    }
    let conv = Conv1d::causal(cin, cout, ks, frames).dilated(dilation);
    ops::conv1d(dev, x, &weight, &bias, &mut out, &conv)?;
    Ok(out)
}

/// One `DecoderResidualUnit`: snake -> conv7(dilated) -> snake -> conv1 -> add.
fn residual_unit(codec: &Codec, x: &Tensor, prefix: &str, dilation: usize) -> Result<Tensor> {
    let dev: &Device = codec.device();
    let channels = x.shape()[0];
    // CPU: fuse each snake into its convolution and the residual into the last one.
    if !dev.is_gpu() {
        let cv1 = codec.snake_conv(x, &format!("{prefix}.act1"), &format!("{prefix}.conv1.conv.weight"),
                                   &format!("{prefix}.conv1.conv.bias"), channels, 7, dilation, None)?;
        return codec.snake_conv(&cv1, &format!("{prefix}.act2"), &format!("{prefix}.conv2.conv.weight"),
                                &format!("{prefix}.conv2.conv.bias"), channels, 1, 1, Some(x));
    }

    let alpha = codec.vector(&format!("{prefix}.act1.alpha"), channels)?;
    let beta = codec.vector(&format!("{prefix}.act1.beta"), channels)?;
    let h = ops::snake_beta(dev, x, &alpha, &beta)?;
    let h3 = causal_conv(codec, &h, &format!("{prefix}.conv1.conv.weight"),
                         &format!("{prefix}.conv1.conv.bias"), channels, 7, dilation)?;
    let alpha2 = codec.vector(&format!("{prefix}.act2.alpha"), channels)?;
    let beta2 = codec.vector(&format!("{prefix}.act2.beta"), channels)?;
    let h4 = ops::snake_beta(dev, &h3, &alpha2, &beta2)?;
    let out = causal_conv(codec, &h4, &format!("{prefix}.conv2.conv.weight"),
                          &format!("{prefix}.conv2.conv.bias"), channels, 1, 1)?;

    let mut sum = dev.zeros(x.shape())?;
    ops::add(dev, &out, x, &mut sum)?;
    Ok(sum)
}

/// One `DecoderBlock`: snake -> transposed conv -> three residual units.
fn decoder_block(codec: &Codec, x: &Tensor, prefix: &str, upsample_rate: usize) -> Result<Tensor> {
    let dev = codec.device();
    let [cin, frames] = x.shape();
    let profile = std::env::var("M5_PROFILE_DEEP").is_ok();
    let mut phase = std::time::Instant::now();
    // GPU work is queued, not executed: drain it before reading the clock so a
    // stage's elapsed includes its own execution (flush is a no-op on CPU).
    let lap = |name: &str, phase: &mut std::time::Instant| {
        if let crate::device::Device::Gpu(gpu) = dev { gpu.sync(); }
        if profile { eprintln!("        {name}: {:.3}s", phase.elapsed().as_secs_f64()); }
        *phase = std::time::Instant::now();
    };
    let cout = cin / 2;

    let alpha = codec.vector(&format!("{prefix}.block.0.alpha"), cin)?;
    let beta = codec.vector(&format!("{prefix}.block.0.beta"), cin)?;
    let h = ops::snake_beta(dev, x, &alpha, &beta)?;

    // CausalTransConvNet(cin, cout, kernel=2*rate, stride=rate), trimmed to frames*rate.
    let ks = 2 * upsample_rate;
    let weight = codec.tensor(&format!("{prefix}.block.1.conv.weight"), [cin, cout * ks])?;
    let bias = codec.vector(&format!("{prefix}.block.1.conv.bias"), cout)?;
    let mut up = dev.zeros([cout, frames * upsample_rate])?;
    if let Device::Gpu(_) = dev {
        up = codec.conv_transpose_phased(&h, &format!("{prefix}.block.1.conv.weight"),
                                         &format!("{prefix}.block.1.conv.bias"),
                                         cout, ks, upsample_rate)?;
    } else {
        ops::conv_transpose1d(dev, &h, &weight, &bias, &mut up, ks, upsample_rate)?;
    }
    lap("transposed conv", &mut phase);

    for (unit, dilation) in [(2usize, 1usize), (3, 3), (4, 9)] {
        up = residual_unit(codec, &up, &format!("{prefix}.block.{unit}"), dilation)?;
        lap(&format!("residual unit (dilation {dilation})"), &mut phase);
    }
    Ok(up)
}

impl Codec {
    /// SEANet tail only: `[latent, frames]` (upsampler output) -> `[1, samples]`.
    ///
    /// Split out so the tests can drive it with a fixed input and bisect GPU/CPU
    /// differences without re-running the whole decode.
    pub fn seanet(&self, upsampled: &Tensor) -> Result<Tensor> {
        self.seanet_impl(upsampled, None)
    }

    /// Same computation, but returns each stage's output so a GPU/CPU divergence
    /// can be localised instead of guessed at.
    pub fn seanet_trace(&self, upsampled: &Tensor) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        let mut trace = Vec::new();
        let pcm = self.seanet_impl(upsampled, Some(&mut trace))?;
        Ok((pcm, trace))
    }

    fn seanet_impl(&self, upsampled: &Tensor, mut trace: Option<&mut Vec<(String, Tensor)>>) -> Result<Tensor> {
        let cfg = &self.cfg;
        let mut h = causal_conv(self, upsampled, "decoder.decoder.0.conv.weight",
                                "decoder.decoder.0.conv.bias", cfg.decoder_dim, 7, 1)?;
        if let Some(t) = trace.as_deref_mut() {
            t.push(("conv0".to_string(), self.device().copy(&h)?));
        }
        let profile = std::env::var("M5_PROFILE").is_ok();
        let mut phase = std::time::Instant::now();
        for (index, rate) in cfg.upsample_rates.iter().enumerate() {
            let prefix = format!("decoder.decoder.{}", index + 1);
            h = decoder_block(self, &h, &prefix, *rate)?;
            if profile {
                eprintln!("      block{} ({}ch -> {:.0}k samples): {:.3}s", index + 1, h.shape()[0],
                          h.shape()[1] as f64 / 1e3, phase.elapsed().as_secs_f64());
                phase = std::time::Instant::now();
            }
            if let Some(t) = trace.as_deref_mut() {
                t.push((format!("block{}", index + 1), self.device().copy(&h)?));
            }
        }
        let channels = h.shape()[0];
        let alpha = self.vector("decoder.decoder.5.alpha", channels)?;
        let beta = self.vector("decoder.decoder.5.beta", channels)?;
        h = ops::snake_beta(self.device(), &h, &alpha, &beta)?;
        if let Some(t) = trace.as_deref_mut() {
            t.push(("snake".to_string(), self.device().copy(&h)?));
        }
        if profile { if let crate::device::Device::Gpu(gpu) = self.device() { gpu.sync(); } eprintln!("      final snake: {:.3}s", phase.elapsed().as_secs_f64()); phase = std::time::Instant::now(); }
        let mut pcm = causal_conv(self, &h, "decoder.decoder.6.conv.weight",
                                  "decoder.decoder.6.conv.bias", 1, 7, 1)?;
        if profile { if let crate::device::Device::Gpu(gpu) = self.device() { gpu.sync(); } eprintln!("      final conv: {:.3}s", phase.elapsed().as_secs_f64()); }

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
        let profile = std::env::var("M5_PROFILE").is_ok();
        let mut phase = std::time::Instant::now();

        let quantized = self.dequantize(codes, frames)?; // [codebook_dim, frames]
        if profile { if let crate::device::Device::Gpu(gpu) = self.device() { gpu.sync(); } eprintln!("    quantise {:.3}s", phase.elapsed().as_secs_f64()); phase = std::time::Instant::now(); }
        let h = causal_conv(self, &quantized, "decoder.pre_conv.conv.weight",
                                "decoder.pre_conv.conv.bias", cfg.latent_dim, 3, 1)?;

        let mut time_major = dev.zeros([frames, cfg.latent_dim])?;
        ops::transpose(dev, &h, &mut time_major)?;
        if profile { if let crate::device::Device::Gpu(gpu) = self.device() { gpu.sync(); } eprintln!("    pre_conv {:.3}s", phase.elapsed().as_secs_f64()); phase = std::time::Instant::now(); }
        let transformed = self.transformer(&time_major)?;
        if profile { if let crate::device::Device::Gpu(gpu) = self.device() { gpu.sync(); } eprintln!("    pre_transformer {:.3}s", phase.elapsed().as_secs_f64()); phase = std::time::Instant::now(); }
        let mut channels_major = dev.zeros([cfg.latent_dim, frames])?;
        ops::transpose(dev, &transformed, &mut channels_major)?;
        let upsampled = self.upsample(&channels_major)?;
        if profile { if let crate::device::Device::Gpu(gpu) = self.device() { gpu.sync(); } eprintln!("    upsample {:.3}s", phase.elapsed().as_secs_f64()); phase = std::time::Instant::now(); }
        // Channels-first stays. A channels-last rewrite (seanet_cl, same math,
        // SNR 112 dB) measures 2.66 s against 2.03 s here on the same input: at
        // these sizes the tail is DRAM-bound, so the better arithmetic intensity of
        // the channels-last kernels does not survive the streaming. See
        // examples/seanet_cl_check.rs. Cutting traffic (fusing snake/residual into
        // the convolution) is the lever that matters, not the layout.
        let pcm = self.seanet(&upsampled)?;
        if profile { eprintln!("    seanet {:.3}s", phase.elapsed().as_secs_f64()); }
        Ok(pcm)
    }
}

impl Codec {
    /// Fused `snake -> conv -> (+residual)` for the CPU path.
    ///
    /// `x` is the tensor the snake applies to; `residual` (when given) is added to
    /// the convolution result. Both fusions exist because the tail is DRAM-bound —
    /// see `cpu_ops::snake_conv1d`.
    fn snake_conv(&self, x: &Tensor, act_prefix: &str, weight_name: &str, bias_name: &str,
                  cout: usize, ks: usize, dilation: usize, residual: Option<&Tensor>) -> Result<Tensor> {
        let dev = self.device();
        let [cin, t_in] = x.shape();
        let weight = self.tensor(weight_name, [cout, cin * ks])?;
        let bias = self.vector(bias_name, cout)?;
        let alpha = self.vector(&format!("{act_prefix}.alpha"), cin)?;
        let beta = self.vector(&format!("{act_prefix}.beta"), cin)?;
        let t_out = t_in;
        let mut out = dev.zeros([cout, t_out])?;
        match dev {
            crate::device::Device::Cpu => cpu_ops::snake_conv1d(
                x.cpu_slice()?, alpha.cpu_slice()?, beta.cpu_slice()?, weight.cpu_slice()?,
                bias.cpu_slice()?, residual.map(|r| r.cpu_slice()).transpose()?, out.cpu_slice_mut()?,
                (cin, cout, ks, dilation, 1, (ks - 1) * dilation, t_in, t_out)),
            _ => anyhow::bail!("fused snake+conv is CPU-only"),
        }
        Ok(out)
    }
}
