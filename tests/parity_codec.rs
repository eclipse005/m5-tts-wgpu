//! Codec decode stages against the reference dump, on both backends.
//!
//! Stage-by-stage because a waveform comparison tells you *that* something is
//! wrong, not where: quantiser, pre_conv and (later) the transformer, upsampler
//! and SEANet stack each get their own gate.

mod common;

use anyhow::Result;
use common::{codec_dir, max_abs_diff, Goldens};
use m5_tts_wgpu::codec::Codec;
use m5_tts_wgpu::device::Device;

/// Generate tighter than this and the stage is wrong, not just reordered.
const TOL: f32 = 2e-4;

#[test]
fn quantiser_and_pre_conv_match_reference() -> Result<()> {
    let Some(goldens) = Goldens::open() else {
        eprintln!("skipping: run scripts/dump_acoustic.py");
        return Ok(());
    };
    let Some(dir) = codec_dir() else {
        eprintln!("skipping: no codec bundle (set M5_CODEC)");
        return Ok(());
    };

    let codes: Vec<u32> = goldens.i64("codes").iter().map(|v| *v as u32).collect();
    let [frames, codebooks] = [goldens.shape("codes")[0], goldens.shape("codes")[1]];
    assert_eq!(codebooks, 16, "codes must be [frames, 16]");

    let (want_shape, want) = goldens.matrix("codec_quantized"); // [512, frames]
    let (want_pre_shape, want_pre) = goldens.matrix("codec_pre_conv"); // [frames, 1024]

    for dev in [Device::Cpu, Device::open("gpu")?] {
        let codec = Codec::load(&dir, dev, 24000)?;
        let dev = codec.device();

        let quantized = codec.dequantize(&codes, frames)?;
        assert_eq!(quantized.shape(), want_shape);
        let got = dev.read(&quantized)?;
        let diff = max_abs_diff(&want, &got);
        eprintln!("[{}] quantizer: max|diff|={diff:.3e}", dev.describe());
        assert!(diff < TOL, "quantizer diff {diff}");

        let pre = codec.pre_transformer_input(&codes, frames)?;
        assert_eq!(pre.shape(), want_pre_shape);
        let got = dev.read(&pre)?;
        let diff = max_abs_diff(&want_pre, &got);
        eprintln!("[{}] pre_conv: max|diff|={diff:.3e}", dev.describe());
        assert!(diff < TOL, "pre_conv diff {diff}");
    }
    Ok(())
}
