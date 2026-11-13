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
use m5_tts_wgpu::ops;

/// Generate tighter than this and the stage is wrong, not just reordered.
const TOL: f32 = 2e-4;

/// Signal-to-noise ratio in dB between the reference and the port.
fn snr_of(want: &[f32], got: &[f32]) -> (f32, f32) {
    let signal: f32 = want.iter().map(|v| v * v).sum();
    let noise: f32 = want.iter().zip(got).map(|(a, b)| (a - b) * (a - b)).sum();
    (10.0 * (signal / noise.max(1e-20)).log10(), max_abs_diff(want, got))
}

#[test]
fn codec_decode_stages_match_reference() -> Result<()> {
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
    let (want_tr_shape, want_tr) = goldens.matrix("codec_transformer"); // [frames, 1024]
    let (want_up_shape, want_up) = goldens.matrix("codec_upsample"); // [1024, frames * 4]
    let want_pcm = goldens.f32("pcm"); // mono, frames * 1920
    let want_pcm_short = goldens.f32("pcm_short");
    let codes_short: Vec<u32> = goldens.i64("codes_short").iter().map(|v| *v as u32).collect();
    let short_frames = codes_short.len() / 16;

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

        let transformed = codec.transformer(&pre)?;
        assert_eq!(transformed.shape(), want_tr_shape);
        let got = dev.read(&transformed)?;
        let diff = max_abs_diff(&want_tr, &got);
        eprintln!("[{}] transformer: max|diff|={diff:.3e}", dev.describe());
        assert!(diff < TOL, "transformer diff {diff}");

        let mut channels_major = dev.zeros([want_tr_shape[1], frames])?;
        ops::transpose(dev, &transformed, &mut channels_major)?;
        let upsampled = codec.upsample(&channels_major)?;
        assert_eq!(upsampled.shape(), want_up_shape);
        let got = dev.read(&upsampled)?;
        let diff = max_abs_diff(&want_up, &got);
        let scale = want_up.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!("[{}] upsample: max|diff|={:.3e} rel={:.2e} |golden|max={}",
                  dev.describe(), diff, diff / scale, scale);
        // Two transposed convs plus two ConvNeXt blocks: the only difference from
        // the reference is fp32 summation order, so gate on relative error and
        // let the PCM SNR below be the audible gate.
        assert!(diff / scale < 2e-4, "upsample rel diff {:.3e}", diff / scale);

        // Full-length PCM is CPU-only for now: the naive conv over the last block
        // is one 18M-output dispatch and the Windows TDR watchdog kills the
        // device. The short slice below exercises every SEANet block on the GPU,
        // and the tiled conv kernel will lift this restriction.
        if !dev.is_gpu() {
            let pcm = codec.decoder(&codes, frames)?;
            assert_eq!(pcm.shape(), [1, want_pcm.len()]);
            let got = dev.read(&pcm)?;
            let diff = max_abs_diff(&want_pcm, &got);
            let (snr, _) = snr_of(&want_pcm, &got);
            eprintln!("[{}] pcm: {} samples, max|diff|={:.3e}, SNR={:.1} dB",
                      dev.describe(), want_pcm.len(), diff, snr);
            assert!(snr > 60.0, "pcm SNR {snr:.1} dB");
        }

        let short = codec.decoder(&codes_short, short_frames)?;
        assert_eq!(short.shape(), [1, want_pcm_short.len()]);
        let got = dev.read(&short)?;
        let (snr, _) = snr_of(&want_pcm_short, &got);
        eprintln!("[{}] pcm({} frames): {} samples, SNR={:.1} dB",
                  dev.describe(), short_frames, want_pcm_short.len(), snr);
        assert!(snr > 60.0, "short pcm SNR {snr:.1} dB");
    }
    Ok(())
}

/// Encoder half: prompt audio -> SEANet latent, compared stage by stage.
#[test]
fn codec_encoder_matches_reference() -> Result<()> {
    let Some(goldens) = Goldens::open() else {
        eprintln!("skipping: run scripts/dump_acoustic.py");
        return Ok(());
    };
    let Some(dir) = codec_dir() else {
        eprintln!("skipping: no codec bundle (set M5_CODEC)");
        return Ok(());
    };
    let audio = goldens.f32("prompt_audio");
    let (want_seanet_shape, want_seanet) = goldens.matrix("prompt_seanet");

    for dev in [Device::Cpu, Device::open("gpu")?] {
        let codec = Codec::load(&dir, dev, 24000)?;
        let dev = codec.device();
        let input = dev.tensor([1, audio.len()], &audio)?;
        let latent = codec.seanet_encode(&input)?;
        assert_eq!(latent.shape(), want_seanet_shape);
        let diff = max_abs_diff(&want_seanet, &dev.read(&latent)?);
        eprintln!("[{}] seanet encode: max|diff|={diff:.3e}", dev.describe());
        assert!(diff < 2e-3, "seanet encoder diff {diff}");

        // transformer works on [frames, channels]
        let (want_tr_shape, want_tr) = goldens.matrix("prompt_transformer");
        let mut time_major = dev.zeros([want_seanet_shape[1], want_seanet_shape[0]])?;
        ops::transpose(dev, &latent, &mut time_major)?;
        let transformed = codec.encoder_transformer(&time_major)?;
        // The golden is channel-major ([1, 512, 86]); the port keeps the
        // transformer input time-major, so transpose back for comparison.
        let mut transformed_channels = dev.zeros([transformed.shape()[1], transformed.shape()[0]])?;
        ops::transpose(dev, &transformed, &mut transformed_channels)?;
        assert_eq!(transformed_channels.shape(), want_tr_shape);
        let diff = max_abs_diff(&want_tr, &dev.read(&transformed_channels)?);
        eprintln!("[{}] encoder transformer: max|diff|={diff:.3e}", dev.describe());
        assert!(diff < 2e-3, "encoder transformer diff {diff}");

        let (want_ds_shape, want_ds) = goldens.matrix("prompt_downsample");
        let mut channels_major = dev.zeros([transformed.shape()[1], transformed.shape()[0]])?;
        ops::transpose(dev, &transformed, &mut channels_major)?;
        let downsampled = codec.encoder_downsample(&channels_major)?;
        assert_eq!(downsampled.shape(), want_ds_shape);
        let diff = max_abs_diff(&want_ds, &dev.read(&downsampled)?);
        eprintln!("[{}] encoder downsample: max|diff|={diff:.3e}", dev.describe());
        assert!(diff < 5e-3, "encoder downsample diff {diff}");

        // SplitRVQ: 1 semantic + 15 acoustic codebooks, frame-major.
        let want_codes = goldens.i64("prompt_codes");
        let code_shape = goldens.shape("prompt_codes"); // [1, 16, frames]
        let want_frames = code_shape[2];
        let codes = codec.encode_codes(&downsampled)?;
        assert_eq!(codes.len(), want_frames * 16);
        // Golden is [1, 16, frames] (codebook-major); the port returns frame-major.
        let mut want_frame_major = Vec::with_capacity(want_frames * 16);
        for t in 0..want_frames {
            for k in 0..16 {
                want_frame_major.push(want_codes[k * want_frames + t] as u32);
            }
        }
        let mismatches = codes.iter().zip(&want_frame_major).filter(|(a, b)| a != b).count();
        eprintln!("[{}] SplitRVQ: {} codes, {} mismatches", dev.describe(), codes.len(), mismatches);
        assert_eq!(mismatches, 0, "quantised codes differ");
    }
    Ok(())
}

/// `prepare_prompt` must reproduce the reference's silence trimming bit for bit:
/// the codec sees exactly this waveform.
#[test]
fn prepare_prompt_matches_reference() -> Result<()> {
    let Some(goldens) = Goldens::open() else {
        eprintln!("skipping: run scripts/dump_acoustic.py");
        return Ok(());
    };
    let want = goldens.f32("prompt_audio");
    let path = std::path::Path::new(r"D:\M5-TTS\assets\aslp_default_voice.wav");
    if !path.is_file() {
        eprintln!("skipping: {} not found", path.display());
        return Ok(());
    }
    let got = m5_tts_wgpu::audio::prepare_prompt(path, 24000)?;
    eprintln!("prepare_prompt: {} samples vs golden {}", got.len(), want.len());
    assert_eq!(got.len(), want.len(), "sample count");
    let mut diff = 0.0f32;
    for (a, b) in want.iter().zip(&got) {
        diff = diff.max((a - b).abs());
    }
    eprintln!("prepare_prompt: max|diff|={diff:.3e}");
    assert!(diff < 1e-6, "prepare_prompt diff {diff}");
    Ok(())
}
