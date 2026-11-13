//! End-to-end parity: same text, same reference audio, same recorded noise as the
//! reference implementation -> same codes and the same waveform.

mod common;

use anyhow::Result;
use common::{Goldens, max_abs_diff};
use m5_tts_wgpu::pipeline::{Args, NoiseSource, Pipeline, RecordedNoise};
use std::path::PathBuf;

fn dirs() -> Option<(PathBuf, PathBuf, PathBuf)> {
    let root = std::env::var("M5_WEIGHTS").unwrap_or_else(|_| r"D:\M5-TTS\weights".to_string());
    let model = PathBuf::from(&root).join("M5-TTS-50M-VoiceClone");
    let codec = PathBuf::from(&root).join("M5-TTS-Audio-Tokenizer");
    let wav = PathBuf::from(r"D:\M5-TTS\assets\aslp_default_voice.wav");
    (model.join("model.safetensors").is_file() && wav.is_file()).then(|| (model, codec, wav))
}

#[test]
fn end_to_end_matches_reference() -> Result<()> {
    let Some(goldens) = Goldens::open() else {
        eprintln!("skipping: run scripts/dump_acoustic.py");
        return Ok(());
    };
    let Some((model, codec, wav)) = dirs() else {
        eprintln!("skipping: weights or reference wav missing");
        return Ok(());
    };
    // The golden's last row is the EOS frame (all-zero tokens, clipped to 0); the
    // port stops *before* appending it, so compare the real frames only.
    let all_codes: Vec<u32> = goldens.i64("codes").iter().map(|v| *v as u32).collect();
    let real_frames = all_codes.len() / 16 - 1;
    let want_codes = &all_codes[..real_frames * 16];
    let want_audio = goldens.f32("final_audio");
    let noise = goldens.f32("noise_all");
    let prompt_text = "猪携带该疾病，然后通过蚊子传染给人类。".to_string();
    let text = "你好，欢迎使用我们的语音合成模型，今天天气很好。".to_string();

    for spec in ["cpu", "gpu"] {
        let pipeline = Pipeline::load(&model, &codec, spec)?;
        let args = Args {
            text: text.clone(),
            prompt_wav: Some(wav.clone()),
            prompt_text: Some(prompt_text.clone()),
            ..Default::default()
        };
        let mut noise_source = RecordedNoise::new(noise.clone());
        let out = pipeline.synthesize(&args, &mut noise_source as &mut dyn NoiseSource)?;

        let generated: Vec<u32> = out.generated.clone();
        let frames = generated.len() / 16;
        eprintln!("[{spec}] generated {frames} frames (reference {})", want_codes.len() / 16);
        let mismatches = generated.iter().zip(want_codes).filter(|(a, b)| a != b).count();
        eprintln!("[{spec}] code mismatches: {mismatches} / {}", generated.len().min(want_codes.len()));
        // The port must stop on its own at the same frame the reference did.
        assert_eq!(frames, real_frames, "frame count (EOS parity)");
        assert_eq!(mismatches, 0, "generated codes differ");

        let diff = max_abs_diff(&want_audio, &out.audio);
        let signal: f32 = want_audio.iter().map(|v| v * v).sum();
        let noise_energy: f32 = want_audio.iter().zip(&out.audio).map(|(a, b)| (a - b) * (a - b)).sum();
        let snr = 10.0 * (signal / noise_energy.max(1e-20)).log10();
        eprintln!("[{spec}] audio {} samples vs {}: max|diff|={diff:.3e}, SNR={snr:.1} dB",
                  out.audio.len(), want_audio.len());
        assert_eq!(out.audio.len(), want_audio.len(), "audio length");
        assert!(snr > 60.0, "audio SNR {snr:.1} dB");
    }
    Ok(())
}
