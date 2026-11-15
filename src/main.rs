//! `m5-tts` CLI — thin wrapper over the library.

use anyhow::{Context, Result};
use clap::Parser;
use m5_tts_wgpu::conditioning::format_conditioned_text;
use m5_tts_wgpu::config::{ConditioningMeta, ModelConfig};
use m5_tts_wgpu::pipeline::{Args, NoiseSource, Pipeline, PrngNoise};
use m5_tts_wgpu::tok::Tok;
use std::path::PathBuf;

/// M5-TTS 50M in Rust (wgpu + hand-written CPU).
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Checkpoint bundle (config.json, conditioning.json, tokenizer/, model.safetensors)
    #[arg(long)]
    model: PathBuf,
    /// Audio tokenizer bundle
    #[arg(long)]
    codec: PathBuf,
    /// cpu | gpu
    #[arg(long, default_value = "gpu")]
    device: String,

    /// Text to synthesise.
    #[arg(long)]
    text: Option<String>,
    /// Reference audio for zero-shot cloning (needs --prompt-text).
    #[arg(long)]
    prompt_wav: Option<PathBuf>,
    /// Exact transcript of the reference audio.
    #[arg(long)]
    prompt_text: Option<String>,
    /// Preset speaker id (CustomVoice checkpoints).
    #[arg(long, default_value_t = 1)]
    speaker_id: u32,
    #[arg(long, default_value = "Chinese")]
    lang: String,
    #[arg(long)]
    accent: Option<String>,
    #[arg(long)]
    caption: Option<String>,
    #[arg(long, default_value_t = 0.9)]
    temperature: f32,
    #[arg(long, default_value_t = 10)]
    topk: usize,
    /// Longest utterance, in milliseconds (reference `max_audio_length_ms`,
    /// default 40 s; generation stops there even without an EOS).
    #[arg(long, default_value_t = 40000.0)]
    max_audio_length_ms: f64,
    #[arg(long, default_value_t = 1234)]
    seed: u64,
    #[arg(long, default_value = "output.wav")]
    output: PathBuf,

    /// Print the conditioned prompt and token ids, then exit.
    #[arg(long)]
    tokenize: Option<String>,
    /// Validate the checkpoint against the port's tensor contract and exit.
    #[arg(long)]
    inspect: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.inspect {
        use m5_tts_wgpu::weights::{lm_specs, validate, Weights};
        let _cfg = ModelConfig::load(&cli.model.join("config.json"))?;
        let meta = ConditioningMeta::load(&cli.model.join("conditioning.json"))?;
        let tok = Tok::load(&cli.model, &meta)?;
        let weights = Weights::load(&cli.model.join("model.safetensors"))?;
        let params = validate(&weights, &lm_specs(tok.vocab_size(), 2051, 16))?;
        println!("tensors: {} | LM params: {:.1}M", weights.len(), params as f64 / 1e6);
        return Ok(());
    }

    if let Some(text) = cli.tokenize {
        let meta = ConditioningMeta::load(&cli.model.join("conditioning.json"))?;
        let tok = Tok::load(&cli.model, &meta)?;
        let speaker = cli.speaker_id.to_string();
        let prompt = format_conditioned_text(&text, Some(&speaker), cli.accent.as_deref(),
                                             Some(&cli.lang), cli.caption.as_deref())?;
        println!("{prompt}");
        println!("{} token ids", tok.encode(&prompt)?.len());
        return Ok(());
    }

    let text = cli.text.clone().context("--text is required")?;
    let load_start = std::time::Instant::now();
    let pipeline = Pipeline::load(&cli.model, &cli.codec, &cli.device)?;
    eprintln!("device: {} (weights resident in {:.2}s)", pipeline.describe(),
              load_start.elapsed().as_secs_f64());

    let args = Args {
        text,
        prompt_wav: cli.prompt_wav,
        prompt_text: cli.prompt_text,
        speaker_id: cli.speaker_id,
        lang: cli.lang,
        accent: cli.accent,
        caption: cli.caption,
        temperature: cli.temperature,
        topk: cli.topk,
        max_audio_frames: 500,
    }
    .with_max_audio_length_ms(cli.max_audio_length_ms)?;
    let mut noise = PrngNoise::new(cli.seed);
    let started = std::time::Instant::now();
    let out = pipeline.synthesize(&args, &mut noise as &mut dyn NoiseSource)?;
    let elapsed = started.elapsed().as_secs_f64();
    let audio_s = out.audio.len() as f64 / out.sample_rate as f64;

    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: out.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&cli.output, spec)?;
    for sample in &out.audio {
        let value = (sample.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        writer.write_sample(value)?;
    }
    writer.finalize()?;
    println!("{} -> {} ({} frames, {:.2}s audio in {:.2}s, RTF {:.2})",
             out.prompt_frames, cli.output.display(), out.frames, audio_s, elapsed, elapsed / audio_s);
    Ok(())
}
