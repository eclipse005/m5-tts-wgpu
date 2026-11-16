//! `m5-tts` — M5-TTS 50M in Rust (wgpu GPU + hand-written CPU).

use anyhow::{bail, Result};
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use m5_tts_wgpu::config::{ConditioningMeta, ModelConfig};
use m5_tts_wgpu::pipeline::{Args, Pipeline};
use m5_tts_wgpu::tok::Tok;
use m5_tts_wgpu::weights::{lm_specs, validate, Weights};
use m5_tts_wgpu::audio;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Instant;

/// M5-TTS 50M speech synthesis — wgpu GPU and hand-written CPU inference.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
#[allow(clippy::large_enum_variant)] // one variant per invocation, never stored
enum Command {
    /// Synthesise speech: zero-shot cloning from a prompt, or a preset speaker.
    Synth(Synth),
    /// Check a checkpoint bundle against this port's tensor and vocabulary contract.
    Inspect(Inspect),
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DeviceChoice {
    /// GPU when one is usable, otherwise the CPU backend (default).
    Auto,
    /// Hand-written CPU backend, no wgpu.
    Cpu,
    /// wgpu GPU (Vulkan / Metal / GL); fails when no adapter is usable.
    #[value(alias = "cuda")]
    Gpu,
}

impl DeviceChoice {
    fn spec(self) -> &'static str {
        match self {
            DeviceChoice::Auto => "auto",
            DeviceChoice::Cpu => "cpu",
            DeviceChoice::Gpu => "gpu",
        }
    }
}

#[derive(ClapArgs, Debug)]
struct Bundle {
    /// M5 bundle: config.json, conditioning.json, tokenizer/, model.safetensors.
    #[arg(long)]
    model: PathBuf,
    /// Audio tokenizer bundle: config.json and its safetensors.
    #[arg(long)]
    codec: PathBuf,
}

#[derive(ClapArgs, Debug)]
struct Synth {
    #[command(flatten)]
    bundle: Bundle,
    /// Text to speak; `-` reads it from standard input.
    #[arg(long)]
    text: String,
    /// Reference recording for zero-shot cloning (needs --prompt-text).
    #[arg(long)]
    prompt_wav: Option<PathBuf>,
    /// Exact transcript of the reference recording.
    #[arg(long)]
    prompt_text: Option<String>,
    /// Preset speaker id (CustomVoice checkpoints only).
    #[arg(long, default_value_t = 1)]
    speaker_id: u32,
    /// Language of the text: Chinese, English, German, French, Japanese, Korean.
    #[arg(long, default_value = "Chinese")]
    lang: String,
    /// Dialect: `chuan` (Sichuanese) or `yue` (Cantonese, also converts the text).
    #[arg(long)]
    accent: Option<String>,
    /// Style caption, e.g. `happy`.
    #[arg(long)]
    caption: Option<String>,
    #[arg(long, default_value_t = 0.9)]
    temperature: f32,
    #[arg(long, default_value_t = 10)]
    topk: usize,
    /// Sampler seed: the same seed and text reproduce the same audio, including how
    /// a long text was split.
    #[arg(long, default_value_t = 1234)]
    seed: u64,
    /// Reference semantics: one utterance of at most this long, truncated with a
    /// warning when it does not fit. Omit it and a long text is read to its end as
    /// several ~40 s utterances joined by a 150 ms pause.
    #[arg(long)]
    max_audio_length_ms: Option<f64>,
    /// WAV file to write; `-` streams 24 kHz mono PCM16 to standard output.
    #[arg(long, default_value = "output.wav")]
    output: String,
    #[arg(long, value_enum, default_value_t = DeviceChoice::Auto)]
    device: DeviceChoice,
}

#[derive(ClapArgs, Debug)]
struct Inspect {
    #[command(flatten)]
    bundle: Bundle,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Synth(a) => synth(a),
        Command::Inspect(a) => inspect(a),
    }
}

fn synth(a: Synth) -> Result<()> {
    let text = if a.text == "-" {
        let mut text = String::new();
        std::io::stdin().read_to_string(&mut text)?;
        text
    } else {
        a.text
    };

    let started = Instant::now();
    let pipeline = Pipeline::load(&a.bundle.model, &a.bundle.codec, a.device.spec())?;
    eprintln!(
        "device: {} (weights resident in {:.2}s)",
        pipeline.describe(),
        started.elapsed().as_secs_f64()
    );

    let mut args = Args {
        text,
        prompt_wav: a.prompt_wav,
        prompt_text: a.prompt_text,
        speaker_id: a.speaker_id,
        lang: a.lang,
        accent: a.accent,
        caption: a.caption,
        temperature: a.temperature,
        topk: a.topk,
        seed: a.seed,
        ..Args::default()
    };
    if let Some(ms) = a.max_audio_length_ms {
        args = args.with_max_audio_length_ms(ms)?;
    }

    let started = Instant::now();
    let out = pipeline.synthesize(&args)?;
    let elapsed = started.elapsed().as_secs_f64();
    let audio_s = out.audio.len() as f64 / out.sample_rate as f64;

    if a.output == "-" {
        // Raw PCM16 on stdout, log lines on stderr: `m5-tts synth … --output - |
        // ffplay -f s16le -ar 24000 -ch_layout mono -`.
        let mut stdout = std::io::stdout().lock();
        let mut write = |bytes: &[u8]| -> Result<()> {
            match stdout.write_all(bytes) {
                Ok(()) => Ok(()),
                // A reader that stops early (`| head -c …`) is not an error.
                Err(err) if err.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
                Err(err) => Err(err.into()),
            }
        };
        for sample in &out.audio {
            write(&((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes())?;
        }
        let _ = stdout.flush();
    } else {
        audio::write_wav(a.output.as_ref(), &out.audio, out.sample_rate)?;
    }
    eprintln!(
        "{:.2}s audio in {:.2}s (RTF {:.2}) from {} frames",
        audio_s,
        elapsed,
        elapsed / audio_s,
        out.frames
    );
    Ok(())
}

fn inspect(a: Inspect) -> Result<()> {
    let config = ModelConfig::load(&a.bundle.model.join("config.json"))?;
    let meta = ConditioningMeta::load(&a.bundle.model.join("conditioning.json"))?;
    let tok = Tok::load(&a.bundle.model, &meta)?;
    let weights = Weights::load(&a.bundle.model.join("model.safetensors"))?;
    let params = validate(&weights, &lm_specs(tok.vocab_size(), 2051, 16))?;

    if !a.bundle.codec.join("config.json").is_file() {
        bail!("{} has no config.json", a.bundle.codec.display());
    }
    println!("language model: {} backbone, {} decoder", config.backbone_flavor, config.decoder_flavor);
    println!("tokenizer: {} tokens, conditioning format {:?}", tok.vocab_size(), meta.format);
    println!(
        "tensors: {} | parameters: {:.1}M (the 50M figure counts backbone + decoder)",
        weights.len(),
        params as f64 / 1e6
    );
    println!("codec: {}", a.bundle.codec.display());
    println!("bundle ok");
    Ok(())
}
