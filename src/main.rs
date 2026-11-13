//! `m5-tts` CLI — thin wrapper over the library.

use anyhow::{Context, Result};
use clap::Parser;
use m5_tts_wgpu::conditioning::format_conditioned_text;
use m5_tts_wgpu::config::{ConditioningMeta, ModelConfig};
use m5_tts_wgpu::tok::Tok;
use std::path::PathBuf;

/// M5-TTS 50M in Rust (wgpu + hand-written CPU).
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Checkpoint bundle directory (config.json, conditioning.json, tokenizer/, model.safetensors)
    #[arg(long)]
    model: PathBuf,
    /// Print the conditioned prompt and token ids, then exit.
    #[arg(long)]
    tokenize: Option<String>,
    /// Validate the checkpoint against the port's tensor contract and exit.
    #[arg(long)]
    inspect: bool,
    #[arg(long, default_value = "Chinese")]
    lang: String,
    #[arg(long)]
    accent: Option<String>,
    #[arg(long)]
    caption: Option<String>,
    #[arg(long, default_value_t = 1)]
    speaker_id: u32,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = ModelConfig::load(&cli.model.join("config.json"))?;
    let meta = ConditioningMeta::load(&cli.model.join("conditioning.json"))?;
    let tok = Tok::load(&cli.model, &meta)?;
    println!(
        "model: {} + {} | vocab {} (text {} + {} control)",
        cfg.backbone_flavor,
        cfg.decoder_flavor,
        tok.vocab_size(),
        meta.base_vocab_size,
        tok.control_ids().len()
    );

    if cli.inspect {
        use m5_tts_wgpu::weights::{lm_specs, validate, DType, Weights};
        let path = cli.model.join("model.safetensors");
        let weights = Weights::load(&path)?;
        let params = validate(&weights, &lm_specs(tok.vocab_size(), 2051, 16))?;
        let mut dtypes = std::collections::BTreeMap::new();
        for name in weights.names() {
            *dtypes.entry(format!("{:?}", weights.get(name)?.dtype)).or_insert(0usize) += 1;
        }
        let _ = DType::F32;
        println!("tensors: {} {dtypes:?}", weights.len());
        println!("LM params: {:.1}M (validated against {} specs)", params as f64 / 1e6,
                 lm_specs(tok.vocab_size(), 2051, 16).len());
    }

    if let Some(text) = cli.tokenize {
        let speaker = if cli.speaker_id == 0 {
            "0".to_string()
        } else {
            cli.speaker_id.to_string()
        };
        let prompt = format_conditioned_text(
            &text,
            Some(&speaker),
            cli.accent.as_deref(),
            Some(&cli.lang),
            cli.caption.as_deref(),
        )
        .context("build conditioning prompt")?;
        println!("{prompt}");
        let ids = tok.encode(&prompt)?;
        println!("{} token ids: {ids:?}", ids.len());
    }
    Ok(())
}
