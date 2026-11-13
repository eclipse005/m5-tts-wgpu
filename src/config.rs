//! `config.json` and `conditioning.json` of an M5 checkpoint bundle.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::Path;

/// `config.json` — the inference-relevant subset. The reference rejects
/// anything that is not the 16-codebook / 2051-token layout, so do the same.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub backbone_flavor: String,
    pub decoder_flavor: String,
    #[serde(default = "default_audio_vocab")]
    pub audio_vocab_size: usize,
    #[serde(default = "default_codebooks")]
    pub audio_num_codebooks: usize,
    #[serde(default = "default_codebooks")]
    pub use_codebooks: usize,
}

fn default_audio_vocab() -> usize {
    2051
}

fn default_codebooks() -> usize {
    16
}

impl ModelConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        let cfg: Self = serde_json::from_str(&text)
            .with_context(|| format!("parse {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<()> {
        if self.audio_vocab_size != 2051 {
            bail!("expected audio_vocab_size=2051, got {}", self.audio_vocab_size);
        }
        if self.audio_num_codebooks != 16 || self.use_codebooks != 16 {
            bail!(
                "expected 16 codebooks, got audio_num_codebooks={} use_codebooks={}",
                self.audio_num_codebooks,
                self.use_codebooks
            );
        }
        Ok(())
    }
}

/// `conditioning.json` — vocabulary contract of the saved tokenizer. The port
/// validates against it exactly like `validate_conditioning_bundle` does.
#[derive(Debug, Clone, Deserialize)]
pub struct ConditioningMeta {
    pub format: String,
    pub base_vocab_size: usize,
    pub vocab_size: usize,
    pub control_token_ids: std::collections::BTreeMap<String, u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub vocab_sha256: Option<String>,
}

pub const CONDITIONING_FORMAT: &str = "m5-caption-accent-speaker-lang-v3";

impl ConditioningMeta {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        let meta: Self = serde_json::from_str(&text)
            .with_context(|| format!("parse {}", path.display()))?;
        if meta.format != CONDITIONING_FORMAT {
            bail!(
                "unexpected conditioning format {:?}, expected {CONDITIONING_FORMAT}",
                meta.format
            );
        }
        Ok(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_codebook_count() {
        let cfg: ModelConfig = serde_json::from_str(
            r#"{"backbone_flavor":"llama-30M","decoder_flavor":"llama-20M","use_codebooks":8}"#,
        )
        .unwrap();
        assert!(cfg.validate().is_err());
    }
}
