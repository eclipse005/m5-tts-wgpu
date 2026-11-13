//! `tokenizer/tokenizer.json` — the LLaMA-3-style BPE with the eight control
//! tokens already appended.
//!
//! The bundle's own post-processor is `TemplateProcessing(bos $A eos)`, which is
//! exactly what `load_conditioning_tokenizer` reinstalls, so the port loads the
//! file unchanged and only re-checks the vocabulary contract.

use anyhow::{bail, Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

use crate::conditioning::CONTROL_TOKENS;
use crate::config::ConditioningMeta;

pub struct Tok {
    inner: Tokenizer,
    control_ids: [u32; 8],
    vocab_size: usize,
}

impl Tok {
    /// `bundle_dir` holds the flat bundle layout used by this port
    /// (`tokenizer.json` at the root); the upstream `tokenizer/` subdirectory is
    /// still accepted so an untouched Hugging Face download works too.
    pub fn load(bundle_dir: &Path, meta: &ConditioningMeta) -> Result<Self> {
        let root = bundle_dir.join("tokenizer.json");
        let path = if root.is_file() { root } else { bundle_dir.join("tokenizer").join("tokenizer.json") };
        let inner = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("load {}: {e}", path.display()))?;

        let mut control_ids = [0u32; 8];
        for (slot, token) in control_ids.iter_mut().zip(CONTROL_TOKENS) {
            *slot = inner
                .token_to_id(token)
                .with_context(|| format!("tokenizer is missing control token {token}"))?;
        }
        let vocab_size = inner.get_vocab_size(true);
        if vocab_size != meta.vocab_size {
            bail!(
                "tokenizer vocab size {vocab_size} != conditioning.json {}",
                meta.vocab_size
            );
        }
        // Control tokens must occupy the final eight ids in declaration order.
        let expected: Vec<u32> = (meta.base_vocab_size as u32..meta.vocab_size as u32).collect();
        if control_ids != expected.as_slice() {
            bail!("control token ids {control_ids:?} != expected {expected:?}");
        }
        for (token, id) in CONTROL_TOKENS.iter().zip(control_ids) {
            let declared = meta
                .control_token_ids
                .get(*token)
                .with_context(|| format!("conditioning.json is missing {token}"))?;
            if *declared != id {
                bail!("{token}: tokenizer id {id} != conditioning.json {declared}");
            }
        }
        let bos = inner.token_to_id("<|begin_of_text|>").context("missing BOS")?;
        let eos = inner.token_to_id("<|end_of_text|>").context("missing EOS")?;
        if bos != meta.bos_token_id || eos != meta.eos_token_id {
            bail!(
                "BOS/EOS ids {bos}/{eos} != conditioning.json {}/{}",
                meta.bos_token_id,
                meta.eos_token_id
            );
        }
        Ok(Self {
            inner,
            control_ids,
            vocab_size,
        })
    }

    /// One encode call, BOS/EOS applied by the post-processor.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn control_ids(&self) -> [u32; 8] {
        self.control_ids
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
