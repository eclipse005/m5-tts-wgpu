//! M5 LM: a llama-30M backbone plus a llama-20M depth decoder.
//!
//! Mirrors `m5/model/model.py` (`CSM`). The backbone consumes one hot-vector per
//! frame — 16 audio codebooks plus the text token, summed into one hidden row —
//! and emits the codebook-0 logits; the depth decoder then resolves the other
//! fifteen codebooks from the same hidden state (see `depth.rs`).
//!
//! Dimensions come from the checkpoint flavours (`llama-30M` /
//! `llama-20M`), which is what the reference pins, so they are validated on load
//! rather than configured.

pub mod backbone;

use anyhow::{bail, Context, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use crate::conditioning::CONTROL_TOKENS;
use crate::config::{ConditioningMeta, ModelConfig};
use crate::device::{Device, Tensor};
use crate::tok::Tok;
use crate::weights::Weights;

/// Fixed geometry of the 50M checkpoint (`config.json` flavours).
pub const HIDDEN: usize = 512;
pub const HEADS: usize = 8;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 64;
pub const INTERMEDIATE: usize = 2048;
pub const BACKBONE_LAYERS: usize = 8;
pub const DEPTH_LAYERS: usize = 5;
pub const AUDIO_VOCAB: usize = 2051;
pub const CODEBOOKS: usize = 16;
/// llama-3 RoPE scaling used by every flavour in `m5/model/flavors.py`.
pub const ROPE_THETA: f64 = 500_000.0;
pub const ROPE_FACTOR: f64 = 32.0;
pub const ROPE_HIGH_FREQ_FACTOR: f64 = 4.0;
pub const ROPE_LOW_FREQ_FACTOR: f64 = 1.0;
pub const ROPE_ORIGINAL_MAX: usize = 2048;

/// An M5 checkpoint bundle with its device-resident weights and host embedding
/// tables.
pub struct Lm {
    pub cfg: ModelConfig,
    pub tok: Tok,
    pub meta: ConditioningMeta,
    dev: Device,
    weights: Weights,
    cache: RefCell<HashMap<String, Rc<Tensor>>>,
    /// Embedding tables stay on the host: one frame needs 17 lookups, and copying
    /// 333 MB of tables to the device for that would be absurd.
    text_emb: Vec<f32>,
    audio_emb: Vec<f32>,
    text_vocab: usize,
}

impl Lm {
    pub fn load(bundle: &Path, dev: Device) -> Result<Self> {
        let cfg = ModelConfig::load(&bundle.join("config.json"))?;
        let meta = ConditioningMeta::load(&bundle.join("conditioning.json"))?;
        let tok = Tok::load(bundle, &meta)?;
        if cfg.backbone_flavor != "llama-30M" || cfg.decoder_flavor != "llama-20M" {
            bail!("unsupported flavours {} / {}: this port pins llama-30M + llama-20M",
                  cfg.backbone_flavor, cfg.decoder_flavor);
        }
        let weights = Weights::load(&bundle.join("model.safetensors"))?;
        let specs = crate::weights::lm_specs(tok.vocab_size(), AUDIO_VOCAB, CODEBOOKS);
        crate::weights::validate(&weights, &specs).context("LM weight contract")?;

        let text_vocab = tok.vocab_size();
        let text_emb = weights.get_f32("text_embeddings.weight")?;
        let audio_emb = weights.get_f32("audio_embeddings.weight")?;
        if text_emb.len() != text_vocab * HIDDEN {
            bail!("text_embeddings has {} values, expected {}", text_emb.len(), text_vocab * HIDDEN);
        }
        if audio_emb.len() != AUDIO_VOCAB * CODEBOOKS * HIDDEN {
            bail!("audio_embeddings has {} values, expected {}", audio_emb.len(),
                  AUDIO_VOCAB * CODEBOOKS * HIDDEN);
        }
        Ok(Self {
            cfg,
            tok,
            meta,
            dev,
            weights,
            cache: RefCell::new(HashMap::new()),
            text_emb,
            audio_emb,
            text_vocab,
        })
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    pub fn text_vocab(&self) -> usize {
        self.text_vocab
    }

    /// Device tensor for `name`, validated against `shape`, shared via `Rc`.
    pub fn tensor(&self, name: &str, shape: [usize; 2]) -> Result<Rc<Tensor>> {
        if let Some(t) = self.cache.borrow().get(name) {
            if t.shape() != shape {
                bail!("{name}: cached shape {:?} != {shape:?}", t.shape());
            }
            return Ok(t.clone());
        }
        let entry = self.weights.get(name)?;
        if entry.shape.iter().product::<usize>() != shape[0] * shape[1] {
            bail!("{name}: checkpoint shape {:?} does not flatten to {shape:?}", entry.shape);
        }
        let tensor = Rc::new(self.dev.tensor(shape, &entry.to_f32()?)?);
        self.cache.borrow_mut().insert(name.to_string(), tensor.clone());
        Ok(tensor)
    }

    pub fn vector(&self, name: &str, len: usize) -> Result<Rc<Tensor>> {
        self.tensor(name, [len, 1])
    }

    /// Control-token ids, in `conditioning.json` order.
    pub fn control_ids(&self) -> [u32; 8] {
        self.tok.control_ids()
    }

    /// Reserved control token check used by the reference before tokenising.
    pub fn is_control_token(&self, token: &str) -> bool {
        CONTROL_TOKENS.contains(&token)
    }
}
