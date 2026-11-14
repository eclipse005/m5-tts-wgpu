//! Qwen 12 Hz speech codec (`ASLP-lab/M5-TTS-Audio-Tokenizer`).
//!
//! Decode path, mirroring `Qwen3TTSTokenizerV2Decoder.forward`:
//! codes -> RVQ split dequantise -> `pre_conv` -> `pre_transformer` ->
//! upsample (x2 transposed conv + ConvNeXt) -> SEANet decoder -> PCM.
//!
//! The encode path (reference audio -> codes) lives in `encode.rs` and shares the
//! weight map. Both graph halves call [`crate::ops`] only, so the same code runs
//! on wgpu and on the CPU twin.

pub mod encoder;
pub mod quantizer;
pub mod seanet;
pub mod seanet_cl;
pub mod transformer;
pub mod upsample;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::path::Path;

use crate::cpu_ops;
use crate::device::{Device, Tensor};
use crate::ops::{self, Conv1d};
use crate::weights::Weights;

/// `decoder_config` / `encoder_config` of the codec's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct CodecConfig {
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub num_quantizers: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub sliding_window: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub layer_scale_initial_scale: f64,
}

/// `encoder_config` of the codec: the encoder transformer has its own geometry
/// (8 heads, no GQA) and its own RoPE theta.
#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_enc_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_enc_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_window")]
    pub sliding_window: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
}

fn default_rope_theta() -> f64 {
    10_000.0
}
fn default_enc_heads() -> usize {
    8
}
fn default_head_dim() -> usize {
    64
}
fn default_enc_layers() -> usize {
    8
}
fn default_intermediate() -> usize {
    2048
}
fn default_window() -> usize {
    250
}
fn default_norm_eps() -> f64 {
    1e-5
}

#[derive(Deserialize)]
struct CodecFile {
    #[serde(default)]
    encoder_config: Option<EncoderConfig>,
    decoder_config: CodecConfig,
    #[serde(default)]
    #[allow(dead_code)]
    input_sample_rate: usize,
    #[serde(default)]
    encode_downsample_rate: usize,
}

impl CodecConfig {
    /// Frames per second of the codec (12.5 Hz for the Qwen 12 Hz tokenizer).
    ///
    /// `encode_downsample_rate` is *samples per frame* (1920), not frames per
    /// second — the reference divides the waveform length by 12.5 directly.
    pub fn frame_rate(&self, sample_rate: usize, encode_downsample_rate: usize) -> f64 {
        sample_rate as f64 / encode_downsample_rate as f64
    }
}

/// A codec bundle: parsed config plus device-resident weights.
pub struct Codec {
    pub cfg: CodecConfig,
    pub sample_rate: usize,
    /// Codec frame rate in Hz (24000 / `encode_downsample_rate`).
    pub frame_rate: f64,
    /// Encoder transformer geometry (its own heads/theta/window).
    pub enc: EncoderConfig,
    dev: Device,
    weights: Weights,
    /// Device tensors are created once per name; the SEANet stack alone is ~450
    /// tensors per decode otherwise.
    cache: RefCell<HashMap<String, Rc<Tensor>>>,
}

impl Codec {
    pub fn load(dir: &Path, dev: Device, sample_rate: usize) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("read {}/config.json", dir.display()))?;
        let file: CodecFile = serde_json::from_str(&text).context("parse codec config.json")?;
        let weights = Weights::load(&dir.join("model.safetensors"))?;
        let frame_rate = file.decoder_config.frame_rate(file.input_sample_rate, file.encode_downsample_rate);
        let enc = file.encoder_config.unwrap_or_else(|| {
            serde_json::from_str("{}").expect("encoder defaults")
        });
        Ok(Self {
            cfg: file.decoder_config,
            sample_rate,
            frame_rate,
            enc,
            dev,
            weights,
            cache: RefCell::new(HashMap::new()),
        })
    }

    /// RoPE theta of the encoder transformer (plain frequencies, no scaling).
    pub fn encoder_rope_theta(&self) -> f64 {
        self.enc.rope_theta
    }

    /// Weight cache, exposed so stage modules can memoise packed weights.
    pub(crate) fn cache_ref(&self) -> &RefCell<HashMap<String, Rc<Tensor>>> {
        &self.cache
    }

    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// Device tensor for `name`, validated against `shape` (row-major).
    ///
    /// Handed out behind `Rc` because weights are read-only and shared; copying an
    /// 11M-float conv weight per call would dominate everything else.
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

    /// 1-D parameter as a `[len, 1]` tensor (kernels take 2-D shapes).
    pub fn vector(&self, name: &str, len: usize) -> Result<Rc<Tensor>> {
        self.tensor(name, [len, 1])
    }

    /// Channels-last packed weight `[(tap*cin + ci) * cout + co]` (cached).
    ///
    /// The staged channels-last conv kernel needs this layout; the checkpoint
    /// stores `[cout, cin, ks]`. Packing is a one-time host pass per weight.
    pub fn packed_conv1d(&self, name: &str, cout: usize, cin: usize, ks: usize) -> Result<Rc<Tensor>> {
        let key = format!("packed.{name}");
        if let Some(t) = self.cache.borrow().get(&key) {
            return Ok(t.clone());
        }
        let raw = self.weights.get_f32(name)?;
        let packed = crate::cpu_ops::pack_conv_weights(&raw, cout, cin, ks);
        let tensor = Rc::new(self.dev.tensor([ks * cin, cout], &packed)?);
        self.cache.borrow_mut().insert(key, tensor.clone());
        Ok(tensor)
    }

    /// The fifteen acoustic codebooks stacked into one table, built once.
    ///
    /// `rvq_rest` stores one tensor per codebook; concatenating them on the host
    /// keeps a decode at one bind group per group instead of fifteen.
    pub fn acoustic_tables(&self) -> Result<(Rc<Tensor>, Rc<Tensor>)> {
        let cfg = &self.cfg;
        let rest = cfg.num_quantizers - 1;
        let dim = cfg.codebook_dim / 2;
        if let (Some(t), Some(u)) = (self.cache.borrow().get("rvq_rest.tables"),
                                     self.cache.borrow().get("rvq_rest.usage")) {
            return Ok((t.clone(), u.clone()));
        }
        let mut tables = vec![0.0f32; rest * cfg.codebook_size * dim];
        let mut usage = vec![0.0f32; rest * cfg.codebook_size];
        for k in 0..rest {
            let table = self.tensor(
                &format!("decoder.quantizer.rvq_rest.vq.layers.{k}._codebook.embedding_sum"),
                [cfg.codebook_size, dim])?;
            let start = k * cfg.codebook_size * dim;
            tables[start..start + cfg.codebook_size * dim].copy_from_slice(&self.dev.read(&table)?);
            let u = self.vector(
                &format!("decoder.quantizer.rvq_rest.vq.layers.{k}._codebook.cluster_usage"),
                cfg.codebook_size)?;
            let start = k * cfg.codebook_size;
            usage[start..start + cfg.codebook_size].copy_from_slice(&self.dev.read(&u)?);
        }
        let tables = Rc::new(self.dev.tensor([rest * cfg.codebook_size, dim], &tables)?);
        let usage = Rc::new(self.dev.tensor([rest * cfg.codebook_size, 1], &usage)?);
        self.cache.borrow_mut().insert("rvq_rest.tables".into(), tables.clone());
        self.cache.borrow_mut().insert("rvq_rest.usage".into(), usage.clone());
        Ok((tables, usage))
    }

    /// `decoder.quantizer.*` dequantise + `pre_conv`, i.e. everything before the
    /// transformer. Returns `[frames, latent_dim]` (time-major), which is the
    /// layout `pre_transformer` consumes.
    pub fn pre_transformer_input(&self, codes: &[u32], frames: usize) -> Result<Tensor> {
        let quantized = quantizer::dequantize(self, codes, frames)?; // [codebook_dim, frames]
        let cfg = &self.cfg;
        let mut projected = self.dev.zeros([cfg.latent_dim, frames])?;
        let weight = self.tensor("decoder.pre_conv.conv.weight", [cfg.latent_dim, cfg.codebook_dim * 3])?;
        let bias = self.vector("decoder.pre_conv.conv.bias", cfg.latent_dim)?;
        let conv = Conv1d::causal(cfg.codebook_dim, cfg.latent_dim, 3, frames);
        ops::conv1d(&self.dev, &quantized, &weight, &bias, &mut projected, &conv)?;
        let mut time_major = self.dev.zeros([frames, cfg.latent_dim])?;
        ops::transpose(&self.dev, &projected, &mut time_major)?;
        Ok(time_major)
    }
}
