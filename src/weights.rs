//! safetensors loading with the checkpoint's own dtype preserved.
//!
//! The VoiceClone bundle stores bf16, CustomVoice and the codec store f32. bf16
//! widens to f32 losslessly (a 16-bit shift), so the CPU path is fp32-exact and
//! the GPU path can keep the packed form for half the bus traffic.

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Bf16,
    I64,
    Other,
}

impl DType {
    fn from_safetensors(dtype: safetensors::tensor::Dtype) -> Self {
        use safetensors::tensor::Dtype as S;
        match dtype {
            S::F32 => DType::F32,
            S::F16 => DType::F16,
            S::BF16 => DType::Bf16,
            S::I64 => DType::I64,
            _ => DType::Other,
        }
    }
}

pub struct TensorEntry {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Vec<u8>,
}

impl TensorEntry {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Widen to f32. bf16 -> f32 is exact; f16 -> f32 is exact as well.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        match self.dtype {
            // The checkpoint's f32 bytes are already the in-memory image, so
            // this is a memcpy: the per-element `from_le_bytes` it used to do
            // ran at a fraction of that, and the codec's transposed-conv
            // weights alone are 75 MB.
            DType::F32 => {
                let len = self.data.len() / 4;
                let mut out = vec![0.0f32; len];
                bytemuck::cast_slice_mut::<f32, u8>(&mut out)
                    .copy_from_slice(&self.data[..len * 4]);
                Ok(out)
            }
            DType::Bf16 => Ok(self
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits(((c[1] as u32) << 24) | ((c[0] as u32) << 16)))
                .collect()),
            DType::F16 => Ok(self
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect()),
            other => bail!("unsupported dtype {other:?} for f32 conversion"),
        }
    }

    pub fn to_i64(&self) -> Result<Vec<i64>> {
        if self.dtype != DType::I64 {
            bail!("expected i64 tensor, got {:?}", self.dtype);
        }
        Ok(self
            .data
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    pub fn require_shape(&self, expected: &[usize], name: &str) -> Result<()> {
        if self.shape != expected {
            bail!("{name}: expected shape {expected:?}, got {:?}", self.shape);
        }
        Ok(())
    }
}

pub struct Weights {
    map: HashMap<String, TensorEntry>,
}

impl Weights {
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let st = SafeTensors::deserialize(&mmap).context("parse safetensors")?;
        // One contiguous copy per tensor, spread over the thread pool: the codec
        // checkpoint is 682 MB and a serial copy of it ran at ~1.5 GB/s (0.45 s
        // of process start-up, which the CLI pays before it says anything).
        let entries: Vec<(String, Vec<usize>, DType, &[u8])> = st
            .iter()
            .map(|(name, view)| {
                (name.to_string(), view.shape().to_vec(),
                 DType::from_safetensors(view.dtype()), view.data())
            })
            .collect();
        let map: HashMap<String, TensorEntry> = entries
            .into_par_iter()
            .map(|(name, shape, dtype, bytes)| {
                (name, TensorEntry { shape, dtype, data: bytes.to_vec() })
            })
            .collect();
        Ok(Self { map })
    }

    pub fn get(&self, name: &str) -> Result<&TensorEntry> {
        self.map
            .get(name)
            .with_context(|| format!("missing tensor {name}"))
    }

    pub fn get_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.get(name)?.to_f32()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(String::as_str)
    }
}



// ---------------------------------------------------------------------------
// Bundle contract
// ---------------------------------------------------------------------------

/// One expected tensor: name plus shape. Kept as data so the error message can
/// say exactly what is wrong instead of failing later inside a kernel.
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

fn llama_specs(prefix: &str, layers: usize, hidden: usize, kv: usize, inter: usize,
               head_dim: usize) -> Vec<TensorSpec> {
    let mut out = vec![TensorSpec {
        name: format!("{prefix}.norm.weight"),
        shape: vec![hidden],
    }];
    for i in 0..layers {
        let p = format!("{prefix}.layers.{i}");
        let specs = [
            (format!("{p}.input_layernorm.weight"), vec![hidden]),
            (format!("{p}.post_attention_layernorm.weight"), vec![hidden]),
            (format!("{p}.self_attn.q_proj.weight"), vec![hidden, hidden]),
            (format!("{p}.self_attn.k_proj.weight"), vec![kv * head_dim, hidden]),
            (format!("{p}.self_attn.v_proj.weight"), vec![kv * head_dim, hidden]),
            (format!("{p}.self_attn.o_proj.weight"), vec![hidden, hidden]),
            (format!("{p}.mlp.gate_proj.weight"), vec![inter, hidden]),
            (format!("{p}.mlp.up_proj.weight"), vec![inter, hidden]),
            (format!("{p}.mlp.down_proj.weight"), vec![hidden, inter]),
        ];
        out.extend(specs.into_iter().map(|(name, shape)| TensorSpec { name, shape }));
    }
    out
}

/// Shapes of the LM half of an M5 bundle (`config.json` + `model.safetensors`).
/// llama-30M backbone = 8 layers / 8 heads / 2 kv / 512 hidden / 2048 inter;
/// llama-20M depth decoder = 5 layers, same widths.
pub fn lm_specs(text_vocab: usize, audio_vocab: usize, codebooks: usize) -> Vec<TensorSpec> {
    let mut specs = vec![
        TensorSpec { name: "text_embeddings.weight".into(), shape: vec![text_vocab, 512] },
        TensorSpec { name: "audio_embeddings.weight".into(), shape: vec![audio_vocab * codebooks, 512] },
        TensorSpec { name: "codebook0_head.weight".into(), shape: vec![audio_vocab, 512] },
        TensorSpec { name: "projection.weight".into(), shape: vec![512, 512] },
        TensorSpec { name: "audio_head".into(), shape: vec![codebooks - 1, 512, audio_vocab] },
    ];
    specs.extend(llama_specs("backbone", 8, 512, 2, 2048, 64));
    specs.extend(llama_specs("decoder", 5, 512, 2, 2048, 64));
    specs
}

/// Verify every spec against the loaded tensors; returns the parameter count.
pub fn validate(w: &Weights, specs: &[TensorSpec]) -> Result<u64> {
    let mut total = 0u64;
    let mut errors = Vec::new();
    for spec in specs {
        match w.map.get(&spec.name) {
            None => errors.push(format!("missing {}", spec.name)),
            Some(entry) => {
                if entry.shape != spec.shape {
                    errors.push(format!(
                        "{}.shape {:?} != {:?}",
                        spec.name, entry.shape, spec.shape
                    ));
                }
                total += entry.numel() as u64;
            }
        }
    }
    if !errors.is_empty() {
        bail!("bundle contract failed:\n  {}", errors.join("\n  "));
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_widens_losslessly() {
        // 0x3f80 bf16 == 1.0f32
        let entry = TensorEntry {
            shape: vec![2],
            dtype: DType::Bf16,
            data: vec![0x80, 0x3f, 0x00, 0xc0],
        };
        assert_eq!(entry.to_f32().unwrap(), vec![1.0, -2.0]);
    }
}
