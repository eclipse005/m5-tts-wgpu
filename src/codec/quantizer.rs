//! Split residual vector quantiser decode (`SplitResidualVectorQuantizer`).
//!
//! Codebook 0 goes through `rvq_first`, codebooks 1..15 through `rvq_rest`; each
//! group has its own 256 -> 512 projection and the two are summed.
//! `embedding_sum / clamp(cluster_usage, 1e-5)` is the reference's formula,
//! evaluated at decode time exactly like it does.

use anyhow::Result;
use rayon::prelude::*;

use super::Codec;
use crate::device::{Device, Tensor};
use crate::ops;

/// Projection of one quantiser group, `[frames, codebook_dim]`.
#[allow(clippy::too_many_arguments)]
fn project(dev: &Device, codec: &Codec, group: &str, tables: &Tensor, usage: &Tensor,
           codes: &[u32], frames: usize, k_start: usize, k_count: usize) -> Result<Tensor> {
    let dim = codec.cfg.codebook_dim / 2;
    let mut gathered = dev.zeros([dim, frames])?;
    ops::gather_sum(dev, codes, tables, usage, &mut gathered, dim, frames, k_start, k_count,
                    codec.cfg.codebook_size, codec.cfg.num_quantizers)?;

    // `output_proj` is a 1x1 conv without bias: a channel mix, not a time op.
    let weight = codec.tensor(&format!("decoder.quantizer.{group}.output_proj.weight"),
                              [codec.cfg.codebook_dim, dim])?;
    let mut time_major = dev.zeros([frames, dim])?;
    ops::transpose(dev, &gathered, &mut time_major)?;
    let mut projected = dev.zeros([frames, codec.cfg.codebook_dim])?;
    ops::linear(dev, &time_major, &weight, None, &mut projected)?;
    Ok(projected)
}

impl Codec {
    /// `SplitResidualVectorQuantizer.decode`, `[codebook_dim, frames]`.
    pub fn dequantize(&self, codes: &[u32], frames: usize) -> Result<Tensor> {
        dequantize(self, codes, frames)
    }
}

/// `SplitResidualVectorQuantizer.decode`, `[codebook_dim, frames]`.
///
/// `codes` is `[frames, num_codebooks]`, frame-major, codebook 0 first.
pub fn dequantize(codec: &Codec, codes: &[u32], frames: usize) -> Result<Tensor> {
    let dev = codec.device();
    let cfg = &codec.cfg;

    let semantic_tables = codec.tensor(
        "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        [cfg.codebook_size, cfg.codebook_dim / 2],
    )?;
    let semantic_usage =
        codec.vector("decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage", cfg.codebook_size)?;
    let semantic = project(dev, codec, "rvq_first", &semantic_tables, &semantic_usage, codes, frames, 0, 1)?;

    let (acoustic_tables, acoustic_usage) = codec.acoustic_tables()?;
    let acoustic = project(dev, codec, "rvq_rest", &acoustic_tables, &acoustic_usage, codes, frames,
                           1, cfg.num_quantizers - 1)?;

    let mut sum = dev.zeros([frames, cfg.codebook_dim])?;
    ops::add(dev, &semantic, &acoustic, &mut sum)?;
    let mut channels = dev.zeros([cfg.codebook_dim, frames])?;
    ops::transpose(dev, &sum, &mut channels)?;
    Ok(channels)
}

impl Codec {
    /// `SplitResidualVectorQuantizer.encode(x, 16)`: 1 semantic + 15 acoustic
    /// codebooks, frame-major (`[frames, num_codebooks]`, codebook 0 first).
    ///
    /// Both branches see the *same* latent (each with its own 512 -> 256
    /// projection); inside a branch the codebooks are greedy residual, exactly like
    /// `MimiResidualVectorQuantizer.encode`. Quantisation itself runs on the host:
    /// 43 frames x 16 codebooks x 2048 entries is ~10 ms with rayon, and it avoids
    /// a gather-heavy kernel for a stage that runs once per utterance.
    pub fn encode_codes(&self, latent: &Tensor) -> Result<Vec<u32>> {
        let [channels, frames] = latent.shape();
        if channels != self.cfg.codebook_dim {
            anyhow::bail!("encode_codes expects {}-channel latents, got {channels}",
                          self.cfg.codebook_dim);
        }
        let host = self.device().read(latent)?;
        let dim = self.cfg.codebook_dim / 2;

        let semantic = self.rvq_encode("semantic", &host, channels, frames, dim, 1)?;
        let acoustic = self.rvq_encode("acoustic", &host, channels, frames, dim,
                                       self.cfg.num_quantizers - 1)?;

        // Frame-major output: [frames, 16].
        let mut codes = vec![0u32; frames * self.cfg.num_quantizers];
        for frame in 0..frames {
            codes[frame * self.cfg.num_quantizers] = semantic[frame][0];
            for k in 0..self.cfg.num_quantizers - 1 {
                codes[frame * self.cfg.num_quantizers + 1 + k] = acoustic[frame][k];
            }
        }
        Ok(codes)
    }

    /// One branch of the split quantiser: project, then greedy residual coding.
    fn rvq_encode(&self, branch: &str, latent: &[f32], channels: usize, frames: usize,
                  dim: usize, layers: usize) -> Result<Vec<Vec<u32>>> {
        let prefix = format!("encoder.quantizer.{branch}_residual_vector_quantizer");
        let proj = self.weights.get_f32(&format!("{prefix}.input_proj.weight"))?;
        // Conv1d(512 -> 256, k=1, no bias): a channel mix.
        let mut residual = vec![0.0f32; dim * frames];
        for (out_channel, row) in proj.chunks_exact(channels).enumerate().take(dim) {
            for frame in 0..frames {
                let mut acc = 0.0f32;
                for c in 0..channels {
                    acc += row[c] * latent[c * frames + frame];
                }
                residual[out_channel * frames + frame] = acc;
            }
        }

        let mut codes = vec![vec![0u32; layers]; frames];
        for layer in 0..layers {
            let table = self.codebook_table(&format!("{prefix}.layers.{layer}.codebook"), dim)?;
            let mut indices = vec![0u32; frames];
            indices.par_iter_mut().enumerate().for_each(|(frame, slot)| {
                let vector: Vec<f32> = (0..dim).map(|d| residual[d * frames + frame]).collect();
                *slot = nearest(&table, &vector) as u32;
            });
            for frame in 0..frames {
                codes[frame][layer] = indices[frame];
                let slot = indices[frame] as usize;
                for d in 0..dim {
                    residual[d * frames + frame] -= table[slot * dim + d];
                }
            }
        }
        Ok(codes)
    }

    /// `embed_sum / clamp(cluster_usage, 1e-5)`, one branch's codebook table.
    fn codebook_table(&self, prefix: &str, dim: usize) -> Result<Vec<f32>> {
        let sums = self.weights.get_f32(&format!("{prefix}.embed_sum"))?;
        let usage = self.weights.get_f32(&format!("{prefix}.cluster_usage"))?;
        let mut table = sums;
        for (index, value) in table.iter_mut().enumerate() {
            let entry = index / dim;
            *value /= usage[entry].max(1e-5);
        }
        Ok(table)
    }
}

/// Nearest entry by squared L2 distance.
fn nearest(table: &[f32], vector: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_distance = f32::INFINITY;
    for (index, entry) in table.chunks_exact(vector.len()).enumerate() {
        let mut distance = 0.0f32;
        for (a, b) in entry.iter().zip(vector) {
            let d = a - b;
            distance += d * d;
        }
        if distance < best_distance {
            best_distance = distance;
            best = index;
        }
    }
    best
}
