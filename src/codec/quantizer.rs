//! Split residual vector quantiser decode (`SplitResidualVectorQuantizer`).
//!
//! Codebook 0 goes through `rvq_first`, codebooks 1..15 through `rvq_rest`; each
//! group has its own 256 -> 512 projection and the two are summed.
//! `embedding_sum / clamp(cluster_usage, 1e-5)` is the reference's formula,
//! evaluated at decode time exactly like it does.

use anyhow::Result;

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
