//! `pre_transformer`: the eight-layer causal transformer between the quantiser
//! and the SEANet decoder.
//!
//! Mirrors `Qwen3TTSTokenizerV2DecoderTransformerModel.forward`: input_proj,
//! per-layer RMSNorm -> sliding-window attention -> layer scale -> RMSNorm ->
//! SwiGLU -> layer scale, then a final RMSNorm and output_proj. Positions are
//! `0..T`, RoPE is the rotate-half form with theta from the config.

use anyhow::Result;

use super::Codec;
use crate::device::{Device, Tensor};
use crate::ops;

/// One transformer layer's weights, resolved once per decode.
struct LayerWeights {
    input_norm: Tensor,
    post_norm: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    attn_scale: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
    mlp_scale: Tensor,
}

fn layer_weights(codec: &Codec, index: usize) -> Result<LayerWeights> {
    let p = format!("decoder.pre_transformer.layers.{index}");
    let cfg = &codec.cfg;
    let h = cfg.hidden_size;
    let i = cfg.intermediate_size;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_attention_heads * cfg.head_dim;
    Ok(LayerWeights {
        input_norm: codec.vector(&format!("{p}.input_layernorm.weight"), h)?,
        post_norm: codec.vector(&format!("{p}.post_attention_layernorm.weight"), h)?,
        q: codec.tensor(&format!("{p}.self_attn.q_proj.weight"), [q_dim, h])?,
        k: codec.tensor(&format!("{p}.self_attn.k_proj.weight"), [kv_dim, h])?,
        v: codec.tensor(&format!("{p}.self_attn.v_proj.weight"), [kv_dim, h])?,
        o: codec.tensor(&format!("{p}.self_attn.o_proj.weight"), [h, q_dim])?,
        attn_scale: codec.vector(&format!("{p}.self_attn_layer_scale.scale"), h)?,
        gate: codec.tensor(&format!("{p}.mlp.gate_proj.weight"), [i, h])?,
        up: codec.tensor(&format!("{p}.mlp.up_proj.weight"), [i, h])?,
        down: codec.tensor(&format!("{p}.mlp.down_proj.weight"), [h, i])?,
        mlp_scale: codec.vector(&format!("{p}.mlp_layer_scale.scale"), h)?,
    })
}

fn layer(codec: &Codec, w: &LayerWeights, x: &Tensor, frames: usize) -> Result<Tensor> {
    let dev: &Device = codec.device();
    let cfg = &codec.cfg;
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;

    let mut normed = dev.zeros([frames, hidden])?;
    ops::rms_norm(dev, x, &w.input_norm, &mut normed, cfg.rms_norm_eps)?;

    let mut q = dev.zeros([frames, q_dim])?;
    let mut k = dev.zeros([frames, q_dim])?;
    let mut v = dev.zeros([frames, q_dim])?;
    ops::linear(dev, &normed, &w.q, None, &mut q)?;
    ops::linear(dev, &normed, &w.k, None, &mut k)?;
    ops::linear(dev, &normed, &w.v, None, &mut v)?;
    ops::rope(dev, &mut q, cfg.num_attention_heads, cfg.head_dim, cfg.rope_theta)?;
    ops::rope(dev, &mut k, cfg.num_attention_heads, cfg.head_dim, cfg.rope_theta)?;

    let mut attended = dev.zeros([frames, q_dim])?;
    ops::attention(dev, &q, &k, &v, &mut attended, cfg.num_attention_heads, cfg.head_dim,
                   cfg.sliding_window)?;
    let mut projected = dev.zeros([frames, hidden])?;
    ops::linear(dev, &attended, &w.o, None, &mut projected)?;
    let mut x1 = dev.zeros([frames, hidden])?;
    ops::add_scaled(dev, x, &projected, &w.attn_scale, &mut x1)?;

    let mut normed2 = dev.zeros([frames, hidden])?;
    ops::rms_norm(dev, &x1, &w.post_norm, &mut normed2, cfg.rms_norm_eps)?;
    let mut gate = dev.zeros([frames, inter])?;
    let mut up = dev.zeros([frames, inter])?;
    ops::linear(dev, &normed2, &w.gate, None, &mut gate)?;
    ops::linear(dev, &normed2, &w.up, None, &mut up)?;
    let mut act = dev.zeros([frames, inter])?;
    ops::silu_mul(dev, &gate, &up, &mut act)?;
    let mut mlp = dev.zeros([frames, hidden])?;
    ops::linear(dev, &act, &w.down, None, &mut mlp)?;
    let mut out = dev.zeros([frames, hidden])?;
    ops::add_scaled(dev, &x1, &mlp, &w.mlp_scale, &mut out)?;
    Ok(out)
}

impl Codec {
    /// `pre_transformer(inputs_embeds=…).last_hidden_state`, `[frames, latent_dim]`.
    pub fn transformer(&self, x: &Tensor) -> Result<Tensor> {
        let dev = self.device();
        let cfg = &self.cfg;
        let [frames, latent] = x.shape();

        let in_weight = self.tensor("decoder.pre_transformer.input_proj.weight", [cfg.hidden_size, latent])?;
        let in_bias = self.vector("decoder.pre_transformer.input_proj.bias", cfg.hidden_size)?;
        let mut hidden = dev.zeros([frames, cfg.hidden_size])?;
        ops::linear(dev, x, &in_weight, Some(&in_bias), &mut hidden)?;

        for index in 0..cfg.num_hidden_layers {
            let weights = layer_weights(self, index)?;
            hidden = layer(self, &weights, &hidden, frames)?;
        }

        let norm = self.vector("decoder.pre_transformer.norm.weight", cfg.hidden_size)?;
        let mut normed = dev.zeros([frames, cfg.hidden_size])?;
        ops::rms_norm(dev, &hidden, &norm, &mut normed, cfg.rms_norm_eps)?;
        let out_weight = self.tensor("decoder.pre_transformer.output_proj.weight", [latent, cfg.hidden_size])?;
        let out_bias = self.vector("decoder.pre_transformer.output_proj.bias", latent)?;
        let mut out = dev.zeros([frames, latent])?;
        ops::linear(dev, &normed, &out_weight, Some(&out_bias), &mut out)?;
        Ok(out)
    }
}
