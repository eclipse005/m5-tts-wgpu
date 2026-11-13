//! Encoder half of the codec: reference audio -> codes.
//!
//! Mirrors `MimiEncoder` as the checkpoint builds it:
//!
//! ```text
//! conv(1->64, k7) -> [resnet(64) elu conv(64->128,k8,s4)] x4 ... -> conv(1024->512,k3)
//!   -> encoder_transformer (8 layers, LayerNorm+affine, GELU MLP, layer scale,
//!      sliding window 250, RoPE theta 1e4)
//!   -> downsample conv(512->512, k4, s2)
//!   -> SplitResidualVectorQuantizer.encode(16)
//! ```
//!
//! Causal strided convs use `pad = kernel - stride` and `out = ceil(in / stride)`,
//! which is what makes the golden 82 211 samples land on 86 frames (82211 / 960).

use anyhow::Result;

use super::Codec;
use crate::device::Tensor;
use crate::ops::{self, Conv1d};

/// `MimiConv1d` geometry for a causal strided convolution.
fn causal(ks: usize, stride: usize, t_in: usize) -> Conv1d {
    Conv1d {
        cin: 0, // filled by the caller
        cout: 0,
        ks,
        dilation: 1,
        stride,
        pad: ks - stride,
        t_in,
        t_out: t_in.div_ceil(stride),
    }
}

impl Codec {
    /// One causal conv of the encoder stack (`bias` present on every layer).
    fn enc_conv(&self, x: &Tensor, weight_name: &str, bias_name: &str, cout: usize,
                ks: usize, stride: usize) -> Result<Tensor> {
        let dev = self.device();
        let [cin, t_in] = x.shape();
        let weight = self.tensor(weight_name, [cout, cin * ks])?;
        let bias = self.vector(bias_name, cout)?;
        let mut geometry = causal(ks, stride, t_in);
        geometry.cin = cin;
        geometry.cout = cout;
        let mut out = dev.zeros([cout, geometry.t_out])?;
        ops::conv1d(dev, x, &weight, &bias, &mut out, &geometry)?;
        Ok(out)
    }

    /// `MimiResnetBlock`: elu -> conv(k3) -> elu -> conv(k1) + residual.
    fn enc_resnet(&self, x: &Tensor, prefix: &str) -> Result<Tensor> {
        let dev = self.device();
        let channels = x.shape()[0];
        let mut h = dev.copy(x)?;
        ops::elu(dev, &mut h)?;
        let h = self.enc_conv(&h, &format!("{prefix}.block.1.conv.weight"),
                              &format!("{prefix}.block.1.conv.bias"), channels / 2, 3, 1)?;
        let mut h = h;
        ops::elu(dev, &mut h)?;
        let out = self.enc_conv(&h, &format!("{prefix}.block.3.conv.weight"),
                                &format!("{prefix}.block.3.conv.bias"), channels, 1, 1)?;
        let mut sum = dev.zeros(x.shape())?;
        ops::add(dev, &out, x, &mut sum)?;
        Ok(sum)
    }

    /// SEANet encoder: `[channels=1, samples]` -> `[512, samples / 960]`.
    pub fn seanet_encode(&self, audio: &Tensor) -> Result<Tensor> {
        let dev = self.device();
        if audio.shape()[0] != 1 {
            anyhow::bail!("seanet_encode expects mono audio, got {} channels", audio.shape()[0]);
        }
        let mut h = self.enc_conv(audio, "encoder.encoder.layers.0.conv.weight",
                                  "encoder.encoder.layers.0.conv.bias", 64, 7, 1)?;
        // Stage pattern taken from the module list: resnet, elu, strided conv.
        for (resnet, conv, stride, cout) in
            [(1usize, 3usize, 4usize, 128usize), (4, 6, 5, 256), (7, 9, 6, 512), (10, 12, 8, 1024)]
        {
            h = self.enc_resnet(&h, &format!("encoder.encoder.layers.{resnet}"))?;
            ops::elu(dev, &mut h)?;
            h = self.enc_conv(&h, &format!("encoder.encoder.layers.{conv}.conv.weight"),
                              &format!("encoder.encoder.layers.{conv}.conv.bias"), cout, 2 * stride, stride)?;
        }
        // Trailing ELU then the last 1024 -> 512 conv.
        ops::elu(dev, &mut h)?;
        self.enc_conv(&h, "encoder.encoder.layers.14.conv.weight",
                      "encoder.encoder.layers.14.conv.bias", 512, 3, 1)
    }
}

impl Codec {
    /// Encoder transformer: 8 layers of pre-norm attention + GELU MLP, each with
    /// a learned layer scale. `LayerNorm` here has affine bias (unlike the
    /// decoder's RMSNorm), and the attention window is 250.
    pub fn encoder_transformer(&self, x: &Tensor) -> Result<Tensor> {
        let dev = self.device();
        let enc = &self.enc;
        let [frames, hidden] = x.shape();
        let inter = enc.intermediate_size;
        let q_dim = enc.num_attention_heads * enc.head_dim;
        let mut h = dev.copy(x)?;

        for index in 0..enc.num_hidden_layers {
            let p = format!("encoder.encoder_transformer.layers.{index}");
            let ln_w = self.vector(&format!("{p}.input_layernorm.weight"), hidden)?;
            let ln_b = self.vector(&format!("{p}.input_layernorm.bias"), hidden)?;
            let mut normed = dev.zeros([frames, hidden])?;
            ops::layer_norm(dev, &h, &ln_w, &ln_b, &mut normed, enc.norm_eps)?;

            let q_w = self.tensor(&format!("{p}.self_attn.q_proj.weight"), [q_dim, hidden])?;
            let k_w = self.tensor(&format!("{p}.self_attn.k_proj.weight"), [q_dim, hidden])?;
            let v_w = self.tensor(&format!("{p}.self_attn.v_proj.weight"), [q_dim, hidden])?;
            let o_w = self.tensor(&format!("{p}.self_attn.o_proj.weight"), [hidden, q_dim])?;
            let mut q = dev.zeros([frames, q_dim])?;
            let mut k = dev.zeros([frames, q_dim])?;
            let mut v = dev.zeros([frames, q_dim])?;
            ops::linear(dev, &normed, &q_w, None, &mut q)?;
            ops::linear(dev, &normed, &k_w, None, &mut k)?;
            ops::linear(dev, &normed, &v_w, None, &mut v)?;
            let theta = self.encoder_rope_theta();
            ops::rope(dev, &mut q, enc.num_attention_heads, enc.head_dim, theta)?;
            ops::rope(dev, &mut k, enc.num_attention_heads, enc.head_dim, theta)?;

            let mut attended = dev.zeros([frames, q_dim])?;
            ops::attention(dev, &q, &k, &v, &mut attended, enc.num_attention_heads, enc.head_dim,
                           enc.sliding_window)?;
            let mut projected = dev.zeros([frames, hidden])?;
            ops::linear(dev, &attended, &o_w, None, &mut projected)?;
            let attn_scale = self.vector(&format!("{p}.self_attn_layer_scale.scale"), hidden)?;
            let mut x1 = dev.zeros([frames, hidden])?;
            ops::add_scaled(dev, &h, &projected, &attn_scale, &mut x1)?;

            let ln2_w = self.vector(&format!("{p}.post_attention_layernorm.weight"), hidden)?;
            let ln2_b = self.vector(&format!("{p}.post_attention_layernorm.bias"), hidden)?;
            let mut normed2 = dev.zeros([frames, hidden])?;
            ops::layer_norm(dev, &x1, &ln2_w, &ln2_b, &mut normed2, enc.norm_eps)?;
            let fc1 = self.tensor(&format!("{p}.mlp.fc1.weight"), [inter, hidden])?;
            let fc2 = self.tensor(&format!("{p}.mlp.fc2.weight"), [hidden, inter])?;
            let mut wide = dev.zeros([frames, inter])?;
            ops::linear(dev, &normed2, &fc1, None, &mut wide)?;
            ops::gelu(dev, &mut wide)?;
            let mut narrow = dev.zeros([frames, hidden])?;
            ops::linear(dev, &wide, &fc2, None, &mut narrow)?;
            let mlp_scale = self.vector(&format!("{p}.mlp_layer_scale.scale"), hidden)?;
            let mut out = dev.zeros([frames, hidden])?;
            ops::add_scaled(dev, &x1, &narrow, &mlp_scale, &mut out)?;
            h = out;
        }
        Ok(h)
    }

    /// `downsample`: causal conv(512 -> 512, k4, s2), no bias.
    pub fn encoder_downsample(&self, x: &Tensor) -> Result<Tensor> {
        let dev = self.device();
        let [channels, t_in] = x.shape();
        let weight = self.tensor("encoder.downsample.conv.weight", [channels, channels * 4])?;
        let zero = dev.zeros([channels, 1])?;
        let mut geometry = causal(4, 2, t_in);
        geometry.cin = channels;
        geometry.cout = channels;
        let mut out = dev.zeros([channels, geometry.t_out])?;
        ops::conv1d(dev, x, &weight, &zero, &mut out, &geometry)?;
        Ok(out)
    }
}
