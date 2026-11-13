//! Hand-written f32 kernels: the CPU twin of the WGSL pipeline.
//!
//! Plain loops with rayon over the independent axis. No NN framework, no BLAS:
//! the shapes here are small enough that a cache-friendly loop beats the call
//! overhead, and keeping the code explicit makes GPU/CPU parity checkable.

use rayon::prelude::*;

/// `y[m,n] = sum_k x[m,k] * w[n,k] + bias[n]`, weight rows are `[out, in]`.
pub fn linear(x: &[f32], w: &[f32], bias: Option<&[f32]>, y: &mut [f32], (m, n, k): (usize, usize, usize)) {
    // `ops` validates shapes; the assertion documents the contract.
    debug_assert_eq!((x.len(), w.len(), y.len()), (m * k, n * k, m * n));
    let row = |r: usize| -> Vec<f32> {
        let xr = &x[r * k..(r + 1) * k];
        (0..n)
            .map(|c| {
                let wr = &w[c * k..(c + 1) * k];
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += xr[i] * wr[i];
                }
                acc + bias.map(|b| b[c]).unwrap_or(0.0)
            })
            .collect()
    };
    // Rows are independent; the closures allocate one row each, which keeps the
    // inner dot product contiguous in both operands.
    y.par_chunks_mut(n)
        .enumerate()
        .for_each(|(r, out)| out.copy_from_slice(&row(r)));
}

/// Causal conv1d, `[channels, time]`, `y = conv(x) + bias`.
///
/// Cache-blocked: one time block of the input window is staged once and reused by
/// every output channel, and the inner loop walks the staged window contiguously
/// with the weight in a register. The naive per-output form re-read the same
/// window `cout` times and was the whole codec's bottleneck.
#[allow(clippy::too_many_arguments)]
pub fn conv1d(x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
              (cin, cout, ks, dilation, stride, pad, t_in, t_out): (usize, usize, usize, usize, usize, usize, usize, usize)) {
    const BT: usize = 128;
    let span = (ks - 1) * dilation + 1;
    for t0 in (0..t_out).step_by(BT) {
        let tn = BT.min(t_out - t0);
        let start = t0 as isize * stride as isize - pad as isize;
        let len = (tn - 1) * stride + span;
        // Window for this block, zero-padded outside the input.
        let mut window = vec![0.0f32; cin * len];
        window
            .par_chunks_mut(len)
            .enumerate()
            .take(cin)
            .for_each(|(ci, row)| {
                for (j, slot) in row.iter_mut().enumerate() {
                    let src = start + j as isize;
                    if src >= 0 && (src as usize) < t_in {
                        *slot = x[ci * t_in + src as usize];
                    }
                }
            });

        let window = &window;
        y.par_chunks_mut(t_out).enumerate().take(cout).for_each(|(co, out)| {
            let mut acc = vec![0.0f32; tn];
            for ci in 0..cin {
                let base = ci * len;
                let wrow = (co * cin + ci) * ks;
                for k in 0..ks {
                    let wv = w[wrow + k];
                    let off = k * dilation;
                    for (j, slot) in acc.iter_mut().enumerate() {
                        *slot += window[base + j * stride + off] * wv;
                    }
                }
            }
            for (j, slot) in acc.iter().enumerate() {
                out[t0 + j] = slot + bias[co];
            }
        });
    }
}

/// SnakeBeta in place: `x + sin(exp(a) x)^2 / (exp(b) + 1e-9)`.
pub fn snake_beta(x: &mut [f32], alpha: &[f32], beta: &[f32], channels: usize, time: usize) {
    if beta.len() < channels || alpha.len() < channels {
        return;
    }
    x.par_chunks_mut(time).enumerate().take(channels).for_each(|(c, row)| {
        let a = alpha[c].exp();
        let b = beta[c].exp() + 1e-9;
        for v in row.iter_mut() {
            let s = (*v * a).sin();
            *v += s * s / b;
        }
    });
}

/// `y = a + b`.
pub fn add(a: &[f32], b: &[f32], y: &mut [f32]) {
    y.par_iter_mut().zip(a.par_iter().zip(b.par_iter())).for_each(|(o, (x, z))| *o = x + z);
}

/// `y += x` (residual connections in the codec).
pub fn residual(y: &mut [f32], x: &[f32]) {
    y.par_iter_mut().zip(x.par_iter()).for_each(|(o, v)| *o += v);
}

/// RVQ dequantise-and-sum: `out[d,t] = sum_k table[k][code[t,k]][d] / clamp(usage, eps)`.
///
/// Tables are addressed *relative to the group* (`cb - k_start`), because
/// `rvq_rest` ships one table per codebook and we stack only its own fifteen.
#[allow(clippy::too_many_arguments)]
pub fn gather_sum(codes: &[u32], tables: &[f32], usage: &[f32], out: &mut [f32],
                  (dim, frames, k_start, k_count, codebook_size, num_codebooks):
                      (usize, usize, usize, usize, usize, usize)) {
    // Output is `[dim, frames]`, so the linear index is `d * frames + t`.
    out.par_iter_mut().enumerate().for_each(|(idx, slot_out)| {
        let d = idx / frames;
        let t = idx % frames;
        let mut acc = 0.0f32;
        for k in 0..k_count {
            let cb = k_start + k;
            let code = codes[t * num_codebooks + cb] as usize;
            let slot = k * codebook_size + code;
            acc += tables[slot * dim + d] / usage[slot].max(1e-5);
        }
        *slot_out = acc;
    });
}

/// `[rows, cols] -> [cols, rows]`.
pub fn transpose(x: &[f32], y: &mut [f32], rows: usize, cols: usize) {
    y.par_chunks_mut(rows).enumerate().for_each(|(c, out)| {
        for (r, slot) in out.iter_mut().enumerate() {
            *slot = x[r * cols + c];
        }
    });
}

/// Row-wise RMSNorm.
pub fn rms_norm(x: &[f32], w: &[f32], y: &mut [f32], rows: usize, cols: usize, eps: f32) {
    y.par_chunks_mut(cols).enumerate().for_each(|(r, out)| {
        let row = &x[r * cols..(r + 1) * cols];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for c in 0..cols {
            out[c] = row[c] * inv * w[c];
        }
    });
}

/// Row-wise LayerNorm with affine parameters.
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], y: &mut [f32], rows: usize, cols: usize, eps: f32) {
    y.par_chunks_mut(cols).enumerate().for_each(|(r, out)| {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            out[c] = (row[c] - mean) * inv * w[c] + b[c];
        }
    });
}

/// `y = silu(gate) * up`.
pub fn silu_mul(gate: &[f32], up: &[f32], y: &mut [f32]) {
    y.par_iter_mut().zip(gate.par_iter().zip(up.par_iter())).for_each(|(o, (g, u))| {
        *o = (g / (1.0 + (-g).exp())) * u;
    });
}

/// In-place per-channel scale on `[rows, cols]`.
pub fn scale_cols(x: &mut [f32], scale: &[f32], cols: usize) {
    x.par_chunks_mut(cols).for_each(|row| {
        for (c, v) in row.iter_mut().enumerate() {
            *v *= scale[c];
        }
    });
}

/// `y = a + scale[c] * b`.
pub fn add_scaled(a: &[f32], b: &[f32], scale: &[f32], y: &mut [f32], cols: usize) {
    y.par_chunks_mut(cols).enumerate().for_each(|(r, out)| {
        let a = &a[r * cols..(r + 1) * cols];
        let b = &b[r * cols..(r + 1) * cols];
        for c in 0..cols {
            out[c] = a[c] + scale[c] * b[c];
        }
    });
}

/// Rotary position embedding, rotate-half form.
pub fn rope(x: &mut [f32], frames: usize, heads: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;
    let stride = heads * head_dim;
    x.par_chunks_mut(stride).enumerate().take(frames).for_each(|(t, row)| {
        for h in 0..heads {
            let base = h * head_dim;
            for j in 0..half {
                let freq = theta.powf(-2.0 * j as f32 / head_dim as f32);
                let angle = t as f32 * freq;
                let (s, c) = angle.sin_cos();
                let a = row[base + j];
                let b = row[base + j + half];
                row[base + j] = a * c - b * s;
                row[base + j + half] = b * c + a * s;
            }
        }
    });
}

/// Causal sliding-window attention over `[T, heads*head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn attention(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32],
                 frames: usize, heads: usize, head_dim: usize, window: usize) {
    let stride = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    out.par_chunks_mut(stride).enumerate().take(frames).for_each(|(t, row)| {
        let start = if t + 1 > window { t + 1 - window } else { 0 };
        let span = t + 1 - start;
        let mut scores = vec![0.0f32; span];
        for (h, head) in (0..heads).enumerate() {
            let qo = head * head_dim;
            let mut maxs = f32::NEG_INFINITY;
            for (j, kpos) in (start..=t).enumerate() {
                let ko = kpos * stride + qo;
                let mut acc = 0.0f32;
                for i in 0..head_dim {
                    acc += q[t * stride + qo + i] * k[ko + i];
                }
                scores[j] = acc * scale;
                maxs = maxs.max(scores[j]);
            }
            let sum: f32 = scores.iter_mut().map(|s| { *s = (*s - maxs).exp(); *s }).sum();
            let inv = 1.0 / sum;
            for i in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, kpos) in (start..=t).enumerate() {
                    acc += scores[j] * inv * v[kpos * stride + qo + i];
                }
                row[h * head_dim + i] = acc;
            }
        }
    });
}

/// Depthwise causal conv1d (`groups == channels`).
pub fn conv1d_depthwise(x: &[f32], w: &[f32], b: &[f32], y: &mut [f32],
                       channels: usize, ks: usize, pad: usize, t_in: usize, t_out: usize) {
    y.par_chunks_mut(t_out).enumerate().take(channels).for_each(|(c, out)| {
        for (t, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..ks {
                let src = t as isize + k as isize - pad as isize;
                if src >= 0 && (src as usize) < t_in {
                    acc += x[c * t_in + src as usize] * w[c * ks + k];
                }
            }
            *slot = acc + b[c];
        }
    });
}

/// Transposed conv1d with `[in, out, kernel]` weights, trimmed to `t_in * stride`.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(x: &[f32], w: &[f32], b: &[f32], y: &mut [f32],
                        cin: usize, cout: usize, ks: usize, stride: usize, t_in: usize) {
    let t_out = t_in * stride;
    y.par_chunks_mut(t_out).enumerate().take(cout).for_each(|(co, out)| {
        for (j, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..ks {
                let num = j as isize - k as isize;
                if num < 0 || num as usize % stride != 0 {
                    continue;
                }
                let t = num as usize / stride;
                if t >= t_in {
                    continue;
                }
                for ci in 0..cin {
                    acc += x[ci * t_in + t] * w[(ci * cout + co) * ks + k];
                }
            }
            *slot = acc + b[co];
        }
    });
}

/// Exact GELU: `0.5 x (1 + erf(x / sqrt 2))`.
pub fn gelu(x: &mut [f32]) {
    x.par_iter_mut().for_each(|v| {
        *v = 0.5 * *v * (1.0 + erf(*v * std::f32::consts::FRAC_1_SQRT_2));
    });
}

/// Abramowitz & Stegun 7.1.26, max error ~1.5e-7 (matches the WGSL version).
fn erf(v: f32) -> f32 {
    let sign = if v >= 0.0 { 1.0 } else { -1.0 };
    let ax = v.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

/// Copy `rows` rows of `src` into `dst` at row `offset`.
pub fn write_rows(dst: &mut [f32], src: &[f32], cols: usize, offset: usize) {
    let start = offset * cols;
    dst[start..start + src.len()].copy_from_slice(src);
}

/// Single-query attention over a KV cache of `len` rows.
pub fn attention_decode(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32],
                        len: usize, heads: usize, head_dim: usize) {
    let stride = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    out.par_chunks_mut(head_dim).enumerate().take(heads).for_each(|(head, dst)| {
        let qo = head * head_dim;
        let mut scores = vec![0.0f32; len];
        let mut maxs = f32::NEG_INFINITY;
        for (j, score) in scores.iter_mut().enumerate() {
            let ko = j * stride + qo;
            let mut acc = 0.0f32;
            for i in 0..head_dim {
                acc += q[qo + i] * k[ko + i];
            }
            *score = acc * scale;
            maxs = maxs.max(*score);
        }
        let mut sum = 0.0f32;
        for score in scores.iter_mut() {
            *score = (*score - maxs).exp();
            sum += *score;
        }
        let inv = 1.0 / sum;
        for i in 0..head_dim {
            let mut acc = 0.0f32;
            for (j, score) in scores.iter().enumerate() {
                acc += score * v[j * stride + qo + i];
            }
            dst[i] = acc * inv;
        }
    });
}

/// Rotary embedding from a cos/sin table, in place.
pub fn rope_tabled(x: &mut [f32], cos: &[f32], sin: &[f32], frames: usize, heads: usize, head_dim: usize) {
    let half = head_dim / 2;
    let stride = heads * head_dim;
    x.par_chunks_mut(stride).enumerate().take(frames).for_each(|(t, row)| {
        for h in 0..heads {
            let base = h * head_dim;
            for j in 0..half {
                let c = cos[t * head_dim + j];
                let s = sin[t * head_dim + j];
                let a = row[base + j];
                let b = row[base + j + half];
                row[base + j] = a * c - b * s;
                row[base + j + half] = b * c + a * s;
            }
        }
    });
}

/// Grouped-query KV repeat.
pub fn repeat_kv(x: &[f32], y: &mut [f32], frames: usize, kv_heads: usize, heads: usize, head_dim: usize) {
    let rep = heads / kv_heads;
    let src_stride = kv_heads * head_dim;
    let dst_stride = heads * head_dim;
    y.par_chunks_mut(dst_stride).enumerate().take(frames).for_each(|(t, row)| {
        for h in 0..heads {
            let src = &x[t * src_stride + (h / rep) * head_dim..][..head_dim];
            row[h * head_dim..(h + 1) * head_dim].copy_from_slice(src);
        }
    });
}
