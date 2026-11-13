//! Hand-written f32 kernels: the CPU twin of the WGSL pipeline.
//!
//! Plain loops with rayon over the independent axis. No NN framework, no BLAS:
//! the shapes here are small enough that a cache-friendly loop beats the call
//! overhead, and keeping the code explicit makes GPU/CPU parity checkable.

use rayon::prelude::*;

/// Below this many elements a rayon fork costs more than the loop it parallelises
/// (measured: a fork is ~10-20 us, a 2k-element elementwise pass is ~2 us). The
/// LM's per-frame ops sit far below this line, which is why a frame of ~1160 ops
/// spent most of its time on fork-join rather than arithmetic.
const PAR_MIN: usize = 8192;

#[inline]
fn wide_enough(n: usize) -> bool {
    n >= PAR_MIN
}

/// `y[m,n] = sum_k x[m,k] * w[n,k] + bias[n]`, weight rows are `[out, in]`.
pub fn linear(x: &[f32], w: &[f32], bias: Option<&[f32]>, y: &mut [f32], (m, n, k): (usize, usize, usize)) {
    // `ops` validates shapes; the assertion documents the contract.
    debug_assert_eq!((x.len(), w.len(), y.len()), (m * k, n * k, m * n));
    // Decode is m = 1: parallelising over rows would leave one thread doing all
    // the work, so spread that case across the output columns instead. This is the
    // difference between ~0.6 and ~10 GFLOP/s for the per-frame GEMMs.
    if m == 1 {
        let xr = &x[..k];
        let vectorised = avx2_available();
        // A single decode step is 512x512: forking rayon for ~0.3 M MAC costs more
        // than the arithmetic, so small GEMVs run serially.
        let parallel = true; // measured: forking beats serial even at 0.3M MAC
        let body = |(c, slot): (usize, &mut f32)| {
            let wr = &w[c * k..(c + 1) * k];
            let dot = if vectorised {
                // SAFETY: guarded by `avx2_available()`.
                unsafe { dot_avx2(xr, wr, k) }
            } else {
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += xr[i] * wr[i];
                }
                acc
            };
            *slot = dot + bias.map(|b| b[c]).unwrap_or(0.0);
        };
        if parallel {
            y.par_iter_mut().enumerate().with_min_len(16).for_each(body);
        } else {
            y.iter_mut().enumerate().for_each(body);
        }
        return;
    }
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
    // Measured on this machine: a fixed 128-sample block beats cache-sized blocks
    // (212 vs 47 GFLOP/s). Smaller blocks pay the staging cost far more often than
    // they gain from L1 residency.
    let bt: usize = 128;
    let span = (ks - 1) * dilation + 1;
    for t0 in (0..t_out).step_by(bt) {
        let tn = bt.min(t_out - t0);
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
        // Stride-1 convolutions (every SEANet residual/pointwise tap) are vectorised
        // over time: the staged window is contiguous, so the inner loop becomes a
        // straight FMA chain. Strided convs keep the scalar path.
        let vectorised = stride == 1 && avx2_available();
        // Four output channels per pass: one window load then feeds four FMAs, which
        // is what lifts this out of the "re-read the window for every co" regime
        // (~60 GFLOP/s in situ versus 212 standalone).
        const CO: usize = 4;
        if vectorised && cout % CO == 0 {
            let bias = bias;
            y.par_chunks_mut(t_out * CO).enumerate().for_each(|(block, out_block)| {
                let co0 = block * CO;
                let mut acc = vec![0.0f32; CO * tn];
                for ci in 0..cin {
                    let base = ci * len;
                    let rows: [&[f32]; CO] = std::array::from_fn(|r| {
                        let start = (co0 + r) * cin * ks + ci * ks;
                        &w[start..start + ks]
                    });
                    // SAFETY: guarded by `avx2_available()`.
                    unsafe { conv_block4_avx2(&window[base..base + len], &rows, &mut acc, tn, ks, dilation) };
                }
                for r in 0..CO {
                    let lane = &acc[r * tn..(r + 1) * tn];
                    let dst = &mut out_block[r * t_out + t0..r * t_out + t0 + tn];
                    for (slot, value) in dst.iter_mut().zip(lane) {
                        *slot = value + bias[co0 + r];
                    }
                }
            });
            continue;
        }
        y.par_chunks_mut(t_out).enumerate().take(cout).for_each(|(co, out)| {
            let mut acc = vec![0.0f32; tn];
            for ci in 0..cin {
                let base = ci * len;
                let wrow = &w[(co * cin + ci) * ks..(co * cin + ci) * ks + ks];
                if vectorised {
                    // SAFETY: guarded by `avx2_available()` above.
                    unsafe { conv_block_avx2(&window[base..base + len], wrow, &mut acc, tn, ks, dilation) };
                } else {
                    for k in 0..ks {
                        let wv = wrow[k];
                        let off = k * dilation;
                        for (j, slot) in acc.iter_mut().enumerate() {
                            *slot += window[base + j * stride + off] * wv;
                        }
                    }
                }
            }
            for (j, slot) in acc.iter().enumerate() {
                out[t0 + j] = slot + bias[co];
            }
        });
    }
}

/// SnakeBeta out of place: `y = x + sin(exp(a) x)^2 / (exp(b) + 1e-9)`.
///
/// Fused into one pass: the previous version copied the tensor and then ran the
/// activation over it, which on the last SEANet block is a redundant 41 MB copy.
pub fn snake_beta_into(x: &[f32], y: &mut [f32], alpha: &[f32], beta: &[f32],
                       channels: usize, time: usize) {
    if beta.len() < channels || alpha.len() < channels {
        return;
    }
    y.par_chunks_mut(time)
        .zip(x.par_chunks(time))
        .enumerate()
        .take(channels)
        .with_min_len(1)
        .for_each(|(c, (out, src))| {
            let a = alpha[c].exp();
            let b = beta[c].exp() + 1e-9;
            for (slot, v) in out.iter_mut().zip(src) {
                let s = fast_sin(*v * a);
                *slot = v + s * s / b;
            }
        });
}

/// Sine accurate to ~1e-7 over the range the codec reaches, without libm's call.
#[inline]
fn fast_sin(x: f32) -> f32 {
    // Range-reduce to [-pi, pi], then a 7th-order odd polynomial.
    use std::f32::consts::PI;
    let mut t = x - (x * (1.0 / (2.0 * PI))).round() * (2.0 * PI);
    if t > PI {
        t -= 2.0 * PI;
    }
    let t2 = t * t;
    t * (1.0 + t2 * (-1.0 / 6.0 + t2 * (1.0 / 120.0 + t2 * (-1.0 / 5040.0 + t2 / 362880.0))))
}

/// SnakeBeta in place (kept for callers that already own a copy).
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
    if wide_enough(y.len()) {
        y.par_iter_mut().zip(a.par_iter().zip(b.par_iter())).with_min_len(16)
            .for_each(|(o, (x, z))| *o = x + z);
    } else {
        for ((o, x), z) in y.iter_mut().zip(a).zip(b) {
            *o = x + z;
        }
    }
}

/// `y += x` (residual connections in the codec).
pub fn residual(y: &mut [f32], x: &[f32]) {
    if wide_enough(y.len()) {
        y.par_iter_mut().zip(x.par_iter()).with_min_len(16).for_each(|(o, v)| *o += v);
    } else {
        for (o, v) in y.iter_mut().zip(x) {
            *o += v;
        }
    }
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
    let body = |(r, out): (usize, &mut [f32])| {
        let row = &x[r * cols..(r + 1) * cols];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for c in 0..cols {
            out[c] = row[c] * inv * w[c];
        }
    };
    // One row (the LM's decode case) has nothing to parallelise.
    if wide_enough(rows * cols) {
        y.par_chunks_mut(cols).enumerate().for_each(body);
    } else {
        y.chunks_mut(cols).enumerate().for_each(body);
    }
}

/// Row-wise LayerNorm with affine parameters.
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], y: &mut [f32], rows: usize, cols: usize, eps: f32) {
    let body = |(r, out): (usize, &mut [f32])| {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            out[c] = (row[c] - mean) * inv * w[c] + b[c];
        }
    };
    if wide_enough(rows * cols) {
        y.par_chunks_mut(cols).enumerate().for_each(body);
    } else {
        y.chunks_mut(cols).enumerate().for_each(body);
    }
}

/// `y = silu(gate) * up`.
pub fn silu_mul(gate: &[f32], up: &[f32], y: &mut [f32]) {
    let body = |(o, (g, u)): (&mut f32, (&f32, &f32))| *o = (g / (1.0 + (-g).exp())) * u;
    if wide_enough(y.len()) {
        y.par_iter_mut().zip(gate.par_iter().zip(up.par_iter())).with_min_len(16).for_each(body);
    } else {
        y.iter_mut().zip(gate.iter().zip(up)).for_each(body);
    }
}

/// In-place per-channel scale on `[rows, cols]`.
pub fn scale_cols(x: &mut [f32], scale: &[f32], cols: usize) {
    let body = |row: &mut [f32]| {
        for (c, v) in row.iter_mut().enumerate() {
            *v *= scale[c];
        }
    };
    if wide_enough(x.len()) {
        x.par_chunks_mut(cols).with_min_len(1).for_each(body);
    } else {
        x.chunks_mut(cols).for_each(body);
    }
}

/// `y = a + scale[c] * b`.
pub fn add_scaled(a: &[f32], b: &[f32], scale: &[f32], y: &mut [f32], cols: usize) {
    let mut body = |(r, out): (usize, &mut [f32])| {
        let a = &a[r * cols..(r + 1) * cols];
        let b = &b[r * cols..(r + 1) * cols];
        for c in 0..cols {
            out[c] = a[c] + scale[c] * b[c];
        }
    };
    if wide_enough(y.len()) {
        y.par_chunks_mut(cols).enumerate().for_each(body);
    } else {
        y.chunks_mut(cols).enumerate().for_each(body);
    }
}

/// Rotary position embedding, rotate-half form.
pub fn rope(x: &mut [f32], frames: usize, heads: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;
    let stride = heads * head_dim;
    let body = |(t, row): (usize, &mut [f32])| {
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
    };
    if wide_enough(x.len()) {
        x.par_chunks_mut(stride).enumerate().take(frames).for_each(body);
    } else {
        x.chunks_mut(stride).enumerate().take(frames).for_each(body);
    }
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

/// Transposed conv1d with `[in, out, kernel]` weights, output length `t_in * stride`.
///
/// Phase decomposition: output `q*S + r` receives the taps `k = r, r+S, r+2S, ...`,
/// i.e. `x[ci, q - m] * w[ci, co, r + m*S]` for tap index `m`. That is two taps when
/// `kernel == 2 * stride` (every SEANet upsampler) and one when `kernel == stride`
/// (the latent upsamplers). The naive gather read `taps x cin` strided values per
/// output; this walks time contiguously instead.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(x: &[f32], w: &[f32], b: &[f32], y: &mut [f32],
                        cin: usize, cout: usize, ks: usize, stride: usize, t_in: usize) {
    debug_assert!(ks >= stride && ks <= 2 * stride, "unexpected kernel/stride {ks}/{stride}");
    let t_out = t_in * stride;
    const BLOCK: usize = 1024;
    y.par_chunks_mut(t_out).enumerate().take(cout).for_each(|(co, out)| {
        let mut acc = vec![0.0f32; stride * BLOCK];
        for q0 in (0..t_in).step_by(BLOCK) {
            let len = BLOCK.min(t_in - q0);
            // Each phase owns its own `stride * BLOCK`-slot lane, so zero those
            // lanes separately: clearing only `[0, stride * len)` leaves stale
            // values in the r >= 1 lanes whenever a call spans several blocks.
            for r in 0..stride {
                for slot in acc[r * BLOCK..r * BLOCK + len].iter_mut() {
                    *slot = 0.0;
                }
            }
            for ci in 0..cin {
                let xrow = &x[ci * t_in..(ci + 1) * t_in];
                let wbase = (ci * cout + co) * ks;
                for r in 0..stride {
                    let lane = &mut acc[r * BLOCK..r * BLOCK + len];
                    let mut tap = 0usize;
                    let mut k = r;
                    while k < ks {
                        let wv = w[wbase + k];
                        for (q, slot) in lane.iter_mut().enumerate() {
                            let index = q0 + q;
                            if index >= tap {
                                *slot += xrow[index - tap] * wv;
                            }
                        }
                        tap += 1;
                        k += stride;
                    }
                }
            }
            for r in 0..stride {
                let lane = &acc[r * BLOCK..r * BLOCK + len];
                for (q, value) in lane.iter().enumerate() {
                    out[(q0 + q) * stride + r] = value + b[co];
                }
            }
        }
    });
}

/// Exact GELU: `0.5 x (1 + erf(x / sqrt 2))`.
pub fn gelu(x: &mut [f32]) {
    x.par_iter_mut().with_min_len(16).for_each(|v| {
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
    let body = |(t, row): (usize, &mut [f32])| {
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
    };
    if wide_enough(x.len()) {
        x.par_chunks_mut(stride).enumerate().take(frames).for_each(body);
    } else {
        x.chunks_mut(stride).enumerate().take(frames).for_each(body);
    }
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

/// ELU (alpha 1), in place.
pub fn elu(x: &mut [f32]) {
    x.par_iter_mut().with_min_len(16).for_each(|v| {
        if *v <= 0.0 {
            *v = v.exp() - 1.0;
        }
    });
}

/// 8-lane FMA dot product, horizontally reduced at the end.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32], k: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= k {
        let av = _mm256_loadu_ps(a.as_ptr().add(i));
        let bv = _mm256_loadu_ps(b.as_ptr().add(i));
        acc = _mm256_fmadd_ps(av, bv, acc);
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum: f32 = lanes.iter().sum();
    while i < k {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

/// True when the CPU can run the AVX2 + FMA path.
fn avx2_available() -> bool {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        false
    }
}

/// `acc += window[k*dilation + j] * w[k]` for the whole time block, 8 lanes at a
/// time. Stride-1 only.
///
/// Time is the outer loop so the accumulator stays in a register across all taps;
/// with the tap loop outside, every FMA needed a load and a store of the same
/// accumulator and ran at a third of this rate.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv_block_avx2(window: &[f32], w: &[f32], acc: &mut [f32], tn: usize, ks: usize,
                          dilation: usize) {
    use std::arch::x86_64::*;
    let acc_ptr = acc.as_mut_ptr();
    let win_ptr = window.as_ptr();
    let mut j = 0usize;
    while j + 8 <= tn {
        let mut a = _mm256_loadu_ps(acc_ptr.add(j));
        for k in 0..ks {
            let wv = _mm256_set1_ps(*w.get_unchecked(k));
            let x = _mm256_loadu_ps(win_ptr.add(k * dilation + j));
            a = _mm256_fmadd_ps(x, wv, a);
        }
        _mm256_storeu_ps(acc_ptr.add(j), a);
        j += 8;
    }
    // Tail: scalar.
    while j < tn {
        let mut sum = *acc_ptr.add(j);
        for k in 0..ks {
            sum += *win_ptr.add(k * dilation + j) * *w.get_unchecked(k);
        }
        *acc_ptr.add(j) = sum;
        j += 1;
    }
}

/// Four-row stride-1 conv block: one window vector load feeds four FMAs.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv_block4_avx2(window: &[f32], w: &[&[f32]; 4], acc: &mut [f32], tn: usize,
                           ks: usize, dilation: usize) {
    use std::arch::x86_64::*;
    let win = window.as_ptr();
    let acc_ptr = acc.as_mut_ptr();
    let mut j = 0usize;
    while j + 8 <= tn {
        let mut a0 = _mm256_loadu_ps(acc_ptr.add(j));
        let mut a1 = _mm256_loadu_ps(acc_ptr.add(tn + j));
        let mut a2 = _mm256_loadu_ps(acc_ptr.add(2 * tn + j));
        let mut a3 = _mm256_loadu_ps(acc_ptr.add(3 * tn + j));
        for k in 0..ks {
            let x = _mm256_loadu_ps(win.add(k * dilation + j));
            a0 = _mm256_fmadd_ps(x, _mm256_set1_ps(*w[0].get_unchecked(k)), a0);
            a1 = _mm256_fmadd_ps(x, _mm256_set1_ps(*w[1].get_unchecked(k)), a1);
            a2 = _mm256_fmadd_ps(x, _mm256_set1_ps(*w[2].get_unchecked(k)), a2);
            a3 = _mm256_fmadd_ps(x, _mm256_set1_ps(*w[3].get_unchecked(k)), a3);
        }
        _mm256_storeu_ps(acc_ptr.add(j), a0);
        _mm256_storeu_ps(acc_ptr.add(tn + j), a1);
        _mm256_storeu_ps(acc_ptr.add(2 * tn + j), a2);
        _mm256_storeu_ps(acc_ptr.add(3 * tn + j), a3);
        j += 8;
    }
    while j < tn {
        for r in 0..4 {
            let mut sum = *acc_ptr.add(r * tn + j);
            for k in 0..ks {
                sum += *win.add(k * dilation + j) * *w[r].get_unchecked(k);
            }
            *acc_ptr.add(r * tn + j) = sum;
        }
        j += 1;
    }
}

// ---------------------------------------------------------------------------
// Fused decode kernels
//
// The LM's per-frame loop is a chain of tiny ops (512-2048 elements each). Every
// `par_*` call is a fork-join (~10-20 us on this machine) while the arithmetic is
// 1-5 us, so a layer spent most of its time in the scheduler. These three fuse the
// layer's op chains into one parallel region each, 14 forks -> 5, and skip the
// intermediate allocations.
// ---------------------------------------------------------------------------

/// `y = W @ rms_norm(x) + bias`: norm and the (one or three) projections in one pass.
///
/// `x` is a single row (`cols` values); `w` is `[rows, cols]` (several projections
/// can be concatenated into one weight matrix).
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_gemv(x: &[f32], norm_w: &[f32], w: &[f32], bias: Option<&[f32]>, y: &mut [f32],
                     cols: usize, rows: usize, eps: f32) {
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(y.len(), rows);
    // Norm preamble: 512 elements, cheaper serial than a fork.
    let mut normed = vec![0.0f32; cols];
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / cols as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for c in 0..cols {
        normed[c] = x[c] * inv * norm_w[c];
    }
    let dot = |r: usize| -> f32 {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        if avx2_available() {
            // SAFETY: guarded by `avx2_available()`.
            acc = unsafe { dot_avx2(&normed, row, cols) };
        } else {
            for c in 0..cols {
                acc += normed[c] * row[c];
            }
        }
        acc + bias.map(|b| b[r]).unwrap_or(0.0)
    };
    y.par_iter_mut().enumerate().with_min_len(16).for_each(|(r, slot)| *slot = dot(r));
}

/// `y = W @ x + residual`: projection and residual add in one pass.
pub fn gemv_add(x: &[f32], w: &[f32], residual: &[f32], y: &mut [f32], k: usize, n: usize) {
    debug_assert_eq!(x.len(), k);
    debug_assert_eq!(y.len(), n);
    let dot = |r: usize| -> f32 {
        let row = &w[r * k..(r + 1) * k];
        let acc = if avx2_available() {
            // SAFETY: guarded by `avx2_available()`.
            unsafe { dot_avx2(x, row, k) }
        } else {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += x[i] * row[i];
            }
            acc
        };
        acc + residual[r]
    };
    y.par_iter_mut().enumerate().with_min_len(16).for_each(|(r, slot)| *slot = dot(r));
}

/// `y = W @ (silu(gate) * up)`: the SwiGLU activation and the down projection fused.
pub fn silu_gemv(gate: &[f32], up: &[f32], w: &[f32], residual: &[f32], y: &mut [f32],
                 inter: usize, n: usize) {
    debug_assert_eq!(y.len(), n);
    let mut act = vec![0.0f32; inter];
    for i in 0..inter {
        let g = gate[i];
        act[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
    let dot = |r: usize| -> f32 {
        let row = &w[r * inter..(r + 1) * inter];
        let acc = if avx2_available() {
            // SAFETY: guarded by `avx2_available()`.
            unsafe { dot_avx2(&act, row, inter) }
        } else {
            let mut acc = 0.0f32;
            for i in 0..inter {
                acc += act[i] * row[i];
            }
            acc
        };
        acc + residual[r]
    };
    y.par_iter_mut().enumerate().with_min_len(16).for_each(|(r, slot)| *slot = dot(r));
}

#[cfg(test)]
mod perf_tests {
    #[test]
    fn fused_kernels_have_sane_throughput() {
        // 4096 x 512 = 2.1M MAC; anything above a millisecond is a scheduling bug.
        let (hidden, inter) = (512usize, 2048usize);
        let x = vec![0.1f32; hidden];
        let norm_w = vec![1.0f32; hidden];
        let w = vec![0.01f32; 2 * inter * hidden];
        let mut out = vec![0.0f32; 2 * inter];
        let start = std::time::Instant::now();
        super::rms_norm_gemv(&x, &norm_w, &w, None, &mut out, hidden, 2 * inter, 1e-5);
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        eprintln!("rms_norm_gemv 4096x512: {elapsed:.3} ms");
        assert!(elapsed < 5.0, "fused gemv took {elapsed:.1} ms");
    }
}

// ---------------------------------------------------------------------------
// Channels-last convolution (the general design, not a per-layer patch)
//
// oneDNN and every fast conv engine work channels-last because it makes the
// reduction axis contiguous: for one output element the (tap, channel) pairs are
// adjacent in memory, so the inner loop is a plain SIMD FMA chain. In `[C, T]`
// layout every tap is a separate strided stream and the weights get re-walked per
// output channel, which is why the channels-first kernel sat at ~60 GFLOP/s.
//
// Weights are packed once at load as `[(tap * cin + ci) * cout + co]`, so for a
// fixed reduction index the `cout` values are contiguous.
// ---------------------------------------------------------------------------

/// Pack `[cout, cin, ks]` weights into the channels-last GEMM layout
/// `[tap * cin + ci][co]`, adding the bias later (biases stay separate).
pub fn pack_conv_weights(w: &[f32], cout: usize, cin: usize, ks: usize) -> Vec<f32> {
    let mut packed = vec![0.0f32; ks * cin * cout];
    for co in 0..cout {
        for ci in 0..cin {
            for tap in 0..ks {
                packed[(tap * cin + ci) * cout + co] = w[(co * cin + ci) * ks + tap];
            }
        }
    }
    packed
}

/// Pack `[cin, cout, ks]` weights (PyTorch's `ConvTranspose1d` layout) into the
/// same `[tap * cin + ci][co]` matrix the forward conv uses.
pub fn pack_conv_transpose_weights(w: &[f32], cin: usize, cout: usize, ks: usize) -> Vec<f32> {
    let mut packed = vec![0.0f32; ks * cin * cout];
    for ci in 0..cin {
        for co in 0..cout {
            for tap in 0..ks {
                packed[(tap * cin + ci) * cout + co] = w[(ci * cout + co) * ks + tap];
            }
        }
    }
    packed
}

/// Channels-last causal conv1d.
///
/// `x` is `[t_in, cin]`, `y` is `[t_out, cout]`, `w` is the packed weight matrix
/// from [`pack_conv_weights`]. One thread owns one timestep and eight output
/// channels, so the reduction is a broadcast-plus-FMA chain: `x[t, ci]` is a
/// scalar, the next eight weights are one contiguous vector load, and the
/// accumulator stays in a register for the whole reduction.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_channels_last(x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
                            cin: usize, cout: usize, ks: usize, dilation: usize, stride: usize,
                            pad: usize, t_in: usize, t_out: usize) {
    debug_assert_eq!(w.len(), ks * cin * cout);
    debug_assert_eq!(y.len(), t_out * cout);
    let blocks = cout / 8;
    let vectorised = avx2_available() && cout % 8 == 0;
    if vectorised {
        // One task per timestep; inside, the eight-channel blocks are independent.
        y.par_chunks_mut(cout).enumerate().take(t_out).for_each(|(t, row)| {
            // SAFETY: guarded by `avx2_available()`; `row` is `cout` floats and the
            // weight matrix is fully bounds-checked by construction.
            unsafe { conv_row_avx2(x, w, bias, row, cin, cout, blocks, ks, dilation, stride, pad, t_in, t) };
        });
        return;
    }
    for (t, row) in y.chunks_mut(cout).enumerate().take(t_out) {
        for (co, slot) in row.iter_mut().enumerate() {
            let mut acc = bias[co];
            for tap in 0..ks {
                let src = (t * stride + tap * dilation) as isize - pad as isize;
                if src < 0 || (src as usize) >= t_in {
                    continue;
                }
                let xrow = &x[src as usize * cin..(src as usize + 1) * cin];
                let wbase = tap * cin * cout;
                for ci in 0..cin {
                    acc += xrow[ci] * w[wbase + ci * cout + co];
                }
            }
            *slot = acc;
        }
    }
}

/// One timestep of [`conv1d_channels_last`], eight output channels per FMA.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_row_avx2(x: &[f32], w: &[f32], bias: &[f32], row: &mut [f32],
                        cin: usize, cout: usize, blocks: usize, ks: usize, dilation: usize,
                        stride: usize, pad: usize, t_in: usize, t: usize) {
    use std::arch::x86_64::*;
    for block in 0..blocks {
        let co = block * 8;
        let mut acc = if !bias.is_empty() {
            _mm256_loadu_ps(bias.as_ptr().add(co))
        } else {
            _mm256_setzero_ps()
        };
        for tap in 0..ks {
            let src = (t * stride + tap * dilation) as isize - pad as isize;
            if src < 0 || (src as usize) >= t_in {
                continue;
            }
            let xrow = x.as_ptr().add(src as usize * cin);
            let wbase = tap * cin * cout + co;
            for ci in 0..cin {
                let a = _mm256_set1_ps(*xrow.add(ci));
                let wv = _mm256_loadu_ps(w.as_ptr().add(wbase + ci * cout));
                acc = _mm256_fmadd_ps(a, wv, acc);
            }
        }
        _mm256_storeu_ps(row.as_mut_ptr().add(co), acc);
    }
}

/// SnakeBeta on channels-last activations (`[t, channels]`).
pub fn snake_beta_channels_last(x: &mut [f32], alpha: &[f32], beta: &[f32], channels: usize) {
    let a: Vec<f32> = alpha.iter().map(|v| v.exp()).collect();
    let b: Vec<f32> = beta.iter().map(|v| v.exp() + 1e-9).collect();
    let body = |block: &mut [f32]| {
        for (i, v) in block.iter_mut().enumerate() {
            let c = i % channels;
            let s = (*v * a[c]).sin();
            *v += s * s / b[c];
        }
    };
    if wide_enough(x.len()) {
        // Chunk by rows-in-bulk, not by single rows: 182k one-row tasks cost more
        // in scheduling than the sine work itself.
        x.par_chunks_mut(channels * 64).for_each(body);
    } else {
        x.chunks_mut(channels * 64).for_each(body);
    }
}

/// Transposed conv1d on channels-last activations.
///
/// `y[q*stride + r, co]` gets `sum_m x[q - m, ci] * w[(r + m*stride) * cin + ci][co]`.
/// One thread owns one output timestep and eight channels, so the reduction is the
/// same broadcast-plus-vector-FMA chain the forward conv uses.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_channels_last(x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
                                      cin: usize, cout: usize, ks: usize, stride: usize, t_in: usize) {
    debug_assert_eq!(w.len(), ks * cin * cout);
    let t_out = t_in * stride;
    debug_assert_eq!(y.len(), t_out * cout);
    if avx2_available() && cout % 8 == 0 {
        y.par_chunks_mut(cout).enumerate().take(t_out).for_each(|(j, row)| {
            // SAFETY: guarded by `avx2_available()`.
            unsafe { transpose_row_avx2(x, w, bias, row, cin, cout, ks, stride, t_in, j) };
        });
        return;
    }
    for (j, row) in y.chunks_mut(cout).enumerate().take(t_out) {
        let q = j / stride;
        let r = j % stride;
        for (co, slot) in row.iter_mut().enumerate() {
            let mut acc = bias[co];
            let mut tap = 0usize;
            let mut k = r;
            while k < ks {
                if q >= tap {
                    let xrow = &x[(q - tap) * cin..(q - tap + 1) * cin];
                    let wbase = k * cin * cout;
                    for ci in 0..cin {
                        acc += xrow[ci] * w[wbase + ci * cout + co];
                    }
                }
                tap += 1;
                k += stride;
            }
            *slot = acc;
        }
    }
}

/// One output timestep of [`conv_transpose1d_channels_last`], eight channels per FMA.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_row_avx2(x: &[f32], w: &[f32], bias: &[f32], row: &mut [f32],
                             cin: usize, cout: usize, ks: usize, stride: usize, t_in: usize,
                             j: usize) {
    use std::arch::x86_64::*;
    let q = j / stride;
    let r = j % stride;
    let _ = t_in;
    for co in (0..cout).step_by(8) {
        let mut acc = if !bias.is_empty() {
            _mm256_loadu_ps(bias.as_ptr().add(co))
        } else {
            _mm256_setzero_ps()
        };
        let mut tap = 0usize;
        let mut k = r;
        while k < ks {
            if q >= tap {
                let xrow = x.as_ptr().add((q - tap) * cin);
                let wbase = k * cin * cout + co;
                for ci in 0..cin {
                    let a = _mm256_set1_ps(*xrow.add(ci));
                    let wv = _mm256_loadu_ps(w.as_ptr().add(wbase + ci * cout));
                    acc = _mm256_fmadd_ps(a, wv, acc);
                }
            }
            tap += 1;
            k += stride;
        }
        _mm256_storeu_ps(row.as_mut_ptr().add(co), acc);
    }
}
