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
                // Equal-length zip: vectorises on every target, not just AVX2.
                xr.iter().zip(wr).map(|(a, b)| a * b).sum()
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
    let row = |r: usize, out: &mut [f32]| {
        let xr = &x[r * k..(r + 1) * k];
        // AVX2 dots per output column: the scalar inner loop ran the codec's
        // pre_transformer GEMMs at ~70 GFLOP/s. Weights (<= 12 MB at these
        // shapes) re-read from L3 per row, which L3 bandwidth absorbs.
        let vectorised = avx2_available();
        for (c, slot) in out.iter_mut().enumerate() {
            let wr = &w[c * k..(c + 1) * k];
            let dot = if vectorised {
                // SAFETY: guarded by `avx2_available()`.
                unsafe { dot_avx2(xr, wr, k) }
            } else {
                xr.iter().zip(wr).map(|(a, b)| a * b).sum()
            };
            *slot = dot + bias.map(|b| b[c]).unwrap_or(0.0);
        }
    };
    y.par_chunks_mut(n)
        .enumerate()
        .for_each(|(r, out)| row(r, out));
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
    // A/B'd via seanet_probe variant 1 at the block-4 shape (same-process):
    // 128 -> 107 ms, 512 -> 60 ms (406 GFLOP/s): staging forks amortise while
    // the staged window (96 ch x 518 x 4 B = 200 KB) stays under L2.
    let bt: usize = 512;
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
                    // Slice-paired loop: equal-length `zip` is what lets the compiler
                    // vectorise on every target (x86 baseline SSE2, ARM NEON), not
                    // just where the hand-written AVX2 path is available.
                    for k in 0..ks {
                        let wv = wrow[k];
                        let off = k * dilation;
                        if stride == 1 {
                            let src = &window[base + off..base + off + tn];
                            for (slot, value) in acc.iter_mut().zip(src) {
                                *slot += value * wv;
                            }
                        } else {
                            // Last tapped sample is (tn-1)*stride + off, not tn*stride.
                            let src = &window[base + off..base + off + (tn - 1) * stride + 1];
                            // `chunks`, not `chunks_exact`: the tail chunk holds the
                            // final (partial-stride) sample.
                            for (slot, value) in acc.iter_mut().zip(src.chunks(stride)) {
                                *slot += value[0] * wv;
                            }
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
                let s = (*v * a).sin();
                *slot = v + s * s / b;
            }
        });
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
            *v = *v + s * s / b;
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
    let taps = ks.div_ceil(stride);
    let t_out = t_in * stride;
    const BLOCK: usize = 512;
    const CO: usize = 4;

    // `out_block` always starts at row `co0`, so the body indexes rows relative.
    let body = |co0: usize, rows: usize, out_block: &mut [f32]| {
        // One accumulator plane per (channel, phase).
        let mut acc = vec![0.0f32; CO * stride * BLOCK];
        for q0 in (0..t_in).step_by(BLOCK) {
            let len = BLOCK.min(t_in - q0);
            for slot in acc.iter_mut() {
                *slot = 0.0;
            }
            for ci in 0..cin {
                let xrow = &x[ci * t_in..(ci + 1) * t_in];
                for tap in 0..taps {
                    for r in 0..stride {
                        let k = r + tap * stride;
                        if k >= ks {
                            continue;
                        }
                        // Four output channels share one x read: the previous version
                        // re-read x for every co and streamed ~1 GB per block.
                        for c in 0..rows {
                            // ConvTranspose1d stores [cin, cout, ks].
                            let wv = w[(ci * cout + co0 + c) * ks + k];
                            let off = (c * stride + r) * BLOCK;
                            let lane = &mut acc[off..off + len];
                            for (q, slot) in lane.iter_mut().enumerate() {
                                let index = q0 + q;
                                if index >= tap {
                                    *slot += xrow[index - tap] * wv;
                                }
                            }
                        }
                    }
                }
            }
            for r in 0..stride {
                for c in 0..rows {
                    let off = (c * stride + r) * BLOCK;
                    let lane = &acc[off..off + len];
                    for (q, value) in lane.iter().enumerate() {
                        out_block[c * t_out + (q0 + q) * stride + r] = value + b[co0 + c];
                    }
                }
            }
        }
    };
    if cout % CO == 0 {
        y.par_chunks_mut(t_out * CO)
            .enumerate()
            .for_each(|(block, out)| body(block * CO, CO, out));
    } else {
        y.par_chunks_mut(t_out * CO)
            .enumerate()
            .for_each(|(block, out)| {
                let co0 = block * CO;
                body(co0, (cout - co0).min(CO), out);
            });
    }
}

/// ConvTranspose1d with `ks == 2*stride` (every codec upsampling block) computed
/// as `stride` kernel-2 causal convolutions sharing the input, then an interleave.
///
/// Output sample `t = m*stride + q` only reads input samples `m` and `m-1` with
/// kernel taps `q` and `q+stride`, so phase `q` is a contiguous stride-1 conv over
/// the whole input. Loop order follows the memory hierarchy: shapes whose weights
/// exceed L3 run channel-outer reading the original `[cin, cout, ks]` layout
/// directly (one 64 B line per (ci, co) serves every phase — no repack pass);
/// the rest go through the tuned `conv1d` kernels (~280 GFLOP/s). Both cost one
/// full-size intermediate plus an interleave pass, and both replace the
/// hand-rolled scatter loop (~80 GFLOP/s). The intermediate is row-major in
/// `[co, q]`: row `co*stride + q`.
pub fn conv_transpose1d_phased(x: &[f32], w: &[f32], b: &[f32], y: &mut [f32],
                               cin: usize, cout: usize, ks: usize, stride: usize, t_in: usize) {
    debug_assert_eq!(ks, 2 * stride, "phased form needs ks == 2*stride, got {ks}/{stride}");
    let t_out = t_in * stride;
    let rows_total = cout * stride;
    let mut tmp = vec![0.0f32; rows_total * t_in];

    if rows_total * cin * 2 * 4 > 24 << 20 {
        conv_transpose_phased_co_outer(x, w, b, &mut tmp, cin, cout, ks, stride, t_in);
    } else {
        // Repack into conv1d rows: row `co*stride + q` = [x[m-1] | x[m]] taps =
        // [W[ci,co,q+stride] | W[ci,co,q]].
        let mut w2 = vec![0.0f32; rows_total * cin * 2];
        for co in 0..cout {
            for q in 0..stride {
                let dst = (co * stride + q) * cin * 2;
                for ci in 0..cin {
                    let src = (ci * cout + co) * ks;
                    w2[dst + ci * 2] = w[src + q + stride];
                    w2[dst + ci * 2 + 1] = w[src + q];
                }
            }
        }
        let b2: Vec<f32> = (0..rows_total).map(|row| b[row / stride]).collect();
        conv1d(x, &w2, &b2, &mut tmp, (cin, rows_total, 2, 1, 1, 1, t_in, t_in));
    }

    // `y[co, m*stride + q] = tmp[(co*stride + q)*t_in + m]`. A small staging tile
    // keeps both sides' streams contiguous instead of striding `t_in` floats
    // between successive elements.
    if stride == 1 {
        y.copy_from_slice(&tmp);
        return;
    }
    const TILE: usize = 64;
    y.par_chunks_mut(t_out).enumerate().for_each(|(co, row)| {
        let mut stage = vec![0.0f32; stride * TILE];
        for m0 in (0..t_in).step_by(TILE) {
            let len = TILE.min(t_in - m0);
            for q in 0..stride {
                let src = &tmp[(co * stride + q) * t_in + m0..][..len];
                stage[q * TILE..q * TILE + len].copy_from_slice(src);
            }
            for m in 0..len {
                for q in 0..stride {
                    row[(m0 + m) * stride + q] = stage[q * TILE + m];
                }
            }
        }
    });
}

/// Channel-outer phase convolution for shapes whose weights exceed L3
/// ([`conv_transpose1d_phased`] dispatches here).
///
/// One task per `CO` output channels; each walks the whole time span, so its
/// weight slab (CO x ks x 4 B per ci) stays resident while `x` is shared through
/// L3 — the weights stream exactly once, unlike a time-outer walk that re-reads
/// them once per time block. Reads the original `[cin, cout, ks]` layout: the
/// line holding (ci, co) carries every phase's taps, so there is no repack.
/// `tmp` rows are `[co, q]` = `co*stride + q`.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose_phased_co_outer(x: &[f32], w: &[f32], b: &[f32], tmp: &mut [f32],
                                             cin: usize, cout: usize, ks: usize, stride: usize,
                                             t_in: usize) {
    const CO: usize = 4;
    // One chunk = CO output channels x all their phase rows.
    tmp.par_chunks_mut(t_in * CO * stride).enumerate().for_each(|(block, out_chunk)| {
        let co0 = block * CO;
        let cos = CO.min(cout - co0);
        // Bias first, then accumulate the two-tap conv over every input channel.
        for r in 0..cos {
            let bv = b[co0 + r];
            for q in 0..stride {
                let acc = &mut out_chunk[(r * stride + q) * t_in..(r * stride + q + 1) * t_in];
                acc.iter_mut().for_each(|slot| *slot = bv);
            }
        }
        for ci in 0..cin {
            let xr = &x[ci * t_in..(ci + 1) * t_in];
            for r in 0..cos {
                // One 64 B line holds all `ks` taps for (ci, co): reused by every
                // phase of this output channel.
                let line = &w[(ci * cout + co0 + r) * ks..][..ks];
                for q in 0..stride {
                    let w0 = line[q + stride]; // multiplies x[m-1]
                    let w1 = line[q]; // multiplies x[m]
                    let acc = &mut out_chunk[(r * stride + q) * t_in..(r * stride + q + 1) * t_in];
                    // y[m] = x[m-1]*w0 + x[m]*w1; equal-length zips vectorise on
                    // any target ISA (same trick as conv1d's fallback).
                    acc[0] += xr[0] * w1;
                    if t_in > 1 {
                        let prev = &xr[..t_in - 1];
                        let cur = &xr[1..];
                        for (a, (p, c)) in acc[1..].iter_mut().zip(prev.iter().zip(cur)) {
                            *a += p * w0 + c * w1;
                        }
                    }
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

/// Fused decode-step attention block: RMS-norm -> q/k/v projection -> rope ->
/// KV-cache append -> attention, one parallel region over query heads.
///
/// The production chain paid four rayon barriers per layer for work totalling
/// ~1.5 MMAC; this collapses the first three stages into one fork (one per head
/// instead of per GEMV). The heads sharing a KV head re-project its 128 k/v rows
/// (0.26 MMAC duplicated — far cheaper than a barrier). Cache layout, rope and
/// score accumulation order are identical to [`rms_norm_gemv_bf16`] +
/// [`rope_tabled`] + [`repeat_kv`] + [`write_rows`] + [`attention_decode`], so
/// results are bit-identical.
#[allow(clippy::too_many_arguments)]
pub fn qkv_attend_bf16(x: &[f32], norm_w: &[f32], qkv_w: &[u8], cache_k: &mut [f32],
                       cache_v: &mut [f32], attended: &mut [f32], cos: &[f32], sin: &[f32],
                       position: usize, heads: usize, kv_heads: usize, head_dim: usize,
                       cols: usize, eps: f32) {
    let q_rows = heads * head_dim;
    let kv_rows = kv_heads * head_dim;
    let stride = q_rows;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let kv_per_head = heads / kv_heads;

    let mut normed = vec![0.0f32; cols];
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / cols as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for c in 0..cols {
        normed[c] = x[c] * inv * norm_w[c];
    }

    // Split the cache: rows before `position` are read-only history, the current
    // row is written per head at disjoint slots.
    let len_hist = position;
    let (hist_k, cur_k) = cache_k.split_at_mut(position * stride);
    let (hist_v, cur_v) = cache_v.split_at_mut(position * stride);
    attended.par_chunks_mut(head_dim).enumerate().take(heads)
        .zip(cur_k.par_chunks_mut(head_dim)).zip(cur_v.par_chunks_mut(head_dim))
        .for_each(|(((h, out), k_slot), v_slot)| {
        let qo = h * head_dim;
        let mut q = vec![0.0f32; head_dim];
        for (i, qv) in q.iter_mut().enumerate() {
            let row = (qo + i) * cols * 2;
            *qv = dot_bf16(&normed, &qkv_w[row..row + cols * 2], cols);
        }
        // The four query heads sharing a KV head each re-project its rows.
        let kvh = h / kv_per_head;
        let mut k = vec![0.0f32; head_dim];
        let mut v = vec![0.0f32; head_dim];
        for (i, kvv) in k.iter_mut().enumerate() {
            let row = (q_rows + kvh * head_dim + i) * cols * 2;
            *kvv = dot_bf16(&normed, &qkv_w[row..row + cols * 2], cols);
        }
        for (i, kvv) in v.iter_mut().enumerate() {
            let row = (q_rows + kv_rows + kvh * head_dim + i) * cols * 2;
            *kvv = dot_bf16(&normed, &qkv_w[row..row + cols * 2], cols);
        }
        // Rope this position's q and k in place (rotate-half, table-driven).
        for j in 0..head_dim / 2 {
            let (c, s) = (cos[j], sin[j]);
            let (a, b) = (q[j], q[j + head_dim / 2]);
            q[j] = a * c - b * s;
            q[j + head_dim / 2] = b * c + a * s;
            let (a, b) = (k[j], k[j + head_dim / 2]);
            k[j] = a * c - b * s;
            k[j + head_dim / 2] = b * c + a * s;
        }
        // Append to the cache at this position, in the expanded q-head layout.
        k_slot.copy_from_slice(&k);
        v_slot.copy_from_slice(&v);
        // Attend over rows 0..=position: history from the cache, the current row
        // (just appended) from this task's own slot.
        let mut scores = vec![0.0f32; len_hist + 1];
        let mut maxs = f32::NEG_INFINITY;
        for (j, score) in scores.iter_mut().enumerate().take(len_hist) {
            let ko = j * stride + qo;
            let mut acc = 0.0f32;
            for i in 0..head_dim {
                acc += q[i] * hist_k[ko + i];
            }
            *score = acc * scale;
            maxs = maxs.max(*score);
        }
        {
            let mut acc = 0.0f32;
            for i in 0..head_dim {
                acc += q[i] * k_slot[i];
            }
            let cur = acc * scale;
            scores[len_hist] = cur;
            maxs = maxs.max(cur);
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
                let vv = if j < len_hist { &hist_v[j * stride + qo + i] } else { &v_slot[i] };
                acc += score * vv;
            }
            out[i] = acc * inv;
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
///
/// Every kernel has a portable scalar path for CPUs without AVX2 (ARM, older
/// x86); `M5_NO_SIMD=1` forces it so the fallback stays tested — it is the path a
/// non-AVX2 machine runs, not dead code.
fn avx2_available() -> bool {
    // Detected once: this is called inside per-row loops (millions of times per
    // frame), and `env::var_os` takes the process environment lock — putting the
    // `M5_NO_SIMD` check here uncached made the LM 9x slower.
    static DETECTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DETECTED.get_or_init(|| {
        if std::env::var_os("M5_NO_SIMD").is_some() {
            return false;
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
        {
            false
        }
    })
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
            x.iter().zip(row).map(|(a, b)| a * b).sum()
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
            act.iter().zip(row).map(|(a, b)| a * b).sum()
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

/// Pack `[cin, cout, ks]` ConvTranspose weights into the phased layout
/// `[(co*stride + q) * taps*cin + tap*cin + ci]` (`taps = ks / stride`) for the
/// GPU phased kernel: one contiguous row per `(co, q)` pair, tap-major inside.
pub fn pack_conv_transpose_phased(w: &[f32], cin: usize, cout: usize, ks: usize,
                                  stride: usize) -> Vec<f32> {
    // Layout: [(tap*cin + ci) * rows_total + (co*stride + q)] - ci-major, so a
    // workgroup's 32 (co,q) slots read one contiguous w2 line per (ci, tap)
    // instead of 32 cache lines (the GPU twin of the CPU co-outer lesson).
    let taps = ks / stride;
    let rows = cout * stride;
    let mut out = vec![0.0f32; rows * taps * cin];
    for tap in 0..taps {
        for ci in 0..cin {
            for co in 0..cout {
                for q in 0..stride {
                    out[(tap * cin + ci) * rows + co * stride + q] =
                        w[(ci * cout + co) * ks + q + tap * stride];
                }
            }
        }
    }
    out
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

// ---------------------------------------------------------------------------
// bf16 weights, widened inside the kernel
//
// The VoiceClone checkpoint stores bf16, and lifting it to f32 at load doubles the
// bytes every GEMV has to pull from memory — which is what the decode loop is
// bound on (75 layer passes per frame stream ~1.1 GB against a 30 MB L3). bf16 ->
// f32 is a lossless shift, so widening per use keeps the arithmetic bit-identical
// while halving the traffic.
// ---------------------------------------------------------------------------

/// Widen one bf16 pair (little-endian) into two f32s.
#[inline]
fn bf16_pair(bytes: &[u8]) -> (f32, f32) {
    let lo = u16::from_le_bytes([bytes[0], bytes[1]]);
    let hi = u16::from_le_bytes([bytes[2], bytes[3]]);
    (
        f32::from_bits((lo as u32) << 16),
        f32::from_bits((hi as u32) << 16),
    )
}

/// `dot(x, bf16 weight row)`, eight weights per FMA when AVX2 is available.
fn dot_bf16(x: &[f32], bytes: &[u8], k: usize) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if avx2_available() {
            // SAFETY: guarded by `avx2_available()`.
            return unsafe { dot_bf16_avx2(x, bytes, k) };
        }
    }
    dot_bf16_scalar(x, bytes, k)
}

/// bf16 -> f32 is a 16-bit shift, so eight weights widen with one instruction pair
/// and feed one FMA; the scalar version does two conversions per pair.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_bf16_avx2(x: &[f32], bytes: &[u8], k: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= k {
        let raw = _mm_loadu_si128(bytes.as_ptr().add(i * 2) as *const __m128i);
        let weights = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
        let activations = _mm256_loadu_ps(x.as_ptr().add(i));
        acc = _mm256_fmadd_ps(activations, weights, acc);
        i += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sum: f32 = lanes.iter().sum();
    while i < k {
        let bits = u16::from_le_bytes([*bytes.get_unchecked(i * 2), *bytes.get_unchecked(i * 2 + 1)]);
        sum += *x.get_unchecked(i) * f32::from_bits((bits as u32) << 16);
        i += 1;
    }
    sum
}

fn dot_bf16_scalar(x: &[f32], bytes: &[u8], k: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut i = 0usize;
    while i + 2 <= k {
        let (a, b) = bf16_pair(&bytes[i * 2..i * 2 + 4]);
        acc += x[i] * a + x[i + 1] * b;
        i += 2;
    }
    while i < k {
        let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
        acc += x[i] * f32::from_bits((bits as u32) << 16);
        i += 1;
    }
    acc
}

/// bf16 twin of [`rms_norm_gemv`]: norm, then project from bf16 weights.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_gemv_bf16(x: &[f32], norm_w: &[f32], w: &[u8], bias: Option<&[f32]>, y: &mut [f32],
                          cols: usize, rows: usize, eps: f32) {
    let mut normed = vec![0.0f32; cols];
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / cols as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for c in 0..cols {
        normed[c] = x[c] * inv * norm_w[c];
    }
    let dot = |r: usize| -> f32 {
        let row = &w[r * cols * 2..(r + 1) * cols * 2];
        dot_bf16(&normed, row, cols) + bias.map(|b| b[r]).unwrap_or(0.0)
    };
    y.par_iter_mut().enumerate().with_min_len(16).for_each(|(r, slot)| *slot = dot(r));
}

/// bf16 twin of [`gemv_add`].
pub fn gemv_add_bf16(x: &[f32], w: &[u8], residual: &[f32], y: &mut [f32], k: usize, n: usize) {
    let dot = |r: usize| -> f32 {
        let row = &w[r * k * 2..(r + 1) * k * 2];
        dot_bf16(x, row, k) + residual[r]
    };
    y.par_iter_mut().enumerate().with_min_len(16).for_each(|(r, slot)| *slot = dot(r));
}

/// bf16 twin of [`silu_gemv`].
pub fn silu_gemv_bf16(gate: &[f32], up: &[f32], w: &[u8], residual: &[f32], y: &mut [f32],
                      inter: usize, n: usize) {
    let mut act = vec![0.0f32; inter];
    for i in 0..inter {
        let g = gate[i];
        act[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
    let dot = |r: usize| -> f32 {
        let row = &w[r * inter * 2..(r + 1) * inter * 2];
        dot_bf16(&act, row, inter) + residual[r]
    };
    y.par_iter_mut().enumerate().with_min_len(16).for_each(|(r, slot)| *slot = dot(r));
}

/// SnakeBeta fused into the conv input, optional residual fused into the output.
///
/// A SEANet residual unit is `snake -> conv7 -> snake -> conv1 -> +x`, i.e. ten
/// passes over a 41 MB activation on the largest block (the decode is DRAM-bound,
/// measured 158 GMAC/s against PyTorch's 232). Staging the *activated* window per
/// time block and writing the residual sum in the same pass halves that: the snake
/// no longer needs its own read+write, and the add does not either.
#[allow(clippy::too_many_arguments)]
pub fn snake_conv1d(x: &[f32], alpha: &[f32], beta: &[f32], w: &[f32], bias: &[f32],
                    residual: Option<&[f32]>, y: &mut [f32],
                    (cin, cout, ks, dilation, stride, pad, t_in, t_out):
                        (usize, usize, usize, usize, usize, usize, usize, usize)) {
    // A/B'd via examples/seanet_probe in one process, both production shapes:
    // 128 samples per block made the per-block window staging and rayon forks
    // ~45% of the kernel at block 4 (92 vs 52 ms) and block 1 is fastest here
    // too (34 vs 29 ms, 1.1 TFLOP/s). 1024 plateaus.
    let bt: usize = 512;
    let span = (ks - 1) * dilation + 1;
    let a: Vec<f32> = alpha.iter().map(|v| v.exp()).collect();
    let b: Vec<f32> = beta.iter().map(|v| v.exp() + 1e-9).collect();

    for t0 in (0..t_out).step_by(bt) {
        let tn = bt.min(t_out - t0);
        let start = t0 as isize * stride as isize - pad as isize;
        let len = (tn - 1) * stride + span;
        let mut window = vec![0.0f32; cin * len];
        window
            .par_chunks_mut(len)
            .enumerate()
            .take(cin)
            .for_each(|(ci, row)| {
                let (ca, cb) = (a[ci], b[ci]);
                for (j, slot) in row.iter_mut().enumerate() {
                    let src = start + j as isize;
                    if src >= 0 && (src as usize) < t_in {
                        let v = x[ci * t_in + src as usize];
                        let s = (v * ca).sin();
                        *slot = v + s * s / cb;
                    }
                }
            });

        let window = &window;
        let vectorised = stride == 1 && avx2_available();
        const CO: usize = 4;
        if vectorised && cout % CO == 0 {
            y.par_chunks_mut(t_out * CO).enumerate().for_each(|(block, out_block)| {
                let co0 = block * CO;
                let mut acc = vec![0.0f32; CO * tn];
                for ci in 0..cin {
                    let base = ci * len;
                    let rows: [&[f32]; CO] = std::array::from_fn(|r| {
                        let s = (co0 + r) * cin * ks + ci * ks;
                        &w[s..s + ks]
                    });
                    // SAFETY: guarded by `avx2_available()`.
                    unsafe { conv_block4_avx2(&window[base..base + len], &rows, &mut acc, tn, ks, dilation) };
                }
                for r in 0..CO {
                    let lane = &acc[r * tn..(r + 1) * tn];
                    let co = co0 + r;
                    let dst = &mut out_block[r * t_out + t0..r * t_out + t0 + tn];
                    match residual {
                        Some(res) => {
                            for (q, slot) in dst.iter_mut().enumerate() {
                                *slot = lane[q] + bias[co] + res[co * t_out + t0 + q];
                            }
                        }
                        None => {
                            for (q, slot) in dst.iter_mut().enumerate() {
                                *slot = lane[q] + bias[co];
                            }
                        }
                    }
                }
            });
            continue;
        }
        // Scalar fallback (strided, unaligned channel counts, or no AVX2). It must
        // still parallelise over output channels: forgetting that made the
        // non-AVX2 path 45x slower than the vector one, not "a bit slower".
        y.par_chunks_mut(t_out).enumerate().take(cout).for_each(|(co, out)| {
            let window = &window;
            let wrow = &w[co * cin * ks..(co + 1) * cin * ks];
            // Tap-outer, time-inner with paired slices: the same shape that lets
            // `conv1d`'s fallback vectorise on any target.
            let mut acc = vec![0.0f32; tn];
            for ci in 0..cin {
                let base = ci * len;
                let wr = &wrow[ci * ks..(ci + 1) * ks];
                for (k, wv) in wr.iter().enumerate() {
                    let off = base + k * dilation;
                    if stride == 1 {
                        let src = &window[off..off + tn];
                        for (slot, value) in acc.iter_mut().zip(src) {
                            *slot += value * wv;
                        }
                    } else {
                        let src = &window[off..off + (tn - 1) * stride + 1];
                        for (slot, value) in acc.iter_mut().zip(src.chunks(stride)) {
                            *slot += value[0] * wv;
                        }
                    }
                }
            }
            for (q, slot) in out[t0..t0 + tn].iter_mut().enumerate() {
                *slot = acc[q] + bias[co]
                    + residual.map(|r| r[co * t_out + t0 + q]).unwrap_or(0.0);
            }
        });
    }
}

/// Eight-row stride-1 conv block: one window load feeds eight FMAs.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn conv_block8_avx2(window: &[f32], w: &[&[f32]; 8], acc: &mut [f32], tn: usize,
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
        let mut a4 = _mm256_loadu_ps(acc_ptr.add(4 * tn + j));
        let mut a5 = _mm256_loadu_ps(acc_ptr.add(5 * tn + j));
        let mut a6 = _mm256_loadu_ps(acc_ptr.add(6 * tn + j));
        let mut a7 = _mm256_loadu_ps(acc_ptr.add(7 * tn + j));
        for k in 0..ks {
            let x = _mm256_loadu_ps(win.add(k * dilation + j));
            a0 = _mm256_fmadd_ps(x, _mm256_set1_ps(*w[0].get_unchecked(k)), a0);
            a1 = _mm256_fmadd_ps(x, _mm256_set1_ps(*w[1].get_unchecked(k)), a1);
            a2 = _mm256_fmadd_ps(x, _mm256_set1_ps(*w[2].get_unchecked(k)), a2);
            a3 = _mm256_fmadd_ps(x, _mm256_set1_ps(*w[3].get_unchecked(k)), a3);
            a4 = _mm256_fmadd_ps(x, _mm256_set1_ps(*w[4].get_unchecked(k)), a4);
            a5 = _mm256_fmadd_ps(x, _mm256_set1_ps(*w[5].get_unchecked(k)), a5);
            a6 = _mm256_fmadd_ps(x, _mm256_set1_ps(*w[6].get_unchecked(k)), a6);
            a7 = _mm256_fmadd_ps(x, _mm256_set1_ps(*w[7].get_unchecked(k)), a7);
        }
        _mm256_storeu_ps(acc_ptr.add(j), a0);
        _mm256_storeu_ps(acc_ptr.add(tn + j), a1);
        _mm256_storeu_ps(acc_ptr.add(2 * tn + j), a2);
        _mm256_storeu_ps(acc_ptr.add(3 * tn + j), a3);
        _mm256_storeu_ps(acc_ptr.add(4 * tn + j), a4);
        _mm256_storeu_ps(acc_ptr.add(5 * tn + j), a5);
        _mm256_storeu_ps(acc_ptr.add(6 * tn + j), a6);
        _mm256_storeu_ps(acc_ptr.add(7 * tn + j), a7);
        j += 8;
    }
    while j < tn {
        for r in 0..8 {
            let mut sum = *acc_ptr.add(r * tn + j);
            for k in 0..ks {
                sum += *win.add(k * dilation + j) * *w[r].get_unchecked(k);
            }
            *acc_ptr.add(r * tn + j) = sum;
        }
        j += 1;
    }
}
