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
#[allow(clippy::too_many_arguments)]
pub fn conv1d(x: &[f32], w: &[f32], bias: &[f32], y: &mut [f32],
              (cin, cout, ks, dilation, stride, pad, t_in, t_out): (usize, usize, usize, usize, usize, usize, usize, usize)) {
    y.par_chunks_mut(t_out).enumerate().for_each(|(co, out)| {
        if co >= cout {
            return;
        }
        for (t, slot) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for ci in 0..cin {
                let wbase = (co * cin + ci) * ks;
                let xbase = ci * t_in;
                for k in 0..ks {
                    let src = (t * stride + k * dilation) as isize - pad as isize;
                    if src >= 0 && (src as usize) < t_in {
                        acc += x[xbase + src as usize] * w[wbase + k];
                    }
                }
            }
            *slot = acc + bias[co];
        }
    });
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
