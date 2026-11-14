//! Does embedding prefetch tasks inside a GEMV's parallel region lift the
//! sequential GEMV chain toward the ~51 GB/s DRAM ceiling?
//!
//! Emulates one depth layer step: qkv (768x512) -> o (512x512) -> gateup
//! (4096x512) -> down (512x2048), cycling 5 weight sets so the 38 MB working
//! set thrashes L3 exactly like the real depth pass. Variants:
//!   A. plain GEMV chain (production shape)
//!   B. each GEMV's par region also contains prefetch tasks for the NEXT
//!      region's weights, so latency-bound dot tasks and pure streams overlap.
//!
//! Run: cargo run --release --example gemv_prefetch_probe
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

#[inline]
fn dot_bf16(x: &[f32], row: &[u8]) -> f32 {
    let mut acc = 0.0f32;
    for (i, chunk) in row.chunks_exact(2).enumerate() {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        acc += x[i] * f32::from_bits((bits as u32) << 16);
    }
    acc
}

fn stream(slab: &[u8]) {
    // Read-only walk; the sum defeats elision without serialising.
    let mut acc = 0u32;
    for v in slab.chunks_exact(4) {
        acc = acc.wrapping_add(u32::from_le_bytes([v[0], v[1], v[2], v[3]]));
    }
    black_box(acc);
}

/// One GEMV region: `rows` dot tasks plus prefetch tasks for `next`'s weights.
fn gemv_region(x: &[f32], w: &[u8], y: &mut [f32], cols: usize, next: &[&[u8]]) {
    let rows = y.len();
    const SLAB: usize = 8192;
    let pf_total: usize = next.iter().map(|p| p.len().div_ceil(SLAB)).sum();
    // Probe-only: rayon's `for_each` is `Fn`, so writes go through a raw pointer
    // (task indices are disjoint by construction).
    struct Slot(*mut f32);
    unsafe impl Send for Slot {}
    unsafe impl Sync for Slot {}
    impl Slot {
        fn set(&self, t: usize, v: f32) {
            unsafe { *self.0.add(t) = v };
        }
    }
    let y_ptr = Slot(y.as_mut_ptr());
    (0..rows + pf_total).into_par_iter().with_min_len(16).for_each(|t| {
        if t < rows {
            y_ptr.set(t, dot_bf16(x, &w[t * cols * 2..(t + 1) * cols * 2]));
        } else {
            // Walk the flattened next-region slabs.
            let mut left = t - rows;
            for p in next {
                let chunks = p.len().div_ceil(SLAB);
                if left < chunks {
                    let end = ((left + 1) * SLAB).min(p.len());
                    stream(&p[left * SLAB..end]);
                    break;
                }
                left -= chunks;
            }
        }
    });
}

fn plain_gemv(x: &[f32], w: &[u8], y: &mut [f32], cols: usize) {
    y.par_iter_mut().enumerate().with_min_len(16).for_each(|(r, slot)| {
        *slot = dot_bf16(x, &w[r * cols * 2..(r + 1) * cols * 2]);
    });
}

fn main() {
    let cols = 512usize;
    let shapes: [usize; 4] = [768, 512, 4096, 512];
    let sets: Vec<Vec<Vec<u8>>> = (0..5)
        .map(|_| shapes.iter().map(|rows| vec![0x3Fu8; rows * cols * 2]).collect())
        .collect();
    let x: Vec<f32> = vec![0.01; cols];
    let mut y = vec![0.0f32; *shapes.iter().max().unwrap()];
    let _ = black_box(&sets);
    let steps = 40;

    for layer in 0..5 {
        for (ri, rows) in shapes.iter().enumerate() {
            plain_gemv(&x, &sets[layer][ri], &mut y[..*rows], cols);
        }
    }

    // A: plain chain
    let s = Instant::now();
    for _ in 0..steps {
        for layer in 0..5 {
            for (ri, rows) in shapes.iter().enumerate() {
                plain_gemv(&x, &sets[layer][ri], &mut y[..*rows], cols);
            }
        }
    }
    let a = s.elapsed().as_secs_f64() * 1e3 / steps as f64;
    println!("A plain 5-layer chain: {:6.2} ms/step", a);

    // B: every region also streams the next region's weights in the same fork.
    // Order per layer: qkv->o->gateup->down; the region before each GEMV
    // prefetches it (down prefetches the next layer's qkv).
    let s = Instant::now();
    for _ in 0..steps {
        for layer in 0..5 {
            let next_layer = (layer + 1) % 5;
            // qkv, prefetching o
            gemv_region(&x, &sets[layer][0], &mut y[..shapes[0]], cols, &[&sets[layer][1]]);
            // o, prefetching gateup
            gemv_region(&x, &sets[layer][1], &mut y[..shapes[1]], cols, &[&sets[layer][2]]);
            // gateup, prefetching down
            gemv_region(&x, &sets[layer][2], &mut y[..shapes[2]], cols, &[&sets[layer][3]]);
            // down, prefetching next layer's qkv
            gemv_region(&x, &sets[layer][3], &mut y[..shapes[3]], cols, &[&sets[next_layer][0]]);
        }
    }
    let b = s.elapsed().as_secs_f64() * 1e3 / steps as f64;
    println!("B embedded-prefetch chain: {:6.2} ms/step", b);
}
