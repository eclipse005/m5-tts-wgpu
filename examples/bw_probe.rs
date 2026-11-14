//! DRAM read-bandwidth calibration: the ceiling every streaming kernel here
//! (depth GEMVs, codec tail, transposed convs) is bounded by. Measures parallel
//! read of a 1 GB buffer at full / P-core-ish thread counts, best-of-5 passes.
//!
//! Run: cargo run --release --example bw_probe
use rayon::prelude::*;
use std::time::Instant;

fn read_gb_s(buf: &[f32], threads: usize) -> f64 {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let len = buf.len();
    // 8 passes, keep the best; sum via black_box so nothing is elided.
    let mut best = f64::MAX;
    for _ in 0..8 {
        let s = Instant::now();
        let total: f32 = pool.install(|| {
            buf.par_chunks(1 << 14)
                .map(|c| c.iter().sum::<f32>())
                .sum()
        });
        let secs = s.elapsed().as_secs_f64();
        std::hint::black_box(total);
        best = best.min(secs);
    }
    len as f64 * 4.0 / best / 1e9
}

fn main() {
    let n: usize = 256 << 20; // 1 GiB of f32
    let buf = vec![0.5f32; n];
    std::hint::black_box(&buf);
    let threads = rayon::current_num_threads();
    println!("threads available: {threads}");
    for t in [threads, 16, 12, 8, 4, 1] {
        if t > threads {
            continue;
        }
        println!("read {t:2} threads: {:6.1} GB/s", read_gb_s(&buf, t));
    }
}
