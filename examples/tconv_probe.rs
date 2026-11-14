//! Transposed-conv (production phased path) at block-1 shape (1536->768, stride 8, t_in 228):
//! the repacked weights are 75 MB and the old time-outer order re-streams them
//! once per 128-sample block. A/B the two loop orders in one process.
//!
//! Run: cargo run --release --example tconv_probe
use m5_tts_wgpu::cpu_ops;
use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let (cin, cout, stride) = (1536usize, 768usize, 8usize);
    let ks = 2 * stride;
    let t_in = 228usize;
    let macs = (cin * cout * ks * t_in) as f64;
    let x = vec![0.01f32; cin * t_in];
    let w = vec![0.01f32; cin * cout * ks];
    let b = vec![0.001f32; cout];
    let mut y = vec![0.0f32; cout * t_in * stride];

    // warmup + correctness reference
    cpu_ops::conv_transpose1d_phased(&x, &w, &b, &mut y, cin, cout, ks, stride, t_in);
    let reference = y.clone();

    for i in 0..4 {
        let s = Instant::now();
        cpu_ops::conv_transpose1d_phased(&x, &w, &b, &mut y, cin, cout, ks, stride, t_in);
        let ms = s.elapsed().as_secs_f64() * 1e3;
        let d = y.iter().zip(&reference).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        println!("run{i}: {ms:7.1} ms  ({:5.1} GFLOP/s)  max|diff vs naive| = {d:.2e}",
                 2.0 * macs / (ms * 1e-3) / 1e9);
        black_box(&y);
    }

}
