//! Block-4 residual unit, costed piece by piece at the real in-situ shape
//! (96 ch x 188k samples): where do the 127 GFLOP/s in situ go, given the
//! standalone conv kernel does 244?
//!
//! Variants (same math, same shapes):
//!   1. conv1d alone                 — the reference kernel number
//!   2. snake -> conv1d              — unfused, one extra full-tensor pass
//!   3. snake -> conv1d -> +res      — the full residual unit, unfused
//!   4. snake_conv1d (fused, no res) — the production kernel
//!   5. snake_conv1d (fused, +res)   — the production residual-unit tail
//!
//! Run: cargo run --release --example seanet_probe
use m5_tts_wgpu::cpu_ops;
use m5_tts_wgpu::device::{Device, Tensor};
use m5_tts_wgpu::ops::{self, Conv1d};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("cpu")?;
    let (cin, cout, ks, t) = (96usize, 96usize, 7usize, 188_000usize);
    let macs = (cin * cout * ks * t) as f64;
    let mut rng: u64 = 0x9E3779B97F4A7C15;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 40) as f32 / 16_777_216.0 - 1.0
    };
    let x = dev.tensor([cin, t], &(0..cin * t).map(|_| next()).collect::<Vec<f32>>())?;
    let w = dev.tensor([cout, cin * ks], &(0..cout * cin * ks).map(|_| next()).collect::<Vec<f32>>())?;
    let bias = dev.tensor([cout, 1], &(0..cout).map(|_| next()).collect::<Vec<f32>>())?;
    let alpha = dev.tensor([cin, 1], &(0..cin).map(|_| next() * 0.1).collect::<Vec<f32>>())?;
    let beta = dev.tensor([cin, 1], &(0..cin).map(|_| next() * 0.1).collect::<Vec<f32>>())?;
    let res = dev.tensor([cout, t], &(0..cout * t).map(|_| next()).collect::<Vec<f32>>())?;

    let gflop = |secs: f64| 2.0 * macs / secs / 1e9;

    // 1. conv1d alone
    let p = Conv1d::causal(cin, cout, ks, t).dilated(1);
    let mut y = dev.zeros([cout, t])?;
    let s = Instant::now();
    ops::conv1d(&dev, &x, &w, &bias, &mut y, &p)?;
    let t1 = s.elapsed().as_secs_f64();
    println!("1. conv1d alone            {:6.1} ms  {:6.1} GFLOP/s", t1 * 1e3, gflop(t1));

    // 2. snake -> conv1d
    let s = Instant::now();
    let h = ops::snake_beta(&dev, &x, &alpha, &beta)?;
    let t_sn = s.elapsed().as_secs_f64();
    let s = Instant::now();
    ops::conv1d(&dev, &h, &w, &bias, &mut y, &p)?;
    let t2 = s.elapsed().as_secs_f64();
    println!("2. snake: {:6.1} ms + conv: {:6.1} ms = {:6.1} ms  (conv {:6.1} GFLOP/s)",
             t_sn * 1e3, t2 * 1e3, (t_sn + t2) * 1e3, gflop(t2));

    // 3. snake -> conv1d -> +res (full unit, unfused)
    let s = Instant::now();
    let h = ops::snake_beta(&dev, &x, &alpha, &beta)?;
    let mut c = dev.zeros([cout, t])?;
    ops::conv1d(&dev, &h, &w, &bias, &mut c, &p)?;
    let mut sum = dev.zeros([cout, t])?;
    ops::add(&dev, &c, &res, &mut sum)?;
    let t3 = s.elapsed().as_secs_f64();
    println!("3. unfused unit (snake+conv+add) {:6.1} ms  (conv {:6.1} GFLOP/s)", t3 * 1e3, gflop(t3 - t_sn));

    // 4. fused snake_conv1d, no residual
    let s = Instant::now();
    let mut y4 = dev.zeros([cout, t])?;
    cpu_ops::snake_conv1d(x.cpu_slice()?, alpha.cpu_slice()?, beta.cpu_slice()?, w.cpu_slice()?,
                          bias.cpu_slice()?, None, y4.cpu_slice_mut()?,
                          (cin, cout, ks, 1, 1, ks - 1, t, t));
    let t4 = s.elapsed().as_secs_f64();
    println!("4. fused snake_conv (no res)     {:6.1} ms  {:6.1} GFLOP/s", t4 * 1e3, gflop(t4));

    // 5. fused snake_conv1d + residual
    let s = Instant::now();
    let mut y5 = dev.zeros([cout, t])?;
    cpu_ops::snake_conv1d(x.cpu_slice()?, alpha.cpu_slice()?, beta.cpu_slice()?, w.cpu_slice()?,
                          bias.cpu_slice()?, Some(res.cpu_slice()?), y5.cpu_slice_mut()?,
                          (cin, cout, ks, 1, 1, ks - 1, t, t));
    let t5 = s.elapsed().as_secs_f64();
    println!("5. fused snake_conv (+res)       {:6.1} ms  {:6.1} GFLOP/s", t5 * 1e3, gflop(t5));

    // Sanity: same-math variants must agree (y2 holds conv(snake(x)) from
    // variant 2; y5 must equal that plus the residual; y4 must equal y2).
    let y2v = dev.read(&y)?;
    let y4v = dev.read(&y4)?;
    let y5v = dev.read(&y5)?;
    let d42 = y4v.iter().zip(&y2v).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
    let d5 = y5v.iter().zip(&y4v).zip(res.cpu_slice()?)
        .map(|((p, q), r)| (p - q - r).abs()).fold(0.0f32, f32::max);
    println!("max|4-2| = {d42:.2e}   max|5-(4+res)| = {d5:.2e}");
    Ok(())
}
