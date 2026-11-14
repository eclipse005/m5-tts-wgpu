//! Memory-primitive probe at codec block-4 sizes (72 MB tensors): is the
//! elementwise slowness the ops, or the allocator / page faults underneath?
//!
//! Run: cargo run --release --example mem_probe
use m5_tts_wgpu::cpu_ops;
use m5_tts_wgpu::device::{Device, Tensor};
use rayon::prelude::*;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    alloc_costs();
    let dev = Device::open("cpu")?;
    let n = 96 * 188_000; // 18.05M f32 = 72.2 MB
    let alpha = vec![0.1f32; n / (96 * 188_000) * 96];
    let _ = alpha; // placeholder to keep names stable

    // 1. First-time zeros allocation (cold heap)
    let s = Instant::now();
    let a = dev.zeros([96, 188_000])?;
    println!("1. dev.zeros first        {:7.1} ms", s.elapsed().as_secs_f64() * 1e3);

    // 2. Second allocation of the same size (heap warm) + drop pattern
    for i in 0..3 {
        let s = Instant::now();
        let t = dev.zeros([96, 188_000])?;
        let t_zeros = s.elapsed().as_secs_f64() * 1e3;
        drop(t);
        println!("2. dev.zeros drop  iter{}  {:7.1} ms", i, t_zeros);
    }

    // 3. Write pass (touch every element) on the warm buffer
    {
        let mut t = dev.zeros([96, 188_000])?;
        let s = Instant::now();
        t.cpu_slice_mut()?.iter_mut().for_each(|v| *v = 1.0);
        println!("3. write 72MB (serial)    {:7.1} ms", s.elapsed().as_secs_f64() * 1e3);
        let s = Instant::now();
        t.cpu_slice_mut()?.par_iter_mut().for_each(|v| *v = 2.0);
        println!("3. write 72MB (rayon)     {:7.1} ms", s.elapsed().as_secs_f64() * 1e3);
        drop(t);
    }

    // 4. copy 72MB (read+write = 144MB)
    let src = dev.tensor([96, 188_000], &vec![0.5f32; n])?;
    for i in 0..3 {
        let s = Instant::now();
        let dst = dev.copy(&src)?;
        let ms = s.elapsed().as_secs_f64() * 1e3;
        println!("4. dev.copy  iter{}        {:7.1} ms  ({:5.1} GB/s)", i, ms, 144.0 / ms * 1e3 / 1e3);
        drop(dst);
    }

    // 5. add: 3 buffers, read a+b write y = 216MB
    let b = dev.tensor([96, 188_000], &vec![0.25f32; n])?;
    let mut y = dev.zeros([96, 188_000])?;
    for i in 0..3 {
        let s = Instant::now();
        cpu_ops::add(src.cpu_slice()?, b.cpu_slice()?, y.cpu_slice_mut()?);
        let ms = s.elapsed().as_secs_f64() * 1e3;
        println!("5. add       iter{}        {:7.1} ms  ({:5.1} GB/s)", i, ms, 216.0 / ms * 1e3 / 1e3);
    }

    // 6. snake in-place, libm sin (current production path)
    {
        let mut t = dev.copy(&src)?;
        for i in 0..3 {
            let s = Instant::now();
            cpu_ops::snake_beta(t.cpu_slice_mut()?, &alpha[..96], &alpha[..96], 96, 188_000);
            let ms = s.elapsed().as_secs_f64() * 1e3;
            println!("6. snake libm iter{}       {:7.1} ms  ({:5.1} GB/s)", i, ms, 144.0 / ms * 1e3 / 1e3);
        }
    }

    // 7. snake out-of-place via snake_beta_into, fast_sin (fused-read variant)
    {
        let x = dev.copy(&src)?;
        let mut out = dev.zeros([96, 188_000])?;
        for i in 0..3 {
            let s = Instant::now();
            cpu_ops::snake_beta_into(x.cpu_slice()?, out.cpu_slice_mut()?, &alpha[..96], &alpha[..96], 96, 188_000);
            let ms = s.elapsed().as_secs_f64() * 1e3;
            println!("7. snake fast  iter{}       {:7.1} ms  ({:5.1} GB/s)", i, ms, 144.0 / ms * 1e3 / 1e3);
        }
    }

    // 8. memcpy reference: copy_from_slice
    {
        let src = vec![0.5f32; n];
        let mut dst = vec![0.0f32; n];
        let s = Instant::now();
        dst.copy_from_slice(&src);
        let ms = s.elapsed().as_secs_f64() * 1e3;
        println!("8. copy_from_slice         {:7.1} ms  ({:5.1} GB/s)", ms, 144.0 / ms * 1e3 / 1e3);
    }
    Ok(())
}



// (appended) 9. Raw allocator costs, separated.
#[allow(dead_code)]
fn alloc_costs() {
    let n = 96 * 188_000;
    for i in 0..3 {
        let s = std::time::Instant::now();
        let v = vec![0.0f32; n];
        println!("9. vec![0.0;72MB]  iter{}  {:6.1} ms", i, s.elapsed().as_secs_f64() * 1e3);
        drop(v);
    }
    for i in 0..3 {
        let s = std::time::Instant::now();
        let v = vec![1.5f32; n];
        println!("9. vec![1.5;72MB]  iter{}  {:6.1} ms", i, s.elapsed().as_secs_f64() * 1e3);
        drop(v);
    }
    for i in 0..3 {
        let s = std::time::Instant::now();
        let v: Vec<f32> = Vec::with_capacity(n);
        println!("9. with_capacity   iter{}  {:6.1} ms", i, s.elapsed().as_secs_f64() * 1e3);
        drop(v);
    }
    for i in 0..3 {
        let s = std::time::Instant::now();
        let v = vec![0.0f32; n];
        let ms_alloc = s.elapsed().as_secs_f64() * 1e3;
        drop(v);
        let ms_drop = s.elapsed().as_secs_f64() * 1e3 - ms_alloc;
        println!("9. alloc {:6.1} + drop {:6.1} ms  iter{}", ms_alloc, ms_drop, i);
    }
}
