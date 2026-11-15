//! What does a big GPU scratch buffer cost?
//!
//! The codec's `causal_conv` allocates two channels-last scratch buffers per
//! convolution with `Gpu::empty` (a plain `create_buffer`, never pooled), at
//! 40+ MB each in the last block. This probe separates creation cost from
//! pooling cost at those sizes.
//!
//!   cargo run --release --example gpu_alloc_probe [megabytes]
use m5_tts_wgpu::device::Device;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let mb: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(41);
    let len = mb * 1_000_000 / 4; // f32 count
    println!("{mb} MB buffers");

    let iters = 20;
    let start = Instant::now();
    for i in 0..iters {
        let buf = gpu.empty(&format!("probe.{i}"), len);
        drop(buf);
    }
    gpu.sync();
    println!("create_buffer + drop   {:7.2} ms each", start.elapsed().as_secs_f64() * 1e3 / iters as f64);

    // Pooled: take, hand straight back, then take again.
    for _ in 0..2 { let b = gpu.take_pooled(len); gpu.recycle(len, b); }
    let start = Instant::now();
    for _ in 0..iters {
        let buf = gpu.take_pooled(len);
        gpu.recycle(len, buf);
    }
    println!("pooled take + recycle  {:7.2} ms each", start.elapsed().as_secs_f64() * 1e3 / iters as f64);

    // The deferral the ops actually hit: a buffer dropped while a batch is open
    // only joins the pool after the next flush.
    let start = Instant::now();
    for _ in 0..iters {
        let buf = gpu.take_pooled(len);
        gpu.recycle_on_drop(len, buf);
        gpu.sync();
    }
    println!("deferred drop + flush  {:7.2} ms each", start.elapsed().as_secs_f64() * 1e3 / iters as f64);
    Ok(())
}
