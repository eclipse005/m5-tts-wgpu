//! What this GPU can actually do: device bandwidth and GEMV launch cost.
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    println!("adapter: {}", dev.describe());
    // 1) device-side copy bandwidth (bandwidth ceiling without the host link)
    let n = 16_000_000usize;
    let src = dev.tensor([1, n], &vec![0.5f32; n])?;
    let start = Instant::now();
    let dst = dev.copy(&src)?;
    dev.gpu()?.flush();
    let _ = dev.read(&dst)?;
    let elapsed = start.elapsed().as_secs_f64();
    println!("copy 64 MB: {:.3}s -> {:.1} GB/s (read+write)", elapsed, 128.0 / elapsed / 1e3);

    // 2) empty-dispatch cost: 2000 trivial elementwise ops on 1k elements
    let small = dev.tensor([1, 1024], &vec![1.0f32; 1024])?;
    let mut out = dev.zeros([1, 1024])?;
    let start = Instant::now();
    for _ in 0..2000 { ops::add(&dev, &small, &small, &mut out)?; }
    dev.gpu()?.flush();
    let _ = dev.read(&out)?;
    println!("trivial dispatch: {:.1} us/op", start.elapsed().as_secs_f64() * 1e6 / 2000.0);

    // 3) GEMV as the LM runs it (m=1, n=512, k=512)
    let x = dev.tensor([1, 512], &vec![0.01f32; 512])?;
    let w = dev.tensor([512, 512], &vec![0.01f32; 512 * 512])?;
    let mut y = dev.zeros([1, 512])?;
    ops::linear(&dev, &x, &w, None, &mut y)?;
    let start = Instant::now();
    for _ in 0..500 { ops::linear(&dev, &x, &w, None, &mut y)?; }
    dev.gpu()?.flush();
    let _ = dev.read(&y)?;
    println!("GEMV 512x512: {:.1} us/op", start.elapsed().as_secs_f64() * 1e6 / 500.0);
    Ok(())
}
