//! conv1d throughput at the heaviest SEANet shape (block4: 96ch, 7 taps, 182400 frames).
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::ops::{self, Conv1d};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let (cin, cout, ks, t) = (96usize, 96usize, 7usize, 182400usize);
    let macs = (cin * cout * ks * t) as f64;
    let p = Conv1d::causal(cin, cout, ks, t);
    // Device memory bandwidth reference (copy 17.5M floats = 70 MB).
    {
        let dev = Device::open("gpu")?;
        let n = 17_500_000usize;
        let src = dev.tensor([1, n], &vec![0.5f32; n])?;
        let start = Instant::now();
        let dst = dev.copy(&src)?;
        dev.gpu()?.flush();
        let kernel_only = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let _ = dev.read(&dst)?;
        let readback = start.elapsed().as_secs_f64();
        println!("gpu copy kernel: {:.3}s ({:.1} GB/s dev-side), readback: {:.3}s ({:.1} GB/s host)",
                 kernel_only, 140.0 / kernel_only / 1e3, readback, 70.0 / readback / 1e3);
    }

    for spec in ["cpu", "gpu"] {
        let dev = Device::open(spec)?;
        let x = dev.tensor([cin, t], &vec![0.01f32; cin * t])?;
        let w = dev.tensor([cout, cin * ks], &vec![0.01f32; cout * cin * ks])?;
        let b = dev.tensor([cout, 1], &vec![0.001f32; cout])?;
        let mut y = dev.zeros([cout, t])?;
        ops::conv1d(&dev, &x, &w, &b, &mut y, &p)?;      // warm
        let start = Instant::now();
        ops::conv1d(&dev, &x, &w, &b, &mut y, &p)?;
        let _ = dev.read(&y)?;
        let s = start.elapsed().as_secs_f64();
        println!("{spec}: {:.3}s  {:.1} GFLOP/s  [{}]", s, 2.0 * macs / s / 1e9, dev.describe());
    }
    Ok(())
}
