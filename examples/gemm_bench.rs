//! Dense GEMM throughput through the op seam — the GPU's achievable compute rate.
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    for (m, n, k) in [(512usize, 2048usize, 512usize), (2048, 2048, 2048)] {
        for spec in ["cpu", "gpu"] {
            let dev = Device::open(spec)?;
            let x = dev.tensor([m, k], &vec![0.01f32; m * k])?;
            let w = dev.tensor([n, k], &vec![0.01f32; n * k])?;
            let mut y = dev.zeros([m, n])?;
            ops::linear(&dev, &x, &w, None, &mut y)?;
            let start = Instant::now();
            ops::linear(&dev, &x, &w, None, &mut y)?;
            if let Ok(g) = dev.gpu() {
                g.flush();
            }
            let _ = dev.read(&y)?;
            let s = start.elapsed().as_secs_f64();
            println!("{spec} m={m} n={n} k={k}: {s:.3}s  {:.1} GFLOP/s", 2.0 * (m * n * k) as f64 / s / 1e9);
        }
    }
    Ok(())
}
