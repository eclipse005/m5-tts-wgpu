//! fp32 FMA ceiling of the adapter.
use m5_tts_wgpu::device::Device;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let threads = 1_000_000usize;
    let iters = 4096u32;
    let out = gpu.empty("fma out", threads);
    let params = gpu.upload_u32("fma params", &[iters, 0]);
    let groups = ((threads / 256) as u32, 1, 1);
    gpu.dispatch("fma_probe", m5_tts_wgpu::shaders::FMA_PROBE, "main",
                 &[(0, &out), (1, &params)], groups);
    gpu.flush();
    let _ = gpu.download(&out, 8);
    let start = Instant::now();
    gpu.dispatch("fma_probe", m5_tts_wgpu::shaders::FMA_PROBE, "main",
                 &[(0, &out), (1, &params)], groups);
    gpu.flush();
    let _ = gpu.download(&out, 8);
    let elapsed = start.elapsed().as_secs_f64();
    let flops = 4.0 * iters as f64 * threads as f64;
    println!("fma probe: {:.3}s  {:.1} GFLOP/s ({} threads x {} iters x 4 chains)",
             elapsed, 2.0 * flops / elapsed / 1e9, threads, iters);
    Ok(())
}
