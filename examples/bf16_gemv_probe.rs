//! Isolated bf16 GEMV throughput at the depth decoder's shapes, back to back.
//!
//!   cargo run --release --example bf16_gemv_probe
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::gpu_ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let k = 512usize;
    for n in [512usize, 768, 2048, 4096] {
        // Packed bf16: two per u32 (all 1.0 => 0x3f80 pairs).
        let packed: Vec<f32> = vec![f32::from_bits(0x3f803f80); n * k / 2];
        let w = dev.tensor([n * k / 2, 1], &packed)?;
        let x = dev.tensor([1, k], &vec![0.5f32; k])?;
        let mut y = dev.zeros([1, n])?;
        let kk = gpu.kernel_params("probe.gemv_bf16", m5_tts_wgpu::shaders::GEMV_BF16_F32,
                                   "ro,ro,ro,rw");
        let zero = dev.tensor([1, 1], &[0.0f32])?;
        let run = || gpu_ops::f_gemv(&kk, gpu, 1, x.gpu_buffer().unwrap(),
                                     w.gpu_buffer().unwrap(), zero.gpu_buffer().unwrap(),
                                     y.gpu_buffer().unwrap(), n, k);
        run();
        gpu.sync();
        let iters = 300;
        let start = Instant::now();
        for _ in 0..iters { run(); }
        gpu.sync();
        let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let bytes = (n * k * 2) as f64;
        println!("gemv_bf16 n={n:5} k={k}: {ms:7.2} us   {:6.1} GB/s   ({:.1} GMAC/s)",
                 bytes / (ms * 1e6), (n * k) as f64 / (ms * 1e6) / 1e3);
        let _ = &mut y;
    }
    Ok(())
}
