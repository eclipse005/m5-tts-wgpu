//! Measure the P104 pipeline-switch penalty: one kernel repeated vs several
//! kernels rotated, same buffers, same batch. Isolates GPU dispatch latency.
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::gpu_ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let n = 512usize;
    let x = dev.tensor([1, n], &vec![0.1f32; n])?;
    let w = dev.tensor([n, n], &vec![0.01f32; n * n])?;
    let nw = dev.tensor([1, n], &vec![1.0f32; n])?;
    let y = dev.zeros([1, n])?;
    let z = dev.zeros([1, n])?;
    let iters = 2000;

    // warm up pipelines
    for _ in 0..50 {
        gpu_ops::rms_norm(gpu, x.gpu_buffer()?, nw.gpu_buffer()?, y.gpu_buffer()?, 1, n, 1e-5);
        gpu_ops::linear(gpu, y.gpu_buffer()?, w.gpu_buffer()?, None, z.gpu_buffer()?, 1, n, n);
        gpu_ops::add(gpu, x.gpu_buffer()?, z.gpu_buffer()?, y.gpu_buffer()?, n);
    }
    gpu.sync();

    let s = Instant::now();
    for _ in 0..iters {
        gpu_ops::linear(gpu, x.gpu_buffer()?, w.gpu_buffer()?, None, y.gpu_buffer()?, 1, n, n);
    }
    gpu.sync();
    println!("single gemv repeat:      {:.1} us/dispatch", s.elapsed().as_secs_f64()*1e6/iters as f64);

    let s = Instant::now();
    for _ in 0..iters {
        gpu_ops::rms_norm(gpu, x.gpu_buffer()?, nw.gpu_buffer()?, y.gpu_buffer()?, 1, n, 1e-5);
        gpu_ops::linear(gpu, y.gpu_buffer()?, w.gpu_buffer()?, None, z.gpu_buffer()?, 1, n, n);
        gpu_ops::add(gpu, x.gpu_buffer()?, z.gpu_buffer()?, y.gpu_buffer()?, n);
    }
    gpu.sync();
    println!("3-kernel rotation:       {:.1} us/dispatch", s.elapsed().as_secs_f64()*1e6/(iters*3) as f64);
    Ok(())
}