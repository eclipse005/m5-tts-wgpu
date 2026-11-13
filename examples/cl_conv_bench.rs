//! Channels-last + packed weights conv on the GPU vs the checkpoint-layout kernel.
use m5_tts_wgpu::cpu_ops;
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::gpu_ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let (cin, cout, ks, t) = (96usize, 96usize, 7usize, 182400usize);
    let macs = (cin * cout * ks * t) as f64;
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;

    // Packed weights [tap*cin+ci][cout], channels-last activations.
    let w = vec![0.01f32; cout * cin * ks];
    let packed = cpu_ops::pack_conv_weights(&w, cout, cin, ks);
    let x = dev.tensor([t, cin], &vec![0.01f32; t * cin])?;
    let wp = dev.tensor([ks * cin, cout], &packed)?;
    let b = dev.tensor([cout, 1], &vec![0.001f32; cout])?;
    let y = dev.zeros([t, cout])?;
    gpu_ops::conv1d_cl(gpu, x.gpu_buffer()?, wp.gpu_buffer()?, b.gpu_buffer()?, y.gpu_buffer()?,
                       cin, cout, ks, 1, 1, ks - 1, t, t);
    gpu.flush();
    let _ = dev.read(&y)?;
    let start = Instant::now();
    gpu_ops::conv1d_cl(gpu, x.gpu_buffer()?, wp.gpu_buffer()?, b.gpu_buffer()?, y.gpu_buffer()?,
                       cin, cout, ks, 1, 1, ks - 1, t, t);
    gpu.flush();
    let _ = dev.read(&y)?;
    let s = start.elapsed().as_secs_f64();
    println!("gpu channels-last packed: {:.3}s  {:.1} GFLOP/s", s, 2.0 * macs / s / 1e9);
    Ok(())
}
