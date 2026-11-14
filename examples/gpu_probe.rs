//! What this GPU can actually do: device bandwidth and GEMV launch cost.
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    dispatch_cost_decomp()?;
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

// (appended) 4) per-dispatch cost decomposition: params-buffer creation vs
// bind-group creation vs plain recording.
#[allow(dead_code)]
fn dispatch_cost_decomp() -> anyhow::Result<()> {
    use m5_tts_wgpu::gpu_ops;
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let n = 1024usize;
    let a = dev.tensor([1, n], &vec![1.0f32; n])?;
    let b = dev.tensor([1, n], &vec![1.0f32; n])?;
    let y = dev.zeros([1, n])?;
    let iters = 5000usize;

    // a) full dispatch() path: upload_u32 (buffer creation) + bind group + record
    let s = Instant::now();
    for _ in 0..iters { gpu_ops::add(gpu, a.gpu_buffer()?, b.gpu_buffer()?, y.gpu_buffer()?, n); }
    gpu.sync();
    println!("a) full dispatch():   {:6.1} us/op", s.elapsed().as_secs_f64() * 1e6 / iters as f64);

    // b) same buffers, one bind group reused: record only (bypasses the wrapper)
    let params = gpu.upload_u32("probe.params", &[n as u32, 0, 0, 0]);
    let pipeline = gpu.pipeline("probe_add", m5_tts_wgpu::shaders::ADD_F32, "main");
    let layout = pipeline.get_bind_group_layout(0);
    let group = gpu.raw_device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("probe"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: a.gpu_buffer()?.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: b.gpu_buffer()?.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y.gpu_buffer()?.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
        ],
    });
    let s = Instant::now();
    for _ in 0..iters {
        gpu.reuse("probe_add", &pipeline, &group, (1, 1, 1));
    }
    gpu.sync();
    println!("b) reused bind group: {:6.1} us/op", s.elapsed().as_secs_f64() * 1e6 / iters as f64);

    // c) params buffer creation only (bind group reused; data wrong, timing only)
    let s = Instant::now();
    for _ in 0..iters {
        let _p = gpu.upload_u32("probe.params", &[n as u32, 0, 0, 0]);
    }
    gpu.sync();
    println!("c) params buffer only: {:5.1} us/op", s.elapsed().as_secs_f64() * 1e6 / iters as f64);

    // d) bind group creation only (params buffer reused)
    let s = Instant::now();
    for _ in 0..iters {
        let _g = gpu.raw_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("probe"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.gpu_buffer()?.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.gpu_buffer()?.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.gpu_buffer()?.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
            ],
        });
    }
    println!("d) bind group only:    {:5.1} us/op", s.elapsed().as_secs_f64() * 1e6 / iters as f64);
    Ok(())
}
