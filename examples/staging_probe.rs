//! Upload rate for the two things that matter: `queue.write_buffer` chunk size
//! (wgpu allocates+maps a fresh staging buffer per call) and whether the
//! destination buffer is fresh or already created.
//!
//!   cargo run --release --example staging_probe
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let mut adapters = pollster::block_on(
        instance.enumerate_adapters(wgpu::Backends::VULKAN | wgpu::Backends::GL));
    adapters.sort_by_key(|a| match a.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 0u8,
        _ => 2,
    });
    let adapter = adapters.remove(0);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))?;
    println!("{}", adapter.get_info().name);

    let total: usize = 384 * 1_000_000; // one decoder block's worth of weights
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC;
    let make = |size: usize, label: &str| device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label), size: size as u64, usage, mapped_at_creation: false,
    });

    let dst = make(total, "dst");
    for chunk_mb in [2usize, 4, 8, 16, 32] {
        let chunk = chunk_mb * 1_000_000;
        let src = vec![0u8; chunk];
        let n = total / chunk;
        let start = Instant::now();
        let mut off = 0usize;
        for _ in 0..n {
            queue.write_buffer(&dst, off as u64, &src);
            off += chunk;
        }
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let s = start.elapsed().as_secs_f64();
        println!("preallocated, {chunk_mb:2} MB chunks {:6.0} ms  {:5.2} GB/s",
                 s * 1e3, (n * chunk) as f64 / s / 1e9);
    }

    // One destination buffer per chunk (what a per-weight upload does: every
    // weight gets its own buffer).
    for chunk_mb in [8usize, 16] {
        let chunk = chunk_mb * 1_000_000;
        let src = vec![0u8; chunk];
        let n = (total / chunk).min(24);
        let start = Instant::now();
        let mut bufs = Vec::new();
        for _ in 0..n {
            let b = make(chunk, "dst3");
            queue.write_buffer(&b, 0, &src);
            bufs.push(b);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let s = start.elapsed().as_secs_f64();
        println!("dst per chunk, {chunk_mb:2} MB chunks {:6.0} ms  {:5.2} GB/s",
                 s * 1e3, (n * chunk) as f64 / s / 1e9);
    }

    // Fresh destination per upload (what a one-shot weight upload does).
    for chunk_mb in [8usize, 16] {
        let chunk = chunk_mb * 1_000_000;
        let src = vec![0u8; chunk];
        let n = total / chunk;
        let start = Instant::now();
        let dst = make(total, "dst2");
        let mut off = 0usize;
        for _ in 0..n {
            queue.write_buffer(&dst, off as u64, &src);
            off += chunk;
        }
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let s = start.elapsed().as_secs_f64();
        println!("fresh dst,  {chunk_mb:2} MB chunks {:6.0} ms  {:5.2} GB/s",
                 s * 1e3, (n * chunk) as f64 / s / 1e9);
    }
    Ok(())
}
