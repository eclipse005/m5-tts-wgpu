//! Why do weight uploads run at 0.85 GB/s inside a real run when a clean
//! process measures 2.5-5.9 GB/s? Separates the candidates: resident-VRAM
//! pressure, buffer freshness, staging-buffer reuse, and the upload API.
//!
//!   cargo run --release --example upload_probe2
use m5_tts_wgpu::device::Device;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let len = 75_000_000 / 4; // one block-1 transposed-conv weight
    let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
    println!("75 MB per upload, 4 uploads per case");

    let time = |label: &str, f: &dyn Fn()| {
        f();
        gpu.sync();
        let start = Instant::now();
        for _ in 0..4 {
            f();
            gpu.sync();
        }
        let secs = start.elapsed().as_secs_f64();
        println!("{label:36} {:6.0} ms  {:5.2} GB/s", secs * 1e3 / 4.0,
                 4.0 * (len * 4) as f64 / secs / 1e9);
    };

    // (a) clean: fresh buffer per upload.
    time("fresh buffer, clean device", &|| {
        let b = gpu.empty("p.a", len);
        gpu.write(&b, &data);
    });

    // Reserve the buffers we will need *before* the device fills up, then write
    // into them later: is the cost in creation or in the write?
    let reserved: Vec<wgpu::Buffer> =
        (0..4).map(|i| gpu.empty(&format!("p.res{i}"), len)).collect();
    time("pre-created buffers, clean", &|| {
        for b in &reserved {
            gpu.write(b, &data);
        }
    });

    // (b) 1.6 GB resident, fresh buffer per upload.
    let hog: Vec<_> = (0..8)
        .map(|_| dev.zeros([50_000_000, 1]))
        .collect::<anyhow::Result<Vec<_>>>()?;
    gpu.sync();
    println!("  (1600 MB resident)");
    time("fresh buffer, 1.6 GB resident", &|| {
        let b = gpu.empty("p.b", len);
        gpu.write(&b, &data);
    });

    // (c) buffers created before the pressure, written under it.
    time("pre-created buffers, resident", &|| {
        for b in &reserved {
            gpu.write(b, &data);
        }
    });

    // (d) mapped at creation under pressure (the handover notes this path used
    // to fail with "Buffer is invalid" at get_mapped_range).
    time("mapped_at_creation, resident", &|| {
        use wgpu::util::DeviceExt;
        let b = gpu.raw_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p.init"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        std::mem::forget(b);
    });

    drop(hog);
    Ok(())
}
