//! PCIe upload bandwidth: how do wgpu's upload paths compare on this card?
//!
//! The codec's decoder wants ~460 MB of weights per run, so the transfer path
//! matters. Measured here: a reused buffer reaches ~6 GB/s, while a fresh
//! create + upload is ~2.5x slower. (In situ, under the run's resident-memory
//! pressure, a 75 MB fresh upload is slower still — see HANDOVER.)
//!
//!   cargo run --release --example upload_probe
use m5_tts_wgpu::device::Device;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let mb = 128usize;
    let len = mb * 1_000_000 / 4;
    let data: Vec<f32> = (0..len).map(|i| i as f32).collect();
    println!("{mb} MB per call, 4 calls");

    let time = |label: &str, f: &dyn Fn()| {
        f();
        gpu.sync();
        let start = Instant::now();
        for _ in 0..4 { f(); }
        gpu.sync();
        let secs = start.elapsed().as_secs_f64();
        println!("{label:28} {:6.0} ms  {:5.2} GB/s", secs * 1e3 / 4.0,
                 4.0 * (len * 4) as f64 / secs / 1e9);
    };

    time("create + write_buffer", &|| {
        let _b = gpu.empty("probe.wb", len);
        gpu.write(&_b, &data);
    });

    let mut pre = gpu.empty("probe.pre", len);
    time("write_buffer (preallocated)", &|| {
        gpu.write(&pre, &data);
    });
    let _ = &mut pre;

    // mapped at creation: the host writes straight into the buffer's memory.
    time("mapped_at_creation", &|| {
        use wgpu::util::DeviceExt;
        let _b = gpu.raw_device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe.init"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
    });
    Ok(())
}
