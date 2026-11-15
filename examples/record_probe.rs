//! Per-dispatch *client-side* cost of the three recording paths, with the GPU
//! forced to run behind (one huge submit, no waits). The fused depth decode
//! spends 21 ms/frame recording 720 dispatches = 29 us each; this says where.
//!
//!   cargo run --release --example record_probe
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::gpu_ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // SAFETY: single-threaded setup before anything is spawned.
    unsafe { std::env::set_var("M5_MAX_PENDING", "100000") };
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let (h, inter) = (512usize, 2048usize);
    let x = dev.tensor([1, h], &vec![0.1f32; h])?;
    let nw = dev.tensor([1, h], &vec![1.0f32; h])?;
    let y = dev.zeros([1, h])?;
    let g = dev.tensor([inter, h], &vec![0.01f32; inter * h])?;
    let u = dev.tensor([inter, h], &vec![0.01f32; inter * h])?;
    let gu = dev.zeros([1, 2 * inter])?;

    let bench = |label: &str, n: usize, f: &dyn Fn()| {
        for _ in 0..32 { f(); }
        gpu.sync();
        let start = Instant::now();
        for _ in 0..n { f(); }
        let secs = start.elapsed().as_secs_f64();
        gpu.sync();
        println!("{label:44} {:6.2} us/dispatch  ({} total)", secs * 1e6 / n as f64, n);
    };

    // 1. cached bind group + cached params (the fused depth path)
    let k = gpu.kernel_params("record.rms", m5_tts_wgpu::shaders::RMS_NORM_F32, "ro,ro,rw");
    bench("Kernel::run_params, same key (params cached)", 2000, &|| {
        k.run_params(gpu, 7, &[1, h as u32, (1e-5f32).to_bits(), 0],
                     &[x.gpu_buffer().unwrap(), nw.gpu_buffer().unwrap(), y.gpu_buffer().unwrap()],
                     (1, 1, 1));
    });

    // 2. cached bind group, params that change every call
    let flip = std::cell::Cell::new(0u32);
    bench("Kernel::run_params, changing params", 2000, &|| {
        flip.set(flip.get().wrapping_add(1));
        k.run_params(gpu, 8, &[1, h as u32, (1e-5f32).to_bits(), flip.get()],
                     &[x.gpu_buffer().unwrap(), nw.gpu_buffer().unwrap(), y.gpu_buffer().unwrap()],
                     (1, 1, 1));
    });

    // 3. the generic op path: params pool + a fresh bind group per dispatch
    bench("gpu_ops::rms_norm (bind group per call)", 2000, &|| {
        gpu_ops::rms_norm(gpu, x.gpu_buffer().unwrap(), nw.gpu_buffer().unwrap(),
                          y.gpu_buffer().unwrap(), 1, h, 1e-5);
    });

    // 4. a big kernel through the generic path
    bench("gpu_ops::norm_gateup (bind group per call)", 500, &|| {
        gpu_ops::norm_gateup_f32(gpu, x.gpu_buffer().unwrap(), nw.gpu_buffer().unwrap(),
                                 g.gpu_buffer().unwrap(), u.gpu_buffer().unwrap(),
                                 gu.gpu_buffer().unwrap(), inter, h, 1e-5);
    });

    // 5. allocation paths
    bench("Device::zeros([1, 512])", 2000, &|| {
        let _ = dev.zeros([1, h]).unwrap();
    });
    bench("Device::copy (1x512)", 2000, &|| {
        let _ = dev.copy(&x).unwrap();
    });
    bench("Device::tensor([1, 512]) upload", 2000, &|| {
        let _ = dev.tensor([1, h], &vec![0.5f32; h]).unwrap();
    });
    Ok(())
}
