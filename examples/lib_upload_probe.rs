//! Same upload pattern through the *library's* Gpu as through raw wgpu, to find
//! why the codec's weight uploads run at ~1 GB/s while a standalone wgpu probe
//! measures 10-16 GB/s.
//!
//!   cargo run --release --example lib_upload_probe
use m5_tts_wgpu::device::Device;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let chunk = 16_000_000usize / 4; // f32 count
    let n = 24;
    let src = vec![0.5f32; chunk * 2];

    let run = |label: &str, f: &dyn Fn()| {
        gpu.sync();
        let before = gpu.upload_host_nanos.get();
        let start = Instant::now();
        f();
        gpu.sync();
        let secs = start.elapsed().as_secs_f64();
        let host = (gpu.upload_host_nanos.get() - before) as f64 / 1e9;
        println!("{label:34} wall {:6.0} ms ({:5.2} GB/s)   host {:6.0} ms ({:5.2} GB/s)",
                 secs * 1e3, (n * chunk * 4) as f64 / secs / 1e9,
                 host * 1e3, (n * chunk * 4) as f64 / host / 1e9);
    };

    // (a) into one preallocated buffer: 24 writes of one chunk each.
    let big = gpu.storage((n * chunk * 4) as u64, "probe.big");
    run("24 writes into one buffer", &|| {
        for i in 0..n { gpu.write(&big, &src[i % 2 * chunk..i % 2 * chunk + chunk]); }
    });

    // (b) one fresh buffer per write (the weight-upload pattern).
    run("24 writes, fresh buffer each", &|| {
        for i in 0..n {
            let b = gpu.storage((chunk * 4) as u64, "probe.one");
            gpu.write(&b, &src[i % 2 * chunk..i % 2 * chunk + chunk]);
            std::mem::forget(b);
        }
    });

    // (c) same, but with a dispatchery batch open in between.
    let x = dev.tensor([1, 512], &vec![0.1f32; 512])?;
    let y = dev.zeros([1, 512])?;
    let w = dev.tensor([1, 512], &vec![1.0f32; 512])?;
    run("24 writes + dispatches between", &|| {
        for _ in 0..n {
            let b = gpu.storage((chunk * 4) as u64, "probe.two");
            gpu.write(&b, &src[..chunk]);
            m5_tts_wgpu::gpu_ops::add(gpu, x.gpu_buffer().unwrap(), w.gpu_buffer().unwrap(),
                                      y.gpu_buffer().unwrap(), 512);
            std::mem::forget(b);
        }
    });
    // (d) fresh buffer per write, queue drained before each (GPU idle).
    run("24 writes, sync before each", &|| {
        for i in 0..n {
            gpu.sync();
            let b = gpu.storage((chunk * 4) as u64, "probe.four");
            gpu.write(&b, &src[i % 2 * chunk..i % 2 * chunk + chunk]);
            std::mem::forget(b);
        }
    });

    // (e) fresh buffer per write, but submitted (not waited) right away.
    run("24 writes, flush after each", &|| {
        for i in 0..n {
            let b = gpu.storage((chunk * 4) as u64, "probe.five");
            gpu.write(&b, &src[i % 2 * chunk..i % 2 * chunk + chunk]);
            gpu.flush();
            std::mem::forget(b);
        }
    });
    // (f) the codec's actual situation: dispatches recorded, then a weight
    // upload - but with a submit (no wait) first, so the batch is empty.
    run("dispatch + flush, then write", &|| {
        for i in 0..n {
            m5_tts_wgpu::gpu_ops::add(gpu, x.gpu_buffer().unwrap(), w.gpu_buffer().unwrap(),
                                      y.gpu_buffer().unwrap(), 512);
            m5_tts_wgpu::gpu_ops::add(gpu, x.gpu_buffer().unwrap(), w.gpu_buffer().unwrap(),
                                      y.gpu_buffer().unwrap(), 512);
            gpu.flush();
            let b = gpu.storage((chunk * 4) as u64, "probe.six");
            gpu.write(&b, &src[i % 2 * chunk..i % 2 * chunk + chunk]);
            std::mem::forget(b);
        }
    });
    // (g) the same uploads with a large host footprint, like a process that has
    // loaded the codec's 682 MB of weights into RAM.
    let hog: Vec<Vec<u8>> = (0..7).map(|_| vec![1u8; 100 * 1_000_000]).collect();
    println!("  (700 MB host resident)");
    run("24 writes, fresh buffer each", &|| {
        for i in 0..n {
            let b = gpu.storage((chunk * 4) as u64, "probe.seven");
            gpu.write(&b, &src[i % 2 * chunk..i % 2 * chunk + chunk]);
            std::mem::forget(b);
        }
    });
    drop(hog);
    Ok(())
}
