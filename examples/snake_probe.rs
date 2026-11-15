//! Why is the GPU SnakeBeta pass at 5-9 GB/s while a plain copy is near DRAM
//! speed? Times the same in-place pass with and without the sine, at the
//! decoder's block-4 shape, next to a copy of the same bytes.
//!
//!   cargo run --release --example snake_probe
use m5_tts_wgpu::device::Device;
use std::time::Instant;

const SNAKE_NOSIN: &str = r#"
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32, 2>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 128u;
    if (idx >= p[0] * p[1]) { return; }
    let c = idx / p[1];
    let v = x[idx];
    let s = v * alpha[c];
    x[idx] = v + s * s / beta[c];
}
"#;

const SNAKE_POLY: &str = r#"
// sin via a degree-7 odd polynomial on the reduced range (|arg| is small here).
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> alpha: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> p: array<u32, 2>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 128u;
    if (idx >= p[0] * p[1]) { return; }
    let c = idx / p[1];
    let v = x[idx];
    let s = sin(v * alpha[c]);
    x[idx] = v + s * s / beta[c];
}
"#;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let (channels, time) = (96usize, 188_000usize);
    let n = channels * time;
    println!("shape {channels} x {time} = {:.1} MB", (n * 4) as f64 / 1e6);
    let alpha = dev.tensor([channels, 1], &vec![0.7f32; channels])?;
    let beta = dev.tensor([channels, 1], &vec![0.9f32; channels])?;
    let x = dev.tensor([channels, time], &vec![0.3f32; n])?;
    let dst = std::cell::RefCell::new(dev.zeros([channels, time])?);

    let time_it = |label: &str, f: &dyn Fn()| {
        for _ in 0..3 { f(); }
        gpu.sync();
        let start = Instant::now();
        let iters = 10;
        for _ in 0..iters { f(); }
        gpu.sync();
        let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!("{label:28} {ms:8.2} ms   {:6.1} GB/s (read+write)",
                 (2 * n * 4) as f64 / (ms * 1e6));
    };

    time_it("copy (dev.copy, sync)", &|| {
        *dst.borrow_mut() = dev.copy(&x).unwrap();
    });
    time_it("snake libm sin", &|| {
        m5_tts_wgpu::gpu_ops::snake_beta(gpu, x.gpu_buffer().unwrap(),
                                         alpha.gpu_buffer().unwrap(),
                                         beta.gpu_buffer().unwrap(), channels, time);
    });
    time_it("snake no sin", &|| {
        gpu.dispatch_params("snake_nosin", SNAKE_NOSIN, "main",
                            &[(0, x.gpu_buffer().unwrap()), (1, alpha.gpu_buffer().unwrap()),
                              (2, beta.gpu_buffer().unwrap())],
                            &gpu.params_u32(&[channels as u32, time as u32]),
                            { let groups = n.div_ceil(128);
                              ((groups.min(65535)) as u32, groups.div_ceil(65535) as u32, 1) });
    });
    Ok(())
}
