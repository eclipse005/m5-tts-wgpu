//! The phased transposed conv at each decoder block's real shape, split into its
//! four steps, to find where block 1's 180 ms goes.
//!
//!   cargo run --release --example tconv_gpu_probe
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::gpu_ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    // (cin, cout, stride, t_in) for the four decoder blocks (rates 8, 5, 4, 3).
    for (cin, cout, stride, t_in) in
        [(1536usize, 768usize, 8usize, 98usize), (768, 384, 5, 784), (384, 192, 4, 3920),
         (192, 96, 3, 15680)]
    {
        let taps = 2usize;
        let rows = cout * stride;
        let x = dev.tensor([cin, t_in], &vec![0.01f32; cin * t_in])?;
        let w = dev.tensor([taps * cin, rows], &vec![0.01f32; taps * cin * rows])?;
        let b = dev.tensor([rows, 1], &vec![0.001f32; rows])?;
        let x_cl = gpu.empty("probe.x_cl", t_in * cin);
        let y_cl = gpu.empty("probe.y_cl", t_in * rows);
        let macs = (rows * t_in * taps * cin) as f64;
        let steps: [(&str, Box<dyn Fn()>); 3] = [
            ("transpose in", Box::new(|| gpu_ops::transpose(gpu, x.gpu_buffer().unwrap(), &x_cl, cin, t_in))),
            ("conv (reg8)", Box::new(|| gpu_ops::conv1d_reg8(gpu, &x_cl, w.gpu_buffer().unwrap(),
                                                             b.gpu_buffer().unwrap(), &y_cl,
                                                             cin, rows, taps, 1, 1, taps - 1,
                                                             t_in, t_in))),
            ("interleave", Box::new(|| {
                let out = dev.zeros([cout, t_in * stride]).unwrap();
                gpu_ops::phased_interleave(gpu, &y_cl, out.gpu_buffer().unwrap(), rows, stride, t_in);
            })),
        ];
        println!("cin {cin} cout {cout} stride {stride} t_in {t_in}  rows {rows}  {:.1} GFLOP",
                 2.0 * macs / 1e9);
        for (label, step) in &steps {
            step();
            gpu.sync();
            let start = Instant::now();
            let iters = 5;
            for _ in 0..iters { step(); }
            gpu.sync();
            let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
            println!("   {label:14} {ms:7.2} ms  {:6.0} GFLOP/s", 2.0 * macs / (ms * 1e-3) / 1e9);
        }
    }
    Ok(())
}
