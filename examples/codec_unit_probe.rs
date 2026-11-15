//! What a decoder residual unit really costs at block 4 (96 ch x 188k samples),
//! op by op and then chained, to separate kernel time from the transposes.
//!
//!   cargo run --release --example codec_unit_probe
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::gpu_ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let (ch, t) = (96usize, 188_000usize);
    let mb = (ch * t * 4) as f64 / 1e6;
    println!("{ch} ch x {t} samples = {mb:.0} MB per tensor");

    let x = dev.tensor([ch, t], &vec![0.1f32; ch * t])?;
    let alpha = dev.tensor([ch, 1], &vec![0.5f32; ch])?;
    let beta = dev.tensor([ch, 1], &vec![0.7f32; ch])?;
    let x_cl = {
        let raw = gpu.empty("probe.x_cl", t * ch);
        drop(raw);
        gpu.empty("probe.x_cl", t * ch)
    };
    let y_cl = gpu.empty("probe.y_cl", t * ch);
    let out = dev.zeros([ch, t])?;
    let b = dev.tensor([ch, 1], &vec![0.01f32; ch])?;
    let w7 = dev.tensor([7 * ch, ch], &vec![0.01f32; 7 * ch * ch])?;
    let w1 = dev.tensor([ch, ch], &vec![0.01f32; ch * ch])?;

    let bench = |label: &str, f: &dyn Fn()| {
        for _ in 0..2 { f(); }
        gpu.sync();
        let start = Instant::now();
        let iters = 5;
        for _ in 0..iters { f(); }
        gpu.sync();
        let ms = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!("{label:38} {ms:7.2} ms   {:6.1} GB/s (2 passes)",
                 (2.0 * mb) / (ms as f64 / 1e3) / 1e3);
    };

    bench("transpose [ch,T] -> [T,ch]", &|| {
        gpu_ops::transpose(gpu, x.gpu_buffer().unwrap(), &x_cl, ch, t);
    });
    bench("transpose [T,ch] -> [ch,T]", &|| {
        gpu_ops::transpose(gpu, &x_cl, out.gpu_buffer().unwrap(), t, ch);
    });
    let snakebuf = std::cell::RefCell::new(dev.copy(&x)?);
    bench("snake (copy + kernel)", &|| {
        *snakebuf.borrow_mut() = dev.copy(&x).unwrap();
        gpu_ops::snake_beta(gpu, snakebuf.borrow().gpu_buffer().unwrap(),
                            alpha.gpu_buffer().unwrap(), beta.gpu_buffer().unwrap(), ch, t);
    });
    bench("add (out + x)", &|| {
        gpu_ops::add(gpu, out.gpu_buffer().unwrap(), x.gpu_buffer().unwrap(),
                     snakebuf.borrow().gpu_buffer().unwrap(), ch * t);
    });
    bench("conv1d_reg8 ks7 dil1", &|| {
        gpu_ops::conv1d_reg8(gpu, &x_cl, w7.gpu_buffer().unwrap(), b.gpu_buffer().unwrap(),
                             &y_cl, ch, ch, 7, 1, 1, 6, t, t);
    });
    bench("conv1d_reg8 ks1", &|| {
        gpu_ops::conv1d_reg8(gpu, &x_cl, w1.gpu_buffer().unwrap(), b.gpu_buffer().unwrap(),
                             &y_cl, ch, ch, 1, 1, 1, 0, t, t);
    });

    // The unit as the codec runs it: snake, transpose+conv+transpose, snake, ... , add.
    let unit = || {
        let mut h = dev.copy(&x).unwrap();
        gpu_ops::snake_beta(gpu, h.gpu_buffer().unwrap(), alpha.gpu_buffer().unwrap(),
                            beta.gpu_buffer().unwrap(), ch, t);
        let h3 = dev.zeros([ch, t]).unwrap();
        let a = gpu.empty("u.a", ch * t);
        let bb = gpu.empty("u.b", ch * t);
        gpu_ops::transpose(gpu, h.gpu_buffer().unwrap(), &a, ch, t);
        gpu_ops::conv1d_reg8(gpu, &a, w7.gpu_buffer().unwrap(), b.gpu_buffer().unwrap(), &bb,
                             ch, ch, 7, 1, 1, 6, t, t);
        gpu_ops::transpose(gpu, &bb, h3.gpu_buffer().unwrap(), t, ch);
        let h4 = dev.copy(&h3).unwrap();
        gpu_ops::snake_beta(gpu, h4.gpu_buffer().unwrap(), alpha.gpu_buffer().unwrap(),
                            beta.gpu_buffer().unwrap(), ch, t);
        let o = dev.zeros([ch, t]).unwrap();
        let c = gpu.empty("u.c", ch * t);
        let d = gpu.empty("u.d", ch * t);
        gpu_ops::transpose(gpu, h4.gpu_buffer().unwrap(), &c, ch, t);
        gpu_ops::conv1d_reg8(gpu, &c, w1.gpu_buffer().unwrap(), b.gpu_buffer().unwrap(), &d,
                             ch, ch, 1, 1, 1, 0, t, t);
        gpu_ops::transpose(gpu, &d, o.gpu_buffer().unwrap(), t, ch);
        let s = dev.zeros([ch, t]).unwrap();
        gpu_ops::add(gpu, o.gpu_buffer().unwrap(), x.gpu_buffer().unwrap(),
                     s.gpu_buffer().unwrap(), ch * t);
    };
    bench("whole unit (10 dispatches)", &unit);
    Ok(())
}
