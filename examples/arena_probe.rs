//! Isolate the uniform-arena kernel mechanism: run af_rms / af_gemv on known
//! inputs and compare against a host reference. If these are wrong, the arena
//! params/bind-group plumbing is broken; if right, the depth bug is elsewhere.
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::gpu_ops;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let n = 512usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let w: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.001)).collect();
    let xt = dev.tensor([1, n], &x)?;
    let wt = dev.tensor([1, n], &w)?;
    let yt = dev.zeros([1, n])?;

    let rms = gpu.arena_kernel("probe.rms", m5_tts_wgpu::shaders::AF_RMS, "ro,ro,rw");
    // 200 dispatches, same key (params repeat within a batch on purpose).
    for _ in 0..200 {
        gpu_ops::af_rms(&rms, gpu, 7, xt.gpu_buffer()?, wt.gpu_buffer()?, yt.gpu_buffer()?, 1, n, 1e-5);
    }
    let got = dev.read(&yt)?;
    let mut ss = 0.0f32;
    for v in &x { ss += v * v; }
    let inv = 1.0 / (ss / n as f32 + 1e-5).sqrt();
    let mut maxerr = 0.0f32;
    for i in 0..n {
        let want = x[i] * inv * w[i];
        maxerr = maxerr.max((got[i] - want).abs());
    }
    println!("af_rms maxerr = {maxerr:.3e}  (got[0]={:.6} want[0]={:.6})", got[0], x[0]*inv*w[0]);

    // af_gemv: y = W x, W [64, 512]
    let m = 64usize;
    let wmat: Vec<f32> = (0..m*n).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let wt2 = dev.tensor([m, n], &wmat)?;
    let bias = dev.tensor([1, 1], &[0.0])?;
    let yt2 = dev.zeros([1, m])?;
    let gemv = gpu.arena_kernel("probe.gemv", m5_tts_wgpu::shaders::AF_GEMV, "ro,ro,ro,rw");
    for _ in 0..200 {
        gpu_ops::af_gemv(&gemv, gpu, 9, xt.gpu_buffer()?, wt2.gpu_buffer()?, bias.gpu_buffer()?, yt2.gpu_buffer()?, m, n);
    }
    let got2 = dev.read(&yt2)?;
    let mut maxerr2 = 0.0f32;
    for r in 0..m {
        let mut acc = 0.0f32;
        for c in 0..n { acc += x[c] * wmat[r*n+c]; }
        maxerr2 = maxerr2.max((got2[r] - acc).abs());
    }
    println!("af_gemv maxerr = {maxerr2:.3e}  (got[0]={:.6} want[0]={:.6})", got2[0], {
        let mut a=0.0; for c in 0..n { a+=x[c]*wmat[c]; } a
    });

    // Mimic depth: af_rms reads a FRESH buffer written by a regular linear op.
    let fresh = gpu.empty("probe.fresh", n);
    let src = dev.tensor([1, n], &x)?;
    let ident = dev.tensor([n, n], &{
        let mut v = vec![0.0f32; n*n];
        for i in 0..n { v[i*n+i] = 1.0; }
        v
    })?;
    let bias0 = dev.tensor([1,1], &[0.0])?;
    let check = |name: &str, got: &[f32]| {
        let mut me = 0.0f32;
        for i in 0..n { me = me.max((got[i] - x[i]*inv*w[i]).abs()); }
        println!("{name}: maxerr={me:.3e} got[1]={:.6} want[1]={:.6}", got[1], x[1]*inv*w[1]);
    };
    // (a) fresh uploaded via write_buffer
    let fresh_up = gpu.empty("probe.fresh_up", n);
    gpu.write(&fresh_up, &x);
    let ya = dev.zeros([1, n])?;
    for _ in 0..50 { gpu_ops::af_rms(&rms, gpu, 8, &fresh_up, wt.gpu_buffer()?, ya.gpu_buffer()?, 1, n, 1e-5); }
    check("af_rms fresh+write_buffer", &dev.read(&ya)?);
    // (b) fresh written by regular linear (synced)
    let fresh = gpu.empty("probe.fresh", n);
    gpu_ops::linear(gpu, src.gpu_buffer()?, ident.gpu_buffer()?, Some(bias0.gpu_buffer()?), &fresh, 1, n, n);
    gpu.sync();
    let yb = dev.zeros([1, n])?;
    for _ in 0..50 { gpu_ops::af_rms(&rms, gpu, 9, &fresh, wt.gpu_buffer()?, yb.gpu_buffer()?, 1, n, 1e-5); }
    check("af_rms fresh+linear(sync)", &dev.read(&yb)?);
    // (c) regular rms_norm reading the same fresh+linear buffer
    let yc = dev.zeros([1, n])?;
    for _ in 0..50 { gpu_ops::rms_norm(gpu, &fresh, wt.gpu_buffer()?, yc.gpu_buffer()?, 1, n, 1e-5); }
    check("regular rms fresh+linear", &dev.read(&yc)?);
    // (d) af_rms pooled input -> FRESH output
    let yd = gpu.empty("probe.yd", n);
    for _ in 0..50 { gpu_ops::af_rms(&rms, gpu, 10, xt.gpu_buffer()?, wt.gpu_buffer()?, &yd, 1, n, 1e-5); }
    check("af_rms pooled->fresh out", &gpu.download(&yd, n));
    Ok(())
}