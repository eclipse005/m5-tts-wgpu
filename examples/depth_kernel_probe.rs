//! Time each fused decode kernel at depth shapes, repeated back-to-back, to find
//! which kernel dominates the GPU depth pass.
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::gpu_ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let (h, inter, qd, kvd) = (512usize, 2048usize, 512usize, 128usize);
    let x = dev.tensor([1, h], &vec![0.1f32; h])?;
    let nw = dev.tensor([1, h], &vec![1.0f32; h])?;
    let qkv_w = dev.tensor([qd + 2*kvd, h], &vec![0.01f32; (qd+2*kvd)*h])?;
    let o_w = dev.tensor([h, qd], &vec![0.01f32; h*qd])?;
    let g_w = dev.tensor([inter, h], &vec![0.01f32; inter*h])?;
    let u_w = dev.tensor([inter, h], &vec![0.01f32; inter*h])?;
    let d_w = dev.tensor([h, inter], &vec![0.01f32; h*inter])?;
    let zero = dev.tensor([1,1], &[0.0])?;
    let normed = dev.zeros([1, h])?;
    let qkv = dev.zeros([1, qd+2*kvd])?;
    let attended = dev.zeros([1, qd])?;
    let x1 = dev.zeros([1, h])?;
    let gate_up = dev.zeros([1, 2*inter])?;
    let xout = dev.zeros([1, h])?;
    let ck = dev.zeros([18, qd])?;
    let cv = dev.zeros([18, qd])?;
    let cos = dev.tensor([2048, 64], &vec![1.0f32; 2048*64])?;
    let sin = dev.tensor([2048, 64], &vec![0.0f32; 2048*64])?;
    let iters = 1000;
    let mut time = |name: &str, f: &dyn Fn()| {
        for _ in 0..50 { f(); }
        gpu.sync();
        let s = Instant::now();
        for _ in 0..iters { f(); }
        gpu.sync();
        println!("{name:24} {:.1} us", s.elapsed().as_secs_f64()*1e6/iters as f64);
    };
    time("rms", &|| gpu_ops::rms_norm(gpu, x.gpu_buffer().unwrap(), nw.gpu_buffer().unwrap(), normed.gpu_buffer().unwrap(), 1, h, 1e-5));
    time("qkv gemv 768x512", &|| gpu_ops::linear(gpu, normed.gpu_buffer().unwrap(), qkv_w.gpu_buffer().unwrap(), None, qkv.gpu_buffer().unwrap(), 1, qd+2*kvd, h));
    time("qkv_attend", &|| gpu_ops::qkv_attend_f32(gpu, qkv.gpu_buffer().unwrap(), ck.gpu_buffer().unwrap(), cv.gpu_buffer().unwrap(), attended.gpu_buffer().unwrap(), cos.gpu_buffer().unwrap(), sin.gpu_buffer().unwrap(), 5, 8, qd));
    time("gemv_add 512x512", &|| gpu_ops::gemv_add_f32(gpu, attended.gpu_buffer().unwrap(), o_w.gpu_buffer().unwrap(), x.gpu_buffer().unwrap(), x1.gpu_buffer().unwrap(), h, qd));
    time("norm_gateup 4096x512", &|| gpu_ops::norm_gateup_f32(gpu, x1.gpu_buffer().unwrap(), nw.gpu_buffer().unwrap(), g_w.gpu_buffer().unwrap(), u_w.gpu_buffer().unwrap(), gate_up.gpu_buffer().unwrap(), inter, h, 1e-5));
    time("silu_down 512x2048", &|| gpu_ops::silu_down_f32(gpu, gate_up.gpu_buffer().unwrap(), d_w.gpu_buffer().unwrap(), x1.gpu_buffer().unwrap(), xout.gpu_buffer().unwrap(), inter, h));
    Ok(())
}