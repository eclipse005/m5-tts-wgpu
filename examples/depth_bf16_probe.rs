//! Kernel-only cost of the fused bf16 depth layer step (6 dispatches, the
//! exact kernels `llama_layer_depth_fused` uses), back to back.
//!
//! The depth frame is ~19 ms for 720 dispatches; this separates "how fast are
//! the kernels" from "how much is dispatch overhead".
//!
//!   cargo run --release --example depth_bf16_probe
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::gpu_ops::{self, f_gemv, f_gemv_add, f_norm_gateup, f_qkv_attend, f_rms, f_silu_down};
use m5_tts_wgpu::shaders;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let (h, inter, qd, kvd) = (512usize, 2048usize, 512usize, 128usize);
    let pack = |n: usize| -> Vec<f32> { vec![f32::from_bits(0x3f803f80); n / 2] };
    let w_qkv = dev.tensor([(qd + 2 * kvd) * h / 2, 1], &pack((qd + 2 * kvd) * h))?;
    let w_o = dev.tensor([h * qd / 2, 1], &pack(h * qd))?;
    let w_gate = dev.tensor([inter * h / 2, 1], &pack(inter * h))?;
    let w_up = dev.tensor([inter * h / 2, 1], &pack(inter * h))?;
    let w_down = dev.tensor([h * inter / 2, 1], &pack(h * inter))?;
    let zero = dev.tensor([1, 1], &[0.0f32])?;
    let x_in = dev.tensor([1, h], &vec![0.1f32; h])?;
    let norm_w = dev.tensor([1, h], &vec![1.0f32; h])?;
    let normed = dev.zeros([1, h])?;
    let qkv = dev.zeros([1, qd + 2 * kvd])?;
    let attended = dev.zeros([1, qd])?;
    let x1 = dev.zeros([1, h])?;
    let gate_up = dev.zeros([1, 2 * inter])?;
    let x_out = dev.zeros([1, h])?;
    let cache_k = dev.zeros([18, qd])?;
    let cache_v = dev.zeros([18, qd])?;
    let cos = dev.tensor([2048, 64], &vec![1.0f32; 2048 * 64])?;
    let sin = dev.tensor([2048, 64], &vec![0.0f32; 2048 * 64])?;

    let rms = gpu.kernel_params("p.rms", shaders::RMS_NORM_F32, "ro,ro,rw");
    let gemv = gpu.kernel_params("p.gemv_bf16", shaders::GEMV_BF16_F32, "ro,ro,ro,rw");
    let attend = gpu.kernel_params("p.qkv_attend", shaders::QKV_ATTEND_F32, "ro,rw,rw,rw,ro,ro");
    let add = gpu.kernel_params("p.gemv_add_bf16", shaders::GEMV_ADD_BF16_F32, "ro,ro,ro,rw");
    let gateup = gpu.kernel_params("p.norm_gateup_bf16", shaders::NORM_GATEUP_BF16_F32, "ro,ro,ro,ro,rw");
    let down = gpu.kernel_params("p.silu_down_bf16", shaders::SILU_DOWN_BF16_F32, "ro,ro,ro,rw");

    let step = || {
        f_rms(&rms, gpu, 1, x_in.gpu_buffer().unwrap(), norm_w.gpu_buffer().unwrap(),
              normed.gpu_buffer().unwrap(), 1, h, 1e-5);
        f_gemv(&gemv, gpu, 2, normed.gpu_buffer().unwrap(), w_qkv.gpu_buffer().unwrap(),
               zero.gpu_buffer().unwrap(), qkv.gpu_buffer().unwrap(), qd + 2 * kvd, h);
        f_qkv_attend(&attend, gpu, 3, qkv.gpu_buffer().unwrap(), cache_k.gpu_buffer().unwrap(),
                     cache_v.gpu_buffer().unwrap(), attended.gpu_buffer().unwrap(),
                     cos.gpu_buffer().unwrap(), sin.gpu_buffer().unwrap(), 5, 8, qd, kvd);
        f_gemv_add(&add, gpu, 4, attended.gpu_buffer().unwrap(), w_o.gpu_buffer().unwrap(),
                   x_in.gpu_buffer().unwrap(), x1.gpu_buffer().unwrap(), h, qd);
        f_norm_gateup(&gateup, gpu, 5, x1.gpu_buffer().unwrap(), norm_w.gpu_buffer().unwrap(),
                      w_gate.gpu_buffer().unwrap(), w_up.gpu_buffer().unwrap(),
                      gate_up.gpu_buffer().unwrap(), inter, h, 1e-5);
        f_silu_down(&down, gpu, 6, gate_up.gpu_buffer().unwrap(), w_down.gpu_buffer().unwrap(),
                    x1.gpu_buffer().unwrap(), x_out.gpu_buffer().unwrap(), inter, h);
    };
    for _ in 0..20 { step(); }
    gpu.sync();
    let iters = 200;
    let start = Instant::now();
    for _ in 0..iters { step(); }
    gpu.sync();
    let per_step = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
    let bytes = ((qd + 2 * kvd) * h + h * qd + 2 * inter * h + h * inter) as f64 * 2.0;
    println!("one layer step (6 dispatch): {per_step:6.2} ms? no, {:6.1} us   {:.0} GB/s",
             per_step * 1e3, bytes / (per_step * 1e3));
    // Per-dispatch floor, same shape of work: 6 trivial dispatches.
    let trivial = gpu.kernel_params("p.rms2", shaders::RMS_NORM_F32, "ro,ro,rw");
    let tiny = dev.zeros([1, 8])?;
    let tiny_out = dev.zeros([1, 8])?;
    let tiny_w = dev.tensor([1, 8], &vec![1.0f32; 8])?;
    let floor = || {
        for _ in 0..6 {
            f_rms(&trivial, gpu, 1, tiny.gpu_buffer().unwrap(), tiny_w.gpu_buffer().unwrap(),
                  tiny_out.gpu_buffer().unwrap(), 1, 8, 1e-5);
        }
    };
    for _ in 0..50 { floor(); }
    gpu.sync();
    let start = Instant::now();
    for _ in 0..iters { floor(); }
    gpu.sync();
    let per_floor = start.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!("six trivial dispatches:      {:6.1} us   (so dispatch tax = {:.1} us each)",
             per_floor * 1e3, per_floor * 1e3 / 6.0);
    let _ = gpu_ops::copy;
    // Per-kernel breakdown of one step.
    let time_one = |label: &str, f: &dyn Fn()| {
        for _ in 0..20 { f(); }
        gpu.sync();
        let start = Instant::now();
        for _ in 0..200 { f(); }
        gpu.sync();
        println!("  {label:34} {:6.1} us", start.elapsed().as_secs_f64() * 1e3 / 200.0 * 1e3);
    };
    time_one("rms_norm 512", &|| f_rms(&rms, gpu, 1, x_in.gpu_buffer().unwrap(),
                                      norm_w.gpu_buffer().unwrap(), normed.gpu_buffer().unwrap(),
                                      1, h, 1e-5));
    time_one("gemv bf16 768x512 (qkv)", &|| f_gemv(&gemv, gpu, 2, normed.gpu_buffer().unwrap(),
                                                   w_qkv.gpu_buffer().unwrap(), zero.gpu_buffer().unwrap(),
                                                   qkv.gpu_buffer().unwrap(), qd + 2 * kvd, h));
    time_one("qkv_attend (8 heads, len 6)", &|| f_qkv_attend(&attend, gpu, 3, qkv.gpu_buffer().unwrap(),
                                                             cache_k.gpu_buffer().unwrap(), cache_v.gpu_buffer().unwrap(),
                                                             attended.gpu_buffer().unwrap(), cos.gpu_buffer().unwrap(),
                                                             sin.gpu_buffer().unwrap(), 5, 8, qd, kvd));
    time_one("gemv_add bf16 512x512 (o)", &|| f_gemv_add(&add, gpu, 4, attended.gpu_buffer().unwrap(),
                                                         w_o.gpu_buffer().unwrap(), x_in.gpu_buffer().unwrap(),
                                                         x1.gpu_buffer().unwrap(), h, qd));
    time_one("norm_gateup bf16 2x2048x512", &|| f_norm_gateup(&gateup, gpu, 5, x1.gpu_buffer().unwrap(),
                                                              norm_w.gpu_buffer().unwrap(), w_gate.gpu_buffer().unwrap(),
                                                              w_up.gpu_buffer().unwrap(), gate_up.gpu_buffer().unwrap(),
                                                              inter, h, 1e-5));
    time_one("silu_down bf16 512x2048", &|| f_silu_down(&down, gpu, 6, gate_up.gpu_buffer().unwrap(),
                                                        w_down.gpu_buffer().unwrap(), x1.gpu_buffer().unwrap(),
                                                        x_out.gpu_buffer().unwrap(), inter, h));
    Ok(())
}
