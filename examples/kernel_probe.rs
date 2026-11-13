//! Micro-benchmark the fused decode kernels in isolation.
use m5_tts_wgpu::cpu_ops;
use std::time::Instant;

fn main() {
    let (hidden, inter) = (512usize, 2048usize);
    let x = vec![0.1f32; hidden];
    let norm_w = vec![1.0f32; hidden];
    let qkv_w = vec![0.01f32; (hidden + 256) * hidden];
    let gate_up_w = vec![0.01f32; 2 * inter * hidden];
    let mut out = vec![0.0f32; 2 * inter];
    let calls = 50;

    let start = Instant::now();
    for _ in 0..calls {
        cpu_ops::rms_norm_gemv(&x, &norm_w, &qkv_w, None, &mut out[..hidden + 256], hidden, hidden + 256, 1e-5);
    }
    println!("rms_norm_gemv 768x512:  {:.1} us", start.elapsed().as_secs_f64() * 1e6 / calls as f64);

    let start = Instant::now();
    for _ in 0..calls {
        cpu_ops::rms_norm_gemv(&x, &norm_w, &gate_up_w, None, &mut out, hidden, 2 * inter, 1e-5);
    }
    println!("rms_norm_gemv 4096x512: {:.1} us", start.elapsed().as_secs_f64() * 1e6 / calls as f64);

    let start = Instant::now();
    for _ in 0..calls {
        let mut y = vec![0.0f32; hidden];
        cpu_ops::gemv_add(&x, &qkv_w[..hidden * hidden], &x, &mut y, hidden, hidden);
    }
    println!("gemv_add 512x512:       {:.1} us", start.elapsed().as_secs_f64() * 1e6 / calls as f64);
}
