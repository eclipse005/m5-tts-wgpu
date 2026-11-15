//! Time one full GPU depth frame (15 codebooks, 5 layers each) in isolation and
//! report the dispatch/submit counts it costs.
//!
//! Answers: is the handover's "depth 65 ms/frame, layer kernels only 11 ms"
//! gap in the fused kernels, the per-codebook ops (audio_head GEMV, sampler,
//! embed+projection), or in submit/record overhead?
//!
//!   cargo run --release --example depth_frame_probe [frames]
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::lm::{Lm, AUDIO_VOCAB, CODEBOOKS, HIDDEN};
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let frames: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(10);
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let lm = Lm::load(Path::new("D:/M5-TTS/weights/M5-TTS-50M-VoiceClone"), dev.shared())?;

    let last_h = dev.tensor([1, HIDDEN], &vec![0.05f32; HIDDEN])?;
    let noise_flat: Vec<f32> = (0..CODEBOOKS * AUDIO_VOCAB).map(|i| 0.3 + (i % 97) as f32 * 0.01).collect();
    let noise = dev.tensor([CODEBOOKS, AUDIO_VOCAB], &noise_flat)?;

    let run = |label: &str, f: &dyn Fn() -> anyhow::Result<Vec<u32>>| -> anyhow::Result<()> {
        f()?;
        f()?;
        gpu.sync();
        let before = gpu.snapshot();
        let start = Instant::now();
        for _ in 0..frames {
            f()?;
        }
        gpu.sync();
        let elapsed = start.elapsed().as_secs_f64();
        let after = gpu.snapshot();
        println!(
            "{label:10} {:7.2} ms/frame   {} dispatch  {} submit  {} per frame",
            elapsed * 1e3 / frames as f64,
            (after.dispatches - before.dispatches) / frames as u64,
            (after.submits - before.submits) / frames as u64,
            (after.dispatches - before.dispatches) / frames as u64,
        );
        Ok(())
    };

    run("default", &|| lm.depth_codes_device(&last_h, &noise, 0.9, 10, true))?;
    run("fused", &|| lm.depth_codes_device_fused(&last_h, &noise, 0.9, 10, true))?;

    // The per-codebook projection the depth loop runs 15 times a frame:
    // [1, 512] @ [2051, 512]^T, i.e. a 4.2 MB warp-per-row GEMV.
    let hidden = dev.tensor([1, HIDDEN], &vec![0.05f32; HIDDEN])?;
    let head = dev.tensor([AUDIO_VOCAB, HIDDEN], &vec![0.001f32; AUDIO_VOCAB * HIDDEN])?;
    let mut logits = dev.zeros([1, AUDIO_VOCAB])?;
    let gemv = || {
        m5_tts_wgpu::gpu_ops::linear(gpu, hidden.gpu_buffer().unwrap(), head.gpu_buffer().unwrap(),
                                     None, logits.gpu_buffer().unwrap(), 1, AUDIO_VOCAB, HIDDEN);
    };
    gemv();
    gpu.sync();
    let start = Instant::now();
    for _ in 0..frames {
        gemv();
    }
    gpu.sync();
    println!("head gemv  {:7.3} ms/frame", start.elapsed().as_secs_f64() * 1e3 / frames as f64);

    // Same GEMV but with the two per-call allocations audio_head_logits makes
    // (a 2 KB row copy and an 8 KB logits buffer), to separate kernel cost from
    // allocation/pool cost.
    let alloc_gemv = || -> anyhow::Result<()> {
        let last = dev.copy(&hidden)?;
        let mut out = dev.zeros([1, AUDIO_VOCAB])?;
        m5_tts_wgpu::gpu_ops::linear(gpu, last.gpu_buffer().unwrap(), head.gpu_buffer().unwrap(),
                                     None, out.gpu_buffer().unwrap(), 1, AUDIO_VOCAB, HIDDEN);
        Ok(())
    };
    alloc_gemv()?;
    gpu.sync();
    let start = Instant::now();
    for _ in 0..frames {
        alloc_gemv()?;
    }
    gpu.sync();
    println!("head+alloc {:7.3} ms/frame", start.elapsed().as_secs_f64() * 1e3 / frames as f64);
    // The sampler on its own: 16 calls a frame in the depth loop.
    let tokens: m5_tts_wgpu::device::Tokens = dev.tokens(&[0u32; CODEBOOKS])?;
    let tok_buf = tokens.buffer.clone().expect("device tokens");
    let sample_once = || {
        m5_tts_wgpu::gpu_ops::sample_topk_offset(gpu, logits.gpu_buffer().unwrap(),
                                                 noise.gpu_buffer().unwrap(), &tok_buf,
                                                 AUDIO_VOCAB, 10, 0.9, true, 0, 0);
    };
    sample_once();
    gpu.sync();
    let start = Instant::now();
    for _ in 0..frames {
        sample_once();
    }
    gpu.sync();
    println!("sample      {:7.3} ms/frame", start.elapsed().as_secs_f64() * 1e3 / frames as f64);
    let start = Instant::now();
    for _ in 0..frames {
        sample_once();
        gpu.sync();
    }
    println!("sample+sync {:7.3} ms/frame", start.elapsed().as_secs_f64() * 1e3 / frames as f64);
    let _ = &mut logits;
    Ok(())
}
