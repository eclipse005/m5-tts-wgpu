//! Micro-benchmark: per-op cost inside one LM frame.
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::lm::backbone::KvCache;
use m5_tts_wgpu::lm::{AUDIO_VOCAB, HIDDEN, Lm};
use m5_tts_wgpu::ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-50M-VoiceClone");
    let lm = Lm::load(dir, Device::Cpu)?;
    let dev = lm.device();
    let x = dev.tensor([1, HIDDEN], &vec![0.01f32; HIDDEN])?;
    let w = lm.tensor("codebook0_head.weight", [AUDIO_VOCAB, HIDDEN])?;
    let mut y = dev.zeros([1, AUDIO_VOCAB])?;
    ops::linear(dev, &x, &w, None, &mut y)?;
    let n = 200;
    let start = Instant::now();
    for _ in 0..n { ops::linear(dev, &x, &w, None, &mut y)?; }
    println!("linear m=1 n=2051 k=512: {:.1} us/call", start.elapsed().as_secs_f64() * 1e6 / n as f64);

    let w2 = lm.tensor("projection.weight", [HIDDEN, HIDDEN])?;
    let mut y2 = dev.zeros([1, HIDDEN])?;
    let start = Instant::now();
    for _ in 0..n { ops::linear(dev, &x, &w2, None, &mut y2)?; }
    println!("linear m=1 n=512 k=512: {:.1} us/call", start.elapsed().as_secs_f64() * 1e6 / n as f64);

    // One full depth step: 15 codebooks.
    let h = dev.tensor([1, HIDDEN], &vec![0.02f32; HIDDEN])?;
    let noise: Vec<Vec<f32>> = (0..16).map(|_| vec![0.5f32; AUDIO_VOCAB]).collect();
    let start = Instant::now();
    let calls = 20;
    for _ in 0..calls { let _ = lm.depth_codes(&h, 0, &noise, 0.9, 10, true)?; }
    println!("depth_codes (15 codebooks): {:.1} ms/call", start.elapsed().as_secs_f64() * 1e3 / calls as f64);

    // A single backbone step.
    let mut kv = KvCache::new(dev, 8, 128)?;
    let _ = lm.backbone_prefill(&dev.tensor([2, HIDDEN], &vec![0.01f32; 2 * HIDDEN])?, &mut kv)?;
    let start = Instant::now();
    for i in 0..calls { let _ = lm.backbone_step(&x, &mut kv, 2 + i)?; }
    println!("backbone_step (8 layers): {:.1} ms/call", start.elapsed().as_secs_f64() * 1e3 / calls as f64);
    Ok(())
}
