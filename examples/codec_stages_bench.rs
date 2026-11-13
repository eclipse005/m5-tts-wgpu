//! Per-stage codec timing, so optimisation targets the dominant stage.
use m5_tts_wgpu::codec::Codec;
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::ops;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-Audio-Tokenizer");
    let codes: Vec<u32> = std::fs::read("goldens/acoustic/codes.bin")?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u32)
        .collect();
    let frames = codes.len() / 16;
    for spec in ["cpu", "gpu"] {
        let dev = Device::open(spec)?;
        let codec = Codec::load(dir, dev, 24000)?;
        let dev = codec.device();
        let t = Instant::now();
        let q = codec.dequantize(&codes, frames)?;
        let t_q = t.elapsed().as_secs_f64();
        let pre = codec.pre_transformer_input(&codes, frames)?;
        let t_pre = t.elapsed().as_secs_f64() - t_q;
        let tr = codec.transformer(&pre)?;
        let t_tr = t.elapsed().as_secs_f64() - t_q - t_pre;
        let mut ch = dev.zeros([1024, frames])?;
        ops::transpose(dev, &tr, &mut ch)?;
        let up = codec.upsample(&ch)?;
        let t_up = t.elapsed().as_secs_f64() - t_q - t_pre - t_tr;
        // 8 frames only on GPU: the naive stages are still slow there.
        let n = if dev.is_gpu() { 8 } else { frames };
        let short: Vec<u32> = codes[..n * 16].to_vec();
        let codec2 = Codec::load(dir, Device::open(spec)?, 24000)?;
        let _ = codec2.dequantize(&short, n)?;
        let t = Instant::now();
        let pcm = codec2.decoder(&short, n)?;
        let _ = codec2.device().read(&pcm)?;
        let t_pcm = t.elapsed().as_secs_f64() * frames as f64 / n as f64;
        println!("{spec}: quantise={t_q:.3}s pre_conv={t_pre:.3}s transformer={t_tr:.3}s upsample={t_up:.3}s seanet≈{t_pcm:.3}s");
    }
    Ok(())
}
