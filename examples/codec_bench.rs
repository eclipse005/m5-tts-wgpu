//! Codec decode timing: CPU vs GPU on the reference code matrix.
use m5_tts_wgpu::codec::Codec;
use m5_tts_wgpu::device::Device;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-Audio-Tokenizer");
    let codes: Vec<u32> = std::fs::read("goldens/acoustic/codes.bin")?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u32)
        .collect();
    let frames = codes.len() / 16;
    let audio_s = frames as f64 / 12.5;

    for spec in ["cpu", "gpu"] {
        let dev = Device::open(spec)?;
        let codec = Codec::load(dir, dev, 24000)?;
        // Load weights that the decode touches, so the timing is steady state.
        let _ = codec.dequantize(&codes, frames)?;
        let start = Instant::now();
        let pcm = codec.decoder(&codes, frames)?;
        let _ = codec.device().read(&pcm)?;
        let elapsed = start.elapsed().as_secs_f64();
        println!("{spec}: {frames} frames -> {:.2}s audio in {elapsed:.3}s (RTF {:.3}) [{}]",
                 audio_s, elapsed / audio_s, codec.device().describe());
    }
    Ok(())
}
