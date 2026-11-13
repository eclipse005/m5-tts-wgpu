//! Debug helper: print one codec stage for the golden codes on both backends.
use m5_tts_wgpu::codec::Codec;
use m5_tts_wgpu::device::Device;

fn main() -> anyhow::Result<()> {
    let codes: Vec<u32> = std::fs::read("goldens/acoustic/codes.bin")?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u32)
        .collect();
    let frames = codes.len() / 16;
    let want: Vec<f32> = std::fs::read("goldens/acoustic/codec_quantized.bin")?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let dir = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-Audio-Tokenizer");
    for spec in ["cpu", "gpu"] {
        let dev = Device::open(spec)?;
        let codec = Codec::load(dir, dev, 24000)?;
        let out = codec.dequantize(&codes, frames)?;
        let got = codec.device().read(&out)?;
        let mut diff = 0.0f32;
        for (a, b) in want.iter().zip(&got) {
            diff = diff.max((a - b).abs());
        }
        println!("{spec}: shape {:?} max|diff|={diff:.3e}", out.shape());
        println!("  want[0..5] {:?}", &want[..5]);
        println!("  got [0..5] {:?}", &got[..5]);
        println!("  want mean {:.4} got mean {:.4}", want.iter().sum::<f32>() / want.len() as f32,
                 got.iter().sum::<f32>() / got.len() as f32);
    }
    Ok(())
}
