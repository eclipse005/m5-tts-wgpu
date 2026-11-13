//! Does the channels-last SEANet reproduce the golden PCM?
use m5_tts_wgpu::codec::Codec;
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::ops;

fn golden(name: &str) -> Vec<f32> {
    std::fs::read(format!("goldens/acoustic/{name}.bin"))
        .unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-Audio-Tokenizer");
    let codec = Codec::load(dir, Device::Cpu, 24000)?;
    let dev = codec.device();
    let want = golden("pcm");
    let (up_shape, up) = {
        let bytes = std::fs::read("goldens/acoustic/codec_upsample.bin")?;
        let data: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        ([1024usize, data.len() / 1024], data)
    };
    // [C, T] -> [T, C]
    let mut time_major = dev.zeros([up_shape[1], up_shape[0]])?;
    let input = dev.tensor([up_shape[0], up_shape[1]], &up)?;
    ops::transpose(dev, &input, &mut time_major)?;

    // Same-process A/B: cross-run numbers drift on this box.
    let t = std::time::Instant::now();
    let _ = codec.seanet(&input)?;
    println!("channels-first: {:.3}s", t.elapsed().as_secs_f64());
    let start = std::time::Instant::now();
    let pcm = codec.seanet_cl(&time_major)?;
    let elapsed = start.elapsed().as_secs_f64();
    let _ = &pcm;
    let got = dev.read(&pcm)?;
    let mut diff = 0.0f32;
    for (a, b) in want.iter().zip(&got) { diff = diff.max((a - b).abs()); }
    let signal: f32 = want.iter().map(|v| v * v).sum();
    let noise: f32 = want.iter().zip(&got).map(|(a, b)| (a - b) * (a - b)).sum();
    println!("channels-last:  {:.3}s", elapsed);
    println!("seanet_cl: {} samples vs {}, max|diff|={diff:.3e}, SNR={:.1} dB",
             got.len(), want.len(), 10.0 * (signal / noise.max(1e-20)).log10());
    Ok(())
}
