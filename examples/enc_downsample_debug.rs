//! Isolated check: golden transformer output -> downsample vs golden downsample.
use m5_tts_wgpu::codec::Codec;
use m5_tts_wgpu::device::Device;

fn golden(name: &str) -> Vec<f32> {
    std::fs::read(format!("goldens/acoustic/{name}.bin"))
        .unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-Audio-Tokenizer");
    let codec = Codec::load(dir, Device::Cpu, 24000)?;
    let dev = codec.device();
    let input = dev.tensor([512, 86], &golden("prompt_transformer"))?;
    let out = codec.encoder_downsample(&input)?;
    let got = dev.read(&out)?;
    let want = golden("prompt_downsample");
    let mut diff = 0.0f32;
    for (a, b) in want.iter().zip(&got) { diff = diff.max((a - b).abs()); }
    println!("downsample(golden input): shape {:?} max|diff|={diff:.3e}", out.shape());
    println!("want[0..4] {:?}", &want[..4]);
    println!("got [0..4] {:?}", &got[..4]);
    let amax = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!("|want|max={amax:.3}");
    Ok(())
}
