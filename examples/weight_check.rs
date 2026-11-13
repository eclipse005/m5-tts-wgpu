//! Compare a checkpoint tensor as read through both backends (upload path check).
use m5_tts_wgpu::codec::Codec;
use m5_tts_wgpu::device::Device;

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-Audio-Tokenizer");
    let name = "decoder.decoder.0.conv.weight";
    let shape = [1536usize, 1024 * 7];
    let mut values = Vec::new();
    for spec in ["cpu", "gpu"] {
        let codec = Codec::load(dir, Device::open(spec)?, 24000)?;
        let t = codec.tensor(name, shape)?;
        let v = codec.device().read(&t)?;
        let sum: f64 = v.iter().map(|x| *x as f64).sum();
        let first: Vec<f32> = v[..4].to_vec();
        let last: Vec<f32> = v[v.len() - 4..].to_vec();
        println!("{spec}: len={} sum={sum:.6} first={first:?} last={last:?}", v.len());
        values.push(v);
    }
    let diff = values[0].iter().zip(&values[1]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("cpu-vs-gpu weight max|diff|={diff:.3e}");
    Ok(())
}
