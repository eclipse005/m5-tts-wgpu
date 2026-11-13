//! Compare the depth decoder's first step against the reference dump.
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::lm::backbone::{rope_table, KvCache};
use m5_tts_wgpu::lm::{Lm, DEPTH_LAYERS, HIDDEN};

fn golden(name: &str) -> Vec<f32> {
    std::fs::read(format!("goldens/acoustic/{name}.bin"))
        .unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-50M-VoiceClone");
    let lm = Lm::load(dir, Device::Cpu)?;
    let dev = lm.device();
    let last_h = dev.tensor([1, HIDDEN], &golden("frame0_h"))?;
    let c0 = 440u32; // frame0_sample[0]

    let projection = lm.tensor("projection.weight", [HIDDEN, HIDDEN])?;
    let e0 = lm.audio_embedding(0, c0)?;
    let mut joined = dev.read(&last_h)?;
    joined.extend_from_slice(&dev.read(&e0)?);
    let joined = dev.tensor([2, HIDDEN], &joined)?;
    let mut projected = dev.zeros([2, HIDDEN])?;
    m5_tts_wgpu::ops::linear(dev, &joined, &projection, None, &mut projected)?;
    println!("depth input max|diff|={:.3e}", diff(&golden("frame0_depth_input"), &dev.read(&projected)?));

    let (cos, sin) = rope_table(&[0, 1]);
    let mut kv = KvCache::new(dev, DEPTH_LAYERS, 18)?;
    let mut hidden = dev.copy(&projected)?;
    hidden = lm.llama_layer("decoder", 0, &hidden, &cos, &sin, Some((&mut kv, 0, true)))?;
    println!("depth layer0 max|diff|={:.3e}", diff(&golden("frame0_depth_layer0"), &dev.read(&hidden)?));
    for layer in 1..DEPTH_LAYERS {
        hidden = lm.llama_layer("decoder", layer, &hidden, &cos, &sin, Some((&mut kv, 0, true)))?;
        let host = dev.read(&hidden)?;
        println!("  after depth layer{layer}: |h|={:.3} first4={:?}",
                 host.iter().fold(0.0f32, |m, v| m.max(v.abs())), &host[..4]);
    }
    println!("depth layer_out max|diff|={:.3e}", diff(&golden("frame0_depth_layer_out"), &dev.read(&hidden)?));
    Ok(())
}
