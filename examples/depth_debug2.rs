//! Compare cb-1 logits against the reference dump, with a few raw values.
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::lm::backbone::{rope_table, KvCache};
use m5_tts_wgpu::lm::{Lm, DEPTH_LAYERS, HIDDEN};

fn golden(name: &str) -> Vec<f32> {
    std::fs::read(format!("goldens/acoustic/{name}.bin"))
        .unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-50M-VoiceClone");
    let lm = Lm::load(dir, Device::Cpu)?;
    let dev = lm.device();
    let want = golden("frame0_ci1_logits");
    let logits = lm.depth_logits_replay(
        &dev.tensor([1, HIDDEN], &golden("frame0_h"))?,
        &[440, 1327, 1356, 1210, 1441, 219, 1049, 1656, 1020, 1858, 1779, 1378, 1556, 737, 872, 1175],
    )?;
    let got = dev.read(&logits[0])?;
    let mut diff = 0.0f32;
    for (a, b) in want.iter().zip(&got) { diff = diff.max((a - b).abs()); }
    println!("cb1 max|diff|={diff:.3e}");
    println!("want[0..6]  {:?}", &want[..6]);
    println!("got [0..6]  {:?}", &got[..6]);
    let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    println!("argmax want {} got {}", am(&want), am(&got));
    // stack4 sanity: my stack output vs the golden stack4
    let _ = DEPTH_LAYERS;
    let (cos, sin) = rope_table(&[0, 1]);
    let mut kv = KvCache::new(dev, DEPTH_LAYERS, 18)?;
    let projection = lm.tensor("projection.weight", [HIDDEN, HIDDEN])?;
    let e0 = lm.audio_embedding(0, 440)?;
    let mut joined = dev.read(&dev.tensor([1, HIDDEN], &golden("frame0_h"))?)?;
    joined.extend_from_slice(&dev.read(&e0)?);
    let joined = dev.tensor([2, HIDDEN], &joined)?;
    let mut hidden = dev.zeros([2, HIDDEN])?;
    m5_tts_wgpu::ops::linear(dev, &joined, &projection, None, &mut hidden)?;
    for layer in 0..DEPTH_LAYERS {
        hidden = lm.llama_layer("decoder", layer, &hidden, &cos, &sin, Some((&mut kv, 0, true)))?;
    }
    let want_stack = golden("frame0_depth_stack4");
    let host = dev.read(&hidden)?;
    let mut d2 = 0.0f32;
    for (a, b) in want_stack.iter().zip(&host) { d2 = d2.max((a - b).abs()); }
    println!("stack4 max|diff|={d2:.3e}");
    Ok(())
}
