//! Compare default vs fused-arena depth codes for the same last_h + noise.
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::lm::{Lm, HIDDEN, CODEBOOKS, AUDIO_VOCAB};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let dev = Device::open("gpu")?;
    let lm = Lm::load(Path::new("D:/M5-TTS/weights/M5-TTS-50M-VoiceClone"), dev.shared())?;
    let last_h = dev.tensor([1, HIDDEN], &vec![0.05f32; HIDDEN])?;
    let noise = dev.tensor([CODEBOOKS, AUDIO_VOCAB], &vec![0.5f32; CODEBOOKS * AUDIO_VOCAB])?;

    let default = lm.depth_codes_device(&last_h, &noise, 0.9, 10, true)?;
    println!("default: {:?}", default);
    let fused = lm.depth_codes_device_fused(&last_h, &noise, 0.9, 10, true)?;
    println!("fused:   {:?}", fused);
    let mism = default.iter().zip(&fused).filter(|(a,b)| a!=b).count();
    println!("mismatches: {mism} / {CODEBOOKS}");
    Ok(())
}