//! Depth decoder per-frame cost: host path vs device path.
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::lm::{AUDIO_VOCAB, CODEBOOKS, HIDDEN, Lm};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-50M-VoiceClone");
    let noise: Vec<Vec<f32>> = (0..CODEBOOKS)
        .map(|i| (0..AUDIO_VOCAB).map(|j| 0.5 + ((i * 7 + j) % 13) as f32 * 0.01).collect())
        .collect();
    let mut flat = Vec::new();
    for row in &noise {
        flat.extend_from_slice(row);
    }

    // Same-process A/B: cross-run numbers drift on this box.
    for noset in [false, true] {
        if noset {
            std::env::set_var("M5_NO_FUSED", "1");
        } else {
            std::env::remove_var("M5_NO_FUSED");
        }
        let lm = Lm::load(dir, Device::Cpu)?;
        let h = lm.device().tensor([1, HIDDEN], &vec![0.02f32; HIDDEN])?;
        let calls = 30;
        let _ = lm.depth_codes(&h, 0, &noise, 0.9, 10, true)?;
        let start = Instant::now();
        for _ in 0..calls { let _ = lm.depth_codes(&h, 0, &noise, 0.9, 10, true)?; }
        println!("cpu {}: {:.1} ms/frame", if noset { "generic " } else { "fused   " },
                 start.elapsed().as_secs_f64() * 1e3 / calls as f64);
    }

    for spec in ["cpu", "gpu"] {
        let lm = Lm::load(dir, Device::open(spec)?)?;
        let dev = lm.device();
        let h = dev.tensor([1, HIDDEN], &vec![0.02f32; HIDDEN])?;
        let calls = 20;

        let start = Instant::now();
        for _ in 0..calls {
            let _ = lm.depth_codes(&h, 0, &noise, 0.9, 10, true)?;
        }
        println!("{spec} host path:   {:.1} ms/frame", start.elapsed().as_secs_f64() * 1e3 / calls as f64);

        if spec == "gpu" {
            let noise_tensor = dev.tensor([CODEBOOKS, AUDIO_VOCAB], &flat)?;
            let _ = lm.depth_codes_device(&h, &noise_tensor, 0.9, 10, true)?;
            let start = Instant::now();
            for _ in 0..calls {
                let _ = lm.depth_codes_device(&h, &noise_tensor, 0.9, 10, true)?;
            }
            println!("{spec} device path: {:.1} ms/frame",
                     start.elapsed().as_secs_f64() * 1e3 / calls as f64);
        }
    }
    Ok(())
}
