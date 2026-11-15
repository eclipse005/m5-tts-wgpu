//! Where do the 84 ms of "prepare one 75 MB transposed-conv weight" go?
//! Splits get_f32 / pack / upload into their own timings.
//!
//!   cargo run --release --example weight_prep_probe
use m5_tts_wgpu::cpu_ops;
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::weights::Weights;
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let file = Path::new("D:/M5-TTS/weights/M5-TTS-Audio-Tokenizer/model.safetensors");
    let start = Instant::now();
    let ws = Weights::load(file)?;
    println!("Weights::load            {:6.1} ms", start.elapsed().as_secs_f64() * 1e3);

    for name in ["decoder.decoder.1.block.1.conv.weight", "decoder.decoder.2.block.1.conv.weight",
                 "decoder.decoder.3.block.1.conv.weight", "decoder.decoder.4.block.1.conv.weight"] {
        let entry = ws.get(name)?;
        let shape = entry.shape.clone();
        let start = Instant::now();
        let raw = ws.get_f32(name)?;
        let t_get = start.elapsed().as_secs_f64() * 1e3;
        let (cin, cout_ks) = (shape[0], shape[1]);
        let stride = cout_ks / (cin / (shape[0] / shape[0])); // unused fallback
        let _ = stride;
        // The decoder's transposed convs: [cin, cout, ks] with ks = 2 * rate.
        let cout_guess = shape[1] / 2;
        let start = Instant::now();
        let packed = cpu_ops::pack_conv_transpose_phased(&raw, shape[0], cout_guess, shape[2],
                                                         shape[2] / 2);
        let t_pack = start.elapsed().as_secs_f64() * 1e3;
        println!("{name}\n  shape {shape:?}  {:.1} MB", raw.len() as f64 * 4.0 / 1e6);
        println!("  get_f32 {t_get:6.1} ms   pack {t_pack:6.1} ms   ({} values)", packed.len());
        drop(packed);
    }

    // Upload of the largest tensor, into a fresh vs a pooled buffer.
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    let raw = ws.get_f32("decoder.decoder.1.block.1.conv.weight")?;
    for label in ["fresh", "pooled"] {
        let start = Instant::now();
        for _ in 0..3 {
            match label {
                "fresh" => { let b = gpu.empty("probe.f", raw.len()); gpu.write(&b, &raw); }
                _ => { let t = dev.tensor([raw.len(), 1], &raw)?; gpu.flush(); let _ = dev.read(&t)?; }
            }
        }
        gpu.sync();
        println!("upload x3 ({label:6})   {:6.1} ms each ({:.1} MB)",
                 start.elapsed().as_secs_f64() * 1e3 / 3.0, raw.len() as f64 * 4.0 / 1e6);
    }
    Ok(())
}
