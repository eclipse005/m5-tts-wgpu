//! Decode the same 98-frame code sequence twice in one process: the second pass
//! has every weight, packed copy, bind group and pipeline already resident, so
//! the difference is exactly what "preparing the decoder" costs per run.
//!
//!   cargo run --release --example codec_repeat_probe [frames]
use m5_tts_wgpu::codec::Codec;
use m5_tts_wgpu::device::Device;
use std::path::Path;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let frames: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(98);
    let dev = Device::open("gpu")?;
    let gpu = dev.gpu()?;
    eprintln!("device open, loading codec weights...");
    let codec = Codec::load(Path::new("D:/M5-TTS/weights/M5-TTS-Audio-Tokenizer"), dev.shared(), 24000)?;
    // A deterministic code sequence: the values only matter for cache behaviour.
    eprintln!("codec loaded");
    let codebooks = codec.cfg.num_quantizers;
    let codes: Vec<u32> = (0..frames * codebooks)
        .map(|i| ((i * 37 + 11) % 2047) as u32)
        .collect();
    println!("loading done, {frames} frames x {codebooks} codebooks");
    for pass in 0..3 {
        eprintln!("  start pass {pass}");
        gpu.sync();
        let start = Instant::now();
        let pcm = codec.decoder(&codes, frames)?;
        gpu.sync();
        let secs = start.elapsed().as_secs_f64();
        let st = gpu.snapshot();
        println!("pass {pass}: {:.3} s  ({} dispatch, {} submits, uploaded {} MB, host write_buffer {:.0} ms)",
                 secs, st.dispatches, st.submits, st.bytes_uploaded / 1_000_000,
                 gpu.upload_host_nanos.get() as f64 / 1e6);
        drop(pcm);
    }
    Ok(())
}
