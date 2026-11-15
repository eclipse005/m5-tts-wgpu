//! Reference-audio front end in isolation: `prepare_prompt` (resample, peak
//! normalise, silence trim) and the encoder's codes, dumped for diffing against
//! the Python reference.
//!
//!   cargo run --release --example prompt_probe -- <wav> <out-prefix>
use m5_tts_wgpu::audio;
use m5_tts_wgpu::codec::Codec;
use m5_tts_wgpu::device::Device;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out = &args[2];
    // `--raw <file.f32>`: encode an already-prepared waveform verbatim, so the
    // encoder can be probed at an exact length (the front end is checked
    // separately, sample for sample).
    let samples = if args[1] == "--raw" {
        let bytes = std::fs::read(&args[3])?;
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
    } else {
        audio::prepare_prompt(Path::new(&args[1]), 24000)?
    };
    println!("prepare_prompt: {} samples ({:.3} s)", samples.len(), samples.len() as f64 / 24000.0);
    let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(format!("{out}.prompt.f32"), bytes)?;

    let dev = Device::open("cpu")?;
    let codec = Codec::load(Path::new("D:/M5-TTS/weights/M5-TTS-Audio-Tokenizer"), dev.shared(), 24000)?;
    let dev = codec.device();
    let audio_t = dev.tensor([1, samples.len()], &samples)?;
    let seanet = codec.seanet_encode(&audio_t)?;
    let mut time_major = dev.zeros([seanet.shape()[1], seanet.shape()[0]])?;
    m5_tts_wgpu::ops::transpose(dev, &seanet, &mut time_major)?;
    let transformed = codec.encoder_transformer(&time_major)?;
    let mut channels = dev.zeros([transformed.shape()[1], transformed.shape()[0]])?;
    m5_tts_wgpu::ops::transpose(dev, &transformed, &mut channels)?;
    let downsampled = codec.encoder_downsample(&channels)?;
    let codes = codec.encode_codes(&downsampled)?;
    println!("codes: {} frames x {}", codes.len() / 16, 16);
    let text: Vec<String> = codes.iter().map(|c| c.to_string()).collect();
    std::fs::write(format!("{out}.codes.txt"), text.join("\n"))?;
    Ok(())
}
