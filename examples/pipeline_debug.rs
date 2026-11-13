//! Print the pipeline's audio length vs the reference golden.
use m5_tts_wgpu::pipeline::{Args, NoiseSource, Pipeline, RecordedNoise};

fn golden(name: &str) -> Vec<f32> {
    std::fs::read(format!("goldens/acoustic/{name}.bin"))
        .unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() -> anyhow::Result<()> {
    let model = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-50M-VoiceClone");
    let codec = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-Audio-Tokenizer");
    let pipeline = Pipeline::load(model, codec, "cpu")?;
    let args = Args {
        max_audio_frames: 56,
        text: "你好，欢迎使用我们的语音合成模型，今天天气很好。".into(),
        prompt_wav: Some(std::path::PathBuf::from(r"D:\M5-TTS\assets\aslp_default_voice.wav")),
        prompt_text: Some("猪携带该疾病，然后通过蚊子传染给人类。".into()),
        ..Default::default()
    };
    let mut noise = RecordedNoise::new(golden("noise_all"));
    let out = pipeline.synthesize(&args, &mut noise as &mut dyn NoiseSource)?;
    let want = golden("final_audio");
    println!("frames {} (prompt {}) audio {} vs golden {}",
             out.frames, out.prompt_frames, out.audio.len(), want.len());
    Ok(())
}
