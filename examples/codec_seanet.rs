//! Bisect the SEANet tail: same input, both backends, per-block max|diff|.
use m5_tts_wgpu::codec::Codec;
use m5_tts_wgpu::device::Device;

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-Audio-Tokenizer");
    let (shape, data) = golden("codec_upsample");
    let [channels, frames] = [shape[0], shape[1]];
    println!("input {shape:?}");

    let cpu = Codec::load(dir, Device::Cpu, 24000)?;
    let out_cpu = cpu.seanet(&cpu.device().tensor([channels, frames], &data)?)?;
    let cpu_pcm = cpu.device().read(&out_cpu)?;

    let gpu = Codec::load(dir, Device::open("gpu")?, 24000)?;
    let out_gpu = gpu.seanet(&gpu.device().tensor([channels, frames], &data)?)?;
    let gpu_pcm = gpu.device().read(&out_gpu)?;

    let mut diff = 0.0f32;
    for (a, b) in cpu_pcm.iter().zip(&gpu_pcm) {
        diff = diff.max((a - b).abs());
    }
    println!("seanet cpu-vs-gpu: {} samples, max|diff|={diff:.3e}", cpu_pcm.len());
    println!("first: cpu {:?}", &cpu_pcm[..4]);
    println!("       gpu {:?}", &gpu_pcm[..4]);
    Ok(())
}

fn golden(name: &str) -> (Vec<usize>, Vec<f32>) {
    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("goldens/acoustic/index.json").unwrap()).unwrap();
    let shape: Vec<usize> = index["tensors"][name]["shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let bytes = std::fs::read(format!("goldens/acoustic/{name}.bin")).unwrap();
    let data: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let cols = *shape.last().unwrap();
    let rows = data.len() / cols;
    (vec![rows, cols], data)
}
