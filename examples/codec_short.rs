//! Bisect the short-slice decode: per-stage max|diff| for both backends.
use m5_tts_wgpu::codec::Codec;
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::ops;

fn golden(name: &str) -> (Vec<usize>, Vec<f32>) {
    let text = std::fs::read_to_string("goldens/acoustic/index.json").unwrap();
    let index: serde_json::Value = serde_json::from_str(&text).unwrap();
    let shape: Vec<usize> = index["tensors"][name]["shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let bytes = std::fs::read(format!("goldens/acoustic/{name}.bin")).unwrap();
    let data: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let cols = *shape.last().unwrap();
    (vec![data.len() / cols, cols], data)
}

fn diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn main() -> anyhow::Result<()> {
    let dir = std::path::Path::new(r"D:\M5-TTS\weights\M5-TTS-Audio-Tokenizer");
    let codes: Vec<u32> = std::fs::read("goldens/acoustic/codes_short.bin")?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u32)
        .collect();
    let frames = codes.len() / 16;
    let (_, want_q) = golden("codec_quantized_short");
    let (_, want_pre) = golden("codec_pre_conv_short");
    let (_, want_tr) = golden("codec_transformer_short");
    let (want_up_shape, want_up) = golden("codec_upsample_short");
    let (_, want_pcm) = golden("pcm_short");

    // Direct conv0 through both backends with the codec's own weight tensor.
    {
        let (want_up_shape, want_up2) = golden("codec_upsample_short");
        let _ = want_up_shape;
        let mut outs = Vec::new();
        for spec in ["cpu", "gpu"] {
            let dev = Device::open(spec)?;
            let codec = Codec::load(dir, dev, 24000)?;
            let dev = codec.device();
            let x = dev.tensor([1024, frames * 4], &want_up2)?;
            let w = codec.tensor("decoder.decoder.0.conv.weight", [1536, 1024 * 7])?;
            let b = codec.vector("decoder.decoder.0.conv.bias", 1536)?;
            let mut y = dev.zeros([1536, frames * 4])?;
            let p = m5_tts_wgpu::ops::Conv1d::causal(1024, 1536, 7, frames * 4);
            m5_tts_wgpu::ops::conv1d(dev, &x, &w, &b, &mut y, &p)?;
            let v = dev.read(&y)?;
            println!("direct conv0 {spec}: first={:?}", &v[..4]);
            outs.push(v);
        }
        println!("direct conv0 cpu-vs-gpu={:.3e}", diff(&outs[0], &outs[1]));
    }

    // SEANet bisect: run the same upsampled input through both backends and
    // compare every stage.
    let (_, want_up2) = golden("codec_upsample_short");
    let mut traces = Vec::new();
    for spec in ["cpu", "gpu"] {
        let dev = Device::open(spec)?;
        let codec = Codec::load(dir, dev, 24000)?;
        let input = codec.device().tensor([1024, frames * 4], &want_up2)?;
        let (_pcm, trace) = codec.seanet_trace(&input)?;
        traces.push((spec, trace.into_iter().map(|(n, t)| (n, codec.device().read(&t).unwrap())).collect::<Vec<_>>()));
    }
    for (index, (name, cpu_vals)) in traces[0].1.iter().enumerate() {
        let gpu_vals = &traces[1].1[index].1;
        println!("seanet {name}: len={} cpu-vs-gpu={:.3e}", cpu_vals.len(), diff(cpu_vals, gpu_vals));
        if index == 0 {
            let amax = |v: &[f32]| v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            println!("  cpu[0..6]={:?} |max|={}", &cpu_vals[..6], amax(cpu_vals));
            println!("  gpu[0..6]={:?} |max|={}", &gpu_vals[..6], amax(gpu_vals));
            println!("  cpu[7168..7174]={:?}", &cpu_vals[7168..7174]);
            println!("  gpu[7168..7174]={:?}", &gpu_vals[7168..7174]);
        }
    }

    for spec in ["cpu", "gpu"] {
        let dev = Device::open(spec)?;
        let codec = Codec::load(dir, dev, 24000)?;
        let dev = codec.device();
        let q = codec.dequantize(&codes, frames)?;
        let pre = codec.pre_transformer_input(&codes, frames)?;
        let tr = codec.transformer(&pre)?;
        let mut ch = dev.zeros([1024, frames])?;
        ops::transpose(dev, &tr, &mut ch)?;
        let up = codec.upsample(&ch)?;
        let pcm = codec.decoder(&codes, frames)?;
        println!("{spec}: quantizer={:.3e} pre_conv={:.3e} transformer={:.3e} upsample={:.3e} (shape {:?} want {:?}) pcm={:.6}",
                 diff(&dev.read(&q)?, &want_q), diff(&dev.read(&pre)?, &want_pre),
                 diff(&dev.read(&tr)?, &want_tr), diff(&dev.read(&up)?, &want_up), up.shape(), want_up_shape,
                 diff(&dev.read(&pcm)?, &want_pcm));
    }
    Ok(())
}
