//! Backbone parity: prompt embeddings, the 8-layer prefill's last hidden row and
//! the codebook-0 logits, per backend.

mod common;

use anyhow::Result;
use common::{Goldens, max_abs_diff};
use m5_tts_wgpu::device::Device;
use m5_tts_wgpu::lm::backbone::KvCache;
use m5_tts_wgpu::lm::{AUDIO_VOCAB, CODEBOOKS, HIDDEN, Lm};
use std::path::PathBuf;

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var("M5_MODEL")
        .unwrap_or_else(|_| r"D:\M5-TTS\weights\M5-TTS-50M-VoiceClone".to_string());
    let path = PathBuf::from(dir);
    path.join("model.safetensors").is_file().then_some(path)
}

#[test]
fn backbone_prefill_matches_reference() -> Result<()> {
    let Some(goldens) = Goldens::open() else {
        eprintln!("skipping: run scripts/dump_acoustic.py");
        return Ok(());
    };
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no M5 bundle (set M5_MODEL)");
        return Ok(());
    };
    let tokens: Vec<u32> = goldens.i64("frame0_tokens").iter().map(|v| *v as u32).collect();
    let frames = goldens.shape("frame0_tokens")[0] / (CODEBOOKS + 1);
    let want_embeds = goldens.f32("frame0_embeds");
    let want_h = goldens.f32("frame0_h");
    let want_c0 = goldens.f32("frame0_c0_logits");
    assert_eq!(want_h.len(), HIDDEN);
    assert_eq!(want_c0.len(), AUDIO_VOCAB);

    for spec in ["cpu", "gpu"] {
        let dev = Device::open(spec)?;
        let lm = Lm::load(&dir, dev)?;
        let embedded = lm.embed_frames(&tokens, frames)?;
        assert_eq!(want_embeds.len(), frames * HIDDEN);
        let diff_emb = max_abs_diff(&want_embeds, &lm.device().read(&embedded)?);
        eprintln!("[{}] prompt embeddings max|diff|={diff_emb:.3e}", lm.device().describe());
        assert!(diff_emb < 1e-4, "embeddings diff {diff_emb}");
        let mut kv = KvCache::new(lm.device(), 8, frames + 512)?;
        let want_layer0 = goldens.f32("frame0_layer0");
        let want_layers: Vec<Vec<f32>> =
            (0..8).map(|i| goldens.f32(&format!("frame0_layer{i}"))).collect();
        let h = lm.backbone_prefill(&embedded, &mut kv)?;
        assert_eq!(h.shape(), [frames, HIDDEN]);

        // Per-layer parity: the first layer that drifts is the one to fix.
        {
            let mut kv0 = KvCache::new(lm.device(), 8, frames + 512)?;
            let l0 = lm.backbone_layer_public(0, &embedded, frames, &mut kv0)?;
            let diff = max_abs_diff(&want_layer0, &lm.device().read(&l0)?);
            eprintln!("[{}] layer0 max|diff|={diff:.3e}", lm.device().describe());
        }
        // Layer-by-layer replay with a fresh cache, to localise the drift.
        {
            let mut cur = lm.device().copy(&embedded)?;
            for index in 0..8 {
                let mut kv_layer = KvCache::new(lm.device(), 8, frames + 512)?;
                cur = lm.backbone_layer_public(index, &cur, frames, &mut kv_layer)?;
                let diff = max_abs_diff(&want_layers[index], &lm.device().read(&cur)?);
                eprintln!("[{}] layer{index} max|diff|={diff:.3e}", lm.device().describe());
            }
        }
        let last = last_row(lm.device(), &h, frames)?;
        let diff_h = max_abs_diff(&want_h, &lm.device().read(&last)?);
        let logits = lm.codebook0_logits(&last)?;
        let got_c0 = lm.device().read(&logits)?;
        let diff_c0 = max_abs_diff(&want_c0, &got_c0);

        eprintln!("[{}] backbone h max|diff|={diff_h:.3e}, c0 logits max|diff|={diff_c0:.3e} \
                   (argmax want {} got {})",
                  lm.device().describe(), argmax(&want_c0), argmax(&got_c0));
        assert!(diff_h < 1e-4, "h diff {diff_h}");
        assert!(diff_c0 < 2e-4, "c0 diff {diff_c0}");
        assert_eq!(argmax(&want_c0), argmax(&got_c0), "c0 argmax");
    }
    Ok(())
}

/// Last hidden row as a `[1, HIDDEN]` tensor (device-side slice-free copy).
fn last_row(dev: &Device, h: &m5_tts_wgpu::device::Tensor, frames: usize) -> Result<m5_tts_wgpu::device::Tensor> {
    let host = dev.read(h)?;
    let start = (frames - 1) * HIDDEN;
    dev.tensor([1, HIDDEN], &host[start..start + HIDDEN])
}

/// Incremental decode must agree with a full prefill: this is the gate on the KV
/// cache (row writes + single-query attention), which the reference goldens do
/// not cover directly.
#[test]
fn kv_cache_step_matches_full_prefill() -> Result<()> {
    let Some(goldens) = Goldens::open() else {
        eprintln!("skipping: run scripts/dump_acoustic.py");
        return Ok(());
    };
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no M5 bundle (set M5_MODEL)");
        return Ok(());
    };
    let tokens: Vec<u32> = goldens.i64("frame0_tokens").iter().map(|v| *v as u32).collect();
    let frames = goldens.shape("frame0_tokens")[0] / (CODEBOOKS + 1);
    let split = 20usize.min(frames - 1);

    for spec in ["cpu", "gpu"] {
        let lm = Lm::load(&dir, Device::open(spec)?)?;
        let dev = lm.device();

        // Reference: full prefill over `split + 1` frames, last row is the target.
        let all = lm.embed_frames(&tokens[..(split + 1) * (CODEBOOKS + 1)], split + 1)?;
        let mut kv_full = KvCache::new(dev, 8, split + 8)?;
        let h_full = lm.backbone_prefill(&all, &mut kv_full)?;
        let want = last_row(dev, &h_full, split + 1)?;

        // Incremental: prefill `split` frames, then one step.
        let head = lm.embed_frames(&tokens[..split * (CODEBOOKS + 1)], split)?;
        let mut kv = KvCache::new(dev, 8, split + 8)?;
        let _ = lm.backbone_prefill(&head, &mut kv)?;
        let row = dev.tensor([1, HIDDEN], &dev.read(&lm.embed_frames(
            &tokens[split * (CODEBOOKS + 1)..(split + 1) * (CODEBOOKS + 1)], 1)?)?)?;
        let got = lm.backbone_step(&row, &mut kv, split)?;

        let diff = max_abs_diff(&dev.read(&want)?, &dev.read(&got)?);
        eprintln!("[{}] kv-cache step vs prefill max|diff|={diff:.3e}", dev.describe());
        assert!(diff < 5e-4, "incremental decode diff {diff}");
    }
    Ok(())
}

/// Depth decoder parity: replay the reference's own codes through the fifteen
/// steps and compare every codebook's logits, then replay its sampler noise and
/// check the sampled tokens.
#[test]
fn depth_decoder_matches_reference() -> Result<()> {
    let Some(goldens) = Goldens::open() else {
        eprintln!("skipping: run scripts/dump_acoustic.py");
        return Ok(());
    };
    let Some(dir) = model_dir() else {
        eprintln!("skipping: no M5 bundle (set M5_MODEL)");
        return Ok(());
    };
    let tokens: Vec<u32> = goldens.i64("frame0_tokens").iter().map(|v| *v as u32).collect();
    let frames = goldens.shape("frame0_tokens")[0] / (CODEBOOKS + 1);
    // Frame 0 in *token* space: the depth decoder consumes logits-space tokens
    // (0..2050), while the codec consumes `token - 1`.
    let frame0: Vec<u32> = goldens.root_i64("frame0_sample").iter().map(|v| *v as u32).collect();
    let noise_tokens = goldens.root_i64("frame0_noise_tokens");
    let noise: Vec<Vec<f32>> = (0..CODEBOOKS).map(|i| goldens.f32(&format!("frame0_noise{i}"))).collect();
    let want_h = goldens.f32("frame0_h");

    for spec in ["cpu", "gpu"] {
        let lm = Lm::load(&dir, Device::open(spec)?)?;
        let dev = lm.device();
        // Prefill the backbone once; the depth decoder only needs its last row.
        let embedded = lm.embed_frames(&tokens, frames)?;
        let mut kv = KvCache::new(dev, 8, frames + 8)?;
        let h = lm.backbone_prefill(&embedded, &mut kv)?;
        let last = last_row(dev, &h, frames)?;
        let diff_h = max_abs_diff(&want_h, &dev.read(&last)?);
        assert!(diff_h < 1e-4, "backbone h drifted: {diff_h}");

        // 0) sampler replay first: it is the decisive gate (the recorded ci logits
        //    golden turned out to be captured from a different iteration).
        let sampled = lm.depth_codes(&last, frame0[0], &noise, 0.9, 10, false)?;
        eprintln!("[{}] depth sample got  {sampled:?}", dev.describe());
        eprintln!("[{}] depth sample want {frame0:?}", dev.describe());

        // 1) logits replay with the recorded codes
        let logits = lm.depth_logits_replay(&last, &frame0)?;
        assert_eq!(logits.len(), CODEBOOKS - 1);
        let mut worst = 0.0f32;
        for (index, logits) in logits.iter().enumerate() {
            let want = goldens.f32(&format!("frame0_ci{}_logits", index + 1));
            let diff = max_abs_diff(&want, &dev.read(logits)?);
            let got = dev.read(logits)?;
            eprintln!("[{}]   ci{}: max|diff|={diff:.3e} (argmax want {}, got {})",
                      dev.describe(), index + 1, argmax(&want), argmax(&got));
            worst = worst.max(diff);
        }
        eprintln!("[{}] depth logits (15 x 2051) max|diff|={worst:.3e}", dev.describe());
        assert!(worst < 2e-4, "depth logits diff {worst}");

        // 2) sampler replay with the recorded noise
        let codes = lm.depth_codes(&last, frame0[0], &noise, 0.9, 10, false)?;
        let want_codes: Vec<u32> = noise_tokens.iter().map(|v| *v as u32).collect();
        eprintln!("[{}] depth sample: got {codes:?}", dev.describe());
        eprintln!("[{}] depth sample: want {want_codes:?}", dev.describe());
        assert_eq!(codes, want_codes, "sampled codes differ on {}", dev.describe());
    }
    Ok(())
}
