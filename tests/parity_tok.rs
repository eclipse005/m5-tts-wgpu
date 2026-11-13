//! Token-level parity with the Python reference: conditioned string and token ids
//! must match exactly (`scripts/dump_golden.py` produces the goldens).

use anyhow::{Context, Result};
use m5_tts_wgpu::conditioning::format_conditioned_text;
use m5_tts_wgpu::config::{ConditioningMeta, ModelConfig};
use m5_tts_wgpu::tok::Tok;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct GoldenFile {
    vocab_size: usize,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    lang: Option<String>,
    accent: Option<String>,
    caption: Option<String>,
    speaker: String,
    normalized: String,
    conditioned: String,
    ids: Vec<u32>,
}

fn model_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("M5_MODEL") {
        return Some(PathBuf::from(dir));
    }
    let default = Path::new(r"D:\M5-TTS\weights\M5-TTS-50M-VoiceClone");
    default.join("model.safetensors").is_file().then(|| default.to_path_buf())
}

#[test]
fn conditioned_text_and_ids_match_reference() -> Result<()> {
    let goldens = Path::new(env!("CARGO_MANIFEST_DIR")).join("goldens/tokenizer.json");
    if !goldens.exists() {
        eprintln!("skipping: {} not found (run scripts/dump_golden.py)", goldens.display());
        return Ok(());
    }
    let Some(bundle) = model_dir() else {
        eprintln!("skipping: no M5 checkpoint bundle (set M5_MODEL)");
        return Ok(());
    };
    let golden: GoldenFile = serde_json::from_str(&std::fs::read_to_string(&goldens)?)?;
    let _cfg = ModelConfig::load(&bundle.join("config.json"))?;
    let meta = ConditioningMeta::load(&bundle.join("conditioning.json"))?;
    let tok = Tok::load(&bundle, &meta).context("load tokenizer")?;
    assert_eq!(tok.vocab_size(), golden.vocab_size);

    let mut checked = 0usize;
    for case in &golden.cases {
        let conditioned = format_conditioned_text(
            &case.normalized,
            Some(&case.speaker),
            case.accent.as_deref(),
            case.lang.as_deref(),
            case.caption.as_deref(),
        )?;
        assert_eq!(conditioned, case.conditioned, "prompt differs for {:?}", case.text);
        let ids = tok.encode(&conditioned)?;
        assert_eq!(ids, case.ids, "ids differ for {:?}", case.text);
        checked += 1;
    }
    assert_eq!(checked, golden.cases.len());
    eprintln!("token parity: {checked} cases exact");
    Ok(())
}
