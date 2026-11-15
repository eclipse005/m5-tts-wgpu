//! Scenario parity across the *feature* surface.
//!
//! `parity_pipeline` pins one case (Chinese zero-shot cloning). This one replays
//! the reference's own sampling noise through every feature the reference
//! exposes: preset voices (CustomVoice), the six languages, both accents, a
//! caption, temperature/top-k variants, and short/long texts. Each scenario
//! golden is `scripts/dump_scenarios.py`'s output: the noise draws, the codes
//! the reference sampled, and the PCM it decoded.
//!
//! Codes must match exactly; audio is gated on SNR (the two implementations
//! accumulate in different orders, so bit equality is not the bar).

mod common;

use anyhow::{Context, Result};
use m5_tts_wgpu::pipeline::{Args, NoiseSource, Pipeline, RecordedNoise};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn weights_root() -> PathBuf {
    PathBuf::from(std::env::var("M5_WEIGHTS").unwrap_or_else(|_| r"D:\M5-TTS\weights".to_string()))
}

/// Every scenario directory under `goldens/scenarios`, sorted for stable output.
fn scenarios() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("goldens/scenarios");
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.join("meta.json").is_file())
                .collect()
        })
        .unwrap_or_default();
    dirs.sort();
    dirs
}

fn f32_file(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

fn i64_file(path: &Path) -> Result<Vec<i64>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(bytemuck::cast_slice::<u8, i64>(&bytes).to_vec())
}

/// Tiny JSON reader: the meta files are flat (`"key": value` lines), and pulling
/// in a serde dependency for a test is not worth it.
fn meta<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{key}\":");
    let start = text.find(&needle)? + needle.len();
    let rest = text[start..].trim_start();
    if let Some(stripped) = rest.strip_prefix('"') {
        // Strings in these files contain no escapes.
        Some(&stripped[..stripped.find('"')?])
    } else {
        let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '-' || c == '.'))?;
        Some(&rest[..end])
    }
}

#[test]
fn scenarios_match_reference() -> Result<()> {
    let dirs = scenarios();
    if dirs.is_empty() {
        eprintln!("skipping: run scripts/dump_scenarios.py");
        return Ok(());
    }
    let root = weights_root();
    let codec_dir = root.join("M5-TTS-Audio-Tokenizer");
    if !codec_dir.join("model.safetensors").is_file() {
        eprintln!("skipping: codec weights missing");
        return Ok(());
    }
    // Both bundles stay resident for one device's pass (2 x ~1 GB) and are
    // dropped before switching device: reloading a second model after dropping
    // the first ran the card out of memory, because the dropped device's memory
    // is only reclaimed on a later poll.
    let mut failures: Vec<String> = Vec::new();

    // Both bundles alive on one device is ~3.3 GB of weights plus scratch, and
    // the second decode after a bundle switch runs the card out of memory (a
    // dropped device's memory is only reclaimed on a later poll). So the GPU
    // gate runs one bundle per invocation; the CPU gate always covers both.
    //   M5_SCENARIO_BUNDLE=preset cargo test --release --test parity_scenarios
    let gpu_bundle = std::env::var("M5_SCENARIO_BUNDLE").unwrap_or_else(|_| "clone".to_string());
    // The card keeps a dropped device's memory until something polls it again, so
    // a full pass over every bundle needs one device per process:
    //   M5_SCENARIO_DEVICE=gpu M5_SCENARIO_BUNDLE=preset cargo test ...
    let only = std::env::var("M5_SCENARIO_DEVICE").ok();
    let devices: Vec<&str> = ["cpu", "gpu"].into_iter()
        .filter(|d| only.as_deref().map(|o| o == *d).unwrap_or(true))
        .collect();

    for spec in devices {
        let mut resident: HashMap<String, Pipeline> = HashMap::new();
        for dir in &dirs {
            let text = std::fs::read_to_string(dir.join("meta.json"))?;
            let name = meta(&text, "name").unwrap_or("?").to_string();
            let kind = meta(&text, "kind").unwrap_or("clone");
            let bundle = if kind == "preset" {
                root.join("M5-TTS-50M-CustomVoice")
            } else {
                root.join("M5-TTS-50M-VoiceClone")
            };
            if !bundle.join("model.safetensors").is_file() {
                eprintln!("[{name}] skipping: {} missing", bundle.display());
                continue;
            }
            let want_codes: Vec<u32> = i64_file(&dir.join("codes.bin"))?
                .into_iter().map(|v| v as u32).collect();
            let want_audio = f32_file(&dir.join("pcm.bin"))?;
            let noise = f32_file(&dir.join("noise.bin"))?;
            let frames = want_codes.len() / 16;

            let mut args = Args {
                text: meta(&text, "text").unwrap_or_default().to_string(),
                temperature: meta(&text, "temperature").and_then(|v| v.parse().ok()).unwrap_or(0.9),
                topk: meta(&text, "topk").and_then(|v| v.parse().ok()).unwrap_or(10),
                ..Default::default()
            };
            // The reference's generation budget matters (it decides where the cap
            // lands), so it travels with the scenario.
            let budget = meta(&text, "max_audio_length_ms")
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(40000.0);
            args = args.with_max_audio_length_ms(budget)?;
            if let Some(lang) = meta(&text, "lang") {
                args.lang = lang.to_string();
            }
            args.accent = meta(&text, "accent").map(str::to_string);
            args.caption = meta(&text, "caption").map(str::to_string);
            if kind == "preset" {
                args.speaker_id =
                    meta(&text, "speaker_id").and_then(|v| v.parse().ok()).unwrap_or(1);
            } else {
                args.prompt_wav = Some(PathBuf::from(meta(&text, "prompt_wav").unwrap_or_default()));
                args.prompt_text = meta(&text, "prompt_text").map(str::to_string);
            }

            let key = bundle.display().to_string();
            if spec == "gpu" && kind != gpu_bundle {
                continue;
            }
            if !resident.contains_key(&key) {
                resident.insert(key.clone(), Pipeline::load(&bundle, &codec_dir, spec)?);
            }
            let pipeline = &resident[&key];
            let mut source = RecordedNoise::new(noise.clone());
            let out = match pipeline.synthesize(&args, &mut source as &mut dyn NoiseSource) {
                Ok(out) => out,
                Err(err) => {
                    // One scenario failing must not hide the other thirty.
                    failures.push(format!("{name}/{spec}: {err:#}"));
                    continue;
                }
            };

            // Same input, same noise, same process -> the same bytes. Run with
            // M5_SCENARIO_REPEAT=1: a kernel that reads memory it never wrote
            // (fixed-size score staging, a missing storage barrier, a recycled
            // pool buffer) shows up here and nowhere else, because it looks
            // like ordinary sampling noise from outside.
            if std::env::var_os("M5_SCENARIO_REPEAT").is_some() {
                let mut again = RecordedNoise::new(noise.clone());
                let second = pipeline.synthesize(&args, &mut again as &mut dyn NoiseSource)?;
                let same = second.audio.len() == out.audio.len()
                    && second.generated == out.generated
                    && second.audio.iter().zip(&out.audio).all(|(a, b)| a.to_bits() == b.to_bits());
                if !same {
                    failures.push(format!("{name}/{spec}: not reproducible run to run"));
                }
            }

            if std::env::var_os("M5_DUMP_PCM").is_some() {
                let bytes: Vec<u8> = bytemuck::cast_slice::<f32, u8>(&out.audio).to_vec();
                std::fs::write(dir.join(format!("pcm_got_{spec}.bin")), bytes)?;
            }
            // Hand the previous scenario's scratch back before measuring the
            // next: this process runs sixteen syntheses back to back.
            if pipeline.device().is_gpu() {
                pipeline.device().gpu()?.release_scratch();
            }
            let got_frames = out.generated.len() / 16;
            let mismatches = out.generated.iter().zip(&want_codes).filter(|(a, b)| a != b).count();
            let diff = if want_audio.len() == out.audio.len() {
                common::max_abs_diff(&want_audio, &out.audio)
            } else {
                f32::INFINITY
            };
            let (signal, noise_energy) = want_audio
                .iter()
                .zip(&out.audio)
                .fold((0.0f64, 0.0f64), |(s, n), (a, b)| {
                    (s + (*a as f64) * (*a as f64), n + ((*a - *b) as f64).powi(2))
                });
            let snr = if want_audio.len() == out.audio.len() {
                10.0 * (signal / noise_energy.max(1e-20)).log10()
            } else {
                f64::NEG_INFINITY
            };
            eprintln!("[{name}/{spec}] frames {got_frames}/{frames} (reference), code mismatches                        {mismatches}/{}, SNR {snr:.1} dB, max|diff| {diff:.2e}", want_codes.len());
            if got_frames != frames || mismatches > 0 || snr < 60.0 {
                failures.push(format!("{name}/{spec}: {got_frames}/{frames} frames,                                        {mismatches} code mismatches, {snr:.1} dB"));
            }
        }
        drop(resident);
    }
    assert!(failures.is_empty(), "scenario parity failures:\n  {}", failures.join("\n  "));
    Ok(())
}
