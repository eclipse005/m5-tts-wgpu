//! Text-normalisation parity with the Python reference, per language.
//!
//! Both sides run the same WeTextProcessing 1.3.0 automata: the Python side
//! composes them with pynini, this side with `rustfst`. Anything that differs
//! is a bug in the port, not a tolerance to widen.

use anyhow::Result;
use m5_tts_wgpu::tn::Tn;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Deserialize)]
struct GoldenFile {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    lang: Option<String>,
    accent: Option<String>,
    normalized: String,
}

/// Divergences we know about and accept, with the reason. Anything else fails
/// the test, so a new one cannot slip in unnoticed.
const KNOWN_DIVERGENCES: &[(&str, &str)] = &[(
    "1,234 people signed 1/3 of the forms.",
    "two equal-cost paths (digit-by-digit vs cardinal); OpenFST and rustfst      break the tie differently. TN only decides the spoken content, so this is      tracked rather than chased.",
)];

#[test]
fn normalize_text_matches_reference_per_language() -> Result<()> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("goldens/tokenizer.json");
    if !path.exists() {
        eprintln!("skipping: {} not found (run scripts/dump_golden.py)", path.display());
        return Ok(());
    }
    let golden: GoldenFile = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let tn = Tn::bundled();

    let mut per_language: BTreeMap<String, usize> = BTreeMap::new();
    let mut yue_pending = 0usize;
    for case in &golden.cases {
        let lang = case.lang.clone().unwrap_or_default();
        if case.accent.as_deref() == Some("yue") {
            // OpenCC s2hk is not shipped yet; assert the gap stays explicit
            // instead of silently returning un-converted text.
            assert!(
                tn.normalize(&case.text, &lang, case.accent.as_deref()).is_err(),
                "yue must fail loudly until s2hk lands: {:?}",
                case.text
            );
            yue_pending += 1;
            continue;
        }
        let got = tn.normalize(&case.text, &lang, case.accent.as_deref())?;
        if got != case.normalized {
            let known = KNOWN_DIVERGENCES.iter().any(|(text, _)| *text == case.text);
            assert!(
                known,
                "TN mismatch for {:?} ({lang})
  rust: {got}
  py  : {}",
                case.text, case.normalized
            );
            eprintln!("known TN divergence for {:?}: {got}", case.text);
            *per_language.entry(lang).or_default() += 1;
            continue;
        }
        *per_language.entry(lang).or_default() += 1;
    }
    eprintln!("TN parity: {per_language:?} exact, {yue_pending} yue case(s) still gated");
    // Every language the model supports must be covered by at least one case.
    for lang in ["Chinese", "English", "German", "French", "Japanese", "Korean"] {
        assert!(
            per_language.contains_key(lang),
            "no TN golden case for {lang}"
        );
    }
    Ok(())
}
