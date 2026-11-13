//! Text normalisation: the reference's compiled pynini automata.
//!
//! `text_normalization.py` calls WeTextProcessing, whose whole rule stack per
//! language (pre-processor, cardinal/date/money/time/fraction grammar,
//! post-processor, option flags) is compiled into two OpenFST binaries. The
//! engine below is the same algorithm the previous port uses
//! (ming-3.0-tts-wgpu `src/tn.rs`): `rustfst` reads those binaries, so the
//! tag -> reorder -> verbalise chain runs the identical automaton.
//!
//! What is *not* reusable from that port is its data or its glue: the FSTs have
//! to be built with M5's options (`remove_interjections=False`,
//! `remove_erhua=False`, `traditional_to_simple=False`, `remove_puncts=False`,
//! `full_to_half=False` for zh; library defaults otherwise), WeTextProcessing
//! 1.3.0 owns the per-language token reorder tables, and Ming's talker
//! pre-processor / regex / number_en layers do not exist here.
//!
//! `tn_data/` holds the M5 automata for all six languages (`<prefix>_tagger.fst`,
//! `<prefix>_verbalizer.fst`) plus the reorder tables dumped from the same
//! WeTextProcessing version (`<prefix>_orders.json`).
//!
//! Cantonese still needs the OpenCC `s2hk` conversion, which is not shipped yet.

use anyhow::{bail, Context, Result};
use rustfst::algorithms::compose::compose;
use rustfst::algorithms::tr_compares::ILabelCompare;
use rustfst::algorithms::tr_sort;
use rustfst::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Language code -> FST prefix in `tn_data/`. Mirrors `_LANGUAGE_PACKAGES`.
pub const LANGUAGES: &[(&str, &str)] = &[
    ("zh", "zh_tn"),
    ("en", "en_tn"),
    ("ja", "ja_tn"),
    ("de", "de_tn"),
    ("fr", "fr_tn"),
    ("ko", "ko_tn"),
];

fn string_fst(text: &str) -> VectorFst<TropicalWeight> {
    let mut fst: VectorFst<TropicalWeight> = VectorFst::new();
    let mut prev = fst.add_state();
    fst.set_start(prev).unwrap();
    for b in text.as_bytes() {
        let next = fst.add_state();
        fst.add_tr(
            prev,
            Tr::new(*b as u32, *b as u32, TropicalWeight::one(), next),
        )
        .unwrap();
        prev = next;
    }
    fst.set_final(prev, TropicalWeight::one()).unwrap();
    fst
}

fn best_path_output(fst: &VectorFst<TropicalWeight>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut state = match fst.start() {
        Some(s) => s,
        None => return out,
    };
    loop {
        let trs = fst.get_trs(state).unwrap();
        let trs = trs.trs();
        if trs.is_empty() {
            break;
        }
        let (olabel, nextstate) = (trs[0].olabel, trs[0].nextstate);
        if olabel != 0 {
            out.push(olabel as u8);
        }
        state = nextstate;
    }
    out
}

/// `pynini.escape`: prefix `\`, `[` and `]` with a backslash (verified
/// character by character against pynini 2.1.7 — `goldens/escape.json`).
fn escape(input: &str) -> std::borrow::Cow<'_, str> {
    if !input.contains(['\\', '[', ']']) {
        return std::borrow::Cow::Borrowed(input);
    }
    let mut out = String::with_capacity(input.len() + 8);
    for ch in input.chars() {
        if matches!(ch, '\\' | '[' | ']') {
            out.push('\\');
        }
        out.push(ch);
    }
    std::borrow::Cow::Owned(out)
}

/// One language's tagger/verbalizer pair plus its reorder table.
pub struct Normalizer {
    tagger: VectorFst<TropicalWeight>,
    verbalizer: VectorFst<TropicalWeight>,
    orders: HashMap<String, Vec<String>>,
}

impl Normalizer {
    /// `prefix` is `zh_tn`, `en_tn`, … inside `tn_dir`.
    pub fn load(tn_dir: &Path, prefix: &str) -> Result<Self> {
        let read = |suffix: &str| -> Result<VectorFst<TropicalWeight>> {
            let path = tn_dir.join(format!("{prefix}_{suffix}.fst"));
            VectorFst::<TropicalWeight>::read(&path)
                .with_context(|| format!("read {}", path.display()))
        };
        let mut tagger = read("tagger")?;
        let mut verbalizer = read("verbalizer")?;
        // pynini writes arc-sorted automata; sorting again is cheap and drops
        // the dependency on that (same approach as ming-3.0-tts-wgpu).
        tr_sort(&mut tagger, ILabelCompare {});
        tr_sort(&mut verbalizer, ILabelCompare {});
        let orders_path = tn_dir.join(format!("{prefix}_orders.json"));
        let orders: HashMap<String, Vec<String>> = serde_json::from_str(
            &std::fs::read_to_string(&orders_path)
                .with_context(|| format!("read {}", orders_path.display()))?,
        )
        .with_context(|| format!("parse {}", orders_path.display()))?;
        Ok(Self {
            tagger,
            verbalizer,
            orders,
        })
    }

    fn apply(fst: &VectorFst<TropicalWeight>, input: &str) -> Result<String> {
        let source = string_fst(&escape(input));
        let lattice: VectorFst<TropicalWeight> = compose::<
            TropicalWeight,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
            _,
            _,
        >(&source, fst)?;
        let best: VectorFst<TropicalWeight> = shortest_path_with_config(
            &lattice,
            ShortestPathConfig {
                nshortest: 1,
                unique: true,
                ..Default::default()
            },
        )?;
        Ok(String::from_utf8_lossy(&best_path_output(&best)).into_owned())
    }

    /// `Processor.tag`: escape -> tagger -> shortest path.
    pub fn tag(&self, input: &str) -> Result<String> {
        Self::apply(&self.tagger, input)
    }

    /// `TokenParser.reorder` with this language's table.
    pub fn reorder_tagged(&self, tagged: &str) -> String {
        reorder(tagged, &self.orders)
    }

    /// `Processor.verbalize`: reorder the tagged fields, then verbalise.
    pub fn verbalize(&self, tagged: &str) -> Result<String> {
        Self::apply(&self.verbalizer, &reorder(tagged, &self.orders))
    }

    /// `Processor.normalize`: tag, reorder the tagged fields, verbalise.
    pub fn normalize(&self, input: &str) -> Result<String> {
        if input.is_empty() {
            return Ok(String::new());
        }
        let tagged = Self::apply(&self.tagger, input)?;
        if tagged.is_empty() {
            return Ok(String::new());
        }
        let reordered = reorder(&tagged, &self.orders);
        Self::apply(&self.verbalizer, &reordered)
    }
}

// ---------------------------------------------------------------------------
// TokenParser (`tn/token_parser.py`): reorder tagged fields per category.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct TagToken {
    name: String,
    order: Vec<String>,
    members: Vec<(String, String)>,
}

impl TagToken {
    fn render(&self, orders: &HashMap<String, Vec<String>>) -> String {
        let mut order = self.order.clone();
        if let Some(want) = orders.get(&self.name) {
            let preserve = self
                .members
                .iter()
                .any(|(k, v)| k == "preserve_order" && v == "true");
            if !preserve {
                order = want.clone();
            }
        }
        let mut out = format!("{} {{", self.name);
        for key in &order {
            if let Some((_, value)) = self.members.iter().find(|(k, _)| k == key) {
                out.push_str(&format!(" {}: \"{}\"", key, value));
            }
        }
        out.push_str(" }");
        out
    }
}

/// Parse `name { key: "value" key: "value" }` exactly like `token_parser.py`.
fn parse_tags(input: &str) -> Vec<TagToken> {
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0usize;
    let mut tokens = Vec::new();
    loop {
        while i < chars.len() && chars[i] == ' ' {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }
        let mut name = String::new();
        while i < chars.len() && (chars[i].is_ascii_alphabetic() || chars[i] == '_') {
            name.push(chars[i]);
            i += 1;
        }
        if name.is_empty() {
            break;
        }
        while i < chars.len() && matches!(chars[i], ' ' | '{') {
            i += 1;
        }
        let mut token = TagToken {
            name,
            ..Default::default()
        };
        loop {
            while i < chars.len() && chars[i] == ' ' {
                i += 1;
            }
            if i >= chars.len() {
                break;
            }
            if chars[i] == '}' {
                i += 1;
                break;
            }
            let mut key = String::new();
            while i < chars.len() && (chars[i].is_ascii_alphabetic() || chars[i] == '_') {
                key.push(chars[i]);
                i += 1;
            }
            // `parse_chars(': "')`: at most one of each, in that order. A greedy
            // skip would eat the value when it starts with a space (`value: " "`).
            for sep in [':', ' ', '"'] {
                if i < chars.len() && chars[i] == sep {
                    i += 1;
                }
            }
            let mut value = String::new();
            while i < chars.len() && chars[i] != '"' {
                let ch = chars[i];
                value.push(ch);
                i += 1;
                if ch == '\\' && i < chars.len() {
                    value.push(chars[i]);
                    i += 1;
                }
            }
            if i < chars.len() {
                i += 1;
            }
            token.order.push(key.clone());
            token.members.push((key, value));
        }
        tokens.push(token);
    }
    tokens
}

fn reorder(tagged: &str, orders: &HashMap<String, Vec<String>>) -> String {
    parse_tags(tagged)
        .iter()
        .map(|t| t.render(orders))
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// `normalize_text` (`text_normalization.py`)
// ---------------------------------------------------------------------------

/// Language aliases accepted by the reference (`_LANGUAGES`).
pub fn language_code(lang: &str) -> Option<&'static str> {
    match lang.trim().to_lowercase().as_str() {
        "zh" | "chinese" => Some("zh"),
        "en" | "english" => Some("en"),
        "ja" | "japanese" => Some("ja"),
        "ko" | "korean" => Some("ko"),
        "de" | "german" => Some("de"),
        "fr" | "french" => Some("fr"),
        _ => None,
    }
}

pub struct Tn {
    dir: PathBuf,
}

impl Tn {
    pub fn new(tn_dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: tn_dir.into(),
        }
    }

    /// Default location: `tn_data/` in the crate root (`M5_TN_DIR` overrides).
    pub fn bundled() -> Self {
        if let Ok(dir) = std::env::var("M5_TN_DIR") {
            return Self::new(dir);
        }
        Self::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("tn_data"))
    }

    /// `normalize_text(text, lang=…, accent=…)`. Unsupported languages pass the
    /// text through untouched, exactly like the reference.
    pub fn normalize(&self, text: &str, lang: &str, accent: Option<&str>) -> Result<String> {
        if text.trim().is_empty() {
            bail!("TN requires nonempty text");
        }
        if text.contains("<|") {
            bail!("do not place model control tokens in text passed to TN");
        }
        let Some(code) = language_code(lang) else {
            return Ok(text.to_string());
        };
        let dialect = accent.unwrap_or("").trim().to_lowercase();
        let known_dialect = matches!(dialect.as_str(), "" | "none" | "null" | "chuan" | "yue");
        if code != "zh" || !known_dialect {
            // `supported_dialect` in the reference: only zh takes dialects.
            if !known_dialect {
                return Ok(text.to_string());
            }
        }
        if dialect == "yue" {
            bail!("yue needs the OpenCC s2hk table, which is not shipped yet");
        }
        let prefix = LANGUAGES
            .iter()
            .find(|(c, _)| *c == code)
            .map(|(_, p)| *p)
            .with_context(|| format!("no TN prefix for language {code}"))?;
        let out = Normalizer::load(&self.dir, prefix)?.normalize(text)?;
        if out.trim().is_empty() {
            bail!("WeTextProcessing returned empty or invalid normalized text");
        }
        Ok(out)
    }
}
