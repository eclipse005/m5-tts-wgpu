//! The conditioning prompt format (`conditioning.py`).
//!
//! Every utterance is wrapped in eight control tokens; the reference builds the
//! string and encodes it once, so BOS/EOS are applied by the post-processor and
//! never by hand. `normalize_condition` deliberately maps "None"/"null" to the
//! empty string: missing is not the same as neutral.

use anyhow::{bail, Result};

pub const CONTROL_TOKENS: [&str; 8] = [
    "<|caption_begin|>",
    "<|caption_end|>",
    "<|lang_begin|>",
    "<|lang_end|>",
    "<|accent_begin|>",
    "<|accent_end|>",
    "<|speaker_start|>",
    "<|speaker_end|>",
];

/// Reference field order, used by every prompt and by the metadata check.
pub const FIELD_ORDER: [&str; 5] = ["caption", "accent", "speaker", "lang", "text"];

/// `normalize_condition(value, name)`.
pub fn normalize_condition(value: Option<&str>, name: &str) -> Result<String> {
    let Some(value) = value else {
        return Ok(String::new());
    };
    let value = value.trim();
    if value.eq_ignore_ascii_case("none") || value.eq_ignore_ascii_case("null") {
        return Ok(String::new());
    }
    for marker in CONTROL_TOKENS {
        if value.contains(marker) {
            bail!("{name} contains a reserved conditioning delimiter");
        }
    }
    Ok(value.to_string())
}

/// `format_conditioned_text(text, speaker, accent, lang, caption)`.
pub fn format_conditioned_text(
    text: &str,
    speaker: Option<&str>,
    accent: Option<&str>,
    lang: Option<&str>,
    caption: Option<&str>,
) -> Result<String> {
    for marker in CONTROL_TOKENS {
        if text.contains(marker) {
            bail!("spoken text contains a reserved conditioning delimiter");
        }
    }
    let caption = normalize_condition(caption, "caption")?;
    let accent = normalize_condition(accent, "accent")?;
    let lang = normalize_condition(lang, "lang")?;
    let speaker = normalize_condition(speaker, "speaker")?;
    let speaker = if speaker.is_empty() || speaker == "0" {
        "None".to_string()
    } else {
        speaker
    };
    Ok(format!(
        "<|caption_begin|>{caption}<|caption_end|>\
         <|accent_begin|>{accent}<|accent_end|>\
         <|speaker_start|>{speaker}<|speaker_end|>\
         <|lang_begin|>{lang}<|lang_end|>{text}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_conditions_are_empty() {
        assert_eq!(normalize_condition(None, "x").unwrap(), "");
        assert_eq!(normalize_condition(Some(" None "), "x").unwrap(), "");
        assert_eq!(normalize_condition(Some("NULL"), "x").unwrap(), "");
        assert_eq!(normalize_condition(Some("") , "x").unwrap(), "");
    }

    #[test]
    fn speaker_zero_and_none_become_literal_none() {
        let s = format_conditioned_text("hi", Some("0"), None, Some("Chinese"), None).unwrap();
        assert!(s.contains("<|speaker_start|>None<|speaker_end|>"), "{s}");
        let s = format_conditioned_text("hi", None, None, None, None).unwrap();
        assert!(s.contains("<|speaker_start|>None<|speaker_end|>"), "{s}");
    }

    #[test]
    fn delimiter_order_matches_reference() {
        let s = format_conditioned_text("你好", Some("1"), Some("chuan"), Some("Chinese"), Some("happy"))
            .unwrap();
        assert_eq!(
            s,
            "<|caption_begin|>happy<|caption_end|>\
             <|accent_begin|>chuan<|accent_end|>\
             <|speaker_start|>1<|speaker_end|>\
             <|lang_begin|>Chinese<|lang_end|>你好"
        );
    }
}
