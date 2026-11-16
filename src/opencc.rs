//! OpenCC `s2hk`: simplified Chinese -> Hong Kong traditional.
//!
//! The reference implementation only uses OpenCC for `accent=yue`, where it runs
//! the ordinary Chinese text normalisation and then converts the *script*: the
//! model was trained to pair the yue condition with Hong Kong traditional
//! characters, so this is a spelling transform, not a pronunciation mode
//! (`text_normalization.py::normalize_text`).
//!
//! `opencc-python-reimplemented` 0.1.7 is a plain Python loop over three text
//! dictionaries, so the port carries the same tables (`tn_data/opencc/`, ~1 MB)
//! and the same algorithm instead of taking a runtime dependency - the same
//! trade the text normaliser already makes for WeTextProcessing's FSTs.
//!
//! The algorithm, faithfully:
//!   * split the input on OpenCC's separator characters (punctuation, spaces),
//!     converting only the non-separator runs;
//!   * apply the conversion chain `[[STPhrases, STCharacters], [HKVariants]]`,
//!     where a *group* means "first match wins" across its dictionaries;
//!   * inside a dictionary, match the longest key (bounded by the parent match's
//!     length hint) at the leftmost position, take the first candidate, and
//!     recurse into the text on both sides;
//!   * re-serialise the parse tree in order and feed it to the next chain entry.
//!
//! `tests/parity_tn.rs` checks it against the reference package on 12 k strings
//! (`tn_data/opencc/validation.tsv`).

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One OpenCC dictionary: `key -> first candidate`, plus its key-length bounds.
struct Dict {
    /// Longest key in chars (OpenCC measures in characters, not bytes).
    max_len: usize,
    min_len: usize,
    map: HashMap<String, String>,
}

impl Dict {
    fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        let mut map = HashMap::with_capacity(64 * 1024);
        let mut max_len = 1usize;
        let mut min_len = usize::MAX;
        for line in text.lines() {
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            let Some((key, value)) = line.split_once('\t') else {
                bail!("{}: line without a tab: {line:?}", path.display());
            };
            // OpenCC keeps the whole candidate list but only ever uses the first
            // (`value.split(' ')[0]` in the reference).
            let value = value.split(' ').next().unwrap_or(value);
            let key_len = key.chars().count();
            max_len = max_len.max(key_len);
            min_len = min_len.min(key_len);
            map.insert(key.to_string(), value.to_string());
        }
        if map.is_empty() {
            bail!("{}: empty dictionary", path.display());
        }
        Ok(Self { max_len, min_len, map })
    }
}

/// One node of the parse tree: a piece of text, or a matched replacement.
struct Node {
    value: String,
    left: Option<usize>,
    right: Option<usize>,
    /// `length_hint`: a child may not match longer than its parent's match.
    hint: Option<usize>,
    matched: bool,
}

struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn new(text: String) -> Self {
        Self { nodes: vec![Node { value: text, left: None, right: None, hint: None, matched: false }] }
    }

    /// Apply one chain entry (one dictionary, or a group of them) to the tree.
    ///
    /// Grouped dictionaries are applied in order, each only to the nodes the
    /// previous ones left unmatched: that is OpenCC's "first match wins".
    fn apply(&mut self, group: &[&Dict]) {
        let mut working = vec![0usize];
        let mut unmatched: Vec<usize> = Vec::new();
        for dict in group {
            while let Some(current) = working.pop() {
                let text = &self.nodes[current].value;
                let hint = self.nodes[current].hint;
                match Self::find_match(text, dict, hint) {
                    Some((value, left, right, matched_len)) => {
                        self.nodes[current].value = value;
                        self.nodes[current].hint = None;
                        self.nodes[current].matched = true;
                        if let Some(left) = left {
                            self.nodes.push(Node { value: left, left: None, right: None,
                                                   hint: Some(matched_len), matched: false });
                            let index = self.nodes.len() - 1;
                            self.nodes[current].left = Some(index);
                            working.push(index);
                        }
                        if let Some(right) = right {
                            self.nodes.push(Node { value: right, left: None, right: None,
                                                   hint: Some(matched_len), matched: false });
                            let index = self.nodes.len() - 1;
                            self.nodes[current].right = Some(index);
                            working.push(index);
                        }
                    }
                    None => {
                        self.nodes[current].hint = None;
                        unmatched.push(current);
                    }
                }
            }
            std::mem::swap(&mut working, &mut unmatched);
        }
    }

    /// Longest key first, leftmost within a length; returns the replacement and
    /// the text on both sides.
    fn find_match(text: &str, dict: &Dict, hint: Option<usize>)
                  -> Option<(String, Option<String>, Option<String>, usize)> {
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut test_len = len.min(dict.max_len);
        if let Some(hint) = hint {
            test_len = test_len.min(hint);
        }
        while test_len >= dict.min_len {
            if test_len > len {
                test_len -= 1;
                continue;
            }
            for start in 0..=len - test_len {
                let candidate: String = chars[start..start + test_len].iter().collect();
                if let Some(value) = dict.map.get(&candidate) {
                    let left = (start > 0).then(|| chars[..start].iter().collect());
                    let right = (start + test_len < len)
                        .then(|| chars[start + test_len..].iter().collect());
                    return Some((value.clone(), left, right, test_len));
                }
            }
            test_len -= 1;
        }
        None
    }

    /// The tree in order: unmatched text stays as it was.
    fn render(&self) -> String {
        let mut out = String::new();
        // Iterative inorder: left, value, right.
        let mut stack: Vec<usize> = Vec::new();
        let mut current = Some(0usize);
        while current.is_some() || !stack.is_empty() {
            while let Some(index) = current {
                stack.push(index);
                current = self.nodes[index].left;
            }
            let index = stack.pop().expect("stack is non-empty");
            out.push_str(&self.nodes[index].value);
            current = self.nodes[index].right;
        }
        out
    }
}

/// Separator characters of OpenCC's `split_chars_re`; whitespace runs are one
/// separator too, and a conversion never crosses one.
fn is_separator(c: char) -> bool {
    matches!(c,
        '-' | ',' | '.' | '?' | '!' | '*' |
        '　' | '，' | '。' | '、' | '；' | '：' | '？' | '！' | '…' |
        '“' | '”' | '‘' | '’' | '『' | '』' | '「' | '」' | '﹁' | '﹂' |
        '—' | '－' | '（' | '）' | '《' | '》' | '〈' | '〉' | '～' | '．' |
        '／' | '＼' | '︒' | '︑' | '︔' | '︓' | '︿' | '﹀' | '︹' | '︺' |
        '︙' | '︐' | '［' | '﹇' | '］' | '﹈' | '︕' | '︖' | '︰' | '︳' |
        '︴' | '︽' | '︾' | '︵' | '︶' | '｛' | '︷' | '｝' | '︸' | '﹃' |
        '﹄' | '【' | '︻' | '】' | '︼')
}

/// The `s2hk` converter: the phrase/character group, then the HK variants.
pub struct S2hk {
    phrases: Dict,
    characters: Dict,
    hk_variants: Dict,
}

impl S2hk {
    pub fn load(dir: &Path) -> Result<Self> {
        Ok(Self {
            phrases: Dict::load(&dir.join("STPhrases.txt"))?,
            characters: Dict::load(&dir.join("STCharacters.txt"))?,
            hk_variants: Dict::load(&dir.join("HKVariants.txt"))?,
        })
    }

    pub fn bundled() -> Self {
        let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("tn_data/opencc");
        Self::load(&dir).expect("bundled OpenCC tables")
    }

    /// `OpenCC("s2hk").convert(text)`.
    pub fn convert(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut run = String::new();
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            if c.is_whitespace() || is_separator(c) {
                if !run.is_empty() {
                    out.push_str(&self.convert_run(&run));
                    run.clear();
                }
                out.push(c);
                if c.is_whitespace() {
                    // `\s+` is one separator: copy the rest of the run as-is.
                    while let Some(next) = chars.peek() {
                        if next.is_whitespace() {
                            out.push(*next);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }
            } else {
                run.push(c);
            }
        }
        if !run.is_empty() {
            out.push_str(&self.convert_run(&run));
        }
        out
    }

    fn convert_run(&self, text: &str) -> String {
        let mut current = text.to_string();
        // conversion_chain: [ [STPhrases, STCharacters], [HKVariants] ]
        for group in [&[&self.phrases, &self.characters][..], &[&self.hk_variants][..]] {
            let mut tree = Tree::new(current);
            tree.apply(group);
            current = tree.render();
        }
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn converter() -> S2hk {
        S2hk::bundled()
    }

    #[test]
    fn converts_simple_to_hong_kong_traditional() {
        let cc = converter();
        // Expected values come from the reference package (see the dump script).
        assert_eq!(cc.convert("发"), "發");
        assert_eq!(cc.convert("银行"), "銀行");
        assert_eq!(cc.convert("里面"), "裏面");
        assert_eq!(cc.convert("大家"), "大家");
        assert_eq!(cc.convert("后发制人"), "後發制人");
        assert_eq!(cc.convert("你好，今天天气很好。"), "你好，今天天氣很好。");
        // Punctuation runs pass through untouched, and a phrase never spans them.
        assert_eq!(cc.convert("干，发"), "幹，發");
    }

    /// The full 12 k-string corpus the reference produced, when it is present
    /// (`scripts/dump_opencc.py` regenerates it; it is committed).
    #[test]
    fn matches_reference_on_corpus() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tn_data/opencc/validation.tsv");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("skipping: {} missing", path.display());
            return;
        };
        let cc = converter();
        let mut checked = 0usize;
        let mut failures = Vec::new();
        for line in text.lines() {
            let Some((input, expected)) = line.split_once('\t') else { continue };
            let got = cc.convert(input);
            checked += 1;
            if got != expected && failures.len() < 10 {
                failures.push(format!("{input:?}: want {expected:?}, got {got:?}"));
            }
        }
        assert!(checked > 10_000, "corpus too small: {checked}");
        assert!(failures.is_empty(), "{checked} strings, {} mismatches:\n  {}",
                failures.len(), failures.join("\n  "));
    }
}
