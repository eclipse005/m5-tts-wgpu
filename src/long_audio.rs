//! Long-audio text splitting.
//!
//! The model speaks one utterance per pass: 2048 positions have to hold both the
//! text and the ~80 ms of audio each text token turns into, so a long script is
//! read as a sequence of utterances, each with the same voice conditioning, and
//! the pieces are joined.
//!
//! Where the cuts go is decided on the text alone - never by predicting how long
//! the audio will be - so the same text always splits the same way. The algorithm
//! follows the ming-3.0 TTS port (`frontend/text_segment_cut.py`), which cuts by
//! *semantic length*: one unit per CJK character, one per Latin word. Cuts are
//! staged from the strongest boundary to the weakest:
//!
//! 1. sentences (。！？, with a hard cut as the last resort),
//! 2. clauses inside a sentence (，;；),
//! 3. an over-long clause is split at its own delimiters, and only a clause with
//!    no delimiter left anywhere is cut between characters.
//!
//! Short tails are merged into the previous fragment rather than left to stand
//! alone (a two-character fragment sounds like a cough, not a sentence), and a
//! fragment that already ends a sentence is not glued to a very short clause.

/// Minimum tail: clauses shorter than this are merged into a neighbour when the
/// result still fits, instead of becoming a fragment of their own.
pub const MIN_TAIL_LENGTH: usize = 5;

/// Characters that end a sentence. A line break ends one too: it is not speech, and
/// it must never reach the model inside a fragment - a newline in the conditioned
/// text makes the 50M model babble instead of reading (measured: a fragment that
/// started on a paragraph break came out as unintelligible syllables).
const SENTENCE_END: [char; 6] = ['。', '！', '？', '!', '?', '\n'];

/// Clauses inside a sentence.
const CLAUSE_END: [char; 4] = ['，', ',', '；', ';'];

/// Longest fragment, in semantic units, for an utterance budget of `frames`.
///
/// The reference implementation's default utterance is 40 s (`max_audio_length_ms`,
/// 500 frames). At the slow end of the narrators measured here (3 units per second,
/// against the 4.2-7.7 that clean readings show) that is `frames * 3 / 12` = 125
/// units for 500 frames; the divisor is 12 rather than 12.5 so the result rounds
/// down. Deliberately conservative: a fragment that overruns its budget is cut
/// mid-word by the model's frame cap, which is the one failure this feature must
/// not produce.
pub fn semantic_budget(frames: usize) -> usize {
    const UNITS_PER_SECOND: usize = 3;
    (frames * UNITS_PER_SECOND / 12).max(MIN_TAIL_LENGTH * 4)
}

/// Semantic length: one per CJK character, one per Latin word.
///
/// A word is a run of letters or digits, and an apostrophe between letters keeps a
/// word together (`don't` is one word), matching the reference splitter.
pub fn semantic_length(text: &str) -> usize {
    let mut count = 0usize;
    let mut in_word = false;
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if is_cjk(ch) {
            count += 1;
            in_word = false;
        } else if ch.is_alphabetic() || ch.is_numeric() {
            if !in_word {
                count += 1;
                in_word = true;
            }
        } else if matches!(ch, '\'' | '\u{2019}')
            && in_word
            && chars.peek().is_some_and(|next| next.is_alphabetic() || next.is_numeric())
        {
            // word-internal apostrophe: the same word continues
        } else {
            in_word = false;
        }
    }
    count
}

fn is_cjk(ch: char) -> bool {
    matches!(ch, '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}' | '\u{F900}'..='\u{FAFF}')
}

/// Collapse every run of whitespace to one space and trim.
///
/// A line break is layout, not speech, and the model has never seen one inside the
/// text it reads: a fragment that started on a paragraph break came out as
/// unintelligible syllables. The single-pass path leaves the text alone (that is
/// what the reference does), so the cleaning belongs to the long-audio split.
pub fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_space = false;
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !in_space && !out.is_empty() {
                out.push(' ');
            }
            in_space = true;
        } else {
            out.push(ch);
            in_space = false;
        }
    }
    out.trim_end().to_string()
}

/// Cut `text` into fragments of at most `max_units` semantic units, at the
/// strongest boundary available.
///
/// Returns the text untouched as a single fragment when it already fits, so a
/// script inside the reference's own limits goes through the single-pass path
/// byte for byte.
pub fn cut(text: &str, max_units: usize) -> Vec<String> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    if text.trim() == text && semantic_length(text) <= max_units {
        return vec![text.to_string()];
    }

    let mut fragments: Vec<String> = Vec::new();
    for sentence in sentences(text) {
        if semantic_length(&sentence) <= max_units {
            append(&mut fragments, sentence, max_units);
            continue;
        }
        for clause in clauses(&sentence) {
            if semantic_length(&clause) <= max_units {
                append(&mut fragments, clause, max_units);
                continue;
            }
            for piece in cut_clause(&clause, max_units) {
                append(&mut fragments, piece, max_units);
            }
        }
    }

    // A short last fragment is a stumble; join it to the one before when that
    // still fits (the same merge `append` does, applied to the final tail).
    if fragments.len() > 1 {
        let tail = semantic_length(fragments.last().unwrap());
        let prev = semantic_length(&fragments[fragments.len() - 2]);
        if tail < MIN_TAIL_LENGTH && prev + tail <= max_units {
            let last = fragments.pop().unwrap();
            fragments.last_mut().unwrap().push_str(&last);
        }
    }
    fragments
}

/// Sentences, keeping the punctuation with the sentence it ends.
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if SENTENCE_END.contains(&ch) {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Clauses inside a sentence, keeping the delimiter with the clause it ends.
fn clauses(sentence: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in sentence.chars() {
        current.push(ch);
        if CLAUSE_END.contains(&ch) {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Split one clause that is over budget: pack its delimited pieces, and only cut
/// between characters when a single piece has no delimiter left.
fn cut_clause(clause: &str, max_units: usize) -> Vec<String> {
    let mut pieces: Vec<String> = Vec::new();
    let mut current = String::new();
    for ch in clause.chars() {
        current.push(ch);
        if CLAUSE_END.contains(&ch) || SENTENCE_END.contains(&ch) || ch == ' ' {
            pieces.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        pieces.push(current);
    }

    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();
    for piece in pieces {
        let units = semantic_length(&piece);
        if units > max_units {
            // No delimiter left: this has to be cut between characters.
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            out.extend(char_windows(&piece, max_units));
            continue;
        }
        if semantic_length(&current) + units > max_units && !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
        current.push_str(&piece);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Break text with no delimiters into `max_units`-sized windows.
fn char_windows(text: &str, max_units: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut units = 0usize;
    for ch in text.chars() {
        let cost = usize::from(is_cjk(ch) || ch.is_alphabetic() || ch.is_numeric());
        if units + cost > max_units && !current.is_empty() {
            out.push(std::mem::take(&mut current));
            units = 0;
        }
        current.push(ch);
        units += cost;
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Add `piece` to the last fragment when both fit, otherwise start a new one.
fn append(fragments: &mut Vec<String>, piece: String, max_units: usize) {
    // Fragments never carry leading or trailing whitespace: the reference strips the
    // text it conditions on, and a fragment boundary can fall inside a run of blanks.
    let piece = piece.trim();
    if piece.is_empty() {
        return;
    }
    let piece = piece.to_string();
    let Some(last) = fragments.last_mut() else {
        fragments.push(piece);
        return;
    };
    let last_units = semantic_length(last);
    let piece_units = semantic_length(&piece);
    // Do not glue a short clause onto a fragment that already ended a sentence.
    let ends_sentence = last.chars().last().is_some_and(|c| SENTENCE_END.contains(&c));
    if last_units + piece_units <= max_units && !(ends_sentence && piece_units < MIN_TAIL_LENGTH) {
        last.push_str(&piece);
    } else {
        fragments.push(piece);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_length_counts_cjk_and_latin_words() {
        assert_eq!(semantic_length("让我们一起"), 5);
        assert_eq!(semantic_length("hello world"), 2);
        assert_eq!(semantic_length("你好 world 2024"), 4);
        assert_eq!(semantic_length("，。！？ "), 0);
        assert_eq!(semantic_length("don't stop"), 2);
    }

    #[test]
    fn short_text_is_returned_untouched() {
        let text = "让我们一起用心去聆听、去感受。";
        assert_eq!(cut(text, 120), vec![text.to_string()]);
    }

    #[test]
    fn clean_collapses_layout() {
        assert_eq!(clean("  你好  \n\n世界\t！ "), "你好 世界 ！");
        assert_eq!(clean("干净文本"), "干净文本");
        assert_eq!(clean("\n\n"), "");
    }

    #[test]
    fn concat_is_lossless() {
        let text = clean(concat!(
            "第一句话。第二句话，还有分句；第三句！英文 too, ok? 结束。",
            "\n\n",
            "这是一个很长很长的句子，长到必须从句子的分句处切开，然后继续切，",
            "直到每一段都不超过预算为止，而拼接回来必须和原文一模一样。"
        ));
        for budget in [5, 8, 12, 20, 40, 200] {
            let pieces = cut(&text, budget);
            // Nothing is lost or reordered; blanks at a cut join up again, because a
            // fragment never keeps leading or trailing whitespace.
            let squeeze = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
            assert_eq!(squeeze(&pieces.concat()), squeeze(&text), "budget {budget}");
            for piece in &pieces {
                assert_eq!(piece.trim(), piece, "budget {budget}: {piece:?} has blank edges");
                assert!(!piece.contains('\n'), "budget {budget}: {piece:?} keeps a line break");
                // Only a single unbreakable unit may exceed the budget.
                assert!(
                    semantic_length(piece) <= budget || !has_delimiter(piece),
                    "budget {budget}: {:?} is {} units",
                    piece,
                    semantic_length(piece)
                );
            }
        }
    }

    fn has_delimiter(text: &str) -> bool {
        text.chars()
            .any(|c| SENTENCE_END.contains(&c) || CLAUSE_END.contains(&c))
    }

    #[test]
    fn cuts_prefer_sentence_boundaries() {
        let text = "第一句有十个字左右。第二句也有十个字左右。第三句同样十个字。";
        let pieces = cut(text, 16);
        assert_eq!(pieces.len(), 3);
        assert!(pieces.iter().all(|p| p.ends_with('。')));
    }

    #[test]
    fn short_tail_is_merged() {
        let text = "这是一个足够长的句子需要被切开。短。";
        let pieces = cut(text, 18);
        assert!(pieces.len() <= 2, "{pieces:?}");
        assert_eq!(pieces.concat(), text);
    }

    #[test]
    fn budget_scales_with_the_utterance() {
        // 40 s (the reference's default) -> 120 units; 90 s -> 270.
        assert_eq!(semantic_budget(500), 125);
        assert_eq!(semantic_budget(1125), 281);
        assert!(semantic_budget(0) >= MIN_TAIL_LENGTH * 4);
    }
}
