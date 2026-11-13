"""Golden dumps from the Python reference (D:\M5-TTS) used by the parity tests.

Run with the environment that runs the reference:
  python scripts/dump_golden.py --out goldens

Everything is written as JSON with plain Rust-friendly shapes so the tests
stay dependency-free.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REFERENCE = os.environ.get("M5_REFERENCE", r"D:\M5-TTS")
sys.path.insert(0, REFERENCE)

CASES = [
    # text, lang, accent, caption, speaker
    ("你好，欢迎使用我们的语音合成模型。", "Chinese", None, None, "0"),
    ("你好，今天的天气很好，适合出去散散步。", "Chinese", None, None, "0"),
    ("我有123元，还没到2026年。", "Chinese", None, None, "0"),
    ("Hello, welcome to our speech synthesis model.", "English", None, None, "0"),
    ("请问，你食咗饭未啊？", "Chinese", "yue", None, "0"),
    ("今天巴适得很哦。", "Chinese", "chuan", None, "0"),
    ("你好，欢迎使用我们的语音合成模型。", "Chinese", None, None, "1"),
    ("你好，欢迎使用我们的语音合成模型。", "Chinese", None, "happy", "1"),
    ("Guten Tag, das Wetter ist schön.", "German", None, None, "0"),
    ("こんにちは、今日はいい天気ですね。", "Japanese", None, None, "0"),
    ("안녕하세요, 오늘 날씨가 좋네요.", "Korean", None, None, "0"),
    ("Bonjour, il fait beau aujourd'hui.", "French", None, None, "0"),
    # TN-heavy cases: cardinal / date / money / time / fraction / measure / ordinal
    ("价格是1234.56元，打八折。", "Chinese", None, None, "0"),
    ("2026年11月13日下午3点05分开会。", "Chinese", None, None, "0"),
    ("电话号码13800138000，房间号305。", "Chinese", None, None, "0"),
    ("三分之一的人同意，另外50%不同意。", "Chinese", None, None, "0"),
    ("今天气温-5℃，湿度百分之五十。", "Chinese", None, None, "0"),
    ("第3章第2节讲了1/2和3.5公里。", "Chinese", None, None, "0"),
    ("I have 3 apples and $12.50, that's 20% off.", "English", None, None, "0"),
    ("It's 3:45 pm on Nov 13, 2026.", "English", None, None, "0"),
    ("1,234 people signed 1/3 of the forms.", "English", None, None, "0"),
    ("2026年11月13日、12時30分です。", "Japanese", None, None, "0"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "goldens"))
    ap.add_argument("--model", default=os.environ.get("CLONE_MODEL"))
    args = ap.parse_args()

    from conditioning import format_conditioned_text
    from inference import load_inference_tokenizer
    from text_normalization import normalize_text

    bundle = Path(args.model)
    if not bundle.is_dir():
        raise SystemExit(f"set --model or CLONE_MODEL to a checkpoint bundle directory ({bundle})")
    tokenizer = load_inference_tokenizer(bundle)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    skipped = []
    for text, lang, accent, caption, speaker in CASES:
        try:
            normalized = normalize_text(text, lang=lang, accent=accent)
        except ImportError as error:
            # Cantonese needs opencc-python-reimplemented; do not install anything
            # into the caller's environment, just report it.
            skipped.append((text, str(error)))
            print(f"SKIP {text!r}: {error}")
            continue
        conditioned = format_conditioned_text(
            normalized, speaker=int(speaker), accent=accent, lang=lang, caption=caption
        )
        ids = tokenizer.encode(conditioned)
        rows.append(
            dict(text=text, lang=lang, accent=accent, caption=caption, speaker=speaker,
                 normalized=normalized, conditioned=conditioned, ids=list(ids))
        )
        print(f"{len(ids):5d} ids  {normalized}")
    (out / "tokenizer.json").write_text(
        json.dumps(dict(vocab_size=len(tokenizer), cases=rows), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"wrote {out / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
