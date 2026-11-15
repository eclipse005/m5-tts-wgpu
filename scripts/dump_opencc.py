"""Extract the OpenCC `s2hk` tables and a validation corpus from the reference env.

Cantonese (`accent=yue`) is not a pronunciation mode: the reference runs the
ordinary Chinese TN and then converts the *script* to Hong Kong traditional with
`opencc-python-reimplemented`'s `s2hk`. That package is a pure-Python
re-implementation over three text dictionaries (`STPhrases`, `STCharacters`,
`HKVariants`), so the port embeds the same tables plus the same algorithm
instead of taking a dependency (`src/opencc.rs`).

This script writes:
  tn_data/opencc/{STPhrases,STCharacters,HKVariants}.txt   (verbatim, ~1 MB)
  tn_data/opencc/validation.tsv                            (input <TAB> expected)

`validation.tsv` covers every `STCharacters` key, a slice of `STPhrases` keys,
and the sentences the TN tests use, all converted by the reference package -
`tests/parity_tn.rs` replays it through the port.

    python scripts/dump_opencc.py [--out tn_data/opencc]
"""

import argparse
import sys
from pathlib import Path

SOURCES = ["STPhrases.txt", "STCharacters.txt", "HKVariants.txt"]

TEST_SENTENCES = [
    "你好，欢迎使用我们的语音合成模型，今天天气很好。",
    "猪携带该疾病，然后通过蚊子传染给人类。",
    "人工智能正在改变我们生活的方方面面。",
    "后发制人，干杯，里面，大家，银行，头发，面条儿。",
    "2024 年 3 月 5 日，我们开了 5 个会。",
    "这本书 3.5 元，那个 12 块。",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tn_data/opencc")
    parser.add_argument("--phrase-keys", type=int, default=8000)
    args = parser.parse_args()

    import opencc
    package = Path(opencc.__file__).parent
    dict_dir = package / "dictionary"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in SOURCES:
        data = (dict_dir / name).read_bytes()
        (out / name).write_bytes(data)
        print(f"{name}: {len(data)} bytes")

    converter = opencc.OpenCC("s2hk")

    inputs = []
    for name in ["STCharacters.txt", "HKVariants.txt"]:
        for line in (dict_dir / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                inputs.append(line.split("\t")[0])
    phrases = [line.split("\t")[0] for line in
               (dict_dir / "STPhrases.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    inputs.extend(phrases[: args.phrase_keys])
    inputs.extend(TEST_SENTENCES)

    seen = set()
    lines = []
    for text in inputs:
        if text in seen:
            continue
        seen.add(text)
        expected = converter.convert(text)
        assert "\t" not in text and "\t" not in expected
        lines.append(f"{text}\t{expected}")
    (out / "validation.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"validation.tsv: {len(lines)} cases")


if __name__ == "__main__":
    sys.exit(main())
