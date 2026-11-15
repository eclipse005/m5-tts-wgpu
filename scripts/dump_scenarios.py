"""Scenario goldens: does the port match the reference across the *feature* surface?

`dump_acoustic.py` pins one case (Chinese zero-shot cloning, seed 1234). This one
pins the rest of the feature matrix the reference exposes
(`inference.py::M5TTS.infer` + `app.py`):
preset speakers (CustomVoice), the six advertised languages, both accents, a
caption, temperature/top-k variations, and short/long texts.

Sampling is stochastic, so each scenario records the reference's own noise draws
(one `Exp(1)` vector per codebook per frame, in draw order) next to the codes it
sampled and the PCM it produced. `tests/parity_scenarios.rs` replays that noise
through the port: codes must match exactly, audio within an SNR gate.

Run with the reference environment:
    python scripts/dump_scenarios.py --out goldens/scenarios [--only preset]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REFERENCE = os.environ.get("M5_REFERENCE", r"D:\M5-TTS")
sys.path.insert(0, REFERENCE)

PROMPT_WAV = r"D:\M5-TTS\assets\aslp_default_voice.wav"
# A 44.1 kHz copy of the same recording: the reference resamples it to the codec
# rate (`torchaudio.functional.resample`), which the port carries as
# `audio::resample`. Real reference audio is usually 44.1/48 kHz.
PROMPT_WAV_44K = r"D:\M5-TTS\assets\aslp_default_voice_44k.wav"
# 13.4 s and stereo copies: `prepare_prompt` branches on recording length (>10 s
# takes a 0.02 silence threshold instead of 0.015) and averages the channels.
PROMPT_WAV_LONG = r"D:\M5-TTS\assets\aslp_default_voice_long10s.wav"
PROMPT_WAV_STEREO = r"D:\M5-TTS\assets\aslp_default_voice_stereo.wav"
PROMPT_TEXT = "猪携带该疾病，然后通过蚊子传染给人类。"

# A bedtime script: 571 characters, 498 text tokens, ~90 s of speech. The
# reference only gets this far by raising its 40 s default budget, and even at
# 90 s it ends on the cap without EOS - the case that exercises positions past
# 1024 (the attention kernels' score staging used to overflow there).
LONG_TEXT = (
    "夜深了，城市的灯一盏一盏慢慢安静下来，你也终于有了一点属于自己的时间。"
    "或许今天并不轻松，可能有些事情没有按照期待发展，也可能只是单纯地感到疲惫。"
    "但没关系，现在这一刻，你不需要证明什么，也不需要赶着去完成什么，"
    "只需要好好待在这里，和自己待一会儿。"
    "想象一下，窗外有一点点微风，轻轻地吹动树叶，发出很细很柔的声音。"
    "这样的声音，不急不躁，就像时间本来的样子。"
    "你可以慢慢地呼吸，让空气一点点进入身体，再轻轻地呼出来。"
    "所有紧绷的情绪，都可以随着呼气一点点放松下来。"
    "有时候我们会对自己太严格，总觉得应该更好一点、更快一点、更完美一点。"
    "但其实，你已经走了很远的路。"
    "那些你以为不起眼的坚持，那些别人看不到的努力，都在一点一点地塑造现在的你。"
    "你不需要一直强大，也可以偶尔脆弱，偶尔停下来。"
    "停下来，并不是失败，而是给自己一点空间，好让内心重新变得柔软。"
    "如果你愿意，可以把今天发生的一件小事轻轻地想一想，"
    "也许是一杯温热的水，一句简单的问候，或者是某个不经意的微笑。"
    "这些细碎的温暖，像夜空里不太显眼的星星，但它们一直都在，默默地陪着你。"
    "现在，就这样慢慢地，把肩膀放松下来，让呼吸变得更轻一点。"
    "你已经做得很好了，真的。"
    "接下来的时间，不需要焦虑未来，也不需要回头纠结过去。"
    "就只是这一刻，你是安全的，是被允许安静下来的。"
    "晚安，愿你在柔软的梦里，被温柔地接住。")


def scenarios():
    clone = dict(kind="clone", prompt_wav=PROMPT_WAV, prompt_text=PROMPT_TEXT)
    preset = dict(kind="preset")

    def s(**kw):
        base = dict(clone if kw.pop("kind", "clone") == "clone" else preset)
        base.update(kw)
        return base

    return {
        # --- cloning: text length and language ---
        "clone_zh_short": s(text="你好。"),
        "clone_zh_medium": s(text="你好，欢迎使用我们的语音合成模型，今天天气很好。"),
        "clone_zh_long": s(text=(
            "人工智能正在改变我们生活的方方面面。从语音助手到自动驾驶，"
            "从医疗诊断到金融风控，这项技术在很短的时间里走过了很长的路。"
            "今天，我们想和你聊聊它接下来会走向哪里。")),
        "clone_en": s(lang="English", text=(
            "Hello, this is the speech synthesis model reading an English sentence. "
            "The weather is unusually nice today.")),
        "clone_ja": s(lang="Japanese", text="こんにちは、今日はとてもいい天気ですね。"),
        "clone_de": s(lang="German", text="Hallo, heute ist das Wetter sehr schön."),
        "clone_fr": s(lang="French", text="Bonjour, le temps est très beau aujourd'hui."),
        "clone_ko": s(lang="Korean", text="안녕하세요, 오늘 날씨가 아주 좋네요."),
        # --- accents and captions ---
        "clone_accent_chuan": s(accent="chuan", text="你好，今天天气很好。"),
        # Cantonese: Chinese TN then OpenCC s2hk (script conversion, not
        # Cantonese verbalisation - see src/opencc.rs).
        "clone_accent_yue": s(accent="yue", text="请问，你食咗饭未啊？今天天气很好。"),
        "clone_caption": s(caption="用温柔一点的语气说", text="你好，今天天气很好。"),
        # Reference recording at 44.1 kHz -> resampling path.
        "clone_prompt_44k": s(prompt_wav=PROMPT_WAV_44K, text="你好，今天天气很好。"),
        # --- sampling knobs ---
        "clone_topk1": s(topk=1, text="你好，今天天气很好。"),
        "clone_topk50": s(topk=50, text="你好，今天天气很好。"),
        "clone_temp05": s(temperature=0.5, text="你好，今天天气很好。"),
        # --- long text: 498 tokens and ~90 s, i.e. positions well past 1024 ---
        "clone_zh_verylong": s(text=LONG_TEXT, max_audio_length_ms=90000),
        # --- reference-audio shapes and the app's text-length boundary ---
        "clone_prompt_long10s": s(prompt_wav=PROMPT_WAV_LONG, text="你好，今天天气很好。"),
        "clone_prompt_stereo": s(prompt_wav=PROMPT_WAV_STEREO, text="你好，今天天气很好。"),
        "clone_text_tiny": s(text="好。"),
        "clone_text_250": s(text=(
            "夜深了，城市的灯一盏一盏慢慢安静下来，你也终于有了一点属于自己的时间。"
            "或许今天并不轻松，可能有些事情没有按照期待发展，也可能只是单纯地感到疲惫。"
            "但没关系，现在这一刻，你不需要证明什么，也不需要赶着去完成什么，"
            "只需要好好待在这里，和自己待一会儿，让呼吸慢下来。")),
        "clone_text_nopunct": s(text="今天天气很好我们出去走走吧"),
        # --- conditioning combinations and sampler extremes ---
        "clone_caption_accent": s(accent="chuan", caption="用开心的语气说",
                                  text="你好，今天天气很好。"),
        "clone_temp01": s(temperature=0.1, text="你好，今天天气很好。"),
        "clone_temp15": s(temperature=1.5, text="你好，今天天气很好。"),
        "clone_topk7": s(topk=7, text="你好，今天天气很好。"),
        # --- preset voices (CustomVoice checkpoint) ---
        "preset_1_zh": s(kind="preset", speaker_id=1, text="你好，今天的天气很好。"),
        "preset_2_zh": s(kind="preset", speaker_id=2, text="你好，今天的天气很好。"),
        "preset_1_en": s(kind="preset", speaker_id=1, lang="English",
                         text="Hello, the weather is nice today."),
        "preset_3_zh": s(kind="preset", speaker_id=3, text="你好，今天的天气很好。"),
        "preset_4_zh": s(kind="preset", speaker_id=4, text="你好，今天的天气很好。"),
        "preset_1_ja": s(kind="preset", speaker_id=1, lang="Japanese",
                         text="こんにちは、今日はいい天気ですね。"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=r"D:\m5-tts-wgpu\goldens\scenarios")
    parser.add_argument("--only", default=None, help="substring filter on scenario names")
    parser.add_argument("--skip-existing", action="store_true",
                        help="keep scenarios already dumped (the reference run is minutes each)")
    args = parser.parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    import torch
    from inference import M5TTS
    import m5.model.model as model_mod

    # The bundles are flat local directories (see HANDOVER "weights"), not Hub
    # snapshots, so point at them directly.
    weights = Path(os.environ.get("M5_WEIGHTS", r"D:\M5-TTS\weights"))
    codec = weights / "M5-TTS-Audio-Tokenizer"
    clone = weights / "M5-TTS-50M-VoiceClone"
    custom = weights / "M5-TTS-50M-CustomVoice"

    torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.environ.get("PY_THREADS", "20")))

    record = {"noise": [], "codes": []}
    orig_sample = model_mod._multinomial_sample_one_no_sync
    orig_frame = model_mod.CSM.generate_frame

    def patched_sample(probs):
        # Same draw as the reference, recorded: torch.empty_like(probs).exponential_(1).
        q = torch.empty_like(probs).exponential_(1)
        record["noise"].append(q.reshape(-1).float().cpu().numpy().copy())
        return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)

    def patched_frame(self, *fargs, **fkwargs):
        sampled = orig_frame(self, *fargs, **fkwargs)
        record["codes"].append(sampled.reshape(-1).cpu().numpy().copy())
        return sampled

    model_mod._multinomial_sample_one_no_sync = patched_sample
    model_mod.CSM.generate_frame = patched_frame

    loaded = {}

    def get(kind):
        name = custom if kind == "preset" else clone
        if name not in loaded:
            print(f"loading {name} ...", flush=True)
            loaded[name] = M5TTS.from_pretrained(name, codec=codec, device="cpu",
                                                 attention="sdpa", local_files_only=True)
        return loaded[name]

    todo = scenarios()
    if args.only:
        todo = {k: v for k, v in todo.items() if args.only in k}
    if args.skip_existing:
        todo = {k: v for k, v in todo.items() if not (out_root / k / "meta.json").is_file()}

    summary = []
    for name, spec in todo.items():
        tts = get(spec["kind"])
        record["noise"].clear()
        record["codes"].clear()
        torch.manual_seed(1234)
        kw = dict(text=spec["text"], lang=spec.get("lang", "Chinese"),
                  accent=spec.get("accent"), caption=spec.get("caption"),
                  temperature=spec.get("temperature", 0.9), topk=spec.get("topk", 10),
                  max_audio_length_ms=spec.get("max_audio_length_ms", 40000))
        if spec["kind"] == "clone":
            kw.update(prompt_wav=spec["prompt_wav"], prompt_text=spec["prompt_text"])
        else:
            kw.update(speaker_id=spec["speaker_id"])
        print(f"[{name}] inferring ...", flush=True)
        pcm = tts.infer(**kw)

        codes = np.stack(record["codes"])                      # [frames+1, 16], last row is EOS
        noise = np.stack(record["noise"]).astype(np.float32)    # [draws, 2051]
        audio = pcm.numpy().astype(np.float32)
        # The last recorded row is the EOS frame the reference drops again — unless
        # generation ran into max_audio_length_ms, where every row is a real frame.
        hit_cap = not np.all(codes[-1] == 0)
        real_frames = codes.shape[0] if hit_cap else codes.shape[0] - 1
        if hit_cap:
            print("  note: no EOS (hit max_audio_length_ms); keeping all frames")

        d = out_root / name
        d.mkdir(exist_ok=True)
        np.ascontiguousarray(noise).tofile(d / "noise.bin")
        np.ascontiguousarray(audio).tofile(d / "pcm.bin")
        meta = dict(spec)
        meta.update(name=name, frames=int(real_frames), codebooks=int(codes.shape[1]),
                    max_audio_length_ms=float(kw["max_audio_length_ms"]),
                    draws=int(noise.shape[0]), pcm_samples=int(audio.size),
                    reference=dict(device="cpu", attention="sdpa", seed=1234, dtype="fp32"))
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        # Store code space (token - 1), the convention the codec and the port's
        # `generated` output use; the reference itself appends `sample - 1` and
        # clamps to the codebook range.
        np.ascontiguousarray(np.clip(codes[:real_frames].astype(np.int64) - 1, 0, 2047)).tofile(
            d / "codes.bin")
        summary.append((name, real_frames, audio.size / 24000.0))
        print(f"  {real_frames} frames, {audio.size / 24000.0:.2f} s audio", flush=True)

    model_mod._multinomial_sample_one_no_sync = orig_sample
    model_mod.CSM.generate_frame = orig_frame
    print("\nscenario                frames   audio")
    for name, frames, seconds in summary:
        print(f"  {name:22} {frames:5} {seconds:7.2f}s")


if __name__ == "__main__":
    main()
