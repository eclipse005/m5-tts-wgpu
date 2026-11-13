"""Acoustic golden dump: prompt codes, LM codes, frame-0 logits and PCM.

These are the numbers the Rust port has to reproduce. Sampling is stochastic, so
the dump pins the reference in two RNG-free ways:

* `pcm` = `decode(codes)` for a fixed code matrix — the hard audio gate;
* `prompt_codes` = `encode(prompt wav)` — deterministic, gates the encoder;
* frame 0 of the LM: backbone hidden `h`, `codebook0_head` logits, and the
  fifteen depth-decoder logits computed from `h` plus the reference's own
  sampled codes — all deterministic, no sampler involved.

Everything is little-endian f32/i64 next to a JSON index, so the Rust tests need
no numpy decoder.

Run with the reference environment:
  python scripts/dump_acoustic.py --out goldens/acoustic
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REFERENCE = os.environ.get("M5_REFERENCE", r"D:\M5-TTS")
sys.path.insert(0, REFERENCE)


def save(name, array, out):
    array = np.ascontiguousarray(array)
    array.tofile(out / f"{name}.bin")
    return {"file": f"{name}.bin", "dtype": str(array.dtype), "shape": list(array.shape)}


def depth_logits(model, last_h, c0, codes):
    """Frame i logits, exactly as `CSM.generate_frame` computes them.

    `codes` are the reference's own samples (already -1 corrected), so this is a
    pure replay: no sampling, no RNG.
    """
    import torch
    from transformers.cache_utils import DynamicCache

    dtype = next(model.parameters()).dtype
    curr_h = torch.cat([last_h.unsqueeze(1), model._embed_audio(0, c0)], dim=1)
    curr_pos = torch.arange(0, curr_h.size(1))[None]
    cache = DynamicCache(config=model.decoder.config)
    out = []
    for i in range(1, model.args.use_codebooks):
        dec_h = model.decoder(
            hidden_states=model.projection(curr_h),
            position_ids=curr_pos,
            attention_mask=None,
            use_cache=True,
            past_key_values=cache,
        ).to(dtype=dtype)
        logits = torch.mm(dec_h[:, -1, :], model.audio_head[i - 1])
        out.append(logits.detach().reshape(-1))
        ci = torch.tensor([[codes[i]]], dtype=c0.dtype)
        curr_h = model._embed_audio(i, ci)
        curr_pos = curr_pos[:, -1:] + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "goldens/acoustic"))
    ap.add_argument("--model", default=os.environ.get("CLONE_MODEL"))
    ap.add_argument("--codec", default=os.environ.get("CODEC"))
    ap.add_argument("--prompt-wav", default=os.path.join(REFERENCE, "assets/aslp_default_voice.wav"))
    ap.add_argument("--prompt-text", default="猪携带该疾病，然后通过蚊子传染给人类。")
    ap.add_argument("--text", default="你好，欢迎使用我们的语音合成模型，今天天气很好。")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch
    from inference import M5TTS, prepare_prompt
    from m5.model.model import CSM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    tts = M5TTS.from_pretrained(args.model, codec=args.codec, device="cpu", attention="sdpa",
                               cache_dir=os.environ.get("M5_CACHE_DIR"), local_files_only=True)
    gen = tts.generator
    model = gen._model

    # Read-only observation: keep every generate_frame call's inputs and result.
    audit = []
    original = CSM.generate_frame

    def spy(self, tokens, tokens_mask, backbone_cache, input_pos, temperature, topk, need_finish, do_sample=False):
        sample = original(self, tokens, tokens_mask, backbone_cache, input_pos, temperature, topk,
                          need_finish, do_sample=do_sample)
        audit.append({"tokens": tokens.detach().clone(), "input_pos": input_pos.detach().clone(),
                      "sample": sample.detach().clone(), "temperature": temperature, "topk": topk})
        return sample

    hidden_capture = {}
    logits_capture = {}
    handles = [
        model.backbone.register_forward_hook(
            lambda _m, _i, o: hidden_capture.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().clone())),
        model.codebook0_head.register_forward_hook(
            lambda _m, _i, o: logits_capture.__setitem__("c0", o.detach().clone())),
    ]
    CSM.generate_frame = spy
    try:
        audio = tts.infer(args.text, prompt_wav=args.prompt_wav, prompt_text=args.prompt_text, lang="Chinese")
        h0 = hidden_capture["h"]
        c0_logits = logits_capture["c0"]
    finally:
        CSM.generate_frame = original
        for handle in handles:
            handle.remove()

    frames = [r["sample"].reshape(-1).cpu().numpy() for r in audit]
    # sampled tokens carry a +1 offset; the reference clamps to [0, 2047] before decoding
    codes = np.clip(np.stack(frames).astype(np.int64) - 1, 0, 2047)
    c0 = np.array([int(frames[0][0])], dtype=np.int64)

    # Depth-decoder replay for frame 0 (deterministic given h and the samples).
    last_h = h0[:, -1, :]
    ci = depth_logits(model, last_h, torch.tensor([[int(frames[0][0])]]), codes[0])

    prompt = prepare_prompt(args.prompt_wav, tts.sample_rate, device="cpu")
    enc = gen._audio_tokenizer.encode(prompt.numpy(), sr=tts.sample_rate)
    prompt_codes = enc.audio_codes[0].cpu().numpy().astype(np.int64)  # (frames, codebooks)

    # Codec stage replay for the same fixed code matrix: `forward` wrapped by hand
    # so every stage boundary is captured (hooks miss `.decode()` entry points).
    codec = gen._audio_tokenizer.model.decoder
    codes_t = torch.from_numpy(codes).long()
    x = codec.quantizer.decode(codes_t.permute(1, 0)[None]) if False else codec.quantizer.decode(codes_t.T[None])
    stage = {"quantized": x}
    stage["pre_conv"] = codec.pre_conv(x).transpose(1, 2)
    stage["transformer"] = codec.pre_transformer(inputs_embeds=stage["pre_conv"]).last_hidden_state
    up = stage["transformer"].permute(0, 2, 1)
    for blocks in codec.upsample:
        for block in blocks:
            up = block(up)
    stage["upsample"] = up
    wav = up
    for block in codec.decoder:
        wav = block(wav)
    stage["wav"] = wav.clamp(min=-1, max=1)
    wavs, sr = gen._audio_tokenizer.decode({"audio_codes": [codes_t]})
    stage = {k: (v.detach() if hasattr(v, "detach") else v) for k, v in stage.items()}
    replay = np.asarray(stage["wav"].reshape(-1), dtype=np.float32)
    api = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    pcm = api
    print(f"replay vs api decode: n={min(replay.size, api.size)} max|diff|="
          f"{np.abs(replay[: min(replay.size, api.size)] - api[: min(replay.size, api.size)]).max():.3e}")

    index = {
        "text": args.text,
        "prompt_text": args.prompt_text,
        "seed": args.seed,
        "sample_rate": int(sr),
        "frame_numbers": [int(r["input_pos"].reshape(-1)[-1]) for r in audit],
        "temperatures": sorted({float(r["temperature"]) for r in audit}),
        "topk": sorted({int(r["topk"]) for r in audit}),
        "tensors": {
            "prompt_codes": save("prompt_codes", prompt_codes, out),
            "codes": save("codes", codes, out),
            "pcm": save("pcm", pcm, out),
            "final_audio": save("final_audio", audio.numpy(), out),
            "codec_quantized": save("codec_quantized", stage["quantized"].numpy(), out),
            "codec_pre_conv": save("codec_pre_conv", stage["pre_conv"].numpy(), out),
            "codec_transformer": save("codec_transformer", stage["transformer"].numpy(), out),
            "codec_upsample": save("codec_upsample", stage["upsample"].numpy(), out),
            "frame0_tokens": save("frame0_tokens", audit[0]["tokens"].reshape(-1).numpy(), out),
            "frame0_pos": save("frame0_pos", audit[0]["input_pos"].reshape(-1).numpy(), out),
            "frame0_h": save("frame0_h", last_h.reshape(-1).numpy(), out),
            "frame0_c0_logits": save("frame0_c0_logits", c0_logits.reshape(-1).numpy(), out),
            **{f"frame0_ci{i}_logits": save(f"frame0_ci{i}_logits", ci[i - 1].numpy(), out)
               for i in range(1, len(ci) + 1)},
        },
        "frame0_tokens_shape": list(audit[0]["tokens"].shape),
        "frame0_sample_c0": int(frames[0][0]),
        "frame0_sample": [int(v) for v in frames[0]],
    }
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"codes {codes.shape}, prompt_codes {prompt_codes.shape}, pcm {pcm.shape} ({pcm.size / sr:.2f}s)")
    print(f"frames {len(audit)}, frame0 c0={index['frame0_sample_c0']} logits {c0_logits.shape}, depth {len(ci)}x{ci[0].shape}")
    print(f"reference audio {audio.numel() / tts.sample_rate:.2f}s; wrote {out / 'index.json'}")


if __name__ == "__main__":
    main()
