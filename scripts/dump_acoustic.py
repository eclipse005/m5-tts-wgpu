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


def _capture_first_embed(store):
    """Keep the prefill's input embeddings (the first backbone call only)."""
    def hook(_module, _args, kwargs):
        store.setdefault("x", kwargs["hidden_states"].detach().clone())
    return hook


def _capture_first_output(store, key):
    """Keep the first call's output. Must return None: a torch forward hook that
    returns a value *replaces* the module output, which would pin every decode
    step to the prefill's tensor and stop the model from ever emitting EOS."""
    def hook(_module, _args, out):
        store.setdefault(key, (out[0] if isinstance(out, tuple) else out).detach().clone())
    return hook


def _capture_first_hidden(store, index=0):
    """Keep the prefill's output of one transformer layer (first call only)."""
    def hook(_module, _args, out):
        value = out[0] if isinstance(out, tuple) else out
        store.setdefault(index, value.detach().clone())
    return hook


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
    captured = {}
    layer_outs = {}
    hooks = [
        layer.register_forward_hook(_capture_first_hidden(layer_outs, index))
        for index, layer in enumerate(model.decoder.layers)
    ]
    captured["input"] = model.projection(curr_h).detach().clone()
    # Single-layer replay of the depth stack, to bisect a layer vs stack mismatch.
    pos0 = torch.arange(0, curr_h.size(1))[None]
    pos_emb = model.decoder.position_embedding(captured["input"], pos0)
    layer0 = model.decoder.layers[0](
        hidden_states=captured["input"], attention_mask=None, position_ids=pos0,
        position_embeddings=pos_emb, use_cache=False, past_key_value=None)
    captured["layer0"] = (layer0[0] if isinstance(layer0, tuple) else layer0).detach().clone()
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
        if i == 1:
            captured["layer_out"] = dec_h.detach().clone()
        ci = torch.tensor([[codes[i]]], dtype=c0.dtype)
        curr_h = model._embed_audio(i, ci)
        curr_pos = curr_pos[:, -1:] + 1
    for hook in hooks:
        hook.remove()
    captured["layers"] = layer_outs
    return out, captured


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

    # Sampler noise: recorded so the Rust sampler can be replayed without
    # reproducing torch's MT19937 stream. `_multinomial_sample_one_no_sync` draws
    # `q ~ Exp(1)` per call and returns argmax(probs / q).
    import m5.model.model as m5model
    noise_log = []
    original_sample = m5model._multinomial_sample_one_no_sync

    def sample_spy(probs):
        q = torch.empty_like(probs).exponential_(1)
        token = torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)
        if len(noise_log) < 16:
            noise_log.append((q.detach().reshape(-1).clone(), token.detach().reshape(-1).clone()))
        return token

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
    embeds_capture = {}
    layer_capture = {}
    handles = [
        model.backbone.register_forward_hook(_capture_first_output(hidden_capture, "h")),
        model.codebook0_head.register_forward_hook(_capture_first_output(logits_capture, "c0")),
        # The backbone is called with kwargs, so the pre-hook needs with_kwargs.
        model.backbone.register_forward_pre_hook(_capture_first_embed(embeds_capture), with_kwargs=True),
        *[layer.register_forward_hook(_capture_first_hidden(layer_capture, index))
          for index, layer in enumerate(model.backbone.layers)],
    ]
    CSM.generate_frame = spy
    m5model._multinomial_sample_one_no_sync = sample_spy
    try:
        audio = tts.infer(args.text, prompt_wav=args.prompt_wav, prompt_text=args.prompt_text, lang="Chinese")
        h0 = hidden_capture["h"]
        e0 = embeds_capture["x"]
        inv_freq = model.backbone.position_embedding.inv_freq.detach().float().clone()
        c0_logits = logits_capture["c0"]
    finally:
        CSM.generate_frame = original
        m5model._multinomial_sample_one_no_sync = original_sample
        for handle in handles:
            handle.remove()

    frames = [r["sample"].reshape(-1).cpu().numpy() for r in audit]
    # sampled tokens carry a +1 offset; the reference clamps to [0, 2047] before decoding
    codes = np.clip(np.stack(frames).astype(np.int64) - 1, 0, 2047)
    c0 = np.array([int(frames[0][0])], dtype=np.int64)

    # Depth-decoder replay for frame 0 (deterministic given h and the samples).
    last_h = h0[:, -1, :]
    # NB: the depth decoder consumes *tokens* (logit space, 0..2050), not the
    # codec codes (`token - 1`). Feeding codes here silently shifts every
    # embedding by one row.
    sample0 = [int(v) for v in frames[0]]
    ci, depth_captured = depth_logits(model, last_h, torch.tensor([[sample0[0]]]), sample0)

    prompt = prepare_prompt(args.prompt_wav, tts.sample_rate, device="cpu")
    # Encoder stages, replayed step by step so the port can be bisected layer by
    # layer (same idea as the decoder's stage captures).
    with torch.inference_mode():
        encoder = gen._audio_tokenizer.model.encoder
        audio = torch.from_numpy(np.asarray(prompt.numpy(), dtype=np.float32))[None, None, :]
        seanet = encoder.encoder(audio, padding_cache=None)
        transformed = encoder.encoder_transformer(seanet.transpose(1, 2))[0].transpose(1, 2)
        downsampled = encoder.downsample(transformed, padding_cache=None)
        encoded = encoder.quantizer.encode(downsampled, 16).transpose(0, 1)
    encoder_stages = {"prompt_audio": prompt.reshape(1, -1), "prompt_seanet": seanet, "prompt_transformer": transformed,
                      "prompt_downsample": downsampled}
    prompt_codes = encoded.cpu().numpy().astype(np.int64)  # (frames, codebooks)

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

    # Short decode: the naive SEANet companion kernel in the Rust port hits the
    # Windows TDR watchdog on the full 109 440-sample tail, so the GPU pipeline is
    # gated on a short slice as well.
    short = codes[:8]
    # Stage replay for the short slice (the GPU path is bisected on these).
    with torch.inference_mode():
        q_short = codec.quantizer.decode(torch.from_numpy(short).long().T[None])
        pre_short = codec.pre_conv(q_short).transpose(1, 2)
        tr_short = codec.pre_transformer(inputs_embeds=pre_short).last_hidden_state
        up_short = tr_short.permute(0, 2, 1)
        for blocks in codec.upsample:
            for block in blocks:
                up_short = block(up_short)
    short_stages = {"codec_quantized_short": q_short, "codec_pre_conv_short": pre_short,
                    "codec_transformer_short": tr_short, "codec_upsample_short": up_short}
    wavs_short, _ = gen._audio_tokenizer.decode({"audio_codes": [torch.from_numpy(short).long()]})
    pcm_short = np.asarray(wavs_short[0], dtype=np.float32).reshape(-1)

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
            **{k: save(k, v.detach().float().numpy(), out) for k, v in encoder_stages.items()},
            "codes": save("codes", codes, out),
            "pcm": save("pcm", pcm, out),
            "codes_short": save("codes_short", short, out),
            "pcm_short": save("pcm_short", pcm_short, out),
            **{k: save(k, v.detach().float().numpy(), out) for k, v in short_stages.items()},
            "final_audio": save("final_audio", audio.numpy(), out),
            "codec_quantized": save("codec_quantized", stage["quantized"].numpy(), out),
            "codec_pre_conv": save("codec_pre_conv", stage["pre_conv"].numpy(), out),
            "codec_transformer": save("codec_transformer", stage["transformer"].numpy(), out),
            "codec_upsample": save("codec_upsample", stage["upsample"].numpy(), out),
            "frame0_tokens": save("frame0_tokens", audit[0]["tokens"].reshape(-1).numpy(), out),
            "frame0_pos": save("frame0_pos", audit[0]["input_pos"].reshape(-1).numpy(), out),
            "frame0_h": save("frame0_h", last_h.reshape(-1).numpy(), out),
            "frame0_embeds": save("frame0_embeds", e0.reshape(-1).numpy(), out),
            **{f"frame0_layer{i}": save(f"frame0_layer{i}", layer_capture[i].reshape(-1).numpy(), out)
               for i in sorted(layer_capture)},
            "backbone_inv_freq": save("backbone_inv_freq", inv_freq.numpy(), out),
            # Frame-0 sampler noise and the token each draw produced.
            "frame0_depth_input": save("frame0_depth_input", depth_captured["input"].reshape(-1).numpy(), out),
            "frame0_depth_layer_out": save("frame0_depth_layer_out", depth_captured["layer_out"].reshape(-1).numpy(), out),
            "frame0_depth_layer0": save("frame0_depth_layer0", depth_captured["layer0"].reshape(-1).numpy(), out),
            **{f"frame0_depth_stack{i}": save(f"frame0_depth_stack{i}",
                                             depth_captured["layers"][i].reshape(-1).numpy(), out)
               for i in sorted(depth_captured["layers"])},
            **{f"frame0_noise{i}": save(f"frame0_noise{i}", q.numpy(), out)
               for i, (q, _tok) in enumerate(noise_log[:16])},
            "frame0_c0_logits": save("frame0_c0_logits", c0_logits.reshape(-1).numpy(), out),
            **{f"frame0_ci{i}_logits": save(f"frame0_ci{i}_logits", ci[i - 1].numpy(), out)
               for i in range(1, len(ci) + 1)},
        },
        "frame0_tokens_shape": list(audit[0]["tokens"].shape),
        "frame0_sample_c0": int(frames[0][0]),
        "frame0_sample": [int(v) for v in frames[0]],
        "frame0_noise_tokens": [int(tok[0]) for _, tok in noise_log[:16]],
        "frame0_noise_count": len(noise_log),
    }
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"codes {codes.shape}, prompt_codes {prompt_codes.shape}, pcm {pcm.shape} ({pcm.size / sr:.2f}s)")
    print(f"frames {len(audit)}, frame0 c0={index['frame0_sample_c0']} logits {c0_logits.shape}, depth {len(ci)}x{ci[0].shape}")
    print(f"reference audio {audio.numel() / tts.sample_rate:.2f}s; wrote {out / 'index.json'}")


if __name__ == "__main__":
    main()
