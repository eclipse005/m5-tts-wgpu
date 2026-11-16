# M5-TTS wgpu

**M5-TTS speech synthesis in Rust with wgpu — 50M multi-token-prediction TTS, zero-shot voice cloning, multi-dialect Chinese, hand-written CPU twin.**

[![Build](https://img.shields.io/github/actions/workflow/status/eclipse005/m5-tts-wgpu/build.yml?branch=main&style=flat-square&label=build)](https://github.com/eclipse005/m5-tts-wgpu/actions/workflows/build.yml) [![GitHub stars](https://img.shields.io/github/stars/eclipse005/m5-tts-wgpu?style=flat-square&logo=github&color=f59e0b)](https://github.com/eclipse005/m5-tts-wgpu/stargazers) [![GitHub forks](https://img.shields.io/github/forks/eclipse005/m5-tts-wgpu?style=flat-square&logo=github)](https://github.com/eclipse005/m5-tts-wgpu/network/members) [![License](https://img.shields.io/github/license/eclipse005/m5-tts-wgpu?style=flat-square&color=16a34a)](LICENSE) [![Rust](https://img.shields.io/badge/Rust-2021-CE422B?style=flat-square&logo=rust&logoColor=white)](#install) [![wgpu](https://img.shields.io/badge/wgpu-30-0F7B6C?style=flat-square)](https://github.com/gfx-rs/wgpu)

[![Windows](https://img.shields.io/badge/Windows-x64-0078D6?style=flat-square)](#install) [![Linux](https://img.shields.io/badge/Linux-x86__64-FCC624?style=flat-square&logo=linux&logoColor=111111)](#install) [![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-111111?style=flat-square&logo=apple&logoColor=white)](#install)

**English** · [简体中文](README.zh-CN.md)

A native Rust implementation of
[M5-TTS-50M](https://huggingface.co/collections/ASLP-lab/m5-tts) — *a mini-scale
multilingual and multi-dialect text-to-speech language model with multi-token
prediction* — using [wgpu](https://github.com/gfx-rs/wgpu) for GPU acceleration.

The goal is simple: run M5-TTS **locally and natively** — no Python, no PyTorch,
no CUDA toolkit at runtime.

### Features

* 🦀 Pure Rust — one binary, nothing else to install
* 🎮 GPU acceleration through wgpu (Vulkan / Metal / GL), one code path for every vendor
* 🖥️ Windows / macOS / Linux
* ⚡ Hand-written CPU backend: no NN framework, just rayon and blocked SIMD kernels
* 🎙️ Zero-shot voice cloning from a prompt recording, or a preset speaker from the CustomVoice bundle
* 🌏 Chinese, English, German, French, Japanese, Korean; Cantonese (`yue`) and Sichuanese (`chuan`) through `--accent`
* 🔤 The reference text normaliser: its compiled pynini FST chain (six languages) plus the OpenCC `s2hk` pass for Cantonese
* 📜 Long scripts: text is cut at sentence and clause boundaries into ~40 s utterances and joined with a uniform 150 ms pause, so a 15-minute narration is one command
* 🎚️ Reproducible: same `--seed` and text produce the same audio, down to the split decisions
* 🧪 Small in-source unit suite, plus the measured parity and RTF tables below

### Install

```bash
git clone https://github.com/eclipse005/m5-tts-wgpu.git
cd m5-tts-wgpu
cargo build --release        # target/release/m5-tts
```

Requires a recent stable Rust (edition 2021) and a GPU driver exposing Vulkan,
Metal or GL — on `--device cpu` not even that. Building does **not** need a CUDA
toolkit, a Python environment or the model itself.

| Dependency | When |
|---|---|
| `tn_data/` | Shipped in this repository: the compiled pynini automata the text normaliser uses. `M5_TN_DIR` overrides the location, `--lang` other than the six supported values passes text through unchanged. |

### Model download

Weights are **not** included in this repository. Download the checkpoints (rights
remain with the original authors) and pass the directories unchanged:

| Bundle | Use |
|---|---|
| [ASLP-lab/M5-TTS-50M-VoiceClone](https://huggingface.co/ASLP-lab/M5-TTS-50M-VoiceClone) | `--prompt-wav` + `--prompt-text` (zero-shot cloning) |
| [ASLP-lab/M5-TTS-50M-CustomVoice](https://huggingface.co/ASLP-lab/M5-TTS-50M-CustomVoice) | `--speaker-id` (preset voice) |
| [ASLP-lab/M5-TTS-Audio-Tokenizer](https://huggingface.co/ASLP-lab/M5-TTS-Audio-Tokenizer) | `--codec`, required by both |

The model directory must contain `model.safetensors`, `config.json`,
`conditioning.json` and `tokenizer/`; the codec directory `config.json` and its
safetensors. `m5-tts inspect` checks both before you spend time on synthesis.

```bash
m5-tts inspect --model ./M5-TTS-50M-CustomVoice --codec ./M5-TTS-Audio-Tokenizer
# language model: llama-30M backbone, llama-20M decoder
# tokenizer: 129992 tokens, conditioning format "m5-caption-accent-speaker-lang-v3"
# tensors: 124 | parameters: 149.9M (the 50M figure counts backbone + decoder)
# bundle ok
```

### Quick start

Preset speaker — load the **CustomVoice** bundle, pick a trained speaker id:

```bash
m5-tts synth \
  --model  ./M5-TTS-50M-CustomVoice \
  --codec  ./M5-TTS-Audio-Tokenizer \
  --speaker-id 1 \
  --text "你好，今天的天气很好，适合出去散散步。" \
  --output ./speaker.wav
```

Clone a voice instead — the **VoiceClone** bundle plus a reference recording and
its exact transcript:

```bash
m5-tts synth \
  --model  ./M5-TTS-50M-VoiceClone \
  --codec  ./M5-TTS-Audio-Tokenizer \
  --text "你好，欢迎使用我们的语音合成模型。" \
  --prompt-wav ./reference.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output ./clone.wav
```

A long script reads from a file instead of the command line, and comes back as
one WAV:

```bash
m5-tts synth --model ./M5-TTS-50M-VoiceClone --codec ./M5-TTS-Audio-Tokenizer \
  --text - --prompt-wav ./reference.wav --prompt-text "…" \
  --output ./bedtime.wav < bedtime.txt
```

The run prints the adapter in use, one line per utterance (characters, semantic
units, frames, duration, units per second), the total and the RTF on **stderr** —
so `--output -` can stream raw 24 kHz mono PCM16 to a player:

```bash
m5-tts synth … --output - | ffplay -f s16le -ar 24000 -ch_layout mono -
```

### Options

| Option | Description |
|---|---|
| `--model <dir>` / `--codec <dir>` | Checkpoint bundles (required) |
| `--text <str>` | Text to speak; `-` reads standard input |
| `--output <file>` | WAV to write; `-` writes raw PCM16 to standard output (default `output.wav`) |
| `--device <name>` | `auto` (default: GPU if usable, else CPU), `cpu`, `gpu`/`cuda` |
| `--prompt-wav <wav>` / `--prompt-text <str>` | Reference recording and its transcript for cloning |
| `--speaker-id <n>` | Preset speaker id (CustomVoice checkpoints), default `1` |
| `--lang <name>` | `Chinese`, `English`, `German`, `French`, `Japanese`, `Korean` |
| `--accent <name>` | `yue` (Cantonese, also converts the text to Traditional), `chuan` (Sichuanese) |
| `--caption <str>` | Style caption, e.g. `happy` |
| `--temperature <f>` / `--topk <n>` | Sampler, defaults `0.9` / `10` (the reference's `infer` defaults) |
| `--seed <n>` | Sampler seed, default `1234` |
| `--max-audio-length-ms <ms>` | Reference semantics: one utterance of at most this long, truncated with a warning when it does not fit. Omit it for long text |

The flag surface matches `infer.py` of the upstream repository apart from the
two length knobs: `--max-audio-length-ms` keeps the reference's meaning, and
omitting it enables long text (below).

### Library

The binary is a thin shell over the crate:

```rust
use m5_tts_wgpu::{Args, Pipeline};

let pipeline = Pipeline::load(
    "M5-TTS-50M-CustomVoice".as_ref(),
    "M5-TTS-Audio-Tokenizer".as_ref(),
    "auto",                       // auto | cpu | gpu
)?;

let args = Args { text: "你好，今天的天气很好。".into(), ..Args::default() };
let out = pipeline.synthesize(&args)?;      // out.audio: Vec<f32> at out.sample_rate
m5_tts_wgpu::audio::write_wav("hello.wav".as_ref(), &out.audio, out.sample_rate)?;
```

`Pipeline::load` uploads the ~1.1 GB of weights once (≈3 s on the card below);
after that a synthesis is a few hundred milliseconds of compute per second of
audio. The pipeline holds `Rc` internally and is **not `Send`**: keep it on the
thread that created it.

### Long text

The model speaks one utterance per pass: its 2048 positions have to hold both the
text and the ~80 ms of audio each text token becomes. The reference lets you ask
for up to 90 s and otherwise truncates at 40 s. This port reads a long script to
its end instead:

* the text is cut **at sentence and clause boundaries** into utterances of about
  one reference utterance (`--max-audio-length-ms`, 40 s by default), decided by
  *semantic length* — one unit per CJK character, one per Latin word — so the
  same text always splits the same way;
* every utterance is a normal pass with the same voice conditioning, so preset and
  cloned voices behave identically, and a text that fits one pass produces
  byte-identical audio to the explicit path;
* seams get a uniform 150 ms pause, because the model's own trailing silence
  ranges from nothing to over a second;
* a fragment that hits the frame cap, or that comes back faster (or slower) than
  speech can be, is halved and redone — truncated or babbled audio is never
  emitted, and only that fragment is repeated.

Measured on a 571-character bedtime script with a cloned voice: 5 utterances,
100.9 s of audio, RTF 0.38, no retries. A 2288-character script: 20 utterances,
556.8 s, RTF 0.38.

### How close is it to the reference?

Every stage was compared against goldens dumped from the Python implementation —
the harness drove the port rather than the runtime, and is not part of this
repository. What it established:

| Stage | Measure | Result |
|---|---|---|
| 31-scenario matrix (both backends, both bundles) | sampled codes, frame by frame | **identical**, 0 mismatching codes |
| decoded PCM per scenario | SNR against the reference waveform | ≥ 60 dB gate, all pass |
| long generation (repeated runs) | bit reproducibility | identical codes and PCM |
| prompt preprocessing | `prepare_prompt` trimming, both silence thresholds | matches, including the 44.1 kHz and stereo sources |
| text normalisation | FST chain, per language | matches the reference's automata |

Speech rate on a Tesla P104 (Vulkan), 7.4 s of audio, prompt included:

| Configuration | RTF ↓ |
|---|---|
| Python reference, PyTorch + CUDA, same card | 0.77 |
| This port, GPU | 0.45 |
| This port, GPU, long script (571 chars) | 0.38 |
| This port, CPU (`--device cpu`) | 0.83 |

RTF is synthesis time divided by generated audio duration; loading is excluded
(the load is ~3 s, of which the weight upload dominates). The card's power state
alone moves a single measurement by up to 2×, so treat these as same-window
comparisons, not absolutes.

### Tests

```bash
cargo test --release         # 18 in-source tests, no assets needed
```

They cover the text front end and the splitter (`long_audio`: lossless
concatenation, budget sweep, sentence-boundary preference), the conditioning
format, the checkpoint contract, bf16 weight handling and the OpenCC tables. The
numerical comparison behind the table above lived in a separate harness and is
gone with the tuning work; what a clean checkout carries is the unit suite, the
shipped `tn_data/`, and the measured numbers.

### Layout

```
src/
  main.rs, lib.rs            CLI (`synth`, `inspect`) and the crate surface
  pipeline.rs                load, prompt encoding, prefill, frame loop, decode
  long_audio.rs              semantic-length splitting for long scripts
  conditioning.rs, tok.rs    the <|caption|>… prompt format, tokenizer
  tn.rs, tn_data/            the reference's compiled pynini automata
  opencc.rs                  s2hk traditional conversion for Cantonese
  audio.rs                   WAV I/O, torchaudio-compatible resampling, prompt trimming
  device.rs, ops.rs          device-independent tensors and the op seam
  gpu.rs, gpu_ops.rs         wgpu device, submit batching, buffer pool
  shaders.rs                 WGSL kernels: GEMV, attention, norms, conv, sampler
  cpu_ops.rs                 hand-written CPU kernels (SIMD + rayon)
  lm/                        backbone (llama-30M), depth decoder (llama-20M), sampler
  codec/                     Qwen 12 Hz speech codec: encoder, SEANet decoder, quantiser
  weights.rs, config.rs      safetensors and the checkpoint contract
```

### Why wgpu?

Instead of depending on CUDA or another vendor-specific runtime, this port uses
**wgpu** as the GPU abstraction, so one Rust binary runs on NVIDIA, AMD, Intel,
Apple and software adapters. Where a kernel is bound by different things on
discrete and integrated GPUs, the tuning is chosen per dispatch shape rather than
per vendor — and on `--device cpu` the port falls back to a hand-written host
implementation of the same pipeline, not to a different model.

DX12 is compiled out on purpose, as in the sibling
[ming-3.0-tts-wgpu](https://github.com/eclipse005/ming-3.0-tts-wgpu) port: wgpu
30's default `dx12` backend pulls `windows = 0.62`, which does not build next to
`gpu-allocator` in a host that pins `windows = 0.61`. Add the feature back if your
host needs it; inference is identical.

### Project status

Functionally complete against `infer.py`: same arguments and checks, cloning and
preset speakers, all six languages plus both dialects, the reference's text
normalisation, and long text as a superset feature. **Streaming is not
implemented**, because the reference does not expose it either — its `generate`
returns one tensor after the decode loop. The ONNX and Android paths of the
upstream repository are not ported, and model directories are read from disk only
(no Hugging Face download).

Known model behaviour, not a porting bug: the 50M LM can lose the thread on a
single very long utterance and drift into unintelligible syllables. That is why
long scripts are cut into reference-sized utterances, and why a fragment whose
duration is outside the range speech can occupy is redone.

### Related

* [M5-TTS](https://github.com/ASLP-lab/M5-TTS) — the official model, reference implementation and demo
* [M5-TTS model collection](https://huggingface.co/collections/ASLP-lab/m5-tts) — checkpoint weights
* [wgpu](https://github.com/gfx-rs/wgpu)
* [ming-3.0-tts-wgpu](https://github.com/eclipse005/ming-3.0-tts-wgpu) — the sibling speech port, same kernel and CPU-backend patterns

### License

Apache-2.0; see [LICENSE](LICENSE).

This repository is an **independent Rust inference implementation** for loading
and running the officially released M5-TTS checkpoints — not an official ASLP-lab
release, and not affiliated with the original authors. Model weights, the
compiled text-normalisation automata and the shippable voices remain under the
terms of their respective owners.
