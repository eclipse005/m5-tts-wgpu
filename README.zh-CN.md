# M5-TTS wgpu

**用 Rust + wgpu 实现 M5-TTS 语音合成 —— 50M 多 token 预测 TTS、零样本音色克隆、中文方言，另带一份手写 CPU 后端。**

[![Build](https://img.shields.io/github/actions/workflow/status/eclipse005/m5-tts-wgpu/build.yml?branch=main&style=flat-square&label=build)](https://github.com/eclipse005/m5-tts-wgpu/actions/workflows/build.yml) [![GitHub stars](https://img.shields.io/github/stars/eclipse005/m5-tts-wgpu?style=flat-square&logo=github&color=f59e0b)](https://github.com/eclipse005/m5-tts-wgpu/stargazers) [![GitHub forks](https://img.shields.io/github/forks/eclipse005/m5-tts-wgpu?style=flat-square&logo=github)](https://github.com/eclipse005/m5-tts-wgpu/network/members) [![License](https://img.shields.io/github/license/eclipse005/m5-tts-wgpu?style=flat-square&color=16a34a)](LICENSE) [![Rust](https://img.shields.io/badge/Rust-2021-CE422B?style=flat-square&logo=rust&logoColor=white)](#安装) [![wgpu](https://img.shields.io/badge/wgpu-30-0F7B6C?style=flat-square)](https://github.com/gfx-rs/wgpu)

[![Windows](https://img.shields.io/badge/Windows-x64-0078D6?style=flat-square)](#安装) [![Linux](https://img.shields.io/badge/Linux-x86__64-FCC624?style=flat-square&logo=linux&logoColor=111111)](#安装) [![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon-111111?style=flat-square&logo=apple&logoColor=white)](#安装)

[English](README.md) · **简体中文**

[M5-TTS-50M](https://huggingface.co/collections/ASLP-lab/m5-tts) 的原生 Rust 实现
——官方定位是*小型、多语言、多方言、带多 token 预测的语音合成语言模型*——GPU 部分用
[wgpu](https://github.com/gfx-rs/wgpu) 加速。

目标很简单：让 M5-TTS **在本地原生跑起来**，运行时不需要 Python、不需要 PyTorch、不需要 CUDA toolkit。

### 特性

* 🦀 纯 Rust —— 一个二进制，不用装别的东西
* 🎮 通过 wgpu 用 GPU（Vulkan / Metal / GL），各家显卡同一条代码路径
* 🖥️ Windows / macOS / Linux
* ⚡ 手写 CPU 后端：不依赖任何神经网络框架，只有 rayon 和分块 SIMD kernel
* 🎙️ 零样本克隆（参考音频 + 转写），或 CustomVoice bundle 的固定音色
* 🌏 中文、英文、德文、法文、日文、韩文；粤语（`yue`）与四川话（`chuan`）用 `--accent` 选择
* 🔤 参考实现同款文本正则：它编译好的 pynini FST 链（六种语言），粤语额外走 OpenCC `s2hk` 转繁体
* 📜 长稿：按句子/分句切成 ~40 s 的 utterance 后用统一 150 ms 停顿拼接，15 分钟口播也是一条命令
* 🎚️ 可复现：同一个 `--seed` 和同一段文本，产出的音频（包括切分决策）完全一致
* 🧪 源码内的小型单测，以及下面实测的对齐表与 RTF 表

### 安装

```bash
git clone https://github.com/eclipse005/m5-tts-wgpu.git
cd m5-tts-wgpu
cargo build --release        # target/release/m5-tts
```

需要较新的 stable Rust（edition 2021）和支持 Vulkan / Metal / GL 的显卡驱动；用
`--device cpu` 连显卡都不需要。编译**不需要** CUDA toolkit、Python 环境或模型权重。

| 依赖 | 什么时候需要 |
|---|---|
| `tn_data/` | 仓库自带：文本正则用的 pynini 编译产物。`M5_TN_DIR` 可覆盖路径；`--lang` 传六种支持语言以外的值则原样跳过正则。 |

### 下载模型

权重**不在**本仓库内。请下载官方 checkpoint（权利归原作者），把目录原样传给 `--model` / `--codec`：

| Bundle | 用途 |
|---|---|
| [ASLP-lab/M5-TTS-50M-VoiceClone](https://huggingface.co/ASLP-lab/M5-TTS-50M-VoiceClone) | `--prompt-wav` + `--prompt-text`（零样本克隆） |
| [ASLP-lab/M5-TTS-50M-CustomVoice](https://huggingface.co/ASLP-lab/M5-TTS-50M-CustomVoice) | `--speaker-id`（固定音色） |
| [ASLP-lab/M5-TTS-Audio-Tokenizer](https://huggingface.co/ASLP-lab/M5-TTS-Audio-Tokenizer) | `--codec`，上面两个都要 |

模型目录需含 `model.safetensors`、`config.json`、`conditioning.json` 和 `tokenizer/`；
codec 目录需含 `config.json` 与其 safetensors。合成之前先自检：

```bash
m5-tts inspect --model ./M5-TTS-50M-CustomVoice --codec ./M5-TTS-Audio-Tokenizer
# language model: llama-30M backbone, llama-20M decoder
# tokenizer: 129992 tokens, conditioning format "m5-caption-accent-speaker-lang-v3"
# tensors: 124 | parameters: 149.9M (the 50M figure counts backbone + decoder)
# bundle ok
```

### 快速开始

固定音色 —— 用 **CustomVoice** bundle，选一个训练过的 speaker id：

```bash
m5-tts synth \
  --model  ./M5-TTS-50M-CustomVoice \
  --codec  ./M5-TTS-Audio-Tokenizer \
  --speaker-id 1 \
  --text "你好，今天的天气很好，适合出去散散步。" \
  --output ./speaker.wav
```

克隆音色 —— 用 **VoiceClone** bundle，外加一段参考录音和它的精确转写：

```bash
m5-tts synth \
  --model  ./M5-TTS-50M-VoiceClone \
  --codec  ./M5-TTS-Audio-Tokenizer \
  --text "你好，欢迎使用我们的语音合成模型。" \
  --prompt-wav ./reference.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output ./clone.wav
```

长稿从文件读进来（`--text -` 走标准输入），输出仍然是一个 WAV：

```bash
m5-tts synth --model ./M5-TTS-50M-VoiceClone --codec ./M5-TTS-Audio-Tokenizer \
  --text - --prompt-wav ./reference.wav --prompt-text "…" \
  --output ./bedtime.wav < bedtime.txt
```

运行时把显卡名字、每段（字数、语义单位、帧数、时长、单位/秒）、总时长和 RTF 打到 **stderr**，
所以 `--output -` 可以直接把 24 kHz 单声道 PCM16 原始流送给播放器：

```bash
m5-tts synth … --output - | ffplay -f s16le -ar 24000 -ch_layout mono -
```

### 参数

| 参数 | 说明 |
|---|---|
| `--model <dir>` / `--codec <dir>` | 两个 bundle 目录（必需） |
| `--text <str>` | 要说的文本；`-` 表示从标准输入读 |
| `--output <file>` | 写 WAV；`-` 表示把 PCM16 原始流写标准输出（默认 `output.wav`） |
| `--device <name>` | `auto`（默认：有可用 GPU 就用，否则 CPU）、`cpu`、`gpu`/`cuda` |
| `--prompt-wav <wav>` / `--prompt-text <str>` | 克隆用的参考录音及其转写 |
| `--speaker-id <n>` | 固定音色的 speaker id（CustomVoice），默认 `1` |
| `--lang <name>` | `Chinese`、`English`、`German`、`French`、`Japanese`、`Korean` |
| `--accent <name>` | `yue`（粤语，同时把文本转繁体）、`chuan`（四川话） |
| `--caption <str>` | 风格标签，例如 `happy` |
| `--temperature <f>` / `--topk <n>` | 采样参数，默认 `0.9` / `10`（与参考 `infer` 默认一致） |
| `--seed <n>` | 采样种子，默认 `1234` |
| `--max-audio-length-ms <ms>` | 参考语义：单条 utterance 最长这么多，装不下就截断并告警。不传则启用长稿模式 |

除两个长度开关外，参数与上游 `infer.py` 对齐：`--max-audio-length-ms` 保留参考含义，
不传它才是长稿模式（下面详述）。

### 作为库使用

命令行只是 crate 的一层薄壳：

```rust
use m5_tts_wgpu::{Args, Pipeline};

let pipeline = Pipeline::load(
    "M5-TTS-50M-CustomVoice".as_ref(),
    "M5-TTS-Audio-Tokenizer".as_ref(),
    "auto",                       // auto | cpu | gpu
)?;

let args = Args { text: "你好，今天的天气很好。".into(), ..Args::default() };
let out = pipeline.synthesize(&args)?;      // out.audio: Vec<f32>，采样率 out.sample_rate
m5_tts_wgpu::audio::write_wav("hello.wav".as_ref(), &out.audio, out.sample_rate)?;
```

`Pipeline::load` 一次性上传约 1.1 GB 权重（下面那张卡上 ≈3 s）；之后每生成 1 秒音频只需几百毫秒。
`Pipeline` 内部持有 `Rc`，**不是 `Send`**：在创建它的线程里使用。

### 长文本

模型一次只能说一条 utterance：2048 个位置要同时装下文本和每个文本 token 对应的 ~80 ms 音频。
参考实现最多让你请求 90 s，否则默认 40 s 截断。本移植会把长稿读完：

* 按**句子与分句边界**切成大约一条参考 utterance（`--max-audio-length-ms`，默认 40 s）的长度，
  用**语义长度**衡量——一个汉字/一个英文单词各算一个单位——因此同一段文本永远切成同一组片段；
* 每个片段都是一次普通合成，条件（固定音色或克隆条件）完全相同，所以两种音色行为一致；
  能装进一次合成的文本，与显式单 pass 路径产出**逐字节相同**的音频；
* 接缝统一 150 ms 停顿，因为模型自己的尾静音从 0 到 1 s 以上不等；
* 撞到帧上限、或语速超出人说话可能范围的片段，会对半切后重做——截断或胡说的音频永远不会输出，
  而且只重做那一段。

实测：571 字的助眠稿 + 克隆音色 → 5 段 / 100.9 s / RTF 0.38，零重做；2288 字 → 20 段 / 556.8 s / RTF 0.38。

### 与参考实现的对齐程度

每个阶段都与 Python 参考导出的 golden 逐个比对过——那套 harness 是拿来驱动移植的，不在本仓库内。
它确立的结论：

| 阶段 | 度量 | 结果 |
|---|---|---|
| 31 个场景矩阵（两种后端、两个 bundle） | 逐帧采样 codes | **完全一致**，0 个不匹配 |
| 各场景解码后的 PCM | 与参考波形的 SNR | ≥ 60 dB 门槛，全部通过 |
| 长时生成（重复运行） | 逐位可复现性 | codes 与 PCM 完全一致 |
| prompt 预处理 | `prepare_prompt` 的裁剪与两个静音阈值 | 一致，含 44.1 kHz 与立体声来源 |
| 文本正则 | 各语言的 FST 链 | 与参考自动机一致 |

速度（Tesla P104 / Vulkan，7.4 s 音频，含 prompt）：

| 配置 | RTF ↓ |
|---|---|
| Python 参考实现，PyTorch + CUDA，同卡 | 0.77 |
| 本移植，GPU | 0.45 |
| 本移植，GPU，长稿（571 字） | 0.38 |
| 本移植，CPU（`--device cpu`） | 0.83 |

RTF = 合成耗时 / 生成音频时长，不含加载（加载 ≈3 s，主要是权重上传）。这张卡的功耗状态本身
就能让同一条命令差 2 倍，所以这些数字只适合同窗口对比，不宜当绝对值。

### 测试

```bash
cargo test --release         # 18 个源码内单测，不需要任何资源
```

覆盖文本前端与切分器（`long_audio`：拼接无损、预算扫、优先句边界）、条件格式、checkpoint 契约、
bf16 权重处理以及 OpenCC 表。上面那张对齐表背后的数值 harness 是独立工具，已随调优工作一起移除；
干净 checkout 里留下的是这套单测、自带的 `tn_data/` 和实测数字。

### 代码结构

```
src/
  main.rs, lib.rs            CLI（`synth`、`inspect`）与 crate 对外表面
  pipeline.rs                加载、prompt 编码、prefill、帧循环、解码
  long_audio.rs              长稿的语义长度切分
  conditioning.rs, tok.rs    <|caption|>… 条件格式、分词器
  tn.rs, tn_data/            参考实现的 pynini 自动机
  opencc.rs                  粤语用的 s2hk 繁体转换
  audio.rs                   WAV 读写、torchaudio 兼容重采样、prompt 裁剪
  device.rs, ops.rs          与设备无关的张量和算子接缝
  gpu.rs, gpu_ops.rs         wgpu 设备、提交批处理、buffer 池
  shaders.rs                 WGSL kernel：GEMV、attention、norm、conv、采样
  cpu_ops.rs                 手写 CPU kernel（SIMD + rayon）
  lm/                        backbone（llama-30M）、depth decoder（llama-20M）、采样器
  codec/                     Qwen 12 Hz 语音 codec：编码器、SEANet 解码器、量化器
  weights.rs, config.rs      safetensors 与 checkpoint 契约
```

### 为什么用 wgpu？

本移植不绑定 CUDA 或任何厂商运行时，而用 **wgpu** 作为 GPU 抽象层，因此同一个 Rust 二进制能跑在
NVIDIA、AMD、Intel、Apple 以及软件适配器上。对离散/集成显卡瓶颈不同的 kernel，按 dispatch 形态
而不是按厂商来选参；`--device cpu` 时退回的是**同一条流水线的手写宿主实现**，不是另一个模型。

和同作者的 [ming-3.0-tts-wgpu](https://github.com/eclipse005/ming-3.0-tts-wgpu) 一样，DX12 被故意
编译掉：wgpu 30 默认的 `dx12` 后端会拉 `windows = 0.62`，在锁 `windows = 0.61` 的宿主里与
`gpu-allocator` 冲突。需要时把 feature 加回来即可，推理结果不变。

### 项目状态

对 `infer.py` 功能完整：参数与校验一致，克隆与固定音色，六种语言加两种方言，参考实现的文本正则，
长文本作为超集特性。**未实现流式**——参考实现也没有暴露流式，它的 `generate` 在解码结束后一次性返回
整段张量。上游的 ONNX 与 Android 路径未移植；模型目录只从本地读取（不做 Hugging Face 下载）。

一个已知的**模型**行为、而非移植缺陷：50M 的 LM 在单条很长的 utterance 上会失去线索，滑向听不懂的
音节。所以长稿会被切成参考量级的 utterance，且时长落在人说话不可能的范围内的片段会被重做。

### 相关

* [M5-TTS](https://github.com/ASLP-lab/M5-TTS) —— 官方模型、参考实现与 demo
* [M5-TTS 模型合集](https://huggingface.co/collections/ASLP-lab/m5-tts) —— 权重
* [wgpu](https://github.com/gfx-rs/wgpu)
* [ming-3.0-tts-wgpu](https://github.com/eclipse005/ming-3.0-tts-wgpu) —— 同作者的另一个语音移植，kernel 与 CPU 后端套路相同

### License

Apache-2.0，见 [LICENSE](LICENSE)。

本仓库是**独立的 Rust 推理实现**，用于加载并运行官方发布的 M5-TTS checkpoint——不是 ASLP-lab 的官方
发布物，也与原作者无关。模型权重、编译好的文本正则自动机以及可发布的音色，权利归各自所有者。
