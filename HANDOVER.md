# M5-TTS Rust 移植 — 交接文档

> 写给接手的人（AI 或人）。**以仓库里的代码和测试为准**，本文的"优化建议"只是参考，
> 不是必须遵守的路线。凡是本文与代码冲突的地方，一律以代码和可复现的测试输出为准。

最后更新：见 `git log -1`。当前 HEAD 的提交信息会告诉你最后一步动了什么。

**2025-11 本轮成果**（提交 603eb7c 起）：RTF 1.21 → **0.75-0.87**（配对 best-of-3 0.75，对
Python CPU 1.36 快 45%，**纯 CPU 已追平 Python CUDA 的 0.77**）。十步：①修好 9 倍回归
（`env::var_os` 环境锁）；②大张量双重分配 → lazy 零页；③转置卷积分相分解；④大权重转置卷积
直读原布局（block1 6.2x，512 GFLOP/s）；⑤snake_conv1d/conv1d 时间块 128→512（fork 摊销，
block4 单元 -43%，406-466 GFLOP/s）；⑥GEMM 路径 AVX2 化（pre_transformer -37%）；
⑦norm+qkv+rope+KV+attention 按 head 融合（depth 26.8→25.2 ms/帧）；⑧libm sin 保住 SNR
109dB；⑨depth 预取实验失败复盘；⑩带宽标定 51.8 GB/s @20 线程（读数天花板）。
新增探针：`mem_probe`、`seanet_probe`（双形状 BT A/B）、`bw_probe`、`tconv_probe`、
`gemv_prefetch_probe`。
剩余预算（4.3s 音频样例）：depth ~1.4s（48%）> codec 解码 ~1.45s > encode ~0.33s。
CPU 已近本架构上限：depth 的 570MB/帧权重流是硬地板（L3 30MB 装不下、串行零重用），
再压只剩 fork 微调（~5%）。下一阶段按计划转 wgpu。

---

## 1. 这是什么

把 [ASLP-lab/M5-TTS](https://huggingface.co/ASLP-lab/M5-TTS)（50M 参数、多语言 TTS、零样本克隆）
用 Rust 重写成两个后端：

* **CPU**：手写算子（rayon + AVX2/FMA，带通用回退），不依赖任何 NN 框架
* **GPU**：wgpu 30 compute shader（Vulkan），零 Python 运行时依赖

目标有两个，缺一不可：

1. **和 Python 参考实现数值对齐**（这是硬约束，见第 4 节）
2. 性能超过参考实现（CPU 是主战场；GPU 已探到瓶颈，见第 6 节）

### 路径

| 用途 | 路径 |
|---|---|
| 本移植 | `D:\m5-tts-wgpu` |
| Python 参考实现（原版） | `D:\M5-TTS` |
| 权重（唯一来源，扁平目录、真文件） | `D:\M5-TTS\weights\{M5-TTS-50M-VoiceClone, M5-TTS-50M-CustomVoice, M5-TTS-Audio-Tokenizer}` |
| 参考音频 | `D:\M5-TTS\assets\aslp_default_voice.wav` |
| Python 环境（跑参考、出 golden） | conda `videolingo`：`C:\Users\ADMIN\miniconda3\envs\videolingo\python.exe` |
| 参考实现的环境变量 | `source D:\M5-TTS\outputs\runenv.sh`（内含 `CLONE_MODEL` / `VOICE_MODEL` / `CODEC`） |

权重是**扁平真文件**（`config.json` + `conditioning.json` + `model.safetensors` + `tokenizer/`），
不是 HF 缓存符号链接。VoiceClone 是 **bf16**，CustomVoice 和 codec 是 **f32** —— 这个差别很重要（见 5.2）。

---

## 2. 五分钟上手

```bash
cd /d/m5-tts-wgpu
cargo test --release                 # 9 个套件，全绿是基线；AVX2 路径
M5_NO_SIMD=1 cargo test --release    # 同上，强制通用（无 AVX2）路径，也必须全绿

# 手动跑一段（克隆）
./target/release/m5-tts.exe \
  --model D:/M5-TTS/weights/M5-TTS-50M-VoiceClone \
  --codec D:/M5-TTS/weights/M5-TTS-Audio-Tokenizer \
  --device cpu \
  --text "你好，欢迎使用我们的语音合成模型，今天天气很好。" \
  --prompt-wav D:/M5-TTS/assets/aslp_default_voice.wav \
  --prompt-text "猪携带该疾病，然后通过蚊子传染给人类。" \
  --output outputs/demo.wav

# 预设音色（换权重、去掉 prompt）
./target/release/m5-tts.exe --model D:/M5-TTS/weights/M5-TTS-50M-CustomVoice \
  --codec D:/M5-TTS/weights/M5-TTS-Audio-Tokenizer --device cpu --speaker-id 1 \
  --text "你好，今天的天气很好。" --output outputs/speaker.wav
```

`--device gpu` 也能跑，功能正确但当前**比 CPU 慢**（原因见 6.3）。

---

## 3. 代码结构（8900 行 Rust）

```
src/lib.rs
src/config.rs         config.json / conditioning.json 解析与校验
src/conditioning.rs   <|caption|>… 条件串（对齐 conditioning.py）
src/tok.rs            tokenizer.json + 8 个控制 token，vocab 契约校验
src/tn.rs             文本规范化：rustfst 读 WeTextProcessing 的 OpenFST 自动机
src/weights.rs        safetensors 读取（保留 bf16/f32 原样）+ LM 张量契约校验
src/audio.rs          wav 读写 + prepare_prompt（静音裁剪，对齐 inference.py）

src/device.rs         Device(Gpu|Cpu) + Tensor + Tokens(u32)
src/ops.rs            ★算子缝：所有图只调这里，内部按 Device 分派并做形状校验
src/cpu_ops.rs        ★手写 CPU 实现（AVX2 + 通用回退 + bf16 权重版本）
src/gpu.rs            wgpu 设备（单 compute pass 批量提交、deferred submit）
src/gpu_ops.rs        WGSL 发射器
src/shaders.rs        WGSL 源码（含注释说明每个 kernel 的调优取值与实测）

src/lm/mod.rs         权重/缓存/几何常量
src/lm/backbone.rs    llama-30M 骨干（预填 + KV cache 单步）+ ★融合 CPU 解码层
src/lm/depth.rs       depth 解码器（15 codebook）+ 设备端零回读路径
src/lm/sample.rs      top-k 采样器（对齐 sample_topk + exponential 噪声）

src/codec/mod.rs      codec 配置（encoder_config / decoder_config）
src/codec/quantizer.rs  SplitRVQ 反量化 + 编码（最近邻）
src/codec/transformer.rs pre_transformer（RoPE + 滑窗 72 + SwiGLU）
src/codec/upsample.rs   2×转置卷积 + ConvNeXt
src/codec/seanet.rs     ★SEANet 尾段（生产路径，channels-first）
src/codec/seanet_cl.rs  channels-last 版本（已实现且对齐，但实测更慢，未接线）
src/codec/encoder.rs    codec 编码器（参考音频 → codes）

src/pipeline.rs       ★端到端：text → codes → PCM，含分阶段计时（M5_PROFILE=1）
src/main.rs           CLI

tests/                见第 4 节
goldens/              从参考实现导出的黄金数据（13 MB，已入库）
tn_data/              6 种语言的 OpenFST 自动机 + token 重排表（22 MB，已入库）
scripts/dump_*.py     生成 goldens（用参考实现的 Python 环境跑）
bench/compare.sh      ★成对基准：Rust vs Python，同会话、同线程、best-of-N
examples/*.rs         25 个微基准/诊断程序，每个文件顶部都写了它回答什么问题
```

**分层原则**：`codec`/`lm` 只通过 `ops` 调算子，从不直接碰 wgpu 或 `Vec`。
加一个 stage = 加一个 op（两个后端各实现一次），不是改图。

---

## 4. 数值对齐（硬约束，别动）

### 4.1 黄金数据与测试

`goldens/acoustic/index.json` 记录了参考实现（PyTorch，`device=cpu`, fp32, sdpa）的一次完整推理，
包含：prompt embeddings、backbone 每层输出、`frame0_h`、`c0_logits`、15 组 `ci_logits`、
参考的采样噪声（`noise_all`, 912×2051）、prompt codes、生成的 codes、codec 各阶段张量、PCM。

| 测试 | 内容 | 判据 |
|---|---|---|
| `parity_tok` | 条件串 + token id | 逐字节相等 |
| `parity_tn` | 6 语言 TN（21 例） | 逐字节相等；1 条英文平局差异已登记；粤语尚未支持 |
| `ops_parity` (15) | 算子两两 GPU↔CPU、无 SIMD↔AVX2、转置卷积 vs 朴素参考 | 见各断言 |
| `parity_codec` (3) | codec 解码逐阶段 + 编码器 + prepare_prompt | 阶段 ≤2e-4；PCM **SNR ≥ 60 dB**；codes **0 mismatch** |
| `parity_lm` (4) | backbone 逐层、depth logits、采样回放、KV cache | h ≤1e-4；codes 完全相等 |
| `parity_pipeline` (1) | 端到端（同一噪声流） | 帧数相同、**codes 0 mismatch**、**SNR ≥ 60 dB** |

当前实测：PCM SNR 108~112 dB，生成 codes 与参考逐值相同。

### 4.2 重新生成 golden

```bash
cd /d/M5-TTS && source outputs/runenv.sh
$PYTHON D:/m5-tts-wgpu/scripts/dump_acoustic.py --out D:/m5-tts-wgpu/goldens/acoustic
$PYTHON D:/m5-tts-wgpu/scripts/dump_golden.py  --out D:/m5-tts-wgpu/goldens
```

**踩过的坑（写脚本时必看）**：

1. 用 torch hook 抓张量时，`dict.setdefault` **有返回值**，hook 返回非 None 会**替换模块输出** ——
   表现为"模型永远不吐 EOS、跑满 500 帧"。必须用显式 `return None` 的函数。
2. 同一个 hook 会为每一帧触发；只想要预填那一帧时必须 `setdefault` 语义（抓第一次）。
3. depth 解码器吃的是 **token**（0..2050），不是 codec 的 **code**（token-1）。搞错会整体错一行。
4. `TransformerDecoder.forward` 结尾有 `self.norm`，漏掉会让第一个 codebook 的 logits 差一个量级。
5. `final_audio` 之类的变量名别被后续代码 shadow（我因此存错过一次 prompt 音频）。
6. golden 的 `codes` 最后一行是 **EOS 帧**（全 0），真实生成帧数 = 行数-1。

---

## 5. 当前状态（以代码为准）

### 5.1 功能

全部打通：克隆（参考音频 + 逐字文本）、预设音色（CustomVoice + speaker-id）、
6 语言、accent（chuan 支持；yue 需要 OpenCC s2hk 表，**未做**）、caption、CLI、wav 输出。
流式解码、场景矩阵、README 都**还没做**。

### 5.2 关键设计决定（都是实测出来的，别凭直觉改）

| 决定 | 依据 |
|---|---|
| **bf16 权重留在内存里，kernel 内展开** | VoiceClone 权重本来就是 bf16；lift 成 f32 会让每帧权重流量翻倍。展开是精确移位，结果逐位一致。CustomVoice 是 f32，走 f32 路径 |
| **CPU decode 用融合层**（norm+QKV / o+残差 / norm+gate·up / silu+down） | 一层从 14 次 fork 降到 5 次；`M5_NO_FUSED=1` 可关闭对照 |
| **CPU `zeros` 用懒提交零页，`copy` 并行分块** | 旧实现 `to_vec()` 把 72MB 逐页提交一遍（31ms/张量，单线程）；`vec![0.0; n]` 本身 0ms，页错误摊进写它的算子（452b528） |
| **转置卷积分相分解**（`conv_transpose1d_phased`） | ks=2r 精确等价于 r 个 kernel-2 因果卷积 + 一次交错写；复用 conv1d 的 280 GFLOP/s，替换 80 GFLOP/s 的 scatter 循环（37469b1） |
| **Snake 用 libm `sin`，不用 fast_sin** | snake 是带宽受限，sin 精度免费；fast_sin 掉 23dB SNR（4e49264） |
| **通用路径用 slice-zip 写法** | 等长 slice 的 `zip`/`chunks` 让编译器在**任何架构**的基线 ISA 上自动向量化（x86 SSE2 / ARM NEON）。标量卷积因此从 51.7 → 198 GFLOP/s，只比手写 AVX2 慢 23% |
| **channels-last 未采用**（`seanet_cl.rs` 保留但未接线） | 单独基准 339 vs 248 GFLOP/s，但真实尺寸下 2.66s vs 2.03s 更慢：这里是 DRAM 带宽受限 |
| **单元级时间块融合未采用** | 同进程 A/B 只快 2% |
| **GPU 单 compute pass 批量提交** | 每 dispatch 一个 pass 时 ~20µs/次；改成一个 pass 重放整批后 9.8µs |
| **GPU GEMV 用 warp-per-row** | 16×16 tile 在 m=1 时算 16 行只用 1 行；专用 kernel 80 → 14µs |
| **GPU 提交上限默认 8** | 单个 submit 超过 ~2s 会被 Windows TDR 重置设备（`M5_MAX_PENDING` 可改） |

### 5.3 环境变量（诊断用）

`M5_PROFILE=1`（pipeline/codec 分阶段计时）、`M5_PROFILE_DEEP=1`（block 级）、
`M5_PROFILE_LAYER=1`（融合层内部，f32 与 bf16 都有）、`M5_NO_SIMD=1`（强制通用路径）、
`M5_NO_FUSED=1`、`M5_NO_BF16=1`、`M5_MAX_PENDING=N`、`M5_TN_DIR`、`M5_MODEL`、
`M5_CODEC`、`RAYON_NUM_THREADS`。

---

## 6. 性能现状（务必先读 6.1 再信任何数字）

### 6.1 测量纪律（这一节比数字更重要）

这台机器上有常驻应用（NinjaTrader / Edge / 微信 / node），随时抢核。我的 rayon 默认用满 20 线程，
所以**后台一忙，读数能差 2~17 倍**。实测过的例子：

| 指标 | 空闲时 | 后台 22% 负载时 |
|---|---|---|
| `rms_norm_gemv 768x512` | 40.8 µs | 441 µs |
| 端到端 RTF | 1.28 | 3.5 / 5.07 |

因此：

* **只用 `bash bench/compare.sh <threads> <runs>` 出结论**：它同会话跑 Rust(AVX2) / Rust(通用) / Python(AVX2) / Python(无 AVX)，报 best-of-N
* 任何跨时间点的对比都不可信（我早期测的 PyTorch 1.22 与后来的 1.29 就是两个时刻）
* 报数时写明：提交号、线程数、负载、best-of
* 微基准（`examples/*_bench.rs`）适合定位热点，不适合下结论

### 6.2 最后一次可信的对齐状态

* 端到端 parity（提交 4e49264，`parity_pipeline`）：帧数相同、**codes 0 mismatch**、
  **CPU SNR 109.4 dB / GPU 108.2 dB**
* 两个 SIMD 模式下 9 个测试套件全绿

### 6.3 已解问题与新发现的机制（都实测过，别重蹈覆辙）

**① 9 倍回归已修复**（603eb7c）：`depth_bench` 27.6→275 ms/帧的根因**不是** `dot_bf16_avx2`，
而是 `avx2_available()` 里每次调用 `env::var_os`——它抢进程环境锁，而这个函数在逐行
循环里每帧被调几百万次。检测缓存进 `OnceLock` 后恢复 26.8 ms/帧。教训：**热循环里
永远不要摸环境变量**（`M5_NO_SIMD` 这类开关必须在首次调用时缓存）。

**② 大张量分配曾是隐藏大头**（452b528）：`add`/`snake` 本身 40+ GB/s，慢的是
`Device::Cpu::zeros` 的双重分配——`vec![0.0; n]`（0ms，Windows 懒提交零页）再 `to_vec()`
把 72MB 逐页提交一遍（~31ms，单线程）。修复后：
* `zeros` 直接返回 alloc_zeroed vec，页错误摊进写它的算子（rayon 并行）；
* `copy` 分块并行；`snake_beta` CPU 走 `snake_beta_into` 单遍免拷贝。
残差单元在 block4 尺寸从 205→82 µs。**教训：先量算子还是量分配**（`examples/mem_probe.rs`）。

**③ 转置卷积分相分解**（37469b1）：`ks=2r` 的 ConvTranspose 精确等价于 r 个共享输入的
kernel-2 因果卷积（输出 t=m·r+q 只读 x[m]、x[m-1] 和 tap q、q+r）——相位 q 是一整条
连续 stride-1 卷积，直接复用 conv1d 的 280 GFLOP/s 机器，再加一次分块交错写。
四次尝试 AVX2 化旧 scatter 循环都失败的原因现在清楚了：**换掉循环结构，而不是向量化它**。

**④ fast_sin 会掉 23dB SNR**（4e49264）：fast_sin 进生产路径后 CPU SNR 109.4→85.0
（gate ≥60 仍绿、codes 仍 0 mismatch，但没必要）。`mem_probe` 证明 snake 是带宽受限，
libm sin 与 fast_sin 整段耗时相同——**精度白扔，性能白赚**。快正弦函数已删。

**⑤ depth 解码器的预取实验（已回退，别盲目重试）**：depth 每帧把 38MB bf16 权重
（5 层 × 7.6MB）从 DRAM 重流一遍（26.8 ms/帧 ≈ 21 GB/s），因为 L3 只有 30MB 装不下、
且 15 个 codebook 严格串行零时间局部性。试过：单线程 `_mm_prefetch` worker 预取下一层
qkv+o——单线程 prefetch 流只有 ~13 GB/s（MLP 受限），7.6MB 要 585µs > 层步 330µs；
只预取 1.3MB 的 qkv+o 时 best-of-5 33.0 vs 35.5 ms，**差值在噪声带内，不满足入库标准**。
GEMV min_len 16→64 也无稳定收益。要再攻这个点：先在安静窗口复测（本机负载 20~45% 波动），
方向是多线程预取池或与 gateup/down 的带宽窗口重叠，收益上限 ~7 ms/帧。
`M5_PROFILE_LAYER=1` 现在对 bf16 融合层也能打印五段耗时（gateup 121 / down 71 /
qkv 58 / o 50 / attn 30 µs，见 55f5a80）。

GPU 端：codec 的 GPU 路径仍是 channels-first + 朴素卷积；已实现的
`conv1d_cl`（打包权重 + 8×4 寄存器分块）已验证正确（≤2e-5），速度 120~185 GFLOP/s，
但**还没接进 codec 的 GPU 路径**。CPU 的分相转置卷积思路（③）大概率也适用于 GPU。

### 6.4 与参考实现的对比（配对基准，best-of-3，提交 4e49264）

| | RTF | 说明 |
|---|---|---|
| **Rust AVX2** | **0.94 ~ 0.95** | 比 Python AVX2 快 ~30%，已破实时 |
| Rust 通用（无 AVX2） | 1.10 | 比 Python 通用快 ~31% |
| PyTorch CPU（AVX2 / oneDNN） | 1.36 | 同会话实测 |
| PyTorch CPU（无 AVX） | 1.59 ~ 1.82 | 同会话实测 |
| PyTorch CUDA（P104，fp32） | 0.77 | 仍是 GPU 目标的参照 |

PyTorch 对 ISA 只敏感 17%（它 101 ms/帧里主要是逐帧 kernel 调度开销）；
我们曾对 ISA 敏感 4.4 倍，现在 codec 只差 15%，与原版同量级。

单帧对比（同一会话、同样受后台干扰）：**LM 生成 ~31 ms/帧 vs 原版 101 ms/帧**
（depth 26.8 + backbone ~4）。本轮 RTF 三级跳的会话内读数：1.21 → 1.02（分配修复）
→ 0.93（转置卷积）→ 0.95（libm sin 回退后复测）。

### 6.5 GPU 硬件事实（别再重复测）

```
纯 FMA 探针          2107 GFLOP/s (1.05 TMAC/s)   ← 卡的实际上限
GEMM 2048³           86.8 GFLOP/s                 ← 我的 kernel，4% 上限
conv（打包 CL，8×4）  120~185 GFLOP/s              ← 我的 kernel
dev copy             0.6 GB/s（≈ x1 PCIe，矿卡）
trivial dispatch     9.8 µs
GEMV 512×512         13.9 µs（warp-per-row）
```

cuDNN 在同一张卡上能让 codec 解码跑 0.2s（≈790 GMAC/s = 75% 上限），所以瓶颈是 kernel 而不是卡。
提升方向只有一个：**提高每字节权重的 FMA 次数**（现在 1~2，需要 8+，即 8×8 累加器 + 权重块进 shared）。

---

## 7. 优化建议（**仅供参考**，按性价比排序）

1. **depth 解码器的权重流**（LM 现在是最大头，~45% 端到端）：26.8 ms/帧 ≈ 38MB 权重 /
   DRAM 带宽，L3 30MB 装不下、codebook 严格串行。可行方向：多线程预取池（单线程
   prefetch 只有 ~13 GB/s，试过不够，见 6.3⑤）、或把 gateup+down 的带宽窗口与
   qkv/o 的延迟重叠。收益上限 ~7 ms/帧。**动手前先在安静窗口（负载 <15%）复测**，
   本机负载 20~45% 波动能吞掉这个量级的信号。
2. **LM 在位的每帧开销**：depth_bench 26.8 ms/帧 vs 端到端 31~34 ms/帧，差的 4~7ms
   是 backbone 步 + 采样 + KV 写。还没逐段profile过，先量再动手。
3. **codec block4 残差单元已近 kernel 峰值**（~250-290 GFLOP/s vs conv_bench 244），
   剩余空间有限。下一个 codec 目标是 upsample（0.17s）与 pre_transformer（0.15s）。
4. **GPU 8×8 寄存器分块**（权重复用 8 次/字节）→ 目标 ≥400 GFLOP/s。然后把
   `conv1d_cl` 接进 codec 的 GPU 路径（需要 GPU 版的 snake / 转置卷积 CL kernel）。
   CPU 的分相转置卷积（37469b1）值得原样搬到 GPU。
5. **torch RNG 对齐**：现在 CLI 的 `--seed` 用自研 xoshiro PRNG（可复现，但与
   `torch.manual_seed` 不同流）。要做到"同 seed 与参考逐 token 相同"，需要复刻 torch CPU 的
   MT19937 + `exponential_`（golden 里 `noise_all` 是参考的真实抽样，可用于校验）。
6. **粤语 TN**：从 `opencc-python-reimplemented` 抽 s2hk 表入库。
7. **流式解码**（chunk 300 / left context 25，参考 `chunked_decode`）、**场景矩阵**、
   **README**。

---

## 8. 约束与红线

* **数值对齐优先于性能**：任何优化都必须让第 4 节的测试保持绿；PCM 出过一次静默错误
  （SNR 112 → -9.8 dB），就是被"看着更快"的改动引入的。
* **每个 SIMD 优化都必须同时给出 AVX2 与 `M5_NO_SIMD=1` 两套结果**（RTF + SNR + 测试）。
* 不要往 `D:\M5-TTS` 的 Python 环境里装包（除了已经装好的 WeTextProcessing 1.3.0 / opencc）；
  更不要在 base 环境装东西。
* 权重只从 `D:\M5-TTS\weights` 读；不要恢复 ONNX（那份 1.9 GB 已删，本项目不用 ONNX）。
* 不要按进程名杀进程（本机跑着 pi 自身和多个服务）；清理只针对自己启动的 PID。
* 提交前跑 `cargo test --release` 与 `M5_NO_SIMD=1 cargo test --release`，两者都要 9 套全绿。
