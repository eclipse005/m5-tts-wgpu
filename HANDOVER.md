# M5-TTS Rust 移植 — 交接文档

> 写给接手的人（AI 或人）。**以仓库里的代码和测试为准**，本文的"优化建议"只是参考，
> 不是必须遵守的路线。凡是本文与代码冲突的地方，一律以代码和可复现的测试输出为准。

最后更新：见 `git log -1`。当前 HEAD 的提交信息会告诉你最后一步动了什么。

**2026-11 第三轮（权重常驻 + 上传路径，6.9）**：GPU RTF **0.86 → 0.56**（同会话 CPU 0.75，
PyTorch CUDA 0.77）。要读这一节的**口径**：RTF 现在只计合成（`--device gpu` 的
`in X s, RTF Y` 行），权重在 `Pipeline::load` 阶段上卡，和 Python 基线的
`from_pretrained` 同口径。整进程 wall 5.3~5.5 s（load 2.94 s + 合成 2.4 s），比上一轮
的 5.1 s 略长——**别把 RTF 的降低当成 wall 的降低**。

**2026-11 第二轮（RTF 攻坚，6.8 有完整复盘）**：GPU RTF **2.66 → 0.86**（同会话 CPU 0.83）。
四个已确认的收益：
①**top-k 采样器是 depth 的真正黑洞**（O(vocab²) 排名，2.8 ms × 16 次/帧 = 45 ms，占 59 ms
的 depth 帧）→ 改成基数选择后 **0.12 ms**，depth 59 → 20 ms/帧，这是单项最大的一笔；
②`Device::zeros` 走 PCIe 上传零页（codec 每块 72 MB）→ 改为设备端 fill，codec 5.2 → 2.6 s，
上传量 4508 → 1240 MB；③conv kernel 的 `stage` 工作组内存 36 KB 把占用率压到 2 组/SM →
收到 8 KB，卷积 **500 → 1300-1700 GFLOP/s**（block1-4 的 conv7 全部 -60%）；
④分相转置卷积改跑 reg8 卷积 + 一次 interleave（50 → ~1000 GFLOP/s）。
另有三处小项：host 端权重打包改并行（50 → 2.3 ms/权重）、safetensors f32 读取改 memcpy、
大权重上传分片。融合 depth 路径现在是默认（`M5_DEPTH_PLAIN=1` 回退）。

**2026-11 第一轮（GPU depth 攻坚，结论见 6.7）**：GPU RTF 2.66 → 2.23（唯一确认的收益是
删掉每层 sync）。融合 depth 路径当时查不出收益，**因为采样器的 45 ms/帧把它盖住了**。

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
examples/*.rs         ~35 个微基准/诊断程序，每个文件顶部都写了它回答什么问题
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
| `parity_scenarios` | 16 个功能场景 × 双后端（6.10 有全表） | codes 逐值相等 + SNR ≥ 60 dB |

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
`M5_DEPTH_PLAIN=1`（depth 走逐 op 路径，默认是融合路径）、`M5_DEPTH_ARENA=1`、
`M5_GATEUP_ROWS=N`（norm_gateup 行/workgroup，默认 32）、
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

* 端到端 parity（本轮结束时，`parity_pipeline`）：帧数相同、**codes 0 mismatch**、
  **CPU SNR 110.9 dB / GPU 108.5 dB**
* 两个 SIMD 模式下 9 个测试套件全绿（GPU 侧的改动也在里面：`parity_codec` 的逐阶段
  ≤2e-4、PCM SNR ≥60 dB 都过）

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
但**还没接进 codec 的 GPU 路径**。

## 6.6 wgpu 阶段计划（2025-11 定稿，全 GPU 路线）

**决策**：除调度外一切都在 GPU 上跑，不从 CPU 优势里借性能（0.75 依赖 20 核/51.8GB/s，
换 8 核机器会退化到 ~1.4）。P104-100（Vulkan，PCIe x1 = 0.6GB/s，无 tensor core）。

**实测基线**（98 帧样例，RTF 5.22 vs CPU 0.87）：depth 169.2 ms/帧（9.3s）、backbone
14.5 ms/帧（0.8s）、codec 解码 11.26s、encode+prefill ~1.1s。

**根因（sync 计时修好后的实测，2025-11）**——注意：GPU 分段计时必须 flush+poll 同步
（`Gpu::sync()` 已加），flush 只提交不等待；异步计时的结论（转置卷积 ~7s）是错的：
1. **12 个残差单元合计 ~10.5s，~32-43 GFLOP/s**（峰值的 2%）——最大头。
   naive conv1d kernel 在 block4 单元实测 ~32 GFLOP/s（不是异步计时反推的 115-128）。
2. 4 个转置卷积合计 ~2.5s，~16-25 GFLOP/s。
3. depth 走 host path：每 codebook logits 回读（flush 批队列）→ 16 次同步/帧；每层步
   18 dispatch × 75 层步 ≈ 1350 dispatch。device path 存在但更慢(130 vs 60ms)。
4. PCIe 上传 2.7GB/run（~0.85GB 是权重一次性；33MB/帧 待查，fuse 后自然消失），
   0.6GB/s 下 ≈ 4.5s（可能与计算重叠）。gpu.upload_labels 可按调用点分解。

**理论天花板（融合后）**：depth 8-12 ms/帧（4 dispatch/层 + bf16 上卡 + 全帧单批提交 +
仅 1 次回读）+ backbone 2-3 ms/帧 + codec 0.6-1.0s（cuDNN 同卡 0.2s 是存在性证明）
+ encode 0.15-0.25s → **RTF 0.40-0.55**，好于 CPU 0.75 和 CUDA 0.77 约一倍。

**CPU 机理 → GPU 平移表**：分相转置卷积 → WGSL kernel；qkv_attend per-head 融合 →
一 dispatch 一 WG=head；snake/残差进 conv epilogue；BT=512 的道理 = dispatch 最小化 +
批量提交；bf16 shader 内移位解码；大权重直读布局（消重排）；lazy 零页 → GPU buffer 池。

**执行顺序**：①✅计时基建（Gpu::sync = flush+poll；stats + upload_labels 打印）→
②✅causal_conv CL 绕行（残差单元 32→~100 GFLOP/s，10.5→3.7s；c166fee）→
③✅转置卷积分相 kernel（16-25→~80 GFLOP/s，2.5→1.13s；5d11439；两个失败变体见提交注记：
runtime-indexed 局部数组会溢出 local memory、浅链 8 累加器无收益）→
④depth device 路径融合（161→8-12 ms/帧：4 dispatch/层 + 全帧单批提交 + 仅 1 次回读，
最大剩余项 8.9s）→ ⑤block4 单元占用率（96ch = 1.5 个 CT-tile，0.47s/个）→
⑥pre_transformer GEMM → ⑦8×8 分块/共享内存版转置卷积（当前单累加器 80 GFLOP/s，
共享内存 staging 是 CL conv 120-185 的同构路线）→ ⑧bf16 上卡 + buffer 池。
每步过双设备 parity 门（ops_parity / parity_codec / parity_pipeline）。

**GPU 进度**：RTF 5.22 → **2.74**。分解：causal_conv CL 绕行（残差单元 -65%）+
转置卷积分相（-55%）+ depth 走 device path（每帧 16 次回读 → 1 次；sample_topk 加
noise 行偏移；2-token prefill 设备端拼接）+ **submit 税发现**（每次 submit+wait
~0.5ms 驱动延迟；M5_MAX_PENDING=100000 单独一项就 RTF 3.77→3.02）+ params 池 +
layout 缓存（dispatch 28.7→~10µs 组合实测；非线性残差未解，探针数据在 gpu_probe a-g）。
PCIe 上传 2.7GB 的 33MB/帧部分待 depth 融合时一并处理。
**depth 层融合已落地**（fb244aa）：18→6 dispatch/层（gemv_add / norm_gateup /
silu_down / 轻量 rope+cache+attention，rope 全表一次上卡按 position 索引，
scratch 池免分配）。depth 161 → 69 ms/帧。过程中 parity 门拦下三个 bug
（kv_rows 来源、参数重排后的 head_dim 错位、非法 select）。

**诚实结论**：depth 在 GPU 上是 dispatch 受限——每层 6 个 pipeline 轮换的 dispatch
在位 ~120µs/个（探针单 pipeline 背靠背只有 26µs），kernel 本身合计 ~100µs/层。
要再降只能更激进的 kernel 合并或持久绑定设计；本卡+驱动下 depth 的 GPU 路径
（69ms/帧）仍落后 CPU 融合路径（24ms/帧）约 3 倍。`M5_DEPTH_HOST=1` 可切回
CPU depth（混合模式 RTF ~2.1 vs 全 GPU 2.65）。

**buffer 池已落地**（4700cd6）——"没做的大块"就是它：每个 codec op 新分配 36-72MB
Vulkan buffer 再销毁，分配器碎化后每周期 ~70ms，每 decode ~50 次 = ~3.5s。
Tensor::Gpu 现在携带设备句柄，drop 时归还按长度分池（deferred：batch 打开时
延迟到 flush 后入池，避免 wgpu 同 pass 内 RO→RW hazard——顺带发现了 mega-pass
(100k dispatch) 与任何 buffer 复用根本不兼容，默认 batch 改 64）。
RTF 2.56 → 2.21，codec 解码 6.1 → 4.8s。
同轮：8×8 寄存器分块 conv（341→384 GFLOP/s，注意生成 kernel 时清零块曾误入
cin-chunk 循环导致 silent garbage，parity_codec 拦截）；snake kernel 的
pow(sin,2)→s·s + 预 exp 表（每元素 2 次 exp→零次）；tiled transpose（两侧合并）。
encoder 的 CL 绕行实测变慢已回退（窄通道形状不划算）。

**当前 GPU 分解**（98 帧样例，RTF 2.44 @ 提交 b99ee29）：codec ~4.8s（残差
单元 12 个 conv @384 GFLOP/s + snake/转置/pre_transformer/quantise ~0.75s + 在位
开销），depth ~3.6s（dispatch 受限：6 dispatch × 75 层），encode ~0.9s，prefill
0.3s。双模式 9 套全绿；codes 0 mismatch；SNR 109.4dB。

**arena 迁移状态**：uniform arena + Kernel 缓存 + depth 19-dispatch ring 化 +
冷启动防御全部落地，但 depth arena 路径仍被 wgpu 使用冲突挡住
（`M5_DEPTH_ARENA=1` 实验）。params-arena 实验已回滚（破坏全部 GPU op，
params pool 恢复 ✓）。完整 WIP 保留在 `arena-wip` 分支（2c8bcb6），含
逐张量验证数据。-depth 的 dispatch 税需要 ming 式持久 scratch 的完全等价
实现（其 Op 层的 buffer 生命周期管理与 wgpu tracker 的交互是关键差异）。

下一步 = **移植 ming-3.0-tts-wgpu 的已验证 GPU 架构**（D:\ming-3.0-tts-wgpu，
同款 P104，RTF 0.60 > CUDA——它踩平了我们所有的坑）：

1. **uniform arena + 动态偏移**（gpu.rs:267 arena、398 alloc_uniform、454
   submit_pending：一个 4096×256B uniform buffer，flush 时一次 write_buffer，
   dispatch 用 `set_bind_group(0, &bind, &[offset])` 选槽）——消灭每 dispatch
   的 params buffer；kernel 的 params binding 声明 has_dynamic_offset。
2. **Kernel + run_cached**（nn.rs:107-190）：bind group 按 u64 key 缓存
   （layer<<8|slot），持久 scratch + 不可变权重跨步存活——DiT 24 层 × 17
   dispatch 零对象创建。我们的 depth 5 层 × 6 kernel 完全同构：每层一套
   持久 scratch（5 套），bind group 按 (layer, kernel) 缓存一次。
3. **可靠性常数**（gpu.rs:80-99）：batch 上限 256（≥1024 TDR 掉卡）、
   **首 submit ≤ 8 dispatch 且 drain**（P104 冷启动 >50 直接 device lost）、
   in-flight = 4（前 3 个已移植 ✓）。
4. **required_limits: adapter.limits()**（gpu.rs:226）——取满适配器上限。
5. in-flight 管线（✅已移植）：flush 后非阻塞 poll，CPU 录制与 GPU 执行重叠。

RO→RW pass hazard 的修正认识（**再次修正**，91077a9）：wgpu 的规则是
**READ_WRITE 是独占用法**——同一 pass 内，一个 buffer 被绑定为 RW 后不能再被
任何 dispatch 以任何用法绑定（包括 RO）。之前 mega-pass 的错误与 arena 路径
稳态的错误都是这个规则：**顺序残差链（读→写 per layer）天然违反它**。
ming 的 DiT 不触雷是因为每步 ~110 dispatch 恰好整体落在一个 pass 内、步间
scratch 不跨 pass 复用；我们的 depth 5 层 × 30 dispatch 需要每层结束一个 pass
（sync），或把层间依赖改为 double-buffer 的纯 RW→RO 形式。

arena 路径现状（91077a9）：6 个 arena kernel + 每层持久 scratch + rope 全表 +
双缓冲 norm_out 已落地，per-dispatch-sync 追踪下数值逐位正确；默认路径仍是
已验证的 6-dispatch 形式（69ms/帧），arena 路径在 M5_DEPTH_ARENA=1 之后。
**arena 路径调查（5d04692 → 31fdf02 → 本轮）**：
1. ✅a_rms 参数语义 bug：eps_bits 传到列数槽 → 每 dispatch 1.7s 的越界循环
   （每层首 kernel 1.7s 之谜的谜底）；修后 rms 输出与参考逐位一致（每层每步验证）。
2. ✅a_gemv 漏改 run_params（仍是 run）——run_params/params_dyn 追踪直接定位。
3. ✅RO→RW 冲突根因确认：**池复用的 buffer 跨 pass 的 RO→RW**——
   M5_NO_POOL_REUSE 下 arena 路径完全正确（codes 逐值一致）。
4. ✅depth 持久 scratch/ring 改专用非池化分配（fresh，2KB 级，永不回收）。
5. ⚠️当前剩余（最终诊断）：depth 瞬态 ring 全覆盖 ✓；常驻表 tensor_fresh ✓；
   arena kernel 的 lazy-compile priming ✓（新 pipeline 首 dispatch 后立即
   flush+drain）。**全 pipeline 仍 device-lost 级联**：prefill 的 146-dispatch
   submit 在 encoder 常驻（~840MB）时超过 P104 冷启动阈值（ming 记录阈值
   降至 ~16 dispatch）→ device fault → 后续全 invalid。**修复方向**：提交规模爬坡 8→16→32→64→128→256（每级 drain）已实现
   （gpu.rs maybe_flush 的 ramp 表），encoder 常驻下稳态 256 批仍触发
   device fault——冷窗口比一次 drain 长，与 ming 的"一次 drain 即稳态"
   不同；差异点在 m5 的 encode 阶段实际 dispatch 了 encoder（~500 dispatch
   + 编码器 pipeline 编译），且 buffer 创建与 submit 交错。下一步：对比
   ming 的 dispatch/创建时序，或把 encode 阶段也纳入 ramp 的冷窗口。
   in-flight=4、batch 256 已按 ming 校准。

**风险**：P104 shared memory 48KB/WG（64×64 tile 失败教训在 shaders.rs:756）；dispatch
下限 9.8µs；TDR 2s（M5_MAX_PENDING=8）；VRAM 8GB 充裕；PCIe x1 禁 per-stage 回读。CPU 的分相转置卷积思路（③）大概率也适用于 GPU。

### 6.4 与参考实现的对比（配对基准，best-of-3；GPU 行是 2026-11 第三轮）

| | RTF | 说明 |
|---|---|---|
| **Rust GPU** | **0.56** | 98 帧样例，负载 8%；权重在 `load` 阶段常驻（与 `from_pretrained` 同口径），见 6.9。整进程 wall 5.3 s |
| **Rust AVX2** | **0.75 ~ 0.82** | 20 线程 / 默认线程数，同会话复测（CPU 路径未改动） |
| Rust 通用（无 AVX2） | 1.10 | 比 Python 通用快 ~31% |
| PyTorch CPU（AVX2 / oneDNN） | 1.36 | 同会话实测 |
| PyTorch CPU（无 AVX） | 1.59 ~ 1.82 | 同会话实测 |
| PyTorch CUDA（P104，fp32） | 0.77 | 仍是 GPU 目标的参照 |

PyTorch 对 ISA 只敏感 17%（它 101 ms/帧里主要是逐帧 kernel 调度开销）；
我们曾对 ISA 敏感 4.4 倍，现在 codec 只差 15%，与原版同量级。CPU 那一行的 0.83 是本轮
同会话复测（改动只在 GPU 路径，数字与 6.4 旧值 0.94~0.95 的差别来自后台负载，别跨会话比）。

单帧对比（同一会话、同样受后台干扰）：**LM 生成 ~31 ms/帧 vs 原版 101 ms/帧**
（depth 26.8 + backbone ~4）。本轮 RTF 三级跳的会话内读数：1.21 → 1.02（分配修复）
→ 0.93（转置卷积）→ 0.95（libm sin 回退后复测）。

### 6.5 GPU 硬件事实（别再重复测）

```
纯 FMA 探针              2107 GFLOP/s (1.05 TMAC/s)   ← 卡的实际上限
conv1d_reg8（ks7，四个块） 1148~1669 GFLOP/s            ← 占用率修好后（6.8③）
conv（打包 CL，8×4，旧）   120~185 GFLOP/s              ← 当时被 36 KB 的 stage 卡住
GEMM 2048³               86.8 GFLOP/s                 ← 旧的 16×16 tile，已过时
snake（72 MB，就地）      0.71 ms = 204 GB/s
dev copy（72 MB）         1.0~1.5 GB/s（含池化分配）
上传 write_buffer（复用）  5.9 GB/s；新建 2.5 GB/s；真流程里 75 MB 只有 0.85 GB/s
trivial dispatch         9.8 µs
GEMV 512×512             13.9 µs（warp-per-row）
```

cuDNN 在同一张卡上能让 codec 解码跑 0.2s（≈790 GMAC/s = 75% 上限）。本轮之后卷积 kernel
已经到上限的 55-80%，**剩余空间不在算术而在“首次准备权重”和 DRAM 往返**（6.8⑦）。

### 6.7 2026-11 第一轮：GPU depth 攻坚复盘（结论被 6.8 部分推翻，见 ① 的说明）

**基线**（98 帧样例，seed 1234，`M5_PROFILE=1`）：RTF 2.66，depth 82 ms/帧、codec 解码
5.7s、encode 0.75s、prefill 0.21s；37524 dispatch / **4849 submit**（每层 sync 把批切成
~6 dispatch 的小提交）。

**① 唯一确认的收益：删掉 GPU 融合层的每层 `gpu.sync()`**（`llama_layer_gpu_fused_step`）。
submit 4849→879，RTF 2.66→**2.23**，depth 82→66 ms/帧，`parity_pipeline` 双设备全绿。
**这推翻了 6.6 里"RO→RW 同 pass 独占"的认识**：wgpu-core 30 的 compute pass 是
**逐 dispatch 的 usage scope**（`wgpu-core/src/command/compute.rs:308` 注释明说），
同一 buffer 上一 dispatch 写、下一 dispatch 读完全合法；ming 的 DiT 在同卡上一个
120-dispatch pass 里对共享 scratch 做 RW→RO→RW 就是活证据。之前那堵"墙"是误诊。
（真正的限制只在**单个 dispatch 内**：同一 buffer 不能同时绑成 ro 和 rw 两个 binding。）

**② 新建融合 depth 路径**（`depth_codes_device_fused` +
`llama_layer_depth_fused` + `DepthFusedGpu`）：6 dispatch/层，走 `kernel_params`
（storage params + **缓存 bind group**），持久 per-layer scratch + 持久 fresh KV cache，
cache key = `(layer<<16)|(step<<8)|op`（**step 必须进 key**：per-key params buffer 会在
同批内 key 重复时被覆盖）。`parity_pipeline` 全绿、codes 0 mismatch。**但 depth 仍 ~65 ms/帧，
没比默认快**——缓存 bind group 不是瓶颈。

**③ bf16 常驻权重**（`GEMV_BF16_F32`/`GEMV_ADD_BF16_F32`/`NORM_GATEUP_BF16_F32`/
`SILU_DOWN_BF16_F32`，2-per-u32 打包、shader 内移位展开、**按 f32 的 lane/lane+32 顺序累加**
所以逐位一致）。`DepthFusedGpu.bf16` 存打包权重，层内分支选用。parity 全绿。
**depth 仍 ~67 ms/帧——权重带宽也不是瓶颈**（减半流量没用）。

**④ 死胡同：uniform arena + 动态偏移**（`ArenaKernel` + `AF_*` shader，ming nn.rs 形式）。
隔离探针正确，但**在 depth 上下文里读"常规 op 写过的 fresh buffer"会读到 0/旧值**，
params 经核对完全正确、bind group 新建也一样错。根因未完全隔离，疑似该 Vulkan/P104
驱动对 dynamic-offset uniform binding 的处理 bug。**storage-params 缓存 bind group
（`kernel_params`）则完全正确**——所以融合路径用的是后者。`ArenaKernel`/`AF_*`/
`arena_kernel()` 是死代码，**建议删**。

**⑤ 顺手修的 WIP bug**：上一轮把 `kernel_params` 的 params binding 从 bgl 里删了
（去掉 `,dyn` 时没补 `,ro`），bind group 与 layout 不匹配。已改回 `format!("{kinds},ro")`。

**⑥ 关键诊断数据（别再重复测）**：
* `switch_probe`：单 kernel 重复 11.5 µs/dispatch，3-kernel 轮换 **8.6 µs**——
  **pipeline 切换不贵**。
* `depth_kernel_probe`（隔离、back-to-back）：rms 10.6 / qkv gemv 13.7 / qkv_attend 15.9 /
  gemv_add 11.6 / **norm_gateup 82.3** / silu_down 29.4 µs。一层合计 ~163 µs × 70 层
  ≈ **11.4 ms/帧**。
* **但实测 depth ~65 ms/帧**—— layer kernel 只占 ~11ms，**剩下 ~54ms 没解释**。
  嫌疑：`sample_topk`（14/帧，单 workgroup、O(vocab²) 排名）、`audio_head` GEMV
  （2051×512，14/帧）、以及真实序列里的逐 dispatch 在位开销（隔离探针 back-to-back
  与真实交错序列的差异）。**下一步先隔离量 `sample_topk` 和 `audio_head` 的耗时**，
  别再动 layer kernel（它们已经只占 11ms）。

**当前 GPU 分解**（本轮结束时，RTF 0.86-0.90）：depth 26.4 ms/帧（1.62s generation）>
codec 1.13-1.46s > encode 0.50s > prefill 0.14s。明细见 6.8。

**给接手的优先级建议**（已由 6.8 兑现 1~2、判定 3 部分无效）：
1. ~~codec 是最大头~~ → 已做（占用率 + zeros 上传 + 分相转置卷积，见 6.8）。
2. ~~depth 先量 `sample_topk`/`audio_head`~~ → 量了，44 ms 在采样器（6.8①）。
3. 清理：删 `ArenaKernel`/`AF_*`/`arena_kernel`（死代码）；删旧 arena 路径
   `llama_layer_depth_arena`/`DepthBf16Weights`/`a_gemv_bf16w`/`M5_DEPTH_ARENA`
   （被融合路径取代，且旧路径 down_proj 的 in_features 误传 HIDDEN 是 bug）。尚未做。
4. 双 SIMD 门已补齐：`cargo test --release` 与 `M5_NO_SIMD=1 cargo test --release` 均 9 套全绿。

**新增探针**：`arena_probe`（uniform arena 机制隔离）、`depth_compare`（default vs fused
逐帧 codes 对比）、`depth_kernel_probe`（各融合 kernel 隔离耗时）、`switch_probe`
（pipeline 切换惩罚）、`depth_frame_probe`（整帧 depth 耗时 + dispatch/submit 计数）、
`conv_block_probe`（四个 decoder 块的卷积形状）、`snake_probe`（snake 是否带宽受限）、
`gpu_alloc_probe`（大 scratch buffer 创建成本）、`upload_probe`（上传路径带宽）、
`tconv_gpu_probe`（分相转置卷积三步拆分）。

---

## 6.8 2026-11 第二轮：GPU RTF 2.66 → 0.86 的复盘

**测法**：所有结论都来自同一会话内的 A/B 或 `M5_PROFILE`/`M5_PROFILE_DEEP` 的分段读数；
机器负载 9~40% 波动，因此下面的绝对值只代表“同一会话内的相对变化”。

**① 采样器是 depth 的黑洞（单项最大收益，45 ms/帧）**
`SAMPLE_TOPK_F32` 用“每个元素和全部元素比一遍”求 top-k cutoff：vocab 2051，单 workgroup
256 轨，每轨 8×2051 次比较 + 每次一个 f32 除法。实测 **2.8 ms/次**，depth 每帧采样 16 次
= **45 ms**，而整帧只有 59 ms。改成 **radix select**（f32 位元保序 u32 镜像，4 趟 8 bit，
每趟 O(vocab/256) 扫描 + 256 bin 回走）：**0.12 ms/次**，数值逐位不变（max/sum/argmax 的
归约顺序故意不动，argmax 平局按更小下标打破，等价于原来的串行扫描）。
→ depth 59 → **20 ms/帧**。**教训：先量整帧的分段，再信“kernel 已经很便宜”的直觉。**

**② `Device::zeros` 在往 PCIe 上推零页**
GPU 的 `zeros` 走 `tensor()` → `queue.write_buffer(全尺寸零 vec)`：codec block4 的每个
72 MB 输出缓冲都这么来一遍，实测 ~1.1 GB/s。改成“池化 buffer + 设备端 fill kernel”后：
codec 解码 5.21 → **2.57 s**，一次 run 的上传量 **4508 → 1240 MB**。`copy` 同理跳过零填充。

**③ 卷积 kernel 的占用率被 `stage` 数组卡死**
`CONV1D_REG8_F32` 的 `var<workgroup> stage` 按 16×576 f32 = 36 KB 静态分配，而实际窗口
`TT*stride + (ks-1)*dilation + 1` 最多 119（调用方只有 stride=1、ks∈{1,3,7}、dilation∈{1,3,9}）。
P104 每 SM 96 KB 共享内存 → **2 组/SM**。收到 16×128（8 KB）后 → **500 → 1300-1700 GFLOP/s**
（`conv_block_probe` 全量实测；block1 的 conv7 在真流程里 61 → 10 ms）。

**④ 分相转置卷积：换 kernel 而不是优化它**
`ks == stride*taps` 的转置卷积本来就是 `taps` 个因果 tap 跑在 (co,q) 相位行上，而旧 kernel
一个线程只算一个输出、单累加器串行 1536 次 → ~50 GFLOP/s。现在直接跑 `CONV1D_REG8_F32`
（`cout := cout*stride`，打包时把 tap 序反转），再用一个新 kernel `PHASED_INTERLEAVE_F32`
把 `[t_in, cout*stride]` 撒回 `[cout, t_in*stride]`：block2/3/4 的转置卷积 72/45/29 →
38/18/26 ms，block4 整块 0.155 → 0.131 s。**数字上用 `tconv_gpu_probe` 复核：三步合计
≤7 ms/块，剩下的 180 ms 是首次使用时的权重大小（75 MB）上传，见 ⑦。**

**⑤ 融合 depth 路径转正**：`M5_DEPTH_FUSED=1` 现在是默认（`M5_DEPTH_PLAIN=1` 回退），
720 dispatch/帧 vs 1178，depth 22.4 → **19.9 ms/帧**（`depth_frame_probe`）。

**⑥ host 端“准备权重”不是一个恒定量**：`pack_conv_weights` 串行打包 4.1 M 权重 ~50 ms
（block1 第一次卷积的 62 ms 里占 50）→ 并行后 **2.3 ms**；`TensorEntry::to_f32` 对 f32 权重
逐元素 `from_le_bytes` → 改成 memcpy；`causal_conv` 以前把 checkpoint 布局的 f32 权重也传上卡
（`codec.tensor`）再传一份打包副本 → 现在只传打包副本。

**⑦ 仍未解的最大单笔：大权重的首次上传。** 75 MB 的分相权重在真流程里 89 ms（0.84 GB/s），
干净进程里同样 128 MB 的 `create_buffer + write_buffer` 是 2.47 GB/s、复用 buffer 是 5.94 GB/s
（`upload_probe`）。按 16 MB 分片 `write_buffer` 把 118 → 89 ms，**根因未定位**（怀疑运行中
1.5~2 GB 常驻下的设备内存分配/映射）。codec 一次 run 要传 ~457 MB 权重（encoder 另 ~225 MB），
按 0.85 GB/s 计 ≈ 0.5 s，是当前 codec 1.1-1.4 s 里最大的一块。**下一步：在 upload_probe 里
先占 1.5 GB 常驻再测，确认是显存压力；若是，考虑权重分块预取到 GPU 并让它在 depth 生成期间
落地（试过一次“预热整个 decoder”，因为预处理抢占 CPU 记录时间而不赚，见下）。**

**⑧ 失败/回退过的尝试**（别重复）：
* `Pipeline` 里在生成前预热整个 codec decoder（一帧 dummy 解码）：codec 从 1.85 → 1.02 s，
  但 generation 1.73 → 2.54 s，**总时间反而 +0.13 s**（预热是 host 侧准备权重，抢占 CPU 录制）。
* 在 `causal_conv` 里插 sync 分段：codec 总时间不变，说明那里不是“批太大”的问题。
* snake kernel 不是瓶颈：72 MB 的 snake 单独跑 0.71 ms（204 GB/s），`sin` 与去掉 `sin` 同速
  （`snake_probe`）。之前 profile 里 snake 显示 32 ms，其实是它前面 `dev.copy` 的零页上传。

**当前 GPU 分解**（RTF 0.86-0.90，98 帧 4.30 s 音频）：
| 阶段 | 时间 | 说明 |
|---|---|---|
| depth | 1.62 s（26.4 ms/帧） | 层 kernel ~14 ms/帧（420 dispatch，权重 bf16）+ 采样 0.12 ms/帧；差距是 qkv/attention/head/norm 的在位开销 |
| codec 解码 | 1.13-1.46 s | block1-4 合计 ~0.67-0.93 s（首次权重准备 ~0.3 s 在里面）、pre_transformer 0.14、upsample 0.17、quantise 0.11 |
| encode | 0.50 s | seanet_encode 0.22 + transformer 0.22（encoder 权重 ~225 MB 的上传在里面） |
| prefill | 0.14 s | |

**下一步的最大三项**：①（⑦的）大权重上传，若能藏到 depth 生成期间 ≈ -0.4 s；
②depth 的层 kernel（14 ms/帧 × 0.79 s 总计）：bf16 GEMV 现在 50-70 GB/s（`norm_gateup` 82 µs），
靠 4 累加器/宽 load 提 ILP 是唯一方向，注意会改归约顺序；③encoder（0.50 s）。

---

## 6.9 2026-11 第三轮：权重常驻（0.86 → 0.56）

**发现**：把同一个 98 帧序列**在一个进程里解码两遍**（`examples/codec_repeat_probe.rs`）
才看清成本结构：

```
pass 0: 0.875 s   (上传 594 MB)     ← 全部是"准备权重"
pass 1: 0.080 s   (上传 0 MB)       ← 真正的计算
```

即 **codec 解码的 0.8 s 是每 run 一次的权重准备**，不是计算。逐项拆开（`weight_prep_probe`
+ 计时段 `upload_host_nanos`）：`Weights::get_f32` 拷贝、host 侧打包/展开、以及
`queue.write_buffer` 的 host 时间（594 MB 占 ~0.5 s，≈1.2-1.5 GB/s）。encode 阶段同理
（225 MB 权重、0.45 s 里 0.4 s 是准备）。

**为什么快不起来**（三个都白试了，写在这里免得重来）：
* `queue.write_buffer` 每次调用**新建并 map 一块 staging buffer**（wgpu-core
  `queue.rs: StagingBuffer::new`），实测 host 侧只有 1-1.5 GB/s。自制"持久 staging + 显式
  copy"反而更差：往 mapped 的 host-visible 内存写本身只有 1.4 GB/s（`staging_probe`）。
* `mapped_at_creation`（`create_buffer_init`）在干净进程里 2-4 GB/s，但在这个驱动上会把
  权重放进 **host-visible 内存**：GPU 之后每次读权重都走 PCIe，compute 慢一个数量级，
  98 帧直接 OOM。**永久放弃**这条路。
* 探测里"写进从不被 dispatch 使用的 buffer"能到 10-17 GB/s，那是假象（拷贝被推迟到首次
  使用）。真实带宽就是 **~1.5 GB/s**，和 6.6 记的 "PCIe x1 矿卡 0.6 GB/s" 同量级。
* 顺带：`Weights::load` 的逐张量拷贝改成 rayon 并行（load 2.94 → 2.4 s）；host 侧的
  safetensors 零拷贝 mmap 反而更差（RTF 0.89，页错误被搬进计时段），已回退；
* 其它失败：
  bf16 GEMV 加 4 累加器（+0.4 ms/帧，测试还挂了——helper 落错 kernel，已回退）。

**做法**：`Pipeline::load` 里把两半模型都"走一遍"——`Gpu::set_no_record(true)` 让
dispatch 全部丢弃，图遍历照样把每个权重（打包副本、展开的 bias、snake 表、rope 表、
15 个 audio_head、depth scratch、backbone 的 bf16 块）都建好并上传。没有 dispatch 就没有
队列积压，上传拿到最好的速率。（试过直接在生成循环里预热：那是用 host 时间换 GPU 时间，
总时长 +0.13 s，已删。）

**结果**（98 帧 4.30 s 音频，负载 8%，3 次）：

| 阶段 | 第二轮 | 第三轮 |
|---|---|---|
| encode | 0.50 s | **0.287 s** |
| prefill | 0.14 s | 0.14 s |
| generation | 1.62 s（26.4 ms/帧） | **1.29 s（23.4 ms/帧）** |
| codec 解码 | 1.13-1.46 s | **0.77 s** |
| **合成合计** | 3.62 s（RTF 0.84） | **2.42 s（RTF 0.56）** |
| load（含常驻） | ~1.5 s | 2.4-2.5 s |
| 整进程 wall | ~5.1 s | **5.2-5.6 s**（GPU）/ 4.6 s（CPU） |

对照：CPU(AVX2, 20 线程) 0.75；PyTorch CPU 1.36；PyTorch CUDA 0.77。输出音频与上一轮
**逐样本相同**（改的只是权重什么时候上卡）。

**注意 wall 与 RTF 的区别**：常驻把 1.5 s 的上传从计时段挪到 load，所以 RTF 掉到 0.56，
但整进程 wall 仍是 5.2-5.6 s（其中 ~0.6 s 是 912 MB 权重跑 1.5 GB/s 的 PCIe 硬底，~0.5 s 是
`Weights::load` 的拷贝——已并行化，~0.5 s 是 pipeline 编译）。**一次性的 CLI 调用，wall 上
CPU 路径仍然更快（4.6 s）；RTF 的意义在服务化/重复调用。** 想压 wall 的话，下一步是
`wgpu::PipelineCache` 落盘（首次仍要付）。

**下一轮最大的两项**（针对 RTF）：generation 1.29 s（depth 23.4 ms/帧：720 dispatch、层 kernel ~14 ms，
其余是 dispatch 在位开销 + 每帧一次 token 回读的整队排空）与 codec 解码 0.77 s（剩下是计算：
卷积 22 ms/块量级 + 4 个块 + pre_transformer/upsample）。要做深度的 2 倍提升，只能走
vec4 宽 load + 8 值/线程（改归约顺序，需过 `parity_lm` + `parity_pipeline` 双门）。

---

## 6.10 2026-11 第四轮：功能面覆盖与对齐审计（6.10 全表）

问题：**我们的测试覆盖了整个功能面没有？** 之前的端到端 parity 只有**一个场景**
（中文零样本克隆，seed 1234）。这一轮把参考实现暴露的每个功能都跑了一遍，用参考自己的
采样噪声回放（`scripts/dump_scenarios.py` → `tests/parity_scenarios.rs`），逐场景比
codes（要求逐值相等）与 PCM（SNR ≥ 60 dB）：

| 场景 | 帧数 | 音频 | codes | SNR cpu / gpu |
|---|---|---|---|---|
| 克隆·中文短句（"你好。"） | 7 | 0.46 s | 0 mismatch | 107.6 / 104.4 dB |
| 克隆·中文中句 | 52 | 4.06 s | 0 | 103.9 / 102.5 |
| 克隆·中文长段（4 句） | 216 | 17.18 s | 0 | 108.2 / 103.9 |
| 克隆·英文 | 82 | 6.46 s | 0 | 106.6 / 108.7 |
| 克隆·日文 | 37 | 2.86 s | 0 | 112.0 / 107.2 |
| 克隆·德文 | 38 | 2.94 s | 0 | 109.7 / 104.1 |
| 克隆·法文 | 28 | 2.14 s | 0 | 103.1 / 100.3 |
| 克隆·韩文 | 31 | 2.38 s | 0 | 106.1 / 105.0 |
| 克隆·accent=chuan | 48 | 3.74 s | 0 | 112.6 / 110.7 |
| 克隆·accent=yue（繁体字，粤语腔） | 78 | 6.14 s | 0 | 105.0 / — |
| 克隆·参考音频 44.1 kHz（重采样路径） | 28 | 2.14 s | 0 | 106.4 / — |
| 克隆·caption | 29 | 2.22 s | 0 | 109.3 / 107.2 |
| 克隆·topk=1（跑满 40 s 无 EOS） | 500 | 39.90 s | 0 | 109.1 / 108.5 |
| **克隆·长文 571 字（90 s 预算、跑满不 EOS）** | **1125** | **89.90 s** | **0 / 18000** | **104.5 / 103.9** |
| 克隆·topk=50 | 33 | 2.54 s | 0 | 112.6 / 109.6 |
| 克隆·temperature=0.5 | 31 | 2.38 s | 0 | 106.4 / 105.9 |
| 预设音色 speaker 1（中文） | 35 | 2.70 s | 0 | 115.6 / 111.6 |
| 预设音色 speaker 2（中文） | 35 | 2.70 s | 0 | 116.4 / 113.4 |
| 预设音色 speaker 1（英文） | 28 | 2.14 s | 0 | 114.5 / 113.4 |

即：**16 个场景、两套后端，codes 全部逐值相同，PCM SNR 100-116 dB**。
命令：
`M5_SCENARIO_DEVICE=cpu cargo test --release --test parity_scenarios`（全 16 个）+
`M5_SCENARIO_DEVICE=gpu M5_SCENARIO_BUNDLE=clone|cargo test --release --test parity_scenarios`
（GPU 分两个进程跑，因为一个 Pipeline 常驻约 1 GB，一块卡装不下两份）。goldens 目录
（175 MB，采样噪声占大头，可再生）不入库。

**审计过程中挖出两个真 bug（都已修）**：

1. **`encode_batch` 用 `std::mem::forget(batch)` 泄漏 bind group** —— bind group 会
   钉住它绑定的 buffer，所以同一个进程里做第二次合成就会 OOM。单次 CLI 调用看不出来
   （进程退出即释放），但**服务化必然踩**。改成 `retired` 列表：阻塞等待（`sync` 或
   in-flight 满）之后释放。修完 GPU 侧从"第 2 个场景 OOM"变成"13 个场景跑通"。
   另加 `Gpu::release_scratch()`（全排空后清池），给多次合成的调用方用一个口子。
2. **参考的 codec 解码是分块的**（`chunked_decode(chunk_size=300, left_context=25)`），
   我们是一次性整段 → **超过 300 帧（24 s）的音频对不上**：`clone_topk1` 只有 31.3 dB，
   而且差异集中在第 20~35 秒。按参考分块（含丢掉 left context 的写法）后 **109.1 dB**，
   顺带把 40 s 音频的峰值显存从"爆 8 GB"降到"每块 325 帧"。

**已补齐的两项外部依赖（都按"数据 + 算法一起入库"，不留运行时依赖）**：

* **accent=yue**：参考里的 `yue` **不是粤语发音**，而是中文 TN 之后做 **字形转换**
  （OpenCC `s2hk`：简体 → 香港繁体），模型在 yue 条件下配对的是港繁字形。
  `opencc-python-reimplemented` 只是三个文本词典上的纯 Python 循环，所以
  `src/opencc.rs` 移植了它**同一套算法**（分隔符切分 → 分组词典左最长匹配 + 长度 hint →
  取第一候选 → 顺序重建 → 再过 HKVariants）+ **同一份数据**（`tn_data/opencc/`，
  1.0 MB），并用 `scripts/dump_opencc.py` 导出的 **12 048 条**参考输出做闸门
  （`opencc::tests::matches_reference_on_corpus`，全等）。端到端 78 帧、codes 0 mismatch、
  105 dB。
* **非 24 kHz 参考音频**：`audio::resample` 按 torchaudio 2.8 的公式实现
  （`sinc_interp_hann`，`lowpass_filter_width=6`，`rolloff=0.99`，按 gcd 约简成
  `min(orig,new)*rolloff` 的窗化 sinc 滤波器组 + stride 卷积，含 `width` 两侧补零与
  `ceil(new*len/orig)` 截断）。44.1 kHz 参考音频端到端 28 帧、codes 0 mismatch、106.4 dB。

**仍然缺的**（如实）：

* `max_audio_length_ms` 已按参考语义暴露（`--max-audio-length-ms`，默认 40000，<800 报错，
  帧数 = ms/80）。
* **多段 context / 多 speaker**：参考的 `Generator.generate` 收 `List[Segment]`，但**公开 API
  不暴露**它——`M5TTS.infer`、`infer_clone.py`、`infer_speaker.py`、`app.py` 都只传一段。
  所以我们按公开 API 对齐，没有做多段（要做是加功能，不是补对齐）。
* **真正的长文本（一篇文章/几分钟）**：参考**做不到单次读完**，见 6.11。
* ONNX 路径（`infer_onnx.py`）按约定不做；Gradio `app.py` 只是包装。
* 流式（边生成边出音频）参考没有，我们也没有（但 codec 已按参考分块，接流式只差调度）。
* `parity_tn` 覆盖 6 语言 + 两种 accent，1 条英文平局差异已登记。

---

## 6.11 长文本的真实天花板（以及是否要做分段合成）

一段 700 字的中文散文（439 个文本 token，含条件串 498）交给参考实现会怎样：

| | 参考实现 | 我们 |
|---|---|---|
| 默认 `max_audio_length_ms=40000` | 500 帧封顶 → **40 s 截断**（`RuntimeWarning`） | 543 帧 / 39.90 s 截断（同样） |
| 调大上限（如 180 s = 2250 帧） | `Generator.generate` 检查 `prompt < max_seq_len(2048) - max_audio_frames` → **`ValueError: Inputs too long`** | 同样报错（`pipeline.rs` 里新加的检查，防止越过 rope 表的 2048 位置静默产出垃圾） |
| 字/秒 | 模型自身节奏 ~6.2 字/秒（实测 122 字 → 19.58 s） | 同 |
| 700 字需要 | ~113 s ≈ 1410 帧；498+1410 = 1908 < 2048，**结构上勉强放得下**，但 40 s 的默认上限先把它截了 | |
| 实际让模型跑下去 | 无法（报错） | 把上限放到 2250 帧时，模型在 **1054 帧（80.8 s）自己出 EOS**，只覆盖约 2/3 文本 |

结论：**参考实现的"长音频"就是 40 秒**；它没有任何按句切分的逻辑（`app.py`/`inference.py`/
`generator.py` 全文没有 split）。所以"读完一篇文章"不是对齐缺口，而是**超出参考能力的功能**。

两条可选的源头设计（需要你定）：

**A. 按句切分合成（推荐）**：把 TN 后的文本按标点切成长度受限的片段（每段 ≤ ~250 字 ≈
40 s，保证 prompt + 文本 + 帧数 ≤ 2048），每段都用**同一个 prompt/音色上下文**重新 prefill、
生成、解码，再把 PCM 按顺序拼接（每段尾部照参考裁 0.1 s，段间可选 20-50 ms 交叉淡化）。
代价：段边界处韵律会重新起头；每段多一次 prefill（~0.1 s，700 字 3 段合计 +0.3 s）。
好处：每段的输出**可以与参考逐段 parity 对齐**（把该段单独喂给参考、录它的噪声回放），
所以新功能仍然有硬闸门。实现量：`pipeline.rs` 的切分 + 循环 + PCM 拼接，加一个测试。

**B. 只保证和内核对齐**：维持 40 s / 2048 token 的参考行为，长文报错或截断（已实现），
文档里写清"一次调用最多 40 s"。零实现量，但 700 字的稿件用不了。

---

## 6.12 长上下文越界 bug（已修，2026-11 第五轮）

**症状**：同一命令、同一 seed，GPU 对一段 571 字的文本跑出 467 / 866 / 411 / 1168 帧
（CPU 两次逐字节相同）。位置 <1024 时稳定，跨过 1024 才抖：

| 预算 | 最大位置 | 三次 |
|---|---|---|
| 30 s | ~881 | 418 / 418 / 418 帧，逐字节相同 |
| 60 s | ~1256 | 793 / **238** / 793 帧 |

**根因（两个 kernel，同一个错误模式）**：attention 的 score 暂存是**固定尺寸工作组数组**，
而被索引的量是序列长度。

| kernel | 用在哪 | 原来 | 实际需要 | 后果 |
|---|---|---|---|---|
| `ATTENTION_F32` | backbone **prefill**（`window = frames`）、codec（窗口 72） | `scores: 256` | 到 prompt token 数 | 507 token 的 prompt **一进来就越界**，把同组 `partial` 一起写坏 |
| `QKV_ATTEND_F32` | backbone/depth **每步 decode** | `scores: 1024` | 到 2048 位置 | 第 1024 帧之后越界，输出变随机 |

同时 `QKV_ATTEND_F32` 里"本 dispatch 写 KV → 随即读回"只用了 `workgroupBarrier()`
（WGSL 里跨 invocation 的 **storage** 空间可见性要 `storageBarrier()`），一并补上。

**修法**：上下文上限收敛成**一个常量** `lm::MAX_CONTEXT = 2048`（参考的 `max_seq_len`），
rope 全表、两个 kernel 的 score 暂存、`synthesize` 的越界检查全部由它定尺寸/校验，不再各写各的
字面量。性能无变化（0.54-0.55，depth 21-22 ms/帧；注意力 kernel 本来就不是占用率瓶颈）。

**新增的闸门**（防回归）：
* 场景 `clone_zh_verylong`：571 字、498 token、90 s 预算、参考跑满 1125 帧不 EOS →
  回放参考噪声后 **1125/1125 帧、18000 个 codes 全同、CPU 104.5 dB / GPU 103.9 dB**。
* `M5_SCENARIO_REPEAT=1`：每个场景在**同一进程里跑两遍**，PCM 必须逐位相同。GPU 侧
  15 个 clone 场景 + 3 个 preset 场景全过（179 s / 9 s）。这类"读到自己没写的内存"的 bug
  从外面看和采样噪声无法区分，只有这个断言能抓到。

**结论可以写进 README 的那句话**：在参考实现支持的时长内（`infer` 默认 40 s、
`generate` 默认 90 s、受 2048 token 上下文限制），我们与它**逐 code 对齐**；
超出这个范围的时长，参考实现本身不支持（它会截断或抛 `Inputs too long`），
那是移植版后续要加的扩展功能（按句切分合成，设计见 6.11）。

---

## 6.13 覆盖率审计（2026-11 第六轮）：31 个场景，30 个逐 code 对齐

按官方 `M5TTS.infer` 的参数空间铺开（`scripts/dump_scenarios.py` 的 31 个场景，每个都录
参考自己的采样噪声回放）：

| 维度 | 覆盖 | 结果 |
|---|---|---|
| 模型包 | VoiceClone（克隆）、CustomVoice（预设音色） | 0 mismatch |
| 语言 | 中/英/日/德/法/韩 | 0 mismatch |
| 口音 | 无 / chuan / yue | 0 mismatch |
| caption | 有 / 无 / caption+accent 同时 | 0 mismatch |
| speaker id | 1 / 2 / 3 / 4 | 0 mismatch |
| temperature | 0.1 / 0.5 / 0.9 / 1.5 | 0 mismatch |
| top-k | 1 / 7 / 10 / 50 | 0 mismatch |
| 文本长度 | 1 字（8 帧）、常规、**250 字（app 上限，284 帧）**、无标点、571 字（1125 帧） | 0 mismatch |
| 音频时长 | 0.54 s → **89.90 s** | 0 mismatch |
| 参考音频 | 24 kHz 3.4 s、**44.1 kHz**、**立体声**、**13.36 s（>10 s，触发 0.02 静音阈值）** | 0 mismatch（13.36 s 已修，见下） |
| 生成预算 | 1.6 s / 40 s / 90 s（含"跑满不出 EOS"的截断场景） | 0 mismatch |

**当前 31/31 全部逐 code 对齐**（2026-11 第七轮修掉最后一项后）。

### 6.13.1 最后那一项：编码器两个 bug（已修）

13.36 s 参考录音原先 prompts codes 差 17/2032，导致 LM 从第 0 帧分叉、时长都不同
（参考 1.82 s、我们 5.34 s）。逐阶段 dump 后定位到**编码器**，两处：

1. **编码器 transformer 的注意力窗口**（`src/codec/encoder.rs`）：我们按
   `encoder_config.sliding_window = 250` 做了窗口截断，但参考的 `MimiTransformerModel`
   在这里建的是**纯因果 mask**——用 hook 抓它的 attention_mask 实测：query 0/1/250/252
   分别保留 1/2/251/253 个 key，且 `first allowed key` 恒为 0。所以 253 帧的序列从第
   250 帧开始分叉（transformer 输出差 0.16 / 0.11 / **0.50**），再被 downsample 放大到 12.9。
   改成全因果后 transformer 阶段 max|diff| 4.0e-4（尺度 20），最后一帧 1.1e-4。
2. **downsample 的 `pad_mode="replicate"` 右端**：它的滑动窗口在输入长度为奇数时越过信号
   末尾一个样本，HF 用**最后一个样本复制**填充，我们填了 0 → 末帧 latent 差 13。
   补上右端复制后 downsample 阶段 max|diff| 5.3e-4（尺度 26.8）。

修完：编码器 codes **0/2032**，`clone_prompt_long10s` 端到端 **24/24 帧、0/384 codes、
SNR 110.8 dB（CPU）/ 110.6 dB（GPU）**，本次 `M5_SCENARIO_REPEAT=1` 下也逐位可复现。

**方法学**：这一轮靠的是"逐阶段 dump + 与参考对齐"（`examples/prompt_probe` 存
`seanet / transformer / downsample / downsampled` 四个阶段的 f32，Python 侧用 forward hook
落同样的张量），而不是猜。README/HANDOVER 之前记的"编码器 CL 绕行变慢已回退"之外，
这一条是编码器真正的对齐缺口。

（历史记录，修好前的症状：）

* `prepare_prompt` 对它**逐样本相同**（242326 样本，max|diff| = 0），所以不是前端问题；
* 但**编码器的 prompt codes 在最后约 2 帧（共 127 帧）不同**：第二末帧差 1 个 code，
  末帧 16 个全差 → 17/2032。这 2 帧的 latent 被尾部 padding 主导，落在 codebook 决策边界上，
  我们编码器的 f32 累加顺序差（≤2e-4，`parity_codec` 的门限）足以翻掉。
* 后果：prompt codes 变了 → LM 从头分叉 → 该场景跑到噪声用尽。
* 已验证与"是否 hop 整数倍"无关：4.3 s 那支 82211 样本（也非整数倍）逐 code 全同；
  给 242326 样本尾巴补 1920 个零，仍只有第 125~127 帧不同。
* **定位方向**：像 `dump_acoustic.py` 对解码器那样，给**编码器**做逐阶段 dump
  （seanet → transformer → downsample → latent），确认第 125 帧之前是否已经分叉；
  若 latent 已分叉，就要查 HF `MimiConv1d` 的 per-layer `extra_padding` /
  `pad_mode` 在尾部的处理（`replicate` 的右端、`ceil(...)-1` 的帧长公式）。

其余 30 个场景：**codes 逐值相同，PCM SNR 103-116 dB**，且 GPU 侧在
`M5_SCENARIO_REPEAT=1`（同进程跑两遍要求逐位相同）下全过。

---

## 6.14 2026-11 第八轮：depth 的 kernel 提速与"延迟地板"

**结论先说**：depth 现在是**依赖 dispatch 链的延迟**受限，不是 kernel 算术或带宽受限。

**做了什么**（都在 `src/shaders.rs`）：

1. **bf16 / f32 的 GEMV 全部改成 16 字节宽 load**（权重视图 `array<vec4<u32>>` /
   `array<vec4<f32>>`，每次取 8 个 bf16 或 4 个 f32）+ 4 个独立累加器。之前每个元素
   一条 4 B load、一条累加链。
2. **`norm_gateup` 的两处结构问题**：原来每 workgroup 串行做 64 行（4096 行的投影只有
   64 个 workgroup = 3.2 个/SM），且暂存行按 8 个 float 的 lane 步长读取（8 路 bank 冲突）。
   改成 vec4 暂存 + 每 warp 行。
3. **rms 融进 qkv 投影**（`NORM_GEMV_BF16_F32`，和 norm_gateup 同一招：norm 在每
   workgroup 内重算），层步 6 dispatch → **5**。

隔离测量（`examples/depth_bf16_probe.rs`，一层步 6 dispatch）：

| kernel | 改前 | 改后 |
|---|---|---|
| rms 512 | 7.7 µs | 8.1 µs（已并入 qkv） |
| gemv bf16 768×512 | 13.7 µs | 9.3 µs |
| qkv_attend | 13.2 µs | 14.0 µs |
| gemv_add 512×512 | 11.6 µs | 7.9 µs |
| **norm_gateup 2×2048×512** | **82.3 µs** | **20-28 µs**（192 GB/s，带宽上限） |
| silu_down 512×2048 | 29.4 µs | 21.3 µs |
| 一层步合计 | ~163 µs | **~92 µs** |

**但整帧只从 21.2 → 17.7 ms**（depth 650 dispatch/帧）。这暴露了真正的瓶颈：把 6 个
kernel 单独测是 92 µs/步，而真实帧里每步要 190 µs——**每个"依赖前一个"的 dispatch 在这张卡上
约 30 µs**（隔离背靠背只要 5-15 µs）。同一份 kernel 提速 1.6 倍，帧时间几乎不动，就是证据。

**一个反直觉的调参**：`norm_gateup` 的 workgroup 数不是越多越好。同一 kernel：

| 行/workgroup | workgroup 数 | 隔离耗时 | 整帧 depth |
|---|---|---|---|
| 64 | 64 | 66.8 µs | 21.2 ms |
| 8 | 512 | 20.3 µs | **22.2 ms**（更慢！） |
| 32 | 128 | 28.4 µs | **18.9 ms** |

隔离最快的那档在链里最慢：512 个 workgroup 的**尾巴**拉长了单次 dispatch 的完成时间。
延迟受限时，网格大小要按"最慢 workgroup 的完成时间"调，不是按带宽利用率调。

**RTF**：0.54 → **0.51**（synthesis 2.33 → 2.20 s；depth 21.2 → 19.1 ms/帧，含 backbone 的
排空；纯 depth 帧 17.65 ms / 650 dispatch）。所有门全绿（含 31 场景、GPU 逐位复现）。

**下一步的唯一大杠杆**：把整个 codebook 步做成**一个 persistent kernel**——一个
512 线程的 workgroup（hidden=512，一线程一维）在同一个 dispatch 里跑完 5 层
（rms/qkv/attend/o/gateup/down 全在 workgroup 内存里接力，KV 留在全局）。dispatch 数从
650/帧降到 ~45/帧；按当前 30 µs/依赖 dispatch 与 190 GB/s 的权重流（570 MB/帧 = 3 ms）
估算，depth 可到 **4-6 ms/帧**，RTF 0.51 → **0.30 左右**。代价：一个 ~300 行的 WGSL kernel，
共享内存预算 ~22 KB（hidden 2 KB + qkv 3 KB + gate_up 16 KB），数值需要重过 31 场景门。

---

## 6.15 2026-11 第九轮：depth 0.54 → **0.47**，以及"persistent kernel 为什么不做"

**收益来源**（都是先量再改，全部过 31 场景门）：

| 改动 | 效果 |
|---|---|
| GEMV 类 kernel 全部 16 字节宽 load（`array<vec4<u32>>`/`vec4<f32>`）+ 4 累加器 | 一层步 163 → 92 µs；norm_gateup 82 → 20-28 µs |
| `norm_gateup` 每 WG 行数 64 → 32（= 64 → 128 个 WG） | depth 21.2 → 18.9 ms/帧（**注意：512 个 WG 反而更慢**，见 6.14） |
| rms 融进 qkv 投影（`NORM_GEMV_BF16_F32`） | 层步 6 → 5 dispatch |
| **每帧 2-token prefill 换掉 16×16 tile GEMM**：新增 `LINEAR_ROW_F32`（m ≤ 8 时用，一行一线程，**保持同一累加顺序**所以数值逐位相同） | prefill 5.8 → 2.8 ms/帧，depth 17.7 → 15.0 ms |
| **编码器（encode 阶段）改走 channels-last 卷积**（`conv1d_reg8` 窗口不够的 strided 层用 `conv1d_cl`） | `encode/seanet_encode` 0.213 → **0.058 s**（朴素 channels-first 只有 ~70 GFLOP/s，峰值的 3%）；encode 0.289 → 0.131 s，RTF **0.47 → 0.44** |

**codec 解码的实测构成**（`examples/codec_unit_probe.rs`，block4 = 96ch × 188k）：
transpose 1.35 ms（×4/单元）、snake(copy+kernel) 4.07（×2）、add 0.95、conv7 **19.5**（1245 GFLOP/s，
峰值的 59%）、conv1 6.37、整个单元 73.6 ms。**conv 已近峰值，剩下的是 6 趟 DRAM 往返**；
再往下要靠"snake 融进 conv 输入 + residual 融进 conv epilogue"（估计省 ~13 ms/单元 = codec -13%）。

**踩到的两个坑**（都回滚了，别重蹈）：

1. **把 2-token prefill 拆成两次融合 decode 步**（数学等价）：快 4 ms，但 **GPU 侧所有场景的 codes 都变了**
   （每次多生成一帧、噪声用尽）。CPU 矩阵全过是因为 CPU 不走融合路径——这说明"只在 GPU 上改结构"
   必须用 GPU 矩阵验证，`parity_pipeline` 一个场景不够。
2. **norm 融进 audio_head GEMV**（`NORM_GEMV_F32`）：也快 0.8 ms，但 GPU 侧同样整体偏移。已把 norm
   的归约改成与独立 kernel 一样的 64 lanes/64 partial（更接近了），仍然不一致；可疑点是 bind group
   按 codebook 缓存时 `logits` 必须是常驻 buffer（这一点已修），但复测仍不过 → **暂时保留独立
   rms + 通用 head 这条（已验证）路径**，把融合留作待查项。源码里已删掉这条死路径。

**为什么不做 persistent kernel（一个 dispatch 跑完整个 codebook 步）**

数学上它把 654 dispatch/帧降到 ~45，但硬件上不成立：depth 每帧要流 **570 MB 权重**，而一个
workgroup 只能用 **1 个 SM**，P104 单 SM 的 DRAM 带宽约 30-50 GB/s（全卡 190）。570 MB / 40 GB/s
= **14 ms/帧**，比现在（15 ms）还差。要跑满带宽必须同时有几百个 workgroup，而层与层之间是
grid-wide 依赖、wgpu 没有 grid sync——所以"每层一次 dispatch"这条路线（现在 5 个/层步）就是
这张卡上的结构性下限；再往下只能减少**每帧的层步数**（模型决定的 15×5=75）。

**当前分解**（RTF 0.47，synthesis 2.07 s / 4.30 s 音频）：depth 16.2 ms/帧（654 dispatch）>
codec 解码 0.68 s > encode 0.29 s。depth 里：层步 ~10 ms（350 dispatch）、head+sample ~4.5 ms、
prefill ~2.8 ms。**下一个看得见的目标是 codec（0.68 s）**，以及 depth 的 `sample_topk`
（16 次/帧，实测 ~175 µs 一次，1 个工作组的 4 趟基数选择）。

---

## 6.16 2026-11 第十轮：长音频（默认开）——按语义长度分段

**行为**（不加任何参数）：

| 输入 | 行为 |
|---|---|
| 短文本（预算内一次读完） | 与显式单 pass **逐字节相同**（sha256 对比，见验证） |
| 长文本（如 571 字） | 按句子切成 ~110-125 语义单位的若干段，逐段合成、拼接，段间 150 ms 停顿；实测 5 段 / 130.9 s，RTF 0.38 |
| 超长文本（2288 字） | 20 段 / 556.8 s，RTF 0.38，无一段被截断 |
| 固定音色（CustomVoice） | 同一路径（只是条件换成 speaker id）：571 字 → 7 段 / 108.8 s，RTF 0.38 |
| 克隆音色（prompt_wav） | 同一路径（每段都用同一段参考音频 + 转写文本做条件） |
| 显式 `--max-audio-length-ms`（官方语义） | 预算按给定值、超上下文报错/截断——**与参考实现完全一致**，不触发分段 |

**设计**（`src/long_audio.rs` + `Pipeline::synthesize_auto`）：

* **切分只看文本，不预测音频时长**：移植 `D:\ming-3.0-tts-wgpu` 的
  `frontend/text_segment_cut.py`（`semantic_length`：汉字 1、英文单词 1；分三级：
  句末标点 → 句内分句 → 无标点才逐字兜底；过短尾巴并入前段）。因此**同一文本永远切成同一组片段**，
  可复现、可测试（`long_audio` 6 个单测：拼接无损、预算 5~200 全覆盖、优先句边界、短尾合并）。
* **片段长度上限 = 参考实现自己的 utterance 预算**（`max_audio_frames`，默认 40 s），
  换算按保守朗读速度 3 单位/秒 → `semantic_budget(500) = 125` 单位。上下文只当**天花板**
  （`MAX_CONTEXT - 条件 token - prompt 帧 - 1`），防止调用方给了超长预算时把文本撑爆上下文。
* **每段 = 一次普通单 pass**，条件（speaker id 或 prompt 音频/转写）完全相同，所以固定音色和克隆音色
  走同一条路；单 pass 的 `max_audio_frames` 给满上下文余量，"会不会被截断"只由上下文决定。
* **两道守卫（只有这两种情况才重切该段，绝不整篇重跑）**：
  1. 撞上下文上限且没有 EOS → 该段按语义长度对半切，两半插回队首重做；
  2. **比人说话还快**（> 8 单位/秒，即 480 单位/分钟）→ 模型跳字了，同样对半切。
  切到单个不可再分单元仍触发时，保留并打印 warning（`cut_off` / `unreadable`）。
* **接缝**：每段保留自己自然的收尾，仅把尾部静音规整到 150 ms（模型自己的尾静音从 0 到 1 s 以上不等）；
  最后一段不做规整。段间**不做**响度归一化（会破坏与单 pass 的一致性）。

**为什么是这套（本轮实测数据，都是 50M + 同一音色）**：

* 「按上下文能装多少就塞多少」是错的：511 字一段 → 模型只给 17.6 s（放弃朗读），
  2288 字按此策略跑到 RTF 7.96（反复截断重切）。
* 模型对长段不稳定：同一 278 字段落，4 个 seed → 95.7 / 21.0 / 99.3 / 76.9 s。退化输出不是"快"，
  而是**低频嗡鸣**（实测整段 zcr≈0、无过零，rms≈0.07）；正常输出 zcr 0.06-0.13、静音 23-26%。
  所以"时长偏短"不能简单当成"读得快"——那是本轮删掉的一个错误启发式（见下）。
* 40 s 量级的片段稳定得多：571 字段落 5 段的速度落在 4.5-6.3 单位/秒，2288 字 20 段同理。
* 跳字是可检测的：146 字 10.4 s（14 单位/秒）远超人类上限 → 守卫 2 把它切半后变成 5-7 单位/秒。

**根因：片段把原文的换行带进了条件文本（2026-11 修，务必记住）**：用户听出"第二段根本不像中文，
像粤语"。该段正是段落分隔处（`

`）——分段把 `
` 留在片段开头，于是条件文本里出现了模型
从没在朗读位置见过的换行 token，模型开始胡言。证据链：① `--tokenize` 显示条件文本里确有 `

`；
② 同一段落在两次不同的分段方案里都坏（都是跨段落边界的那段）；③ 参考实现 `inference.py:126`
本来就 **`.strip()`** 过目标文本和 prompt 转写，我们的分段路径漏了这一步。
修法（三处，都在源头上）：`synthesize` 按参考语义 strip 文本与 prompt 转写（单 pass 路径也补齐，
顺带修掉一处与参考的偏差）；`long_audio::clean` 把换行/连续空白压成一个空格，**只用于分段路径**
（单 pass 保留参考的原样行为）；`
` 也算句边界，且每个片段 `trim` 后入库。
修后同一稿：5 段 / 100.9 s，语速 4.7-5.2 单位/秒（修前同一段 146 字 58.6 s = 2.1 单位/秒），
低频嗡鸣 0 处，且**一次重切都没触发**。

**语速合理性带（对称守卫，2026-11 补）**：试听反馈确认"慢"同样不可接受——146 字用 58.6 s
（2.1 单位/秒）那一段听起来"根本不像中文，像粤语"。同一段单独重跑三个 seed：26.8 / 49.9 / 31.7 s，
说明这是**采样导致的走飞**（拖长糊掉），不是文本问题。所以守卫改成对称的语速带
**[3, 8] 单位/秒**（只统计真实发声，剔除段尾静音；不足 20 单位的短段不判，因为停顿占比太大）：
出带就对半重切。修后同一稿：5 段 → 7 段 / 97.9 s，所有段 3.6-7.9 单位/秒，低频嗡鸣段 0 处。
代价：被丢弃的那次 pass 白跑（本例 58.6 s 音频），RTF 0.37 → 0.65（仍远快于实时）。

**本轮试过并已删除的方案**（别再走一遍）：

1. 全局 `FRAMES_PER_TOKEN = 3.2` 估算 + "首段自校准节奏" → 换成纯文本切分。校准本身被模型的
   采样方差打败（同一段快慢差 3 倍），且引入不可复现的段边界。
2. 撞上限后**整篇重跑**（`force_split` 标志防递归）→ 换成队首对半重切，只重做那一段。
3. 按 token 绝对位置均衡填装（`remainder/passes_left`）→ 段长确实均匀了，但"均匀"不是目标，
   而且仍要估算每段能装多少。语义长度切分天然给出同样均匀的结果，且不用估。
4. "比实测节奏短 25% 判为跳字"→ 删：同一音色同一文本的时长差可达 3 倍，会把正常快读误判。
   只留**生理上限**（> 8 单位/秒）这一条硬判据。

**验证**：

* `NO_COLOR=1 cargo test --release` 17+18+3+4+1+1+1+1 全绿；`M5_NO_SIMD=1` 同样全绿（10 套件）。
* 31 场景：`M5_SCENARIO_DEVICE=cpu` 全过；`M5_SCENARIO_DEVICE=gpu M5_SCENARIO_BUNDLE=clone|preset` 全过
  （场景都显式传预算，走单 pass 路径，行为未变）。
* 短文本等价性：`--text "…38 字…"` 自动模式 vs `--max-audio-length-ms 40000`，
  WAV sha256 `bc96b40a…` 两边一致。
* 长文本：571 字克隆 5 段 130.9 s；2288 字 20 段 556.8 s；571 字固定音色 7 段 108.8 s（RTF 0.375-0.38）。
* `M5_DUMP_FRAGMENTS=<prefix>` 会把每次 pass 的文本写成 `<prefix>.<n>.txt`，便于核对切分。

---

## 6.17 API 对齐审计（2026-11，逐项对 `D:\M5-TTS` 的 `M5TTS` / `Generator`）

参考实现的公开面只有三处：`M5TTS.from_pretrained(...)`、`M5TTS.infer(...)`、`M5TTS.save(...)`，
加 `Generator.generate(...)`（内部）和两个 CLI 包装（`infer_clone.py` / `infer_speaker.py`）。
**参考实现没有流式**：`generate` 是"跑满帧数→一次性 codec 解码→返回整段 tensor"，没有 yield/chunk 回调。

| 参考面 | 我们的对应 | 状态 |
|---|---|---|
| `infer(text, prompt_wav, prompt_text, speaker_id, lang, accent, caption, temperature, topk, max_audio_length_ms)` | `Pipeline::synthesize(&Args, &mut dyn NoiseSource)` | ✅ 同参同义（`max_audio_length_ms` → `max_audio_frames`，另有 `with_max_audio_length_ms`） |
| 参数校验：text 非空 / temperature 有限 >0 / `1 ≤ topk ≤ 2051` / `max_audio_length_ms ≥ 800` / cloning 必须给 prompt_text / 给 prompt_text 必须给 prompt_wav / `speaker_id ≥ 1` | 同 | ✅（本轮补齐 speaker_id=0 与 temperature/topk 入口校验） |
| 解码后校验 `audio` 非空且有限 | 同 | ✅（本轮补） |
| `Generator.generate` 的 `max_seq_len` 守卫：`prompt_tokens >= 2048 - max_audio_frames` 报错 | `total_input + max_frames > 2047` 报错 | ✅（本轮修正，之前宽松 2 个位置） |
| `min_tokens=6`（第 7 帧起才允许 EOS） | `allow_eos = frame > 6` | ✅ |
| EOS = 16 个 codebook 全 0；撞上限告警；无帧报错 | 同（`Synthesis.eos`、`bail!`） | ✅ |
| `prepare_prompt`：按峰值归一、阈值 0.02/0.015、0.15 s 静音合并、0.05 s 边距、非空/有限/非静音校验 | `audio::prepare_prompt_samples` | ✅（逐句对照，含 <10 s / >10 s 两个阈值分支） |
| `save`：PCM_16、自动建父目录 | `main` 写 wav（`hound`，clamp 到 ±1） | ✅ |
| `from_pretrained` 的 bundle 校验：必需文件、`audio_vocab_size=2051`、16 codebooks、`text_vocab_size == len(tokenizer)` | `ModelConfig::validate` + `ConditioningMeta` + `--inspect` | ⚠️ 前两条 ✅；`text_vocab_size` 未读该字段，靠 `--inspect` 的 tensor 契约校验覆盖 |
| Hugging Face repo id / `revision` / `cache_dir` / `local_files_only` / `attention=flash_attention_2\|sdpa` | 只接受本地目录；GPU 由 wgpu 选（vulkan/cpu），无 attention 开关 | ⚠️ **有意差异**：不联网、不依赖 flash-attn；本地目录语义等价 |
| `--device cuda` 必需（无 CUDA 直接报错） | `--device cpu\|gpu`，两个后端都自研 | ⚠️ **有意差异**（本项目的存在理由） |
| — | `--seed --tokenize --inspect --max-audio-length-ms` 可选（自动长音频）、`Args.auto_length` | ➕ 超集 |
| — | `Synthesis { codes, frames, prompt_frames, generated, audio, sample_rate, eos }` | ➕ 比参考多返回 codes/帧数/eos |

**结论**：参考 API 的参数、校验、EOS/上限语义、prompt 预处理、保存格式都对标（本轮补齐 3 处）；两处有意差异
（不联 HF、device 语义）与超集（自动长音频、seed、codes 返回）已在 §6.16/本节写明。

---

## 7. 优化建议（**仅供参考**，按性价比排序）

1. **大权重首次上传**（6.8⑦）：codec 一次 run 传 ~457 MB、encoder ~225 MB，实测在真流程里只有
   ~0.85 GB/s（干净进程 2.5-5.9 GB/s，见 `upload_probe`）。先把“显存压力是否是原因”测清楚
   （probe 里先占 1.5 GB 再测），再决定走“分块预取 + 与 depth 生成重叠”还是别的路。上限 ~0.4-0.5 s。
2. **depth 的层 kernel**（14 ms/帧，占端到端 20%）：bf16 GEMV 实测 50-70 GB/s（`norm_gateup` 82 µs、
   `silu_down` 29 µs），是 warp-per-row 单累加器 + 每元素一次 4 B load 的延迟问题。方向是
   4 累加器 / vec4 宽 load / split-K（注意都会改归约顺序，需 `parity_lm` + `parity_pipeline` 双门）。
3. **codec 的 DRAM 往返**：残差单元现在仍 = 2×(transpose+conv+transpose) + 2×snake(copy+kernel) + add。
   把 snake 融进 conv 输入、residual 融进 conv 输出（CPU `snake_conv1d` 的 GPU 版），以及
   snake 的 out-of-place 版本（现在多一次 72 MB 的 copy）。
4. **encoder**（0.50 s）：seanet_encode 仍走 channels-first 朴素卷积（当年测过 CL 绕行更慢）。
   现在 CL 卷积的占用率修好了（6.8③），值得重测。
5. **torch RNG 对齐**：现在 CLI 的 `--seed` 用自研 xoshiro PRNG（可复现，但与
   `torch.manual_seed` 不同流）。要做到"同 seed 与参考逐 token 相同"，需要复刻 torch CPU 的
   MT19937 + `exponential_`（golden 里 `noise_all` 是参考的真实抽样，可用于校验）。
6. **粤语 TN**：从 `opencc-python-reimplemented` 抽 s2hk 表入库。
7. **流式解码**（chunk 300 / left context 25，参考 `chunked_decode`）、**场景矩阵**、
   **README**。
8. **清理死代码**：`ArenaKernel`/`AF_*`/`arena_kernel`、旧 arena depth 路径
   （`llama_layer_depth_arena`/`DepthBf16Weights`/`M5_DEPTH_ARENA`，其中 down_proj 的
   in_features 误传 HIDDEN 是 bug）。

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
