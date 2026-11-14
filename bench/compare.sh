#!/usr/bin/env bash
# Paired CPU benchmark: Rust (AVX2 and general) against the Python reference.
#
# Cross-run numbers on this box drift by 2x when anything else is busy, so every
# implementation is measured in the same session, with the same text, the same
# thread count, and best-of-N is reported. Run it while the machine is otherwise
# idle and note the commit under test.
#
#   bash bench/compare.sh [threads] [runs]
set -u

THREADS="${1:-20}"
RUNS="${2:-3}"
TEXT="你好，欢迎使用我们的语音合成模型，今天天气很好。"
PROMPT_TEXT="猪携带该疾病，然后通过蚊子传染给人类。"
WAV="D:/M5-TTS/assets/aslp_default_voice.wav"
WEIGHTS="D:/M5-TTS/weights"
BIN="./target/release/m5-tts.exe"

best() { sort -n | head -1; }

echo "commit: $(git rev-parse --short HEAD)  threads: $THREADS  runs: $RUNS"
echo "load:   $(powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).LoadPercentage" 2>/dev/null | tr -d '\r')%"

peak=0
for i in $(seq "$RUNS"); do
  out=$(RAYON_NUM_THREADS="$THREADS" "$BIN" \
    --model "$WEIGHTS/M5-TTS-50M-VoiceClone" --codec "$WEIGHTS/M5-TTS-Audio-Tokenizer" \
    --device cpu --text "$TEXT" --prompt-wav "$WAV" --prompt-text "$PROMPT_TEXT" \
    --output "outputs/bench_rust_$i.wav" 2>&1 | grep -oE "in [0-9.]+s, RTF [0-9.]+")
  rtf=$(echo "$out" | grep -oE "RTF [0-9.]+" | awk '{print $2}')
  echo "  rust avx2  run$i: RTF $rtf"
  echo "$rtf"
done | tee /tmp/rust_avx.txt >/dev/null
rust_avx=$(awk '{print $NF}' /tmp/rust_avx.txt | grep -E '^[0-9]' | best)

for i in $(seq "$RUNS"); do
  out=$(RAYON_NUM_THREADS="$THREADS" M5_NO_SIMD=1 "$BIN" \
    --model "$WEIGHTS/M5-TTS-50M-VoiceClone" --codec "$WEIGHTS/M5-TTS-Audio-Tokenizer" \
    --device cpu --text "$TEXT" --prompt-wav "$WAV" --prompt-text "$PROMPT_TEXT" \
    --output "outputs/bench_rust_gen_$i.wav" 2>&1 | grep -oE "in [0-9.]+s, RTF [0-9.]+")
  rtf=$(echo "$out" | grep -oE "RTF [0-9.]+" | awk '{print $2}')
  echo "  rust plain run$i: RTF $rtf"
done | tee /tmp/rust_gen.txt >/dev/null
rust_gen=$(awk '{print $NF}' /tmp/rust_gen.txt | grep -E '^[0-9]' | best)

cd /d/M5-TTS || exit 1
source outputs/runenv.sh
PY_MEASURE=$(cat <<'PY'
import os, sys, time, torch
sys.path.insert(0, r"D:\M5-TTS")
torch.set_num_threads(int(os.environ["PY_THREADS"]))
from inference import M5TTS
tts = M5TTS.from_pretrained(os.environ["CLONE_MODEL"], codec=os.environ["CODEC"],
                            device="cpu", attention="sdpa", local_files_only=True)
text = "你好，欢迎使用我们的语音合成模型，今天天气很好。"
pt = "猪携带该疾病，然后通过蚊子传染给人类。"
wav = r"D:\M5-TTS\assets\aslp_default_voice.wav"
tts.infer("你好。", prompt_wav=wav, prompt_text=pt, lang="Chinese")
t = time.perf_counter()
audio = tts.infer(text, prompt_wav=wav, prompt_text=pt, lang="Chinese")
elapsed = time.perf_counter() - t
print(f"  {os.environ['PY_LABEL']} run{os.environ['PY_RUN']}: RTF {elapsed / (audio.numel() / 24000):.2f}")
PY
)
for i in $(seq "$RUNS"); do
  PY_THREADS="$THREADS" PY_LABEL="python avx2 " PY_RUN="$i" $PYTHON -c "$PY_MEASURE" 2>/dev/null | tee -a /tmp/py_avx.txt
done
py_avx=$(grep -oE "RTF [0-9.]+" /tmp/py_avx.txt | awk '{print $2}' | best)

: > /tmp/py_plain.txt
for i in $(seq "$RUNS"); do
  ATEN_CPU_CAPABILITY=default ONEDNN_MAX_CPU_ISA=SSE41 PY_THREADS="$THREADS" \
    PY_LABEL="python plain" PY_RUN="$i" $PYTHON -c "$PY_MEASURE" 2>/dev/null | tee -a /tmp/py_plain.txt
done
py_plain=$(grep -oE "RTF [0-9.]+" /tmp/py_plain.txt | awk '{print $2}' | best)

echo
echo "=== best-of-$RUNS, $THREADS threads, RTF lower is better ==="
printf "  rust   avx2 : %s\n" "$rust_avx"
printf "  python avx2 : %s   (rust %+.0f%%)\n" "$py_avx" \
  "$(awk -v a="$rust_avx" -v b="$py_avx" 'BEGIN {print (a/b-1)*100}')"
echo "  --- without AVX2 ---"
printf "  rust   plain: %s\n" "$rust_gen"
printf "  python plain: %s   (rust %+.0f%%)\n" "$py_plain" \
  "$(awk -v a="$rust_gen" -v b="$py_plain" 'BEGIN {print (a/b-1)*100}')"
