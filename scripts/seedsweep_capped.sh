#!/usr/bin/env bash
# Seed sweep on the capped-crop step-2500 checkpoint, under the SAME conditions
# as validate_step3000 (medium-base, 20s, 50 steps, cfg 3, same 10 seeds) so the
# distributions are directly comparable.
#
# The question this answers: did the capped-crop (melody-only) training lift the
# FLOOR? i.e. fewer T1/T2 single-instrument failures, more consistent dense
# timbre across seeds — vs step3000's timbre 2.90 with several T1/T2 seeds.
set -euo pipefail
cd "$(dirname "$0")/.."

LORA="lora_out/ckz_melody_capped/epoch=54-step=2500.ckpt"
OUT="sweep_rater/campaigns/seedsweep_capped2500"
mkdir -p "$OUT"
MODEL="medium-base"; DURATION=20; STEPS=50; CFG=3

PROMPTS=(
  "2. cKz! ICE Dmin 135 bpm @crushed_keyz"
  "cKz! AzUL Fmin 140 bpm @crushed_keyz"
)
SEEDS=(7 42 101 123 200 314 420 555 777 999)

slug() { echo "$1" | tr -cd '[:alnum:] ' | tr ' ' '_' | cut -c1-28; }

for p in "${PROMPTS[@]}"; do
  s=$(slug "$p")
  for seed in "${SEEDS[@]}"; do
    echo ">>> $s | seed $seed"
    uv run stable-audio --model "$MODEL" --lora-ckpt-path "$LORA" \
      --prompt "$p" --duration "$DURATION" --steps "$STEPS" --cfg-scale "$CFG" --seed "$seed" \
      --output "$OUT/${s}__seed${seed}.wav"
  done
done

echo ""
echo "Rate (timbre + melody):"
echo "  python sweep_rater/rate_server.py --folder $OUT"
