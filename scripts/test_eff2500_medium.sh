#!/usr/bin/env bash
# Test the winning eff-2500 checkpoint on the POST-TRAINED `medium` model, to see
# if it beats medium-base (which scored timbre 3.50, floor 0 on this checkpoint).
# Same prompts + seeds as the medium-base sweep -> direct A/B.
#
# Post-trained regime: --steps 8 (not 50), and cfg_scale has NO effect (omitted).
set -euo pipefail
cd "$(dirname "$0")/.."

LORA="lora_out/ckz_melody_capped/epoch=54-step=2500.ckpt"
OUT="sweep_rater/campaigns/eff2500_medium"
mkdir -p "$OUT"
DURATION=20; STEPS=8

PROMPTS=(
  "2. cKz! ICE Dmin 135 bpm @crushed_keyz"
  "cKz! AzUL Fmin 140 bpm @crushed_keyz"
)
SEEDS=(7 42 123 314 555 999)

slug() { echo "$1" | tr -cd '[:alnum:] ' | tr ' ' '_' | cut -c1-20; }

for p in "${PROMPTS[@]}"; do
  s=$(slug "$p")
  for seed in "${SEEDS[@]}"; do
    echo ">>> medium | $s | seed $seed"
    uv run stable-audio --model medium --lora-ckpt-path "$LORA" \
      --prompt "$p" --duration "$DURATION" --steps "$STEPS" --seed "$seed" \
      --output "$OUT/medium__${s}__seed${seed}.wav"
  done
done

echo ""
echo "Rate, then compare to eff2500 on medium-base (T 3.50 / M 2.67 / floor 0):"
echo "  python sweep_rater/rate_server.py --folder $OUT"
