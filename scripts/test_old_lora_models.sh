#!/usr/bin/env bash
# Hold the OLD (timbre-perfect) 46-file LoRA constant and compare inference models:
#   medium-base  (what all your gold results used)   vs   medium (post-trained, never tried)
# Same prompts + seed, so the ONLY variable is the inference model.
set -euo pipefail
cd "$(dirname "$0")/.."

OLD_LORA="lora_out/ckz_5000_dora/epoch=108-step=5000.ckpt"
OUT="sweep_rater/campaigns/old_lora_model_test"
mkdir -p "$OUT"
DURATION=12
SEED=42

PROMPTS=(
  "2. cKz! ICE Dmin 135 bpm @crushed_keyz"
  "cKz! AzUL Fmin 140 bpm @crushed_keyz"
  "cKz! Midnight Bmin 110 bpm @crushed_keyz"
)

slug() { echo "$1" | tr -cd '[:alnum:] ' | tr ' ' '_' | cut -c1-32; }

for p in "${PROMPTS[@]}"; do
  s=$(slug "$p")
  # medium-base: needs ~50 steps + cfg (this reproduces your gold-result settings)
  echo ">>> medium-base | $p"
  uv run stable-audio --model medium-base --lora-ckpt-path "$OLD_LORA" \
    --prompt "$p" --duration "$DURATION" --steps 50 --cfg-scale 3 --seed "$SEED" \
    --output "$OUT/base__${s}__seed${SEED}.wav"
  # medium (post-trained): ~8 steps, cfg has no effect
  echo ">>> medium (post-trained) | $p"
  uv run stable-audio --model medium --lora-ckpt-path "$OLD_LORA" \
    --prompt "$p" --duration "$DURATION" --steps 8 --seed "$SEED" \
    --output "$OUT/posttrain__${s}__seed${SEED}.wav"
done

echo ""
echo "Compare (base vs posttrain, same old LoRA):"
echo "  python sweep_rater/rate_server.py --folder $OUT"
