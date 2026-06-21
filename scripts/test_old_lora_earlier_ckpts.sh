#!/usr/bin/env bash
# Test Matt's overfit hypothesis: compare EARLIER checkpoints of the old 46-file
# LoRA against the final (step 5000) one, with MULTIPLE SEEDS per cell.
#
# The tell: if step5000 produces near-identical melodies across seeds (memorized)
# while step1000/3000 give varied melodies for the same prompt, it's over-fit and
# an earlier checkpoint is the keeper. The loss minimum was ~step 3100.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="sweep_rater/campaigns/old_lora_earlier_ckpts"
mkdir -p "$OUT"
MODEL="medium-base"
STEPS=50
CFG=3
DURATION=12

# earlier (less overfit) → final (what we've been using)
CKPTS=(
  "step1000:lora_out/ckz_5000_dora/epoch=21-step=1000.ckpt"
  "step3000:lora_out/ckz_5000_dora/epoch=65-step=3000.ckpt"
  "step5000:lora_out/ckz_5000_dora/epoch=108-step=5000.ckpt"
)
PROMPTS=(
  "2. cKz! ICE Dmin 135 bpm @crushed_keyz"
  "cKz! AzUL Fmin 140 bpm @crushed_keyz"
)
SEEDS=(7 42 123 999)

slug() { echo "$1" | tr -cd '[:alnum:] ' | tr ' ' '_' | cut -c1-28; }

for entry in "${CKPTS[@]}"; do
  label="${entry%%:*}"; ckpt="${entry#*:}"
  if [[ ! -f "$ckpt" ]]; then echo "SKIP missing: $ckpt"; continue; fi
  for p in "${PROMPTS[@]}"; do
    s=$(slug "$p")
    for seed in "${SEEDS[@]}"; do
      echo ">>> $label | seed $seed | $p"
      uv run stable-audio --model "$MODEL" --lora-ckpt-path "$ckpt" \
        --prompt "$p" --duration "$DURATION" --steps "$STEPS" --cfg-scale "$CFG" --seed "$seed" \
        --output "$OUT/${label}__${s}__seed${seed}.wav"
    done
  done
done

echo ""
echo "Rate (timbre + melody). Within each checkpoint+prompt, listen across the 4"
echo "seeds: do the melodies VARY (good) or are they all the same (over-fit)?"
echo "  python sweep_rater/rate_server.py --folder $OUT"
