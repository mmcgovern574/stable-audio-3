#!/usr/bin/env bash
# Seed sweep across the 3 capped-crop candidate checkpoints, same conditions, so
# we pick the winner on a tight/varied distribution instead of single-seed demos.
#   eff2500 = ckz_melody_capped/epoch=54-step=2500
#   eff3000 = ckz_melody_capped/epoch=65-step=3000
#   eff4000 = ckz_melody_capped_cont/epoch=21-step=1000  (effective 4000)
#
# Decide on the FLOOR + variety, not the average:
#   - floor: fewest T1/T2 (single-instrument) clips = most consistent timbre
#   - variety: do the seeds give DIFFERENT melodies (good) or the same (overfit)
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="sweep_rater/campaigns/seedsweep_candidates"
mkdir -p "$OUT"
MODEL="medium-base"; DURATION=20; STEPS=50; CFG=3

CKPTS=(
  "eff2500:lora_out/ckz_melody_capped/epoch=54-step=2500.ckpt"
  "eff3000:lora_out/ckz_melody_capped/epoch=65-step=3000.ckpt"
  "eff4000:lora_out/ckz_melody_capped_cont/epoch=21-step=1000.ckpt"
)
PROMPTS=(
  "2. cKz! ICE Dmin 135 bpm @crushed_keyz"
  "cKz! AzUL Fmin 140 bpm @crushed_keyz"
)
SEEDS=(7 42 123 314 555 999)

slug() { echo "$1" | tr -cd '[:alnum:] ' | tr ' ' '_' | cut -c1-20; }

for entry in "${CKPTS[@]}"; do
  label="${entry%%:*}"; ckpt="${entry#*:}"
  [[ -f "$ckpt" ]] || { echo "SKIP missing: $ckpt"; continue; }
  for p in "${PROMPTS[@]}"; do
    s=$(slug "$p")
    for seed in "${SEEDS[@]}"; do
      echo ">>> $label | $s | seed $seed"
      uv run stable-audio --model "$MODEL" --lora-ckpt-path "$ckpt" \
        --prompt "$p" --duration "$DURATION" --steps "$STEPS" --cfg-scale "$CFG" --seed "$seed" \
        --output "$OUT/${label}__${s}__seed${seed}.wav"
    done
  done
done

echo ""
echo "Rate (timbre + melody), then I'll break it down by checkpoint:"
echo "  python sweep_rater/rate_server.py --folder $OUT"
