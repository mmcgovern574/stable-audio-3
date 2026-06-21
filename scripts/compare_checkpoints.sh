#!/usr/bin/env bash
# Generate the same Crushed Keyz prompts from every raw-name checkpoint so you
# can A/B them in one folder and find the quality peak (before over-training).
#
# Each clip uses a FIXED seed so checkpoints are compared apples-to-apples.
# Output: sweep_rater/campaigns/checkpoint_compare/<effstep>__<prompt>.wav
# Then rate/compare:
#   python sweep_rater/rate_server.py --folder sweep_rater/campaigns/checkpoint_compare
#
# NOTE: stop any running training first (Ctrl-C) so the GPU is free.
# NOTE: each clip reloads the model (~30-60s), so this is slow but reliable.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="sweep_rater/campaigns/checkpoint_compare_posttrain"
mkdir -p "$OUT"

# Post-trained (ARC) inference recipe — the intended high-quality path.
# Per the docs: use the post-trained `medium` model, ~8 steps, and cfg_scale /
# negative_prompt have NO effect on post-trained checkpoints (base-only).
MODEL="medium"          # post-trained, NOT medium-base
STEPS=8                 # post-trained uses ~8 steps (50 is for -base only)
DURATION=12
SEED=42

# EXACT Crushed Keyz training caption strings (= the mp3 filename stems the
# model was actually trained on). Swap in any from the dataset, e.g.:
#   "10. cKz! JUNGLE C#min 140 bpm @crushed_keyz"
#   "21. cKz! Panda Cmin 90 bpm @crushed_keyz"
#   "cKz! Scenic Cmin 110 bpm @crushed_keyz"
#   "25. cKz! CoSmoS Dmin 132 bpm @crushed_keyz"
PROMPTS=(
  "2. cKz! ICE Dmin 135 bpm @crushed_keyz"
  "cKz! AzUL Fmin 140 bpm @crushed_keyz"
  "10. cKz! JUNGLE C#min 140 bpm @crushed_keyz"
)

# label (effective step) : checkpoint path  — edit/trim as you like
CKPTS=(
  "eff0500:lora_out/hiphop_rawnames/epoch=0-step=500.ckpt"
  "eff1000:lora_out/hiphop_rawnames/epoch=1-step=1000.ckpt"
  "eff1500:lora_out/hiphop_rawnames/epoch=2-step=1500.ckpt"
  "eff2000:lora_out/hiphop_rawnames/epoch=3-step=2000.ckpt"
  "eff2500:lora_out/hiphop_rawnames/epoch=4-step=2500.ckpt"
  "eff3000:lora_out/hiphop_rawnames_cont/epoch=0-step=500.ckpt"
  "eff3500:lora_out/hiphop_rawnames_cont/epoch=1-step=1000.ckpt"
  "eff4000:lora_out/hiphop_rawnames_cont/epoch=2-step=1500.ckpt"
  "eff4500:lora_out/hiphop_rawnames_cont/epoch=3-step=2000.ckpt"
  "eff5000:lora_out/hiphop_rawnames_cont/epoch=4-step=2500.ckpt"
  "eff5500:lora_out/hiphop_rawnames_cont/epoch=4-step=3000.ckpt"
  "eff6000:lora_out/hiphop_rawnames_cont/epoch=5-step=3500.ckpt"
)

slug() { echo "$1" | tr -cd '[:alnum:] ' | tr ' ' '_' | cut -c1-40; }

for entry in "${CKPTS[@]}"; do
  label="${entry%%:*}"; ckpt="${entry#*:}"
  if [[ ! -f "$ckpt" ]]; then echo "SKIP (missing): $ckpt"; continue; fi
  for p in "${PROMPTS[@]}"; do
    out="$OUT/${label}__$(slug "$p")__seed${SEED}.wav"
    echo ">>> $label  |  $p"
    uv run stable-audio \
      --model "$MODEL" \
      --lora-ckpt-path "$ckpt" \
      --prompt "$p" \
      --duration "$DURATION" \
      --steps "$STEPS" \
      --seed "$SEED" \
      --output "$out"
  done
done

echo ""
echo "Done. Compare them:"
echo "  python sweep_rater/rate_server.py --folder $OUT"
