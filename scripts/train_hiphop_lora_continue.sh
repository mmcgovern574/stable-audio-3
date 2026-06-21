#!/usr/bin/env bash
# Continue the hip-hop LoRA from the step=2500 checkpoint for more steps.
# Use when the demo curve was still improving at step 2000 and you want to
# push further without re-doing the first ~1.5 h.
#
# Notes:
#   * Resumes by loading the LoRA WEIGHTS (optimizer/global_step reset), so
#     --steps below is the number of ADDITIONAL steps. global_step restarts at
#     0, so new checkpoints are named step=500..2500 in the new save_dir but
#     represent EFFECTIVE steps 3000..5000.
#   * New save_dir so the first run's checkpoints are not overwritten.
#   * Watch the ICE (D minor 135) / Midnight (B minor 110) demos — if they start
#     copying the real loops, stop and step back a checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops}"
FROM_CKPT="${FROM_CKPT:-./lora_out/hiphop_rawnames/epoch=4-step=2500.ckpt}"
SAVE_DIR="${SAVE_DIR:-./lora_out/hiphop_rawnames_cont}"
STEPS="${STEPS:-100000}"    # effectively unlimited — stop with Ctrl-C when it sounds right
DURATION="${DURATION:-12}"
RANK="${RANK:-16}"

if [[ ! -f "$FROM_CKPT" ]]; then
  echo "Missing $FROM_CKPT"; exit 1
fi

uv sync --extra lora

uv run python scripts/train_lora.py \
  --model medium-base \
  --data_dir "$DATA_DIR" \
  --lora_checkpoint "$FROM_CKPT" \
  --duration "$DURATION" \
  --rank "$RANK" \
  --adapter_type dora-rows \
  --exclude seconds_total \
  --steps "$STEPS" \
  --batch_size 1 \
  --checkpoint_every 500 \
  --demo_every 500 \
  --demo_prompts_file ./data/demo_prompts_hiphop.txt \
  --demo_cfg_scales 4 \
  --save_dir "$SAVE_DIR" \
  --name hiphop-full-lora-ext

echo ""
echo "Checkpoints: $SAVE_DIR   (step=500..2500 here = EFFECTIVE 3000..5000)"
echo "The step=1 demo of this run ≈ your current step=2500 model (the starting point)."
echo "Keep the LATEST checkpoint whose ICE/Midnight demos still sound original."
