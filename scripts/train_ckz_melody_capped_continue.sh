#!/usr/bin/env bash
# Continue the capped-crop CKz run from step 3000, same settings, +2000 steps
# (effective ~5000). Saves to a new dir so the 0.5k–3k checkpoints stay intact.
# NOTE: resume loads weights only (global_step restarts at 0), so checkpoints
# here named step=500..2000 = EFFECTIVE 3500..5000.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-./data/crushed_keyz_lora}"
FROM_CKPT="${FROM_CKPT:-./lora_out/ckz_melody_capped/epoch=65-step=3000.ckpt}"
SAVE_DIR="${SAVE_DIR:-./lora_out/ckz_melody_capped_cont}"
STEPS="${STEPS:-2000}"
DURATION="${DURATION:-30}"
MAX_OFFSET="${MAX_OFFSET:-15}"
RANK="${RANK:-16}"

[[ -f "$FROM_CKPT" ]] || { echo "Missing $FROM_CKPT"; exit 1; }

uv sync --extra lora

uv run python scripts/train_lora.py \
  --model medium-base \
  --data_dir "$DATA_DIR" \
  --lora_checkpoint "$FROM_CKPT" \
  --duration "$DURATION" \
  --max_crop_offset_sec "$MAX_OFFSET" \
  --rank "$RANK" \
  --adapter_type dora-rows \
  --exclude seconds_total \
  --steps "$STEPS" \
  --batch_size 1 \
  --checkpoint_every 500 \
  --demo_every 500 \
  --demo_prompts_file ./data/demo_prompts_ckz_exact.txt \
  --demo_duration 12 \
  --num_inpaint_demos 0 \
  --demo_cfg_scales 4 \
  --save_dir "$SAVE_DIR" \
  --name ckz-melody-capped-cont

echo ""
echo "Checkpoints: $SAVE_DIR (step=500..2000 = effective 3500..5000)"
echo "After this, sweep 2500/3000/4000/5000 to pick the winner — the latest is NOT"
echo "automatically best (see step3000 > step5000 last time)."
