#!/usr/bin/env bash
# Retrain on the 46 Crushed Keyz loops with CAPPED random crops, so every crop
# stays in the dense melody body and never samples the trailing step-outs.
#
# Why: the "single instrument / muffled" low-timbre failures were the model
# reproducing the step-out tails of the loops. With duration 30s and
# --max_crop_offset_sec 15, crops live in [0, 45s] — always melody, zero
# step-outs — while keeping random-crop augmentation (offset 0–15s) to fight
# overfitting. Stop near the loss minimum (~3000 steps), not 108 epochs.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-./data/crushed_keyz_lora}"     # 46 CKz, filename captions
SAVE_DIR="${SAVE_DIR:-./lora_out/ckz_melody_capped}"
STEPS="${STEPS:-3000}"            # ~loss-minimum sweet spot; ckpts every 500
DURATION="${DURATION:-30}"        # crop window
MAX_OFFSET="${MAX_OFFSET:-15}"    # crop start in [0,15] -> crop in [0,45], melody-only
RANK="${RANK:-16}"

if [[ ! -d "$DATA_DIR" ]]; then echo "Missing $DATA_DIR"; exit 1; fi

uv sync --extra lora

uv run python scripts/train_lora.py \
  --model medium-base \
  --data_dir "$DATA_DIR" \
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
  --name ckz-melody-capped

echo ""
echo "Checkpoints: $SAVE_DIR  (every 500 steps)"
echo "Pick the best by ear/rating around ~2000–3000 (loss min). Listen for: full"
echo "DENSE timbre on every seed, no single-instrument/muffled outputs."
