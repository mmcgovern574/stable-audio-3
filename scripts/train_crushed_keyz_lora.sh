#!/usr/bin/env bash
# LoRA fine-tune on Crushed Keyz melody loops (medium-base).
# Runs on CUDA or Apple Silicon (MPS); precision auto-picks per device.
# Prepare data first:
#   uv run python scripts/prepare_crushed_keyz_lora_data.py
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-./data/crushed_keyz_lora}"
SAVE_DIR="${SAVE_DIR:-./lora_out/crushed_keyz}"
STEPS="${STEPS:-2000}"
DURATION="${DURATION:-30}"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing $DATA_DIR — run: uv run python scripts/prepare_crushed_keyz_lora_data.py"
  exit 1
fi

uv sync --extra lora

uv run python scripts/train_lora.py \
  --model medium-base \
  --data_dir "$DATA_DIR" \
  --duration "$DURATION" \
  --rank 16 \
  --adapter_type dora-rows \
  --exclude seconds_total \
  --steps "$STEPS" \
  --batch_size 1 \
  --checkpoint_every 500 \
  --demo_every 500 \
  --save_dir "$SAVE_DIR" \
  --name crushed-keyz-lora

echo ""
echo "Checkpoints: $SAVE_DIR"
echo "Inference example (use latest epoch=*-step=*.ckpt in $SAVE_DIR):"
echo "  uv run stable-audio --model medium-base \\"
echo "    --lora-ckpt-path $SAVE_DIR/epoch=0-step=${STEPS}.ckpt \\"
echo "    -p \"cKz! Soulful Shadows Cmin 95 @crushed_key, sparse melodic keys loop\" \\"
echo "    --duration 30 --steps 8 --cfg 1.0 --out ckz_lora_test.wav"
