#!/usr/bin/env bash
# LoRA fine-tune on the FULL hip-hop melody-loop library (Crushed Keyz + Cymatics),
# tuned to avoid the memorization seen in the 46-file run.
#
# Prereq — generate captions + filelist once:
#   uv run python scripts/build_hiphop_lora_dataset.py --write
#
# Key differences vs scripts/train_crushed_keyz_lora.sh:
#   * data_dir points at the whole library; a filelist.txt limits it to the
#     623 full loops (stems/reverse excluded).
#   * ~4 epochs instead of ~43. 623 files x 4 = ~2500 steps. The old run was
#     2000 steps / 46 files = 43 passes, which is why it memorized.
#   * duration 12s ~ one full melodic loop, so each sample is a complete idea.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops}"
SAVE_DIR="${SAVE_DIR:-./lora_out/hiphop_rawnames}"
STEPS="${STEPS:-2500}"     # ~4 epochs over 623 files
DURATION="${DURATION:-12}" # full-loop-sized window
RANK="${RANK:-16}"         # keep timbre fidelity; drop to 8 if it still copies

if [[ ! -f "$DATA_DIR/filelist.txt" ]]; then
  echo "Missing $DATA_DIR/filelist.txt — run:"
  echo "  uv run python scripts/build_hiphop_lora_dataset.py --write"
  exit 1
fi

uv sync --extra lora

uv run python scripts/train_lora.py \
  --model medium-base \
  --data_dir "$DATA_DIR" \
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
  --name hiphop-full-lora

echo ""
echo "Checkpoints: $SAVE_DIR  (epoch=*-step=*.ckpt)"
echo ""
echo "PICK THE RIGHT CHECKPOINT — do not assume the last one is best:"
echo "  Generate from a NOVEL prompt (not a training title), e.g.:"
echo "    \"crushed keyz, dark, hip hop melody loop, F minor, 140 bpm\""
echo "  Walk the checkpoints 500 -> 2500. Keep the LATEST one whose novel-prompt"
echo "  output still sounds original. The moment outputs start matching specific"
echo "  training loops, that checkpoint is over-fit — step back one."
