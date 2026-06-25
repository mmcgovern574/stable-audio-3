#!/usr/bin/env bash
# Continue training the AUGMENTED cKz model from aug-8000 (~23.5k cumulative steps)
# to test whether novel-prompt quality keeps climbing (the sweet-spot sweep showed
# it was still rising at the latest checkpoint — possibly UNDER-trained, unlike v1).
#
# CLEAR NAMING: this run's checkpoints land in ckz_aug_cum23k. The step counter
# RESTARTS at 0 here, so a file "step=2000" = CUMULATIVE ~25500 (23500 base + 2000).
# After running, regenerate the manifest to get exact cumulative numbers:
#     uv run python scripts/checkpoint_manifest.py    # updates lora_out/CHECKPOINTS.md
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-./data/ckz_aug}"                                   # the 204-clip augmented set
FROM_CKPT="${FROM_CKPT:-./lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt}"  # aug-8000 (~23.5k cum)
SAVE_DIR="${SAVE_DIR:-./lora_out/ckz_aug_cum23k}"                        # step=N here = ~23500+N cumulative
STEPS="${STEPS:-6000}"                                                   # -> ~29.5k cumulative
DURATION="${DURATION:-30}"
MAX_OFFSET="${MAX_OFFSET:-15}"
RANK="${RANK:-16}"
CKPT_EVERY="${CKPT_EVERY:-2000}"     # checkpoints at cum ~25.5k / 27.5k / 29.5k
DEMO_EVERY="${DEMO_EVERY:-2000}"

[[ -f "$FROM_CKPT" ]] || { echo "Missing $FROM_CKPT"; exit 1; }
[[ -d "$DATA_DIR" ]]  || { echo "Missing $DATA_DIR"; exit 1; }

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
  --checkpoint_every "$CKPT_EVERY" \
  --demo_every "$DEMO_EVERY" \
  --demo_prompts_file ./data/demo_prompts_ckz_novel.txt \
  --demo_duration 12 \
  --num_inpaint_demos 0 \
  --demo_cfg_scales 3 \
  --save_dir "$SAVE_DIR" \
  --name ckz-aug-cum23k

echo ""
echo "Done. Checkpoints in $SAVE_DIR  (step=N  =>  CUMULATIVE ~23500+N)."
echo "1) Update the manifest:  uv run python scripts/checkpoint_manifest.py"
echo "2) Track each new ckpt:   uv run python scripts/track_training.py --ckpt $SAVE_DIR/<ckpt> --label cum<NNk>"
echo "3) Rate by ear (3 axes):  uv run python sweep_rater/rate_server.py --folder sweep_rater/campaigns/track_cum<NNk> --port 8799"
echo "Compare novel melody+novelty vs the aug-8000-base baseline you already rated."
