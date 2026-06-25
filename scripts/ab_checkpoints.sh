#!/usr/bin/env bash
# A/B the SAME prompts+seeds across multiple LoRA checkpoints into one folder,
# so the rater can line up the same title+seed across checkpoints.
#
# GOAL: does any pitch-aug checkpoint reach the v1 (46-sample) champion's TIMBRE?
# v1-2500 is included as the target. All checkpoints run at identical settings,
# so the only variable is the checkpoint.
#
# Filenames lead with title+seed so name-sort in the rater groups the A/B:
#   <kind>__<title>__seed<N>__<label>_cfg<N>.wav
#
# Run with the training job STOPPED (needs the MPS GPU):
#   bash scripts/ab_checkpoints.sh
#   # then: uv run python sweep_rater/rate_server.py --folder <OUT_DIR>
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-medium-base}"
STEPS="${STEPS:-50}"
CFG="${CFG:-4}"
DURATION="${DURATION:-20}"
STRENGTH="${STRENGTH:-1.0}"   # must be 1.0 — lowering muffles (eval finding)
OUT_DIR="${OUT_DIR:-./sweep_rater/campaigns/ab_timbre_v1_vs_aug}"
read -ra SEEDS <<< "${SEEDS:-7 123}"
mkdir -p "$OUT_DIR"

# label|checkpoint   (label sorts/groups in the rater). eff = raw step + 5500.
CKPTS=(
  "v1-2500|./lora_out/ckz_melody_capped/epoch=54-step=2500.ckpt"     # TIMBRE TARGET (46-sample, T3.5)
  "aug-07500|./lora_out/ckz_aug_v1_long/epoch=9-step=2000.ckpt"
  "aug-11500|./lora_out/ckz_aug_v1_long/epoch=29-step=6000.ckpt"
  "aug-15500|./lora_out/ckz_aug_v1_long/epoch=49-step=10000.ckpt"
)

# kind|prompt
PROMPTS=(
  "train|2. cKz! ICE Dmin 135 bpm @crushed_keyz"   # both v1+aug trained on it -> fair timbre A/B
  "novel|cKz! Mirage Gmin 120 bpm @crushed_keyz"   # generalization check
)

sanitize(){ printf '%s' "$1" | tr ' ' '_' | tr -d '/\\:|?*<>"'"'" | tr -s '_' | cut -c1-80; }

n=0
for pair in "${CKPTS[@]}"; do
  label="${pair%%|*}"; ckpt="${pair##*|}"
  if [[ ! -f "$ckpt" ]]; then echo "!! MISSING $label: $ckpt"; continue; fi
  for pp in "${PROMPTS[@]}"; do
    kind="${pp%%|*}"; prompt="${pp##*|}"
    base=$(sanitize "$prompt")
    for seed in "${SEEDS[@]}"; do
      out="$OUT_DIR/${kind}__${base}__seed${seed}__${label}_cfg${CFG}.wav"
      if [[ -f "$out" ]]; then echo "skip: $(basename "$out")"; continue; fi
      n=$((n+1))
      echo ">> [$label / $kind / seed $seed] $prompt"
      uv run stable-audio \
        --model "$MODEL" --lora-ckpt-path "$ckpt" --lora-strength "$STRENGTH" \
        -p "$prompt" --duration "$DURATION" --steps "$STEPS" --cfg-scale "$CFG" \
        --seed "$seed" --out "$out"
    done
  done
done

echo
echo "Generated $n clips -> $OUT_DIR"
echo "Rate: uv run python sweep_rater/rate_server.py --folder $OUT_DIR"
echo "Sort by name to compare v1-2500 vs aug-* on the SAME title+seed."
