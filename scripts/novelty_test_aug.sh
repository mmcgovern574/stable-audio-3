#!/usr/bin/env bash
# Novelty vs memorization audition for the pitch-aug long run.
#
# Generates NOVEL (never-trained) cKz titles PLUS a couple TRAINING-title
# references from ONE checkpoint, across 2 seeds, into one folder so the rater
# groups them. The 5 training-prompt demos can't tell learned-style from
# memorization; novel titles can.
#
# LISTEN FOR:
#   novel__*  -> distinct, fresh-but-still-cKz melodies = GENERALIZATION (the win)
#               mush / generic / a copy of a known song  = memorization overfit
#   train__*  -> the memorized sound, as a reference for "too close to training"
#
# Usage (stop the training run first — it needs the MPS GPU):
#   bash scripts/novelty_test_aug.sh
#   # then: uv run python sweep_rater/rate_server.py --folder <OUT_DIR>
set -euo pipefail
cd "$(dirname "$0")/.."

LORA_CKPT="${LORA_CKPT:-./lora_out/ckz_aug_v1_long/epoch=49-step=10000.ckpt}"  # eff-15500
MODEL="${MODEL:-medium-base}"
STEPS="${STEPS:-50}"
CFG="${CFG:-4}"          # matches the training-demo cfg you've been rating
DURATION="${DURATION:-20}"
STRENGTH="${STRENGTH:-1.0}"   # must be 1.0 — lowering muffles (eval finding)
OUT_DIR="${OUT_DIR:-./sweep_rater/campaigns/novelty_aug_eff15500}"
read -ra SEEDS <<< "${SEEDS:-7 123}"
mkdir -p "$OUT_DIR"

[[ -f "$LORA_CKPT" ]] || { echo "Missing checkpoint: $LORA_CKPT"; exit 1; }

# NOVEL titles — verified absent from data/crushed_keyz_lora. Format mirrors
# the training captions so only the title/key/bpm are new.
NOVEL=(
  "cKz! Skyline Fmin 140 bpm @crushed_keyz"
  "cKz! Voltage C#min 150 bpm @crushed_keyz"
  "cKz! Mirage Gmin 120 bpm @crushed_keyz"
)
# TRAINING references — exact captions the model saw (expect memorized melody).
TRAIN=(
  "2. cKz! ICE Dmin 135 bpm @crushed_keyz"
  "cKz! AzUL Fmin 140 bpm @crushed_keyz"
)

sanitize() {
  printf '%s' "$1" | tr ' ' '_' | tr -d '/\\:|?*<>"'"'" | tr -s '_' | cut -c1-120
}

gen() {
  local kind="$1" prompt="$2" seed="$3"
  local base out
  base=$(sanitize "$prompt")
  out="$OUT_DIR/${kind}__${base}__seed${seed}_cfg${CFG}_steps${STEPS}.wav"
  if [[ -f "$out" ]]; then echo "skip (exists): $(basename "$out")"; return; fi
  echo ">> [$kind seed=$seed] $prompt"
  uv run stable-audio \
    --model "$MODEL" \
    --lora-ckpt-path "$LORA_CKPT" \
    --lora-strength "$STRENGTH" \
    -p "$prompt" \
    --duration "$DURATION" --steps "$STEPS" --cfg-scale "$CFG" \
    --seed "$seed" \
    --out "$out"
}

echo "checkpoint: $LORA_CKPT"
echo "out:        $OUT_DIR"
echo "config:     cfg=$CFG steps=$STEPS dur=${DURATION}s strength=$STRENGTH seeds=${SEEDS[*]}"
echo
for seed in "${SEEDS[@]}"; do
  for p in "${NOVEL[@]}"; do gen novel "$p" "$seed"; done
  for p in "${TRAIN[@]}"; do gen train "$p" "$seed"; done
done

echo
echo "Done -> $OUT_DIR"
echo "Rate:  uv run python sweep_rater/rate_server.py --folder $OUT_DIR"
