#!/usr/bin/env bash
# Layered evaluation of the Crushed Keyz LoRA.
# Runs five stages of increasing distance from the training distribution:
#   1. Exact training captions       — memorization check.
#   2. Same titles, new key/BPM      — small variation.
#   3. Novel cKz-style prompts       — full generalization.
#   4. Cross-genre w/ cKz signature  — testing if the @crushed_keyz tag pulls style across genres.
#   5. Off-distribution edge cases   — finds where the model breaks.
#
# Filenames are stage-prefixed so `ls -1 eval_out/` reads top-to-bottom from
# "closest to training" to "furthest from training".
#
# Usage:   bash scripts/eval_ckz_lora.sh
# Output:  ./eval_out/<stage>_<sanitized prompt>_cfg{N}_steps{N}_dur{N}s.wav
set -euo pipefail
cd "$(dirname "$0")/.."

LORA_CKPT="${LORA_CKPT:-./lora_out/ckz_5000_dora/epoch=108-step=5000.ckpt}"
MODEL="${MODEL:-medium-base}"
DURATION="${DURATION:-30}"
STEPS="${STEPS:-50}"
CFG="${CFG:-7}"
OUT_DIR="${OUT_DIR:-./eval_out}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$LORA_CKPT" ]]; then
  echo "Missing checkpoint: $LORA_CKPT"
  echo "Look in ./lora_out/ckz_5000_dora/ for the correct filename and set"
  echo "LORA_CKPT=... before running."
  ls -1 ./lora_out/ckz_5000_dora/ 2>/dev/null || true
  exit 1
fi

sanitize() {
  local s="$1"
  s=$(printf '%s' "$s" | tr ' ' '_' | tr -d '/\\:|?*<>"'"'" | tr -s '_')
  printf '%s' "${s:0:170}"
}

run_prompt() {
  local stage="$1" prompt="$2"
  local clean filename out
  clean=$(sanitize "$prompt")
  filename="${stage}_${clean}_cfg${CFG}_steps${STEPS}_dur${DURATION}s.wav"
  out="$OUT_DIR/$filename"
  if [[ -f "$out" ]]; then
    echo "  skip (exists): $filename"
    return
  fi
  echo "  $prompt"
  echo "    -> $filename"
  uv run stable-audio \
    --model "$MODEL" \
    --lora-ckpt-path "$LORA_CKPT" \
    -p "$prompt" \
    --duration "$DURATION" --steps "$STEPS" --cfg-scale "$CFG" \
    --output "$out"
}

# ─── Stage 1: Exact training captions ────────────────────────────────────
STAGE1=(
  "2. cKz! ICE Dmin 135 bpm @crushed_keyz"
  "10. cKz! JUNGLE C#min 140 bpm @crushed_keyz"
  "cKz! FlaMEnCO Bmin 125 bpm @crushed_keyz"
  "16. cKz! Rent Free Cmin 140 bpm @crushed_keyz @justryda_"
  "cKz! Azkaban Gmin 110 bpm @crushed_keyz"
)

# ─── Stage 2: Same titles, new keys/BPMs ─────────────────────────────────
STAGE2=(
  "cKz! ICE Emin 100 bpm @crushed_keyz"
  "cKz! JUNGLE Bmin 95 bpm @crushed_keyz"
  "cKz! FlaMEnCO Amin 160 bpm @crushed_keyz"
  "cKz! Rent Free Gmin 110 bpm @crushed_keyz"
  "cKz! Azkaban C#min 145 bpm @crushed_keyz"
)

# ─── Stage 3: Novel cKz-style prompts (in distribution) ──────────────────
STAGE3=(
  "cKz! Midnight Bmin 110 bpm @crushed_keyz"
  "cKz! Glassland F#min 125 bpm @crushed_keyz"
  "cKz! Velvet Amin 90 bpm @crushed_keyz"
  "cKz! Ember Dmin 135 bpm @crushed_keyz"
  "cKz! Paloma Gmin 150 bpm @crushed_keyz"
)

# ─── Stage 4: Cross-genre with cKz signature ─────────────────────────────
# Tests if the @crushed_keyz tag pulls cKz timbre into unrelated genres.
STAGE4=(
  "cKz! HouseGroove Cmin 124 bpm @crushed_keyz, four on the floor"
  "cKz! Lo-Fi Sunset Fmin 75 bpm @crushed_keyz, hip-hop drums"
  "cKz! Phonk Amin 130 bpm @crushed_keyz, slowed cowbell"
  "cKz! Drill Pattern G#min 145 bpm @crushed_keyz, sliding 808"
  "cKz! Afrobeats Dmin 110 bpm @crushed_keyz, log drums"
)

# ─── Stage 5: Off-distribution edge cases ────────────────────────────────
# Nothing cKz-style about these. Tests how the LoRA degrades.
STAGE5=(
  "Heavy metal guitar riff with double-kick drums Emin 180 bpm"
  "Solo violin classical piece in A major, romantic style"
  "Glitchy IDM percussion no melody"
  "Ambient drone, no rhythm, low frequencies"
  "cKz! @crushed_keyz"
)

echo "Eval checkpoint: $LORA_CKPT"
echo "Config: cfg=$CFG steps=$STEPS duration=${DURATION}s"
echo "Outputs: $OUT_DIR/"
echo

echo "── Stage 1: Exact training captions ──"
for p in "${STAGE1[@]}"; do run_prompt "1" "$p"; done
echo

echo "── Stage 2: Same titles, new key/BPM ──"
for p in "${STAGE2[@]}"; do run_prompt "2" "$p"; done
echo

echo "── Stage 3: Novel cKz-style ──"
for p in "${STAGE3[@]}"; do run_prompt "3" "$p"; done
echo

echo "── Stage 4: Cross-genre with cKz tag ──"
for p in "${STAGE4[@]}"; do run_prompt "4" "$p"; done
echo

echo "── Stage 5: Off-distribution ──"
for p in "${STAGE5[@]}"; do run_prompt "5" "$p"; done
echo

echo "Done."
ls -lh "$OUT_DIR/" | sort
