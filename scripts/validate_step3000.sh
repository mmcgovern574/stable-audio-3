#!/usr/bin/env bash
# Validate the NEW best checkpoint — ckz_5000_dora @ step3000 (loss minimum,
# beat step5000 on both timbre+melody) — under your GOLD conditions: 20s, a real
# seed sweep, and the init_audio recipe. step3000 has never been tested this way.
#
# Phase 1: text-to-music seed sweep  -> does a good seed produce 4-5★ clips?
# Phase 2: init_audio noise band     -> the recipe your original 5★s came from,
#                                        now on the correct (non-overfit) checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."

MEL="/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops"
LORA="lora_out/ckz_5000_dora/epoch=65-step=3000.ckpt"
OUT="sweep_rater/campaigns/validate_step3000"
mkdir -p "$OUT"
MODEL="medium-base"
DURATION=20
STEPS=50
CFG=3

slug() { echo "$1" | tr -cd '[:alnum:] ' | tr ' ' '_' | cut -c1-28; }

# ── Phase 1: t2m seed sweep ──────────────────────────────────────────────
SWEEP_PROMPTS=(
  "2. cKz! ICE Dmin 135 bpm @crushed_keyz"
  "cKz! AzUL Fmin 140 bpm @crushed_keyz"
)
SEEDS=(7 42 101 123 200 314 420 555 777 999)
for p in "${SWEEP_PROMPTS[@]}"; do
  s=$(slug "$p")
  for seed in "${SEEDS[@]}"; do
    echo ">>> t2m | $s | seed $seed"
    uv run stable-audio --model "$MODEL" --lora-ckpt-path "$LORA" \
      --prompt "$p" --duration "$DURATION" --steps "$STEPS" --cfg-scale "$CFG" --seed "$seed" \
      --output "$OUT/t2m_${s}__seed${seed}.wav"
  done
done

# ── Phase 2: init_audio noise band (C minor) ─────────────────────────────
INIT_PROMPT="cKz! Scenic Cmin 110 bpm @crushed_keyz"
INITS=(
  "ckzPanda:$MEL/Crushed_keyz_melody_loops/21. cKz! Panda Cmin 90 bpm @crushed_keyz.mp3"
  "cymEmotional:$MEL/Cymatics - Dragon Pt. 5/Sad/Cymatics - Dragon Sad Loop - Emotional - 140 BPM C Min.wav"
)
for entry in "${INITS[@]}"; do
  label="${entry%%:*}"; init="${entry#*:}"
  [[ -f "$init" ]] || { echo "SKIP missing: $init"; continue; }
  for nz in 0.42 0.46 0.52; do
    echo ">>> init | $label | noise $nz"
    uv run stable-audio --model "$MODEL" --lora-ckpt-path "$LORA" \
      --prompt "$INIT_PROMPT" --duration "$DURATION" --steps "$STEPS" --cfg-scale "$CFG" --seed 42 \
      --init-audio "$init" --init-noise-level "$nz" \
      --output "$OUT/init_${label}_noise${nz}__seed42.wav"
  done
done

echo ""
echo "Rate (timbre + melody):"
echo "  python sweep_rater/rate_server.py --folder $OUT"
