#!/usr/bin/env bash
# init_audio on the WINNING eff-2500 checkpoint (clean floor, timbre 3.50) to lift
# MELODY: supply the melody via init_audio, let the LoRA skin it in CKz timbre.
# All inits are C minor (key-match rule); prompt is a CKz C-min title.
# medium-base (the confirmed model), 50 steps, cfg 3, noise band 0.42/0.46/0.52.
set -euo pipefail
cd "$(dirname "$0")/.."

MEL="/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops"
LORA="lora_out/ckz_melody_capped/epoch=54-step=2500.ckpt"
OUT="sweep_rater/campaigns/init_audio_eff2500"
mkdir -p "$OUT"
MODEL="medium-base"; DURATION=20; STEPS=50; CFG=3; SEED=42
PROMPT="cKz! Scenic Cmin 110 bpm @crushed_keyz"

# t2m baseline (no init) for reference
echo ">>> baseline (no init)"
uv run stable-audio --model "$MODEL" --lora-ckpt-path "$LORA" \
  --prompt "$PROMPT" --duration "$DURATION" --steps "$STEPS" --cfg-scale "$CFG" --seed "$SEED" \
  --output "$OUT/t2m_noinit__seed${SEED}.wav"

INITS=(
  "ckzPanda:$MEL/Crushed_keyz_melody_loops/21. cKz! Panda Cmin 90 bpm @crushed_keyz.mp3"
  "cymEmotional:$MEL/Cymatics - Dragon Pt. 5/Sad/Cymatics - Dragon Sad Loop - Emotional - 140 BPM C Min.wav"
  "cymOracleDark9:$MEL/Cymatics Oracle/Melodies/Cymatics - Oracle Dark Melody Loop 9 - 140 BPM C Min.wav"
)
for entry in "${INITS[@]}"; do
  label="${entry%%:*}"; init="${entry#*:}"
  [[ -f "$init" ]] || { echo "SKIP missing: $init"; continue; }
  for nz in 0.42 0.46 0.52; do
    echo ">>> $label | noise $nz"
    uv run stable-audio --model "$MODEL" --lora-ckpt-path "$LORA" \
      --prompt "$PROMPT" --duration "$DURATION" --steps "$STEPS" --cfg-scale "$CFG" --seed "$SEED" \
      --init-audio "$init" --init-noise-level "$nz" \
      --output "$OUT/${label}_noise${nz}__seed${SEED}.wav"
  done
done

echo ""
echo "Rate (timbre + melody). Compare to t2m baseline + to eff2500 t2m melody 2.67:"
echo "  python sweep_rater/rate_server.py --folder $OUT"
