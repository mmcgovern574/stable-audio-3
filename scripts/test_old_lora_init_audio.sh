#!/usr/bin/env bash
# Retest your PROVEN gold recipe: old 46-file LoRA + init_audio on medium-base.
# Supply the MELODY via init_audio and let the LoRA paint the Crushed Keyz TIMBRE
# on top — the path your 5-star results actually came from, never retested here.
#
# All inits are C minor (key-match rule). Prompt = a CKz C-min title (the timbre
# target). Sweeps your calibrated noise band 0.42 / 0.46 / 0.52:
#   lower noise  = more of the init melody kept (more melody, less CKz timbre)
#   higher noise = more regenerated         (more CKz timbre, less of the melody)
set -euo pipefail
cd "$(dirname "$0")/.."

MEL="/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops"
OLD_LORA="lora_out/ckz_5000_dora/epoch=108-step=5000.ckpt"
OUT="sweep_rater/campaigns/old_lora_init_audio"
mkdir -p "$OUT"

PROMPT="cKz! Scenic Cmin 110 bpm @crushed_keyz"   # CKz timbre target (C min)
SEED=42
DURATION=12
STEPS=50
CFG=3
NOISES=(0.42 0.46 0.52)

# label : init audio file (all C minor)
INITS=(
  "ckzPanda:$MEL/Crushed_keyz_melody_loops/21. cKz! Panda Cmin 90 bpm @crushed_keyz.mp3"
  "cymEmotional:$MEL/Cymatics - Dragon Pt. 5/Sad/Cymatics - Dragon Sad Loop - Emotional - 140 BPM C Min.wav"
  "cymOracleDark9:$MEL/Cymatics Oracle/Melodies/Cymatics - Oracle Dark Melody Loop 9 - 140 BPM C Min.wav"
)

# Pure text-to-music baseline (no init) for reference.
echo ">>> baseline (no init) | $PROMPT"
uv run stable-audio --model medium-base --lora-ckpt-path "$OLD_LORA" \
  --prompt "$PROMPT" --duration "$DURATION" --steps "$STEPS" --cfg-scale "$CFG" --seed "$SEED" \
  --output "$OUT/t2m_noinit__seed${SEED}.wav"

for entry in "${INITS[@]}"; do
  label="${entry%%:*}"; init="${entry#*:}"
  if [[ ! -f "$init" ]]; then echo "SKIP missing: $init"; continue; fi
  for nz in "${NOISES[@]}"; do
    echo ">>> $label | noise=$nz"
    uv run stable-audio --model medium-base --lora-ckpt-path "$OLD_LORA" \
      --prompt "$PROMPT" --duration "$DURATION" --steps "$STEPS" --cfg-scale "$CFG" --seed "$SEED" \
      --init-audio "$init" --init-noise-level "$nz" \
      --output "$OUT/${label}_noise${nz}__seed${SEED}.wav"
  done
done

echo ""
echo "Rate (timbre + melody):"
echo "  python sweep_rater/rate_server.py --folder $OUT"
