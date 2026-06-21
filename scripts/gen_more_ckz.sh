#!/usr/bin/env bash
# Generate a broader set of EXACT Crushed Keyz training-set names on the two
# best-rated configs, so we can settle old-LoRA vs new-LoRA across many prompts.
#   newPT   = new raw-name eff6000 checkpoint  on the POST-TRAINED `medium`  (best new result: JUNGLE T4/M4)
#   oldBASE = old 46-file ckz_5000_dora LoRA   on `medium-base`              (best old result: AzUL T4/M3)
# Same seed everywhere. Filenames encode config + prompt so the rater groups them.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="sweep_rater/campaigns/ckz_more"
mkdir -p "$OUT"
SEED=42
DURATION=12

NEW_LORA="lora_out/hiphop_rawnames_cont/epoch=5-step=3500.ckpt"   # eff6000
OLD_LORA="lora_out/ckz_5000_dora/epoch=108-step=5000.ckpt"

# EXACT names from the 46-file training set (verbatim captions the model saw).
PROMPTS=(
  "2. cKz! ICE Dmin 135 bpm @crushed_keyz"
  "cKz! AzUL Fmin 140 bpm @crushed_keyz"
  "10. cKz! JUNGLE C#min 140 bpm @crushed_keyz"
  "21. cKz! Panda Cmin 90 bpm @crushed_keyz"
  "25. cKz! CoSmoS Dmin 132 bpm @crushed_keyz"
  "cKz! Scenic Cmin 110 bpm @crushed_keyz"
  "24. cKz! JuPitER Cmin 120 bpm @crushed_keyz"
  "9. cKz! Steppa Fmin 110 bpm @crushed_keyz"
  "cKz! Mars C#min 125 bpm @crushed_keyz"
  "11. cKz! CANOPY Amin 125 bpm @crushed_keyz"
)

slug() { echo "$1" | tr -cd '[:alnum:] ' | tr ' ' '_' | cut -c1-34; }

for p in "${PROMPTS[@]}"; do
  s=$(slug "$p")
  echo ">>> newPT | $p"
  uv run stable-audio --model medium --lora-ckpt-path "$NEW_LORA" \
    --prompt "$p" --duration "$DURATION" --steps 8 --seed "$SEED" \
    --output "$OUT/newPT__${s}__seed${SEED}.wav"
  echo ">>> oldBASE | $p"
  uv run stable-audio --model medium-base --lora-ckpt-path "$OLD_LORA" \
    --prompt "$p" --duration "$DURATION" --steps 50 --cfg-scale 3 --seed "$SEED" \
    --output "$OUT/oldBASE__${s}__seed${SEED}.wav"
done

echo ""
echo "Rate them (timbre + melody):"
echo "  python sweep_rater/rate_server.py --folder $OUT"
