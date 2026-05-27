#!/usr/bin/env bash
# Batch-generate Crushed Keyz LoRA outputs against the exact training-set
# captions. Prompts are read directly from data/crushed_keyz_lora/*.txt so
# the script stays in sync with the training data — no hard-coded list.
#
# Filenames encode the full prompt text + config so each file is
# self-describing and different CFG/STEPS/DURATION runs co-exist in
# the same output directory.
#
# Usage:   bash scripts/gen_ckz_batch.sh
# Output:  ./gen_out/<sanitized prompt>_cfg{N}_steps{N}_dur{N}s.wav
set -euo pipefail
cd "$(dirname "$0")/.."

LORA_CKPT="${LORA_CKPT:-./lora_out/ckz_2000/epoch=43-step=2000.ckpt}"
MODEL="${MODEL:-medium-base}"
DATA_DIR="${DATA_DIR:-./data/crushed_keyz_lora}"
DURATION="${DURATION:-30}"
STEPS="${STEPS:-8}"
CFG="${CFG:-7}"
OUT_DIR="${OUT_DIR:-./gen_out}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$LORA_CKPT" ]]; then
  echo "Missing checkpoint: $LORA_CKPT"
  echo "Set LORA_CKPT=... or check ./lora_out/"
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "Missing data dir: $DATA_DIR"
  echo "Set DATA_DIR=... or check the training data was prepared."
  exit 1
fi

# Build a filesystem-safe filename from the prompt + config.
#   - Replace spaces with underscores
#   - Strip path separators and shell-unfriendly chars: / \ : | ? * < > " '
#   - Collapse repeated underscores
#   - Cap prompt portion at 180 chars (leaves room for the config suffix)
#   - Append config suffix
sanitize_filename() {
  local prompt="$1"
  local cleaned
  cleaned=$(printf '%s' "$prompt" \
    | tr ' ' '_' \
    | tr -d '/\\:|?*<>"'"'" \
    | tr -s '_')
  cleaned="${cleaned:0:180}"
  printf '%s_cfg%s_steps%s_dur%ss.wav' \
    "$cleaned" "$CFG" "$STEPS" "$DURATION"
}

# ───────────────────────────────────────────────────────────────────────────
# Load all training prompts (one per .txt file in DATA_DIR), sorted so
# numbered tracks group in order and unnumbered ones follow.
# ───────────────────────────────────────────────────────────────────────────
PROMPTS=()
while IFS= read -r -d '' txt_file; do
  # The caption file contains exactly one line — the training caption.
  prompt=$(head -n1 "$txt_file" | tr -d '\r')
  # Skip blank lines just in case.
  [[ -n "$prompt" ]] && PROMPTS+=("$prompt")
done < <(find "$DATA_DIR" -maxdepth 1 -type f -name '*.txt' -print0 | sort -z)

total=${#PROMPTS[@]}
if [[ "$total" -eq 0 ]]; then
  echo "No .txt prompt files found in $DATA_DIR"
  exit 1
fi

echo "Found $total training prompts in $DATA_DIR"
echo "Checkpoint: $LORA_CKPT"
echo "Output dir: $OUT_DIR"
echo "Config: cfg=$CFG steps=$STEPS duration=${DURATION}s"
echo

# ───────────────────────────────────────────────────────────────────────────
# Run them all
# ───────────────────────────────────────────────────────────────────────────
i=0
for prompt in "${PROMPTS[@]}"; do
  i=$((i + 1))
  filename=$(sanitize_filename "$prompt")
  out="$OUT_DIR/$filename"

  if [[ -f "$out" ]]; then
    echo "[$i/$total] skip (exists): $filename"
    continue
  fi

  echo "[$i/$total] $prompt"
  echo "          → $filename"
  uv run stable-audio \
    --model "$MODEL" \
    --lora-ckpt-path "$LORA_CKPT" \
    -p "$prompt" \
    --duration "$DURATION" --steps "$STEPS" --cfg "$CFG" \
    --out "$out"
done

echo
echo "Done. Outputs in $OUT_DIR/"
ls -lh "$OUT_DIR/"
