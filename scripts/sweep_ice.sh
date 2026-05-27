#!/usr/bin/env bash
# Parameter sweep on a known-good training prompt to find the best settings.
# Holds prompt and duration fixed; sweeps one parameter at a time so you can
# isolate what each knob does to the output.
#
# Defaults (held constant when not being swept):
#   cfg-scale       = 7
#   steps           = 50
#   seed            = 42
#   lora-strength   = 1.0 (the model default; CLI omits the flag entirely)
#
# Usage:    bash scripts/sweep_ice.sh
# Output:   ./sweep_out/<axis>_<value>_cfg{N}_steps{N}_seed{N}_strength{V}.wav
set -euo pipefail
cd "$(dirname "$0")/.."

LORA_CKPT="${LORA_CKPT:-./lora_out/ckz_5000_dora/epoch=108-step=5000.ckpt}"
MODEL="${MODEL:-medium-base}"
PROMPT="${PROMPT:-2. cKz! ICE Dmin 135 bpm @crushed_keyz}"
DURATION="${DURATION:-20}"
OUT_DIR="${OUT_DIR:-./sweep_out}"
mkdir -p "$OUT_DIR"

if [[ ! -f "$LORA_CKPT" ]]; then
  echo "Missing checkpoint: $LORA_CKPT"
  exit 1
fi

# Defaults — overridden one at a time per sweep
DEF_CFG=7
DEF_STEPS=50
DEF_SEED=42
DEF_STRENGTH=1.0

# Generic runner.
run() {
  local axis="$1" value="$2" cfg="$3" steps="$4" seed="$5" strength="$6"
  local fn="${axis}_${value}_cfg${cfg}_steps${steps}_seed${seed}_strength${strength}.wav"
  local out="$OUT_DIR/$fn"
  if [[ -f "$out" ]]; then
    echo "  skip (exists): $fn"
    return
  fi
  echo "  $axis=$value  → $fn"

  local lora_args=(--lora-ckpt-path "$LORA_CKPT")
  # Pass --lora-strength only when it's not the default 1.0, to keep behavior
  # explicit. (CLI default is None which the model treats as 1.0.)
  if [[ "$strength" != "1.0" ]]; then
    lora_args+=(--lora-strength "$strength")
  fi

  uv run stable-audio \
    --model "$MODEL" \
    "${lora_args[@]}" \
    -p "$PROMPT" \
    --duration "$DURATION" \
    --steps "$steps" \
    --cfg-scale "$cfg" \
    --seed "$seed" \
    --output "$out"
}

echo "Sweep prompt: $PROMPT"
echo "Fixed: duration=${DURATION}s, output=$OUT_DIR/"
echo "Defaults: cfg=$DEF_CFG steps=$DEF_STEPS seed=$DEF_SEED strength=$DEF_STRENGTH"
echo

# ─── Axis 1: cfg-scale ────────────────────────────────────────────────────
# How literally the model follows the prompt. Low = looser/more creative,
# high = more literal. Past ~10 you usually get artifacts.
echo "── Axis: cfg-scale (1, 2, 4, 7, 10) ──"
for cfg in 1 2 4 7 10; do
  run "cfg" "$cfg" "$cfg" "$DEF_STEPS" "$DEF_SEED" "$DEF_STRENGTH"
done
echo

# ─── Axis 2: diffusion steps ──────────────────────────────────────────────
# More steps = more detail / better quality but slower. SA3 is designed for
# ~50; below that quality drops fast, above ~80 there's diminishing return.
echo "── Axis: steps (8, 16, 32, 50, 80) ──"
for steps in 8 16 32 50 80; do
  run "steps" "$steps" "$DEF_CFG" "$steps" "$DEF_SEED" "$DEF_STRENGTH"
done
echo

# ─── Axis 3: lora-strength ────────────────────────────────────────────────
# Scales the LoRA delta applied on top of base weights. <1 = less cKz flavor,
# >1 = over-applied (often distorts). This is the cleanest "how much LoRA"
# dial at inference time.
echo "── Axis: lora-strength (0.25, 0.5, 0.75, 1.0, 1.25) ──"
for s in 0.25 0.5 0.75 1.0 1.25; do
  run "strength" "$s" "$DEF_CFG" "$DEF_STEPS" "$DEF_SEED" "$s"
done
echo

# ─── Axis 4: seed (same params, different noise) ──────────────────────────
# Shows variety — how different valid outputs can be from the same prompt
# under identical settings. Helps you judge whether the model is producing
# meaningfully varied takes or just one frozen rendition.
echo "── Axis: seed (0, 1, 42, 123, 999) ──"
for seed in 0 1 42 123 999; do
  run "seed" "$seed" "$DEF_CFG" "$DEF_STEPS" "$seed" "$DEF_STRENGTH"
done
echo

echo "Done."
ls -lh "$OUT_DIR/" | sort
