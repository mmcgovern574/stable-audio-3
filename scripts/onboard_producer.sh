#!/usr/bin/env bash
# =============================================================================
# onboard_producer.sh  —  clone ANY producer's melody style into Stable Audio 3
# and ship a rated catalog. Point it at a folder of named loops; it runs the
# whole pipeline, pausing at the two "judge by ear" gates.
#
#   STAGE FLOW          gate
#   prep -> augment -> train  ──(A: pick best checkpoint)
#                       gen    ──(B: rate keepers)
#                       catalog -> catalog_build/<slug>/  (mp3s + catalog.json)
#
# USAGE
#   # 1) up to GATE A (prep + augment + train):
#   bash scripts/onboard_producer.sh --src "/path/to/ProducerMelodies" --name myproducer
#
#   # 2) after you pick a checkpoint by ear, up to GATE B (generate catalog batch):
#   bash scripts/onboard_producer.sh --name myproducer --stage gen \
#        --ckpt lora_out/myproducer/epoch=NN-step=NNNN.ckpt
#
#   # 3) after you rate the batch, build the shippable catalog:
#   bash scripts/onboard_producer.sh --name myproducer --stage catalog
#
# The producer ONLY has to provide a folder of correctly-named loops (see
# PRODUCER_ONBOARDING.md for the naming spec). Style prefix + @trigger are read
# from the filenames automatically.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

# ---- args ----
SRC="" ; NAME="" ; STAGE="train" ; CKPT="" ; RUN="uv run python"
# tunables (env-overridable)
STEPS="${STEPS:-3000}"; CKPT_EVERY="${CKPT_EVERY:-500}"
SIGMA="${SIGMA:-0.75}"; SEEDS="${SEEDS:-7,123}"; N_LOOPS="${N_LOOPS:-24}"
GEN_FLAGS="${GEN_FLAGS:---tritone --minor-only --dense-init --shuffle}"
MIN_RATING="${MIN_RATING:-4}"
PRODUCER_NAME="${PRODUCER_NAME:-}"; MODEL_NAME="${MODEL_NAME:-}"
while [[ $# -gt 0 ]]; do case "$1" in
  --src) SRC="$2"; shift 2;;
  --name) NAME="$2"; shift 2;;
  --stage) STAGE="$2"; shift 2;;
  --ckpt) CKPT="$2"; shift 2;;
  --producer-name) PRODUCER_NAME="$2"; shift 2;;
  --model-name) MODEL_NAME="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 2;;
esac; done

[[ -n "$NAME" ]] || { echo "ERROR: --name <slug> is required"; exit 2; }
SLUG="$(echo "$NAME" | tr '[:upper:] ' '[:lower:]-' | tr -cd 'a-z0-9-_')"

STAGE_DIR="data/${SLUG}_src"
AUG_DIR="data/${SLUG}_aug"
LORA_DIR="lora_out/${SLUG}"
CAMP_DIR="sweep_rater/campaigns/${SLUG}_catalog"
OUT_DIR="catalog_build/${SLUG}"
CONF="producers/${SLUG}.conf"
mkdir -p producers

hr(){ printf '%s\n' "──────────────────────────────────────────────────────────────"; }

# ---- ensure deps for the heavy stages ----
sync_env(){ uv sync --extra lora >/dev/null 2>&1 || true; }

# =============================================================================
# STAGE: prep  (validate naming, stage captioned copies, capture style/trigger)
# =============================================================================
do_prep(){
  [[ -n "$SRC" ]] || { echo "ERROR: --src <folder> required for prep"; exit 2; }
  hr; echo "STAGE prep — validating + staging $SRC"; hr
  local out; out="$($RUN scripts/prep_captions.py --src "$SRC" --stage "$STAGE_DIR")" || {
    echo "$out"; echo "PREP FAILED — fix the file names and retry."; exit 2; }
  echo "$out"
  local style trig nloops demo
  style="$(grep '^PREP_STYLE_PREFIX=' <<<"$out" | cut -d= -f2-)"
  trig="$(grep  '^PREP_TRIGGER='      <<<"$out" | cut -d= -f2-)"
  nloops="$(grep '^PREP_N_LOOPS='     <<<"$out" | cut -d= -f2-)"
  demo="$(grep  '^PREP_DEMO_PROMPTS=' <<<"$out" | cut -d= -f2-)"
  : "${PRODUCER_NAME:=${trig#@}}"
  : "${MODEL_NAME:=${SLUG}-aug}"
  cat > "$CONF" <<EOF
SLUG=$SLUG
STYLE_PREFIX=$style
TRIGGER=$trig
N_LOOPS=$nloops
STAGE_DIR=$STAGE_DIR
AUG_DIR=$AUG_DIR
LORA_DIR=$LORA_DIR
CAMP_DIR=$CAMP_DIR
OUT_DIR=$OUT_DIR
DEMO_PROMPTS=$demo
PRODUCER_NAME=$PRODUCER_NAME
MODEL_NAME=$MODEL_NAME
EOF
  echo "wrote config -> $CONF"
}

# =============================================================================
# STAGE: augment  (pitch ±2 semitones -> 3x dataset)
# =============================================================================
do_augment(){
  source "$CONF"
  hr; echo "STAGE augment — pitch ±2 -> $AUG_DIR"; hr
  sync_env
  $RUN scripts/augment_pitch.py --src "$STAGE_DIR" --out "$AUG_DIR"
}

# =============================================================================
# STAGE: train  (fresh LoRA from medium-base, capped crop) -> GATE A
# =============================================================================
do_train(){
  source "$CONF"
  hr; echo "STAGE train — fresh LoRA on $AUG_DIR ($STEPS steps) -> $LORA_DIR"; hr
  sync_env
  $RUN scripts/train_lora.py \
    --model medium-base --data_dir "$AUG_DIR" \
    --duration 30 --max_crop_offset_sec 15 \
    --rank 16 --adapter_type dora-rows --exclude seconds_total \
    --steps "$STEPS" --batch_size 1 \
    --checkpoint_every "$CKPT_EVERY" --demo_every "$CKPT_EVERY" \
    --demo_prompts_file "$DEMO_PROMPTS" \
    --demo_duration 12 --num_inpaint_demos 0 --demo_cfg_scales 3 \
    --save_dir "$LORA_DIR" --name "$SLUG"
  hr; echo "GATE A — pick the best checkpoint BY EAR (never from demos)."; hr
  cat <<EOF
Checkpoints written to: $LORA_DIR
1) Generate a quick compare batch for a few candidate checkpoints, e.g.:
     for c in $LORA_DIR/epoch=*-step=2000.ckpt $LORA_DIR/epoch=*-step=2500.ckpt $LORA_DIR/epoch=*-step=3000.ckpt; do
       $RUN scripts/ckz_factory.py --ckpt "\$c" \\
         --style-prefix "$STYLE_PREFIX" --trigger "$TRIGGER" \\
         --out sweep_rater/campaigns/${SLUG}_pick_\$(basename "\$c" .ckpt)
     done
2) Rate them (3-axis player):
     $RUN sweep_rater/rate_server.py --folder sweep_rater/campaigns/${SLUG}_pick_<ckpt> --port 8799
3) Then continue with the winner:
     bash scripts/onboard_producer.sh --name $SLUG --stage gen --ckpt <chosen.ckpt>
EOF
}

# =============================================================================
# STAGE: gen  (Discovery-mode catalog batch on the chosen ckpt) -> GATE B
# =============================================================================
do_gen(){
  source "$CONF"
  [[ -n "$CKPT" ]] || { echo "ERROR: --ckpt <chosen checkpoint> required for stage gen"; exit 2; }
  hr; echo "STAGE gen — Discovery σ$SIGMA on $CKPT -> $CAMP_DIR"; hr
  sync_env
  $RUN scripts/init_sweep.py \
    --ckpt "$CKPT" \
    --style-prefix "$STYLE_PREFIX" --trigger "$TRIGGER" \
    --rand-titles --no-baseline $GEN_FLAGS \
    --sigmas "$SIGMA" --seeds "$SEEDS" --n-loops "$N_LOOPS" \
    --out "$CAMP_DIR"
  # remember the ckpt for provenance in the catalog stage
  grep -q '^GEN_CKPT=' "$CONF" && sed -i.bak "/^GEN_CKPT=/d" "$CONF"
  echo "GEN_CKPT=$CKPT" >> "$CONF"
  hr; echo "GATE B — rate the batch, keep the ≥${MIN_RATING}★ clips:"; hr
  echo "     $RUN sweep_rater/rate_server.py --folder $CAMP_DIR --port 8799"
  echo "Then build the shippable catalog:"
  echo "     bash scripts/onboard_producer.sh --name $SLUG --stage catalog"
}

# =============================================================================
# STAGE: catalog  (filter ≥N★ -> mp3 + catalog.json with provenance)
# =============================================================================
do_catalog(){
  source "$CONF"
  local ckpt="${GEN_CKPT:-$CKPT}"
  hr; echo "STAGE catalog — filter ≥${MIN_RATING}★ -> $OUT_DIR"; hr
  $RUN scripts/build_catalog.py \
    --campaign "$CAMP_DIR" --out "$OUT_DIR" --min-rating "$MIN_RATING" \
    --producer-name "$PRODUCER_NAME" --producer-handle "$TRIGGER" \
    --model-name "$MODEL_NAME" --lora-ckpt "$ckpt" --init-noise "$SIGMA"
  hr; echo "DONE — catalog in $OUT_DIR (catalog.json + catalog_meta.json + mp3s)."; hr
}

# ---- dispatch ----
case "$STAGE" in
  prep)    do_prep ;;
  augment) [[ -f "$CONF" ]] || do_prep; do_augment ;;
  train)   do_prep; do_augment; do_train ;;        # default: run through to GATE A
  gen)     do_gen ;;
  catalog) do_catalog ;;
  *) echo "unknown --stage: $STAGE (prep|augment|train|gen|catalog)"; exit 2;;
esac
