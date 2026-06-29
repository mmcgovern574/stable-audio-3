# Onboard a producer → ship a catalog

End-to-end pipeline to clone **any** producer's melody style into Stable Audio 3 and
produce a rated catalog of novel loops — the generalized version of how the live
Crushed Keyz (cKz) catalog on soundsauce.ai was made.

**The producer only provides one thing: a folder of correctly-named melody loops.**
Everything else (style trigger, captions, augmentation, training, generation, catalog
assembly) is automated. Two human checkpoints remain, by design — you judge quality
**by ear** when picking the trained model and when keeping catalog clips (the project's
validated "pure ears" finding; auto-scoring was tried and dropped).

This reproduces the exact recipe that generated the current soundsauce.ai melodies:
**aug-8000-style LoRA on Stable Audio 3 `medium-base`, generated via Discovery mode
(foreign-init at σ0.75).**

---

## 0. What the producer hands you: the naming spec

Drop the producer's melody loops (≈40–50 is plenty) into one folder, each named:

```
[NN.] <STYLE>! <Title> <Key><min|maj> <bpm> bpm @<trigger>.<wav|mp3|flac>
```

Examples:

```
01. cKz! Mirage Dmin 140 bpm @crushed_keyz.wav
7. cKz! ICE D#min 135 bpm @crushed_keyz.wav
```

Rules:
- The optional leading `NN.` index is ignored.
- `<STYLE>!` is one token ending in `!` — the **style trigger** (e.g. `cKz!`). Use the
  same one on every file.
- `<Key>` = a letter `A`–`G`, optional `#`, immediately followed by `min` or `maj`
  (`Dmin`, `A#min`).
- `<bpm>` = a 2–3 digit number followed by the literal word `bpm`.
- `@<trigger>` = the producer's unique tag, the **last** token. Same on every file.
- Everything between `<STYLE>!` and `<Key>` is the **title** (can be multiple words).

The style prefix and `@trigger` become the model's style tokens, so they must be
**identical across the whole folder**. The pipeline reads them straight off the
filenames — you don't configure them anywhere.

> Pick a trigger that's a rare token (e.g. `@crushed_keyz`, not `@piano`) so it doesn't
> collide with anything the base model already knows.

---

## 1. One command per stage

### Stage 1 — train (prep → augment → train), stops at **Gate A**

```bash
cd ~/stable-audio-3 && bash scripts/onboard_producer.sh \
  --src "/path/to/ProducerMelodies" --name myproducer
```

This validates names, stages captioned copies, pitch-augments ±2 semitones (×3 data),
trains a fresh LoRA from `medium-base` (capped-crop recipe, ~3000 steps, checkpoint
every 500), then **pauses** and prints how to pick the best checkpoint.

### Gate A — pick the best checkpoint by ear

Generate a small compare batch on a few candidate checkpoints and rate them (never pick
from the training demos — they're single-seed and misleading). The driver prints ready
commands; the shape is:

```bash
cd ~/stable-audio-3 && uv run python scripts/ckz_factory.py \
  --ckpt lora_out/myproducer/epoch=NN-step=2500.ckpt \
  --style-prefix 'myStyle!' --trigger '@mytrigger' \
  --out sweep_rater/campaigns/myproducer_pick_2500

cd ~/stable-audio-3 && uv run python sweep_rater/rate_server.py \
  --folder sweep_rater/campaigns/myproducer_pick_2500 --port 8799
```

Audition, rate timbre/melody/novelty, and choose the checkpoint with the best **novel**
output. (For a ~150-clip augmented set the sweet spot is usually in the 2000–3000 range;
confirm by ear.)

### Stage 2 — gen (Discovery catalog batch), stops at **Gate B**

```bash
cd ~/stable-audio-3 && bash scripts/onboard_producer.sh \
  --name myproducer --stage gen \
  --ckpt lora_out/myproducer/epoch=NN-step=2500.ckpt
```

Runs Discovery mode (foreign-init σ0.75 + tritone + minor-only + dense-init + shuffle)
to over-generate novel loops into a campaign folder, then **pauses**.

### Gate B — rate the batch, keep the winners

```bash
cd ~/stable-audio-3 && uv run python sweep_rater/rate_server.py \
  --folder sweep_rater/campaigns/myproducer_catalog --port 8799
```

Rate everything; anything averaging ≥4★ across the three axes ships.

### Stage 3 — catalog (filter → MP3 → manifest)

```bash
cd ~/stable-audio-3 && bash scripts/onboard_producer.sh \
  --name myproducer --stage catalog
```

Filters to ≥4★, converts keepers to MP3, and writes:

```
catalog_build/myproducer/
  catalog.json        # same schema the Supabase uploader already consumes
  catalog_meta.json   # producer + model + recipe provenance for this batch
  *.mp3
```

That folder is the shippable catalog — feed `catalog.json` to the existing
`upload_catalog.py` / Supabase flow.

---

## 2. Knobs (env vars, all optional)

| Var | Default | What |
|---|---|---|
| `STEPS` | `3000` | training steps (fresh LoRA) |
| `CKPT_EVERY` | `500` | checkpoint/demo interval |
| `SIGMA` | `0.75` | Discovery init-noise level (the validated sweet spot) |
| `SEEDS` | `7,123` | generation seeds |
| `N_LOOPS` | `24` | foreign init loops sampled per Discovery run |
| `GEN_FLAGS` | `--tritone --minor-only --dense-init --shuffle` | Discovery flags |
| `MIN_RATING` | `4` | catalog keeper threshold (mean of 3 axes) |
| `PRODUCER_NAME` / `MODEL_NAME` | derived | provenance labels |

Example: `STEPS=2500 N_LOOPS=40 bash scripts/onboard_producer.sh --src ... --name ...`

---

## 3. Shared infrastructure (set up once, reused for every producer)

- **Foreign init library** — Discovery mode seeds melodies from a big folder of *foreign*
  loops (the cKz build used Cymatics packs). Defaults live in `scripts/init_sweep.py`
  (`LOOPS_ROOT` / `DEFAULT_DIRS`); override with `--loops-root` / `--init-dirs`. This pool
  is producer-agnostic — the LoRA repaints its own timbre over the foreign melody.
- **Rubberband** — formant-preserving pitch shift for augmentation: `brew install rubberband`.
- **ffmpeg** — wav→mp3 for the catalog: `brew install ffmpeg`.
- **`uv sync --extra lora`** — the driver runs this before the heavy stages.

---

## 4. What each piece does

| File | Role |
|---|---|
| `scripts/onboard_producer.sh` | the orchestrator (this runbook in code) |
| `scripts/prep_captions.py` | validate naming spec, read style/trigger, stage captioned copies |
| `scripts/augment_pitch.py` | pitch ±2 semitones, transpose caption keys → 3× dataset |
| `scripts/train_lora.py` | fine-tune the LoRA (capped-crop, `medium-base`) |
| `scripts/ckz_factory.py` | t2m engine (checkpoint compare + alt novelty engine) |
| `scripts/init_sweep.py` | Discovery engine (foreign-init σ0.75) — the catalog generator |
| `sweep_rater/rate_server.py` | the 3-axis by-ear rater |
| `scripts/build_catalog.py` | filter ≥N★, wav→mp3, emit `catalog.json` + provenance |

**Why it's shaped this way:** timbre lives in the LoRA (run at full strength); melody is the
seed/init lottery; the two can't be decoupled at inference — so the path to "good every
time" is over-generate + pick. Pitch augmentation breaks song-memorization so the model
learns *style*, not specific loops. See `PRODUCER_PIPELINE.md` and the strategy memory for
the deeper rationale.
```
