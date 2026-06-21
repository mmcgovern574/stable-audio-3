# Producer-style LoRA → novel-loop pipeline (reproducible runbook)

How to clone a producer's *timbre* into Stable Audio 3 and mass-produce novel,
high-quality melody loops in that style — then auto-select the best. This is the
generalized version of the Crushed Keyz (cKz) build; swap the placeholders and
repeat for any producer.

**Per-artist things to swap** (everything else is fixed/validated):
- `<ARTIST_LOOPS>` — folder of the producer's melody loops to clone (the *timbre* source). ~40–50 loops is enough.
- `@trigger` — a unique tag put in every caption (e.g. `@crushed_keyz`); it becomes the style trigger.
- `<TITLES>` — vocabulary of title words for prompts (pull from the training filenames).
- `<INIT_LIBRARY>` — a big folder of diverse *foreign* melody loops (e.g. Cymatics) used as the *melody/novelty* source at generation.

---

## Mental model (why the pipeline is shaped this way)
- **Timbre = the LoRA** (a holistic transform; must run at full strength). **Melody = the seed / init loop** (a lottery). Inference can NOT decouple them — the SAME autoencoder is a *semantic-acoustic* latent that entangles harmony+timbre by design.
- **"Muffling" = mode-averaging**: weak/off-manifold conditioning makes the model hedge toward its low-variance mean → dull/quiet/HF-poor. Cure = stay in-distribution + full strength.
- **Quality is a lottery** (~10–15% of generations are keepers). So the path to "good every time" is **over-generate + auto-select**, not a magic setting.

---

## Step 1 — Prepare training data (captions = exact filenames)
Caption each `<ARTIST_LOOPS>/*.wav` with a sidecar `.txt` containing the **exact filename stem** (NOT attribute-style descriptions — that failed). Every caption must contain `@trigger`.
- Helper: `scripts/build_hiphop_lora_dataset.py` (use the `--raw-names` style).
- Filename format that works as a prompt later: `"{N}. {style}! {Title} {Key} {bpm} bpm @trigger"`.
- Missing `.txt` → the clip is silently dropped, so verify every loop has one.

## Step 2 — Train the LoRA (capped-crop, ~2500 steps)
```bash
cd ~/stable-audio-3 && DATA_DIR=./data/<ARTIST_LOOPS> SAVE_DIR=./lora_out/<artist>_capped \
  MAX_OFFSET=15 RANK=16 STEPS=3000 bash scripts/train_ckz_melody_capped.sh
```
Key settings (validated): `--model medium-base`, `--adapter_type dora-rows`, `--exclude seconds_total`, `--duration 30`, **`--max_crop_offset_sec 15`** (CAPPED crop — essential; random full crops sample trailing "step-outs" → single-instrument/muffled failures), lr 1e-4, batch 1, ckpt every 500.
- **Stop ~2500 steps** — later checkpoints overfit and score worse.
- Rank 16 is the baseline; try 32 if timbre is thin.

## Step 3 — Pick the checkpoint (seed-sweep, don't trust loss)
Generate a small seed sweep on a few exact-format prompts across checkpoints (2000/2500/3000), rate timbre+melody, pick the one with the **lowest floor** (fewest T≤2). Use `sweep_rater/sweep_matrix.py` + `rate_server.py`.

## Step 4 — The validated inference recipe (do NOT re-tune these)
`model medium-base` · `cfg 3` · `steps 50` · `sampler dpmpp` · LoRA `strength 1.0` (full, no gating) · prompts in **exact training format**, **well-covered keys**, real titles.

**Ruled out — don't waste time re-testing (all empirically dead):**
- high cfg or steps (flat 2–7 / 50≈100), LoRA strength <1.0 (muffles), time-gated LoRA (muffles), per-layer LoRA filter (no melody/timbre decouple), cfg_interval gating (muffles), apg/rescale (inert), post-trained `medium` model (worse than base), init for *melody preservation* (entangled), **stems as init** (too sparse/quiet), training past ~2500 (overfit), diluting training data with other producers (halves timbre).

## Step 5 — Two ways to make novel melodies (both keep timbre)
1. **Seed-hunt** (pure text-to-music, random noise): most reliable quality, fully novel. Best-rated runs historically.
2. **Init-audio** (foreign loop → re-skinned in the style): novelty comes from the loop; you can *aim* it. Rules: use **dense FULL loops** (not stems), `init_noise_level 0.6`, **match generation duration to the loop length** (else the tail goes silent), key-match the prompt to the loop. **Loop quality dominates** — a few loops are gold, most are filler; bias toward proven anchors after a first pass.

## Step 6 — Mass-generate the library
First curate the init loops into a manifest (once per init-library / loop pack):
```bash
cd ~/stable-audio-3 && uv run python scripts/curate_init_loops.py \
  --src "/path/to/<INIT_LIBRARY>" --out sweep_rater/<artist>_loops.json --per-key 4
```
`curate_init_loops.py` scans the folder, keeps DENSE full loops (sparse → sparse output),
parses key+bpm, and balances across keys. Then generate:
```bash
cd ~/stable-audio-3 && uv run python sweep_rater/build_library.py \
  --manifest sweep_rater/<artist>_loops.json \
  --trigger "@trigger" --style-prefix "<PREFIX>!" --titles-file <TITLES>.txt \
  --prompts-per-loop 4 --seeds 7,123,777 --out sweep_rater/campaigns/<artist>_lib
```
`build_library.py` builds key-matched `<PREFIX> {Title} {Key} {bpm} bpm @trigger` prompts,
generates at the recipe, **duration-matches each clip** (no silent tails), and writes
`library_index.csv` (clip → prompt → init loop). All artist-specific bits are flags now:
`--manifest`, `--trigger`, `--style-prefix`, `--titles-file`, `--lora-ckpt`, `--model`.
`--titles-file` is one title per line (or omit for the built-in flavor list).

## Step 7 — Auto-select the keepers (the scorer)
Needs a seed set of ratings to train (more = better; ~1600 cKz ratings gave Spearman 0.66 timbre / 0.55 melody, and **top-25 picks = 48% keepers vs 11.5% base**).
```bash
# build/refresh the feature cache from everything you've rated:
cd ~/stable-audio-3 && uv run python sweep_rater/scorer/extract_features.py --limit 5000
# rank a generated folder, copy the top-K to <folder>/picks/:
cd ~/stable-audio-3 && uv run python sweep_rater/scorer/pick_best.py --folder sweep_rater/campaigns/<artist>_lib --top 75
# browse:
cd ~/stable-audio-3 && python sweep_rater/rate_server.py --folder sweep_rater/campaigns/<artist>_lib/picks --port 8799
```
`pick_best.py` has a silence guardrail and writes `picks_ranked.csv`.

## Step 8 — The harvest loop (maximize novel keepers per hour)
1. Generate big batches, **biased toward proven anchor loops** (highest keeper rate).
2. Auto-pick; **listen down the ranked list** (not file order); stop when keepers thin.
3. **Rate as you go** → re-run `extract_features.py` → the scorer sharpens → trust deeper cuts next time.
Don't manually scan whole batches in file order (~11% hit rate); the picker concentrates the keepers at the top.

## Bootstrapping a brand-new artist (cold-start)
The scorer needs ratings. For artist #1 you rate by hand to seed it. For later artists, the *timbre* axis of the scorer is largely style-agnostic (it keys on brightness/dynamics/muffle), so it transfers as a first-pass timbre filter; melody scoring you re-seed with a few dozen ratings per new artist.

---

## Key files
- Training: `scripts/train_ckz_melody_capped.sh`, `scripts/train_lora.py`, `scripts/build_hiphop_lora_dataset.py`
- Init curation: `scripts/curate_init_loops.py` (loop folder → dense, key-balanced manifest)
- Generation: `sweep_rater/build_library.py` (fully parameterized: `--manifest/--trigger/--style-prefix/--titles-file/--lora-ckpt`), `sweep_rater/sweep_matrix.py`
- Selection: `sweep_rater/scorer/{extract_features,train_scorer,pick_best}.py`
- Rating UI: `sweep_rater/rate_server.py` + `rate.html`
- Records: `EXPERIMENT_MATRIX.md` (all sweeps), `ANALYSIS_muffling_and_improvements.md` (why), memory files `ckz_*`.
