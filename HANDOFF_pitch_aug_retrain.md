# cKz Pitch-Augmentation Retrain — Handoff
_Context dump to continue in a fresh chat. Written 2026-06-22._

## The goal (unchanged, north star)
Get a **novel 4★+ Crushed Keyz (cKz) melody every time, with variety** — automated,
not hand-cherry-picked. Model = Stable Audio 3 LoRA on an M4 Max Mac (MPS).

## Why we're doing a pitch-augmentation retrain
The current best LoRA (`lora_out/ckz_melody_capped/epoch=54-step=2500.ckpt`, "eff-2500")
**MEMORIZED its training songs**. With only 46 loops, the high-quality outputs are
reproductions of ~5 training songs (ACTiVE Cmin etc.), not novel melodies. Every
inference lever was exhausted and found flat (cfg=3, steps=50, strength=1.0 required,
time-gate dead, init_audio hurts quality though it's a novelty source). Conclusion:
the wall is **memorization from too little data**. Two fixes, both in progress:
1. **More real data** — Mike dropped 22 new loops (46 → 68).
2. **Pitch augmentation** — transpose each loop ±2 semitones (formant-preserving),
   rewriting the caption key to match. Multiplies 68 → ~204 clips across more keys,
   breaking per-song memorization and forcing the model to learn the STYLE.

## What was built this session (all in `scripts/`)
- **`augment_pitch.py`** — offline pitch-aug. Reads `data/crushed_keyz_lora`, for each
  loop writes original + −2 + +2 (formant-preserving via the `rubberband` CLI, R3
  engine), and **transposes the caption key** (Cmin→Dmin for +2, etc.; chromatic,
  sharp spelling). BPM unchanged (pitch-shift preserves tempo). Output → `data/ckz_aug`
  as 32-bit float WAV + `.txt` captions. Has **resume** (skips already-written) and
  skips captions with no matching audio (e.g. `MANIFEST.txt`).
- **`detect_key.py`** — key QA. librosa Krumhansl-Schmuckler (auto-uses `essentia`
  `edma` profile if installed). **Assumes minor** (cKz is ~all minor), compares only the
  **tonic** pitch class, collapses relative-major detections, and reports mismatches as
  `OFF ±Nst`. Writes `data/key_check.csv`. (~70–85% accurate — trust ears over it.)
- **`fix_keys.py`** — applies ear-confirmed key corrections: renames source audio+`.txt`+
  caption content and clears the loop's stale `data/ckz_aug` copies so re-augment rebuilds
  them. Idempotent; handles INSERT for keyless names (ICEBERG). CORR map holds all 16
  corrections (8 round-1 originals + 7 round-2 new + ICEBERG).
- **`import_new_loops.py`** — copies newly-dropped loops (by mtime) from
  `/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops/Crushed_keyz_melody_loops` into
  `data/crushed_keyz_lora`, creates `.txt` captions (= filename stem), normalizes
  short-form keys (`F#m`→`F#min`).

## Key corrections made (ear-confirmed by Mike — detector was right)
Round 1 (originals): DreamDoor Emin→**Bmin**, Rent Free Cmin→**Gmin**, DiMELO Amin→**Emin**,
El Yunque F#min→**C#min**, Pressure F#min→**C#min**, dAYS OFF Gmin→**Dmin**,
Steppa Fmin→**Emin**, StarLight Amin→**G#min**.
Round 2 (new loops): Assault Bmin→**Emin**, MEDUSA G#min→**C#min**, Come Back Bmin→**F#min**,
BAMBOO Amin→**Dmin**, BadBunny Amin→**Dmin**, WAVY Cmin→**Gmin**, DMT G#min→**C#min**,
ICEBERG (no key)→insert **F#min**.
**Panda stays Cmin** — the one confirmed FALSE flag (detector picked the dominant).
NOTE: cKz labels were frequently off by a 4th/5th (IV/V); the detector flags these and
Mike's ears confirmed them real. Always verify by ear, but the detector earned trust here.

## CURRENT STATE
- `data/crushed_keyz_lora/` = **68 source loops** + `.txt` captions (46 original + 22 new
  imported). `MANIFEST.txt` has no audio (ignored). All keys now QA'd; round-1 fixes
  already applied + verified OK by the scan.
- `data/ckz_aug/` = augmented set, **partially built** — round-1 corrected trios were
  cleared (pending rebuild); 22 new + ICEBERG not yet augmented.
- Round-2 fixes (7 new + ICEBERG) are coded in `fix_keys.py` but **not yet applied**.

## IMMEDIATE NEXT STEP (run on the Mac)
```
cd ~/stable-audio-3 && uv run python scripts/fix_keys.py && uv run python scripts/augment_pitch.py
```
Applies the round-2 key fixes, then builds the full augmented set: 68 loops × 3 =
~**204 clips** in `data/ckz_aug` (rebuilds cleared trios, builds 22 new + ICEBERG, skips
untouched). After it runs: spot-check a couple ±2 pairs by ear (timbre must survive).

## THEN: train v2 LoRA on the augmented set
```
cd ~/stable-audio-3 && DATA_DIR=./data/ckz_aug SAVE_DIR=./lora_out/ckz_aug_v1 bash scripts/train_ckz_melody_capped.sh
```
Known-good recipe: `medium-base`, dora-rows rank 16, `--exclude seconds_total`, capped
crop (duration 30 / max_crop_offset 15 = melody-only), ckpt every 500. Stop ~2000–2500.
Watch for: dense timbre on every seed, no single-instrument/muffled outputs.

## THEN: eval v2 vs baseline + LISTEN for novelty
```
cd ~/stable-audio-3 && uv run python scripts/eval_checkpoint.py --ckpt ./lora_out/ckz_aug_v1/<ckpt>.ckpt
```
Baseline to beat (eff-2500): predicted **T2.62 / M2.57, keeper 3%**, density 0.973.
The scorer measures QUALITY, not novelty — so ALSO listen (use `sweep_rater/rate_server.py`
and/or `scripts/checkpoint_novelty.py`) to confirm outputs are genuinely NEW melodies and
not the old memorized songs, with cKz timbre intact. Decision: if timbre held + more
novel → adopt v2. If pitch-shift degraded timbre → reduce to ±1 st or revert.

## THEN: production loop (the realized end-goal tool)
Over-generate → `sweep_rater/scorer/pick_best.py` ranks by predicted timbre+melody (with
density guard) → keep top 5–10% (validated 50–70% keepers vs 12% base). "Novel 4★ every
time" = augmented model (raises novelty floor) + auto-picker (selects quality).

## Gotchas / conventions (preserve)
- Pitch-shift: `rubberband` CLI (formant-preserving, R3), **±2 st only** (timbre risk
  above). Installed at `/opt/homebrew/bin/rubberband`.
- `pyrubberband`/`librosa` go in the **.venv** via `uv pip install` (not anaconda base).
- Float WAV (format 3) is VALID; Python stdlib `wave` can't read it — use soundfile.
- Terminal: prefix `cd ~/stable-audio-3 &&`; NO inline `#` comments (zsh chokes).
- GitHub: all work → PRIVATE repo `stable-audio-3-private` (remote `private`). NEVER push
  the public fork `mmcgovern574/stable-audio-3`.
- Memory files (auto-loaded): ckz_lora_eval_results, ckz_lora_training_setup,
  ckz_lora_init_audio_findings, ckz_autopicker, ckz_timegate_lora, full-terminal-commands.
- Pipeline is parameterized for any producer (see `PRODUCER_PIPELINE.md`,
  `EXPERIMENT_MATRIX.md`, `ANALYSIS_muffling_and_improvements.md`).
