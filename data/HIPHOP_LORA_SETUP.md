# Hip-hop melody-loop LoRA — dataset setup

## The gotcha you asked about

Captions are **not** taken from the folder path or the filename. The trainer
(`scripts/train_lora.py` → `caption_metadata_fn`) reads a sidecar **`.txt`** file
with the same stem sitting next to each audio file. If that `.txt` is missing,
the clip is **rejected and silently dropped**.

So a naive run on the whole library would have trained on the 46 Crushed Keyz
files again (the only ones with `.txt`), and every Cymatics loop would have been
discarded — with no error.

`scripts/build_hiphop_lora_dataset.py` fixes this by writing a `.txt` for every
full loop, with the musically useful fields pulled from **both** the folder
(mood/genre) and the filename (key, BPM).

## What got built

Run: `uv run python scripts/build_hiphop_lora_dataset.py --write`

| | count |
|---|---|
| Full loops captioned | **623** (46 Crushed Keyz + 577 Cymatics) |
| Stems excluded | 1,356 |
| Reverse clips excluded | 18 |

Outputs:
- a `.txt` caption beside each of the 623 loops
- `filelist.txt` at the library root → trainer loads only these 623, never the stems
- `data/hiphop_lora_captions.csv` → review every caption

**Why exclude stems:** ~1,390 of the ~2,000 files are isolated layers
("Loop Stems"/"Stems") or reversed clips. Training on stems teaches the model to
emit single layers, not full melodic loops. Re-include with `--include-stems` /
`--include-reverse` if you ever want them.

## Caption design (built to fight memorization)

Captions use only **shared attributes** — never a unique track name, because a
per-file identifier is exactly what lets the model memorize caption→clip.

- Crushed Keyz: `crushed keyz, hip hop melody loop, <key>, <bpm> bpm`
- Everything else: `<mood/genre>, hip hop melody loop, <key>, <bpm> bpm`

The `crushed keyz` trigger appears **only** on the CKz files, so that token stays
bound to the CKz timbre while melodic variety is learned from the whole set. At
inference, `crushed keyz, dark, hip hop melody loop, F minor, 140 bpm` should give
the CKz sound carrying a dark F-minor melody.

## Why this should fix what you saw

Your model reproduced training melodies because it saw each of 46 files ~43 times
(2000 steps ÷ 46). With 623 files at ~4 epochs each file is seen ~4 times — too
few to memorize, enough to learn the genre's shared melodic structure.

## Train

```bash
bash scripts/train_hiphop_lora.sh
```

Defaults: 2,500 steps (~4 epochs), duration 12s, rank 16, checkpoints every 500.

**Evaluate on NOVEL prompts**, not training titles. Walk the checkpoints and keep
the latest one whose output still sounds original; when generations start matching
specific training loops, that checkpoint is over-fit — step back one. If it still
copies even early, drop `RANK=8`.

## Head-to-head eval vs the old model

`sweep_rater/configs/hiphop_winners_eval.json` re-runs your two top-ranked
prompts on the NEW model, in the new caption vocabulary, with the SAME axes as
the old runs (cfg 3, strength 1.0, steps 50, 20 seeds, duration 20) so the 1-5
ratings line up directly.

Old-model baselines to beat (mean rating):
- ICE / D minor 135 bpm → **3.35**
- Midnight / B minor 110 bpm → **3.08**

After training:
```bash
# 1. pick a checkpoint from the demos, then point the config at it:
#    edit "lora_ckpt" in sweep_rater/configs/hiphop_winners_eval.json
# 2. generate the 40 clips (2 prompts x 20 seeds)
uv run python sweep_rater/sweep_matrix.py --config sweep_rater/configs/hiphop_winners_eval.json
# 3. rate them and compare means to the baselines above
python sweep_rater/rate_server.py --folder sweep_rater/campaigns/hiphop_winners_eval
python sweep_rater/analyze_ratings.py --folder sweep_rater/campaigns/hiphop_winners_eval
```

## Undo

The only changes to your library are added `.txt` files + `filelist.txt`. Remove with:
```bash
find "/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops" -name '*.txt' -delete
```
