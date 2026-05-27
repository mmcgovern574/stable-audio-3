# sweep_rater

Browser-based rater for any folder of .wav files. Single-page UI with
waveform visualization, star ratings, keyboard navigation. Stdlib HTTP server,
no Python dependencies, no npm.

## The model

* **A folder of wav files** is the unit of work. Generation and rating are
  completely decoupled.
* **Ratings live alongside the audio** as `<folder>/ratings.json`.
* **Filenames carry the metadata** — params like `cfg7`, `strength0.75`,
  `seed42` are extracted by regex for display chips and analysis. Filenames
  without recognized params still work; they just show no chips.

## Quick start

```bash
# Rate an existing folder of wavs
python sweep_rater/rate_server.py --folder ./eval_out
python sweep_rater/rate_server.py --folder ./favorites
python sweep_rater/rate_server.py --folder ./gen_out

# Aggregate ratings (parses params from filenames)
python sweep_rater/analyze_ratings.py --folder ./eval_out
```

Browser opens at http://localhost:8765.

## Generating a matrix sweep

`sweep_matrix.py` is an optional helper that runs `stable-audio` once per
parameter combination and writes wavs with parameter-encoding filenames into
a folder under `./sweep_rater/campaigns/`.

```bash
# Default sweep: cfg × strength × seed on the ICE training prompt
python sweep_rater/sweep_matrix.py

# Custom sweep — point at a JSON config:
python sweep_rater/sweep_matrix.py --config myconfig.json

# Then rate the output folder
python sweep_rater/rate_server.py --folder sweep_rater/campaigns/cfg_strength_v1
```

Generation and rating can run concurrently — the server re-scans the folder
on each refresh, so click ↻ refresh in the UI to pick up newly-generated
clips.

## Config file format (for sweep_matrix.py)

```json
{
  "name": "my_sweep",
  "lora_ckpt": "./lora_out/ckz_5000_dora/epoch=108-step=5000.ckpt",
  "model": "medium-base",
  "duration": 20,
  "prompt": "cKz! Midnight Bmin 110 bpm @crushed_keyz",
  "axes": {
    "cfg":      [2, 4, 6, 8, 10],
    "strength": [0.75, 1.0],
    "steps":    [50],
    "seed":     [42]
  }
}
```

## Keyboard shortcuts (in the rater UI)

* `←` / `→` or `j` / `k` — prev / next clip
* `space` — play / pause
* `0` – `5` — set rating
* `c` — clear rating
* `r` — jump to random clip
* `u` — jump to next unrated clip

## Filename convention for parameter extraction

Any tokens matching `<known-key><number>` are pulled out automatically. Known
keys: `cfg`, `strength`, `steps`, `seed`, `duration`, `rank`, `lr`.
Filenames that don't match still work — they just show no chips and the
analyzer treats them as having no params.
