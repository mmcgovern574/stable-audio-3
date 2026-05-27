# Best prompts — Crushed_Keyz stem (variation)

Personal notes from tuning **Stable Audio 3 MLX** on a melodic loop stem. Paths assume you run commands from **`optimized/mlx/`**.

Kept renders live in **`output/best/`** (archived WAVs section below).

Favourite **`σ`** band (listening): **`0.40`**–**`0.46`** (**`prompt ""`**, **`steps`** **4**) — quartet in **`output/best/no_text_sigma_sweet_strip/`** and **`ckz_nt_sig0p*`**. **`0.46`** snapshot: **`output/best/ckz_no_text_var_sigma046.wav`**.

**Strong alt** (good, not favourites): **`σ`** **`0.475`** & **`0.485`** → **`output/ckz_nt_above46_sig*.wav`** (**`output/best/`** keeps copies too).

> Recreate **`output/ckz_seed30.wav`** (ffmpeg below) if it’s missing. Relative **`--out`** WAVs resolve under **`output/`**.

## Prerequisite: 30 s reference WAV (`output/ckz_seed30.wav`)

Recreate **`ckz_seed30`** anytime (from a 44.1 kHz stereo export of the loop):

```bash
# From a WAV master (trim first 30s)
ffmpeg -y -i /path/to/loop.wav -t 30 -ar 44100 -ac 2 -sample_fmt s16 output/ckz_seed30.wav

# Or decode from your original MP3, then trim
ffmpeg -y -i "/path/to/track.mp3" -t 30 -ar 44100 -ac 2 -sample_fmt s16 output/ckz_seed30.wav
```

`./sa3` only accepts **WAV**, 44.1 kHz, 16‑bit PCM (convert MP3s with ffmpeg first).

---

## Best (current): text-free variation (`--prompt ""`)

**Why:** **`--prompt ""`** is unconditional text conditioning (learned pad embeddings only). With **`--init-audio`**, the model biases off the stem **without** extra lexical pull toward specific instruments—“sparse keys texture” wording was still nudging semantics. **`--cfg 1.0`** keeps a single DiT forward (no CFG doubling).

**Canonical command:** set **`INIT_SIG`** in **`0.40`** … **`0.46`** first (table below); **`above46`** WAVs optional.

### **`σ`** (**`--init-noise-level`**) bands that tested well

Same sweep setup: **`prompt ""`**, **`steps` 4**, **`cfg` 1**, **`medium`/`same-l`**, **`ckz_seed30`**, **`seconds`** **30**.

**Primary band (favourites)** — **`σ`** **`0.40`**–**`0.46`** (“closer stem”):

| `σ` | Example file |
|-----|----------------|
| 0.40 | `output/ckz_nt_sig0p40.wav` |
| 0.42 | `output/ckz_nt_sig0p42.wav` |
| 0.44 | `output/ckz_nt_sig0p44.wav` |
| 0.46 | `output/ckz_nt_sig0p46.wav` |

**Archived:** `output/best/no_text_sigma_sweet_strip/` (those four WAVs).

**Higher `σ` — strong alternates** (more drift; good listens, not top tier for this stem): **`0.475`**, **`0.485`**:

| `σ` | Example file |
|-----|----------------|
| 0.475 | `output/ckz_nt_above46_sig0p475.wav` |
| 0.485 | `output/ckz_nt_above46_sig0p485.wav` |

```bash
INIT_SIG=0.42   # or 0.40–0.46 favourite; alt 0.475 / 0.485 if you prefer more variation

cd /Users/mikemcgovern/stable-audio-3/optimized/mlx

./sa3 --prompt "" \
      --dit medium --decoder same-l \
      --seconds 30 \
      --init-audio output/ckz_seed30.wav \
      --init-noise-level "${INIT_SIG}" \
      --cfg 1.0 --steps 4 \
      --out "ckz_no_text_sigma_${INIT_SIG}.wav"
```

**Next lever:** **`--seed`** sweep at **`σ`** **`0.42`** (**or** **`0.44`**) inside the primary band.

---

## Archived “best audio” snaps (`output/best/`)

Snapshots kept on disk:

| File | Notes |
|------|--------|
| **`output/best/no_text_sigma_sweet_strip/`** | Favourite **`σ`** **0.40**–**0.46** (no‑text quartet) |
| **`output/best/ckz_no_text_var_sigma046.wav`** | **`σ`** **0.46**, **`steps`** 4 baseline |
| `output/best/ckz_nt_above46_sig0p475.wav` | Alt — **`σ`** **0.475** (good, not favourite band) |
| `output/best/ckz_nt_above46_sig0p485.wav` | Alt — **`σ`** **0.485** (good, not favourite band) |
| `output/best/ckz_var_steps4_sig0.46.wav` | Best earlier run with **text** prompt (`σ`=0.46) |
| `output/best/ckz_var_steps4.wav` | Text prompt, **`σ`**=**0.48** |

---

## Earlier strong recipe (explicit text prompt)

If you ever want linguistic steering again, this was the best **text** line before empty-prompt won:

```bash
./sa3 --prompt "lightly varied phrase same sparse keys texture" \
      --dit medium --decoder same-l \
      --seconds 30 \
      --init-audio output/ckz_seed30.wav \
      --init-noise-level 0.46 \
      --cfg 1.0 --steps 4 \
      --out ckz_var_steps4_sig0.46.wav
```

Slightly bolder: same block with **`--init-noise-level 0.48`** → `ckz_var_steps4.wav`-style naming.

---

## Quick knobs (`--prompt ""` or any fixed prompt string)

| Goal | Try |
|------|-----|
| Favourite **`σ`** band | **`0.40`** … **`0.46`** (`ckz_nt_sig0p*` / **`no_text_sigma_sweet_strip/`) |
| More variation (alt tier) | **`0.475`**, **`0.485`** (`ckz_nt_above46_sig*`) — good, not top pick here |
| Even closer to **`ckz_seed30`** inside that band | lower **`σ`** (e.g. **0.39**–**0.41**) |
| Braver inside favourite band | **`0.45`**–**`0.46`**; beyond that try **`0.475`** / **`0.485`** as alternates |
| Different micro-take | **`--seed N`** (**`σ`** + **`steps`** fixed) |
| Sharper / more denoiser “finish” vs stem-stickiness | **`--steps` 6** or **8** (often best A/B vs **`steps` 4** at mid **`σ`**) |
| Less lexical bias | **`--prompt ""`** (canonical for this stem) |

---

## Approaches that were weaker *for this stem*

- **Inpainting continuation** (`--inpaint-range` + short init): good for freezing a prefix exact, but the **new segment** often sounded busier / less like the stem.
- **High `--cfg` + long `--negative-prompt`**: tended toward **dense, “jumbled”** mixes for this use case.
- **`--cfg` 1.25–2 vs 1.0** (same init / σ): **1.0** routinely preferred — higher CFG tended to sound **heavier**, not sharper.

---

## License

Stable Audio weights: **Stability AI Community License** — see https://stability.ai/license .