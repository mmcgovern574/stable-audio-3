# Crushed Keyz LoRA — Findings

Compiled from a multi-session effort training and evaluating a style LoRA on
Stable Audio 3 `medium-base` against 46 melody loops from the producer
Crushed Keyz, running entirely on Apple Silicon (MPS).

---

## TL;DR

> **The LoRA encodes cKz timbre reliably. The seed encodes random melody.
> At a known-good prompt + best params, ~25% of seeds land at 5★ and 75%
> at 4–5★ — cherry-picking from mass-seed generation IS the production
> workflow.**

Validated empirically: 20 clips at `cKz! Midnight Bmin 110 bpm @crushed_keyz`
with `cfg=3 strength=1.0 steps=50`, varying only seed → mean 3.95, 25% 5★.
Higher quality than any parameter sweep. `--init-audio` is now a "more
control" upgrade, no longer required to fix output quality.

---

## Setup

- **Branch:** `feat/mps-training` of `mmcgovern574/stable-audio-3` (fork of
  Stability-AI/stable-audio-3 with Mac-specific patches).
- **Training data:** 46 Crushed Keyz melody loops, ~15 s each, captioned
  with their filename stems (e.g. `cKz! ICE Dmin 135 bpm @crushed_keyz`).
- **Base model:** `medium-base` (2.3 B params total: 1.5 B DiT + 198 K
  conditioner + 852 M VAE).
- **LoRA:** `dora-rows` (DoRA with per-output-neuron magnitude — the repo's
  documented default), `rank=16`, 5000 steps, ~108 epochs over the dataset.
- **Final checkpoint:** `./lora_out/ckz_5000_dora/epoch=108-step=5000.ckpt`
- **Hardware:** M-series Max with 64 GB unified memory.
- **Training time:** ~3.5 hours wall clock at ~0.5 it/s.
- **fp16 throughout** training and inference on MPS.

---

## Production-ready parameters

Lock these in for any new generation against this LoRA:

| Parameter        | Value          | Why                                                      |
| ---              | ---            | ---                                                      |
| `cfg-scale`      | **3**          | Best on every controlled prompt tested (ICE/Midnight/AzUL). |
| `lora-strength`  | **0.5 – 1.0**  | Monotonically helps; 0.25 is a hard cliff (avoid).       |
| `steps`          | **50**         | Quality plateaus; 8 is too rough; 80+ diminishing.       |
| `seed`           | **123 or 999** | `seed=42` is consistently ~0.5★ worse across n=44.       |
| `duration`       | **20 s**       | Trained at 30 s, infers fine at 10–30 s.                 |
| `negative-prompt`| not yet tested | Worth trying for non-cKz prompts if LoRA bleeds.         |

---

## What the LoRA learned

**Timbre: excellent.** Sound design, instrument character, voicing colors,
mix aesthetic — all reliably present in every output regardless of seed or
prompt. This is what "style transfer" means and the LoRA nailed it.

**Melody: not learned.** Diffusion audio models struggle with long-range
melodic coherence even at the base layer. A LoRA on 46 clips can shift the
timbral distribution but doesn't have enough data to teach the model new
melodic vocabulary. The base model's generic melodic priors dominate. Each
seed produces a different random melody, mostly mediocre, occasionally
great.

**Prompt: mostly an anchor.** At a fixed seed, varying the title (e.g.
`Twilight` → `Crystal`) produces only minor perturbations in the output —
the noise tensor dominates the structural outcome. *Across many seeds*,
some prompts have higher base-quality distributions (Midnight: 3.46 mean;
ICE/AzUL: ~2.4 mean), but you can't see this from a single sample.

---

## What each axis does

| Axis            | Effect                                            | Notes                          |
| ---             | ---                                               | ---                            |
| `cfg-scale`     | Strong — lower CFG lets the LoRA breathe         | 3 > 5 > 9 > 7 across all prompts |
| `lora-strength` | Strong — more = more cKz character (up to 1.0)  | 0.25 = LoRA basically dead     |
| `seed`          | Strongest for *variety* — dominates melodic skeleton | seed=42 unlucky on aggregate |
| Prompt title    | Marginal at fixed seed                           | Affects quality distribution across seeds, not single outputs |
| Prompt key      | TBD (key_sweep not yet rated)                    |                                |
| Prompt BPM      | TBD (bpm_sweep not yet rated)                    |                                |
| `duration`      | Longer = less coherent (more melodic drift)      | 20s sweet spot                 |
| `steps`         | Quality saturates by 50                          | 8 is preview-only              |

---

## Per-prompt quality distributions (n=48 each, full controlled sweep)

```
Midnight (Bmin 110 bpm)   mean=3.46  ← outlier; "easy" for the LoRA
ICE      (Dmin 135 bpm)   mean=2.44
AzUL     (Fmin 140 bpm)   mean=2.33
```

Same parameter sweep, same LoRA, three different titles → one full star
difference in mean rating. **The prompt you give matters more than the
parameters you tune.** Some property of "Midnight Bmin 110 bpm" makes it
cooperate with this LoRA more than other prompts. Probably some
combination of training-data key/BPM distribution and how T5Gemma encodes
the title semantics.

---

## Standout 5★ outputs

All from `cfg_strength_midnight` or `cfg_strength_azul` sweeps. Stored in
`./favorites/` to survive cleanup.

| File                                                    | Params                                   |
| ---                                                     | ---                                      |
| Midnight Bmin 110 bpm                                   | cfg=3 strength=0.5 seed=999              |
| Midnight Bmin 110 bpm                                   | cfg=3 strength=1.0 seed=123              |
| Midnight Bmin 110 bpm                                   | cfg=5 strength=0.5 seed=999              |
| Midnight Bmin 110 bpm                                   | cfg=7 strength=0.75 seed=123             |
| Midnight Bmin 110 bpm                                   | cfg=9 strength=0.75 seed=123             |
| AzUL Fmin 140 bpm                                       | cfg=3 strength=0.5 seed=999              |
| AzUL Fmin 140 bpm                                       | cfg=5 strength=0.5 seed=999              |

Pattern: 5★s span every cfg from 3–9 and strengths 0.5–1.0, but always
seed 123 or 999, never 42. The cfg/strength combo just shifts the
distribution slightly; the seed determines whether you land in a "good
melody" region of latent space.

---

## Pitfalls discovered (and worked around)

These are real bugs we encountered. Bake them into any future workflow.

1. **`torch.compile` produces NaN audio silently on this MPS stack.**
   The Inductor backend's Metal shader codegen has bugs (references
   undefined variables). Some kernels fall back to eager and work; others
   produce NaN that propagates through diffusion sampling. Output WAVs are
   the right size and the script reports "saved" — but every sample is NaN.
   **Fix:** `--compile` is opt-in only in `sweep_matrix.py`. Off by default.

2. **`adapter_type=lora-xs` checkpoints are unreliable.** The
   `name_is_lora` filter in `stable_audio_3/models/lora/utils.py:76`
   excludes the U/V SVD-basis buffers from saved state_dicts. At inference,
   `add_lora()` recomputes SVD non-deterministically → trained `M_xs` gets
   multiplied against different bases → near-zero effective delta.
   **Fix:** use `dora-rows` (the repo's documented default).

3. **macOS dataloader spawn can't pickle local lambdas or unimported names.**
   `worker_init_fn=lambda ...` inside `train()` crashes on spawn; module-level
   `from pathlib import Path` is invisible to workers. **Fix:** import inside
   functions used in workers; drop the lambda (Lightning's `seed_everything(workers=True)`
   already seeds each worker).

4. **HF `xet` chunked downloader stalls silently.** Showed 5.8 GB partial
   download with 0 progress and no error. **Fix:** `HF_HUB_DISABLE_XET=1`
   by default in `train_lora.py`. Legacy downloader is slower at peak but
   reliable and has a working progress bar.

5. **GitHub secret scanning blocks pushes if any commit history contains
   a token.** Even if you scrub it from the file, the old commit still has
   it. **Fix:** rotate the token *immediately* (assume compromised), then
   rewrite history with `git filter-repo`.

6. **The repo's `pyproject.toml` puts `matplotlib` in the `ui` extra**,
   but `training/diffusion.py` imports it transitively via
   `interface/aeiou.py`. `uv sync --extra lora` doesn't pull it. **Fix:**
   added matplotlib to the `lora` extra.

7. **`wandb` is imported unconditionally in `training/utils.py`**, breaking
   any user who doesn't have wandb installed. **Fix:** wrapped import in
   try/except — wandb is only used when `--logger wandb` is selected.

8. **`t2m demo sample_size` defaulted to model max (380 s).** At
   `duration=30`, demos were being generated at 380 s causing ~12× slower
   demo phase. **Fix:** align demo `sample_size` to `--duration` in
   `train_lora.py`.

---

## Performance optimizations applied

| Change                                              | Speedup                              |
| ---                                                 | ---                                  |
| In-process model load (one `from_pretrained()` for 48 clips vs 48 subprocesses) | ~35% per-clip; saves 24 min on 48-clip sweep |
| `ThreadingHTTPServer` in rate UI                    | Eliminates multi-audio request blocking |
| WaveSurfer instance reuse (load() vs destroy+create) | ~200–400 ms saved per clip nav      |
| `<audio>` lazy creation in "previously rated" list  | Stops media decoder leak at 100+ rows |
| Skip `renderRated()` on every nav                   | O(N) → O(1) per keystroke           |
| Cap rated list to top 20 (with "show all" toggle)   | DOM size stays bounded               |

---

## Open questions / promising next experiments

1. **Validate the mass-seed workflow on a second prompt.** The
   seed_variety result is at one known-good prompt (Midnight). Does the
   75%-at-4★+ hit rate generalize to other prompts that scored well
   individually (AzUL Fmin 140 had 2 of 36 hit 5★)? If yes, the workflow
   is universal. If no, prompt selection matters more than we think.

2. **Build `mass_generate.py`.** Production workflow is now empirically
   validated. Wrap it in a single script: takes a prompt list, generates
   N seeds per prompt at locked-in best params, writes to a campaign
   folder ready for the rater. ~50 lines on top of the sweep_matrix
   plumbing.

3. **`--init-audio` workflow.** Was on the critical path before the
   seed_variety result; now an upgrade for *more deterministic* output
   rather than a fix for *bad* output. Still worth exploring — provides
   melodic control instead of seed lottery.

3. **Negative prompt experiments.** Could potentially reduce LoRA bleed on
   non-cKz prompts (e.g. metal/violin/IDM). Not yet tested.

4. **Per-checkpoint comparison.** We have step 1000/2000/3000/4000/5000
   checkpoints saved. An A/B against the same prompt+seed could tell us if
   training plateau happened earlier than 5000 — and whether late training
   added overfit-risk without quality gain.

5. **Larger dataset for melody learning.** 46 clips wasn't enough for the
   LoRA to learn melodic structure. With 200+ clips, the structural prior
   might shift. Same producer's catalog or curated multi-producer cKz-style
   collection. Significant data work — only worth it if `--init-audio`
   doesn't satisfy the use case.

---

## Workflow reference

### Train a new style LoRA on a different producer's loops:

```bash
uv run python scripts/train_lora.py \
  --model medium-base \
  --data_dir ./data/<producer>_loops \
  --duration 30 --rank 16 --adapter_type dora-rows \
  --exclude seconds_total \
  --steps 5000 --batch_size 1 \
  --checkpoint_every 1000 --demo_every 1000 \
  --num_t2m_demos 4 --demo_steps 50 \
  --demo_prompts_file ./data/<producer>_demo_prompts.txt \
  --save_dir ./lora_out/<producer>_5000_dora --name <producer>-5000
```

### Inference at production defaults:

```bash
uv run stable-audio \
  --model medium-base \
  --lora-ckpt-path ./lora_out/ckz_5000_dora/epoch=108-step=5000.ckpt \
  --lora-strength 1.0 \
  -p '<your prompt>' \
  --duration 20 --steps 50 --cfg-scale 3 --seed 999 \
  --output out.wav
```

### Run a parameter sweep + rate:

```bash
uv run python sweep_rater/sweep_matrix.py --config sweep_rater/configs/<config>.json
python sweep_rater/rate_server.py --folder sweep_rater/campaigns/<config_name>
python sweep_rater/analyze_ratings.py --folder ./sweep_rater/campaigns/<config_name>
```
