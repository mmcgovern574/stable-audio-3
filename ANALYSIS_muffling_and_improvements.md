# Deep analysis — muffling, the floor problem, and how to improve cKz output
_2026-06-20 · over 1,602 rated clips + SA3 paper + codebase_

## CORRECTION (2026-06-20, acoustic-correlation pass over 1,618 clips)
**"Muffling" was the wrong diagnosis.** Correlating every acoustic feature with rating:
the good/bad axis is **DENSITY + DARKNESS**, not muffle. Worse clips: **silence r=-0.30**
(the #1 killer), **dynamic-range r=-0.26**, **hf_ratio r=-0.23 (BRIGHTER = worse)**,
flatness r=-0.14. Better clips: more low/low-mid band energy (r=+0.16..0.22), lower
centroid/rolloff (darker), more tonal, louder. So the cKz "good" sound is **dense, dark,
warm, full-low-end, loud, tonal**; the dominant FAILURE is **sparse / thin / bright / gappy**
(the original "step-out" problem — still #1). Low-HF/dark = GOOD (it's the cKz signature);
my earlier "low-HF=muffled=bad" was backwards, over-read from a few gate/strength runs.
Also: **melodic-movement (chroma/flux) ≈ 0 correlation** with rating — quality is the SOUND
(density/darkness), not melodic content. And **cfg/steps/strength/checkpoint do NOT separate
top from bottom quartiles** (cfg 3.97 vs 3.84 etc.) — confirming it's manifold-landing
(seed/conditioning), not a knob. Implications: (a) picker should add a density/sparseness
guard; (b) v2 retrain target = **density** (kill step-outs / sparse mode); (c) selection
remains the path. Supersedes the "muffling mechanism" framing in §2 below where they conflict.


## TL;DR
The problem isn't the ceiling, it's the **floor**: most generations are mediocre/muffled
(40–70% of clips score T≤2), and only ~3–20% are keepers. Muffling has one root cause —
**the diffusion model producing low-variance, "mode-averaged" latents**, which the SAME
autoencoder's fixed-scale bottleneck decodes into dull, quiet, high-frequency-poor audio.
Anything that weakens the conditioning's push away from the mean, or drags the latent off
the cKz manifold, increases muffling. The 288-clip library's **68% floor** was largely
self-inflicted (LoRA strength 0.85). Top fixes: regenerate at strength **1.0**, raise
**steps**, and ultimately **retrain the LoRA** (the distribution ceiling).

---

## 1. The data picture (1,602 rated clips)

- Most campaigns sit at **T2.3–3.5 with high floors (40–70% T≤2)** and low keeper rates
  (3–20%). The quality distribution has a **heavy bad tail** — generation is a lottery.
- The new **ckz_library (288)**: T2.31 / **68% floor / 3% keepers** — the duddiest recent run.
- **ckz_eff2500_full_melodies** (init, **strength 1.0**, noise 0.7): **27% floor** — less than
  half the library's floor. The main difference vs the library: **strength (1.0 vs 0.85)**.
- **Global LoRA strength**: 1.0 minimizes the floor; lowering it muffles (re-confirmed across runs).
- **Steps**: 100 → T3.5 (n18) vs 50 → T2.6 (n1088). Confounded, but consistent with theory.
- **Best-ever rated**: legacy `title_sweep` / `seed_variety` (OLD ckz_5000 checkpoint, cfg 7,
  curated titles/seeds) at T4.0+. The current pipeline rarely matches them — worth noting the
  old checkpoint at higher cfg produced strong results.
- Conclusion: the central lever is **raising the floor** (fewer muffled duds), not chasing
  the occasional 5-star.

## 2. First-principles: why output gets muffled / loses highs

**Verified mechanism.** The SAME autoencoder uses a `SoftNormBottleneck`
(`models/bottleneck.py`): encode does `x / running_std` (line 32), decode does
`x * running_std` (line 52), with `running_std` **frozen at inference**. This applies a
*fixed* scale — it does **not** correct a variance mismatch.

High-frequency, "sharp" audio requires **high-variance, on-manifold latents**. Diffusion's
characteristic failure under weak/uncertain conditioning is **mode-averaging**: it hedges
toward the mean of its latent distribution, which is spectrally smooth (averaging cancels
high-frequency content). A low-variance latent → decode → **dull, quiet, HF-poor = "muffled."**

So: **muffling = the model hedging toward the mean.** Every muffling case we hit maps to
"the conditioning's push got weaker" or "the latent went off-manifold":

| Observed | Mechanism |
|---|---|
| Global LoRA strength <1.0 → muffled | weaker push toward cKz mode → lower-variance latent |
| LoRA gated off early → muffled (lost highs) | structure forms off-manifold; late LoRA can't resolve HF |
| init noise too high / sparse stems → sparse/muffled | weak anchor → model uncertain → mode-average |
| cfg_interval gated early → slight muffle | guidance (the de-meaning force) absent early |
| stems (55% silent, 9× quieter) → sparse | almost no anchor signal → averaging |

**Secondary magnitude-compressors at our defaults** (each dulls at the margin):
- `apg_scale=1.0` (our default) uses only the *orthogonal* guidance component → less magnitude
  push than vanilla CFG (`apg_scale=0.0`).
- `rescale_cfg=True` (our default) normalizes output std → compresses dynamics/brightness.
- `soft_clip` (tanh on decode) compresses peaks if enabled.
- Too few steps → the final HF-refinement steps are insufficient (the model's default is 8,
  tuned for the *distilled* model; base needs more).

## 3. The library's 68% floor — diagnosed

The 288-clip library used **strength 0.85** (chosen from a 12-clip, single-loop tune on
dark128) across **48 diverse loops in all keys** at noise 0.6. Two compounding errors:
1. **strength 0.85 < 1.0** → weaker timbre push → mode-average → muffle. Global data says 1.0
   minimizes floor; full_melodies at 1.0 had 27% floor vs the library's 68% at 0.85.
2. **48 off-manifold foreign loops** (many non-cKz, some not dense) → many uncertain
   generations → muffle.

The "0.85-helps-with-init" result was **single-loop overfitting**; the aggregate refutes it.
This is exactly the single-seed/single-loop trap we kept hitting — caught here by the 1,602-clip aggregate.

## 3b. CORRECTION (fidelity_ab A/B, 16 clips) — it's the LOOP, not strength/steps
A controlled A/B (strength 0.85 vs 1.0 × steps 50 vs 100, on dark128 + oracle145) **refuted
§3's strength theory**: strength 1.0 was *not* better (T2.88 vs 3.25 at 0.85) and the floor
was identical (38% both). Steps 100 ≈ 50. The variance was entirely **per-loop**: dark128 =
T4–5 everywhere, oracle145 = T2–3 everywhere, regardless of settings. So the library's 68%
floor is a **loop-quality problem**: the 48-loop set contained many poor anchors. The
"full_melodies 27% vs library 68%" gap was confounded by loop selection (6 hand-picked dense
loops vs 48 mixed), not strength. **Lever = which loops you feed it**, and good-vs-bad anchors
aren't predictable from simple density/RMS (both test loops were dense). This still fits the
muffling mechanism (off-manifold init → mode-average → muffle) — it just means the controllable
knob is init-loop *manifold proximity*, not strength.

## 4. What to try next (ranked by confidence × cost) — REVISED
- **Curate init loops by proximity to the cKz training distribution** (e.g. CLAP/embedding
  cosine similarity between each Melody_Loops file and the 46 cKz training loops; keep the
  closest). On-manifold loops anchor cleanly → higher floor. Principled, and the similarity
  score doubles as a scorer feature.
- **Auto-picker** — now clearly the main lever: loop quality is large, unpredictable, and
  uncontrollable a priori, so over-generate across loops × seeds and **let the scorer keep the
  dark128-quality hits and discard the oracle145 duds**. Selection IS loop-curation.
- **Retrain LoRA v2** — still the structural ceiling-raiser (rank, crop, denser data).
- Strength/steps: **do NOT** spend more here — A/B says flat/loop-dominated.
- (Optional) brightness A/B (apg/rescale) — small, may give marginal HF.

## 4-OLD. (superseded by §3b/§4-REVISED) earlier ranked list

**A. Regenerate the library at strength 1.0 (cheap, highest-confidence).** Expect the floor to
drop toward full_melodies' ~27%. Confirm first with a small strength×steps A/B (below) so we
don't spend 2h on a hunch.

**B. Brightness A/B (cheap).** On a fixed good init set, test `apg_scale` 0.0 vs 1.0 and
`rescale_cfg` off vs on, rated specifically for brightness/HF. Our defaults (apg 1.0, rescale
on) both compress magnitude; turning them off may de-muffle. (We found them "inert" before, but
never measured brightness at scale or combined.)

**C. Latent-variance rescale (novel, experimental).** `return_latents=True`, scale the generated
latent to match the training-latent std, then decode. Directly attacks the mode-averaging
mechanism. Higher risk (may amplify noise) but cheap to prototype.

**D. Retrain the LoRA v2 (the real ceiling-raiser, multi-hour).** The LoRA's learned
distribution caps everything downstream — a tighter, brighter adapter raises the *whole* floor.
Levers: higher rank (16→32), fix the mid-melody crop offset, denser/cleaner data, early-stop by
seed-sweep. This is the highest-leverage structural move.

**E. Auto-picker (after A/D).** Selects above the floor; most valuable once the candidate pool
is less duddy.

## 5. Recommendation
1. **Confirm + fix the floor**: small strength×steps A/B → regenerate the library at the winner
   (likely strength 1.0). 
2. **Brightness A/B** (apg/rescale) — quick, may give free HF.
3. **Commit to the v2 retrain** as the structural fix.
4. **Build the auto-picker** once the floor is up.
