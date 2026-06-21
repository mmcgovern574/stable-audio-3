# Crushed Keyz LoRA — checkpoints & testing summary

_Shared settings across all runs: medium-base, dora-rows, rank 16 / alpha 16,
exclude seconds_total, lr 1e-4, batch 1, fp16 on M4 Max (MPS), random crops._

## Trained LoRAs (in `lora_out/`)

| Dir | Data | Captions | Crop | Steps | Status / character |
|---|---|---|---|---|---|
| **ckz_5000_dora** | 46 Crushed Keyz (100%) | filenames | ~30s | 5000 (~108 ep), ckpts 1k–5k | **GOLD / original.** Strongest CKz timbre; melody weak. Best timbre in all testing. |
| ckz_2000 | 46 CKz | filenames | ~30s | 2000, ckpts 0.5k–2k | Old/superseded (tiny files; pre-dates the dora run). |
| hiphop_full | 623 (46 CKz + 577 Cymatics) | attribute-style | 12s | 2500, ckpts 0.5k–2.5k | **Abandoned** — wrong caption format. |
| hiphop_full_ext | 623 | attribute-style | 12s | +2500 (eff 3k–5k) | Abandoned continuation of above. |
| **hiphop_rawnames** | 623 (46 CKz + 577 Cymatics) | exact filenames | 12s | 2500 (~4 ep), ckpts 0.5k–2.5k | New main run. Melody ~same, timbre diluted. |
| **hiphop_rawnames_cont** | 623 | exact filenames | 12s | resumed → eff 3k–6k (folder 0.5k–3.5k) | Continuation. `eff6000` (epoch=5-step=3500) = current best of the new runs. |
| mps_smoke | — | — | — | 20 | Smoke test, disposable. |

Key data fact: in the 623 runs, Crushed Keyz is only **46/623 = 7%** of the mix →
each CKz file seen ~10× vs ~108× in the gold run. This is the dilution.

## Testing done

| Folder | What | Result |
|---|---|---|
| checkpoint_compare (29) | new raw-name ckpts (eff500–6000) on **medium-base**, exact CKz prompts | Low timbre everywhere (mostly T1–2); best Midnight T3. |
| checkpoint_compare_posttrain (36) | same ckpts on **post-trained medium** | Improved with more training; `eff6000` best (JUNGLE **T4/M4**), rest T1–3. Reversed the "over-training" fear. |
| old_lora_model_test (6) | **old LoRA**, base vs post-trained, 3 prompts | Mixed; old+base AzUL **T4/M3**; base ≈ post-train for the old LoRA. |
| **ckz_more (20)** | 10 exact CKz names: new-eff6000-posttrained vs old-base | **Decisive:** old timbre **2.80** vs new **1.40**; melody ~tied (2.1 vs 2.2). Confirms dilution. |
| old_lora_init_audio (10) | old LoRA + init_audio on medium-base, 3 inits × noise 0.42/0.46/0.52 + baseline | init_audio did **not** beat plain t2m baseline; all ~T2–3, generic; noise 0.42 best. |

_Ratings use the two-axis player (timbre + melody), 0–5._

## Bottom-line findings

1. **Old 46-file LoRA = best timbre by far** (2.80 vs 1.40). Timbre is the hard-won asset.
2. **Adding 623 loops did almost nothing for melody** (2.10 → 2.20) while halving timbre — the dilution tradeoff, now measured.
3. **Everything caps at ~T2–3 with a generic, "samey" timbre** across every approach (old/new LoRA, base/post-train, init_audio).
4. That convergence → suspected **systemic ceiling**, leading hypothesis = **flash-attn / MPS** (Medium requires Flash Attention 2, CUDA-only; missing it is a documented cause of degraded Medium output). **Open — awaiting Matt + a CUDA smoke test.**
5. Methodology caveats: recent tests were **single-seed (42)** and 12s (gold was multi-seed, 20s).

## Next steps (in priority order)
- Resolve flash-attn/MPS (Matt's reply + one clean CUDA generation). If timbre lifts on CUDA, the ceiling is a hardware artifact, not a recipe problem.
- If recipe-limited: rebalance (oversample the 46 CKz) to recover timbre; revisit init_audio with proper **seed sweeps**.
