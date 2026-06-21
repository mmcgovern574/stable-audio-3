# Crushed Keyz LoRA — experiment matrix

Living record of every training run and inference sweep. Ratings are timbre/melody (0–5)
from `sweep_rater`. **Floor%** = share of clips at T≤2 (the single-instrument failures).
Updated 2026-06-19.

> Note: legacy runs (★) were rated on the OLD single-score scale, so their timbre=melody.
> Not directly comparable to the new two-axis ratings.

## Settled conclusions
- **Recipe:** `ckz_melody_capped/epoch=54-step=2500.ckpt` · medium-base · cfg 3 · steps 50 · strength 1.0 · t2m · seed-hunt.
- **Ruled out:** post-trained `medium` (worse), Cymatics 623-loop dilution, init_audio, training past ~2500.
- **Open levers being tested:** negative_prompt, sampler_type, apg_scale, rescale_cfg, LoRA interval.

---

## Training runs (LoRA checkpoints)

| checkpoint dir | data | captions | crop | steps | verdict |
|---|---|---|---|---|---|
| `ckz_5000_dora` | 46 CKz | filenames | 30s rand | 5000 | old gold; **overfit** — step3000 > step5000 |
| `ckz_2000` | 46 CKz | filenames | 30s | 2000 | superseded |
| `hiphop_full` / `_ext` | 623 (CKz+Cymatics) | attribute | 12s | 2500/+ | abandoned (wrong captions) |
| `hiphop_rawnames` / `_cont` | 623 | filenames | 12s | →6000 | melody≈, **timbre diluted** (CKz=7%) |
| **`ckz_melody_capped`** | 46 CKz | filenames | **30s capped[0,45]** | 3000 | **WINNER → eff-2500**; fixes single-instrument floor |
| `ckz_melody_capped_cont` | 46 CKz | filenames | capped | →5000 | overfits past 2500 |

---

## Inference experiments

| run | ckpt | model | what varied (held constant) | clips | T / M / floor | learning |
|---|---|---|---|---|---|---|
| old_lora_earlier_ckpts | ckz_5000 1k/3k/5k | base | checkpoint | 24 | 2.2/1.9/62% | **step3000 best** (5000 overfit) |
| validate_step3000 | step3000 | base | seed sweep + init | 26 | 2.7/2.4/50% | step3000 hits T4 on good seeds; init flat |
| ckz_more | new-6000 vs old | mixed | LoRA+model | 20 | 2.1/2.2/65% | **dilution confirmed** (old timbre 2.8 vs new 1.4) |
| checkpoint_compare | rawnames | base | checkpoint | 29 | 1.5/1.5/93% | diluted LoRA weak on base |
| checkpoint_compare_posttrain | rawnames | **post** | checkpoint | 36 | 1.7/1.7/77% | post-trained didn't save it |
| **seedsweep_candidates** | capped 2500/3000/4000 | base | checkpoint (ICE+AzUL ×10 seed) | 36 | 3.3/2.5/8% | **eff-2500 wins**: T3.5, floor 0 |
| eff2500_medium | eff-2500 | **post** | model | 12 | 2.8/2.0/41% | **base beats post** (post floor 5) |
| ckz_eff2500_allprompts | eff-2500 | base | 46 prompts ×4 seed | 184 | 2.1/2.0/74% | **prompt-dependent**; 14% keepers; ACTiVE/AzUL top |
| ckz_eff2500_cfg_sweep | eff-2500 | base | **cfg [2-7]** ×4 prompt | 60 | 2.7/2.6/48% | **cfg FLAT**; 3AM dead at every cfg |
| ckz_eff2500_steps_strength | eff-2500 | base | **strength[.7/1/1.3] × steps[50/100]** | 36 | 3.5/2.9/8% | **strength 1.0 best**, steps FLAT |
| ckz_eff2500_seedhunt | eff-2500 | base | 8 top prompts ×16 seed | 128 | _partial_ | seed-hunt batch (rating) |
| ckz_eff2500_prompt_ablation | eff-2500 | base | **prompt wording** (ACTiVE) ×8 seed | 56 | _partial_ | does wording move the floor? |
| ckz_eff2500_negprompt | eff-2500 | base | **negative off/on** ×3 prompt | 36 | _pending_ | does neg-prompt lift floor? |
| ckz_eff2500_sampler_sweep | eff-2500 | base | **sampler × apg × rescale_cfg** (ACTiVE seed7) | 24 | T4 on euler/dpmpp; rk4+pingpong=garbage(T1) | **dpmpp best** (M4); apg & rescale INERT; rk4/pingpong broken |
| ckz_eff2500_cfg_wide | eff-2500 | base | **cfg [0→100]** (ACTiVE seed7, dpmpp) | 9 | useful band 1–8, flat plateau 1–4; garbage at 0 & ≥32 | cfg 3 confirmed dead-center; **inference knobs exhausted** |
| ckz_eff2500_word_ablation | eff-2500 | base | **prompt words** (remove 1/2/3 units, ACTiVE seed7) | 68 | got **T5/M5** ("15. cKz! ACTiVE Cmin") | **WORDING MATTERS.** trigger +1.1, collab −0.7 (remove), title ~0; notes: `155 bpm`→loud shaker, collab→vocal stem; step-outs persist |
| ckz_eff2500_wordwin_seeds | eff-2500 | base | top 5 prompt forms (±bpm/collab) × 12 seeds | 60 | **full prompt best: T3.6/M3.2/floor 0%**; stripping hurts (floor→66%) | **REVERSES word_ablation** — those were seed-7 artifacts. Use FULL exact prompt. bpm/collab help. ACTiVE = reliable (0 duds/12 seeds) |
| ckz_eff2500_descriptor_sweep | eff-2500 | base | **descriptor WORD** (ACTiVE→15 words) × 4 seeds | 60 | superseded by isolation test | — |
| ckz_eff2500_word_isolation | eff-2500 | base | **descriptor word @ FIXED seed 7** (24 words) | 24 | adjectives = "not much diff" | **invented adjectives are COSMETIC** at cfg3; seed dominates skeleton. Need trained tokens (key/bpm/title) |
| ckz_eff2500_melody_variety | eff-2500 | base | **FIXED seed 7, strong tokens**: key sweep / real training prompts / stacked names / bpm sweep | 25 | **KEY + BPM = real levers** (Gmin/Bmin/130/160 = M4); creative words cosmetic; **timbre ALWAYS T3-4** (trigger holds it) | novelty engine = key×bpm×seed, NOT creative titles. Stacked names = weak melody (M2). Whole training prompts = variety but middling |
| ckz_eff2500_key_bpm_grid | eff-2500 | base | **key×bpm grid** (Cmin/F#min/Gmin/Bmin × 130/155/160) × 2 seeds + creative-title arm @ Bmin160 | 30 | _running_ | confirm key×bpm sweet spot beyond single-seed; show creative words = timbre-safe but melody-flat |
| ckz_eff2500_depths_grid | eff-2500 | base | **key×bpm grid on DEPTHS title** (Cmin/F#min/Gmin/Bmin × 130/140/160) × 2 seeds | 24 | _not run — superseded_ | key×bpm shown unreliable (2-seed); pivoting to LoRA-strength lever |
| ckz_eff2500_key_bpm_grid (RATED) | eff-2500 | base | key×bpm × 2 seeds | 30 | T2.8/M3.1; **Cmin(base) best key M3.5, Gmin/F#min M2.2** | **key/bpm wins were single-seed flukes**; prompt is a WEAK lever — everything converges. Cause = narrow LoRA, not the words |
| ckz_eff2500_lora_strength_variety | eff-2500 | base | **LoRA STRENGTH 0.3-1.0** × 4 seeds (fixed ACTiVE prompt) | 20 | **REFUTED**: T1.0→3.5 as str 0.3→1.0; melody flat (~2.25); lowering str = muffled/saturated, no variety gain | LoRA-strength (global) NOT a variety knob. 1.0 required for timbre |
| ckz_eff2500_timegate_novelty | eff-2500 | base | **TIME-GATED LoRA** `lora_start_sigma` [1.0/0.7/0.5/0.35/0.2] × 4 seeds | 20 | gate1.0 T4-5; **all gate≤0.7 = T1 (no timbre)** | gates too aggressive — all past the cliff. Wiring CONFIRMED correct via gate_debug |
| ckz_gate_debug | eff-2500 | base | gate [1.0/0.95/0.9/0.7] seed7 + CKZ_GATE_DEBUG readback | 4 | **gate 0.95 = T4 (timbre intact!)**; 0.9 T2; 0.7 T2 | **VIABLE BAND EXISTS.** Wiring correct (buffer toggles 0→1, readback confirms). Schedule lingers high-σ: gate0.95 = LoRA-off first ~40% steps yet timbre holds. Break between 0.95↔0.9 |
| ckz_eff2500_timegate_band | eff-2500 | base | **viable band** gate [1.0/0.97/0.95/0.93/0.90] × 4 seeds | 20 | T 4.25→1.25 as gate drops; **melody UNCHANGED** ("same melody, worse timbre"); muffled/low-pass | time-gate = DEAD for novelty. Confirms melody=seed, timbre=LoRA. Cause: SAME latent is semantic-acoustic (entangled by design, per paper §2.1) |
| ckz_eff2500_layerfilter | eff-2500 | base | **native lora_layer_filter** (off self_attn/cross_attn/ff; LoRA-only-ff; LoRA-only-self_attn) × 2 seeds | 12 | off self_attn=T4(inert); off ff/cross=timbre dies; **LoRA-only-ff also dies (T1.5)**; melody UNCHANGED every arm | **NO decoupling.** Timbre = holistic transform (needs ff+cross_attn together), self_attn LoRA ~inert. **3rd axis confirming melody=seed, timbre=holistic-LoRA.** Inference-lever search CLOSED |
| ckz_eff2500_cfg_interval | eff-2500 | base | **native cfg_interval** [(0,1)/(0,.7)/(0,.5)/(0,.3)] × 4 seeds | 16 | _queued_ | time-gated GUIDANCE, LoRA full (timbre-safe). Does removing early prompt-pull diversify melody without muffling? |
| ckz_eff2500_timegate_transfer | eff-2500 | base | **gate + init** (Cymatics loops × noise[.55/.7] × gate[1.0/.45/.3]) | 12 | **gate1.0(full LoRA)=T3.5/M3.2, 3× "dope NEW melody" T4/M4**; gate≤0.45 DEAD (T1.2, "piano not cKz", muffled) | **gate dead (5th confirm).** BUT **init+FULL LoRA = novelty WIN**: foreign loop → new cKz melody, timbre holds. Reopens init as a NOVELTY engine (≠ melody-preservation) |
| ckz_eff2500_init_stems | eff-2500 | base | **init TYPE** (Bad Energy Cmin: keys/bells/full/ambience) × noise[.55/.7] × 2 seeds, FULL LoRA | 16 | full=T3/M3 (best); keys/bells/amb=T2.5/M2.2 "sparse melody" | **stems too SPARSE+QUIET** (bells 55% silent/2.4s gaps, 9× quieter than full); full mix wins. **Use DENSE FULL loops as init, not stems** |
| ckz_eff2500_full_melodies | eff-2500 | base | **6 dense full Cymatics Cmin loops** (80-160bpm, varied moods) × noise0.7 × 2 seeds, FULL LoRA | 12 | T3.0/M2.8; best dark128(T4M3)/oracle145(T3.5/3.5)/monster80("dope new"); cobra/tokyo="like past sweep" | init engine WORKS + beats stems, but hit-rate ≈ seed-hunt. 2nd timbre-safe variety axis (loop), inconsistent per-loop |
| ckz_eff2500_init_strength_grid | eff-2500 | base | **init_noise[.6/.75] × LoRA strength[.85/1.0/1.15]** on dark128 × 2 seeds | 12 | **best noise0.6×str0.85 = T3.5/M3.5 (T4M4 hit)**; str1.15 worse; noise0.75="sparse" | **with init, LOWER LoRA (0.85)>1.0** (opposite of pure-t2m); over-apply REFUTED; noise0.6 anchors dense loops best |
| ckz_eff2500_init_strength_fine | eff-2500 | base | **LoRA strength[.75/.85/.95]** × 2 loops (dark128/monster80) × 2 seeds, noise0.6 | 12 | _superseded by fidelity_ab_ | — |
| **ckz_library** (288) | eff-2500 | base | init library, str0.85, noise0.6, 48 loops all keys | 288 | T2.31 / **68% floor** / 3% keep | floor driven by POOR-ANCHOR loops, not strength (see fidelity_ab) |
| ckz_eff2500_fidelity_ab | eff-2500 | base | **strength[.85/1.0] × steps[50/100]** × 2 loops × 2 seeds, noise0.6 | 16 | str0.85 T3.25 ≥ str1.0 T2.88; **floor 38% both**; steps flat; **dark128 T4-5 vs oracle145 T2-3 regardless** | **REFUTES strength/steps floor-fix.** Floor = LOOP QUALITY (manifold proximity), not settings → curate loops + auto-pick |

### Time-gated LoRA — implementation (the image-style-transfer analog)
- **Why:** diffusion is coarse-to-fine in t — early/high-σ steps set global structure (melody/key/rhythm), late/low-σ steps set fine detail (timbre). Full-time LoRA imposes BOTH → narrow melody + cKz timbre. Gating the adapter to low-σ only = "structure from base/init, style painted late" = the audio analog of Picasso-style transfer.
- **Code:** `model.py` `_LoraGatedBackbone` proxies the DiT, calls `set_lora_strength(σ→strength)` before each forward (only backbone matters mid-sampling; conditioner LoRA is computed once up front). `generate(..., lora_strength_schedule=fn)`. Sweep axis `lora_start_sigma` (+ optional `lora_low`, default 0.0); harness builds the σ-threshold closure, resets `last_strength` after gated combos. `gate=1.0` == old behavior (exact control).

### init_audio line (all ruled out for quality)
| run | result | note |
|---|---|---|
| old_lora_init_audio | T2.3/M1.8/70% | flat, on overfit ckpt |
| init_audio_eff2500 | T2.4/M2.6/50% | flat on the clean ckpt too — baseline (no init) won |
| init_audio_cmin/bmin/cfg/noise* ★ | ~2.0–3.0 | old: 0.46 band = key/BPM anchor only, not quality |

### Legacy ckz_5000_dora runs ★ (single-score scale)
| run | score/floor | note |
|---|---|---|
| title_sweep ★ | 4.1 / 0% | novel cKz titles — strong (but legacy scale) |
| seed_variety ★ | 4.0 / 5% | Midnight seed sweep |
| cfg_strength_midnight ★ | 3.5 / 10% | |
| multi_prompt_validation ★ | 2.9 / 35% | |
| cfg_strength_azul/v1 ★ | 2.3–2.4 / 45–66% | |

---

## Pending / next
- Finish rating: seedhunt, prompt_ablation, negprompt, sampler_sweep.
- LoRA **interval** sweep (needs separate plumbing — set on model, not a generate() arg).
- The actual tool: **auto-scorer + rejection sampling** trained on the ~900 ratings.
