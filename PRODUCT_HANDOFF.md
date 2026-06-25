# cKz Loop Generator — Product/UI Handoff

Context for building a UI (→ website → VST) around the Crushed Keyz (cKz) melody generator.
Everything below is settled by experiment; don't re-derive it. Repo: `~/stable-audio-3`
(Mac, M4 Max, MPS). Private remote `private` (stable-audio-3-private), branch `feat/mps-training`
— NEVER push `origin` (public fork).

## GOAL
A tool that (1) generates many DIVERSE, high-quality cKz melody loops to browse ("discovery"),
and (2) when the user picks a "cool" one, generates more loops with the SAME melody but varied
timbre ("refine"). UI first, then website, then VST.

## THE MODEL
- Base: Stable Audio 3 `medium-base` (NOT post-trained `medium`).
- LoRA: `lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt` ("aug-8000") — a pitch-augmentation
  retrain (46 cKz loops × ±2 semitone transposes → 204 clips). It beat the older model on NOVEL
  prompts (melody 4.0 vs 2.8) and is the production model. ~23.5k cumulative steps.
- Output: 44.1kHz stereo. Generation ≈ 1 min per 20s clip at 50 steps on M4 Max (MPS) — latency
  matters for a real-time/VST product; plan a persistent backend that loads the model ONCE.

## HOW TO GENERATE (exact API — see scripts/init_sweep.py & scripts/mode2_sweep.py for working code)
```python
from stable_audio_3 import StableAudioModel
model = StableAudioModel.from_pretrained("medium-base")
model.load_lora(["lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt"])
model.set_lora_strength(1.0)                       # ALWAYS 1.0 (lowering hurts; proven)
sr = model.model_config.get("sample_rate") or model.model.sample_rate
audio = model.generate(
    prompt="cKz! Mirage Gmin 120 bpm @crushed_keyz",
    duration=20, steps=50, cfg_scale=3.0, seed=7, sampler_type="dpmpp",
    init_audio=(sr, wav_tensor),     # (sr, tensor[channels, samples]); omit for pure t2m
    init_noise_level=0.75,           # σ — only used with init_audio
)
# audio[0].cpu() -> torchaudio.save(...)
```

## DISCOVERY MODE (different melodies) — LOCKED CONFIG
Engine = **foreign-init**: use a foreign melody loop as `init_audio`, repaint in cKz timbre.
- init loop: a **minor-key, DENSE** loop from the Cymatics packs at
  `/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops/` (dirs: Cymatics_Dragon_Platinum_Expansion,
  Cymatics_Gems_Vol_1_Trap_Melodies, Cymatics_2020_Melody_Collection, "Cymatics Gems Vol 6 - Future RnB
  Melodies"). Filenames carry key/bpm ("… 140 BPM D Min"). Skip MAJOR-key loops (0 keepers) and
  sparse loops (silence > 0.12).
- prompt: `cKz! <NovelTitle> <Key> <bpm> bpm @crushed_keyz`. Title = any evocative one-word noun
  (Mirage, Obsidian, Cascade, Solaris, Wraith, Nocturne, Voltage, Velvet, Lotus, Pulse, Cipher…)
  — interchangeable, NOT a diversity lever. Key = the loop's own key (match) OR a TRITONE away
  (both equal quality; tritone = a 2nd distinct melody from the same loop = free 2× diversity).
- `init_noise_level=0.75`, cfg 3.0, steps 50, dpmpp, strength 1.0, ~20s.
- Result: ~45% keepers, ~5% duds. Vary the LOOP (biggest), KEY (match+tritone), SEED.
- Reference impl + flags: `scripts/init_sweep.py --sigmas 0.75 --no-baseline --minor-only
  --dense-init --tritone --n-loops N --seeds ...`

## REFINE MODE (same melody, vary timbre) — LOCKED CONFIG
Engine = re-init from the PICKED clip itself (use the saved keeper wav as `init_audio`).
- init_audio = the picked clip; prompt = that clip's own prompt; **vary the SEED** for variations.
- `init_noise_level`: **0.45 = subtle** timbre tweak (top quality) · **0.75 = bold** new timbre
  (still high quality, melody preserved). AVOID 0.5–0.6 ("half-baked valley") and 0.8+ (degrades).
- strength 1.0, steps 50, cfg 3.0. Melody is preserved at all σ (init anchors it).
- Reference impl: `scripts/mode2_sweep.py`.

## WHAT CONTROLS DIVERSITY (the core mental model)
- **MELODY diversity** ← (1) the init loop [foreign = biggest, injects out-of-distribution structure]
  > (2) the seed > (3) the key (match/tritone). This is where variety lives.
- **TIMBRE diversity** ← the seed only, and it's NARROW: the LoRA pins everything to the cKz
  "crushed keys" signature. NO inference knob widens timbre (proven — see dead-ends). Timbre stays
  cKz by design; melody is the diversifiable axis. (Wider timbre would need a TRAINING change:
  timbre-descriptor retrain or more timbre-diverse data.)

## SETTINGS THAT ARE SETTLED — do NOT re-sweep (all tested):
- cfg = **3.0** (higher distorts with init). steps = **50** (150 identical). LoRA strength = **1.0**
  (lower hurts quality, no diversity gain). sampler = **dpmpp** (euler ok; rk4/pingpong bad).
- σ: 0.6–0.75 useful; 0.45 subtle / 0.75 bold; 0.8+ degrades; melody preserved across σ via init.
- prompt key: match OR tritone (equal); mild mismatch (±2) WORSE; no-key hurts melody. bpm: INERT
  with init (don't need to match). Foreign loops: MINOR-key + dense only.
- time-gated LoRA = DEAD (cKz timbre is set EARLY/high-σ and is holistic; removing LoRA early →
  base-model timbre). descriptor words (dark/bright) = weak/undirected (model not trained on them).
- Auto-scorer = DROPPED (biased toward memorized sound, can't see novelty) → judge BY EAR. The one
  safe automation is a DENSITY-only reject gate (`scripts/density_gate.py`) that drops dead-air duds
  (silence is the #1 quality killer: duds ~11% silence vs keepers ~2%).
- Memorization: prompting an exact TRAINING label reproduces that song; NOVEL titles → novel melodies.

## METHODOLOGY CAVEATS (learned the hard way)
- Generation IS deterministic: same seed + params → bit-identical audio (`torch.manual_seed` per call).
  So you CAN cache/reproduce by (prompt, seed, σ, init).
- Human RATINGS drift ~1.5★ session-to-session. Compare options WITHIN one rating session and
  include a known anchor clip; never compare absolute ratings across sessions.

## INFRASTRUCTURE
- Scripts (all in `scripts/`): `init_sweep.py` (discovery), `mode2_sweep.py`/`mode2_timbre.py` (refine),
  `ckz_factory.py` (pure-t2m factory), `augment_pitch.py`/`detect_key.py`/`fix_keys.py`/`import_new_loops.py`
  (data pipeline), `density_gate.py` (floor filter), `track_training.py`, `checkpoint_manifest.py`.
- Rater (reference UI to study): `sweep_rater/rate_server.py` + `sweep_rater/rate.html` — stdlib HTTP
  server, 3-axis stars (timbre/melody/novelty), per-segment rating, init-audio reference player,
  central `sweep_rater/ratings.json` keyed by repo-relative path.
- Training data: `data/crushed_keyz_lora` (68 cKz source loops, mp3+txt captions), `data/ckz_aug`
  (204 augmented). Generated audio + checkpoints are gitignored (19GB/5GB — local only).
- Memory/notes: see `HANDOFF_pitch_aug_retrain.md` and `lora_out/CHECKPOINTS.md`.

## SUGGESTED FIRST STEPS FOR THE UI
1. **Backend service** (FastAPI/Flask): load aug-8000 ONCE at startup; expose `/discover`
   (random minor+dense loop → match+tritone prompt → generate, return wav + metadata) and `/refine`
   (accept a clip id/wav + σ → re-init → generate). Keep the model resident (load is slow).
2. **Frontend**: a "Generate" feed of diverse clips (waveform + play + keep/discard); clicking
   "More like this" on a kept clip calls /refine (σ0.45 subtle / σ0.75 bold toggle). Mirror the
   rater's UX (sweep_rater/rate.html is a good reference).
3. Pre-index the foreign loop library (parse key/bpm, filter minor + dense) once into a manifest the
   backend samples from. `scripts/init_sweep.py:select_loops` already does this selection logic.
4. Latency plan: M4 Max ≈ 1 min/clip. For web/VST, consider a GPU backend or batching; VST will need
   a backend service (the model is PyTorch — not embeddable in-process; a local server or cloud API).

## ONE-LINE RECIPE SUMMARY
Discovery = aug-8000 · foreign minor+dense loop as init · `cKz! <noun> <key|tritone> <bpm> bpm
@crushed_keyz` · σ0.75 · cfg3 · 50 steps · dpmpp · strength1.0 · vary loop/key/seed → ~45% keepers.
Refine = same model · picked clip as init · σ0.45 (subtle) or 0.75 (bold) · vary seed → same melody,
new timbre. Judge by ear. Timbre is fixed-cKz; melody is the diverse axis.
