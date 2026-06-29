# SoundSauce — Scalable Generation → Catalog Pipeline

How loops flow from the model to the live product, structured so it scales to many
campaigns, models, and producers — and so the `#`-style corruption can never happen again.

## The one principle that fixes everything
**Decouple identity, storage, and display.** Today the *filename* is all three at once
(`0001__A#min_146_m__Cascade__loop-…__seed7.wav`), which is why a `#` in a key name
corrupted Supabase storage and why ratings/picks/catalog are fragile to renames.

Instead: **mint a UUID for each clip at generation time and carry it through every stage.**
- **Identity** = the UUID (never derived from title/key).
- **Storage key** = `<uuid>.mp3` — URL-safe forever (no `#`, spaces, unicode, or collisions, ever).
- **Display** = `title / key / bpm` — plain metadata you can rename anytime with zero storage impact.
- The pretty download name (`Cascade_A#min_146bpm_@crushed_keyz.mp3`) is generated at click time, as the app already does.

## Two data planes
- **Local "firehose"** — *everything* generated (incl. rejects), full params, all ratings.
  Stays on your machine (JSONL/CSV) for retraining, analysis, and re-picking. Cheap, unbounded.
- **Cloud catalog** — *only the keepers you ship*, in Supabase. Lean, fast, what the product reads.

Keepers carry their provenance into the cloud; rejects never leave the firehose.

## Cloud schema (Supabase)
```sql
producers   (id uuid pk, name, handle '@crushed_keyz', links jsonb, created_at)
models      (id uuid pk, producer_id→producers, name 'cKz-aug-8000',
             base_model 'stable-audio-3-medium', base_model_rev,
             lora_ckpt 'epoch=…-step=8000.ckpt', lora_sha256, ckpt_sha256,
             default_recipe jsonb {cfg,steps,sampler,strength,init_noise}, notes, created_at)
generations (id uuid pk, model_id→models, prompt, seed, engine 't2m|foreign-init',
             init_audio, init_audio_sha256, init_offset_sec, init_len_sec, init_noise,
             cfg, steps, sampler, strength, guidance, neg_prompt,
             duration_sec, sample_rate, dtype 'bf16', device, deterministic bool,
             code_commit, sa3_version, torch_version,
             provenance_quality 'logged|backfilled',   -- never lie about what was captured
             params jsonb, created_at)
samples     (id uuid pk = generations.id, generation_id→generations, producer_id→producers,
             title, key, bpm, duration_sec, sample_rate,
             storage_key '<id>.mp3' unique, audio_path,         -- decoupled from title
             output_sha256, peak_dbfs, lufs,                    -- integrity + loudness norm
             is_exclusive, status 'public|hidden|claimed|archived', created_at)
ratings     (id uuid pk, sample_id→samples, rater 'mike|scorer|user:<uid>',
             melody, novelty, timbre, notes, source 'human|model', created_at)
embeddings  (sample_id→samples pk, kind 'chroma|clap', vec, created_at)  -- dedup / similarity / “more like this”
```
Why the extra fields (you asked what else to save for reproducibility):
- **Hashes, not just names** — `ckpt_sha256`, `lora_sha256`, `init_audio_sha256`, `output_sha256`.
  Filenames lie (that's how the `#` bug hid); hashes don't. `output_sha256` also gives you
  instant exact-dup detection and would have caught the sharp-key collisions immediately.
- **Code + model versions** — `code_commit`, `sa3_version`, `torch_version`, `base_model_rev`.
  A seed only reproduces a loop against the *same code*; capture the commit so it survives refactors.
- **Determinism context** — `dtype`, `device`, `deterministic`. SA3 isn't guaranteed bit-identical
  across GPUs/drivers; record the environment so "close vs exact" is honest.
- **Loudness** — `peak_dbfs`, `lufs` for consistent playback levels across the catalog.
- **Embedding** — store the chroma/CLAP vector per loop; powers dedup, the similarity heatmap,
  "more like this," and marketplace search without re-analyzing audio every time.
The player keeps querying `samples` (one tiny change: `audio_path` now points at `<uuid>.mp3`).
`producers`/`models` is exactly the structure the marketplace needs later — built once, here.

## Per-clip provenance record (the new source of truth)
Your generators — `scripts/init_sweep.py` (discovery / foreign-init) and `scripts/mode2_sweep.py`
(refine) — write one JSONL line per clip to `generation_manifest.jsonl`
(replaces the filename-encoding + thin `library_index.csv`):
```json
{"id":"d290f1ee-…","campaign":"ckz_lib_2026-06-28","file":"d290f1ee-….wav",
 "model":{"name":"cKz-aug-8000","base":"stable-audio-3-medium","lora_ckpt":"…step=8000.ckpt"},
 "engine":"foreign-init","init_audio":"Cymatics - … 146 BPM A# Min.wav","init_audio_hash":"…",
 "seed":7,"prompt":"cKz! Cascade A#min 146 bpm @crushed_keyz","init_noise":0.6,
 "cfg":3,"steps":50,"sampler":"dpmpp","strength":1.0,
 "title":"Cascade","key":"A#min","bpm":146,"mood":"s75","duration_sec":7.9,
 "sample_rate":44100,"created_at":"2026-06-28T…"}
```
Everything needed to **reproduce the loop exactly** lives in one record, keyed by the UUID.

## The pipeline, stage by stage (current tool → scalable change)
0. **Registry (once per producer/model)** — insert a `producers` and `models` row; the model
   captures the checkpoint + validated recipe. Small one-time seed.
1. **Generate** — `init_sweep.py` / `mode2_sweep.py`, lightly modified: mint `id=uuid4()`, save
   audio as `<id>.wav`, append the full provenance line to `generation_manifest.jsonl`. *Recipe
   params become persisted per clip instead of just printed.* (No change to the validated recipe.)
2. **Rate** — `rate_server.py`/`ratings.json` keyed by **id** (the file is `<id>.wav`); store
   `{melody, novelty, timbre, notes, rater, ts}`.
3. **Auto-select** — `scorer/` features + `pick_best.py` keyed by **id** → `picks_ranked.csv`
   (id, score). Listen down the ranked list; mark ear-approved keepers (pure-ears final call).
4. **Import keepers** — new `import_keepers.py` (replaces `catalog.json` + `upload_catalog.py`):
   for each approved id → transcode `<id>.wav`→`<id>.mp3` → upload to `catalog/<id>.mp3`
   (URL-safe) → upsert `generations` + `samples` from the manifest + ratings. **Idempotent by id.**

## Adding new audio later
Run stages 1→4 on a new campaign. Import is an idempotent upsert by UUID, so re-runs are safe and
you never touch existing rows. No filename collisions are possible by construction.

## Migrating the current 69 loops = the first run of stage 4
Mint a UUID per existing local file, upload each as `<uuid>.mp3`, write `samples` + repoint
`audio_path`. Provenance is **complete** — the recipe was recovered from the generation sessions
and verified on disk, so these import as `provenance_quality='logged'`:
- from `catalog.json`: title, key, bpm, prompt, seed, init_loop, melody/novelty/timbre.
- recovered recipe: generator `scripts/init_sweep.py`; model aug-8000
  (`lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt`), base `medium-base`, strength 1.0;
  engine foreign-init; **σ0.75** (= the `s75` filename tag); cfg 3.0; steps 50; sampler dpmpp; seeds 7/123.
- from the filename: key-type `m`=match / `t`=tritone(+6).
- computed from the files: duration, sample_rate, output_sha256, peak/LUFS, embedding.

This **permanently fixes the sharp-key corruption** (no `#` ever reaches storage) and seeds the
schema in one pass. Keep all 69 — they're confirmed melodically distinct.

## Minimal code changes
- `init_sweep.py` / `mode2_sweep.py`: UUID + `<id>.wav` + `generation_manifest.jsonl` writer (~30 lines each, shared helper).
- `rate_server.py` / `scorer/*`: key on id (they already key on the file stem → now the stem *is* the id).
- new `import_keepers.py`: manifest + ratings → mp3 → Storage `<id>.mp3` → upsert samples/generations.
- new migration `0005_provenance.sql`: the 4 tables above (+ backfill-friendly columns on `samples`).
- frontend: none beyond `audio_path` now being `<uuid>.mp3`.
