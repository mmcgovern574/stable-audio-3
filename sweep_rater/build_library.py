#!/usr/bin/env python3
"""Crushed Keyz LIBRARY generator.

Generates a large, varied library of cKz melodies using the init-audio novelty
engine, applying everything we learned:

  * init_audio + FULL LoRA (the gate is dead; full LoRA keeps timbre)
  * DENSE full melody loops as init (stems are too sparse)
  * init_noise_level 0.6 (anchors dense loops; higher goes sparse)
  * LoRA strength ~0.85 (lower-helps-with-init)
  * KEY-MATCHED prompts: each loop's key+bpm are parsed from its filename and
    written into an exact cKz training-format prompt -> unlocks any key
  * varied cKz titles for prompt variety
  * dpmpp / cfg 3 / steps 50

Variety comes from three timbre-preserving axes: loop x prompt-title x seed.

Reads a manifest (sweep_rater/library_loops.json) of
  [{"path": <abs wav>, "key": "Cmin", "bpm": 130, "mood": "dark"}, ...]

Resumable: skips clips whose output already exists. Saves to
sweep_rater/campaigns/ckz_library/ — rate/browse with rate_server.py.

Usage:
    uv run python sweep_rater/build_library.py
    uv run python sweep_rater/build_library.py --prompts-per-loop 4 --seeds 7,123,777
    uv run python sweep_rater/build_library.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Default title vocabulary (real cKz training titles) for prompt variety. Override
# per-artist with --titles-file (one title per line). Titles are just flavor words.
DEFAULT_TITLES = [
    "ACTiVE", "ICE", "AzUL", "JUNGLE", "Panda", "CoSmoS", "Midnight", "Ember",
    "Velvet", "Mirage", "Depths", "Twilight", "Dusk", "Halo", "Crystal",
    "StarLight", "Mars", "JuPitER", "Veil", "Echo", "Pressure", "POWER",
    "Nasty", "Coldhearted", "Scenic", "LowKey", "Emotions", "Soulful",
    "FlaMEnCO", "Glassland", "Steppa", "TrapSoul", "Cry", "Coupe", "BANDS",
    "LATE", "CANOPY", "Halo", "Rent_Free", "DreamDoor",
]


def build_prompts(loop, n, rng, titles, prefix, trigger):
    """n key+bpm-matched prompts with distinct titles for one loop.

    Prompt format mirrors the exact training filename strings, e.g.
    'cKz! AzUL Fmin 140 bpm @crushed_keyz' (prefix='cKz!', trigger='@crushed_keyz').
    Deterministic given the rng seed. Swap prefix/trigger/titles per artist.
    """
    chosen = rng.sample(titles, min(n, len(titles)))
    key, bpm = loop["key"], loop["bpm"]
    return [(t, " ".join(x for x in (prefix, t, key, f"{bpm} bpm", trigger) if x))
            for t in chosen]


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9!#@.]+", "_", str(s)).strip("_")[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(REPO_ROOT / "sweep_rater/library_loops.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "sweep_rater/campaigns/ckz_library"))
    ap.add_argument("--lora-ckpt", default="./lora_out/ckz_melody_capped/epoch=54-step=2500.ckpt")
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--prompts-per-loop", type=int, default=3)
    ap.add_argument("--seeds", default="7,123")
    ap.add_argument("--noise", type=float, default=0.6)
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--max-loops", type=int, default=0, help="cap loops (0=all)")
    ap.add_argument("--trigger", default="@crushed_keyz", help="style trigger tag in every prompt")
    ap.add_argument("--style-prefix", default="cKz!", help="prompt prefix before the title")
    ap.add_argument("--titles-file", default=None, help="file of title words (one per line); default = built-in list")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    titles = DEFAULT_TITLES
    if args.titles_file:
        titles = [ln.strip() for ln in open(args.titles_file) if ln.strip()]
        if not titles:
            sys.exit(f"no titles in {args.titles_file}")

    loops = json.load(open(args.manifest))
    if args.max_loops:
        loops = loops[: args.max_loops]
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    rng = random.Random(42)

    # Build the full job list: (loop, title, prompt, seed). Prompts are chosen
    # deterministically per loop so re-runs line up (resumability).
    jobs = []
    for loop in loops:
        for title, prompt in build_prompts(loop, args.prompts_per_loop, rng,
                                           titles, args.style_prefix, args.trigger):
            for seed in seeds:
                jobs.append((loop, title, prompt, seed))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def out_path(i, loop, title, seed):
        loopstem = sanitize(Path(loop["path"]).stem)[:26]
        fn = f"{i:04d}__{loop['key']}_{loop['bpm']}_{loop['mood']}__{sanitize(title)}__loop-{loopstem}__seed{seed}.wav"
        return out_dir / fn

    def write_index():
        import csv as _csv
        with open(out_dir / "library_index.csv", "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["file", "key", "bpm", "mood", "title", "seed", "prompt", "init_loop", "init_path"])
            for i, (loop, title, prompt, seed) in enumerate(jobs, 1):
                w.writerow([out_path(i, loop, title, seed).name, loop["key"], loop["bpm"],
                            loop["mood"], title, seed, prompt, Path(loop["path"]).name, loop["path"]])

    todo = []
    for i, (loop, title, prompt, seed) in enumerate(jobs, 1):
        p = out_path(i, loop, title, seed)
        if p.exists() and p.stat().st_size > 0:
            continue
        todo.append((i, loop, title, prompt, seed, p))

    print(f"Library generator")
    print(f"  loops:            {len(loops)}")
    print(f"  prompts/loop:     {args.prompts_per_loop}   seeds: {seeds}")
    print(f"  total clips:      {len(jobs)}   (to generate now: {len(todo)})")
    print(f"  recipe:           init_noise {args.noise} | LoRA strength {args.strength} | "
          f"cfg {args.cfg} | {args.steps} steps | dpmpp | FULL LoRA")
    print(f"  out:              {out_dir}")
    if args.dry_run:
        for i, loop, title, prompt, seed, p in todo[:12]:
            print(f"  [{i:04d}] {loop['key']}/{loop['bpm']} s{seed}  «{prompt}»  <= {Path(loop['path']).name[:40]}")
        print(f"\n--dry-run: {len(todo)} clips would be generated. ~{len(todo)*25/60:.0f} min est.")
        print(f"Rate later: python sweep_rater/rate_server.py --folder {out_dir}")
        return
    if not todo:
        print("\nAll clips already exist. Nothing to do.")
        return

    # Write the ground-truth index only on a real run (never in --dry-run, so a
    # preview can't clobber an existing folder's index).
    write_index()
    print(f"  index: {out_dir / 'library_index.csv'}")

    print(f"\nLoading model ({args.model}) + LoRA — once...")
    t0 = time.time()
    import torch  # noqa
    import torchaudio
    from stable_audio_3 import StableAudioModel

    model = StableAudioModel.from_pretrained(args.model)
    ckpt = (REPO_ROOT / args.lora_ckpt).resolve()
    if not ckpt.exists():
        sys.exit(f"LoRA checkpoint not found: {ckpt}")
    model.load_lora([str(ckpt)])
    sample_rate = model.model_config.get("sample_rate") or model.model.sample_rate
    model.set_lora_strength(args.strength)  # constant across the whole library
    print(f"Model ready in {time.time()-t0:.1f}s. sample_rate={sample_rate}, "
          f"LoRA strength fixed at {args.strength}.\n")

    # init_audio cache: load each loop once (reused across its prompts/seeds).
    init_cache: dict[str, tuple] = {}

    def load_init(path):
        if path not in init_cache:
            wav, sr = torchaudio.load(path)
            init_cache[path] = (int(sr), wav)
        return init_cache[path]

    gen_start = time.time()
    done = fail = 0
    for n, (i, loop, title, prompt, seed, p) in enumerate(todo, 1):
        try:
            sr, wav = load_init(loop["path"])
        except Exception as e:
            print(f"[{n}/{len(todo)}] SKIP init load fail ({loop['path']}): {e}")
            fail += 1
            continue
        elapsed = time.time() - gen_start
        eta = (elapsed / max(n - 1, 1)) * (len(todo) - n + 1) if n > 1 else 0
        print(f"[{n:>4}/{len(todo)}] {loop['key']}/{loop['bpm']} {loop['mood']:7} "
              f"s{seed}  «{prompt[:48]}»   ETA {eta/60:4.0f}m")
        # Match generation duration to the init loop's length (capped at --duration).
        # If we generate longer than the loop, the back half anchors to padded
        # silence and the output goes silent after the loop ends (the "silent
        # after Ns" defect). Matching duration keeps every clip fully audible.
        loop_dur = wav.shape[-1] / sr
        gen_dur = round(min(loop_dur, args.duration), 1)
        try:
            audio = model.generate(
                prompt=prompt,
                duration=gen_dur,
                steps=args.steps,
                cfg_scale=args.cfg,
                seed=seed,
                init_audio=(sr, wav),
                init_noise_level=args.noise,
                sampler_type="dpmpp",
            )
            torchaudio.save(str(p), audio[0].cpu(), sample_rate)
            done += 1
        except Exception as e:
            print(f"          FAILED: {type(e).__name__}: {e}")
            fail += 1

    dt = time.time() - gen_start
    print(f"\nDone: {done} generated, {fail} failed in {dt/60:.1f} min.")
    print(f"Library: {out_dir}")
    print(f"Browse/rate: python sweep_rater/rate_server.py --folder {out_dir} --port 8799")


if __name__ == "__main__":
    main()
