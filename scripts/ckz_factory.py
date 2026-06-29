#!/usr/bin/env python3
"""cKz loop factory — over-generate NOVEL melodies on aug-8000, then pick the best.

The winning recipe (from the A/Bs): a novel evocative one-word TITLE in the exact
cKz caption format `cKz! <Title> <Key> <bpm> bpm @crushed_keyz`, on the augmented
model — which turns each novel title into a fresh melody with the cKz timbre.

This sweeps ~36 novel titles spread across ALL 12 minor keys (the coverage the
pitch-augmentation added) × varied bpm × seeds, into one campaign folder. Then run
pick_best.py to rank them and surface a keeper shortlist to audition + rate.

    uv run python scripts/ckz_factory.py --dry-run
    uv run python scripts/ckz_factory.py
    # then:
    uv run python sweep_rater/scorer/pick_best.py --folder sweep_rater/campaigns/ckz_factory_v1 --top 30
    uv run python sweep_rater/rate_server.py --folder sweep_rater/campaigns/ckz_factory_v1/picks --port 8799
"""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUG = "lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt"

TITLES = ["Mirage","Obsidian","Voltage","Riptide","Phantom","Ember","Cascade",
          "Velvet","Tundra","Solaris","Nocturne","Skyline","Eclipse","Glacier",
          "Inferno","Onyx","Cobalt","Specter","Halcyon","Maelstrom","Nimbus",
          "Vortex","Abyss","Citadel","Quartz","Zephyr","Cinder","Fathom","Harbor",
          "Lattice","Meridian","Opal","Pyre","Sable","Umbra","Wraith"]
KEYS = ["Cmin","C#min","Dmin","D#min","Emin","Fmin","F#min","Gmin","G#min","Amin","A#min","Bmin"]
BPMS = [90,100,110,120,125,130,135,140,145,150,155,160]


def build(titles, style="cKz!", trigger="@crushed_keyz"):
    rows = []
    for i, t in enumerate(titles):
        key = KEYS[i % len(KEYS)]
        bpm = BPMS[(i * 5) % len(BPMS)]      # stride decorrelates key<->bpm
        rows.append((t, key, bpm, f"{style} {t} {key} {bpm} bpm {trigger}"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "sweep_rater/campaigns/ckz_factory_v1"))
    ap.add_argument("--ckpt", default=AUG)
    ap.add_argument("--seeds", default="7,123")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--titles-file", default=None, help="one title per line (default: built-in 36)")
    ap.add_argument("--style-prefix", default="cKz!",
                    help="producer style token that OPENS every prompt (must match training captions)")
    ap.add_argument("--trigger", default="@crushed_keyz",
                    help="producer trigger tag that CLOSES every prompt (must match training captions)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    titles = [l.strip() for l in open(args.titles_file) if l.strip()] if args.titles_file else TITLES
    rows = build(titles, args.style_prefix, args.trigger)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = []  # (prompt, seed, title, key, bpm, path)
    for i, (t, key, bpm, p) in enumerate(rows):
        for seed in seeds:
            fn = f"{i:02d}_{t}_{key}_{bpm}__seed{seed}.wav"
            jobs.append((p, seed, t, key, bpm, out / fn))
    print(f"model: {args.ckpt}")
    print(f"titles: {len(rows)}  keys: all 12 minor  seeds: {seeds}")
    print(f"settings: cfg {args.cfg}, {args.steps} steps, {args.duration}s, dpmpp")
    print(f"total clips: {len(jobs)}  -> {out}\n")
    for t, key, bpm, p in rows[:6]:
        print(f"   «{p}»")
    print("   ...")
    if args.dry_run:
        print(f"\n--dry-run: would generate {len(jobs)} clips."); return

    # write index BEFORE generating so the picker can map prompts even mid-run
    write_index(out, jobs)

    import torch, torchaudio  # noqa
    from stable_audio_3 import StableAudioModel
    print(f"loading aug model ...")
    model = StableAudioModel.from_pretrained(args.model)
    model.load_lora([str((REPO / args.ckpt).resolve())])
    model.set_lora_strength(1.0)
    sr = model.model_config.get("sample_rate") or model.model.sample_rate
    t0 = time.time(); n = 0
    for p, seed, t, key, bpm, path in jobs:
        if path.exists():
            n += 1; continue
        try:
            a = model.generate(prompt=p, duration=args.duration, steps=args.steps,
                               cfg_scale=args.cfg, seed=seed, sampler_type="dpmpp")
            torchaudio.save(str(path), a[0].cpu(), sr); n += 1
            if n % 8 == 0: print(f"   ...{n}/{len(jobs)}  ({t} {key})")
        except Exception as e:
            print(f"   FAIL {t} {key} s{seed}: {type(e).__name__}: {e}")
    print(f"\nDone: {n}/{len(jobs)} in {(time.time()-t0)/60:.1f}m -> {out}")
    print(f"\nNext — rank & shortlist:")
    print(f"  uv run python sweep_rater/scorer/pick_best.py --folder {out} --top 30")
    print(f"  uv run python sweep_rater/rate_server.py --folder {out}/picks --port 8799")


def write_index(out, jobs):
    with open(out / "library_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file","key","bpm","mood","title","seed",
                                          "prompt","init_loop","init_path"])
        w.writeheader()
        for p, seed, t, key, bpm, path in jobs:
            w.writerow({"file": path.name, "key": key, "bpm": bpm, "mood": "",
                        "title": t, "seed": seed, "prompt": p,
                        "init_loop": "", "init_path": ""})


if __name__ == "__main__":
    main()
