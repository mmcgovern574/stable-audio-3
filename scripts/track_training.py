#!/usr/bin/env python3
"""Generate the fixed demo set for a checkpoint — PURE-EARS progress tracking.

Decision (2026-06-23): the DSP auto-scorer is DROPPED. It can't see novelty,
was trained on the old memorized-sound distribution, and scored a 42%-keeper
batch as 0% — it would bias us toward reproductions and hide good clips. Quality
and novelty are judged BY EAR on the 3-axis rater.

This just renders a FIXED, deterministic demo set (same novel prompts + 2 training
controls × same seeds) on a checkpoint so checkpoints are directly comparable, then
you rate them. The only auto-number kept is DIVERSITY (acoustic spread among the
novel demos) — a scorer-free mode-collapse canary: if it FALLS as training
continues, the model is losing variety. It is NOT a quality judge.

Progress = compare your RATINGS across checkpoints (all land in ratings.json under
the track_<label> campaign), watching novel-prompt melody + novelty rise while
diversity holds.

    uv run python scripts/track_training.py --ckpt lora_out/<run>/<ckpt>.ckpt --label <name>
    uv run python sweep_rater/rate_server.py --folder sweep_rater/campaigns/track_<label> --port 8799
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sweep_rater/scorer"))
from extract_features import feats  # numpy DSP features — used ONLY for the diversity canary

FEAT = REPO / "sweep_rater/scorer/features.jsonl"
NORM_CACHE = REPO / "sweep_rater/scorer/train_feats_cache.json"
TRACK_CSV = REPO / "sweep_rater/training_track.csv"

NOVEL = [
    "cKz! Mirage Gmin 120 bpm @crushed_keyz",
    "cKz! Obsidian D#min 130 bpm @crushed_keyz",
    "cKz! Cascade F#min 135 bpm @crushed_keyz",
    "cKz! Wraith Bmin 140 bpm @crushed_keyz",
    "cKz! Nocturne Amin 95 bpm @crushed_keyz",
    "cKz! Voltage C#min 150 bpm @crushed_keyz",
]
TRAIN_TITLES = ["ACTiVE", "AzUL"]   # memorization controls — rate, expect low novelty
SEEDS = [7, 123]


def find_stem(title):
    for f in sorted(os.listdir(REPO / "data/crushed_keyz_lora")):
        if f.endswith(".txt") and title.lower() in f.lower():
            return os.path.splitext(f)[0]
    return None


def feat_names():
    return sorted(json.loads(open(FEAT).readline())["f"].keys())


def norm_stats(names):
    """Per-feature mean/std (from the 68 training loops) for a STABLE, cross-checkpoint
    diversity normalization. Cached. Scorer-free — just feature statistics."""
    if NORM_CACHE.exists():
        d = json.load(open(NORM_CACHE))
        if d.get("names") == names:
            return np.array(d["mean"]), np.array(d["std"])
    src = REPO / "data/crushed_keyz_lora"
    X = []
    for p in src.iterdir():
        if p.suffix.lower() in (".mp3", ".wav", ".flac"):
            try: X.append([feats(str(p))[n] for n in names])
            except Exception: pass
    X = np.array(X, float); mean = X.mean(0); std = X.std(0) + 1e-9
    json.dump({"names": names, "mean": mean.tolist(), "std": std.tolist()}, open(NORM_CACHE, "w"))
    return mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    label = args.label or "__".join(Path(args.ckpt).parts[-2:]).replace(".ckpt", "")
    out = Path(args.out or REPO / f"sweep_rater/campaigns/track_{label.replace('/','_')}")
    train_stems = {t: find_stem(t) for t in TRAIN_TITLES}
    jobs = [("novel", p) for p in NOVEL] + [("train", train_stems[t]) for t in TRAIN_TITLES if train_stems[t]]
    print(f"checkpoint: {args.ckpt}")
    print(f"demo set: {len(NOVEL)} novel + {sum(1 for t in train_stems.values() if t)} train-control"
          f" × {len(SEEDS)} seeds = {len(jobs)*len(SEEDS)} clips  -> {out}")
    if args.dry_run:
        for kind, p in jobs: print(f"   [{kind:5}] «{p[:52]}»")
        print("\n--dry-run."); return
    out.mkdir(parents=True, exist_ok=True)

    names = feat_names(); mean, std = norm_stats(names)
    import torch, torchaudio  # noqa
    from stable_audio_3 import StableAudioModel
    print("loading model + LoRA ...")
    model = StableAudioModel.from_pretrained(args.model)
    model.load_lora([str((REPO / args.ckpt).resolve())])
    model.set_lora_strength(1.0)
    sr = model.model_config.get("sample_rate") or model.model.sample_rate

    novel_vecs = []
    t0 = time.time()
    for kind, prompt in jobs:
        for seed in SEEDS:
            p = out / f"{kind}_{re.sub(r'[^A-Za-z0-9]+','_',prompt)[:20]}__seed{seed}.wav"
            try:
                if not p.exists():
                    a = model.generate(prompt=prompt, duration=args.duration, steps=args.steps,
                                       cfg_scale=args.cfg, seed=seed, sampler_type="dpmpp")
                    torchaudio.save(str(p), a[0].cpu(), sr)
                if kind == "novel":
                    fe = feats(str(p))
                    novel_vecs.append((np.array([fe[n] for n in names], float) - mean) / std)
            except Exception as e:
                print(f"  FAIL {prompt[:24]} s{seed}: {type(e).__name__}: {e}")

    nv = np.array(novel_vecs)
    diversity = float(np.mean([np.sqrt(((nv[i]-nv[j])**2).sum())
                               for i in range(len(nv)) for j in range(i+1, len(nv))])) if len(nv) > 1 else 0.0
    prev = None
    if TRACK_CSV.exists():
        rows = list(csv.DictReader(open(TRACK_CSV))); prev = rows[-1] if rows else None
    write_header = not TRACK_CSV.exists()
    with open(TRACK_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "n_novel", "diversity", "ts"])
        if write_header: w.writeheader()
        w.writerow({"label": label, "n_novel": len(nv), "diversity": round(diversity, 2),
                    "ts": time.strftime("%Y-%m-%d %H:%M")})

    d = ""
    if prev and prev.get("diversity"):
        try: d = f"   (Δ {diversity-float(prev['diversity']):+.2f} vs {prev['label']})"
        except Exception: d = ""
    print(f"\n=== {label}  ({(time.time()-t0)/60:.1f}m) ===")
    print(f"  diversity (variety canary, NOT quality): {diversity:.2f}{d}")
    print(f"  log: {TRACK_CSV}")
    print(f"\nRATE BY EAR (3 axes) — this is the verdict:")
    print(f"  uv run python sweep_rater/rate_server.py --folder {out} --port 8799")
    print("Progress = novel melody+novelty RISING across checkpoints; STOP if diversity falls.")


if __name__ == "__main__":
    main()
