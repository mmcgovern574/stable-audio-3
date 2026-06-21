#!/usr/bin/env python3
"""Objective checkpoint eval — does a retrain actually raise the floor?

Generates a FIXED pure-text-to-music exact-label seed sweep from a given LoRA
checkpoint (no init), scores every clip with the auto-scorer (trained on the
1600+ ratings), and reports predicted keeper-rate + DENSITY (the quality signal
we found: dense/dark = good, sparse = the #1 killer). The label×seed set is
deterministic, so running this on two checkpoints is directly comparable.

    # baseline
    uv run python scripts/eval_checkpoint.py --ckpt ./lora_out/ckz_melody_capped/epoch=54-step=2500.ckpt
    # candidate
    uv run python scripts/eval_checkpoint.py --ckpt ./lora_out/ckz_dense_v2/<ckpt>.ckpt
    # then compare the two printed summaries.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sweep_rater/scorer"))
from extract_features import feats  # numpy DSP features (no torch)

FEAT = REPO / "sweep_rater/scorer/features.jsonl"


def norm_key(s):
    s = s.lower().replace("sharp", "#")
    m = re.search(r"([a-g])(#?)\s*min", s)
    return (m.group(1).upper()+m.group(2)+"min") if m else None


def exact_labels(labels_dir, n):
    """Deterministic spread of real cKz labels across keys (caption stems)."""
    rows = []
    for f in sorted(os.listdir(labels_dir)):
        if f.endswith(".txt") and norm_key(f):
            rows.append(os.path.splitext(f)[0])
    bykey = {}
    for s in rows: bykey.setdefault(norm_key(s), []).append(s)
    picked, keys, i = [], sorted(bykey), 0
    while len(picked) < n and any(bykey.values()):
        k = keys[i % len(keys)]
        if bykey[k]: picked.append(bykey[k].pop(0))
        i += 1
    return picked


def train_scorer():
    from sklearn.ensemble import RandomForestRegressor
    rows = [json.loads(l) for l in open(FEAT)]
    names = sorted(rows[0]["f"].keys())
    X = np.array([[r["f"][n] for n in names] for r in rows], float)
    yt = np.array([r["timbre"] for r in rows], float)
    ym = np.array([r["melody"] for r in rows], float)
    mt = RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X, yt)
    mm = RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X, ym)
    return names, mt, mm


def density(f):
    sil = float(f.get("silence", 0.0)); dyn = float(f.get("dyn", 18.0))
    return (1.0 - 0.5*sil) * (1.0 - min(max(dyn-22, 0)/40.0, 0.35))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--labels-dir", default=str(REPO/"data/crushed_keyz_lora"))
    ap.add_argument("--n-labels", type=int, default=8)
    ap.add_argument("--seeds", default="7,123,777,1234")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="folder for clips (default scratch by ckpt name)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    labels = exact_labels(args.labels_dir, args.n_labels)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out or REPO/f"sweep_rater/campaigns/eval_{Path(args.ckpt).parent.name}")
    print(f"checkpoint: {args.ckpt}")
    print(f"eval set: {len(labels)} exact labels × {len(seeds)} seeds = {len(labels)*len(seeds)} clips (pure t2m)")
    for L in labels: print(f"   «{L}»")
    if args.dry_run:
        print("\n--dry-run: would generate the above × seeds, then score."); return
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nloading {args.model} + LoRA + scorer ...")
    import torch, torchaudio  # noqa
    from stable_audio_3 import StableAudioModel
    model = StableAudioModel.from_pretrained(args.model)
    model.load_lora([str((REPO/args.ckpt).resolve()) if not os.path.isabs(args.ckpt) else args.ckpt])
    sr_out = model.model_config.get("sample_rate") or model.model.sample_rate
    model.set_lora_strength(1.0)
    names, mt, mm = train_scorer()

    res = []  # (predT, predM, density, silence, dyn)
    t0 = time.time()
    for i, L in enumerate(labels):
        for seed in seeds:
            p = out / f"{i:02d}_seed{seed}.wav"
            try:
                a = model.generate(prompt=L, duration=args.duration, steps=args.steps,
                                   cfg_scale=args.cfg, seed=seed, sampler_type="dpmpp")
                torchaudio.save(str(p), a[0].cpu(), sr_out)
                fe = feats(str(p))
                x = np.array([[fe[n] for n in names]], float)
                res.append((float(mt.predict(x)[0]), float(mm.predict(x)[0]),
                            density(fe), float(fe["silence"]), float(fe["dyn"])))
            except Exception as e:
                print(f"  FAIL {L[:30]} s{seed}: {type(e).__name__}: {e}")
    if not res: print("no clips."); return
    pt = np.array([r[0] for r in res]); pm = np.array([r[1] for r in res])
    de = np.array([r[2] for r in res]); si = np.array([r[3] for r in res]); dy = np.array([r[4] for r in res])
    keep = np.mean((pt>=4) & (pm>=4))
    print(f"\n===== CHECKPOINT SCORE  ({Path(args.ckpt).name}) =====")
    print(f"  predicted timbre   {pt.mean():.2f}")
    print(f"  predicted melody   {pm.mean():.2f}")
    print(f"  est. keeper-rate   {100*keep:.0f}%   (pred T>=4 & M>=4)")
    print(f"  mean DENSITY       {de.mean():.3f}   (1.0=dense; higher=better)")
    print(f"  mean silence frac  {si.mean():.3f}   (lower=better)")
    print(f"  mean dyn range dB  {dy.mean():.1f}    (lower=denser)")
    print(f"  generated in {(time.time()-t0)/60:.1f}m → {out}")
    print(f"  (rate these for ground truth: rate_server.py --folder {out})")


if __name__ == "__main__":
    main()
