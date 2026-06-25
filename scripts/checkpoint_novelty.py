#!/usr/bin/env python3
"""Memorization vs novelty A/B across checkpoints.

The eff-2500 LoRA MEMORIZED its training songs (ACTiVE Cmin = 26% of keepers).
Earlier checkpoints have memorized less, so they may give NOVEL melodies (at some
timbre cost). The scorer can't judge this (it rewards the memorized sound) — you
must LISTEN. This generates the most-memorized exact labels across several
checkpoints × seeds into one folder; compare same title+seed across steps:
does the earlier checkpoint give a *different* (novel) melody that still sounds cKz?

    uv run python scripts/checkpoint_novelty.py
"""
from __future__ import annotations
import argparse, os, re, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]

def find_label(titles, labels_dir):
    out = {}
    for f in sorted(os.listdir(labels_dir)):
        if not f.endswith(".txt"): continue
        stem = os.path.splitext(f)[0]
        for ti in titles:
            if ti.lower() in stem.lower() and ti not in out:
                out[ti] = stem
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=str(REPO/"lora_out/ckz_melody_capped"))
    ap.add_argument("--steps", default="500,1000,1500,2000,2500")
    ap.add_argument("--titles", default="ACTiVE,AzUL,ICE")  # memorized training songs
    ap.add_argument("--seeds", default="7,123")
    ap.add_argument("--out", default=str(REPO/"sweep_rater/campaigns/checkpoint_novelty"))
    ap.add_argument("--model", default="medium-base")
    args = ap.parse_args()

    labels = find_label(args.titles.split(","), str(REPO/"data/crushed_keyz_lora"))
    steps = [int(s) for s in args.steps.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    ckpts = {}
    for s in steps:
        hits = [f for f in os.listdir(args.ckpt_dir) if re.search(rf"step={s}\b", f) or f.endswith(f"step={s}.ckpt")]
        if hits: ckpts[s] = os.path.join(args.ckpt_dir, hits[0])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"labels: {labels}")
    print(f"checkpoints: {sorted(ckpts)}  seeds: {seeds}")
    print(f"total: {len(labels)*len(ckpts)*len(seeds)} clips → {out}\n")

    import torch, torchaudio  # noqa
    from stable_audio_3 import StableAudioModel
    t0 = time.time(); n = 0
    for step in sorted(ckpts):
        print(f"--- loading checkpoint step {step} ---")
        model = StableAudioModel.from_pretrained(args.model)
        model.load_lora([ckpts[step]])
        sr = model.model_config.get("sample_rate") or model.model.sample_rate
        model.set_lora_strength(1.0)
        for ti, prompt in labels.items():
            for seed in seeds:
                p = out / f"step{step:04d}__{ti}__seed{seed}.wav"
                if p.exists(): continue
                try:
                    a = model.generate(prompt=prompt, duration=20, steps=50,
                                       cfg_scale=3.0, seed=seed, sampler_type="dpmpp")
                    torchaudio.save(str(p), a[0].cpu(), sr); n += 1
                    print(f"   step{step} {ti} seed{seed}")
                except Exception as e:
                    print(f"   FAIL step{step} {ti} s{seed}: {e}")
        del model
        try:
            import gc; gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
        except Exception: pass
    print(f"\nDone: {n} clips in {(time.time()-t0)/60:.1f}m → {out}")
    print(f"LISTEN (sort by name → same title+seed grouped across steps):")
    print(f"  python sweep_rater/rate_server.py --folder {out} --port 8799")
    print("Compare step0500 vs step2500 for each title+seed: is the earlier one a")
    print("DIFFERENT melody (novel) that still sounds cKz? If yes → earlier ckpt = novelty.")

if __name__ == "__main__":
    main()
