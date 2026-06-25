#!/usr/bin/env python3
"""Head-to-head: v1 @ step2000 (novelty sweet spot) vs aug @ 8000.

Reproduces the best-ever sample FIRST (v1-2000, exact ACTiVE label, seed7, with
checkpoint_novelty's exact settings: cfg 3.0, 50 steps, 20s, dpmpp), then runs
BOTH models across that prompt + other training labels + novel prompts so you can
A/B them with the timbre/melody/novelty axes.

Filenames sort so the two models sit back-to-back per prompt+seed.

    uv run python scripts/headtohead_novelty.py --dry-run
    uv run python scripts/headtohead_novelty.py
"""
from __future__ import annotations
import argparse, os, re, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# v1-2000 listed FIRST so it loads first; the ACTiVE+seed7 repro is generated first.
MODELS = {
    "v1-2000":  "lora_out/ckz_melody_capped/epoch=43-step=2000.ckpt",
    "aug-8000": "lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt",
}

def find_stem(title):
    for f in sorted(os.listdir(REPO / "data/crushed_keyz_lora")):
        if f.endswith(".txt") and title.lower() in f.lower():
            return os.path.splitext(f)[0]
    return None

# training labels (resolved to exact captions) + novel prompts
TRAIN_TITLES = ["ACTiVE", "AzUL", "ICE"]
NOVEL_PROMPTS = [
    "cKz! Mirage Gmin 120 bpm @crushed_keyz",      # aug won this in last A/B
    "cKz! Obsidian D#min 130 bpm @crushed_keyz",   # aug FIRE last A/B
    "cKz! Riptide Bmin 160 bpm @crushed_keyz",     # v1 won this last A/B
]
SEEDS = [7, 123]
REPRO = ("ACTiVE", 7)   # the exact sample to make first


def slug(p):
    m = re.search(r"cKz!\s*(.+?)\s+[A-G]#?\s*(?:min|maj)?\b", p) or re.search(r"\b([A-Za-z][\w ]+?)\s+[A-G]#?(?:min|maj)", p)
    t = (m.group(1) if m else p[:18]).strip().replace(" ", "_")
    k = re.search(r"\b([A-G]#?(?:min|maj))\b", p)
    return (t[:16] + (("_" + k.group(1)) if k else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "sweep_rater/campaigns/headtohead_v1_2000_vs_aug"))
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    train = {t: find_stem(t) for t in TRAIN_TITLES}
    missing = [t for t, s in train.items() if not s]
    if missing:
        print("WARNING missing training stems:", missing)
    # ordered prompt list: training first (ACTiVE leads), then novel
    prompts = []
    for i, t in enumerate(TRAIN_TITLES):
        if train[t]:
            prompts.append((f"T{i}_{t}", train[t], "train"))
    for j, p in enumerate(NOVEL_PROMPTS):
        prompts.append((f"N{j}_{slug(p)}", p, "novel"))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"out: {out}")
    print(f"settings: cfg {args.cfg}, steps {args.steps}, {args.duration}s, dpmpp, strength 1.0")
    print(f"models: {list(MODELS)}   prompts: {len(prompts)}   seeds: {SEEDS}")
    print(f"total: {len(MODELS)*len(prompts)*len(SEEDS)} clips\n")
    for tag, p, kind in prompts:
        print(f"   [{kind:5}] {tag:18} «{p[:48]}»")
    print(f"\nFIRST (reproduction): v1-2000 · «{train['ACTiVE']}» · seed 7")
    if args.dry_run:
        print("\n--dry-run: no generation."); return

    import torch, torchaudio  # noqa
    from stable_audio_3 import StableAudioModel
    t0 = time.time(); n = 0
    for mname, ckpt in MODELS.items():
        print(f"\n--- loading {mname}: {ckpt} ---")
        model = StableAudioModel.from_pretrained(args.model)
        model.load_lora([str((REPO / ckpt).resolve())])
        model.set_lora_strength(1.0)
        sr = model.model_config.get("sample_rate") or model.model.sample_rate
        # build jobs; for v1-2000 force the ACTiVE+seed7 reproduction first
        jobs = [(tag, p, kind, seed) for (tag, p, kind) in prompts for seed in SEEDS]
        if mname == "v1-2000":
            jobs.sort(key=lambda j: 0 if (j[0].endswith("ACTiVE") and j[3] == REPRO[1]) else 1)
        for tag, p, kind, seed in jobs:
            path = out / f"{tag}__seed{seed}__{mname}.wav"
            if path.exists():
                n += 1; continue
            try:
                a = model.generate(prompt=p, duration=args.duration, steps=args.steps,
                                   cfg_scale=args.cfg, seed=seed, sampler_type="dpmpp")
                torchaudio.save(str(path), a[0].cpu(), sr); n += 1
                print(f"   {mname} {tag} seed{seed}")
            except Exception as e:
                print(f"   FAIL {mname} {tag} s{seed}: {type(e).__name__}: {e}")
        del model
        try:
            import gc; gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
        except Exception: pass
    print(f"\nDone: {n} clips in {(time.time()-t0)/60:.1f}m -> {out}")
    print(f"Rate:  uv run python sweep_rater/rate_server.py --folder {out} --port 8799")
    print("First check: does v1-2000 · ACTiVE · seed7 match your best-ever sample?")


if __name__ == "__main__":
    main()
