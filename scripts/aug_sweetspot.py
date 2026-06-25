#!/usr/bin/env python3
"""Find the aug model's NOVELTY sweet-spot checkpoint.

Novelty peaks early in training and decays as the model memorizes (v1's best-ever
clip came from step2000, not the converged 2500). The aug model was trained in
sequential runs (v1 -> cont -> long -> long2); cumulative steps span ~1k to ~23k.
We're currently running the LATEST (aug-8000 = long2, ~23k cum) — possibly past
peak. This sweeps 5 checkpoints across the whole arc on the SAME novel prompts
(+ one training label as a memorization control), so you can rate where novelty
is highest while quality still holds.

Filenames sort prompt-first, then checkpoint stage 1->5 (early->late), so each
prompt's checkpoints play back-to-back in training order — you hear the novelty
trajectory directly.

    uv run python scripts/aug_sweetspot.py --dry-run
    uv run python scripts/aug_sweetspot.py
"""
from __future__ import annotations
import argparse, os, re, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# (stage, short tag, path) ordered EARLY -> LATE by cumulative training (mtime order)
CHECKPOINTS = [
    ("1", "v1-1k",    "lora_out/ckz_aug_v1/epoch=4-step=1000.ckpt"),
    ("2", "v1-3k",    "lora_out/ckz_aug_v1/epoch=14-step=3000.ckpt"),
    ("3", "long-4k",  "lora_out/ckz_aug_v1_long/epoch=19-step=4000.ckpt"),
    ("4", "long-10k", "lora_out/ckz_aug_v1_long/epoch=49-step=10000.ckpt"),  # ~the old "aug-15500"
    ("5", "long2-8k", "lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt"),  # current aug-8000
]

NOVEL = [
    "cKz! Mirage Gmin 120 bpm @crushed_keyz",
    "cKz! Obsidian D#min 130 bpm @crushed_keyz",
    "cKz! Cascade F#min 135 bpm @crushed_keyz",
    "cKz! Wraith Bmin 140 bpm @crushed_keyz",
]
TRAIN_TITLE = "ACTiVE"   # memorization control — should get MORE memorized at late stages
SEEDS = [7, 123]


def find_stem(title):
    for f in sorted(os.listdir(REPO / "data/crushed_keyz_lora")):
        if f.endswith(".txt") and title.lower() in f.lower():
            return os.path.splitext(f)[0]
    return None


def slug(p):
    m = re.search(r"cKz!\s*([A-Za-z]+)", p)
    t = (m.group(1) if m else p[:10])
    k = re.search(r"\b([A-G]#?min)\b", p)
    return f"{t}_{k.group(1) if k else 'X'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "sweep_rater/campaigns/aug_sweetspot"))
    ap.add_argument("--seeds", default="7,123")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    train_stem = find_stem(TRAIN_TITLE)
    prompts = [(f"{i:02d}", slug(p), p, "novel") for i, p in enumerate(NOVEL)]
    if train_stem:
        prompts.append((f"{len(NOVEL):02d}", f"TRAIN_{TRAIN_TITLE}", train_stem, "train"))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints (early->late):")
    for st, tag, ck in CHECKPOINTS:
        print(f"   stage{st} {tag:9} {ck}  {'OK' if (REPO/ck).exists() else 'MISSING!'}")
    print(f"prompts: {len(prompts)} ({len(NOVEL)} novel + {'1 train' if train_stem else '0 train'})  seeds: {seeds}")
    print(f"total: {len(CHECKPOINTS)*len(prompts)*len(seeds)} clips -> {out}\n")
    if args.dry_run:
        print("--dry-run: no generation."); return

    import torch, torchaudio  # noqa
    from stable_audio_3 import StableAudioModel
    t0 = time.time(); n = 0
    for st, tag, ck in CHECKPOINTS:
        ckp = REPO / ck
        if not ckp.exists():
            print(f"SKIP missing {ck}"); continue
        print(f"\n--- loading stage{st} {tag} ---")
        model = StableAudioModel.from_pretrained(args.model)
        model.load_lora([str(ckp.resolve())])
        model.set_lora_strength(1.0)
        sr = model.model_config.get("sample_rate") or model.model.sample_rate
        for pidx, sl, prompt, kind in prompts:
            for seed in seeds:
                path = out / f"{pidx}_{sl}__stage{st}_{tag}__seed{seed}.wav"
                if path.exists():
                    n += 1; continue
                try:
                    a = model.generate(prompt=prompt, duration=args.duration, steps=args.steps,
                                       cfg_scale=args.cfg, seed=seed, sampler_type="dpmpp")
                    torchaudio.save(str(path), a[0].cpu(), sr); n += 1
                except Exception as e:
                    print(f"   FAIL {tag} {sl} s{seed}: {type(e).__name__}: {e}")
        del model
        try:
            import gc; gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
        except Exception: pass
        print(f"   stage{st} done ({n} total)")
    print(f"\nDone: {n} clips in {(time.time()-t0)/60:.1f}m -> {out}")
    print(f"Rate:  uv run python sweep_rater/rate_server.py --folder {out} --port 8799")
    print("For each prompt, listen stage1->5: where is NOVELTY highest with quality intact?")
    print("Watch the ACTiVE control — it should sound MORE memorized at late stages.")


if __name__ == "__main__":
    main()
