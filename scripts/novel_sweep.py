#!/usr/bin/env python3
"""Novel-prompt A/B — does the augmented model give NOVEL keepers vs v1?

Exact-label evals only surface memorization (asking for a training song by name
returns that song). This instead prompts BOTH models with NOVEL titles the model
never trained on (Mirage, Voltage, Obsidian...), matching the training caption
FORMAT (cKz! <Title> <Key> <bpm> bpm @crushed_keyz) so the timbre still fires,
across varied keys/bpm and a couple of seeds. Pure text-to-music, no init.

Filenames sort so the same prompt+seed puts aug8k and v1 back-to-back for A/B
listening. Rate by ear for: (a) is the melody NOVEL (not a training song), and
(b) did the cKz timbre survive.

    uv run python scripts/novel_sweep.py --dry-run
    uv run python scripts/novel_sweep.py            # generates both models
    uv run python scripts/novel_sweep.py --aug-only
"""
from __future__ import annotations
import argparse, re, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Novel titles the model never saw, spread across keys + bpm (training format).
NOVEL_PROMPTS = [
    "cKz! Mirage Gmin 120 bpm @crushed_keyz",
    "cKz! Voltage C#min 150 bpm @crushed_keyz",
    "cKz! Skyline Fmin 140 bpm @crushed_keyz",
    "cKz! Obsidian D#min 130 bpm @crushed_keyz",
    "cKz! Nocturne Amin 95 bpm @crushed_keyz",
    "cKz! Riptide Bmin 160 bpm @crushed_keyz",
    "cKz! Ember Emin 145 bpm @crushed_keyz",
    "cKz! Cascade F#min 110 bpm @crushed_keyz",
    "cKz! Phantom Dmin 155 bpm @crushed_keyz",
    "cKz! Velvet A#min 90 bpm @crushed_keyz",
    "cKz! Tundra Gmin 138 bpm @crushed_keyz",
    "cKz! Solaris Cmin 128 bpm @crushed_keyz",
]

MODELS = {
    "aug8k": "lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt",
    "v1":    "lora_out/ckz_melody_capped/epoch=54-step=2500.ckpt",
}


def slug(p):
    m = re.search(r"cKz!\s*(.+?)\s+[A-G]", p)
    title = (m.group(1) if m else p).strip().replace(" ", "_")
    k = re.search(r"\b([A-G]#?min)\b", p)
    return f"{title}_{k.group(1) if k else 'X'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="7,123")
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--aug-only", action="store_true")
    ap.add_argument("--out", default=str(REPO / "sweep_rater/campaigns/novel_ab_8k_vs_v1"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models = {"aug8k": MODELS["aug8k"]} if args.aug_only else MODELS
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plan = []
    for i, p in enumerate(NOVEL_PROMPTS):
        for seed in seeds:
            for mname in models:
                fn = f"{i:02d}_{slug(p)}__seed{seed}__{mname}.wav"
                plan.append((p, seed, mname, out / fn))
    print(f"novel prompts: {len(NOVEL_PROMPTS)}  seeds: {seeds}  models: {list(models)}")
    print(f"total clips: {len(plan)}  -> {out}\n")
    for p in NOVEL_PROMPTS:
        print(f"   «{p}»")
    if args.dry_run:
        print("\n--dry-run: no generation."); return

    import torch, torchaudio  # noqa
    from stable_audio_3 import StableAudioModel
    t0 = time.time(); n = 0
    # load one model at a time (reload base per checkpoint, like checkpoint_novelty)
    for mname, ckpt in models.items():
        todo = [job for job in plan if job[2] == mname and not job[3].exists()]
        if not todo:
            print(f"--- {mname}: all done, skipping ---"); continue
        print(f"--- loading {mname}: {ckpt} ---")
        model = StableAudioModel.from_pretrained(args.model)
        model.load_lora([str((REPO / ckpt).resolve())])
        model.set_lora_strength(1.0)
        sr = model.model_config.get("sample_rate") or model.model.sample_rate
        for p, seed, _, path in todo:
            try:
                a = model.generate(prompt=p, duration=args.duration, steps=args.steps,
                                   cfg_scale=args.cfg, seed=seed, sampler_type="dpmpp")
                torchaudio.save(str(path), a[0].cpu(), sr); n += 1
                print(f"   {mname} «{p[:34]}» seed{seed}")
            except Exception as e:
                print(f"   FAIL {mname} {p[:30]} s{seed}: {type(e).__name__}: {e}")
        del model
        try:
            import gc; gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
        except Exception: pass
    print(f"\nDone: {n} clips in {(time.time()-t0)/60:.1f}m -> {out}")
    print(f"Rate (aug8k & v1 sort back-to-back per prompt+seed):")
    print(f"  uv run python sweep_rater/rate_server.py --folder {out} --port 8799")
    print("Listen for: NOVEL melody (not a training song) + cKz timbre intact.")


if __name__ == "__main__":
    main()
