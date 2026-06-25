#!/usr/bin/env python3
"""Mode 2 sweep — hold a PICKED melody, vary the TIMBRE.

Once the user picks a clip as "cool", we want more loops with the SAME melody/key but
DIFFERENT timbres. Mechanism (uncertain — fights the semantic-acoustic entanglement):
use the picked clip AS the init at MODERATE σ (preserve melody) and vary the timbre via
descriptor words + seed. Low σ keeps melody but timbre barely moves; high σ varies timbre
but melody drifts. This sweeps σ × timbre-words × seeds to find the sweet spot (if any).

The reference (picked) clip is wired as the rater's init-audio reference, so you A/B each
clip against it. RATING for this sweep:
  timbre star  = sound quality (as always)
  melody star  = catchiness (as always)
  novelty star = DIFFERENCE from the init/reference clip (1 = identical, 5 = most different)
  notes        = tag DRIFT if the MELODY changed from the picked one (else melody assumed kept)
A Mode-2 WIN = novelty 3-4 (the sound MOVED) + high timbre + NOT tagged DRIFT (melody kept).
Read across σ: novelty 1 = timbre won't budge (σ too low); high novelty + DRIFT = melody
drifted (σ too high); sweet σ = high novelty WITHOUT drift = same melody, new timbre.

    uv run python scripts/mode2_sweep.py --dry-run
    uv run python scripts/mode2_sweep.py            # auto-picks top-rated keeper as ref
    uv run python scripts/mode2_sweep.py --ref sweep_rater/campaigns/<...>.wav
"""
from __future__ import annotations
import argparse, csv, json, os, re, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUG = "lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt"
DESCS = ["", "dark", "bright", "warm", "gritty", "lush"]   # "" = plain (seed-only variation)
SIGMAS = [0.3, 0.45, 0.6]
SEEDS = [7, 123]


def autopick_ref():
    r = json.load(open(REPO / "sweep_rater/ratings.json"))
    cands = []
    for k, v in r.items():
        if v.get("timbre") is None or "#s" in k:
            continue
        if not any(c in k for c in ["init_harvest", "ckz_factory"]):
            continue
        # rank by total, then MELODY (we preserve it), then timbre
        cands.append(((v["timbre"] + v["melody"] + (v.get("novelty") or 0),
                       v["melody"], v["timbre"]), k))
    if not cands:
        return None
    cands.sort(reverse=True)
    return cands[0][1]


def ref_prompt(refpath):
    idx = refpath.parent / "library_index.csv"
    if idx.exists():
        for row in csv.DictReader(open(idx)):
            if row["file"] == refpath.name:
                return row.get("prompt", "")
    # fallback: parse title/key from filename
    stem = refpath.stem
    m = re.search(r"\d+_([A-Za-z]+)_([A-G]#?min)", stem)
    return f"cKz! {m.group(1)} {m.group(2)} 140 bpm @crushed_keyz" if m else ""


def variant(base, d):
    if not d:
        return base
    return base.replace("@crushed", f"{d} @crushed", 1) if "@crushed" in base else f"{base} {d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="path to the picked reference clip (default: auto-pick top-rated)")
    ap.add_argument("--out", default=str(REPO / "sweep_rater/campaigns/mode2_sweep"))
    ap.add_argument("--ckpt", default=AUG)
    ap.add_argument("--sigmas", default="0.3,0.45,0.6")
    ap.add_argument("--seeds", default="7,123")
    ap.add_argument("--descs", default=",".join(DESCS))
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", default="50", help="comma list, e.g. 50,150 (swept; in filename)")
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ref = args.ref or autopick_ref()
    if not ref:
        raise SystemExit("no reference clip (none rated yet) — pass --ref")
    refpath = (REPO / ref) if not os.path.isabs(ref) else Path(ref)
    if not refpath.exists():
        raise SystemExit(f"ref not found: {refpath}")
    base = ref_prompt(refpath)
    if not base:
        raise SystemExit(f"could not resolve a prompt for {refpath.name}")
    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    descs = args.descs.split(",")
    steps_list = [int(s) for s in args.steps.split(",") if s.strip()]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plan = []
    for sig in sigmas:
        for st in steps_list:
            for d in descs:
                prompt = variant(base, d.strip())
                dl = d.strip() or "plain"
                for seed in seeds:
                    sttag = f"_st{st}" if len(steps_list) > 1 else ""
                    fn = f"sig{int(sig*100):02d}_{dl}{sttag}__seed{seed}.wav"
                    plan.append((sig, st, dl, prompt, seed, out / fn))
    print(f"REFERENCE (picked melody): {refpath.relative_to(REPO) if refpath.is_relative_to(REPO) else refpath}")
    print(f"base prompt: «{base}»")
    print(f"σ: {sigmas}   timbre words: {[d.strip() or 'plain' for d in descs]}   seeds: {seeds}")
    print(f"→ {len(plan)} clips -> {out}\n")
    if args.dry_run:
        for sig in sigmas:
            print(f"  σ{sig}: " + " | ".join(variant(base, d.strip()).split('cKz! ')[-1][:30] for d in descs))
        print("\n--dry-run."); return

    write_index(out, plan, refpath)
    import torch, torchaudio
    from stable_audio_3 import StableAudioModel
    print("loading aug model ...")
    model = StableAudioModel.from_pretrained(args.model)
    model.load_lora([str((REPO / args.ckpt).resolve())]); model.set_lora_strength(1.0)
    sr_out = model.model_config.get("sample_rate") or model.model.sample_rate
    wav_ref, sr_ref = torchaudio.load(str(refpath))
    info = torchaudio.info(str(refpath)); gen = round(min(info.num_frames/info.sample_rate, args.max_dur), 1)
    wav_ref = wav_ref[:, :int(gen*sr_ref)]
    n = 0; t0 = time.time()
    for sig, st, dl, prompt, seed, path in plan:
        if path.exists(): n += 1; continue
        try:
            a = model.generate(prompt=prompt, duration=gen, steps=st, cfg_scale=args.cfg,
                               seed=seed, sampler_type="dpmpp", init_audio=(int(sr_ref), wav_ref),
                               init_noise_level=sig)
            torchaudio.save(str(path), a[0].cpu(), sr_out); n += 1
            if n % 6 == 0: print(f"   ...{n}/{len(plan)}")
        except Exception as e:
            print(f"   FAIL σ{sig} {dl} s{seed}: {type(e).__name__}: {e}")
    print(f"\nDone {n}/{len(plan)} in {(time.time()-t0)/60:.1f}m -> {out}")
    print(f"cd ~/stable-audio-3 && uv run python sweep_rater/rate_server.py --folder {out} --port 8799")
    print("Rate: timbre=quality, melody=catchiness, NOVELTY=difference-from-init (1 same .. 5 most different).")
    print("In NOTES tag DRIFT if the melody changed. Win = novelty 3-4 + high timbre + no DRIFT (same melody, new timbre).")


def write_index(out, plan, refpath):
    with open(out / "library_index.csv", "w", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=["file","key","bpm","mood","title","seed","prompt","init_loop","init_path"])
        w.writeheader()
        for sig, st, dl, prompt, seed, path in plan:
            w.writerow({"file": path.name, "key": "", "bpm": "", "mood": f"σ{sig}/{st}st/{dl}",
                        "title": dl, "seed": seed, "prompt": prompt,
                        "init_loop": f"REFERENCE: {refpath.name}", "init_path": str(refpath)})


if __name__ == "__main__":
    main()
