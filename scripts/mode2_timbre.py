#!/usr/bin/env python3
"""Mode 2 timbre-DIVERSITY levers — loosen the cKz LoRA's grip at σ0.75.

σ0.75 is the bold-refine ceiling (more σ degrades). The remaining cap is that the LoRA
clamps timbre to a narrow cKz center during the high-σ (timbre-forming) steps. The init
holds the melody, so we can try UNCLAMPING timbre without losing the tune:

  ls10  = LoRA strength 1.0 (baseline = current bold)
  ls08  = LoRA strength 0.8
  ls06  = LoRA strength 0.6   (less cKz-pinning -> more base-model timbre range?)
  gate  = LoRA OFF early (σ>0.5), ON late (σ<=0.5): diverse timbre forms LoRA-free with the
          init anchoring melody, cKz character painted late (style-transfer setup; init +
          time-gate never combined before)

All at σ0.75, plain prompt, one fixed reference, several seeds. Rate: timbre=quality,
novelty=difference-from-ref, DRIFT if melody slips. WIN = MORE diverse/different timbre
(higher novelty, more spread across seeds) WITHOUT losing quality or melody.

    uv run python scripts/mode2_timbre.py --dry-run
    uv run python scripts/mode2_timbre.py
"""
from __future__ import annotations
import argparse, csv, json, os, re, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
AUG = "lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt"
REF_DEFAULT = "sweep_rater/campaigns/init_harvest_floor/02_Cascade_A#min__s60__seed7.wav"
# (name, lora_strength, gate_below)   gate_below=None -> uniform strength; else LoRA on for σ<=gate
ARMS = [("ls10", 1.0, None), ("ls08", 0.8, None), ("ls06", 0.6, None), ("gate", 1.0, 0.5)]


def ref_prompt(refpath):
    idx = refpath.parent / "library_index.csv"
    if idx.exists():
        for row in csv.DictReader(open(idx)):
            if row["file"] == refpath.name:
                return row.get("prompt", "")
    m = re.search(r"\d+_([A-Za-z]+)_([A-G]#?min)", refpath.stem)
    return f"cKz! {m.group(1)} {m.group(2)} 140 bpm @crushed_keyz" if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF_DEFAULT)
    ap.add_argument("--out", default=str(REPO / "sweep_rater/campaigns/mode2_timbre"))
    ap.add_argument("--ckpt", default=AUG)
    ap.add_argument("--sigma", type=float, default=0.75)
    ap.add_argument("--seeds", default="7,123,777,1234")
    ap.add_argument("--strengths", default="", help="comma LoRA strengths (e.g. 0.6,0.5,0.4) -> ls arms, no gate")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    refpath = (REPO / args.ref) if not os.path.isabs(args.ref) else Path(args.ref)
    if not refpath.exists():
        raise SystemExit(f"ref not found: {refpath}")
    base = ref_prompt(refpath)
    if not base:
        raise SystemExit(f"no prompt resolved for {refpath.name}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = ([(f"ls{int(float(s)*100):02d}", float(s), None) for s in args.strengths.split(",") if s.strip()]
            if args.strengths else ARMS)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plan = [(name, ls, gate, seed, out / f"{name}__seed{seed}.wav")
            for (name, ls, gate) in arms for seed in seeds]
    print(f"REFERENCE: {refpath.name}\nbase prompt: «{base}»")
    print(f"σ={args.sigma}  arms: {[a[0] for a in arms]}  seeds: {seeds}  → {len(plan)} clips -> {out}")
    for name, ls, gate in arms:
        print(f"   {name}: lora_strength={ls}{'' if gate is None else f', LoRA ON only for σ<={gate}'}")
    if args.dry_run:
        print("\n--dry-run."); return

    write_index(out, plan, refpath, args.sigma)
    import torch, torchaudio
    from stable_audio_3 import StableAudioModel
    print("\nloading aug model ...")
    model = StableAudioModel.from_pretrained(args.model)
    model.load_lora([str((REPO / args.ckpt).resolve())])
    sr_out = model.model_config.get("sample_rate") or model.model.sample_rate
    wav_ref, sr_ref = torchaudio.load(str(refpath))
    info = torchaudio.info(str(refpath)); gen = round(min(info.num_frames/info.sample_rate, args.max_dur), 1)
    wav_ref = wav_ref[:, :int(gen*sr_ref)]
    n = 0; t0 = time.time()
    for name, ls, gate, seed, path in plan:
        if path.exists(): n += 1; continue
        model.set_lora_strength(1.0 if gate is not None else ls)
        kw = dict(prompt=base, duration=gen, steps=args.steps, cfg_scale=args.cfg, seed=seed,
                  sampler_type="dpmpp", init_audio=(int(sr_ref), wav_ref), init_noise_level=args.sigma)
        if gate is not None:
            g = float(gate); kw["lora_strength_schedule"] = (lambda s, _g=g: 1.0 if float(s) <= _g else 0.0)
        try:
            a = model.generate(**kw)
            torchaudio.save(str(path), a[0].cpu(), sr_out); n += 1
            if n % 4 == 0: print(f"   ...{n}/{len(plan)}  ({name})")
        except Exception as e:
            print(f"   FAIL {name} s{seed}: {type(e).__name__}: {e}")
    print(f"\nDone {n}/{len(plan)} in {(time.time()-t0)/60:.1f}m -> {out}")
    print(f"cd ~/stable-audio-3 && uv run python sweep_rater/rate_server.py --folder {out} --port 8799")
    print("Rate: timbre=quality, novelty=diff-from-ref, DRIFT if melody slips. WIN = more diff + quality held.")
    print("Compare: does ls08/ls06 or gate give MORE timbre variety than ls10 (baseline) without quality/melody loss?")


def write_index(out, plan, refpath, sigma):
    with open(out / "library_index.csv", "w", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=["file","key","bpm","mood","title","seed","prompt","init_loop","init_path"])
        w.writeheader()
        for name, ls, gate, seed, path in plan:
            w.writerow({"file": path.name, "key": "", "bpm": "", "mood": f"σ{sigma}/{name}",
                        "title": name, "seed": seed, "prompt": ref_prompt(refpath),
                        "init_loop": f"REFERENCE: {refpath.name}", "init_path": str(refpath)})


if __name__ == "__main__":
    main()
