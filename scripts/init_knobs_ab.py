#!/usr/bin/env python3
"""Final init knobs — cfg × init, and bpm-match. (aug-8000, foreign-init σ0.75, minor-only)

The last untested init axes. Per loop (init key matched, σ0.75):
  cfg3   : cfg 3.0, prompt bpm = init bpm   (current baseline)
  cfg5   : cfg 5.0, prompt bpm = init bpm
  cfg7   : cfg 7.0, prompt bpm = init bpm
  bpmX   : cfg 3.0, prompt bpm = FAR from init bpm  (does prompt tempo need to match init?)

Tells us: (a) does higher cfg sharpen/help with init or just distort, (b) does the prompt
bpm matter when the init carries the actual tempo (like the key-match result). Rate T/M/N;
in notes say if bpmX sounds off-tempo or fine.

    uv run python scripts/init_knobs_ab.py --dry-run
    uv run python scripts/init_knobs_ab.py
"""
from __future__ import annotations
import argparse, csv, re, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOOPS_ROOT = Path("/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops")
AUG = "lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt"
DIRS = ["Cymatics_Dragon_Platinum_Expansion", "Cymatics_Gems_Vol_1_Trap_Melodies",
        "Cymatics_2020_Melody_Collection", "Cymatics Gems Vol 6 - Future RnB Melodies"]
TITLES = ["Mirage", "Obsidian", "Cascade", "Solaris", "Wraith", "Nocturne"]
FLAT2SHARP = {"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}
CY = re.compile(r"(\d{2,3})\s*BPM\s+([A-Ga-g])(#|b)?\s*Min", re.I)
# (name, cfg, bpm_mode)
CONDS = [("cfg3", 3.0, "match"), ("cfg5", 5.0, "match"), ("cfg7", 7.0, "match"), ("bpmX", 3.0, "mismatch")]


def parse_cy(name):
    m = CY.search(name)
    if not m: return None
    note = m.group(2).upper() + (m.group(3) or "")
    return FLAT2SHARP.get(note, note) + "min", int(m.group(1))


def mm_bpm(b):
    return b - 40 if b >= 115 else b + 40   # clearly different tempo, plausible range


def pick(root, n):
    from collections import defaultdict
    bykey = defaultdict(list)
    for d in DIRS:
        p = Path(root) / d
        if not p.is_dir(): continue
        for f in sorted(p.glob("*.wav")) + sorted(p.glob("*.mp3")):
            kb = parse_cy(f.name)
            if kb: bykey[kb[0]].append((kb[0], kb[1], f))
    keys = sorted(bykey); out, i = [], 0
    while len(out) < n and any(bykey[k] for k in keys):
        k = keys[i % len(keys)]
        if bykey[k]: out.append(bykey[k].pop(0))
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops-root", default=str(LOOPS_ROOT))
    ap.add_argument("--out", default=str(REPO / "sweep_rater/campaigns/init_knobs_ab"))
    ap.add_argument("--ckpt", default=AUG)
    ap.add_argument("--sigma", type=float, default=0.75)
    ap.add_argument("--seeds", default="7,123")
    ap.add_argument("--n-loops", type=int, default=6)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    loops = pick(args.loops_root, args.n_loops)
    if not loops: raise SystemExit("no minor foreign loops found")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plan = []
    for i, (ikey, bpm, f) in enumerate(loops):
        title = TITLES[i % len(TITLES)]
        for cond, cfg, bmode in CONDS:
            pbpm = bpm if bmode == "match" else mm_bpm(bpm)
            prompt = f"cKz! {title} {ikey} {pbpm} bpm @crushed_keyz"
            for seed in seeds:
                fn = f"{i:02d}_{title}_{ikey}_init{bpm}_prompt{pbpm}__{cond}__seed{seed}.wav"
                plan.append((cond, cfg, bpm, pbpm, ikey, f, title, seed, prompt, out / fn))
    print(f"model: {args.ckpt}  σ={args.sigma}")
    print(f"loops ({len(loops)}): " + ", ".join(f"{k}/{b}" for k, b, _ in loops))
    print(f"conds: {[c for c,_,_ in CONDS]}  seeds: {seeds}  → {len(plan)} clips")
    if args.dry_run:
        for i,(ikey,bpm,f) in enumerate(loops):
            print(f"  loop{i} {ikey} init={bpm}bpm  bpmX={mm_bpm(bpm)}bpm")
        print("\n--dry-run."); return

    write_index(out, plan)
    import torch, torchaudio
    from stable_audio_3 import StableAudioModel
    print("\nloading aug model ...")
    model = StableAudioModel.from_pretrained(args.model)
    model.load_lora([str((REPO / args.ckpt).resolve())]); model.set_lora_strength(1.0)
    sr_out = model.model_config.get("sample_rate") or model.model.sample_rate
    cache = {}
    def load_init(f, dur):
        if f not in cache: cache[f] = torchaudio.load(str(f))
        wav, sr = cache[f]; return sr, wav[:, :int(dur*sr)]
    n = 0; t0 = time.time()
    for cond, cfg, bpm, pbpm, ikey, f, title, seed, prompt, path in plan:
        if path.exists(): n += 1; continue
        info = torchaudio.info(str(f)); gen = round(min(info.num_frames/info.sample_rate, args.max_dur), 1)
        sr_i, wav_i = load_init(f, gen)
        try:
            a = model.generate(prompt=prompt, duration=gen, steps=args.steps, cfg_scale=cfg,
                               seed=seed, sampler_type="dpmpp", init_audio=(int(sr_i), wav_i),
                               init_noise_level=args.sigma)
            torchaudio.save(str(path), a[0].cpu(), sr_out); n += 1
            if n % 6 == 0: print(f"   ...{n}/{len(plan)}")
        except Exception as e:
            print(f"   FAIL {title} {cond} s{seed}: {type(e).__name__}: {e}")
    print(f"\nDone {n}/{len(plan)} in {(time.time()-t0)/60:.1f}m -> {out}")
    print(f"cd ~/stable-audio-3 && uv run python sweep_rater/rate_server.py --folder {out} --port 8799")
    print("Read: does cfg5/cfg7 beat cfg3? does bpmX (tempo mismatch) hold up or go off-tempo?")


def write_index(out, plan):
    with open(out / "library_index.csv", "w", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=["file","key","bpm","mood","title","seed","prompt","init_loop","init_path"])
        w.writeheader()
        for cond, cfg, bpm, pbpm, ikey, f, title, seed, prompt, path in plan:
            w.writerow({"file": path.name, "key": ikey, "bpm": f"prompt={pbpm}/init={bpm}",
                        "mood": cond, "title": title, "seed": seed, "prompt": prompt,
                        "init_loop": f.name, "init_path": str(f)})


if __name__ == "__main__":
    main()
