#!/usr/bin/env python3
"""Does the PROMPT key need to match the INIT-audio key? (aug-8000, σ0.7)

Every init harvest so far set the prompt key = the foreign loop's key. This isolates
that choice: fix the init loop + title + seed, vary ONLY the prompt's key relative to
the init loop's actual key:

  match    : prompt key = init key            (what we've always done)
  up2      : prompt key = init key + 2 semis  (mild mismatch)
  tritone  : prompt key = init key + 6 semis  (max clash)
  nokey    : prompt has NO key token          (does the text key matter at all w/ init?)

Rate T/M/N, and in notes say which key it sounds in (init's? prompt's? off?). Tells us:
(1) does matching matter at σ0.7 on aug, (2) does the model follow the init or the text
key, (3) is mismatch a transposition lever or just degradation.

    uv run python scripts/key_match_ab.py --dry-run
    uv run python scripts/key_match_ab.py
"""
from __future__ import annotations
import argparse, csv, re, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOOPS_ROOT = Path("/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops")
AUG = "lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt"
DIRS = ["Cymatics_Dragon_Platinum_Expansion", "Cymatics_Gems_Vol_1_Trap_Melodies",
        "Cymatics_2020_Melody_Collection", "Cymatics Gems Vol 6 - Future RnB Melodies"]
TITLES = ["Mirage", "Obsidian", "Cascade", "Solaris", "Wraith", "Nocturne", "Velvet", "Voltage"]
KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
FLAT2SHARP = {"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}
CY = re.compile(r"(\d{2,3})\s*BPM\s+([A-Ga-g])(#|b)?\s*Min", re.I)
CONDS_ALL = {"match": 0, "up2": 2, "up5": 7, "tritone": 6, "nokey": None}


def parse_cy(name):
    m = CY.search(name)
    if not m: return None
    note = m.group(2).upper() + (m.group(3) or "")
    return FLAT2SHARP.get(note, note) + "min", int(m.group(1))


def transpose(keymin, delta):
    note = keymin[:-3]; i = KEYS.index(FLAT2SHARP.get(note, note))
    return KEYS[(i + delta) % 12] + "min"


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
    ap.add_argument("--out", default=str(REPO / "sweep_rater/campaigns/key_match_ab"))
    ap.add_argument("--ckpt", default=AUG)
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--seeds", default="7,123")
    ap.add_argument("--n-loops", type=int, default=5)
    ap.add_argument("--conds", default="match,up2,tritone,nokey", help="comma list from: " + ",".join(CONDS_ALL))
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    conds = [(c, CONDS_ALL[c]) for c in args.conds.split(",") if c in CONDS_ALL]
    loops = pick(args.loops_root, args.n_loops)
    if not loops: raise SystemExit("no minor foreign loops found")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plan = []
    for i, (ikey, bpm, f) in enumerate(loops):
        title = TITLES[i % len(TITLES)]
        for cond, delta in conds:
            pkey = None if delta is None else transpose(ikey, delta)
            prompt = (f"cKz! {title} {bpm} bpm @crushed_keyz" if pkey is None
                      else f"cKz! {title} {pkey} {bpm} bpm @crushed_keyz")
            for seed in seeds:
                fn = f"{i:02d}_{title}_init{ikey}_prompt{pkey or 'none'}__{cond}__seed{seed}.wav"
                plan.append((cond, pkey, ikey, bpm, f, title, seed, prompt, out / fn))
    print(f"model: {args.ckpt}  σ={args.sigma}")
    print(f"loops ({len(loops)}): " + ", ".join(f"{k}" for k, _, _ in loops))
    print(f"conditions: {[c for c,_ in conds]}  seeds: {seeds}  → {len(plan)} clips\n")
    for i, (ikey, bpm, f) in enumerate(loops):
        mapping = "  ".join(f"{c}={('none' if d is None else transpose(ikey,d))}" for c, d in conds)
        print(f"  loop{i} init={ikey} {bpm}bpm  ->  {mapping}")
    if args.dry_run:
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
    for cond, pkey, ikey, bpm, f, title, seed, prompt, path in plan:
        if path.exists(): n += 1; continue
        info = torchaudio.info(str(f)); gen = round(min(info.num_frames/info.sample_rate, args.max_dur), 1)
        sr_i, wav_i = load_init(f, gen)
        try:
            a = model.generate(prompt=prompt, duration=gen, steps=args.steps, cfg_scale=args.cfg,
                               seed=seed, sampler_type="dpmpp", init_audio=(int(sr_i), wav_i),
                               init_noise_level=args.sigma)
            torchaudio.save(str(path), a[0].cpu(), sr_out); n += 1
            if n % 5 == 0: print(f"   ...{n}/{len(plan)}")
        except Exception as e:
            print(f"   FAIL {title} {cond} s{seed}: {type(e).__name__}: {e}")
    print(f"\nDone {n}/{len(plan)} in {(time.time()-t0)/60:.1f}m -> {out}")
    print(f"cd ~/stable-audio-3 && uv run python sweep_rater/rate_server.py --folder {out} --port 8799")
    print("In notes: which key does it sound in — init's, prompt's, or off? Does match beat mismatch?")


def write_index(out, plan):
    with open(out / "library_index.csv", "w", newline="") as fc:
        w = csv.DictWriter(fc, fieldnames=["file","key","bpm","mood","title","seed","prompt","init_loop","init_path"])
        w.writeheader()
        for cond, pkey, ikey, bpm, f, title, seed, prompt, path in plan:
            w.writerow({"file": path.name, "key": f"prompt={pkey or 'none'}/init={ikey}", "bpm": bpm,
                        "mood": cond, "title": title, "seed": seed, "prompt": prompt,
                        "init_loop": f.name, "init_path": str(f)})


if __name__ == "__main__":
    main()
