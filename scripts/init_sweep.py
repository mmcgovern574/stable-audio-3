#!/usr/bin/env python3
"""Init-audio sweep — does FOREIGN-loop init at the RIGHT noise level beat t2m?

We only ever tested init at one noise level (0.6), with matched/offkey cKz loops,
on v1, never rated for novelty. This tests it properly on aug-8000:

  init source = FOREIGN melodies (Cymatics packs) — out-of-distribution structure
  init_noise_level (sigma_max) SWEEP = 0.6 / 0.75 / 0.85 / 0.92   (the real variable)
  + a NO-INIT (pure t2m) baseline arm per loop+seed for direct comparison
  prompt = cKz! <novel title> <init's key> <init's bpm> bpm @crushed_keyz

Mechanism: x_start = init*(1-sigma) + noise*sigma, denoise from sigma->0.
Low sigma keeps the init (incl. its foreign TIMBRE -> bleed). High sigma keeps only
the melodic skeleton and repaints timbre as cKz. We're hunting that sweet spot.

Rate timbre/melody/novelty; the init loop is wired as the reference player so you
can hear whether the cKz version kept the foreign melody's structure.

    uv run python scripts/init_sweep.py --dry-run
    uv run python scripts/init_sweep.py
"""
from __future__ import annotations
import argparse, csv, os, re, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOOPS_ROOT = Path("/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops")
AUG = "lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt"
DEFAULT_DIRS = ["Cymatics_Dragon_Platinum_Expansion", "Cymatics_Gems_Vol_1_Trap_Melodies"]
TITLES = ["Mirage", "Obsidian", "Cascade", "Wraith", "Nocturne", "Voltage", "Solaris", "Velvet"]
FLAT2SHARP = {"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}
KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
CY = re.compile(r"(\d{2,3})\s*BPM\s+([A-Ga-g])(#|b)?\s*(Min|Maj)", re.I)


def transpose(keymin, semis):
    """Transpose a 'Xmin'/'Xmaj' key by `semis` semitones (sharp spelling)."""
    qual = "min" if keymin.endswith("min") else "maj"
    note = FLAT2SHARP.get(keymin[:-3], keymin[:-3])
    return KEYS[(KEYS.index(note) + semis) % 12] + qual


def parse_cy(name):
    m = CY.search(name)
    if not m:
        return None
    bpm = int(m.group(1))
    note = m.group(2).upper() + (m.group(3) or "")
    note = FLAT2SHARP.get(note, note)
    qual = "min" if m.group(4).lower().startswith("min") else "maj"
    return note + qual, bpm


def select_loops(dirs, n, root=LOOPS_ROOT, minor_only=False, dense=False, max_silence=0.12):
    """Pick n foreign loops, round-robin across keys for spread (minor preferred).
    Allows MULTIPLE loops per key (harvest mode) so n can exceed the 12 minor keys.
    minor_only: skip major-key loops (they clash with the minor cKz model -> 0 keepers).
    dense: skip loops whose own silence-fraction > max_silence (a sparse init seeds a
    sparse output — sparseness is the #1 floor killer). Checked lazily (only on picked)."""
    from collections import defaultdict
    sil_of = None
    if dense:
        import sys as _s; _s.path.insert(0, str(REPO / "sweep_rater/scorer"))
        from extract_features import feats as _f
        def sil_of(path):
            try: return float(_f(str(path)).get("silence", 0.0))
            except Exception: return 0.0
    bykey = defaultdict(list)
    for d in dirs:
        p = Path(root) / d
        if not p.is_dir():
            continue
        for f in sorted(p.glob("*.wav")) + sorted(p.glob("*.mp3")):
            kb = parse_cy(f.name)
            if kb and not (minor_only and kb[0].endswith("maj")):
                bykey[kb[0]].append((kb[0], kb[1], f))
    keys = [k for k in bykey if k.endswith("min")] + [k for k in bykey if k.endswith("maj")]
    picked, skipped, i = [], 0, 0
    while len(picked) < n and any(bykey[k] for k in keys):
        k = keys[i % len(keys)]; i += 1
        if not bykey[k]:
            continue
        cand = bykey[k].pop(0)
        if dense and sil_of(cand[2]) > max_silence:
            skipped += 1; continue                 # sparse loop -> would seed sparse output
        picked.append(cand)
    if dense and skipped:
        print(f"  dense filter: skipped {skipped} sparse foreign loops (silence > {max_silence})")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loops-root", default=str(LOOPS_ROOT))
    ap.add_argument("--init-dirs", nargs="*", default=DEFAULT_DIRS)
    ap.add_argument("--out", default=str(REPO / "sweep_rater/campaigns/init_sweep_aug"))
    ap.add_argument("--ckpt", default=AUG)
    ap.add_argument("--sigmas", default="0.6,0.75,0.85,0.92")
    ap.add_argument("--seeds", default="7,123")
    ap.add_argument("--n-loops", type=int, default=5)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--no-baseline", action="store_true", help="skip the no-init arm (harvest mode)")
    ap.add_argument("--minor-only", action="store_true", help="skip major-key foreign loops (they underperform)")
    ap.add_argument("--dense-init", action="store_true", help="skip sparse foreign loops (silence>0.12) — raises the floor")
    ap.add_argument("--max-init-silence", type=float, default=0.12)
    ap.add_argument("--tritone", action="store_true", help="also generate each loop with prompt key a TRITONE off (2x melodies/loop)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    loops = select_loops(args.init_dirs, args.n_loops, args.loops_root, args.minor_only,
                         args.dense_init, args.max_init_silence)
    if not loops:
        raise SystemExit("no foreign loops found — check --init-dirs")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plan = []  # (cond, sigma_or_None, title, key, bpm, loop_path, seed, prompt, filename)
    for i, (key, bpm, f) in enumerate(loops):
        title = TITLES[i % len(TITLES)]
        # key variants: matched (prompt=init key) + optional tritone (proven equal-quality,
        # DIFFERENT melody -> 2x melodies per loop). Init audio is the SAME loop either way.
        kvariants = [("m", key)] + ([("t", transpose(key, 6))] if args.tritone else [])
        arms = ([] if args.no_baseline else [("noinit", None)]) + [(f"s{int(s*100):02d}", s) for s in sigmas]
        for ktag, pkey in kvariants:
            prompt = f"cKz! {title} {pkey} {bpm} bpm @crushed_keyz"
            for cond, sig in arms:
                for seed in seeds:
                    kt = f"_{ktag}" if args.tritone else ""
                    fn = f"{i:02d}_{title}_{pkey}{kt}__{cond}__seed{seed}.wav"
                    plan.append((cond, sig, title, pkey, bpm, f, seed, prompt, fn))

    print(f"model: {args.ckpt}")
    print(f"foreign init loops ({len(loops)}):")
    for key, bpm, f in loops:
        print(f"   [{key:6} {bpm:3}bpm]  {f.name[:54]}")
    print(f"sigmas: {sigmas}  +noinit baseline   seeds: {seeds}")
    print(f"total clips: {len(plan)}  -> {out}\n")
    if args.dry_run:
        print("arms per loop:", ([] if args.no_baseline else ["noinit"]) + [f"s{int(s*100)}" for s in sigmas])
        print("\n--dry-run: no generation."); return

    write_index(out, plan)
    import torch, torchaudio  # noqa
    from stable_audio_3 import StableAudioModel
    print("loading aug model ...")
    model = StableAudioModel.from_pretrained(args.model)
    model.load_lora([str((REPO / args.ckpt).resolve())])
    model.set_lora_strength(1.0)
    sr_out = model.model_config.get("sample_rate") or model.model.sample_rate

    initcache = {}
    def load_init(path, dur):
        if path not in initcache:
            wav, sr = torchaudio.load(str(path))
            initcache[path] = (wav, sr)
        wav, sr = initcache[path]
        n = int(dur * sr)
        return (sr, wav[:, :n])

    t0 = time.time(); n = 0
    for cond, sig, title, key, bpm, f, seed, prompt, fn in plan:
        path = out / fn
        if path.exists():
            n += 1; continue
        # generation duration = min(loop length, max-dur) so the init fills the clip
        info = torchaudio.info(str(f))
        loop_dur = info.num_frames / info.sample_rate
        gen_dur = round(min(loop_dur, args.max_dur), 1)
        kw = dict(prompt=prompt, duration=gen_dur, steps=args.steps,
                  cfg_scale=args.cfg, seed=seed, sampler_type="dpmpp")
        if sig is not None:
            sr_i, wav_i = load_init(f, gen_dur)
            kw["init_audio"] = (int(sr_i), wav_i)
            kw["init_noise_level"] = sig
        try:
            a = model.generate(**kw)
            torchaudio.save(str(path), a[0].cpu(), sr_out); n += 1
            if n % 6 == 0: print(f"   ...{n}/{len(plan)}  ({title} {cond})")
        except Exception as e:
            print(f"   FAIL {title} {cond} s{seed}: {type(e).__name__}: {e}")
    print(f"\nDone: {n}/{len(plan)} in {(time.time()-t0)/60:.1f}m -> {out}")
    print(f"Rate (init loop plays as reference; compare structure):")
    print(f"  uv run python sweep_rater/rate_server.py --folder {out} --port 8799")
    print("Question: does any sigma beat the noinit baseline on novelty WITHOUT losing timbre?")


def write_index(out, plan):
    with open(out / "library_index.csv", "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["file","key","bpm","mood","title","seed",
                                             "prompt","init_loop","init_path"])
        w.writeheader()
        for cond, sig, title, key, bpm, f, seed, prompt, fn in plan:
            w.writerow({"file": fn, "key": key, "bpm": bpm, "mood": cond, "title": title,
                        "seed": seed, "prompt": prompt,
                        "init_loop": (f.name if sig is not None else ""),
                        "init_path": (str(f) if sig is not None else "")})


if __name__ == "__main__":
    main()
