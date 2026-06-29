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


# cKz loop filenames put KEY *before* BPM, e.g. "cKz! JUNGLE C#min 140 bpm @crushed_keyz",
# "... Fsharpmin 105 bpm ...", "... Bmin 160 @crushed_key" (no 'bpm' word). The accidental
# may be '#', 'sharp', 'b', or 'flat'. This is the in-distribution init path.
CKZ = re.compile(r"\b([A-Ga-g])\s*(#|sharp|b|flat)?\s*(min|maj)\b[^0-9]*?(\d{2,3})", re.I)


def parse_ckz(name):
    m = CKZ.search(name)
    if not m:
        return None
    note = m.group(1).upper()
    acc = (m.group(2) or "").lower()
    if acc in ("#", "sharp"):
        note += "#"
    elif acc in ("b", "flat"):
        note = FLAT2SHARP.get(note + "b", note)  # flat -> sharp spelling
    qual = "min" if m.group(3).lower() == "min" else "maj"
    bpm = int(m.group(4))
    return note + qual, bpm


def name_pool():
    """Big on-brand single-word title pool. Reuses the catalog's namegen list so the many
    titles stay in the cKz dark/cosmic/luxe aesthetic and don't collide with the 8 defaults.
    Falls back to the 8 TITLES if pipeline/namegen isn't importable."""
    try:
        import sys as _s
        _s.path.insert(0, str(REPO / "pipeline"))
        from namegen import POOL
        return list(dict.fromkeys(list(TITLES) + list(POOL)))  # de-duped, defaults first
    except Exception:
        return list(TITLES)


def select_loops(dirs, n, root=LOOPS_ROOT, minor_only=False, dense=False, max_silence=0.12,
                 shuffle=False, loop_seed=None, parser=parse_cy, include=None):
    """Pick n foreign loops, round-robin across keys for spread (minor preferred).
    Allows MULTIPLE loops per key (harvest mode) so n can exceed the 12 minor keys.
    minor_only: skip major-key loops (they clash with the minor cKz model -> 0 keepers).
    dense: skip loops whose own silence-fraction > max_silence (a sparse init seeds a
    sparse output — sparseness is the #1 floor killer). Checked lazily (only on picked).
    shuffle: randomize each key's bucket + the key order so repeated runs draw a
    DIFFERENT diverse subset instead of always the alphabetically-first loops. Pass
    loop_seed for a reproducible shuffle (same seed -> same selection)."""
    from collections import defaultdict
    import random as _random
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
            if include and not any(s.lower() in f.name.lower() for s in include):
                continue  # restrict to high-yield source loops (substring match)
            kb = parser(f.name)
            if kb and not (minor_only and kb[0].endswith("maj")):
                bykey[kb[0]].append((kb[0], kb[1], f))
    minkeys = [k for k in bykey if k.endswith("min")]
    majkeys = [k for k in bykey if k.endswith("maj")]
    if shuffle:
        rng = _random.Random(loop_seed)        # loop_seed=None -> fresh each run
        for v in bykey.values():
            rng.shuffle(v)                       # different loop wins each key bucket
        rng.shuffle(minkeys); rng.shuffle(majkeys)  # vary which keys lead the round-robin
    keys = minkeys + majkeys
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
    ap.add_argument("--shuffle", action="store_true", help="randomize loop selection so each run draws a DIFFERENT diverse subset (not the same alphabetical front-runners)")
    ap.add_argument("--loop-seed", type=int, default=None, help="seed the --shuffle for a reproducible selection (default: fresh each run)")
    ap.add_argument("--ckz-init", action="store_true",
                    help="use the IN-DISTRIBUTION cKz training loops as init (data/ckz_aug) "
                         "instead of foreign Cymatics packs. Parses the cKz 'KEY BPM' filename "
                         "order. Variation comes from HIGH sigma + seeds + transposed prompt key.")
    ap.add_argument("--reverse-init", action="store_true",
                    help="time-reverse (retrograde) EVERY init loop before use (whole campaign "
                         "reversed). For the pure forward-vs-reversed A/B. To MIX reversed arms "
                         "into a normal run instead, use --reverse-sigmas.")
    ap.add_argument("--reverse-sigmas", default="",
                    help="comma sigmas that add EXTRA reverse-init arms ALONGSIDE the forward "
                         "ones (tagged r<nn>), e.g. '0.8'. This is how you 'mix in some reverse "
                         "init' in a variation harvest without reversing the whole run.")
    ap.add_argument("--init-loops", nargs="*", default=None,
                    help="restrict init selection to source loops whose FILENAME contains any of "
                         "these substrings (case-insensitive), e.g. ICEBERG Bando CANOPY. Use to "
                         "seed only from the high-yield loops the ratings found.")
    ap.add_argument("--rand-titles", action="store_true",
                    help="give every loop+key-variant its OWN unique evocative title drawn from "
                         "the namegen pool (~140 words) instead of cycling the 8 defaults. The #1 "
                         "lever for melodic variety — distinct title -> distinct melody.")
    ap.add_argument("--rand-seeds", type=int, default=0,
                    help="use N RANDOM seeds instead of --seeds. FRESH each run by default "
                         "(system entropy) so every run is new variation. More seeds = more "
                         "variation per title/arm; the seeds used are printed + saved in the index.")
    ap.add_argument("--seed-base", type=int, default=None,
                    help="optional: fix the RNG base for --rand-seeds so the SAME seeds are drawn "
                         "every run (reproducible / resumable). Default None = fresh each run.")
    ap.add_argument("--unique-seeds", action="store_true",
                    help="give EVERY clip its own distinct random seed (no seed reused across "
                         "loops/arms/seeds). Max variation. --rand-seeds N = N unique-seeded takes "
                         "per loop+key+arm. Total clips = loops × keyvars × arms × N.")
    ap.add_argument("--style-prefix", default="cKz!",
                    help="producer style token that OPENS every prompt (must match the training captions, e.g. 'cKz!')")
    ap.add_argument("--trigger", default="@crushed_keyz",
                    help="producer trigger tag that CLOSES every prompt (must match the training captions, e.g. '@crushed_keyz')")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # cKz in-distribution init: flip root/dirs/parser unless the user overrode them.
    parser = parse_cy
    if args.ckz_init:
        parser = parse_ckz
        if args.loops_root == str(LOOPS_ROOT):
            args.loops_root = str(REPO)
        if args.init_dirs == DEFAULT_DIRS:
            args.init_dirs = ["data/ckz_aug"]
        if args.out == str(REPO / "sweep_rater/campaigns/init_sweep_aug"):
            args.out = str(REPO / ("sweep_rater/campaigns/ckz_init_reversed"
                                   if args.reverse_init else "sweep_rater/campaigns/ckz_init_variation"))

    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    rev_sigmas = [float(s) for s in args.reverse_sigmas.split(",") if s.strip()]
    import random as _rs
    seedrng = _rs.Random(args.seed_base)   # seed_base=None -> system entropy, fresh each run
    _used_seeds = set()
    def fresh_seed():
        while True:
            s = seedrng.randint(1, 9_999_999)
            if s not in _used_seeds:
                _used_seeds.add(s); return s
    unique = args.unique_seeds and args.rand_seeds and args.rand_seeds > 0
    n_per_combo = args.rand_seeds if (args.rand_seeds and args.rand_seeds > 0) else None
    if unique:
        seeds = None  # drawn fresh per clip in the plan loop below
    elif n_per_combo:
        seeds = sorted({fresh_seed() for _ in range(n_per_combo)})  # shared list, distinct
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    loops = select_loops(args.init_dirs, args.n_loops, args.loops_root, args.minor_only,
                         args.dense_init, args.max_init_silence,
                         shuffle=args.shuffle, loop_seed=args.loop_seed, parser=parser,
                         include=args.init_loops)
    if not loops:
        raise SystemExit("no foreign loops found — check --init-dirs")

    # title source: 8 defaults, or the big de-duped namegen pool (unique per loop+key-variant)
    pool = name_pool() if args.rand_titles else list(TITLES)
    if args.rand_titles:
        import random as _rt
        _rt.Random(args.loop_seed).shuffle(pool)
    tcount = [0]
    def next_title(i):
        if not args.rand_titles:
            return TITLES[i % len(TITLES)]
        t = pool[tcount[0] % len(pool)]; tcount[0] += 1; return t

    # arms = (cond_tag, sigma_or_None, reverse_bool). noinit baseline + forward s-arms + reverse r-arms.
    base = ([] if args.no_baseline else [("noinit", None, False)])
    fwd  = [(f"s{int(s*100):02d}", s, args.reverse_init) for s in sigmas]
    rev  = [(f"r{int(s*100):02d}", s, True) for s in rev_sigmas]
    arms = base + fwd + rev

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plan = []  # (cond, sigma_or_None, rev, title, key, bpm, loop_path, seed, prompt, filename)
    for i, (key, bpm, f) in enumerate(loops):
        # key variants: matched (prompt=init key) + optional tritone (DIFFERENT melody -> 2x/loop).
        kvariants = [("m", key)] + ([("t", transpose(key, 6))] if args.tritone else [])
        for ktag, pkey in kvariants:
            title = next_title(i)
            prompt = f"{args.style_prefix} {title} {pkey} {bpm} bpm {args.trigger}"
            for cond, sig, rv in arms:
                combo_seeds = [fresh_seed() for _ in range(n_per_combo)] if unique else seeds
                for seed in combo_seeds:
                    kt = f"_{ktag}" if args.tritone else ""
                    fn = f"{i:02d}_{title}_{pkey}{kt}__{cond}__seed{seed}.wav"
                    plan.append((cond, sig, rv, title, pkey, bpm, f, seed, prompt, fn))

    print(f"model: {args.ckpt}")
    print(f"init loops ({len(loops)}):")
    for key, bpm, f in loops:
        print(f"   [{key:6} {bpm:3}bpm]  {f.name[:54]}")
    armtags = [a[0] for a in arms]
    print(f"arms: {armtags}   (fwd sigmas {sigmas}, reverse sigmas {rev_sigmas or 'none'})")
    seedinfo = (f"{n_per_combo} UNIQUE per clip (all distinct, {len(_used_seeds)} total)"
                if unique else f"{len(seeds)} shared: {seeds}")
    print(f"titles: {'namegen pool (unique/loop)' if args.rand_titles else 'default 8'}   seeds: {seedinfo}")
    print(f"total clips: {len(plan)}  -> {out}\n")
    if args.dry_run:
        print("sample filenames:")
        for _,_,_,_,_,_,_,_,_,fn in plan[:8]:
            print("   ", fn)
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
    for cond, sig, rv, title, key, bpm, f, seed, prompt, fn in plan:
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
            if rv:
                wav_i = torch.flip(wav_i, dims=[-1])   # retrograde -> OOD init
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
        for cond, sig, rv, title, key, bpm, f, seed, prompt, fn in plan:
            w.writerow({"file": fn, "key": key, "bpm": bpm, "mood": cond, "title": title,
                        "seed": seed, "prompt": prompt,
                        "init_loop": ((f.name + (" (reversed)" if rv else "")) if sig is not None else ""),
                        "init_path": (str(f) if sig is not None else "")})


if __name__ == "__main__":
    main()
