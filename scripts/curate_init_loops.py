#!/usr/bin/env python3
"""Curate a diverse, DENSE set of init loops from any melody-loop folder → manifest.

Generalizes the cKz init-library curation for a new producer / new loop pack.
Scans --src for full (non-stem) melody loops with a parseable key+BPM, filters to
DENSE ones (low silence / decent level — sparse loops cause sparse output), and
selects a balanced spread across keys/moods. Writes a manifest JSON consumed by
build_library.py:  [{"path","key","bpm","mood"}, ...]

Stores ABSOLUTE paths (works when run natively on the machine that will generate).

    python scripts/curate_init_loops.py --src "/path/to/Melody_Loops" \
        --out sweep_rater/library_loops.json --per-key 4
"""
from __future__ import annotations
import argparse, glob, json, os, re, random, collections
import numpy as np
import soundfile as sf

MOOD_TAGS = ["sad", "dark", "ethnic", "r&b", "rnb", "lofi", "trap", "soul",
             "nostalgic", "orchestral", "various", "cobra", "dragon", "gems",
             "oracle", "monster", "drill", "west coast", "classic", "simple"]


def parse_key_bpm(base):
    mk = re.search(r"\b([A-G][#b]?)\s*(Min|Maj)\b", base, re.I)
    mb = re.search(r"(\d{2,3})\s*BPM", base, re.I)
    if not mk or not mb:
        return None, None
    note = mk.group(1)[0].upper() + mk.group(1)[1:]
    return f"{note}{'min' if mk.group(2).lower()=='min' else 'maj'}", int(mb.group(1))


def mood(path):
    p = path.lower()
    for t in MOOD_TAGS:
        if t in p:
            return t
    return os.path.basename(os.path.dirname(path)).lower()[:10] or "misc"


def density(path):
    """(is_dense, silence_pct, rms). Dense = low trailing/internal silence + level."""
    try:
        a, sr = sf.read(path, dtype="float32")
    except Exception:
        return False, 100.0, 0.0
    if a.ndim > 1:
        a = a.mean(1)
    if len(a) < sr * 3:
        return False, 100.0, 0.0
    fl = int(0.05 * sr); nf = len(a) // fl
    rms = np.array([np.sqrt((a[i*fl:(i+1)*fl]**2).mean()+1e-12) for i in range(nf)])
    sil = 100 * (20*np.log10(rms+1e-9) < -45).mean()
    return (sil < 12 and a.std() > 0.03), float(sil), float(a.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of melody loops to scan")
    ap.add_argument("--out", default="sweep_rater/library_loops.json")
    ap.add_argument("--per-key", type=int, default=4)
    ap.add_argument("--quality", choices=["min", "maj", "both"], default="min")
    ap.add_argument("--scan-per-key", type=int, default=40, help="max candidates density-checked per key")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cands = []
    for p in glob.glob(os.path.join(args.src, "**", "*.wav"), recursive=True):
        if "loop stems" in p.lower():
            continue
        if re.search(r"\b(bass|drums?|808|kick|hat|perc|vocal)\b", os.path.basename(p), re.I):
            continue
        k, b = parse_key_bpm(os.path.basename(p))
        if not k:
            continue
        if args.quality != "both" and not k.endswith(args.quality):
            continue
        cands.append((k, b, os.path.abspath(p)))
    bykey = collections.defaultdict(list)
    for k, b, p in cands:
        bykey[k].append((b, p))
    print(f"{len(cands)} candidate loops across {len(bykey)} keys")

    selected = []
    for k in sorted(bykey):
        pool = bykey[k][:]; rng.shuffle(pool)
        picked = 0; seen = set()
        for b, p in pool[: args.scan_per_key]:
            if picked >= args.per_key:
                break
            ok, sil, r = density(p)
            if not ok:
                continue
            mo = mood(p)
            if mo in seen and picked >= 2:
                continue
            seen.add(mo)
            selected.append({"path": p, "key": k, "bpm": b, "mood": mo})
            picked += 1
    json.dump(selected, open(args.out, "w"), indent=1)
    cnt = collections.Counter(s["key"] for s in selected)
    print(f"selected {len(selected)} dense loops across {len(cnt)} keys → {args.out}")
    print("per key:", dict(sorted(cnt.items())))
    for s in selected[:6]:
        print(f"  {s['key']:6} {s['bpm']:>3}bpm {s['mood']:10} {os.path.basename(s['path'])[:50]}")


if __name__ == "__main__":
    main()
