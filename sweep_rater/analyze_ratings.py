#!/usr/bin/env python3
"""Aggregate ratings from the central sweep_rater/ratings.json.

Optionally filter to a single folder. Parses params from filenames and
prints marginal mean rating per parameter value plus top-N clips.

Usage:
    # All rated clips across all folders
    python sweep_rater/analyze_ratings.py

    # Only clips in a specific folder
    python sweep_rater/analyze_ratings.py --folder ./eval_out
    python sweep_rater/analyze_ratings.py --folder ./sweep_rater/campaigns/cfg_strength_v1
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
RATINGS_DB = SCRIPT_DIR / "ratings.json"

KNOWN_PARAMS = ("cfg", "strength", "steps", "seed", "duration", "rank", "lr")
_PARAM_RE = re.compile(
    r"(?:^|[_\W])(" + "|".join(KNOWN_PARAMS) + r")(-?\d+(?:\.\d+)?)(?=[_\W]|$)",
    flags=re.IGNORECASE,
)

# init_noise_level is emitted by sweep_matrix.py as `noise<val>` (e.g. `noise0.9`).
# Parse it as its own numeric axis so the analyzer can report it.
_NOISE_RE = re.compile(r"noise([0-9]+(?:\.[0-9]+)?)")
# init_audio appears as `init=<stem>` in the filename — capture the stem as a
# categorical axis. The stem is bounded by `__` separators that sweep_matrix
# uses to delimit param-block / prompt-block.
_INIT_AUDIO_RE = re.compile(r"init=([^_].*?)(?=_(?:noise|cfg|strength|steps|seed|__))")


def parse_filename_params(filename: str) -> dict:
    stem = Path(filename).stem
    out = {}
    for m in _PARAM_RE.finditer(stem):
        key = m.group(1).lower()
        val = m.group(2)
        try:
            out[key] = float(val) if "." in val else int(val)
        except ValueError:
            pass
    m_noise = _NOISE_RE.search(stem)
    if m_noise:
        out["init_noise"] = float(m_noise.group(1))
    m_init = _INIT_AUDIO_RE.search(stem)
    if m_init:
        # Keep the first ~28 chars so it's a readable categorical key.
        out["init_audio"] = m_init.group(1)[:28]
    return out


def entry_timbre(r):
    if r.get("timbre") is not None: return r["timbre"]
    if r.get("rating") is not None: return r["rating"]   # legacy single rating
    return None


def entry_melody(r):
    if r.get("melody") is not None: return r["melody"]
    if r.get("rating") is not None: return r["rating"]   # legacy single rating
    return None


def entry_score(r):
    vals = [x for x in (entry_timbre(r), entry_melody(r)) if x is not None]
    return mean(vals) if vals else None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", type=Path, default=None,
                   help="Optional: filter to ratings under this folder")
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()

    if not RATINGS_DB.exists():
        raise SystemExit(f"No ratings yet at {RATINGS_DB}\nRun rate_server.py and rate some clips first.")
    ratings = json.loads(RATINGS_DB.read_text())
    if not ratings:
        print("ratings.json is empty.")
        return

    filter_prefix = None
    if args.folder:
        try:
            rel = args.folder.resolve().relative_to(REPO_ROOT.resolve())
            filter_prefix = str(rel).rstrip("/") + "/"
            if str(rel) == ".":
                filter_prefix = ""
        except ValueError:
            raise SystemExit(f"--folder must be inside the repo: {args.folder}")

    rows = []
    for key, r in ratings.items():
        if filter_prefix is not None and not key.startswith(filter_prefix):
            continue
        sc = entry_score(r)
        if sc is None:
            continue
        rows.append({
            "key":      key,
            "filename": Path(key).name,
            "folder":   str(Path(key).parent),
            "timbre":   entry_timbre(r),
            "melody":   entry_melody(r),
            "score":    sc,
            "notes":    r.get("notes", ""),
            "ts":       r.get("ts", ""),
            "params":   parse_filename_params(Path(key).name),
        })

    if not rows:
        scope = f"folder {args.folder}" if args.folder else "the DB"
        print(f"No ratings in {scope}.")
        return

    tvals = [r["timbre"] for r in rows if r["timbre"] is not None]
    mvals = [r["melody"] for r in rows if r["melody"] is not None]
    print(f"DB:           {RATINGS_DB}")
    if args.folder:
        print(f"Filtered to:  {args.folder}")
    print(f"Rated clips:  {len(rows)}")
    if tvals:
        print(f"Timbre mean:  {mean(tvals):.2f}  (σ={pstdev(tvals):.2f}, n={len(tvals)})")
    if mvals:
        print(f"Melody mean:  {mean(mvals):.2f}  (σ={pstdev(mvals):.2f}, n={len(mvals)})")

    all_axes = sorted({k for r in rows for k in r["params"]})
    if not all_axes:
        print("\nNo recognized parameters in filenames. Showing only top clips.")
    else:
        for axis in all_axes:
            agg = defaultdict(lambda: {"t": [], "m": []})
            for r in rows:
                if axis in r["params"]:
                    if r["timbre"] is not None: agg[r["params"][axis]]["t"].append(r["timbre"])
                    if r["melody"] is not None: agg[r["params"][axis]]["m"].append(r["melody"])
            if not agg:
                continue
            rankings = []
            for v, d in agg.items():
                tm = mean(d["t"]) if d["t"] else None
                mm = mean(d["m"]) if d["m"] else None
                n = max(len(d["t"]), len(d["m"]))
                rankings.append((v, tm, mm, n))
            rankings.sort(key=lambda x: (-((x[1] or 0) + (x[2] or 0)), x[0]))
            print(f"\n  {axis}  (best → worst by timbre+melody):")
            for v, tm, mm, n in rankings:
                ts = f"{tm:.2f}" if tm is not None else " -- "
                ms = f"{mm:.2f}" if mm is not None else " -- "
                print(f"    {axis}={str(v):<10}  timbre={ts}  melody={ms}  n={n}")

    rows.sort(key=lambda r: -r["score"])
    print(f"\n  Top {min(args.top, len(rows))} clips by (timbre+melody)/2:")
    for r in rows[:args.top]:
        param_str = " ".join(f"{k}={v}" for k, v in r["params"].items()) or "(no params)"
        tt = r["timbre"] if r["timbre"] is not None else "-"
        mm = r["melody"] if r["melody"] is not None else "-"
        line = f"    T{tt} M{mm}  {param_str}"
        if r["notes"]:
            line += f'   "{r["notes"]}"'
        line += f"   → {r['key']}"
        print(line)


if __name__ == "__main__":
    main()
