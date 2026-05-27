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
    return out


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
        rows.append({
            "key":      key,
            "filename": Path(key).name,
            "folder":   str(Path(key).parent),
            "rating":   r["rating"],
            "notes":    r.get("notes", ""),
            "ts":       r.get("ts", ""),
            "params":   parse_filename_params(Path(key).name),
        })

    if not rows:
        scope = f"folder {args.folder}" if args.folder else "the DB"
        print(f"No ratings in {scope}.")
        return

    scores = [r["rating"] for r in rows]
    print(f"DB:           {RATINGS_DB}")
    if args.folder:
        print(f"Filtered to:  {args.folder}")
    print(f"Rated clips:  {len(rows)}")
    print(f"Mean rating:  {mean(scores):.2f}  (σ={pstdev(scores):.2f}, "
          f"min={min(scores)}, max={max(scores)})")

    all_axes = sorted({k for r in rows for k in r["params"]})
    if not all_axes:
        print("\nNo recognized parameters in filenames. Showing only top clips.")
    else:
        for axis in all_axes:
            by_val = defaultdict(list)
            for r in rows:
                if axis in r["params"]:
                    by_val[r["params"][axis]].append(r["rating"])
            if not by_val:
                continue
            rankings = [(v, mean(s), len(s)) for v, s in by_val.items()]
            rankings.sort(key=lambda x: (-x[1], x[0]))
            print(f"\n  {axis} (best → worst):")
            for v, m, n in rankings:
                bar = "█" * round(m * 4)
                print(f"    {axis}={str(v):<8}  mean={m:.2f}  n={n:<3}  {bar}")

    rows.sort(key=lambda r: -r["rating"])
    print(f"\n  Top {min(args.top, len(rows))} clips by rating:")
    for r in rows[:args.top]:
        param_str = " ".join(f"{k}={v}" for k, v in r["params"].items()) or "(no params)"
        line = f"    {r['rating']}★  {param_str}"
        if r["notes"]:
            line += f'   "{r["notes"]}"'
        line += f"   → {r['key']}"
        print(line)


if __name__ == "__main__":
    main()
