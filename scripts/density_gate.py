#!/usr/bin/env python3
"""Conservative density reject-gate — pull sparse/dead-air duds out of a harvest folder.

The #1 floor killer (across 1,706 rated clips) is SILENCE: duds average 11% dead air
vs 2% for keepers (r=-0.30); gappy dynamic range is #2 (duds 28dB vs 18dB). This gate
removes clips with egregious silence/gaps so you don't waste ears on obvious duds.

It is the SAFE slice of the dropped auto-scorer: it judges ONLY sparseness (silence,
dyn) — objective, novelty-neutral, NO melody/timbre/memorization bias. It never PICKS
winners; survivors are still rated by ear. Reversible: flagged clips move to <folder>/
_sparse/ (not deleted). Run AFTER a harvest finishes generating (don't move mid-write).

    uv run python scripts/density_gate.py --folder sweep_rater/campaigns/<name> --dry-run
    uv run python scripts/density_gate.py --folder sweep_rater/campaigns/<name>
"""
from __future__ import annotations
import argparse, shutil, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sweep_rater/scorer"))
from extract_features import feats  # numpy DSP (silence, dyn) — no torch, no scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--max-silence", type=float, default=0.10,
                    help="reject if silence fraction exceeds this (keepers avg 0.02, duds 0.11)")
    ap.add_argument("--max-dyn", type=float, default=30.0,
                    help="reject if dynamic-range dB exceeds this (keepers avg 18, duds 28)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")
    wavs = sorted(p for p in folder.glob("*.wav"))
    flagged, kept = [], 0
    for p in wavs:
        try:
            fe = feats(str(p))
        except Exception:
            kept += 1; continue
        sil = float(fe.get("silence", 0.0)); dyn = float(fe.get("dyn", 18.0))
        if sil > args.max_silence or dyn > args.max_dyn:
            flagged.append((sil, dyn, p))
        else:
            kept += 1

    print(f"folder: {folder}")
    print(f"thresholds: silence > {args.max_silence}  OR  dyn > {args.max_dyn} dB")
    print(f"clips: {len(wavs)}   keep: {kept}   flag-sparse: {len(flagged)} "
          f"({100*len(flagged)/max(len(wavs),1):.0f}%)\n")
    for sil, dyn, p in sorted(flagged, reverse=True)[:25]:
        print(f"   sparse  silence={sil:.2f} dyn={dyn:.0f}  {p.name[:50]}")
    if args.dry_run:
        print("\n--dry-run: nothing moved. Re-run without --dry-run to move these to _sparse/.")
        print("Tune with --max-silence / --max-dyn (lower = more aggressive).")
        return
    if not flagged:
        print("nothing to move."); return
    sparse = folder / "_sparse"; sparse.mkdir(exist_ok=True)
    for sil, dyn, p in flagged:
        shutil.move(str(p), str(sparse / p.name))
    print(f"\nmoved {len(flagged)} sparse clips -> {sparse}/  (reversible).")
    print(f"Now rate only the dense survivors:")
    print(f"  cd ~/stable-audio-3 && uv run python sweep_rater/rate_server.py --folder {folder} --port 8799")


if __name__ == "__main__":
    main()
