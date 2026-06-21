#!/usr/bin/env python3
"""Trim each Crushed Keyz loop to its dense MELODY section, dropping the trailing
gap + step-outs (isolated solo instruments) that teach the model a sparse,
single-instrument mode.

Heuristic: each loop is ~60s of continuous melody, then a silence gap, then the
step-outs. We find the FIRST sustained silence gap occurring after --min-keep
seconds and cut the file there. Files with no such gap are kept whole.

Run with --dry-run first to review the cut points, then --write.

Usage:
  uv run python scripts/trim_ckz_melody.py --dry-run
  uv run python scripts/trim_ckz_melody.py --write
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import torch, torchaudio

SRC = Path("/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops/Crushed_keyz_melody_loops")
DEST = Path("data/ckz_melody_trimmed")


def find_melody_end(wav: torch.Tensor, sr: int, *, min_keep: float, gap_sec: float,
                    thresh_db: float) -> float:
    """Return seconds at which to cut: start of the first sustained silence gap
    after `min_keep` seconds. Returns full length if none found."""
    mono = wav.mean(0)
    frame = int(0.05 * sr)                      # 50 ms frames
    n = mono.shape[0] // frame
    if n == 0:
        return mono.shape[0] / sr
    rms = torch.stack([mono[i*frame:(i+1)*frame].pow(2).mean().sqrt() for i in range(n)])
    peak = rms.max().clamp(min=1e-9)
    db = 20 * torch.log10((rms / peak).clamp(min=1e-9))
    silent = db < thresh_db                     # True where ~silence
    gap_frames = max(1, int(gap_sec / 0.05))
    start_frame = int(min_keep / 0.05)
    run = 0
    for i in range(start_frame, n):
        run = run + 1 if silent[i] else 0
        if run >= gap_frames:
            return (i - gap_frames + 1) * 0.05   # start of the gap
    return mono.shape[0] / sr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--min-keep", type=float, default=30.0, help="never cut before this many seconds")
    ap.add_argument("--gap-sec", type=float, default=1.2, help="silence duration that counts as the gap")
    ap.add_argument("--thresh-db", type=float, default=-38.0, help="below this (rel. to peak) = silence")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.write:
        args.dry_run = True

    files = sorted(p for p in args.src.iterdir() if p.suffix.lower() in {".mp3", ".wav", ".flac"})
    if args.write:
        args.dest.mkdir(parents=True, exist_ok=True)
    print(f"{'orig':>7} {'trim':>7}  file")
    kept_whole = 0
    for p in files:
        wav, sr = torchaudio.load(str(p))
        orig = wav.shape[1] / sr
        cut = find_melody_end(wav, sr, min_keep=args.min_keep, gap_sec=args.gap_sec, thresh_db=args.thresh_db)
        if cut >= orig - 0.5:
            kept_whole += 1
        flag = "  (whole)" if cut >= orig - 0.5 else ""
        print(f"{orig:7.1f} {cut:7.1f}  {p.name}{flag}")
        if args.write:
            trimmed = wav[:, : int(cut * sr)]
            out = args.dest / p.name
            torchaudio.save(str(out), trimmed, sr)
            (args.dest / (p.stem + ".txt")).write_text(p.stem + "\n", encoding="utf-8")

    print(f"\n{len(files)} files, {kept_whole} kept whole (no gap found).")
    if args.write:
        print(f"Wrote trimmed loops + .txt captions to {args.dest}")
    else:
        print("DRY RUN — review cut points, then re-run with --write")


if __name__ == "__main__":
    main()
