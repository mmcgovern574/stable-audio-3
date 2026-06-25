#!/usr/bin/env python3
"""Import newly-dropped cKz loops into the training source folder.

Copies loops added to the raw drop folder into data/crushed_keyz_lora, creates a
.txt caption for each (caption = filename stem, matching the existing set), and
normalizes short-form minor keys (e.g. 'F#m' -> 'F#min') so the key parser and
augment_pitch.py can read them.

"New" loops are selected by modification time (added after --since, default
2026-01-01), so it never re-touches or duplicates the original 2025 set.

    uv run python scripts/import_new_loops.py --dry-run
    uv run python scripts/import_new_loops.py
"""
from __future__ import annotations
import argparse, re, shutil, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW_DEFAULT = "/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops/Crushed_keyz_melody_loops"
KEYRE = re.compile(r"\b[A-Ga-g](#|sharp)?\s*(min|maj)\b", re.I)
SHORTRE = re.compile(r"\b([A-Ga-g](?:#|b)?)m\b")   # 'F#m' short minor -> 'F#min'
AUDIO = (".mp3", ".wav", ".flac")


def normalize_key(stem):
    """Ensure the stem has a parseable full key token; fix short-form 'Xm'."""
    if KEYRE.search(stem.lower().replace("sharp", "#")):
        return stem, False
    new, n = SHORTRE.subn(lambda m: m.group(1) + "min", stem, count=1)
    return (new, True) if n else (stem, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--dst", default=str(REPO / "data/crushed_keyz_lora"))
    ap.add_argument("--since", default="2026-01-01", help="import loops modified after this date")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    raw, dst = Path(args.raw), Path(args.dst)
    cutoff = time.mktime(time.strptime(args.since, "%Y-%m-%d"))

    new = [p for p in raw.iterdir()
           if p.suffix.lower() in AUDIO and p.stat().st_mtime > cutoff]
    new.sort(key=lambda p: p.name)
    print(f"raw: {raw}\nnew loops (modified after {args.since}): {len(new)}\n")

    imported = 0
    for p in new:
        stem = p.stem
        new_stem, fixed = normalize_key(stem)
        m = KEYRE.search(new_stem.lower().replace("sharp", "#"))
        key = (m.group(0)) if m else "NO KEY"
        tag = "  (F#m->F#min)" if fixed else ""
        dst_audio = dst / (new_stem + p.suffix)
        exists = dst_audio.exists()
        print(f"  [{key:8}] {new_stem[:54]}{tag}{'  ALREADY THERE' if exists else ''}")
        if args.dry_run or exists:
            continue
        shutil.copy2(p, dst_audio)
        (dst / (new_stem + ".txt")).write_text(new_stem)
        imported += 1

    if args.dry_run:
        print(f"\nDRY-RUN: would import {len(new)} loops + captions into {dst}")
    else:
        print(f"\nimported {imported} loops + captions -> {dst}")
        print("next:  uv run python scripts/fix_keys.py  ->  detect_key.py  ->  augment_pitch.py")


if __name__ == "__main__":
    main()
