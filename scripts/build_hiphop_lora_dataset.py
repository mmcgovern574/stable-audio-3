#!/usr/bin/env python3
"""Generate .txt caption sidecars for the full hip-hop melody-loop library.

Why this exists
---------------
The trainer (scripts/train_lora.py -> caption_metadata_fn) builds a clip's prompt
by reading a sidecar .txt file with the SAME stem next to the audio:

    foo.wav  ->  foo.txt   (prompt = contents of foo.txt)

If the .txt is MISSING, the clip is rejected (__reject__) and silently dropped.
Captions are NOT derived from the folder path or the filename automatically.

Consequences for the Cymatics packs:
  * They ship with NO .txt sidecars -> every one would be dropped, so a naive
    run on the whole library would train on the 46 Crushed Keyz files again.
  * The musically useful info is split: MOOD/GENRE lives in the FOLDER path
    (Dark / Sad / Happy / Ethnic / Nostalgic / Orchestral / Classic / R&B /
    Trap / Future RnB), while KEY + BPM live in the FILENAME.

This script fixes both: it parses key/BPM/mood from path+filename and writes an
attribute-based caption to a sidecar .txt for every *full loop* we want to keep.

Design choices (all to fight the memorization you observed)
-----------------------------------------------------------
* EXCLUDE stems + reverse by default. ~1,390 of ~2,000 files are isolated
  layers ("Loop Stems"/"Stems") or reversed clips. Training on stems teaches
  the model to emit single layers, not full melodic loops. Skipping them also
  works automatically with the trainer: files we don't caption get rejected.
* NO unique track names in captions. Including a per-file identifier is exactly
  what lets the model memorize caption->clip. Captions are built only from
  SHARED attributes (mood, key, BPM, genre) so the model must generalize.
* "crushed keyz" trigger token ONLY on the Crushed Keyz files, so the CKz
  timbre stays summonable at inference while melodic variety is learned from
  the whole set.

Usage
-----
  # dry run: print sample captions + counts, write nothing
  uv run python scripts/build_hiphop_lora_dataset.py --dry-run

  # write .txt sidecars in place + a review CSV/manifest
  uv run python scripts/build_hiphop_lora_dataset.py --write

  # options
  --source DIR       library root (default: SoundSauce/Melody_Loops)
  --include-stems    also caption stem layers (off by default)
  --include-reverse  also caption reversed loops (off by default)
  --report DIR       where to write captions.csv + MANIFEST (default: repo data/)
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff"}

DEFAULT_SOURCE = Path(
    "/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops"
)

# Folder/filename tokens that carry musical meaning -> kept as descriptors.
# Order matters for multi-word genres (check "future rnb" before "rnb").
MOOD_GENRE_PATTERNS = [
    ("future rnb", "future r&b"),
    ("future r&b", "future r&b"),
    ("r&b", "r&b"),
    ("rnb", "r&b"),
    ("trap", "trap"),
    ("orchestral", "orchestral"),
    ("ethnic", "ethnic"),
    ("nostalgic", "nostalgic"),
    ("classic", "classic"),
    ("dark", "dark"),
    ("sad", "sad"),
    ("happy", "happy"),
]

# Brand / packaging noise we never want in a caption.
NOISE_TOKENS = {
    "cymatics", "dragon", "gems", "oracle", "cobra", "platinum", "expansion",
    "collection", "ultimate", "edition", "vol", "pt", "part", "various",
    "melody", "melodies", "loop", "loops", "stems", "stem", "melodics",
    "instrument", "processed", "dry", "wet",
}

CKZ_DIR_HINT = "crushed_keyz"

# --- parsing helpers -------------------------------------------------------

KEY_RE = re.compile(
    r"\b([A-G])\s*(#|b|sharp|flat)?\s*(harmonic\s+)?(min|maj|minor|major)\b",
    re.IGNORECASE,
)
# also handle "Fsharpmin", "Dsharp min", "Cmin", "Amaj" (no space)
KEY_RE_TIGHT = re.compile(
    r"\b([A-G])(sharp|flat|#|b)?\s*(min|maj)\b", re.IGNORECASE
)
BPM_RE = re.compile(r"(\d{2,3})\s*bpm", re.IGNORECASE)
# fallback: a bare 2-3 digit number sitting right before the key token
BPM_BARE_RE = re.compile(r"\b(\d{2,3})\b\s*(?=[A-G](?:#|b|sharp|flat)?\s*(?:harmonic\s+)?(?:min|maj))", re.IGNORECASE)


def normalize_key(text: str) -> str | None:
    m = KEY_RE.search(text) or KEY_RE_TIGHT.search(text)
    if not m:
        return None
    letter = m.group(1).upper()
    acc = (m.group(2) or "").lower()
    acc = {"sharp": "#", "flat": "b", "#": "#", "b": "b", "": ""}.get(acc, "")
    harmonic = ""
    if KEY_RE.search(text) and m.re is KEY_RE and m.group(3):
        harmonic = "harmonic "
    quality_raw = m.group(4) if m.re is KEY_RE else m.group(3)
    quality = "minor" if quality_raw.lower().startswith("min") else "major"
    return f"{letter}{acc} {harmonic}{quality}".strip()


def parse_bpm(text: str) -> int | None:
    m = BPM_RE.search(text)
    if m:
        return int(m.group(1))
    m = BPM_BARE_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def extract_mood_genre(path_text: str) -> list[str]:
    low = path_text.lower()
    found: list[str] = []
    for needle, label in MOOD_GENRE_PATTERNS:
        if needle in low and label not in found:
            found.append(label)
    # "future r&b" already implies r&b — drop the redundant generic tag
    if "future r&b" in found and "r&b" in found:
        found.remove("r&b")
    return found


def is_stem(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    return any("stem" in p for p in parts)


def is_reverse(path: Path) -> bool:
    return any(p.lower() == "reverse" for p in path.parts)


def is_ckz(path: Path) -> bool:
    return any(CKZ_DIR_HINT in p.lower() for p in path.parts)


def build_caption(path: Path) -> tuple[str, dict]:
    """Return (caption, parsed_fields). Caption uses only shared attributes."""
    name = path.stem
    full_text = str(path)  # include folder path so mood/genre is visible

    key = normalize_key(name) or normalize_key(full_text)
    bpm = parse_bpm(name) or parse_bpm(full_text)
    moods = extract_mood_genre(full_text)
    ckz = is_ckz(path)

    parts: list[str] = []
    if ckz:
        parts.append("crushed keyz")
    # genre/mood descriptors
    parts.extend(moods)
    parts.append("hip hop melody loop")
    if key:
        parts.append(f"{key}")
    if bpm:
        parts.append(f"{bpm} bpm")

    caption = ", ".join(parts)
    return caption, {"key": key or "", "bpm": bpm or "", "mood": "|".join(moods), "ckz": ckz}


def iter_audio(source: Path):
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES:
            yield p


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--report", type=Path, default=repo_root / "data")
    ap.add_argument("--include-stems", action="store_true")
    ap.add_argument("--include-reverse", action="store_true")
    ap.add_argument("--raw-names", action="store_true",
                    help="use the exact filename stem as the caption (raw names), "
                         "instead of the composed attribute caption")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true", help="write .txt sidecars in place")
    g.add_argument("--dry-run", action="store_true", help="print samples, write nothing")
    args = ap.parse_args()

    if not args.dry_run and not args.write:
        args.dry_run = True  # safe default

    if not args.source.is_dir():
        raise SystemExit(f"Source not found: {args.source}")

    rows = []
    skipped_stems = skipped_reverse = no_key = no_bpm = 0
    kept = 0

    for audio in iter_audio(args.source):
        if is_stem(audio) and not args.include_stems:
            skipped_stems += 1
            continue
        if is_reverse(audio) and not args.include_reverse:
            skipped_reverse += 1
            continue
        if args.raw_names:
            caption = audio.stem
            fields = {"key": "", "bpm": "", "mood": "", "ckz": is_ckz(audio)}
        else:
            caption, fields = build_caption(audio)
        if not fields["key"]:
            no_key += 1
        if not fields["bpm"]:
            no_bpm += 1
        rows.append((audio, caption, fields))
        kept += 1
        if args.write:
            audio.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")

    if args.write:
        # filelist.txt at the library root makes the trainer load ONLY these
        # full loops (get_audio_filenames honors filelist.txt), so it never
        # scans or reject-churns through the 1,356 excluded stem files.
        filelist = args.source / "filelist.txt"
        filelist.write_text(
            "\n".join(str(a.relative_to(args.source)) for a, _, _ in rows) + "\n",
            encoding="utf-8",
        )
        print(f"filelist written  : {filelist} ({kept} entries)")

    # report
    args.report.mkdir(parents=True, exist_ok=True)
    csv_path = args.report / "hiphop_lora_captions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["relpath", "caption", "key", "bpm", "mood", "ckz"])
        for audio, caption, fields in rows:
            rel = audio.relative_to(args.source)
            w.writerow([str(rel), caption, fields["key"], fields["bpm"], fields["mood"], fields["ckz"]])

    print(f"source            : {args.source}")
    print(f"kept (full loops) : {kept}")
    print(f"  of which CKz     : {sum(1 for _, _, fx in rows if fx['ckz'])}")
    print(f"skipped stems     : {skipped_stems}")
    print(f"skipped reverse   : {skipped_reverse}")
    print(f"missing key       : {no_key}")
    print(f"missing bpm       : {no_bpm}")
    print(f"review CSV        : {csv_path}")
    print(f"mode              : {'WROTE .txt sidecars' if args.write else 'DRY RUN (no files written)'}")
    print()
    print("=== sample captions ===")
    import random
    random.seed(0)
    for audio, caption, _ in random.sample(rows, min(24, len(rows))):
        print(f"  {audio.name[:54]:54s} | {caption}")


if __name__ == "__main__":
    main()
