#!/usr/bin/env python3
"""Copy Crushed Keyz melody loops into a LoRA training folder with .txt captions.

Each audio file gets a matching .txt whose content is the filename stem (no extension),
as required by scripts/train_lora.py.

Usage:
  uv run python scripts/prepare_crushed_keyz_lora_data.py
  uv run python scripts/prepare_crushed_keyz_lora_data.py --source /path/to/loops --dest ./data/crushed_keyz_lora
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg"}


def prepare(source: Path, dest: Path, *, link: bool = False) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p
        for p in source.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )
    if not files:
        raise SystemExit(f"No audio files found in {source}")

    manifest_lines = [
        "# Crushed Keyz LoRA dataset — caption = filename stem",
        f"# source: {source}",
        f"# dest:   {dest}",
        "# idx\taudio\tcaption",
    ]

    for i, src in enumerate(files, start=1):
        dst_audio = dest / src.name
        if link:
            if dst_audio.exists() or dst_audio.is_symlink():
                dst_audio.unlink()
            dst_audio.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst_audio)

        caption = src.stem
        txt_path = dest / f"{src.stem}.txt"
        txt_path.write_text(caption + "\n", encoding="utf-8")
        manifest_lines.append(f"{i:03d}\t{dst_audio.name}\t{caption}")

    manifest_path = dest / "MANIFEST.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print(f"Prepared {len(files)} clips in {dest}")
    print(f"Manifest: {manifest_path}")
    return len(files)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_source = Path(
        "/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops/Crushed_keyz_melody_loops"
    )
    default_dest = repo_root / "data" / "crushed_keyz_lora"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=default_source)
    p.add_argument("--dest", type=Path, default=default_dest)
    p.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink audio instead of copying (saves disk space)",
    )
    args = p.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"Source directory not found: {args.source}")

    prepare(args.source, args.dest, link=args.symlink)


if __name__ == "__main__":
    main()
