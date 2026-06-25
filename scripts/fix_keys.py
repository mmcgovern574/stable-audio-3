#!/usr/bin/env python3
"""Correct mislabeled keys in the cKz source set, then clear their stale
augmented copies so augment_pitch.py regenerates them from the right root.

These 8 were confirmed by ear (detector right, label wrong); Panda was NOT
corrected (detector picked the dominant — label Cmin is right). Each fix renames
the source audio + .txt + .txt content in crushed_keyz_lora, and deletes every
matching file in ckz_aug (all 3 of its transposed copies) so re-running
augment_pitch.py rebuilds the trio with correct keys.

    uv run python scripts/fix_keys.py --dry-run    # preview
    uv run python scripts/fix_keys.py              # apply
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KEYRE = re.compile(r"\b[A-Ga-g](#|sharp)?\s*(min|maj)\b")

# title substring -> corrected key (minor, sharp spelling). Idempotent: loops
# already at the target key become no-ops. ICEBERG has NO key in its name, so the
# key is INSERTED after the title rather than replaced.
CORR = {
    # round 1 (originals) — already applied, kept for reproducibility (no-ops)
    "DreamDoor": "Bmin",
    "Rent Free": "Gmin",
    "DiMELO":    "Emin",
    "El Yunque": "C#min",
    "Pressure":  "C#min",
    "dAYS OFF":  "Dmin",
    "Steppa":    "Emin",
    "StarLight": "G#min",
    # round 2 (new loops, ear-confirmed)
    "Assault":   "Emin",
    "MEDUSA":    "C#min",
    "Come Back": "F#min",
    "BAMBOO":    "Dmin",
    "BadBunny":  "Dmin",
    "WAVY":      "Gmin",
    "DMT":       "C#min",
    "ICEBERG":   "F#min",   # insert (no key in original name)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "data/crushed_keyz_lora"))
    ap.add_argument("--aug", default=str(REPO / "data/ckz_aug"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    src, aug = Path(args.src), Path(args.aug)

    for title, newkey in CORR.items():
        cands = [p for p in src.iterdir()
                 if p.suffix.lower() in (".mp3", ".wav", ".flac")
                 and title.lower() in p.stem.lower()]
        if len(cands) != 1:
            print(f"!! {title}: expected 1 source audio, found {len(cands)} -> SKIP")
            continue
        ap_ = cands[0]; stem = ap_.stem
        m = KEYRE.search(stem)
        if m:
            newstem = KEYRE.sub(newkey, stem, count=1)
            change = f"{m.group(0)} -> {newkey}"
        else:
            # no key in name -> INSERT it right after the title token
            cut = stem.lower().find(title.lower()) + len(title)
            newstem = stem[:cut] + f" {newkey}" + stem[cut:]
            change = f"(no key) -> insert {newkey}"
        if newstem == stem:
            print(f"\n{title}:  already {newkey} (no-op)")
            continue
        stale = [p for p in aug.iterdir() if title.lower() in p.stem.lower()]
        print(f"\n{title}:  {change}")
        print(f"   src: '{stem}'")
        print(f"     -> '{newstem}'")
        print(f"   aug stale to delete: {len(stale)} files "
              f"({sorted(set(p.suffix for p in stale))})")
        if args.dry_run:
            continue
        ap_.rename(src / (newstem + ap_.suffix))
        txt = src / (stem + ".txt")
        if txt.exists():
            nt = src / (newstem + ".txt")
            txt.rename(nt); nt.write_text(newstem)
        for p in stale:
            p.unlink()

    if args.dry_run:
        print("\nDRY-RUN only — nothing changed.")
    else:
        print("\nApplied. Now regenerate the corrected trios:")
        print("  cd ~/stable-audio-3 && uv run python scripts/augment_pitch.py")


if __name__ == "__main__":
    main()
