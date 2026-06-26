#!/usr/bin/env python3
"""Pre-index the foreign loop library into app/loops_manifest.json.

Run this ONCE on the Mac (the Cymatics wavs live there). The server samples
from the resulting manifest — minor-key + dense loops only, exactly the set the
discovery engine is tuned for.

    uv run python -m app.index_loops              # build manifest
    uv run python -m app.index_loops --no-dense   # skip silence filter (faster)
    uv run python -m app.index_loops --dry-run    # just report counts
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime

from . import config as C
from . import loops as L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(C.LOOPS_ROOT))
    ap.add_argument("--dirs", nargs="*", default=C.LOOP_DIRS)
    ap.add_argument("--max-silence", type=float, default=C.MAX_INIT_SILENCE)
    ap.add_argument("--no-dense", action="store_true", help="skip the silence filter")
    ap.add_argument("--keep-major", action="store_true", help="include major keys (not recommended)")
    ap.add_argument("--out", default=str(C.LOOPS_MANIFEST))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from pathlib import Path
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(
            f"loop root not found: {root}\n"
            f"Set CKZ_LOOPS_ROOT or pass --root to your SoundSauce/Melody_Loops path."
        )

    loops = L.build_manifest(
        root=root, dirs=args.dirs,
        minor_only=not args.keep_major,
        dense=not args.no_dense, max_silence=args.max_silence,
    )
    by_key = Counter(lp["key"] for lp in loops)
    print(f"\nindexed {len(loops)} loops across {len(by_key)} keys")
    for k, n in sorted(by_key.items()):
        print(f"   {k:6} {n}")

    if args.dry_run:
        print("\n--dry-run: manifest NOT written.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "built": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "dirs": args.dirs,
        "minor_only": not args.keep_major,
        "dense": not args.no_dense,
        "max_silence": args.max_silence,
        "count": len(loops),
        "loops": loops,
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
