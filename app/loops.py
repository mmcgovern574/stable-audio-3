"""Foreign init-loop library: parse, filter (minor + dense), index, sample.

Mirrors scripts/init_sweep.py:select_loops. The discovery engine repaints a
foreign minor-key, dense melody loop in cKz timbre, so the library we sample
from must be exactly those loops. Build the manifest once on the Mac (the wavs
live there) with index_loops.py; the server samples from the JSON manifest.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from . import config as C

# "… 140 BPM D Min" / "… 97 BPM A# Min"
_CY = re.compile(r"(\d{2,3})\s*BPM\s+([A-Ga-g])(#|b)?\s*(Min|Maj)", re.I)


def parse_cy(name: str) -> Optional[tuple[str, int]]:
    """Return ('A#min', 140) parsed from a Cymatics filename, or None."""
    m = _CY.search(name)
    if not m:
        return None
    bpm = int(m.group(1))
    note = m.group(2).upper() + (m.group(3) or "")
    note = C.FLAT2SHARP.get(note, note)
    qual = "min" if m.group(4).lower().startswith("min") else "maj"
    return note + qual, bpm


def scan_dirs(root: Path, dirs: list[str], minor_only: bool = True) -> list[dict]:
    """Walk the loop dirs, parse every wav/mp3 we can read a key+bpm from."""
    found = []
    for d in dirs:
        p = Path(root) / d
        if not p.is_dir():
            continue
        for f in sorted(p.glob("*.wav")) + sorted(p.glob("*.mp3")):
            kb = parse_cy(f.name)
            if not kb:
                continue
            key, bpm = kb
            if minor_only and key.endswith("maj"):
                continue
            found.append({"path": str(f), "key": key, "bpm": bpm, "name": f.name})
    return found


def build_manifest(
    root: Path = C.LOOPS_ROOT,
    dirs: Optional[list[str]] = None,
    minor_only: bool = True,
    dense: bool = True,
    max_silence: float = C.MAX_INIT_SILENCE,
) -> list[dict]:
    """Build the discovery loop manifest. Runs on the Mac (reads the wavs).

    `dense` checks each loop's own silence-fraction and drops sparse ones (a
    sparse init seeds a sparse output — the #1 floor killer). Requires soundfile.
    """
    dirs = dirs or C.LOOP_DIRS
    loops = scan_dirs(root, dirs, minor_only=minor_only)
    if dense:
        from .density import silence_fraction

        kept, skipped = [], 0
        for lp in loops:
            sil = silence_fraction(lp["path"])
            if sil is None:  # unreadable — keep, let runtime decide
                lp["silence"] = None
                kept.append(lp)
                continue
            lp["silence"] = round(sil, 3)
            if sil > max_silence:
                skipped += 1
                continue
            kept.append(lp)
        loops = kept
        print(f"  dense filter: dropped {skipped} sparse loops (silence > {max_silence})")
    return loops


class LoopLibrary:
    """Runtime sampler over the prebuilt manifest."""

    def __init__(self, loops: list[dict]):
        self.loops = loops
        self._bykey: dict[str, list[dict]] = defaultdict(list)
        for lp in loops:
            self._bykey[lp["key"]].append(lp)

    @classmethod
    def load(cls, manifest: Path = C.LOOPS_MANIFEST) -> "LoopLibrary":
        if not manifest.exists():
            return cls([])
        data = json.loads(manifest.read_text())
        return cls(data.get("loops", data) if isinstance(data, dict) else data)

    def __len__(self) -> int:
        return len(self.loops)

    @property
    def keys(self) -> list[str]:
        return sorted(self._bykey)

    def sample(self, n: int, rng: Optional[random.Random] = None) -> list[dict]:
        """Pick n loops spread across keys (round-robin), allowing repeats per key."""
        rng = rng or random
        if not self.loops:
            return []
        pools = {k: list(v) for k, v in self._bykey.items()}
        for v in pools.values():
            rng.shuffle(v)
        keys = sorted(pools)
        rng.shuffle(keys)
        out, i = [], 0
        guard = 0
        while len(out) < n and guard < n * 50:
            guard += 1
            k = keys[i % len(keys)]
            i += 1
            if pools[k]:
                out.append(pools[k].pop())
            elif not any(pools.values()):
                # exhausted unique loops — allow reuse to fill the request
                out.append(rng.choice(self.loops))
        return out
