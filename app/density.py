"""Density reject-gate — the ONE safe automation (see PRODUCT_HANDOFF.md).

Judges ONLY sparseness (silence / dynamic-range gaps): objective, novelty-neutral,
no melody/timbre/memorization bias. Silence is the #1 floor killer. We reuse the
existing DSP in sweep_rater/scorer/extract_features.py so the math is identical to
scripts/density_gate.py. If soundfile/scorer is unavailable we degrade gracefully
(treat as non-sparse) rather than crash.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from . import config as C

_FEATS = None
_TRIED = False


def _feats():
    global _FEATS, _TRIED
    if _TRIED:
        return _FEATS
    _TRIED = True
    try:
        sys.path.insert(0, str(C.REPO / "sweep_rater" / "scorer"))
        from extract_features import feats  # type: ignore

        _FEATS = feats
    except Exception:
        _FEATS = None
    return _FEATS


def silence_fraction(path: str) -> Optional[float]:
    """Fraction of the clip below -45 dB, or None if it can't be measured."""
    f = _feats()
    if f is None:
        return None
    try:
        fe = f(str(path))
        if not fe:
            return None
        return float(fe.get("silence", 0.0))
    except Exception:
        return None


def gate(path: str,
         max_silence: float = C.GATE_MAX_SILENCE,
         max_dyn: float = C.GATE_MAX_DYN) -> dict:
    """Return {'sparse': bool, 'silence': float|None, 'dyn': float|None}.

    sparse == True means: drop-air dud (silence or gappy dynamics too high).
    Never a hard delete — the UI just flags it so ears aren't wasted.
    """
    f = _feats()
    if f is None:
        return {"sparse": False, "silence": None, "dyn": None}
    try:
        fe = f(str(path)) or {}
    except Exception:
        return {"sparse": False, "silence": None, "dyn": None}
    sil = float(fe.get("silence", 0.0))
    dyn = float(fe.get("dyn", 18.0))
    return {
        "sparse": (sil > max_silence) or (dyn > max_dyn),
        "silence": round(sil, 3),
        "dyn": round(dyn, 1),
    }
