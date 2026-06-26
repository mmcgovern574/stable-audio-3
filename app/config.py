"""Locked cKz generation settings — single source of truth for the UI backend.

Every value here is SETTLED by experiment (see PRODUCT_HANDOFF.md). Do NOT tune
these to "improve" output; they were chosen the hard way. The server and engine
read everything from this module so there is exactly one place the recipe lives.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Repo / model ────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]

BASE_MODEL = "medium-base"  # NOT post-trained "medium"
LORA_CKPT = "lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt"  # "aug-8000", production
LORA_STRENGTH = 1.0  # ALWAYS 1.0 — lowering hurts, no diversity gain (proven)

# ── Generation params (settled — do not re-sweep) ───────────────────────────
CFG_SCALE = 3.0          # higher distorts with init
STEPS = 50               # 150 is identical
SAMPLER = "dpmpp"        # euler ok; rk4/pingpong bad
MAX_DURATION = 20.0      # seconds; gen_dur = min(loop_len, MAX_DURATION)

# ── σ (init_noise_level) ────────────────────────────────────────────────────
# Discovery repaints a foreign melody skeleton in cKz timbre.
SIGMA_DISCOVER = 0.75
# Refine holds the picked melody, moves the timbre.
SIGMA_REFINE_SUBTLE = 0.45   # top quality, gentle tweak
SIGMA_REFINE_BOLD = 0.75     # bold new timbre, melody still preserved
# AVOID 0.5–0.6 ("half-baked valley") and 0.8+ (degrades).

# ── Discovery: foreign init-loop library ────────────────────────────────────
# Minor-key + DENSE loops only. Major keys = 0 keepers; sparse seeds sparse out.
LOOPS_ROOT = Path(
    os.environ.get("CKZ_LOOPS_ROOT", "/Users/mikemcgovern/Documents/SoundSauce/Melody_Loops")
)
LOOP_DIRS = [
    "Cymatics_Dragon_Platinum_Expansion",
    "Cymatics_Gems_Vol_1_Trap_Melodies",
    "Cymatics_2020_Melody_Collection",
    "Cymatics Gems Vol 6 - Future RnB Melodies",
]
MAX_INIT_SILENCE = 0.12  # skip foreign loops whose own silence-fraction exceeds this

# Evocative one-word noun titles. Interchangeable — NOT a diversity lever; a
# novel (non-training-label) title is what yields a novel melody.
TITLES = [
    "Mirage", "Obsidian", "Cascade", "Solaris", "Wraith", "Nocturne",
    "Voltage", "Velvet", "Lotus", "Pulse", "Cipher", "Ember",
    "Halcyon", "Onyx", "Zephyr", "Cobalt", "Specter", "Aurora",
]

# ── Density reject-gate (the one safe automation) ───────────────────────────
# Silence is the #1 floor killer (duds ~11% vs keepers ~2%). Flag, don't delete.
GATE_MAX_SILENCE = 0.10
GATE_MAX_DYN = 30.0

# ── Storage ─────────────────────────────────────────────────────────────────
APP_DIR = REPO / "app"
LOOPS_MANIFEST = APP_DIR / "loops_manifest.json"   # built by index_loops.py
GEN_DIR = APP_DIR / "gen"                          # generated wavs (gitignored)
KEPT_DIR = APP_DIR / "kept"                        # kept wavs (gitignored)
LIBRARY_JSON = KEPT_DIR / "library.json"           # kept-clip manifest

# ── Key helpers (sharp spelling) ────────────────────────────────────────────
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT2SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def transpose(keyqual: str, semis: int) -> str:
    """Transpose a 'Xmin'/'Xmaj' key by `semis` semitones (sharp spelling)."""
    qual = "min" if keyqual.endswith("min") else "maj"
    note = keyqual[:-3]
    note = FLAT2SHARP.get(note, note)
    return KEYS[(KEYS.index(note) + semis) % 12] + qual


def tritone(keyqual: str) -> str:
    """A tritone away (+6). Proven equal-quality, distinct melody from same loop."""
    return transpose(keyqual, 6)
