#!/usr/bin/env python3
"""Pitch-augment the cKz training set to fight memorization (46 loops -> ~3x).

Each loop is pitch-shifted ±N semitones (default ±2), FORMANT-PRESERVING so the
cKz "crushed keys" timbre survives, and the caption's KEY is transposed to match
(Cmin -> Dmin for +2, etc.). BPM is unchanged (pitch-shift preserves tempo).
Presenting each melody in multiple keys breaks song-memorization and forces the
LoRA to learn the STYLE. Output is a new training dir (audio + .txt sidecars).

  python scripts/augment_pitch.py --dry-run        # preview caption key rewrites
  uv run python scripts/augment_pitch.py            # write the augmented set

Pitch-shift backend: pyrubberband (formant-preserving, best — needs `brew install
rubberband` + `pip install pyrubberband`); falls back to torchaudio PitchShift
(phase-vocoder, tempo-preserving but NOT formant-preserving) with a warning.
"""
from __future__ import annotations
import argparse, os, re, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def parse_key(stem):
    s = stem.lower().replace("sharp", "#")
    m = re.search(r"\b([a-g])(#?)\s*(min|maj)\b", s)
    if not m: return None, None
    return (m.group(1).upper() + m.group(2)), ("min" if m.group(3) == "min" else "maj")

def transpose(stem, shift):
    """Return stem with its key token transposed by `shift` semitones (sharp convention)."""
    note, qual = parse_key(stem)
    if note is None: return None
    new = KEYS[(KEYS.index(note) + shift) % 12] + qual
    # replace the first key token (e.g. 'Cmin', 'C#min', 'C min', 'Fsharpmin') with new
    return re.sub(r"\b[A-Ga-g](#|sharp)?\s*(min|maj)\b", new, stem, count=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "data/crushed_keyz_lora"))
    ap.add_argument("--out", default=str(REPO / "data/ckz_aug"))
    ap.add_argument("--shifts", default="-2,2", help="comma semitone shifts (original always kept)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    shifts = [int(s) for s in args.shifts.split(",") if s.strip()]

    src = Path(args.src); out = Path(args.out)
    audio_ext = {}
    for p in src.iterdir():
        if p.suffix.lower() in (".mp3", ".wav", ".flac"):
            audio_ext[p.stem] = p.suffix
    # only caption files that actually have a matching audio loop (skips MANIFEST etc.)
    all_txt = sorted(p.stem for p in src.glob("*.txt"))
    stems = [s for s in all_txt if s in audio_ext]
    skipped_noaudio = [s for s in all_txt if s not in audio_ext]
    if skipped_noaudio:
        print(f"skipping {len(skipped_noaudio)} caption(s) with no audio: {skipped_noaudio}")
    print(f"source loops: {len(stems)}   shifts: {shifts}  → {len(stems)*(1+len(shifts))} total\n")

    # preview / collect jobs
    bad = 0
    print("=== caption key rewrites (sample) ===")
    shown = 0
    plan = []
    for stem in stems:
        note, _ = parse_key(stem)
        if note is None:
            bad += 1
            plan.append((stem, 0, stem))  # keep original, just can't transpose it
            if shown < 30: print(f"  [keep-only, no key in name] {stem[:50]}")
            continue
        plan.append((stem, 0, stem))  # original
        for sh in shifts:
            ns = transpose(stem, sh)
            plan.append((stem, sh, ns))
            if shown < 8:
                print(f"  {note+'min':6} {('+' if sh>0 else '')}{sh} → {parse_key(ns)[0]+'min':6} | {ns[:54]}")
        shown += 1
    print(f"\nplanned outputs: {len(plan)}  (loops with parseable key: {len(stems)-bad}, skipped: {bad})")
    if args.dry_run:
        print("\n--dry-run: no audio written. Re-run without --dry-run to generate.")
        return

    # ---- actually write ----
    out.mkdir(parents=True, exist_ok=True)
    import shutil as _sh, subprocess, tempfile
    import numpy as np, soundfile as sf, torchaudio, torch
    RB = _sh.which("rubberband")
    if RB:
        print(f"pitch backend: rubberband CLI ({RB}) — formant-preserving, R3 engine\n")
    else:
        print("WARNING: `rubberband` binary not on PATH — falling back to torchaudio "
              "PitchShift (NOT formant-preserving). Install: brew install rubberband\n")

    def shift(wav, sr, semis):
        """Pitch-shift tensor [C,N] by `semis` semitones, formant-preserving."""
        if RB:
            with tempfile.TemporaryDirectory() as td:
                ip, op = f"{td}/i.wav", f"{td}/o.wav"
                sf.write(ip, wav.numpy().T, sr)            # soundfile wants [N,C]
                r = subprocess.run([RB, "--pitch", str(semis), "--formant", "--fine",
                                    ip, op], capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"rubberband failed: {r.stderr.strip()[:200]}")
                y, _ = sf.read(op, dtype="float32")
                if y.ndim == 1: y = y[:, None]
                return torch.from_numpy(y.T.copy())
        return torchaudio.transforms.PitchShift(sr, n_steps=semis)(wav).detach()

    cache = {}
    def load(stem):
        if stem not in cache:
            cache[stem] = torchaudio.load(str(src / (stem + audio_ext[stem])))
        return cache[stem]
    n = 0
    for stem, sh, new_stem in plan:
        outwav = out / (new_stem + ".wav")
        txtout = out / (new_stem + ".txt")
        if outwav.exists() and txtout.exists():   # resume: skip already-written
            n += 1; continue
        wav, sr = load(stem)
        ys = wav if sh == 0 else shift(wav, sr, sh)
        torchaudio.save(str(outwav), ys, sr)
        txtout.write_text(new_stem)
        n += 1
        if n % 20 == 0: print(f"  ...{n}/{len(plan)}")
    print(f"\nwrote {n} clips + captions → {out}")
    print(f"train on it:  DATA_DIR={out} SAVE_DIR=./lora_out/ckz_aug_v1 bash scripts/train_ckz_melody_capped.sh")

if __name__ == "__main__":
    main()
