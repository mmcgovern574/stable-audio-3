#!/usr/bin/env python3
"""Cross-check the KEY in each cKz loop's filename against audio key-detection.

Before pitch-augmenting, verify the original labels are right — a wrong label
propagates into every transposed copy. This detects each loop's key from the
AUDIO and compares it to the label parsed from the filename, flagging the ones
worth re-checking by ear.

Since the cKz set is almost entirely MINOR, this assumes minor by default and
compares only the TONIC (root) pitch class — major/minor mode-detection is the
least reliable part of any key detector and is nearly constant here anyway. A
relative-major detection (e.g. D#maj for a Cmin label) is collapsed to its minor
tonic, so it counts as agreement. Mismatches are reported as a SEMITONE OFFSET
(e.g. OFF +2st = audio sounds two semitones above the label), which tells you
exactly how to fix the filename. Pass --allow-major to also detect mode.

Backend: prefers essentia (KeyExtractor, 'edma' profile — best for electronic/
hip-hop); falls back to librosa Krumhansl-Schmuckler template matching.

    uv pip install librosa        # easy path
    # (optional, more accurate)  pip install essentia
    uv run python scripts/detect_key.py

CAVEAT: auto key-detection is ~70-85% accurate and also confuses the dominant
(±5/7 semitones). Treat offsets as "listen to this one," not proof.
"""
from __future__ import annotations
import argparse, csv, os, re, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT2SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
              "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}

KS_MAJ = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KS_MIN = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])


def norm_note(n):
    n = n.strip().capitalize()
    return FLAT2SHARP.get(n, n)


def idx(note):
    return KEYS.index(norm_note(note))


def label_key(stem):
    s = stem.lower().replace("sharp", "#")
    m = re.search(r"\b([a-g])(#?|b?)\s*(min|maj)\b", s)
    if not m:
        return None
    note = norm_note(m.group(1).upper() + (m.group(2) or ""))
    return note, ("minor" if m.group(3) == "min" else "major")


def minor_tonic_pc(note, mode):
    """Collapse any key to the pitch class of its (relative) MINOR tonic."""
    i = idx(note)
    return i if mode == "minor" else (i - 3) % 12   # major Y -> relative minor Y-3


def detect_essentia(path, assume_minor):
    import essentia.standard as es
    audio = es.MonoLoader(filename=path)()
    key, scale, strength = es.KeyExtractor(profileType="edma")(audio)
    return (norm_note(key), scale), float(strength)


def detect_librosa(path, assume_minor):
    import librosa
    y, sr = librosa.load(path, mono=True)
    yt, _ = librosa.effects.trim(y, top_db=30)
    if len(yt) > sr:
        y = yt
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    T = min(chroma.shape[1], len(rms))
    prof = (chroma[:, :T] * rms[:T]).sum(1)            # energy-weighted chroma
    prof = prof / (prof.sum() + 1e-9)
    best = None
    for r in range(12):
        cands = [("minor", np.corrcoef(prof, np.roll(KS_MIN, r))[0, 1])]
        if not assume_minor:
            cands.append(("major", np.corrcoef(prof, np.roll(KS_MAJ, r))[0, 1]))
        for mode, c in cands:
            if best is None or c > best[2]:
                best = (KEYS[r], mode, c)
    return (best[0], best[1]), float(best[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(REPO / "data/crushed_keyz_lora"))
    ap.add_argument("--csv", default=str(REPO / "data/key_check.csv"))
    ap.add_argument("--allow-major", action="store_true",
                    help="also detect major mode (default: assume minor, compare tonic only)")
    args = ap.parse_args()
    assume_minor = not args.allow_major

    try:
        import essentia.standard  # noqa
        detect, backend = detect_essentia, "essentia (edma)"
    except Exception:
        try:
            import librosa  # noqa
            detect, backend = detect_librosa, "librosa (Krumhansl-Schmuckler)"
        except Exception:
            sys.exit("Need a backend: `uv pip install librosa`  (or pip install essentia)")
    print(f"backend: {backend}   mode: {'tonic-only (assume minor)' if assume_minor else 'full key'}\n")

    src = Path(args.src)
    audio_for = {p.stem: p for p in src.iterdir()
                 if p.suffix.lower() in (".mp3", ".wav", ".flac")}
    rows = []
    for stem in sorted(audio_for):
        lab = label_key(stem)
        try:
            det, conf = detect(str(audio_for[stem]), assume_minor)
        except Exception as e:
            print(f"  FAIL {stem[:40]}: {e}"); continue
        if lab is None:
            verdict, delta = "no-label", None
        else:
            lt, dt = minor_tonic_pc(*lab), minor_tonic_pc(*det)
            d = (dt - lt) % 12
            delta = d if d <= 6 else d - 12          # signed nearest
            verdict = "OK" if delta == 0 else f"OFF {delta:+d}st"
        rows.append({"verdict": verdict, "delta": delta, "conf": conf,
                     "label": f"{lab[0]}{lab[1][:3]}" if lab else "—",
                     "detected": f"{det[0]}{det[1][:3]}", "file": stem})

    def rank(r):
        if r["verdict"] == "OK": return (3, 0)
        if r["verdict"] == "no-label": return (2, 0)
        return (0, -abs(r["delta"]), -r["conf"])   # mismatches first, biggest+confident on top
    rows.sort(key=rank)

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["verdict", "conf", "label", "detected", "file"])
        w.writeheader()
        for r in rows:
            w.writerow({"verdict": r["verdict"], "conf": f"{r['conf']:.2f}",
                        "label": r["label"], "detected": r["detected"], "file": r["file"]})

    n_ok = sum(r["verdict"] == "OK" for r in rows)
    n_off = sum(r["verdict"].startswith("OFF") for r in rows)
    n_nl = sum(r["verdict"] == "no-label" for r in rows)
    print(f"{'verdict':9} {'conf':5} {'label':7} {'detect':7}  file")
    print("-" * 74)
    for r in rows:
        mark = "  <-- check by ear" if r["verdict"].startswith("OFF") else ""
        print(f"{r['verdict']:9} {r['conf']:.2f}  {r['label']:7}{r['detected']:7}  {r['file'][:32]}{mark}")
    print(f"\nOK(tonic matches):{n_ok}   OFF(mismatch):{n_off}   no-label:{n_nl}   → {args.csv}")
    print("OFF +Nst = the audio sounds N semitones above the label. Listen; if the audio")
    print("is right, bump the filename key by N, THEN re-run augment_pitch.py.")


if __name__ == "__main__":
    main()
