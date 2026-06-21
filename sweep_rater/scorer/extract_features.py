#!/usr/bin/env python3
"""Extract lightweight audio features for every rated clip → features.jsonl (resumable).

Feasibility step for the auto-picker: do cheap spectral/temporal features predict
the timbre + melody ratings well enough to RANK clips? Timbre (brightness/muffle)
should be very learnable from spectral features; melody is the open question.

numpy-only DSP (no torch/librosa). Resumable: re-run until it reports 0 remaining.

    python sweep_rater/scorer/extract_features.py --limit 400   # process a chunk
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import soundfile as sf

REPO = str(Path(__file__).resolve().parents[2])   # repo root (portable: sandbox or Mac)
RAT = os.path.join(REPO, "sweep_rater/ratings.json")
OUT = os.path.join(REPO, "sweep_rater/scorer/features.jsonl")

BAND_EDGES = [0, 250, 500, 1000, 2000, 4000, 8000, 16000, 22050]


def feats(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr > 24000:                      # decimate ~2x for speed
        a = a[::2]; sr //= 2
    a = a[: sr * 12]                    # first 12s
    if len(a) < 2048:
        return None
    N, H = 2048, 1024
    nf = 1 + (len(a) - N) // H
    if nf < 4:
        return None
    idx = np.arange(N)[None, :] + H * np.arange(nf)[:, None]
    frames = a[idx] * np.hanning(N)[None, :]
    mag = np.abs(np.fft.rfft(frames, axis=1)) + 1e-9   # (nf, 1025)
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    psum = mag.sum(1)
    cent = (mag * freqs).sum(1) / psum
    # rolloff 85%
    cums = np.cumsum(mag, 1)
    roll = freqs[(cums >= 0.85 * cums[:, -1:]).argmax(1)]
    bw = np.sqrt((mag * (freqs[None, :] - cent[:, None]) ** 2).sum(1) / psum)
    flat = np.exp(np.log(mag).mean(1)) / (mag.mean(1) + 1e-9)
    hf = mag[:, freqs > 6000].sum(1) / psum            # high-freq ratio (muffle inverse)
    # sub-band log energies
    bands = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        sel = (freqs >= lo) & (freqs < hi)
        bands.append(np.log((mag[:, sel].sum(1)).mean() + 1e-9))
    # spectral flux (onset/activity)
    nmag = mag / psum[:, None]
    flux = np.abs(np.diff(nmag, axis=0)).sum(1).mean() if nf > 1 else 0.0
    # frame RMS / silence / dynamics
    frms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    fdb = 20 * np.log10(frms + 1e-9)
    silence = float((fdb < -45).mean())
    rms = float(np.sqrt((a ** 2).mean() + 1e-12))
    crest = float(np.abs(a).max() / (rms + 1e-9))
    dyn = float(fdb.max() - np.percentile(fdb, 10))
    # crude chroma movement: fold freqs->12 pitch classes, std over time
    pc = (12 * np.log2(np.maximum(freqs, 1e-6) / 440.0) + 69).astype(int) % 12
    chroma = np.zeros((nf, 12))
    for c in range(12):
        chroma[:, c] = mag[:, pc == c].sum(1)
    chroma /= chroma.sum(1, keepdims=True) + 1e-9
    chroma_move = float(np.abs(np.diff(chroma, axis=0)).sum(1).mean()) if nf > 1 else 0.0
    chroma_var = float(chroma.std(0).mean())

    out = {
        "rms": rms, "log_rms": float(np.log(rms + 1e-9)), "crest": crest,
        "silence": silence, "dyn": dyn,
        "cent_m": float(cent.mean()), "cent_s": float(cent.std()),
        "roll_m": float(roll.mean()), "bw_m": float(bw.mean()),
        "flat_m": float(flat.mean()), "hf_ratio": float(hf.mean()),
        "flux": float(flux), "chroma_move": chroma_move, "chroma_var": chroma_var,
    }
    for i, b in enumerate(bands):
        out[f"band{i}"] = float(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--budget", type=float, default=40.0, help="seconds")
    args = ap.parse_args()

    d = json.load(open(RAT))
    def tt(r): return r.get("timbre", r.get("rating"))
    def mm(r): return r.get("melody", r.get("rating"))
    jobs = []
    for k, v in d.items():
        a, b = tt(v), mm(v)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            continue
        p = os.path.join(REPO, k)
        if os.path.exists(p):
            jobs.append((k, p, a, b))

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try: done.add(json.loads(line)["key"])
            except Exception: pass
    todo = [j for j in jobs if j[0] not in done]
    print(f"rated+on-disk: {len(jobs)} | cached: {len(done)} | remaining: {len(todo)}")

    t0 = time.time(); n = 0
    with open(OUT, "a") as f:
        for k, p, a, b in todo:
            if n >= args.limit or time.time() - t0 > args.budget:
                break
            try:
                fe = feats(p)
            except Exception:
                fe = None
            if fe is None:
                continue
            f.write(json.dumps({"key": k, "timbre": a, "melody": b, "f": fe}) + "\n")
            n += 1
    print(f"extracted {n} this run in {time.time()-t0:.1f}s. "
          f"total cached now ~{len(done)+n}/{len(jobs)}. "
          f"{'DONE' if len(todo)-n<=0 else 'run again for more'}")


if __name__ == "__main__":
    main()
