#!/usr/bin/env python3
"""Auto-picker: score a folder of generated clips and surface the best.

Trains a RandomForest scorer on the cached rated features (features.jsonl) each
run (fast, ~10s, no pickle/version issues), then for every .wav in --folder
extracts the same features, predicts timbre + melody, ranks by predicted T+M,
writes a ranked CSV, and copies the top-K into <folder>/picks/.

Workflow:
  1) generate a big batch (build_library.py, or any sweep)
  2) python sweep_rater/scorer/pick_best.py --folder <campaign> --top 25
  3) browse <campaign>/picks/  (rate_server.py also works on it)

Validated: top ~5-10% by predicted score were 50-70% actual keepers vs 11.5% base.
Re-run extract_features.py first if you've added new ratings, to refresh training.
"""
from __future__ import annotations
import argparse, csv, json, os, re, shutil, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestRegressor


def tail_audible_ratio(path):
    """Fraction of the clip that is audible before it goes silent (1.0 = full).
    Catches the 'silent after Ns' defect (init loop shorter than generation)."""
    try:
        a, sr = sf.read(path, dtype="float32")
    except Exception:
        return 1.0
    if a.ndim > 1:
        a = a.mean(1)
    if len(a) < sr:
        return 1.0
    fl = int(0.1 * sr); nf = len(a) // fl
    rms = np.array([np.sqrt((a[i*fl:(i+1)*fl]**2).mean()+1e-12) for i in range(nf)])
    aud = np.where(20*np.log10(rms+1e-9) > -45)[0]
    if len(aud) == 0:
        return 0.0
    return float((aud[-1] + 1) / nf)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import feats  # reuse the exact same feature extractor

REPO = str(Path(__file__).resolve().parents[2])
FEAT = os.path.join(REPO, "sweep_rater/scorer/features.jsonl")


def load_training():
    rows = [json.loads(l) for l in open(FEAT)]
    names = sorted(rows[0]["f"].keys())
    X = np.array([[r["f"][n] for n in names] for r in rows], float)
    yt = np.array([r["timbre"] for r in rows], float)
    ym = np.array([r["melody"] for r in rows], float)
    return names, X, yt, ym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="campaign folder of .wav to rank")
    ap.add_argument("--top", type=int, default=25, help="copy top-K to picks/")
    ap.add_argument("--weight-timbre", type=float, default=1.0)
    ap.add_argument("--weight-melody", type=float, default=1.0)
    ap.add_argument("--skip-rated", action="store_true",
                    help="exclude clips already rated in this campaign's picks → each run surfaces NEW candidates")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"not a folder: {folder}")

    names, X, yt, ym = load_training()
    print(f"training scorer on {len(X)} rated clips ...")
    mt = RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X, yt)
    mm = RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(X, ym)

    wavs = [p for p in sorted(folder.glob("*.wav"))]
    print(f"scoring {len(wavs)} clips in {folder.name} ...")
    rows = []
    t0 = time.time()
    for p in wavs:
        try:
            fe = feats(str(p))
        except Exception:
            fe = None
        if fe is None:
            continue
        x = np.array([[fe[n] for n in names]], float)
        pt = float(mt.predict(x)[0]); pm = float(mm.predict(x)[0])
        score = args.weight_timbre * pt + args.weight_melody * pm
        # DENSITY guard — sparseness is the #1 quality killer (silence r=-0.30,
        # dynamic-range r=-0.26). Combine trailing-silence (loop-length defect)
        # with internal-silence + dynamic-range from the features. 1.0 = dense.
        tail = tail_audible_ratio(str(p))
        sil = float(fe.get("silence", 0.0))            # fraction of frames silent
        dyn = float(fe.get("dyn", 18.0))               # dB range; >~25 = gappy
        density = tail * (1.0 - 0.5 * sil) * (1.0 - min(max(dyn - 22, 0) / 40.0, 0.35))
        if density < 0.85:
            score *= density / 0.85
        rows.append((score, pt, pm, density, p.name))
    rows.sort(reverse=True)
    print(f"scored {len(rows)} in {time.time()-t0:.1f}s")

    if args.skip_rated:
        rp = os.path.join(REPO, "sweep_rater/ratings.json")
        ratings = json.load(open(rp)) if os.path.exists(rp) else {}
        tag = f"/{folder.name}/picks/"
        rated_orig = set()
        for k, v in ratings.items():
            if tag in k and (v.get("timbre") is not None or v.get("rating") is not None):
                rated_orig.add(re.sub(r'^\d+_T[\d.]+_M[\d.]+__', '', os.path.basename(k)))
        before = len(rows)
        rows = [r for r in rows if r[4] not in rated_orig]
        print(f"  --skip-rated: dropped {before-len(rows)} already-rated → {len(rows)} fresh candidates")

    # ranked CSV
    csv_path = folder / "picks_ranked.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "pred_score", "pred_timbre", "pred_melody", "density", "file"])
        for i, (s, pt, pm, dens, n) in enumerate(rows, 1):
            w.writerow([i, f"{s:.3f}", f"{pt:.2f}", f"{pm:.2f}", f"{dens:.2f}", n])

    # copy top-K
    picks = folder / "picks"
    shutil.rmtree(picks, ignore_errors=True)   # best-effort (cross-owner safe)
    picks.mkdir(exist_ok=True)
    # carry the init-loop mapping (from the parent library_index.csv) into picks/
    # under the RENAMED filenames, so the rater can show the prompt + play the
    # init loop while you rate the picks.
    parent_idx = {}
    pidx = folder / "library_index.csv"
    if pidx.exists():
        for r in csv.DictReader(open(pidx)):
            parent_idx[r["file"]] = r
    picks_rows = []
    for i, (s, pt, pm, dens, n) in enumerate(rows[: args.top], 1):
        newname = f"{i:02d}_T{pt:.1f}_M{pm:.1f}__{n}"
        shutil.copy2(folder / n, picks / newname)
        info = parent_idx.get(n, {})
        picks_rows.append({"file": newname, "key": info.get("key", ""),
                           "bpm": info.get("bpm", ""), "mood": info.get("mood", ""),
                           "title": info.get("title", ""), "seed": info.get("seed", ""),
                           "prompt": info.get("prompt", ""),
                           "init_loop": info.get("init_loop", ""),
                           "init_path": info.get("init_path", "")})
    if picks_rows and any(r["init_loop"] for r in picks_rows):
        with open(picks / "library_index.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(picks_rows[0].keys()))
            w.writeheader(); w.writerows(picks_rows)

    print(f"\nranked CSV : {csv_path}")
    print(f"top {args.top} → {picks}/")
    print(f"\ntop 10 predicted:")
    for i, (s, pt, pm, dens, n) in enumerate(rows[:10], 1):
        flag = "" if dens >= 0.85 else f"  [sparse {dens*100:.0f}%]"
        print(f"  {i:2}. T{pt:.1f}/M{pm:.1f}  {n[:64]}{flag}")
    print(f"\nBrowse picks: python sweep_rater/rate_server.py --folder {picks} --port 8799")


if __name__ == "__main__":
    main()
