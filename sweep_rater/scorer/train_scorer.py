#!/usr/bin/env python3
"""Train + cross-validate the auto-scorer on cached features → can it RANK?

Answers the feasibility question for the auto-picker:
  * timbre: predictable from spectral features? (expect yes — muffle = low HF)
  * melody: predictable at all? (the hard one)
  * selection: if we keep the top-K by predicted score, are they actually keepers?
"""
from __future__ import annotations
import json, os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import spearmanr

REPO = "/sessions/zen-intelligent-meitner/mnt/stable-audio-3"
FEAT = os.path.join(REPO, "sweep_rater/scorer/features.jsonl")

rows = [json.loads(l) for l in open(FEAT)]
names = sorted(rows[0]["f"].keys())
X = np.array([[r["f"][n] for n in names] for r in rows], dtype=float)
yt = np.array([r["timbre"] for r in rows], dtype=float)
ym = np.array([r["melody"] for r in rows], dtype=float)
print(f"samples: {len(rows)}  features: {len(names)}")
print(f"base rates — keeper(T>=4&M>=4): {100*np.mean((yt>=4)&(ym>=4)):.1f}%  "
      f"T>=4: {100*np.mean(yt>=4):.1f}%  floor(T<=2): {100*np.mean(yt<=2):.1f}%\n")

cv = KFold(5, shuffle=True, random_state=0)
def model(): return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=0)

def evaluate(y, label):
    pred = cross_val_predict(model(), X, y, cv=cv, n_jobs=-1)
    rho = spearmanr(pred, y).correlation
    mae = np.mean(np.abs(pred - y))
    print(f"  {label:8} Spearman {rho:.3f}  MAE {mae:.2f}")
    return pred

print("=== cross-validated prediction accuracy ===")
pt = evaluate(yt, "timbre")
pm = evaluate(ym, "melody")
pc = evaluate(yt + ym, "T+M")

print("\n=== SELECTION test (rank by predicted T+M, out-of-fold) ===")
order = np.argsort(-pc)
actual_TM = yt + ym
keeper = (yt >= 4) & (ym >= 4)
for frac in (0.05, 0.10, 0.20):
    k = max(1, int(len(rows) * frac))
    top = order[:k]
    print(f"  top {int(frac*100):>2}% by score (n={k}): "
          f"actual keeper rate {100*keeper[top].mean():4.1f}%  "
          f"(base {100*keeper.mean():.1f}%)  "
          f"mean actual T{yt[top].mean():.2f}/M{ym[top].mean():.2f}")
# timbre-only gate (its prediction is easier): how clean is the top by predicted timbre?
ot = np.argsort(-pt)
k = int(len(rows)*0.2)
print(f"\n  timbre-gate: top 20% by predicted timbre → actual Tavg {yt[ot[:k]].mean():.2f} "
      f"(overall {yt.mean():.2f}); floor in that top set {100*np.mean(yt[ot[:k]]<=2):.1f}%")

print("\n=== top feature importances (full-fit) ===")
for axis, y in (("timbre", yt), ("melody", ym)):
    m = model().fit(X, y)
    imp = sorted(zip(names, m.feature_importances_), key=lambda x: -x[1])[:8]
    print(f"  {axis}: " + ", ".join(f"{n}={v:.2f}" for n, v in imp))
