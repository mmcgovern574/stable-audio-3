#!/usr/bin/env python3
"""Make the LoRA checkpoints LEGIBLE — writes lora_out/CHECKPOINTS.md.

Problem: each continuation run loads weights via --lora_checkpoint but starts a
FRESH Lightning step counter, so `step=8000` in a later run is NOT 8000 cumulative.
You had to read file mtimes to know which checkpoint is newest / most-trained.

This scans lora_out/, orders runs chronologically (mtime), and for the aug-model
lineage computes APPROX cumulative steps (assuming each run continued from the
prior run's last checkpoint), flags the newest, and writes a readable table.

    uv run python scripts/checkpoint_manifest.py
"""
from __future__ import annotations
import re, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LORA = REPO / "lora_out"


def step_of(name):
    m = re.search(r"step=(\d+)", name)
    return int(m.group(1)) if m else -1


def main():
    runs = {}
    for d in sorted(p for p in LORA.iterdir() if p.is_dir()):
        cks = sorted(d.glob("*.ckpt"), key=lambda p: step_of(p.name))
        if cks:
            runs[d.name] = [(step_of(p.name), p.stat().st_mtime, p) for p in cks]
    if not runs:
        print("no checkpoints found"); return

    # newest checkpoint overall
    newest = max((c for v in runs.values() for c in v), key=lambda c: c[1])[2]

    # aug lineage in mtime order -> cumulative
    aug = sorted([r for r in runs if r.startswith("ckz_aug")],
                 key=lambda r: min(m for _, m, _ in runs[r]))
    cum_base = {}
    acc = 0
    for r in aug:
        cum_base[r] = acc
        acc += max(s for s, _, _ in runs[r])

    lines = ["# LoRA checkpoint manifest", "",
             f"_auto-generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/checkpoint_manifest.py_", "",
             "**Gotcha:** continuation runs restart the step counter at 0, so the `step=` in a",
             "filename is the SEGMENT step, not cumulative. Cumulative below is APPROX (assumes",
             "each run continued from the prior run's last checkpoint).", "",
             f"**NEWEST checkpoint:** `{newest.parent.name}/{newest.name}`", ""]

    # aug family with cumulative
    lines.append("## aug model (current production lineage)\n")
    lines.append("| run (chrono) | file | segment step | ~cumulative | modified |")
    lines.append("|---|---|---|---|---|")
    for r in aug:
        for s, m, p in runs[r]:
            cum = cum_base[r] + s
            star = "  ⭐NEWEST" if p == newest else ""
            lines.append(f"| {r} | {p.name} | {s} | ~{cum} | {time.strftime('%m-%d %H:%M', time.localtime(m))}{star} |")
    lines.append(f"\n→ **current best = `ckz_aug_v1_long2/epoch=39-step=8000.ckpt` ≈ cumulative ~{cum_base.get('ckz_aug_v1_long2',0)+8000} steps** (aug-8000).\n")

    # other runs, no cumulative
    others = [r for r in runs if not r.startswith("ckz_aug")]
    if others:
        lines.append("## other runs\n")
        lines.append("| run | checkpoints (step) | newest mtime |")
        lines.append("|---|---|---|")
        for r in sorted(others, key=lambda r: max(m for _, m, _ in runs[r]), reverse=True):
            steps = ",".join(str(s) for s, _, _ in runs[r])
            mt = time.strftime('%m-%d %H:%M', time.localtime(max(m for _, m, _ in runs[r])))
            lines.append(f"| {r} | {steps} | {mt} |")

    out = LORA / "CHECKPOINTS.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
