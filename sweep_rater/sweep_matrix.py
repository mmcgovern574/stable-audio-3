#!/usr/bin/env python3
"""Generate a matrix sweep of stable-audio LoRA inference outputs.

Writes .wav files into an output folder with parameter-encoding filenames.
That's it — no manifest, no extra state. The rater UI reads whatever wav
files exist in the folder (params extracted from filenames at display time).

Usage:
    # Default config (cfg × strength × seed sweep on the ICE training prompt)
    python sweep_rater/sweep_matrix.py

    # Custom config
    python sweep_rater/sweep_matrix.py --config myconfig.json

    # Print combinations without generating audio (sanity check first)
    python sweep_rater/sweep_matrix.py --dry-run

Then rate:
    python sweep_rater/rate_server.py --folder sweep_rater/campaigns/cfg_strength_v1
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_CONFIG = {
    "name": "cfg_strength_v1",
    "lora_ckpt": "./lora_out/ckz_5000_dora/epoch=108-step=5000.ckpt",
    "model": "medium-base",
    "duration": 20,
    "prompt": "2. cKz! ICE Dmin 135 bpm @crushed_keyz",
    "axes": {
        "cfg":      [3, 5, 7, 9],
        "strength": [0.25, 0.5, 0.75, 1.0],
        "steps":    [50],
        "seed":     [42, 123, 999],
    },
}


def sanitize(s: str, max_len: int = 60) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[/\\:|?*<>\"']", "", s)
    s = re.sub(r"_+", "_", s)
    return s[:max_len].rstrip("_.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="JSON config file (defaults to built-in)")
    p.add_argument("--out-root", type=Path, default=SCRIPT_DIR / "campaigns",
                   help="Where output folders live (default: sweep_rater/campaigns)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan, don't run stable-audio")
    p.add_argument("--cwd", type=Path, default=REPO_ROOT,
                   help="Working directory for stable-audio (default: repo root)")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text()) if args.config else DEFAULT_CONFIG

    out_dir = args.out_root / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    axes_order = list(cfg["axes"].keys())
    axes_values = [cfg["axes"][k] for k in axes_order]
    combos = list(itertools.product(*axes_values))
    sanitized_prompt = sanitize(cfg["prompt"])

    print(f"Sweep:        {cfg['name']}")
    print(f"Prompt:       {cfg['prompt']}")
    print(f"LoRA ckpt:    {cfg['lora_ckpt']}")
    print(f"Output:       {out_dir}/")
    print(f"Combinations: {len(combos)}")
    for name in axes_order:
        print(f"  {name}: {cfg['axes'][name]}")
    print(f"\nLaunch the rater now in another terminal:")
    print(f"  python sweep_rater/rate_server.py --folder {out_dir}\n")

    failed = 0
    for i, combo in enumerate(combos, start=1):
        params = dict(zip(axes_order, combo))
        param_str = "_".join(f"{k}{params[k]}" for k in axes_order)
        filename = f"{i:04d}__{param_str}__{sanitized_prompt}.wav"
        out_path = out_dir / filename

        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[{i:>3}/{len(combos)}] skip (exists)  {filename}")
            continue

        cmd = [
            "uv", "run", "stable-audio",
            "--model", cfg["model"],
            "--lora-ckpt-path", cfg["lora_ckpt"],
            "--lora-strength",  str(params.get("strength", 1.0)),
            "-p", cfg["prompt"],
            "--duration",  str(cfg["duration"]),
            "--steps",     str(params.get("steps", 50)),
            "--cfg-scale", str(params.get("cfg", 7)),
            "--seed",      str(params.get("seed", 42)),
            "--output",    str(out_path),
        ]
        print(f"[{i:>3}/{len(combos)}] {filename}")
        if args.dry_run:
            print(f"            would run: {' '.join(cmd)}")
            continue

        result = subprocess.run(cmd, cwd=args.cwd)
        if result.returncode != 0:
            failed += 1
            print(f"            FAILED (exit {result.returncode}); continuing")

    print(f"\nDone. Output: {out_dir}/")
    if failed:
        print(f"Failures: {failed}")
    print(f"Rate: python sweep_rater/rate_server.py --folder {out_dir}")


if __name__ == "__main__":
    main()
