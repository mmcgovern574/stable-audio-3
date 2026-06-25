#!/usr/bin/env python3
"""Split concatenated training-demo wavs into one clip PER MELODY, for per-melody rating.

The trainer writes all demo prompts concatenated into one wav per demo step
(demo_cfg_<c>_<step>.wav, 12s per prompt). The rater scores per FILE, so you'd get
one rating for all 8 melodies. This slices each demo into per-prompt clips and writes
a library_index.csv, so the EXISTING 3-axis rater shows each melody's prompt and lets
you give it its own timbre/melody/novelty stars.

Clips are named so the SAME prompt sorts together across demo steps (compare a melody
as training progresses). Pure stdlib (PCM wav) — no deps. Re-run as new demos appear.

    uv run python scripts/split_demos.py --demos-dir lora_out/ckz_aug_cum23k/demos
    uv run python sweep_rater/rate_server.py --folder lora_out/ckz_aug_cum23k/demos/split --port 8799
"""
from __future__ import annotations
import argparse, csv, re, wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def slug(p):
    m = re.search(r"cKz!\s*([A-Za-z][\w ]*?)\s+[A-G]#?", p) or re.search(r"\b([A-Z][A-Za-z]+)\b", p)
    t = (m.group(1).strip() if m else p[:12]).replace(" ", "_")
    k = re.search(r"\b([A-G]#?min|[A-G]#?maj)\b", p)
    return f"{t[:16]}_{k.group(1) if k else 'X'}"


def parse_step(name):
    m = re.search(r"demo_cfg_[\d.]+_(\d+)\.wav", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos-dir", required=True)
    ap.add_argument("--prompts", default=str(REPO / "data/demo_prompts_ckz_novel.txt"))
    ap.add_argument("--seg-seconds", type=float, default=12.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    demos = Path(args.demos_dir)
    out = Path(args.out or demos / "split"); out.mkdir(parents=True, exist_ok=True)
    prompts = [l.strip() for l in open(args.prompts) if l.strip()]
    train_titles = ["ACTiVE", "AzUL", "ICE", "JUNGLE"]  # mark training-label controls
    rows = []
    n = 0
    for wav in sorted(demos.glob("demo_cfg_*.wav")):
        step = parse_step(wav.name)
        if step is None:
            continue
        w = wave.open(str(wav), "rb")
        sr, ch, sw, total = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        seg = int(args.seg_seconds * sr)
        for i, prompt in enumerate(prompts):
            start = i * seg
            if start >= total:
                break
            w.setpos(start)
            frames = w.readframes(min(seg, total - start))
            kind = "train" if any(t.lower() in prompt.lower() for t in train_titles) else "novel"
            name = f"{i:02d}_{kind}_{slug(prompt)}__step{step:08d}.wav"
            op = out / name
            if not op.exists():
                o = wave.open(str(op), "wb")
                o.setnchannels(ch); o.setsampwidth(sw); o.setframerate(sr)
                o.writeframes(frames); o.close()
                n += 1
            key = re.search(r"\b([A-G]#?min|[A-G]#?maj)\b", prompt)
            bpm = re.search(r"(\d{2,3})\s*bpm", prompt, re.I)
            rows.append({"file": name, "key": key.group(1) if key else "", "bpm": bpm.group(1) if bpm else "",
                         "mood": kind, "title": slug(prompt).split("_")[0], "seed": f"step{step}",
                         "prompt": prompt, "init_loop": "", "init_path": ""})
        w.close()

    with open(out / "library_index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["file","key","bpm","mood","title","seed","prompt","init_loop","init_path"])
        wr.writeheader(); wr.writerows(rows)

    print(f"split {n} new clips -> {out}  ({len(rows)} total, prompt shown via library_index.csv)")
    print(f"\ncd ~/stable-audio-3 && uv run python sweep_rater/rate_server.py --folder {out} --port 8799")


if __name__ == "__main__":
    main()
