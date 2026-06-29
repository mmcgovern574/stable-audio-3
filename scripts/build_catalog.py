#!/usr/bin/env python3
"""Turn a RATED generation campaign into a shippable catalog (the missing final stage).

Reads:
  - a campaign dir  (sweep_rater/campaigns/<name>/)  with *.wav + library_index.csv
  - the central rater DB  (sweep_rater/ratings.json)  — 3-axis scores keyed by repo-rel path
Keeps every clip whose ratings clear the bar, converts it to MP3, and writes:
  - <out>/<clip>.mp3                  (one per keeper)
  - <out>/catalog.json               (array; same schema the Supabase uploader already consumes)
  - <out>/catalog_meta.json          (producer + model + recipe provenance for this batch)

This generalizes the ad-hoc step that built the original cKz catalog_build/, so any
producer's rated campaign becomes a catalog with one command.

Usage:
  python scripts/build_catalog.py \
      --campaign sweep_rater/campaigns/<producer>_catalog \
      --out catalog_build/<producer> \
      --min-rating 4 \
      --producer-name "Crushed Keyz" --producer-handle @crushed_keyz \
      --model-name cKz-aug-8000 --lora-ckpt lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt
"""
from __future__ import annotations
import argparse, csv, json, shutil, subprocess, sys
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
AXES = ("timbre", "melody", "novelty")


def load_index(campaign: Path) -> dict:
    idx = {}
    f = campaign / "library_index.csv"
    if f.exists():
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                idx[row["file"]] = row
    return idx


def rating_key_candidates(wav: Path):
    """Keys the rater DB might store this clip under (repo-relative path, then bare name)."""
    try:
        yield str(wav.resolve().relative_to(REPO))
    except ValueError:
        pass
    yield wav.name


def duration_sec(wav: Path):
    try:
        import soundfile as sf
        info = sf.info(str(wav))
        return round(info.frames / info.samplerate, 2)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True, help="campaign dir with *.wav + library_index.csv")
    ap.add_argument("--out", required=True, help="output dir for mp3s + catalog.json")
    ap.add_argument("--ratings", default=str(REPO / "sweep_rater/ratings.json"))
    ap.add_argument("--min-rating", type=float, default=4.0, help="keep if MEAN of the 3 axes >= this")
    ap.add_argument("--min-axis", type=float, default=0.0, help="also require EACH axis >= this (0 = ignore)")
    ap.add_argument("--mp3-quality", default="2", help="ffmpeg libmp3lame -q:a (0=best..9)")
    # provenance (stamped into catalog_meta.json)
    ap.add_argument("--producer-name", default="")
    ap.add_argument("--producer-handle", default="")
    ap.add_argument("--model-name", default="")
    ap.add_argument("--base-model", default="medium-base")
    ap.add_argument("--lora-ckpt", default="")
    ap.add_argument("--init-noise", type=float, default=0.75)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    campaign = Path(args.campaign).resolve()
    if not campaign.is_dir():
        print(f"ERROR: campaign dir not found: {campaign}", file=sys.stderr); sys.exit(2)
    ratings_path = Path(args.ratings)
    if not ratings_path.exists():
        print(f"ERROR: ratings DB not found: {ratings_path}", file=sys.stderr); sys.exit(2)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg and not args.dry_run:
        print("ERROR: ffmpeg not on PATH (needed for wav->mp3). brew install ffmpeg", file=sys.stderr); sys.exit(2)

    ratings = json.loads(ratings_path.read_text())
    index = load_index(campaign)
    wavs = sorted(campaign.glob("*.wav"))
    if not wavs:
        print(f"ERROR: no .wav in {campaign}", file=sys.stderr); sys.exit(2)

    out = Path(args.out);
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    catalog, kept, skipped_unrated, skipped_low = [], 0, 0, 0
    for wav in wavs:
        rec = None
        for k in rating_key_candidates(wav):
            if k in ratings:
                rec = ratings[k]; break
        if not rec or any(rec.get(a) is None for a in AXES):
            skipped_unrated += 1; continue
        scores = [float(rec[a]) for a in AXES]
        avg = mean(scores)
        if avg < args.min_rating or (args.min_axis and min(scores) < args.min_axis):
            skipped_low += 1; continue

        meta = index.get(wav.name, {})
        mp3_name = wav.stem + ".mp3"
        if not args.dry_run:
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(wav),
                            "-codec:a", "libmp3lame", "-q:a", args.mp3_quality,
                            str(out / mp3_name)], check=True)
        catalog.append({
            "title": meta.get("title", ""),
            "key": meta.get("key", ""),
            "bpm": int(meta["bpm"]) if meta.get("bpm", "").isdigit() else meta.get("bpm", ""),
            "prompt": meta.get("prompt", ""),
            "seed": meta.get("seed", ""),
            "mood": meta.get("mood", ""),
            "init_loop": meta.get("init_loop", ""),
            "duration_sec": duration_sec(wav),
            "timbre": rec["timbre"], "melody": rec["melody"], "novelty": rec["novelty"],
            "rating": round(avg, 2),
            "mp3": mp3_name,
        })
        kept += 1

    print(f"kept {kept}  |  skipped {skipped_unrated} unrated, {skipped_low} below bar "
          f"(min-rating {args.min_rating}, min-axis {args.min_axis})")
    if args.dry_run:
        for e in catalog[:8]:
            print(f"   {e['rating']}  {e['title']} {e['key']} {e['bpm']}bpm  ({e['mp3']})")
        print(f"--dry-run: would write {kept} mp3s + catalog.json to {out}")
        return

    catalog.sort(key=lambda e: (-e["rating"], e["title"]))
    (out / "catalog.json").write_text(json.dumps(catalog, indent=2))
    (out / "catalog_meta.json").write_text(json.dumps({
        "producer": {"name": args.producer_name, "handle": args.producer_handle},
        "model": {"name": args.model_name, "base_model": args.base_model, "lora_ckpt": args.lora_ckpt},
        "recipe": {"engine": "foreign-init", "init_noise": args.init_noise, "cfg": args.cfg,
                   "steps": args.steps, "sampler": "dpmpp", "strength": 1.0},
        "campaign": str(campaign.relative_to(REPO)) if campaign.is_relative_to(REPO) else str(campaign),
        "min_rating": args.min_rating, "min_axis": args.min_axis, "n_keepers": kept,
    }, indent=2))
    print(f"wrote {kept} mp3s + catalog.json + catalog_meta.json -> {out}")


if __name__ == "__main__":
    main()
