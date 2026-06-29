#!/usr/bin/env python3
"""PDMX -> cKz init pipeline.

Turn public-domain SYMBOLIC scores (PDMX MusicXML, or any .mxl/.musicxml/.mid/.krn/.abc)
into out-of-distribution INIT loops for scripts/init_sweep.py — the legal way to get the
foreign-init novelty (the "discover" engine was Cymatics-init; this swaps in public domain).

Per score:
  1. filter   -> MINOR-key; --min-notes density floor
  2. trim      -> slice the MELODY-ACTIVE window (densest onsets) so it starts at bar 1
  3. render    -> harmonic-rich synth (or fluidsynth + --soundfont for full instruments)
  4. name      -> "PDMX NN - <bpm> BPM <Key> Min.wav"  (init_sweep's foreign parser reads this)

Diversity levers:
  --texture-mix  spread the selection across voice-counts (single-line .. complex polyphony)
  --key-spread   transpose each loop to cover all 12 minor keys (guaranteed key variety)

Usage:
  python3 pipeline/pdmx_init.py --corpus --out data/pdmx_init --n 18 --texture-mix --key-spread --min-notes 40
  python3 pipeline/pdmx_init.py --src /path/to/PDMX --out data/pdmx_init --n 64 --texture-mix --key-spread
"""
import argparse, glob, os, random, subprocess, tempfile
from collections import defaultdict
import numpy as np

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PC = {k: i for i, k in enumerate(KEYS)}
FLAT2SHARP = {"D-": "C#", "E-": "D#", "G-": "F#", "A-": "G#", "B-": "A#",
              "C-": "B", "F-": "E"}


def harmonic_wave(x):
    """Fundamental + a couple of softly-decaying harmonics. Kept lean (3 partials) so the
    no-soundfont fallback stays clear instead of buzzy/muffled. For best results pass
    --soundfont <GM.sf2> to render real instruments."""
    return np.sin(x) + 0.35 * np.sin(2 * x) + 0.18 * np.sin(3 * x)


def to_melody_line(s):
    """Reduce a score to its melodic SKYLINE (top note at each moment) — a clean single
    hook, which the ratings showed cKz repaints best. Used when --melody-line is set."""
    from music21 import stream, note as m21note
    ch = s.chordify()
    mel = stream.Part()
    for el in ch.recurse().notes:
        try:
            p = max(el.pitches, key=lambda x: x.ps) if el.isChord else el.pitch
            n = m21note.Note(p); n.quarterLength = el.quarterLength
            mel.append(n)
        except Exception:
            continue
    return mel if len(mel.recurse().notes) else s


def list_scores(args):
    if args.corpus:
        from music21 import corpus
        ps = [str(p) for p in corpus.getPaths()]
    else:
        exts = ("*.mxl", "*.musicxml", "*.xml", "*.mid", "*.midi", "*.krn", "*.abc")
        ps = []
        for e in exts:
            ps += glob.glob(os.path.join(args.src, "**", e), recursive=True)
    random.Random(args.seed).shuffle(ps)
    return ps


def parse_score(path, args):
    from music21 import corpus, converter
    return corpus.parse(path) if args.corpus else converter.parse(path)


def densest_window(pm, win):
    onsets = sorted(n.start for inst in pm.instruments for n in inst.notes)
    if not onsets:
        return 0.0
    end = max(onsets)
    best_t, best_c, t = 0.0, -1, 0.0
    while t <= max(0.0, end - win) + 0.01:
        c = sum(1 for o in onsets if t <= o < t + win)
        if c > best_c:
            best_c, best_t = c, t
        t += 1.0
    return best_t


def sharpname(tonic):
    return FLAT2SHARP.get(tonic, tonic)


def render(stream, keyl, qual, bpm, idx, args):
    """music21 stream -> MIDI -> pretty_midi -> render_pm."""
    import pretty_midi
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
        mid = tf.name
    stream.write('midi', mid)
    pm = pretty_midi.PrettyMIDI(mid)
    os.remove(mid)
    return render_pm(pm, keyl, qual, bpm, idx, args)


# ── PDMX (MusPy-JSON) support ────────────────────────────────────────────────
def pdmx_paths(args):
    """Read PDMX.csv; keep target-genre, light-track scores; return absolute json paths."""
    import csv
    csv.field_size_limit(10 ** 9)
    want = set(g.strip().lower() for g in args.genres.split(",") if g.strip())
    root = args.pdmx
    out = []
    with open(os.path.join(root, "PDMX.csv")) as f:
        for row in csv.DictReader(f):
            nt = row.get("n_tracks", "")
            if not (nt.isdigit() and args.min_voices <= int(nt) <= args.max_voices):
                continue
            g = (row.get("genres") or "").lower()
            if want and not any(d in g for d in want):
                continue
            if args.best_only and str(row.get("is_best_arrangement", "")).lower() not in ("true", "1"):
                continue
            rel = (row.get("path") or "").lstrip("./")
            if rel:
                out.append(os.path.join(root, rel))
                if len(out) >= args.scan_limit:
                    break
    random.Random(args.seed).shuffle(out)
    return out


# Krumhansl-Schmuckler key profiles — fast key estimate straight from the notes
# (PDMX key signatures are frequently absent; the music21 round-trip is too slow at scale).
_KMAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KMIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def estimate_key(pm):
    import numpy as np
    h = np.zeros(12)
    for inst in pm.instruments:
        for n in inst.notes:
            h[n.pitch % 12] += (n.end - n.start)
    if h.sum() == 0:
        return None, None
    best = (-2.0, 0, 'minor')
    for mode, prof in (('major', _KMAJ), ('minor', _KMIN)):
        p = np.array(prof)
        for t in range(12):
            c = np.corrcoef(h, np.roll(p, t))[0, 1]
            if c > best[0]:
                best = (c, t, mode)
    return KEYS[best[1]], best[2]


def pdmx_read(path):
    """PDMX MusPy-JSON -> pretty_midi (or None if too few notes)."""
    import muspy
    pm = muspy.to_pretty_midi(muspy.load_json(path))
    if sum(len(i.notes) for i in pm.instruments) < 12:
        return None
    return pm


def transpose_pm(pm, semis):
    if semis:
        for inst in pm.instruments:
            for n in inst.notes:
                n.pitch = max(0, min(127, n.pitch + semis))
    return pm


def render_pm(pm, keyl, qual, bpm, idx, args):
    import soundfile as sf
    start = densest_window(pm, args.dur)
    if args.soundfont and os.path.isfile(args.soundfont):
        audio = pm.fluidsynth(fs=args.sr, sf2_path=args.soundfont)
    else:
        audio = pm.synthesize(fs=args.sr, wave=harmonic_wave)
    s0 = int(start * args.sr)
    seg = audio[s0:s0 + int(args.dur * args.sr)]
    if seg.size < args.sr:
        seg = audio[:int(args.dur * args.sr)]
    if seg.size == 0:
        os.remove(mid); return None
    seg = seg / (np.max(np.abs(seg)) + 1e-9) * 0.9
    raw = mid + ".wav"
    sf.write(raw, seg, args.sr)
    fn = os.path.join(args.out, f"PDMX {idx:02d} - {bpm} BPM {keyl} {qual}.wav")
    af = "loudnorm" + (f",atempo={args.tempo}" if abs(args.tempo - 1.0) > 1e-3 else "")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-af", af, fn], check=True)
    os.remove(mid); os.remove(raw)
    return fn


def select(cands, args):
    """cands: list of (stream, key, notes, nparts, name). Return up to n, spread across
    voice-count buckets when --texture-mix so the batch has single-line AND complex pieces."""
    if not args.texture_mix:
        return cands[:args.n]
    buckets = defaultdict(list)
    for c in cands:
        buckets[min(c[3], 5)].append(c)        # 1,2,3,4,5+ voices
    bk = sorted(buckets)
    out, i = [], 0
    while len(out) < args.n and any(buckets[b] for b in bk):
        b = bk[i % len(bk)]; i += 1
        if buckets[b]:
            out.append(buckets[b].pop(0))
    return out


def run_pdmx(args, bpms):
    print(f"PDMX: scanning PDMX.csv (genres={args.genres or 'all'}, {args.min_voices}-{args.max_voices} tracks"
          f"{', best-arrangement only' if args.best_only else ''}) ...")
    paths = pdmx_paths(args)
    print(f"  {len(paths)} candidate scores; reading + key-analyzing until {args.n} minor loops ...")
    made = []
    for p in paths:
        if len(made) >= args.n:
            break
        try:
            pm, k = pdmx_read(p)
            if pm is None or k is None or (args.minor_only and k.mode != 'minor'):
                continue
            nn = sum(len(i.notes) for i in pm.instruments)
            if not (args.min_notes <= nn <= args.max_notes):
                continue
            idx = len(made) + 1
            srcname = sharpname(k.tonic.name)
            if args.key_spread and srcname in PC:
                tgt = KEYS[(idx - 1) % 12]
                d = (PC[tgt] - PC[srcname]) % 12
                if d > 6:
                    d -= 12
                pm = transpose_pm(pm, d)
                keyl = tgt
            else:
                keyl = srcname
            bpm = bpms[(idx - 1) % len(bpms)]
            fn = render_pm(pm, keyl, "Min", bpm, idx, args)
            if fn:
                made.append(fn)
                print(f"  [{len(made):2}/{args.n}] {keyl:3} min  {len(pm.instruments)}-trk  {os.path.basename(p)[:22]}")
        except Exception:
            continue
    print(f"\nDone: {len(made)} PDMX init loops -> {args.out}")
    print("Feed init_sweep:\n  uv run python scripts/init_sweep.py --no-baseline "
          f"--loops-root ~/stable-audio-3 --init-dirs {args.out} --sigmas 0.75 --dense-init "
          "--rand-titles --loop-seed 1 --seeds 7,123 --out sweep_rater/campaigns/pdmx_real")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--src", help="directory of MusicXML/.mid/.krn/.abc score files")
    src.add_argument("--corpus", action="store_true", help="use music21's bundled public-domain corpus")
    src.add_argument("--pdmx", help="path to extracted PDMX dir (reads PDMX.csv + MusPy-JSON scores)")
    ap.add_argument("--genres", default="soundtrack,electronic,hiphop,pop,classical-soundtrack,pop-electronic",
                    help="PDMX genre substrings to keep (dark/cinematic/modern); empty = all")
    ap.add_argument("--best-only", action="store_true", help="PDMX: keep only is_best_arrangement scores")
    ap.add_argument("--out", default="data/pdmx_init")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--min-notes", type=int, default=60, help="density floor (reject too-sparse / ambiguous)")
    ap.add_argument("--max-notes", type=int, default=900, help="density ceiling (reject cluttered chord-stacks)")
    ap.add_argument("--min-voices", type=int, default=1, help="reject scores with fewer parts")
    ap.add_argument("--max-voices", type=int, default=3, help="LIGHT TEXTURE: reject dense polyphony (ratings: 5-voice worst)")
    ap.add_argument("--melody-line", action="store_true", help="reduce each score to its top-line melody (clearest hook)")
    ap.add_argument("--minor-only", dest="minor_only", action="store_true", default=True)
    ap.add_argument("--allow-major", dest="minor_only", action="store_false")
    ap.add_argument("--texture-mix", action="store_true", help="spread selection across voice-counts (single..complex)")
    ap.add_argument("--key-spread", action="store_true", help="transpose loops to cover all 12 minor keys")
    ap.add_argument("--scan-limit", type=int, default=400, help="max scores to parse while gathering candidates")
    ap.add_argument("--dur", type=float, default=16.0)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--tempo", type=float, default=1.0, help="atempo speed-up (e.g. 1.4 for more trap energy)")
    ap.add_argument("--soundfont", default="", help="optional .sf2 for full-instrument fluidsynth render")
    ap.add_argument("--bpms", default="130,138,140,145,150")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import warnings; warnings.filterwarnings("ignore")
    os.makedirs(args.out, exist_ok=True)
    bpms = [int(b) for b in args.bpms.split(",") if b.strip()]
    if args.pdmx:
        run_pdmx(args, bpms); return
    paths = list_scores(args)
    print(f"scanning up to {args.scan_limit} scores (minor, >={args.min_notes} notes) ...")

    cands = []
    for p in paths:
        if len(cands) >= args.scan_limit:
            break
        try:
            s = parse_score(p, args)
            k = s.analyze('key')
            if args.minor_only and k.mode != 'minor':
                continue
            nnotes = len(list(s.recurse().notes))
            if not (args.min_notes <= nnotes <= args.max_notes):
                continue                                  # density sweet-spot (clarity, not clutter)
            nparts = len(s.parts) if (hasattr(s, 'parts') and len(s.parts)) else 1
            if not (args.min_voices <= nparts <= args.max_voices):
                continue                                  # light texture: drop dense polyphony
            cands.append((s, k, nnotes, nparts, os.path.basename(p)))
        except Exception:
            continue

    sel = select(cands, args)
    made = []
    for idx, (s, k, n, nparts, name) in enumerate(sel, 1):
        srcname = sharpname(k.tonic.name)
        if args.key_spread:
            target = KEYS[(idx - 1) % 12]
            if srcname in PC:
                d = (PC[target] - PC[srcname]) % 12
                if d > 6:
                    d -= 12
                if d != 0:
                    try:
                        s = s.transpose(d)
                    except Exception:
                        pass
            keyl = target
        else:
            keyl = srcname
        if args.melody_line:
            try:
                s = to_melody_line(s)
            except Exception:
                pass
        bpm = bpms[(idx - 1) % len(bpms)]
        try:
            fn = render(s, keyl, "Min", bpm, idx, args)
            if fn:
                made.append(fn)
                print(f"  [{len(made):2}/{args.n}] {keyl:3} min  {nparts}-voice  {name[:42]}")
        except Exception:
            continue

    print(f"\nDone: {len(made)} init loops -> {args.out}")
    print("Feed init_sweep:\n  uv run python scripts/init_sweep.py --no-baseline "
          f"--loops-root ~/stable-audio-3 --init-dirs {args.out} --sigmas 0.75 --dense-init "
          "--rand-titles --seeds 7,123 --out sweep_rater/campaigns/pdmx_real")


if __name__ == "__main__":
    main()
