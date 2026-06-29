#!/usr/bin/env python3
"""Stage a producer's melody folder for the onboarding pipeline.

Given a folder of named melody loops (see the naming spec below), this:
  1. validates every audio file's name parses into  STYLE! Title Key bpm @trigger
  2. checks the STYLE prefix and @trigger are CONSISTENT across the whole folder
     (they become the model's style tokens, so they must match everywhere)
  3. copies each loop into a clean staging dir and writes a `.txt` caption sidecar
     (caption = the exact filename stem, which is what train_lora.py + augment_pitch.py expect)
  4. emits PREP_* key=value lines the onboarding driver captures (style, trigger, count, paths)
  5. writes a demo_prompts.txt for the training run's t2m demos

NAMING SPEC (one loop per file):
    [NN.] <STYLE>! <Title> <Key><min|maj> <bpm> bpm @<trigger>.<wav|mp3|flac>
  examples:
    01. cKz! Mirage Dmin 140 bpm @crushed_keyz.wav
    7. cKz! ICE D#min 135 bpm @crushed_keyz.wav
  rules:
    - optional leading "NN." index is ignored
    - <STYLE>! = a single token ending in '!' (the style trigger, e.g. cKz!)
    - <Key> = a letter A-G, optional '#', immediately followed by 'min' or 'maj' (Dmin, A#min)
    - <bpm> = integer 2-3 digits, followed by the literal word 'bpm'
    - @<trigger> = the producer's unique tag, the LAST token
    - everything between <STYLE>! and <Key> is the Title (may be multiple words)

Usage:
    python scripts/prep_captions.py --src "/path/to/ProducerMelodies" --stage data/<slug>_src
"""
from __future__ import annotations
import argparse, re, shutil, sys
from pathlib import Path

AUDIO_EXT = (".wav", ".mp3", ".flac")
KEY_RE   = re.compile(r"\b([A-Ga-g])(#?)\s*(min|maj)\b", re.I)
BPM_RE   = re.compile(r"\b(\d{2,3})\s*bpm\b", re.I)
STYLE_RE = re.compile(r"^\s*(?:\d+\.\s*)?(\S+!)\s+")     # token ending in '!', after optional NN.
TRIG_RE  = re.compile(r"(@\S+)\s*$")                      # last @token


def parse_stem(stem: str):
    """Return dict(style,title,key,bpm,trigger) or None if the name doesn't match the spec."""
    sm = STYLE_RE.search(stem)
    tm = TRIG_RE.search(stem)
    km = KEY_RE.search(stem)
    bm = BPM_RE.search(stem)
    if not (sm and tm and km and bm):
        return None
    style = sm.group(1)
    trigger = tm.group(1)
    key = f"{km.group(1).upper()}{km.group(2)}{km.group(3).lower()}"
    bpm = int(bm.group(1))
    # title = text between the style prefix and the key token
    mid = stem[sm.end():km.start()].strip()
    title = mid if mid else "Untitled"
    return {"style": style, "title": title, "key": key, "bpm": bpm, "trigger": trigger}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder of the producer's named melody loops")
    ap.add_argument("--stage", required=True, help="staging dir to write captioned copies into")
    ap.add_argument("--demo-out", default=None, help="path for the generated demo_prompts.txt")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    if not src.is_dir():
        print(f"PREP_OK=0\nERROR: --src is not a folder: {src}", file=sys.stderr); sys.exit(2)

    audio = sorted(p for p in src.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not audio:
        print(f"PREP_OK=0\nERROR: no audio ({', '.join(AUDIO_EXT)}) in {src}", file=sys.stderr); sys.exit(2)

    parsed, bad = {}, []
    for p in audio:
        info = parse_stem(p.stem)
        (parsed.__setitem__(p, info) if info else bad.append(p.name))

    if bad:
        print(f"PREP_OK=0\nERROR: {len(bad)} file(s) do not match the naming spec:", file=sys.stderr)
        for n in bad[:20]:
            print(f"   ✗ {n}", file=sys.stderr)
        print("\nFix the names to:  [NN.] <STYLE>! <Title> <Key>min <bpm> bpm @<trigger>.wav", file=sys.stderr)
        print("e.g.               01. cKz! Mirage Dmin 140 bpm @crushed_keyz.wav", file=sys.stderr)
        sys.exit(2)

    styles  = {i["style"]   for i in parsed.values()}
    triggers = {i["trigger"] for i in parsed.values()}
    if len(styles) != 1 or len(triggers) != 1:
        print(f"PREP_OK=0\nERROR: STYLE/trigger not consistent across the folder.", file=sys.stderr)
        print(f"   styles found:   {sorted(styles)}", file=sys.stderr)
        print(f"   triggers found: {sorted(triggers)}", file=sys.stderr)
        print("Every loop must use the SAME <STYLE>! and @<trigger> — they are the model's style tokens.", file=sys.stderr)
        sys.exit(2)

    style   = styles.pop()
    trigger = triggers.pop()
    titles  = sorted({i["title"] for i in parsed.values()})

    print(f"# {len(parsed)} loops OK   style={style}   trigger={trigger}   {len(titles)} distinct titles")
    if args.dry_run:
        print("PREP_OK=1  (dry-run, nothing written)")
        print(f"PREP_STYLE_PREFIX={style}")
        print(f"PREP_TRIGGER={trigger}")
        print(f"PREP_N_LOOPS={len(parsed)}")
        return

    stage = Path(args.stage); stage.mkdir(parents=True, exist_ok=True)
    n = 0
    for p, info in parsed.items():
        dst = stage / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        (stage / (p.stem + ".txt")).write_text(p.stem)   # caption = exact stem
        n += 1

    # demo prompts: the producer's own titles across a few keys (matches training distribution)
    demo_out = Path(args.demo_out) if args.demo_out else stage.parent / f"{stage.name}_demo_prompts.txt"
    demo_keys = ["Dmin", "F#min", "Bmin", "Amin", "Cmin"]
    demo_titles = titles[:8] if len(titles) >= 8 else titles
    lines = []
    for i, t in enumerate(demo_titles):
        k = demo_keys[i % len(demo_keys)]
        lines.append(f"{style} {t} {k} 140 bpm {trigger}")
    demo_out.write_text("\n".join(lines) + "\n")

    print(f"# staged {n} loops + captions -> {stage}")
    print(f"# wrote {len(lines)} demo prompts -> {demo_out}")
    print("PREP_OK=1")
    print(f"PREP_STYLE_PREFIX={style}")
    print(f"PREP_TRIGGER={trigger}")
    print(f"PREP_N_LOOPS={len(parsed)}")
    print(f"PREP_STAGE={stage}")
    print(f"PREP_DEMO_PROMPTS={demo_out}")


if __name__ == "__main__":
    main()
