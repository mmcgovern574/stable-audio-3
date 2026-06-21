#!/usr/bin/env python3
"""Controlled A/B: EXACT cKz label prompt × {matched-init, off-key init, no-init}.

Tests the hypothesis: the best melodies come from a REAL cKz label (exact
training title+key+bpm) paired with an init loop whose key+bpm match it.

For N real cKz labels (parsed from data/crushed_keyz_lora captions), generate 3
conditions per label × seeds:
  - matched : init = library loop in the SAME key, closest bpm
  - offkey  : init = library loop a TRITONE away (deliberate key mismatch)
  - noinit  : pure text-to-music (the exact label, no init)
Everything else fixed at the validated recipe. Rate, then compare per-condition.

    python scripts/exact_label_ab.py --dry-run        # verify pairing (no torch)
    uv run python scripts/exact_label_ab.py --n-labels 9 --seeds 7,123
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KEYS = ["Cmin","C#min","Dmin","D#min","Emin","Fmin","F#min","Gmin","G#min","Amin","A#min","Bmin"]

def norm_key(s):
    s = s.lower().replace("sharp", "#")
    m = re.search(r"([a-g])(#?)\s*min", s)
    return (m.group(1).upper()+m.group(2)+"min") if m else None

def parse_label(stem):
    k = norm_key(stem); mb = re.search(r"(\d{2,3})\s*bpm", stem, re.I)
    mt = re.search(r"cKz!\s+(.+?)\s+[A-Ga-g][#]?(?:sharp)?min", stem)
    return {"prompt": stem, "key": k, "bpm": int(mb.group(1)) if mb else None,
            "title": (mt.group(1).strip() if mt else "?")}

def offkey(k):
    return KEYS[(KEYS.index(k) + 6) % 12] if k in KEYS else k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default=str(REPO/"data/crushed_keyz_lora"))
    ap.add_argument("--manifest", default=str(REPO/"sweep_rater/library_loops.json"))
    ap.add_argument("--out", default=str(REPO/"sweep_rater/campaigns/ckz_exactlabel_ab"))
    ap.add_argument("--lora-ckpt", default="./lora_out/ckz_melody_capped/epoch=54-step=2500.ckpt")
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--n-labels", type=int, default=9)
    ap.add_argument("--seeds", default="7,123")
    ap.add_argument("--noise", type=float, default=0.6)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--max-duration", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # exact cKz labels (caption stems), one per key for diversity, up to n
    labels = []
    for f in sorted(os.listdir(args.labels_dir)):
        if not f.endswith(".txt"): continue
        L = parse_label(os.path.splitext(f)[0])
        if L["key"] and L["bpm"]: labels.append(L)
    bykey = {}
    for L in labels: bykey.setdefault(L["key"], []).append(L)
    picked, ki = [], 0
    keyrot = [k for k in KEYS if k in bykey]
    while len(picked) < args.n_labels and keyrot:
        k = keyrot[ki % len(keyrot)]
        if bykey[k]: picked.append(bykey[k].pop(0))
        else: keyrot.remove(k); continue
        ki += 1
    labels = picked

    # manifest loops grouped by key
    loops = json.load(open(args.manifest))
    lbk = {}
    for lp in loops: lbk.setdefault(lp["key"], []).append(lp)
    def closest(key, bpm):
        pool = lbk.get(key, [])
        return min(pool, key=lambda lp: abs(int(lp["bpm"]) - bpm)) if pool else None

    # build jobs
    jobs = []
    for L in labels:
        mloop = closest(L["key"], L["bpm"])
        oloop = closest(offkey(L["key"]), L["bpm"])
        conds = [("matched", mloop), ("offkey", oloop), ("noinit", None)]
        for cond, loop in conds:
            if cond != "noinit" and loop is None: continue
            for seed in seeds:
                jobs.append((L, cond, loop, seed))

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    def fn(i, L, cond, seed):
        ti = re.sub(r"[^A-Za-z0-9]+","_",L["title"])[:14]
        return f"{i:03d}__{cond}__{L['key']}_{L['bpm']}__{ti}__seed{seed}.wav"

    print(f"labels: {len(labels)}  conditions: matched/offkey/noinit  seeds: {seeds}")
    print(f"total clips: {len(jobs)}  out: {out_dir}\n")
    # index for rater + analysis
    with open(out_dir/"library_index.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["file","condition","key","bpm","title","seed","prompt","init_loop","init_path"])
        for i,(L,cond,loop,seed) in enumerate(jobs,1):
            w.writerow([fn(i,L,cond,seed),cond,L["key"],L["bpm"],L["title"],seed,L["prompt"],
                        os.path.basename(loop["path"]) if loop else "", loop["path"] if loop else ""])

    if args.dry_run:
        for i,(L,cond,loop,seed) in enumerate(jobs,1):
            ln = os.path.basename(loop["path"])[:46] if loop else "(none)"
            if seed==seeds[0]:
                print(f"  {cond:7} «{L['prompt'][:42]}»  init[{loop['key'] if loop else '-'}] <= {ln}")
        print(f"\n--dry-run: {len(jobs)} clips. index written.")
        return

    todo=[(i,L,cond,loop,seed,out_dir/fn(i,L,cond,seed)) for i,(L,cond,loop,seed) in enumerate(jobs,1)
          if not (out_dir/fn(i,L,cond,seed)).exists()]
    if not todo: print("all exist."); return
    print(f"Loading {args.model} + LoRA...")
    import torch, torchaudio  # noqa
    from stable_audio_3 import StableAudioModel
    model = StableAudioModel.from_pretrained(args.model)
    model.load_lora([str((REPO/args.lora_ckpt).resolve())])
    sr_out = model.model_config.get("sample_rate") or model.model.sample_rate
    model.set_lora_strength(args.strength)
    cache={}
    def load_init(p):
        if p not in cache: cache[p]=torchaudio.load(p)
        wav,sr=cache[p]; return sr,wav
    t0=time.time()
    for n,(i,L,cond,loop,seed,p) in enumerate(todo,1):
        kw={}
        if loop is not None:
            sr,wav=load_init(loop["path"]); kw["init_audio"]=(int(sr),wav)
            kw["init_noise_level"]=args.noise
            dur=round(min(wav.shape[-1]/int(sr), args.max_duration),1)
        else:
            dur=args.max_duration
        eta=(time.time()-t0)/max(n-1,1)*(len(todo)-n+1) if n>1 else 0
        print(f"[{n:>3}/{len(todo)}] {cond:7} {L['key']}/{L['bpm']} s{seed} dur{dur} ETA{eta/60:4.0f}m")
        try:
            a=model.generate(prompt=L["prompt"],duration=dur,steps=args.steps,cfg_scale=args.cfg,
                             seed=seed,sampler_type="dpmpp",**kw)
            torchaudio.save(str(p),a[0].cpu(),sr_out)
        except Exception as e:
            print(f"      FAILED {type(e).__name__}: {e}")
    print(f"\nDone in {(time.time()-t0)/60:.1f}m → {out_dir}")
    print(f"Rate: python sweep_rater/rate_server.py --folder {out_dir} --port 8799")

if __name__ == "__main__":
    main()
