#!/usr/bin/env python3
"""Import keeper loops into the SoundSauce catalog with UUID storage keys + full provenance.

Replaces catalog_build/upload_catalog.py. The storage key is a UUID (<uuid>.mp3), so a '#'
or any other character in a title can never corrupt storage again. Idempotent: re-runs upsert
by stable id (no duplicates).

Backfill the existing catalog (fixes the sharp-key corruption + seeds the new schema):

  cd ~/stable-audio-3 && \
  SUPABASE_URL=https://ycpyncjkwgtsxvozqnzz.supabase.co \
  SUPABASE_SERVICE_KEY=<service_role key> \
  python3 pipeline/import_keepers.py --catalog catalog_build/catalog.json --audio-dir catalog_build --replace

Preview without touching the network (computes ids + audio stats, prints the rows):

  python3 pipeline/import_keepers.py --catalog catalog_build/catalog.json --audio-dir catalog_build --dry-run

Get the service_role key from Supabase -> Project Settings -> API. Requires migration 0005.
"""
import os, sys, json, re, argparse, datetime, urllib.request, urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as prov

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


def _req(method, path, data=None, headers=None, raw=False):
    h = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
    if headers:
        h.update(headers)
    body = data if raw else (json.dumps(data).encode() if data is not None else None)
    r = urllib.request.Request(URL + path, data=body, method=method, headers=h)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def upsert(table, row, on_conflict="id"):
    st, b = _req("POST", f"/rest/v1/{table}?on_conflict={on_conflict}", data=row,
                 headers={"Content-Type": "application/json",
                          "Prefer": "resolution=merge-duplicates,return=minimal"})
    if st not in (200, 201, 204):
        print(f"    ! {table} upsert -> {st}: {b[:200]}")
    return st in (200, 201, 204)


def upload_object(key, blob):
    st, b = _req("POST", f"/storage/v1/object/catalog/{key}", data=blob, raw=True,
                 headers={"Content-Type": "audio/mpeg", "x-upsert": "true"})
    if st not in (200, 201):
        print(f"    ! upload {key} -> {st}: {b[:200]}")
    return st in (200, 201)


def key_type(filename):
    # init_sweep.py names: ..._{key}{m|t}__s{nn}__seed{n}.wav  (m=match key, t=tritone +6)
    mt = re.search(r"_(m|t)__s\d", filename)
    return {"m": "match", "t": "tritone"}.get(mt.group(1) if mt else "", None)


def build_rows(cat, adir, idmap, base_url):
    rows = []
    for m in cat:
        src = m["mp3"]
        sid = idmap.get(src) or prov.stable_id(src)
        idmap[src] = sid
        path = adir / src
        kt = key_type(src)
        # generated_at: real render time. For this backfill we only have the file mtime
        # (an approximation — the true gen time pre-dates the manifest); future gens log it exactly.
        gen_at = (datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).isoformat()
                  if path.exists() else None)
        stats = {"output_sha256": None, "duration_sec": m.get("duration_sec"),
                 "sample_rate": None, "peak_dbfs": None, "embedding": None}
        if path.exists():
            stats = prov.audio_stats(str(path))
        seed = m.get("seed")
        seed_int = int(seed) if str(seed).lstrip("-").isdigit() else None
        # Per-clip engine + sigma if the catalog row carries them (new harvests are
        # self-describing); else fall back to the fixed recipe stamp (old backfills).
        engine = m.get("engine") or prov.CKZ_AUG_RECIPE["engine"]
        init_noise = m["init_noise"] if "init_noise" in m else prov.CKZ_AUG_RECIPE["init_noise"]
        gen = {
            "id": sid, "model_id": prov.MODEL["id"], "prompt": m.get("prompt"), "seed": seed_int,
            "engine": engine, "init_audio": m.get("init_loop"),
            "init_noise": init_noise, "cfg": prov.CKZ_AUG_RECIPE["cfg"],
            "steps": prov.CKZ_AUG_RECIPE["steps"], "sampler": prov.CKZ_AUG_RECIPE["sampler"],
            "strength": prov.CKZ_AUG_RECIPE["strength"], "duration_sec": stats["duration_sec"],
            "sample_rate": stats["sample_rate"], "provenance_quality": "logged",
            "generated_at": gen_at,
            "params": {"key_type": kt, "mood": m.get("mood"),
                       "base_model": prov.CKZ_AUG_RECIPE["base_model"],
                       "lora_ckpt": prov.CKZ_AUG_RECIPE["lora_ckpt"], "source_file": src,
                       "generated_at_source": ("file_mtime_approx" if gen_at else None)},
        }
        smp = {
            "id": sid, "generation_id": sid, "producer_id": prov.PRODUCER["id"],
            "title": m.get("title"), "key": m.get("key"), "bpm": m.get("bpm"),
            "duration_sec": stats["duration_sec"], "sample_rate": stats["sample_rate"],
            "storage_key": f"{sid}.mp3",
            "audio_path": f"{base_url}/storage/v1/object/public/catalog/{sid}.mp3",
            "output_sha256": stats["output_sha256"], "peak_dbfs": stats["peak_dbfs"],
            "is_exclusive": False, "is_claimed": False, "status": "public",
            "rating": m.get("rating"), "prompt": m.get("prompt"), "seed": str(seed),
            "tags": [m.get("title"), m.get("mood"), m.get("key")],
            "meta": {"melody": m.get("melody"), "novelty": m.get("novelty"),
                     "timbre": m.get("timbre"), "mood": m.get("mood"),
                     "init_loop": m.get("init_loop"), "key_type": kt},
        }
        rows.append({"src": src, "sid": sid, "path": path, "kt": kt, "stats": stats,
                     "gen": gen, "smp": smp, "m": m})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True, help="catalog.json with the keeper metadata")
    ap.add_argument("--audio-dir", required=True, help="folder holding the .mp3 files")
    ap.add_argument("--replace", action="store_true", help="delete existing non-exclusive catalog rows first")
    ap.add_argument("--dry-run", action="store_true", help="compute + print only; no uploads/inserts")
    ap.add_argument("--skip-upload", action="store_true", help="skip storage upload (files already uploaded); DB rows only")
    a = ap.parse_args()
    if not a.dry_run and (not URL or not KEY):
        sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY (or use --dry-run).")

    cat = json.load(open(a.catalog))
    adir = Path(a.audio_dir)
    idmap_path = adir / "imported_ids.json"
    idmap = json.loads(idmap_path.read_text()) if idmap_path.exists() else {}
    base_url = URL or "<SUPABASE_URL>"

    rows = build_rows(cat, adir, idmap, base_url)
    if not a.dry_run:
        idmap_path.write_text(json.dumps(idmap, indent=2))

    miss = [r["src"] for r in rows if not r["path"].exists()]
    emb = sum(1 for r in rows if r["stats"]["embedding"])
    print(f"{'DRY-RUN: ' if a.dry_run else ''}{len(rows)} loops | audio found {len(rows)-len(miss)}/{len(rows)} "
          f"| embeddings {emb}/{len(rows)} | id map -> {idmap_path}")
    if miss:
        print(f"  (missing audio for {len(miss)}: {miss[:3]}{'…' if len(miss) > 3 else ''})")

    if a.dry_run:
        for r in rows[:4]:
            s = r["stats"]
            print(f"  • {r['src']}\n      id={r['sid']}  store={r['sid']}.mp3  "
                  f"{r['m'].get('key')} {r['m'].get('bpm')}bpm seed{r['m'].get('seed')} keytype={r['kt']}  "
                  f"dur={s['duration_sec']} sr={s['sample_rate']} peak={s['peak_dbfs']}dB sha={str(s['output_sha256'])[:10]}")
        from collections import Counter
        engmix = Counter(f"{r['gen']['engine']}"
                         + (f" σ{r['gen']['init_noise']}" if r['gen']['init_noise'] is not None else "")
                         for r in rows)
        print(f"  …({len(rows)} total). Engines (per-clip): "
              + ", ".join(f"{k}×{v}" for k, v in sorted(engmix.items()))
              + f"  | cfg{prov.CKZ_AUG_RECIPE['cfg']} steps{prov.CKZ_AUG_RECIPE['steps']} "
              f"{prov.CKZ_AUG_RECIPE['sampler']} -> model {prov.MODEL['name']}")
        print("\nDry run only — nothing uploaded. Re-run with SUPABASE_URL + SUPABASE_SERVICE_KEY (no --dry-run).")
        return

    # registry + optional clear
    upsert("producers", prov.PRODUCER)
    upsert("models", {**prov.MODEL, "producer_id": prov.PRODUCER["id"]}, on_conflict="name")
    if a.replace:
        st, b = _req("DELETE", "/rest/v1/samples?is_exclusive=eq.false",
                     headers={"Prefer": "return=minimal"})
        print(f"  cleared existing non-exclusive samples -> {st}")

    ok = 0
    for r in rows:
        sid, src = r["sid"], r["src"]
        if not a.skip_upload and r["path"].exists():
            upload_object(f"{sid}.mp3", r["path"].read_bytes())
        upsert("gen_runs", r["gen"])
        good = upsert("samples", r["smp"])
        m = r["m"]
        if any(m.get(k) is not None for k in ("melody", "novelty", "timbre")):
            upsert("ratings", {"id": prov.stable_id("rating:mike:" + src), "sample_id": sid,
                               "rater": "mike", "source": "human", "melody": m.get("melody"),
                               "novelty": m.get("novelty"), "timbre": m.get("timbre")})
        if r["stats"]["embedding"]:
            upsert("embeddings", {"sample_id": sid, "kind": "chroma", "vec": r["stats"]["embedding"]},
                   on_conflict="sample_id,kind")
        ok += 1 if good else 0
        print(f"  ✓ {m.get('title')} {m.get('key')} -> {sid}.mp3")
    print(f"\nDone. {ok}/{len(rows)} samples upserted. The player will now stream <uuid>.mp3 — sharp-key bug gone.")


if __name__ == "__main__":
    main()
