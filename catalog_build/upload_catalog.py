#!/usr/bin/env python3
"""Upload the SoundSauce catalog to Supabase.

Reads catalog.json + the .mp3 files in this folder, uploads each mp3 to the public
'catalog' storage bucket, and inserts a row into public.samples. Idempotent: re-runs
overwrite the mp3s and skip rows already inserted (matched by public URL).

Run it (stdlib only — no pip installs needed):

  cd ~/stable-audio-3 && \
  SUPABASE_URL=https://ycpyncjkwgtsxvozqnzz.supabase.co \
  SUPABASE_SERVICE_KEY=<your service_role key> \
  python3 catalog_build/upload_catalog.py

Get the service_role key from Supabase -> Project Settings -> API -> service_role (secret).
Requires the 0003_catalog.sql migration to have been run first.
"""
import os, sys, json, urllib.request, urllib.error

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
HERE = os.path.dirname(os.path.abspath(__file__))

if not URL or not KEY:
    sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables (see header).")

manifest = json.load(open(os.path.join(HERE, "catalog.json")))


def req(method, path, data=None, headers=None):
    h = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
    if headers:
        h.update(headers)
    r = urllib.request.Request(URL + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# Which rows already exist (so re-runs don't duplicate).
status, body = req("GET", "/rest/v1/samples?select=audio_path")
existing = set()
if status == 200:
    try:
        existing = {row["audio_path"] for row in json.loads(body)}
    except Exception:
        pass

uploaded = inserted = skipped = 0
for m in manifest:
    mp3 = m["mp3"]
    public_url = f"{URL}/storage/v1/object/public/catalog/{mp3}"

    # 1) upload the mp3 (upsert so re-runs are safe)
    with open(os.path.join(HERE, mp3), "rb") as fh:
        blob = fh.read()
    status, b = req("POST", f"/storage/v1/object/catalog/{mp3}", data=blob,
                    headers={"Content-Type": "audio/mpeg", "x-upsert": "true"})
    if status not in (200, 201):
        print(f"  upload FAILED {mp3}: {status} {b[:120]}")
        continue
    uploaded += 1

    # 2) insert the sample row (skip if already present)
    if public_url in existing:
        skipped += 1
        continue
    row = {
        "title": m["title"], "key": m["key"], "bpm": m["bpm"], "prompt": m["prompt"],
        "seed": str(m["seed"]), "duration_sec": m["duration_sec"], "audio_path": public_url,
        "rating": m["rating"], "is_exclusive": False,
        "tags": [m["title"], m["mood"], m["key"]],
        "meta": {"melody": m["melody"], "novelty": m["novelty"], "timbre": m["timbre"],
                 "mood": m["mood"], "init_loop": m["init_loop"]},
    }
    status, b = req("POST", "/rest/v1/samples", data=json.dumps(row).encode(),
                    headers={"Content-Type": "application/json", "Prefer": "return=minimal"})
    if status in (200, 201):
        inserted += 1
    else:
        print(f"  insert FAILED {mp3}: {status} {b[:160]}")

print(f"\nDone. uploaded={uploaded}  inserted={inserted}  skipped(existing)={skipped}  total={len(manifest)}")
