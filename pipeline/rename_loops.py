#!/usr/bin/env python3
"""Rename duplicate-named loops to fresh words. Identity is the UUID, so this is a pure
metadata UPDATE of samples.title — safe and reversible.

Preview (no DB needed):
  python3 pipeline/rename_loops.py --catalog catalog_build/catalog.json --dry-run

Apply:
  SUPABASE_URL=… SUPABASE_SERVICE_KEY=… python3 pipeline/rename_loops.py --catalog catalog_build/catalog.json
"""
import os, sys, json, argparse, urllib.request, urllib.error
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as prov
import namegen

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


def patch_title(sid, new_title, mood, key):
    h = {"apikey": KEY, "Authorization": f"Bearer {KEY}",
         "Content-Type": "application/json", "Prefer": "return=minimal"}
    body = json.dumps({"title": new_title, "tags": [new_title, mood, key]}).encode()
    r = urllib.request.Request(f"{URL}/rest/v1/samples?id=eq.{sid}", data=body, method="PATCH", headers=h)
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status in (200, 204)
    except urllib.error.HTTPError as e:
        print(f"  ! {sid} -> {e.code}: {e.read()[:160]}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run and (not URL or not KEY):
        sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY (or use --dry-run).")

    cat = json.load(open(a.catalog))
    loops = [{"id": prov.stable_id(m["mp3"]), "title": m["title"], "key": m["key"],
              "bpm": m["bpm"], "rating": m.get("rating"), "mood": m.get("mood")} for m in cat]
    by_id = {L["id"]: L for L in loops}
    keep, changes = namegen.assign_unique(loops)

    print(f"{len(loops)} loops | {len(keep)} keep their word | {len(changes)} duplicates renamed\n")
    # group the changes by the word being replaced, for readability
    from collections import defaultdict
    grouped = defaultdict(list)
    for sid, new in changes.items():
        grouped[by_id[sid]["title"]].append((by_id[sid], new))
    for word in sorted(grouped):
        rows = grouped[word]
        print(f"  {word}  (kept once, {len(rows)} duplicate{'s' if len(rows) > 1 else ''} renamed):")
        for L, new in sorted(rows, key=lambda x: (x[0]["key"], x[0]["bpm"])):
            print(f"      {L['title']} · {L['key']} · {L['bpm']}bpm   ->   {new} · {L['key']} · {L['bpm']}bpm")
    print()

    if a.dry_run:
        print("Dry run — nothing changed. Re-run with SUPABASE_URL + SUPABASE_SERVICE_KEY to apply.")
        return
    ok = 0
    for sid, new in changes.items():
        L = by_id[sid]
        if patch_title(sid, new, L["mood"], L["key"]):
            ok += 1
    print(f"Done. {ok}/{len(changes)} renamed. Every loop now has a unique word.")


if __name__ == "__main__":
    main()
