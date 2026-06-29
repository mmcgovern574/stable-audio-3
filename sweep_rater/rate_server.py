#!/usr/bin/env python3
"""Tiny HTTP server for rating audio files. Stdlib only.

Now uses a single central ratings DB: `sweep_rater/ratings.json` (keyed by
repo-relative path) so ratings survive across folders and the UI can show
everything you've ever rated.

Usage:
    python sweep_rater/rate_server.py --folder ./eval_out
    python sweep_rater/rate_server.py --folder ./favorites
    python sweep_rater/rate_server.py --folder ./sweep_rater/campaigns/cfg_v1

Endpoints:
    GET  /                         — rate.html
    GET  /api/files                — wavs in --folder, with filename + params
    GET  /api/ratings              — full central ratings dict (path → rating info)
    GET  /audio/<filename>         — wav from --folder (current campaign)
    GET  /audio_at?p=<rel_path>    — wav at arbitrary repo-relative path (safe)
    POST /api/rating               — {path, rating, notes} → upsert into ratings.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

# Reversed-init arms fed the model a TIME-REVERSED loop, but the file on disk is forward.
# Reverse on the fly (ffmpeg areverse) so the reference player matches what seeded the clip.
_REV_CACHE: dict = {}
def reversed_wav(path):
    key = str(path)
    if key in _REV_CACHE:
        return _REV_CACHE[key]
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        return None
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-af", "areverse", "-f", "wav", "-"],
                         capture_output=True)
    data = out.stdout if (out.returncode == 0 and out.stdout) else None
    _REV_CACHE[key] = data
    return data

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
HTML_PATH  = SCRIPT_DIR / "rate.html"
RATINGS_DB = SCRIPT_DIR / "ratings.json"   # central, repo-wide

KNOWN_PARAMS = ("cfg", "strength", "steps", "seed", "duration", "rank", "lr")
_PARAM_RE = re.compile(
    r"(?:^|[_\W])(" + "|".join(KNOWN_PARAMS) + r")(-?\d+(?:\.\d+)?)(?=[_\W]|$)",
    flags=re.IGNORECASE,
)


def parse_filename_params(filename: str) -> dict:
    stem = Path(filename).stem
    out = {}
    for m in _PARAM_RE.finditer(stem):
        key = m.group(1).lower()
        val = m.group(2)
        try:
            out[key] = float(val) if "." in val else int(val)
        except ValueError:
            pass
    return out


def list_wavs(folder: Path) -> list[dict]:
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() != ".wav" or p.name.startswith("."):
            continue
        out.append({"filename": p.name, "params": parse_filename_params(p.name)})
    return out


def load_index(folder: Path) -> dict:
    """Load library_index.csv (if present) → {clip_filename: {prompt, init_loop, init_path}}.

    Lets the rater show, for each generated clip, its exact prompt + the init
    loop it was built from (and play that loop as reference).
    """
    idx = folder / "library_index.csv"
    out: dict = {}
    if not idx.is_file():
        return out
    try:
        with open(idx, newline="") as f:
            for row in csv.DictReader(f):
                fn = row.get("file")
                if fn:
                    out[fn] = {
                        "prompt":    row.get("prompt", ""),
                        "init_loop": row.get("init_loop", ""),
                        "init_path": row.get("init_path", ""),
                    }
    except (OSError, csv.Error):
        pass
    return out


def load_ratings() -> dict:
    if not RATINGS_DB.exists():
        return {}
    try:
        return json.loads(RATINGS_DB.read_text())
    except json.JSONDecodeError:
        return {}


def save_ratings(data: dict) -> None:
    tmp = RATINGS_DB.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(RATINGS_DB)


def migrate_legacy_ratings() -> int:
    """Pull in any leftover per-folder ratings.json files into the central DB.

    Walks the repo for any sibling ratings.json files (i.e. NOT the central
    one), merges their entries into the central ratings.json keyed by
    `<rel_folder>/<filename>`. Leaves the old files in place so users can
    confirm migration; just prints a one-time notice.
    """
    central_resolved = RATINGS_DB.resolve()
    merged = load_ratings()
    moved = 0
    for path in REPO_ROOT.rglob("ratings.json"):
        if path.resolve() == central_resolved:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            rel_folder = path.parent.resolve().relative_to(REPO_ROOT.resolve())
        except ValueError:
            continue
        for filename, entry in data.items():
            # Old format: {filename: {rating, notes, ts}}
            if not isinstance(entry, dict) or "rating" not in entry:
                continue
            key = str(rel_folder / filename) if str(rel_folder) != "." else filename
            if key not in merged:
                merged[key] = entry
                moved += 1
    if moved:
        save_ratings(merged)
        print(f"Migrated {moved} legacy per-folder ratings into {RATINGS_DB.name}")
    return moved


class Handler(BaseHTTPRequestHandler):
    folder: Path = Path()  # set by main()
    segments: list = []          # prompt strings, in demo concatenation order
    segment_seconds: float = 0.0 # seconds per segment (demo_duration)
    index: dict = {}             # clip filename → {prompt, init_loop, init_path}

    def log_message(self, fmt, *args):
        pass

    # ── helpers ─────────────────────────────────────────────────────────
    def _send(self, status: int, ctype: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _send_json(self, data, status: int = 200):
        self._send(status, "application/json", json.dumps(data).encode())

    def _send_404(self, msg: str = "not found"):
        self._send(404, "text/plain", msg.encode())

    def _serve_audio(self, abs_path: Path) -> None:
        """Serve a wav file, guarding against path traversal outside REPO_ROOT."""
        try:
            resolved = abs_path.resolve()
            resolved.relative_to(REPO_ROOT.resolve())
        except (ValueError, OSError):
            return self._send_404("invalid path")
        if not resolved.is_file() or resolved.suffix.lower() != ".wav":
            return self._send_404(f"no such audio")
        self._send(200, "audio/wav", resolved.read_bytes())

    @staticmethod
    def _rel(folder: Path) -> str:
        """Return folder path relative to REPO_ROOT (or absolute if outside)."""
        try:
            return str(folder.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            return str(folder)

    # ── GET ─────────────────────────────────────────────────────────────
    def do_GET(self):
        url = urlparse(self.path)
        path = url.path

        if path in ("/", "/index.html"):
            if not HTML_PATH.exists():
                return self._send_404("rate.html missing")
            return self._send(200, "text/html; charset=utf-8", HTML_PATH.read_bytes())

        if path == "/viz.js":
            vp = SCRIPT_DIR / "viz.js"
            if not vp.exists():
                return self._send_404("viz.js missing")
            return self._send(200, "application/javascript; charset=utf-8", vp.read_bytes())

        if path == "/api/files":
            files = list_wavs(self.folder)
            for f in files:
                info = self.index.get(f["filename"])
                if info:
                    f["init"] = {"prompt": info.get("prompt", ""),
                                 "init_loop": info.get("init_loop", "")}
            return self._send_json({
                "folder":     str(self.folder),
                "rel_folder": self._rel(self.folder),
                "files":      files,
            })

        if path == "/api/ratings":
            return self._send_json(load_ratings())

        if path == "/api/segments":
            return self._send_json({"prompts": self.segments, "seconds": self.segment_seconds})

        if path.startswith("/audio/"):
            filename = unquote(path[len("/audio/"):])
            return self._serve_audio(self.folder / filename)

        if path == "/audio_at":
            qs = parse_qs(url.query or "")
            rel = (qs.get("p") or [""])[0]
            if not rel:
                return self._send_404("missing p")
            return self._serve_audio(REPO_ROOT / rel)

        if path == "/init":
            # Serve the INIT loop for a given generated clip. Allowlisted: only
            # init_path values that appear in library_index.csv are reachable,
            # so this is safe even though those files live outside REPO_ROOT.
            qs = parse_qs(url.query or "")
            fn = unquote((qs.get("file") or [""])[0])
            info = self.index.get(fn)
            if not info or not info.get("init_path"):
                return self._send_404("no init for clip")
            ip = Path(info["init_path"])
            if not ip.is_file() or ip.suffix.lower() != ".wav":
                return self._send_404("init file missing")
            # r-arm clips (filename __r<nn>__ or init_loop marked "(reversed)") seeded the
            # model with a reversed loop — serve the reversed audio so the preview matches.
            if re.search(r"__r\d", fn) or "(reversed)" in (info.get("init_loop") or ""):
                rev = reversed_wav(ip)
                if rev is not None:
                    return self._send(200, "audio/wav", rev)
            return self._send(200, "audio/wav", ip.read_bytes())

        return self._send_404()

    # ── POST ────────────────────────────────────────────────────────────
    def do_POST(self):
        if self.path != "/api/rating":
            return self._send_404()
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length).decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._send_json({"error": "invalid json"}, status=400)

        key = payload.get("path")
        if not isinstance(key, str) or not key:
            return self._send_json({"error": "path required"}, status=400)

        ratings = load_ratings()
        prev = ratings.get(key, {})

        def _valid(x):
            return x is None or (isinstance(x, int) and 0 <= x <= 5)

        timbre = payload.get("timbre")
        melody = payload.get("melody")
        novelty = payload.get("novelty", prev.get("novelty"))
        # Back-compat: a bare "rating" still works (applies to both dims).
        if "timbre" not in payload and "melody" not in payload and "rating" in payload:
            timbre = melody = payload.get("rating")

        if not (_valid(timbre) and _valid(melody) and _valid(novelty)):
            return self._send_json(
                {"error": "timbre/melody/novelty must be int 0..5 or null"}, status=400)

        notes_val = payload.get("notes") if payload.get("notes") is not None else prev.get("notes", "")
        if timbre is None and melody is None and novelty is None and not (notes_val or "").strip():
            ratings.pop(key, None)   # truly empty (e.g. clear) -> remove
        else:
            ratings[key] = {
                "timbre": timbre,
                "melody": melody,
                "novelty": novelty,
                "notes":  notes_val,
                "ts":     datetime.now().isoformat(timespec="seconds"),
            }
        save_ratings(ratings)
        return self._send_json({"ok": True})


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", required=True, type=Path,
                   help="Folder of .wav files for the active rating session.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--segments", type=Path, default=None,
                   help="Text file of prompts (one per line) concatenated into each "
                        "demo clip, in order. Shown as a clickable timeline in the UI.")
    p.add_argument("--segment-seconds", type=float, default=12.0,
                   help="Seconds per segment (the demo_duration). Default 12.")
    args = p.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        sys.exit(f"Not a directory: {folder}")

    Handler.folder = folder
    Handler.index = load_index(folder)
    if Handler.index:
        print(f"Index:     {len(Handler.index)} clips mapped to init loops (library_index.csv)")
    if args.segments and args.segments.is_file():
        Handler.segments = [ln.strip() for ln in args.segments.read_text().splitlines() if ln.strip()]
        Handler.segment_seconds = args.segment_seconds
        print(f"Segments:  {len(Handler.segments)} prompts x {args.segment_seconds}s")
    migrate_legacy_ratings()

    # Pick an available port; bump if --port is busy.
    addr = ("127.0.0.1", args.port)
    server = None
    last_err = None
    for port in range(args.port, args.port + 10):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            addr = ("127.0.0.1", port)
            break
        except OSError as e:
            last_err = e
            continue
    if server is None:
        sys.exit(f"Couldn't bind any port in {args.port}..{args.port+9}: {last_err}")

    url = f"http://{addr[0]}:{addr[1]}/"
    n_wavs = len(list_wavs(folder))
    n_rated = len(load_ratings())
    print(f"Folder:    {folder}")
    print(f"Wavs:      {n_wavs}")
    print(f"DB:        {RATINGS_DB}  ({n_rated} clips rated overall)")
    print(f"URL:       {url}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
