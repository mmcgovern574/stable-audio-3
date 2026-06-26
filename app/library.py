"""Kept-clip store: copy a generated wav into app/kept/ and record it in
library.json. Self-contained so /refine can re-init straight from a kept wav.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from . import config as C
from .jobs import Clip

_lock = threading.Lock()


def _load() -> list[dict]:
    if not C.LIBRARY_JSON.exists():
        return []
    try:
        return json.loads(C.LIBRARY_JSON.read_text())
    except json.JSONDecodeError:
        return []


def _save(items: list[dict]) -> None:
    C.KEPT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = C.LIBRARY_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=2))
    tmp.replace(C.LIBRARY_JSON)


def list_kept() -> list[dict]:
    return _load()


def keep(clip: Clip) -> dict:
    """Copy the clip's generated wav into kept/ and append a manifest entry."""
    if not clip.out_rel:
        raise ValueError("clip has no generated audio yet")
    src = C.APP_DIR / clip.out_rel
    if not src.is_file():
        raise FileNotFoundError(f"generated wav missing: {src}")
    with _lock:
        items = _load()
        # de-dupe: same source clip already kept → return existing
        for it in items:
            if it.get("clip_id") == clip.id:
                return it
        kid = uuid.uuid4().hex[:12]
        C.KEPT_DIR.mkdir(parents=True, exist_ok=True)
        fn = f"{kid}.wav"
        shutil.copy2(src, C.KEPT_DIR / fn)
        entry = {
            "id": kid,
            "clip_id": clip.id,
            "kind": clip.kind,
            "prompt": clip.prompt,
            "title": clip.title,
            "key": clip.key,
            "bpm": clip.bpm,
            "seed": clip.seed,
            "sigma": clip.sigma,
            "init_loop": clip.init_loop,
            "init_path": clip.init_path,
            "wav": f"kept/{fn}",
            "duration": clip.duration,
            "silence": clip.silence,
            "dyn": clip.dyn,
            "sparse": clip.sparse,
            "kept_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        items.insert(0, entry)
        _save(items)
        return entry


def get_kept(kid: str) -> Optional[dict]:
    for it in _load():
        if it["id"] == kid:
            return it
    return None


def unkeep(kid: str) -> bool:
    with _lock:
        items = _load()
        keep_items = [it for it in items if it["id"] != kid]
        if len(keep_items) == len(items):
            return False
        for it in items:
            if it["id"] == kid:
                wav = C.APP_DIR / it["wav"]
                if wav.is_file():
                    wav.unlink()
        _save(keep_items)
        return True
