"""Clip registry + single-worker generation queue.

The model can only do one generate at a time, so every request's clips are
serialized through ONE background worker. Each request (discover/refine) creates
a Job that owns several Clips; the frontend polls the job and renders each clip
as it flips to `done`. Output paths are a hash of (prompt, seed, σ, init) so a
repeat request reuses the cached wav instead of regenerating (generation is
deterministic — see PRODUCT_HANDOFF.md).
"""
from __future__ import annotations

import hashlib
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from . import config as C
from . import density
from .engine import Engine


def _short() -> str:
    return uuid.uuid4().hex[:12]


def _out_rel(prompt: str, seed: int, sigma: Optional[float], init_path: Optional[str]) -> str:
    h = hashlib.sha1(f"{prompt}|{seed}|{sigma}|{init_path}".encode()).hexdigest()[:16]
    return f"gen/{h}.wav"


@dataclass
class Clip:
    id: str
    kind: str                       # "discover" | "refine"
    prompt: str
    seed: int
    sigma: Optional[float]
    init_path: Optional[str]        # foreign loop (discover) or source wav (refine)
    title: str = ""
    key: str = ""
    bpm: int = 0
    key_mode: str = ""              # "match" | "tritone"
    init_loop: str = ""             # display name of init
    refine_mode: str = ""           # "subtle" | "bold"
    source_id: str = ""             # source clip/kept id for refine
    status: str = "pending"         # pending | running | done | error
    error: str = ""
    out_rel: Optional[str] = None
    duration: Optional[float] = None
    sparse: Optional[bool] = None
    silence: Optional[float] = None
    dyn: Optional[float] = None
    kept: bool = False
    created: float = field(default_factory=time.time)
    finished: Optional[float] = None

    def public(self) -> dict:
        d = asdict(self)
        d["audio_url"] = f"/audio/{self.out_rel}" if (self.status == "done" and self.out_rel) else None
        d["init_url"] = f"/audio/init/{self.id}" if self.init_path else None
        return d


@dataclass
class Job:
    id: str
    kind: str
    clip_ids: list[str]
    created: float = field(default_factory=time.time)


class JobManager:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.clips: dict[str, Clip] = {}
        self.jobs: dict[str, Job] = {}
        self._q: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ── public API ──────────────────────────────────────────────────────
    def add_job(self, kind: str, clips: list[Clip]) -> Job:
        job = Job(id=_short(), kind=kind, clip_ids=[c.id for c in clips])
        with self._lock:
            self.jobs[job.id] = job
            for c in clips:
                self.clips[c.id] = c
                self._q.put(c.id)
        return job

    def job_view(self, job_id: str) -> Optional[dict]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        clips = [self.clips[cid].public() for cid in job.clip_ids if cid in self.clips]
        done = sum(1 for c in clips if c["status"] in ("done", "error"))
        return {"job_id": job.id, "kind": job.kind, "done": done,
                "total": len(clips), "clips": clips}

    def get_clip(self, clip_id: str) -> Optional[Clip]:
        return self.clips.get(clip_id)

    # ── worker ──────────────────────────────────────────────────────────
    def _run(self):
        while True:
            cid = self._q.get()
            clip = self.clips.get(cid)
            if clip is None:
                continue
            clip.status = "running"
            try:
                out_rel = _out_rel(clip.prompt, clip.seed, clip.sigma, clip.init_path)
                out_path = C.APP_DIR / out_rel
                if not out_path.exists():
                    res = self.engine.generate(
                        prompt=clip.prompt, seed=clip.seed, sigma=clip.sigma,
                        init_path=clip.init_path, out_path=out_path,
                    )
                    clip.duration = res.duration
                else:
                    clip.duration = self.engine.probe_duration(str(out_path))
                g = density.gate(str(out_path))
                clip.sparse = g["sparse"]
                clip.silence = g["silence"]
                clip.dyn = g["dyn"]
                clip.out_rel = out_rel
                clip.status = "done"
            except Exception as e:  # keep the worker alive
                clip.status = "error"
                clip.error = f"{type(e).__name__}: {e}"
            finally:
                clip.finished = time.time()
                self._q.task_done()
