"""cKz Loop Generator — backend service.

Loads aug-8000 ONCE (resident) and exposes the two locked workflows:
  • /api/discover  — foreign minor+dense loop → cKz-repainted NEW melodies
  • /api/refine    — hold a picked melody, move the timbre (subtle/bold σ)
plus keep/library and audio serving, and serves the static browse UI at /.

Run on the Mac:
    uv run python -m app.server                 # real model (loads aug-8000)
    uv run python -m app.server --mock           # placeholder audio (no model)
    uv run python -m app.server --port 8800
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import config as C
from . import library as lib
from .engine import make_engine
from .jobs import Clip, JobManager, _short
from .loops import LoopLibrary

STATIC = C.APP_DIR / "static"

app = FastAPI(title="cKz Loop Generator")
STATE: dict = {}  # engine, jobs, loops, mock — set in build_app()


# ── request models ──────────────────────────────────────────────────────────
class DiscoverReq(BaseModel):
    count: int = 4
    key_mode: str = "both"        # "both" | "match" | "tritone"
    seed: Optional[int] = None    # base seed for reproducibility; else random


class RefineReq(BaseModel):
    source: str                   # "clip:<id>" or "kept:<id>"
    mode: str = "subtle"          # "subtle" (σ0.45) | "bold" (σ0.75)
    count: int = 4
    seed: Optional[int] = None


class KeepReq(BaseModel):
    clip_id: str


# ── helpers ─────────────────────────────────────────────────────────────────
def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed) if seed is not None else random.Random()


def _plan_discover(req: DiscoverReq) -> list[Clip]:
    loops: LoopLibrary = STATE["loops"]
    if len(loops) == 0:
        raise HTTPException(503, "loop library empty — run `uv run python -m app.index_loops` "
                                 "on the Mac to build app/loops_manifest.json")
    rng = _rng(req.seed)
    count = max(1, min(req.count, 24))
    mode = req.key_mode if req.key_mode in ("both", "match", "tritone") else "both"
    # "both" emits match+tritone per loop (free 2x melodies), so sample fewer loops
    per_loop = 2 if mode == "both" else 1
    n_loops = (count + per_loop - 1) // per_loop
    sampled = loops.sample(n_loops, rng)
    clips: list[Clip] = []
    for lp in sampled:
        variants = []
        if mode in ("both", "match"):
            variants.append(("match", lp["key"]))
        if mode in ("both", "tritone"):
            variants.append(("tritone", C.tritone(lp["key"])))
        for kmode, key in variants:
            if len(clips) >= count:
                break
            title = rng.choice(C.TITLES)
            seed = rng.randint(0, 2**31 - 1)
            prompt = f"cKz! {title} {key} {lp['bpm']} bpm @crushed_keyz"
            clips.append(Clip(
                id=_short(), kind="discover", prompt=prompt, seed=seed,
                sigma=C.SIGMA_DISCOVER, init_path=lp["path"], title=title, key=key,
                bpm=lp["bpm"], key_mode=kmode, init_loop=lp.get("name", Path(lp["path"]).name),
            ))
    return clips[:count]


def _resolve_source(source: str) -> tuple[str, str, dict]:
    """→ (init_path, base_prompt, meta). meta carries title/key/bpm for display."""
    jm: JobManager = STATE["jobs"]
    if source.startswith("clip:"):
        clip = jm.get_clip(source[5:])
        if not clip or clip.status != "done" or not clip.out_rel:
            raise HTTPException(404, "source clip not found or not finished")
        init_path = str((C.APP_DIR / clip.out_rel).resolve())
        return init_path, clip.prompt, {"title": clip.title, "key": clip.key, "bpm": clip.bpm,
                                        "init_loop": clip.init_loop}
    if source.startswith("kept:"):
        k = lib.get_kept(source[5:])
        if not k:
            raise HTTPException(404, "kept clip not found")
        init_path = str((C.APP_DIR / k["wav"]).resolve())
        return init_path, k["prompt"], {"title": k["title"], "key": k["key"], "bpm": k["bpm"],
                                        "init_loop": k.get("init_loop", "")}
    raise HTTPException(400, "source must be 'clip:<id>' or 'kept:<id>'")


def _plan_refine(req: RefineReq) -> list[Clip]:
    init_path, base_prompt, meta = _resolve_source(req.source)
    sigma = C.SIGMA_REFINE_BOLD if req.mode == "bold" else C.SIGMA_REFINE_SUBTLE
    rng = _rng(req.seed)
    count = max(1, min(req.count, 24))
    clips = []
    for _ in range(count):
        seed = rng.randint(0, 2**31 - 1)
        clips.append(Clip(
            id=_short(), kind="refine", prompt=base_prompt, seed=seed, sigma=sigma,
            init_path=init_path, title=meta.get("title", ""), key=meta.get("key", ""),
            bpm=meta.get("bpm", 0) or 0, refine_mode=req.mode, source_id=req.source,
            init_loop=f"REFERENCE: {meta.get('title','clip')}",
        ))
    return clips


# ── routes ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    f = STATIC / "index.html"
    if not f.is_file():
        return HTMLResponse("<h1>index.html missing</h1>", status_code=404)
    return HTMLResponse(f.read_text())


@app.get("/api/status")
def status():
    loops: LoopLibrary = STATE["loops"]
    return {
        "engine": STATE["engine"].name,
        "mock": STATE["engine"].name == "mock",
        "loops": len(loops),
        "loop_keys": loops.keys,
        "kept": len(lib.list_kept()),
        "settings": {
            "model": C.LORA_CKPT, "cfg": C.CFG_SCALE, "steps": C.STEPS,
            "sampler": C.SAMPLER, "strength": C.LORA_STRENGTH,
            "sigma_discover": C.SIGMA_DISCOVER,
            "sigma_subtle": C.SIGMA_REFINE_SUBTLE, "sigma_bold": C.SIGMA_REFINE_BOLD,
            "max_duration": C.MAX_DURATION,
        },
    }


@app.post("/api/discover")
def discover(req: DiscoverReq):
    clips = _plan_discover(req)
    job = STATE["jobs"].add_job("discover", clips)
    return STATE["jobs"].job_view(job.id)


@app.post("/api/refine")
def refine(req: RefineReq):
    clips = _plan_refine(req)
    job = STATE["jobs"].add_job("refine", clips)
    return STATE["jobs"].job_view(job.id)


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    v = STATE["jobs"].job_view(job_id)
    if not v:
        raise HTTPException(404, "job not found")
    return v


@app.post("/api/keep")
def keep(req: KeepReq):
    clip = STATE["jobs"].get_clip(req.clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    try:
        entry = lib.keep(clip)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    clip.kept = True
    return entry


@app.get("/api/library")
def library():
    return {"items": lib.list_kept()}


@app.delete("/api/library/{kid}")
def del_kept(kid: str):
    return {"removed": lib.unkeep(kid)}


# ── audio serving ───────────────────────────────────────────────────────────
def _safe_under(base: Path, rel: str) -> Path:
    p = (base / rel).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise HTTPException(404, "invalid path")
    if not p.is_file() or p.suffix.lower() != ".wav":
        raise HTTPException(404, "no such audio")
    return p


@app.get("/audio/gen/{filename}")
def audio_gen(filename: str):
    return FileResponse(_safe_under(C.GEN_DIR, filename), media_type="audio/wav")


@app.get("/audio/kept/{filename}")
def audio_kept(filename: str):
    return FileResponse(_safe_under(C.KEPT_DIR, filename), media_type="audio/wav")


@app.get("/audio/init/{clip_id}")
def audio_init(clip_id: str):
    """Serve a clip's init audio (foreign loop / source clip) as the reference
    player. Allowlisted: only paths attached to a known clip are reachable."""
    clip = STATE["jobs"].get_clip(clip_id)
    if not clip or not clip.init_path:
        raise HTTPException(404, "no init for clip")
    p = Path(clip.init_path)
    if not p.is_file():
        raise HTTPException(404, "init file missing")
    media = "audio/wav" if p.suffix.lower() == ".wav" else "audio/mpeg"
    return FileResponse(str(p), media_type=media)


def build_app(mock: bool = False) -> FastAPI:
    C.GEN_DIR.mkdir(parents=True, exist_ok=True)
    C.KEPT_DIR.mkdir(parents=True, exist_ok=True)
    engine = make_engine(mock=mock)
    STATE["engine"] = engine
    STATE["jobs"] = JobManager(engine)
    STATE["loops"] = LoopLibrary.load()
    print(f"[server] engine={engine.name}  loops={len(STATE['loops'])}  kept={len(lib.list_kept())}")
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="placeholder audio, no model (dev)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    args = ap.parse_args()
    import uvicorn
    build_app(mock=args.mock)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
