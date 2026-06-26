# cKz Loop Generator — UI backend

A persistent service that loads the **aug-8000** model once and exposes the two
locked cKz workflows from `PRODUCT_HANDOFF.md`, plus a browse/keep web UI. This
is the "web first" step toward the eventual website and VST.

- **Discover** — sample a foreign minor + dense loop, repaint it in cKz timbre →
  new melodies. (`init_sweep.py` recipe: σ0.75, cfg3, 50 steps, dpmpp, strength1.)
- **Refine** — hold a picked clip's melody, move the timbre: **subtle** (σ0.45) or
  **bold** (σ0.75), vary the seed. (`mode2_sweep.py` recipe.)

Every generation setting lives in one place: [`app/config.py`](config.py). Nothing
is re-derived; the values are the settled ones from the handoff.

## Layout

```
app/
  config.py        locked settings (model, cfg/steps/σ, loop dirs, titles, keys)
  engine.py        RealEngine (aug-8000, loaded once) + MockEngine (no model)
  loops.py         foreign-loop parse/filter/sample (minor + dense)
  index_loops.py   CLI: build loops_manifest.json (run once on the Mac)
  density.py       silence/dyn reject-gate (the one safe automation)
  jobs.py          single-worker generation queue + clip registry
  library.py       kept-clip store (kept/ + library.json)
  server.py        FastAPI app + audio serving
  static/index.html  browse feed, waveforms, keep/discard, "More like this"
```

## Run it (on the Mac — the model needs MPS)

```bash
cd ~/stable-audio-3

# 1. install backend deps (FastAPI/uvicorn/soundfile)
uv sync --extra app          # or: uv pip install -e ".[app]"

# 2. build the foreign-loop manifest ONCE (reads your Cymatics wavs)
uv run python -m app.index_loops
#   override the loop folder if needed:
#   CKZ_LOOPS_ROOT="/path/to/Melody_Loops" uv run python -m app.index_loops

# 3. start the service (loads aug-8000 — slow the first time)
uv run python -m app.server          # http://127.0.0.1:8800
```

Open **http://127.0.0.1:8800**, set how many clips, pick the melody spread
(both / match / tritone), hit **Generate**. Clips stream into the feed as each
finishes (≈1 min/clip on M4 Max). **Keep** the good ones; **More ▾ → subtle/bold**
runs Refine on that clip. The **Library** tab holds everything you've kept.

## Dev without the model (mock engine)

No MPS / no weights needed — synthesizes placeholder audio so you can work on the
API and frontend:

```bash
uv run python -m app.server --mock
```

(The smoke test in this PR exercises the whole stack this way.)

## API

| Method | Path | Body / notes |
|---|---|---|
| `GET`  | `/api/status` | engine, loop count, locked settings |
| `POST` | `/api/discover` | `{count, key_mode:"both"\|"match"\|"tritone", seed?}` → job |
| `POST` | `/api/refine` | `{source:"clip:<id>"\|"kept:<id>", mode:"subtle"\|"bold", count, seed?}` → job |
| `GET`  | `/api/jobs/{id}` | poll: `{done,total,clips:[…]}` |
| `POST` | `/api/keep` | `{clip_id}` → copies to `kept/` + `library.json` |
| `GET`  | `/api/library` | kept clips |
| `DELETE` | `/api/library/{id}` | unkeep |
| `GET`  | `/audio/gen/{f}`, `/audio/kept/{f}`, `/audio/init/{clip_id}` | wav |

Requests are queued and run by **one** worker (the model isn't concurrency-safe).
Generation is deterministic, so output files are named by a hash of
`(prompt, seed, σ, init)` and identical requests reuse the cached wav.

## Notes / guardrails baked in

- **σ is fixed per mode** (0.75 discover · 0.45 subtle · 0.75 bold). The handoff's
  "half-baked valley" (0.5–0.6) and degrade zone (0.8+) aren't exposed.
- **Timbre is fixed-cKz by design** — melody is the diversifiable axis (loop > seed
  > key). There is intentionally no "timbre width" knob.
- **Density gate** flags sparse/dead-air duds (`sparse` badge) but never deletes —
  you still judge by ear.
- `app/gen/`, `app/kept/`, and `app/loops_manifest.json` are gitignored (local only).

## Toward the VST

The model is PyTorch (not embeddable in-process), so the VST will talk to this
same service over HTTP — keep this backend resident and the plugin becomes a thin
client of `/discover` and `/refine`.
