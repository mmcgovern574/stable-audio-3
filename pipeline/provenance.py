"""Shared provenance + audio-analysis helpers for the SoundSauce pipeline.

Stdlib-only core (ids, hashes). numpy + ffmpeg/ffprobe are used *best-effort* for audio
stats and the content embedding — if they're missing those fields come back None and the
import still works. See SCALABLE_PIPELINE.md.
"""
import os, json, uuid, hashlib, subprocess, shutil

# Fixed namespace so deterministic ids are stable across runs/machines.
NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


# ---- ids -------------------------------------------------------------------
def stable_id(source_name: str) -> str:
    """Deterministic UUID for a source file — lets the backfill be re-run idempotently
    (same input file -> same id -> upsert, never a duplicate)."""
    return str(uuid.uuid5(NS, "sample:" + source_name))

def new_id() -> str:
    """Fresh random UUID — what generators mint per clip at generation time."""
    return str(uuid.uuid4())


# ---- hashing ---------------------------------------------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---- audio stats (best effort) --------------------------------------------
def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def _decode_mono(path: str, sr: int = 22050):
    """ffmpeg -> mono float32 numpy. Returns (np.ndarray | None, sr)."""
    if not _have("ffmpeg"):
        return None, sr
    try:
        import numpy as np
    except Exception:
        return None, sr
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True,
    )
    if out.returncode != 0 or not out.stdout:
        return None, sr
    return np.frombuffer(out.stdout, dtype=np.float32).copy(), sr

def _probe(path: str):
    """(duration_sec, sample_rate) via ffprobe, else (None, None)."""
    if not _have("ffprobe"):
        return None, None
    try:
        d = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=sample_rate",
             "-of", "json", path], capture_output=True, text=True)
        j = json.loads(d.stdout or "{}")
        dur = j.get("format", {}).get("duration")
        dur = round(float(dur), 3) if dur else None
        sr = next((int(s["sample_rate"]) for s in j.get("streams", []) if s.get("sample_rate")), None)
        return dur, sr
    except Exception:
        return None, None

def _chroma(x, sr) -> list | None:
    """12-d mean chroma (pitch-class profile), normalized — a compact content embedding
    for dedup / similarity / 'more like this'."""
    try:
        import numpy as np
    except Exception:
        return None
    n_fft, hop = 2048, 512
    if x.size < n_fft:
        return None
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    pc = np.full(freqs.shape, -1)
    m = freqs > 0
    pc[m] = np.round(12 * np.log2(freqs[m] / 440.0)).astype(int) % 12
    win = np.hanning(n_fft).astype(np.float32)
    nfr = max(1, 1 + (x.size - n_fft) // hop)
    C = np.zeros(12, np.float64)
    for i in range(nfr):
        mag = np.abs(np.fft.rfft(x[i * hop:i * hop + n_fft] * win))
        for p in range(12):
            C[p] += mag[pc == p].sum()
    s = C.sum()
    if s <= 0:
        return None
    return [round(float(v), 5) for v in (C / s)]

def audio_stats(path: str) -> dict:
    """sha256 (always) + duration/sample_rate/peak_dbfs/embedding (best effort)."""
    dur, sr = _probe(path)
    stats = {"output_sha256": sha256_file(path), "duration_sec": dur, "sample_rate": sr,
             "peak_dbfs": None, "embedding": None}
    x, xr = _decode_mono(path, 22050)
    if x is not None and x.size > 0:
        import numpy as np
        if stats["duration_sec"] is None:
            stats["duration_sec"] = round(x.size / xr, 3)
        pk = float(np.max(np.abs(x)))
        stats["peak_dbfs"] = round(float(20 * np.log10(pk)), 2) if pk > 0 else None
        stats["embedding"] = _chroma(x, xr)
    return stats


# ---- known registry rows (Crushed Keyz) -----------------------------------
# Deterministic ids so the importer can reference them without a read round-trip.
PRODUCER = {
    "id": str(uuid.uuid5(NS, "producer:@crushed_keyz")),
    "name": "Crushed Keyz",
    "handle": "@crushed_keyz",
    "links": {"linktree": "https://linktr.ee/crushedkeyz"},
}
MODEL = {
    "id": str(uuid.uuid5(NS, "model:cKz-aug-8000")),
    "name": "cKz-aug-8000",
    "base_model": "stable-audio-3-medium-base",
    "lora_ckpt": "lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt",
    "default_recipe": {"cfg": 3.0, "steps": 50, "sampler": "dpmpp", "strength": 1.0, "init_noise": 0.75},
}

# Recovered + verified 2026-06-28 (see memory: soundsauce_catalog_dedup). The recipe the
# existing 69 catalog loops were generated with (scripts/init_sweep.py, foreign-init).
CKZ_AUG_RECIPE = {
    "engine": "foreign-init", "init_noise": 0.75, "cfg": 3.0, "steps": 50,
    "sampler": "dpmpp", "strength": 1.0, "base_model": "medium-base",
    "lora_ckpt": "lora_out/ckz_aug_v1_long2/epoch=39-step=8000.ckpt",
}
