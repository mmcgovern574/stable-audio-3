"""End-to-end smoke test of the cKz UI backend (app/) using the MOCK engine.

No model / MPS needed. Routes all generated artifacts into a tmp dir so the repo
stays clean. Run:  uv run pytest tests/test_app_smoke.py -q
(requires the `app` extra: fastapi/uvicorn/soundfile)
"""
import json
import time
import wave
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from starlette.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app.config as C
    monkeypatch.setattr(C, "APP_DIR", tmp_path)
    monkeypatch.setattr(C, "GEN_DIR", tmp_path / "gen")
    monkeypatch.setattr(C, "KEPT_DIR", tmp_path / "kept")
    monkeypatch.setattr(C, "LIBRARY_JSON", tmp_path / "kept" / "library.json")
    monkeypatch.setattr(C, "LOOPS_MANIFEST", tmp_path / "loops_manifest.json")
    C.LOOPS_MANIFEST.write_text(json.dumps({"loops": [
        {"path": str(tmp_path / "A.wav"), "key": "Amin",  "bpm": 140, "name": "A 140 BPM A Min.wav"},
        {"path": str(tmp_path / "B.wav"), "key": "C#min", "bpm": 150, "name": "B 150 BPM C# Min.wav"},
        {"path": str(tmp_path / "C.wav"), "key": "Fmin",  "bpm": 130, "name": "C 130 BPM F Min.wav"},
        {"path": str(tmp_path / "D.wav"), "key": "Gmin",  "bpm": 144, "name": "D 144 BPM G Min.wav"},
    ]}))
    import app.server as S
    S.build_app(mock=True)
    return TestClient(S.app)


def _poll(client, job_id, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["done"] >= j["total"]:
            return j
        time.sleep(0.1)
    raise AssertionError("job timed out")


def test_status_locked_settings(client):
    s = client.get("/api/status").json()
    assert s["engine"] == "mock" and s["loops"] == 4
    e = s["settings"]
    assert (e["cfg"], e["steps"], e["sampler"], e["strength"]) == (3.0, 50, "dpmpp", 1.0)
    assert e["sigma_discover"] == 0.75 and e["sigma_subtle"] == 0.45 and e["sigma_bold"] == 0.75


def test_discover_prompts_and_keys(client):
    job = client.post("/api/discover", json={"count": 4, "key_mode": "both"}).json()
    clips = job["clips"]
    assert len(clips) == 4
    assert {c["key_mode"] for c in clips} == {"match", "tritone"}
    assert all(c["prompt"].startswith("cKz! ") and c["prompt"].endswith("@crushed_keyz") for c in clips)
    assert all(c["sigma"] == 0.75 for c in clips)
    done = _poll(client, job["job_id"])
    assert all(c["status"] == "done" and c["audio_url"] for c in done["clips"])
    assert all(c["sparse"] is not None for c in done["clips"])  # density gate ran


def test_audio_is_valid_wav(client):
    job = client.post("/api/discover", json={"count": 1, "key_mode": "match"}).json()
    c = _poll(client, job["job_id"])["clips"][0]
    import app.config as C
    wpath = C.GEN_DIR / Path(c["audio_url"]).name
    with wave.open(str(wpath), "rb") as w:
        assert w.getnchannels() == 2 and w.getframerate() == 44100
        assert abs(w.getnframes() / w.getframerate() - 20.0) < 0.2


def test_determinism_caches(client):
    a = _poll(client, client.post("/api/discover", json={"count": 2, "key_mode": "match", "seed": 7}).json()["job_id"])
    b = _poll(client, client.post("/api/discover", json={"count": 2, "key_mode": "match", "seed": 7}).json()["job_id"])
    assert [c["audio_url"] for c in a["clips"]] == [c["audio_url"] for c in b["clips"]]


def test_keep_library_refine_unkeep(client):
    job = _poll(client, client.post("/api/discover", json={"count": 2, "key_mode": "match"}).json()["job_id"])
    c0 = job["clips"][0]

    k = client.post("/api/keep", json={"clip_id": c0["id"]}).json()
    assert k["prompt"] == c0["prompt"]
    assert len(client.get("/api/library").json()["items"]) == 1
    client.post("/api/keep", json={"clip_id": c0["id"]})  # idempotent
    assert len(client.get("/api/library").json()["items"]) == 1

    # refine from clip (bold) — same melody prompt, varied seeds, σ flips to 0.75
    rj = client.post("/api/refine", json={"source": f"clip:{c0['id']}", "mode": "bold", "count": 3}).json()
    assert all(c["sigma"] == 0.75 and c["prompt"] == c0["prompt"] for c in rj["clips"])
    assert len({c["seed"] for c in rj["clips"]}) == 3
    _poll(client, rj["job_id"])

    # refine from kept (subtle) — σ0.45
    rj2 = client.post("/api/refine", json={"source": f"kept:{k['id']}", "mode": "subtle", "count": 2}).json()
    assert all(c["sigma"] == 0.45 for c in rj2["clips"])
    _poll(client, rj2["job_id"])

    client.delete(f"/api/library/{k['id']}")
    assert client.get("/api/library").json()["items"] == []
