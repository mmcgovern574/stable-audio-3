"""Generation engines. The model is loaded ONCE and kept resident (load is slow,
~1 min/clip generation on M4 Max). The server holds a single Engine instance and
serializes calls through one worker thread (the model is not concurrency-safe).

Two implementations behind one interface:
  • RealEngine — Stable Audio 3 medium-base + aug-8000 LoRA. Exact generate()
    contract copied from scripts/init_sweep.py and scripts/mode2_sweep.py.
  • MockEngine — synthesizes placeholder audio (numpy + stdlib wave, NO torch),
    so the API + frontend can be developed and tested without the model/MPS.

Generation is deterministic: same (prompt, seed, σ, init) → identical audio. The
server names output files by a hash of that tuple, so identical requests reuse
the cached wav and never regenerate.
"""
from __future__ import annotations

import hashlib
import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config as C


@dataclass
class GenResult:
    path: Path
    sr: int
    duration: float


def _key_root_hz(prompt: str) -> float:
    """Pull the key out of a cKz prompt → a root frequency (for mock melodies)."""
    for tok in prompt.replace("@crushed_keyz", "").split():
        t = tok.rstrip("minaj")
        base = t.rstrip("min").rstrip("maj")
        if base and base[0] in "ABCDEFG":
            note = base[:2] if len(base) > 1 and base[1] == "#" else base[:1]
            if note in C.KEYS:
                # A4 = 440; map pitch class into a low-ish octave for a bassy loop
                semis = C.KEYS.index(note) - C.KEYS.index("A")
                return 220.0 * (2 ** (semis / 12.0))
    return 220.0


class Engine:
    name = "base"
    sr = 44100

    def probe_duration(self, path: Optional[str]) -> float:
        raise NotImplementedError

    def generate(self, *, prompt: str, seed: int, out_path: Path,
                 sigma: Optional[float] = None,
                 init_path: Optional[str] = None) -> GenResult:
        raise NotImplementedError

    def _gen_dur(self, init_path: Optional[str]) -> float:
        if init_path:
            return round(min(self.probe_duration(init_path), C.MAX_DURATION), 1)
        return C.MAX_DURATION


# ── Real model ──────────────────────────────────────────────────────────────
class RealEngine(Engine):
    name = "real"

    def __init__(self, model_name: str = C.BASE_MODEL, ckpt: str = C.LORA_CKPT):
        import torch  # noqa
        import torchaudio  # noqa
        from stable_audio_3 import StableAudioModel

        self._torch = torch
        self._ta = torchaudio
        print(f"[engine] loading {model_name} + LoRA {ckpt} (resident; this is slow)…")
        self.model = StableAudioModel.from_pretrained(model_name)
        self.model.load_lora([str((C.REPO / ckpt).resolve())])
        self.model.set_lora_strength(C.LORA_STRENGTH)
        self.sr = self.model.model_config.get("sample_rate") or self.model.model.sample_rate
        self._initcache: dict[str, tuple] = {}
        print(f"[engine] ready. sample_rate={self.sr}")

    def probe_duration(self, path: Optional[str]) -> float:
        if not path:
            return C.MAX_DURATION
        info = self._ta.info(str(path))
        return info.num_frames / info.sample_rate

    def _load_init(self, path: str, dur: float):
        if path not in self._initcache:
            self._initcache[path] = self._ta.load(str(path))
        wav, sr = self._initcache[path]
        return sr, wav[:, : int(dur * sr)]

    def generate(self, *, prompt, seed, out_path, sigma=None, init_path=None) -> GenResult:
        gen_dur = self._gen_dur(init_path)
        kw = dict(prompt=prompt, duration=gen_dur, steps=C.STEPS,
                  cfg_scale=C.CFG_SCALE, seed=seed, sampler_type=C.SAMPLER)
        if init_path is not None and sigma is not None:
            sr_i, wav_i = self._load_init(init_path, gen_dur)
            kw["init_audio"] = (int(sr_i), wav_i)
            kw["init_noise_level"] = sigma
        a = self.model.generate(**kw)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._ta.save(str(out_path), a[0].cpu(), self.sr)
        return GenResult(path=out_path, sr=self.sr, duration=gen_dur)


# ── Mock model (sandbox / frontend dev) ─────────────────────────────────────
class MockEngine(Engine):
    name = "mock"

    def __init__(self, sr: int = 44100):
        self.sr = sr

    def probe_duration(self, path: Optional[str]) -> float:
        if path and Path(path).is_file() and str(path).lower().endswith(".wav"):
            try:
                with wave.open(str(path), "rb") as w:
                    return w.getnframes() / w.getframerate()
            except Exception:
                pass
        return C.MAX_DURATION

    def generate(self, *, prompt, seed, out_path, sigma=None, init_path=None) -> GenResult:
        import numpy as np

        gen_dur = self._gen_dur(init_path)
        n = int(gen_dur * self.sr)
        h = int(hashlib.sha1(f"{prompt}|{seed}|{sigma}".encode()).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        t = np.arange(n) / self.sr
        root = _key_root_hz(prompt)

        # minor scale degrees (semitones) for an arpeggiated placeholder loop
        scale = [0, 3, 5, 7, 10, 12]
        bpm = 120
        for tok in prompt.split():
            if tok.isdigit():
                bpm = int(tok)
                break
        step = 60.0 / bpm / 2.0  # eighth notes
        n_steps = max(1, int(gen_dur / step))
        left = np.zeros(n, dtype=np.float64)
        right = np.zeros(n, dtype=np.float64)
        pattern = rng.integers(0, len(scale), size=n_steps)
        for i, deg in enumerate(pattern):
            f = root * (2 ** (scale[int(deg)] / 12.0))
            s0 = int(i * step * self.sr)
            s1 = min(int((i + 1) * step * self.sr), n)
            if s1 <= s0:
                continue
            env = np.exp(-3.0 * (np.arange(s1 - s0) / self.sr) / step)
            tone = (np.sin(2 * np.pi * f * t[s0:s1])
                    + 0.5 * np.sin(2 * np.pi * 2 * f * t[s0:s1])
                    + 0.25 * np.sin(2 * np.pi * 3 * f * t[s0:s1]))
            seg = (tone * env).astype(np.float64)
            # σ widens the "timbre": more noise + brighter partial at higher σ
            if sigma:
                seg += float(sigma) * 0.15 * rng.standard_normal(s1 - s0) * env
            pan = 0.5 + 0.4 * math.sin(i)  # gently moving stereo image
            left[s0:s1] += seg * (1 - pan)
            right[s0:s1] += seg * pan

        stereo = np.stack([left, right], axis=0)
        peak = np.max(np.abs(stereo)) or 1.0
        stereo = (stereo / peak) * 0.85
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = (stereo.T * 32767).astype("<i2")
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(data.tobytes())
        return GenResult(path=out_path, sr=self.sr, duration=gen_dur)


def make_engine(mock: bool = False) -> Engine:
    return MockEngine() if mock else RealEngine()
