#!/usr/bin/env python3
"""Matrix parameter sweep — fast in-process version.

Loads the medium-base model + LoRA ONCE, then loops generation in a single
Python process. The previous subprocess-based version paid ~30 s/clip in
model-loading overhead. This pays it once.

Wall-time comparison (48-clip sweep on M-Max):
   Old: ~55 min  (50% on overhead, 50% on actual generation)
   New: ~33 min  (~30 s startup, then pure generation)

Usage:
    uv run python sweep_rater/sweep_matrix.py                    # default config
    uv run python sweep_rater/sweep_matrix.py --config foo.json  # custom
    uv run python sweep_rater/sweep_matrix.py --dry-run          # print plan only

Then rate:
    python sweep_rater/rate_server.py --folder sweep_rater/campaigns/<name>
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Match HF_HUB_DISABLE_XET behavior used in train_lora.py.  Stable-audio's
# from_pretrained may pull weights through HF Hub if not cached.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

# torch.compile is OFF by default. The PyTorch MPS Inductor backend is a
# prototype (per the upstream warning) and was observed to produce NaN audio
# on some prompts even when individual kernels appeared to fall back to eager
# — the partial-graph behaviour can introduce silent numerical errors. The
# in-process model-load speedup is real and lossless; compile is a separate
# opt-in gamble. Enable explicitly with --compile if you want to experiment.
if "--compile" in sys.argv:
    os.environ.setdefault("ENABLE_TORCH_COMPILE", "1")
    os.environ.setdefault("TORCH_LOGS", "")
    os.environ.setdefault("TORCHDYNAMO_VERBOSE", "0")
    import logging as _logging
    _logging.getLogger("torch._dynamo").setLevel(_logging.ERROR)
    _logging.getLogger("torch._inductor").setLevel(_logging.ERROR)

DEFAULT_CONFIG = {
    "name": "cfg_strength_v1",
    "lora_ckpt": "./lora_out/ckz_5000_dora/epoch=108-step=5000.ckpt",
    "model": "medium-base",
    "duration": 20,
    "prompt": "2. cKz! ICE Dmin 135 bpm @crushed_keyz",
    "axes": {
        "cfg":      [3, 5, 7, 9],
        "strength": [0.25, 0.5, 0.75, 1.0],
        "steps":    [50],
        "seed":     [42, 123, 999],
    },
}


def sanitize(s: str, max_len: int = 60) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[/\\:|?*<>\"']", "", s)
    s = re.sub(r"_+", "_", s)
    return s[:max_len].rstrip("_.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",   type=Path, default=None, help="JSON config file")
    p.add_argument("--out-root", type=Path, default=SCRIPT_DIR / "campaigns")
    p.add_argument("--dry-run",  action="store_true")
    p.add_argument("--compile", action="store_true",
                   help="Opt-in: enable torch.compile. Faster but prototype-grade "
                        "on MPS — has been observed to produce NaN audio for some "
                        "prompts. Off by default.")
    p.add_argument("--warmup-steps", type=int, default=4,
                   help="Short pre-generation pass to trigger torch.compile JIT "
                        "before the timed loop. Only used when --compile is set.")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text()) if args.config else DEFAULT_CONFIG

    out_dir = args.out_root / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    axes_order  = list(cfg["axes"].keys())
    axes_values = [cfg["axes"][k] for k in axes_order]
    combos      = list(itertools.product(*axes_values))

    # "prompt" can now be an axis (string-valued) alongside the numeric ones.
    # If present, the per-combo prompt is params["prompt"]; otherwise the
    # global cfg["prompt"] is used. Filenames always show the prompt suffix.
    prompt_is_axis = "prompt" in axes_order
    default_prompt = cfg.get("prompt", "")

    print(f"Sweep:        {cfg['name']}")
    if prompt_is_axis:
        print(f"Prompt axis:  {len(cfg['axes']['prompt'])} prompts")
        for pr in cfg["axes"]["prompt"]:
            print(f"  - {pr}")
    else:
        print(f"Prompt:       {default_prompt}")
    print(f"LoRA ckpt:    {cfg['lora_ckpt']}")
    print(f"Output:       {out_dir}/")
    print(f"Combinations: {len(combos)}")
    for name in axes_order:
        if name == "prompt":
            continue   # already shown above
        print(f"  {name}: {cfg['axes'][name]}")
    print(f"\nLaunch the rater now in another terminal:")
    print(f"  python sweep_rater/rate_server.py --folder {out_dir}\n")

    def _short_axis_value(k, v):
        """Render an axis value compactly for the filename."""
        if k == "init_audio":
            # Just the basename stem, no extension, no path. Cap length.
            stem = Path(str(v)).stem
            return re.sub(r"[^A-Za-z0-9!#@\.]+", "_", stem)[:35]
        return v

    # Plan first: which combos still need generating?
    todo = []
    for i, combo in enumerate(combos, start=1):
        params = dict(zip(axes_order, combo))
        this_prompt = params.get("prompt", default_prompt)
        # Build the param string without the prompt key (it's appended below).
        param_items = []
        for k in axes_order:
            if k == "prompt":
                continue
            v = _short_axis_value(k, params[k])
            if k == "init_audio":
                param_items.append(f"init={v}")
            elif k == "init_noise_level":
                param_items.append(f"noise{v}")
            elif k == "negative_prompt":
                param_items.append("negOn" if str(params[k]).strip() else "negOff")
            elif k == "sampler":
                param_items.append(f"samp-{params[k]}")
            elif k == "apg":
                param_items.append(f"apg{params[k]}")
            elif k == "rescale_cfg":
                param_items.append("rescaleOn" if params[k] else "rescaleOff")
            elif k == "lora_start_sigma":
                param_items.append(f"gate{v}")
            elif k == "lora_low":
                param_items.append(f"low{v}")
            elif k == "lora_layer_filter":
                safe = re.sub(r"[^A-Za-z0-9]+", "-", str(params[k]).strip()).strip("-")
                param_items.append(f"lf-{safe or 'all'}")
            elif k == "cfg_interval":
                ci = params[k]
                param_items.append(f"cfgi{ci[0]}-{ci[1]}")
            else:
                param_items.append(f"{k}{v}")
        param_str = "_".join(param_items)
        filename = f"{i:04d}__{param_str}__{sanitize(this_prompt)}.wav"
        out_path = out_dir / filename
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[{i:>3}/{len(combos)}] skip (exists)  {filename}")
            continue
        todo.append((i, params, filename, out_path))

    if args.dry_run:
        print(f"\n--dry-run: would generate {len(todo)} clips.")
        for i, params, filename, _ in todo:
            print(f"  [{i:>3}/{len(combos)}] {params}  →  {filename}")
        return

    if not todo:
        print("\nAll clips already exist. Nothing to do.")
        return

    # ── One-time model load ─────────────────────────────────────────────
    print(f"\nLoading model ({cfg['model']}) — this takes ~30–60 s on first run,")
    print("then we'll generate all clips without reloading...")
    t0 = time.time()

    import torch  # noqa: E402  (deferred until we know we need it)
    import torchaudio
    from stable_audio_3 import StableAudioModel

    # ── init_audio cache ────────────────────────────────────────────────
    # If init_audio is an axis, the same file path appears in many combos.
    # Load each file once, keep on CPU; .generate() handles device transfer
    # internally via the model's audio_preprocessor.
    _init_audio_cache: dict[str, tuple[int, "torch.Tensor"]] = {}

    def load_init_audio(rel_path: str) -> tuple[int, "torch.Tensor"]:
        """Load (and cache) an init-audio file relative to the repo root.

        Returns (sample_rate, tensor) as required by StableAudioModel.generate().
        Tensor shape is (channels, samples); mono inputs are kept mono — the
        model's audio preprocessor will replicate to stereo as needed.
        """
        if rel_path in _init_audio_cache:
            return _init_audio_cache[rel_path]
        full = (REPO_ROOT / rel_path).resolve()
        if not full.exists():
            sys.exit(f"init_audio file not found: {full}")
        wav, sr = torchaudio.load(str(full))
        # Cache on CPU; the model moves to MPS as part of its preprocessing.
        _init_audio_cache[rel_path] = (int(sr), wav)
        print(f"loaded init_audio: {rel_path}  ({wav.shape[0]}ch, "
              f"{wav.shape[1]/sr:.1f}s @ {sr}Hz)")
        return _init_audio_cache[rel_path]

    model = StableAudioModel.from_pretrained(cfg["model"])
    if cfg["lora_ckpt"]:
        ckpt_path = (REPO_ROOT / cfg["lora_ckpt"]).resolve()
        if not ckpt_path.exists():
            sys.exit(f"LoRA checkpoint not found: {ckpt_path}")
        model.load_lora([str(ckpt_path)])

    # StableAudioModel doesn't expose sample_rate directly; pull from the
    # model_config dict, with a fallback to the inner wrapped model.
    sample_rate = model.model_config.get("sample_rate") or model.model.sample_rate
    load_secs = time.time() - t0
    print(f"Model ready in {load_secs:.1f} s. sample_rate={sample_rate}")

    compile_on = os.environ.get("ENABLE_TORCH_COMPILE") == "1"
    print(f"torch.compile: {'ON' if compile_on else 'off'}")

    # ── Warm-up pass ────────────────────────────────────────────────────
    # If torch.compile is on, the FIRST generate() call triggers JIT
    # compilation (can be 30-90 s on MPS). Do a short throwaway pass with
    # only a few diffusion steps so the JIT cost happens BEFORE the timed
    # loop, and individual clips report honest steady-state times.
    if compile_on and args.warmup_steps > 0:
        first_combo = combos[0]
        first_params = dict(zip(axes_order, first_combo))
        warmup_strength = float(first_params.get("strength", 1.0))
        model.set_lora_strength(warmup_strength)
        last_strength = warmup_strength
        print(f"\nWarming up torch.compile ({args.warmup_steps} steps; "
              f"first JIT compile may take ~30–90 s on MPS)...")
        wt0 = time.time()
        try:
            _ = model.generate(
                prompt=first_params.get("prompt", default_prompt),
                duration=cfg["duration"],
                steps=args.warmup_steps,
                cfg_scale=float(first_params.get("cfg", 7)),
                seed=42,
            )
            print(f"Warm-up done in {time.time()-wt0:.1f} s.\n")
        except Exception as e:
            print(f"Warm-up failed ({type(e).__name__}: {e})")
            print("Continuing without compile benefit; subsequent clips may "
                  "re-trigger JIT on each unique shape.\n")
    else:
        last_strength = None
        print()

    # ── Loop ────────────────────────────────────────────────────────────
    # last_strength is already set above (either by warm-up or to None).
    failed = 0
    gen_start = time.time()
    for n, (i, params, filename, out_path) in enumerate(todo, start=1):
        # Only change LoRA strength when it actually differs.
        s = float(params.get("strength", 1.0))
        if s != last_strength:
            model.set_lora_strength(s)
            last_strength = s

        steps_n = int(params.get("steps", 50))
        cfg_n   = float(params.get("cfg", 7))
        seed_n  = int(params.get("seed", 42))

        this_prompt = params.get("prompt", default_prompt)
        print(f"[{n:>3}/{len(todo)}] (combo {i:>3}/{len(combos)}) "
              f"cfg={cfg_n} strength={s} steps={steps_n} seed={seed_n}")
        if prompt_is_axis:
            print(f"            prompt: {this_prompt}")

        # Optional init_audio / init_noise_level (img2img-style for audio).
        # Pass through only when those axes are present in the config so
        # behaviour is unchanged for sweeps that don't use them.
        gen_kwargs: dict = {}
        if "init_audio" in params:
            gen_kwargs["init_audio"] = load_init_audio(params["init_audio"])
            print(f"            init_audio: {params['init_audio']}")
        if "init_noise_level" in params:
            gen_kwargs["init_noise_level"] = float(params["init_noise_level"])
            print(f"            init_noise_level: {gen_kwargs['init_noise_level']}")
        # Negative prompt: per-combo axis value, else a global cfg["negative_prompt"].
        # Empty string = no negative prompt (the "off" arm of an A/B).
        neg = params.get("negative_prompt", cfg.get("negative_prompt"))
        if neg and str(neg).strip():
            gen_kwargs["negative_prompt"] = neg
            print(f"            negative_prompt: {neg}")
        # New sampler / guidance levers (pass through to sample_diffusion).
        if "apg" in params:
            gen_kwargs["apg_scale"] = float(params["apg"])
        if "sampler" in params:
            gen_kwargs["sampler_type"] = str(params["sampler"])
        if "rescale_cfg" in params:
            gen_kwargs["rescale_cfg"] = bool(params["rescale_cfg"])

        # Time-gated LoRA (the style-transfer lever). lora_start_sigma is the
        # sigma at/below which the adapter switches to full strength; above it
        # (early/high-noise structure steps) the adapter runs at lora_low
        # (default 0.0 = base model decides the melody). sigma runs from
        # sigma_max (1.0, or init_noise_level) down to 0.
        gated = "lora_start_sigma" in params
        if gated:
            start_sigma = float(params["lora_start_sigma"])
            lora_low = float(params.get("lora_low", 0.0))
            lora_full = float(params.get("strength", 1.0))

            def _schedule(sigma, _s=start_sigma, _lo=lora_low, _hi=lora_full):
                return _hi if sigma <= _s else _lo

            gen_kwargs["lora_strength_schedule"] = _schedule
            print(f"            lora gate: sigma<= {start_sigma} -> {lora_full}, "
                  f"else {lora_low}")

        # Native LoRA layer filter (exclusion: matched layer-name substrings get
        # the adapter DISABLED; all others stay on). Passes through to the DiT
        # forward, which calls filter_lora_layers() each step.
        if "lora_layer_filter" in params:
            lf = str(params["lora_layer_filter"])
            gen_kwargs["lora_layer_filter"] = lf
            if os.environ.get("CKZ_GATE_DEBUG"):
                try:
                    from stable_audio_3.models.lora.utils import get_lora_layers
                    from stable_audio_3.models.lora.model import _expand
                    pats = [x.strip().lower() for x in lf.split(",") if x.strip()]
                    allL = get_lora_layers(model.dit)
                    def _hit(n):
                        n = n.lower()
                        return any(e in n for p in pats for e in _expand(p))
                    dis = sum(1 for n, _ in allL if pats and _hit(n))
                    print(f"            layer_filter '{lf or '(none)'}': "
                          f"{dis}/{len(allL)} LoRA layers OFF, {len(allL)-dis} active")
                except Exception as e:
                    print(f"            (layer_filter debug failed: {type(e).__name__}: {e})")

        # Native time-gated CFG: cfg_interval=(min,max) sigma where guidance is
        # applied; outside it CFG is skipped (LoRA stays fully on => timbre safe).
        if "cfg_interval" in params:
            ci = params["cfg_interval"]
            gen_kwargs["cfg_interval"] = (float(ci[0]), float(ci[1]))
            print(f"            cfg_interval: {gen_kwargs['cfg_interval']}")

        clip_t0 = time.time()
        try:
            audio = model.generate(
                prompt=this_prompt,
                duration=cfg["duration"],
                steps=steps_n,
                cfg_scale=cfg_n,
                seed=seed_n,
                **gen_kwargs,
            )
            # generate() returns (batch, channels, samples); take first row.
            torchaudio.save(str(out_path), audio[0].cpu(), sample_rate)
            print(f"            saved in {time.time()-clip_t0:.1f} s  →  {filename}")
        except Exception as e:
            failed += 1
            print(f"            FAILED: {type(e).__name__}: {e}")

        # A gated generation mutates the backbone LoRA buffers per-step, outside
        # the harness's last_strength tracking. Force the next combo to re-apply
        # its global strength so non-gated combos aren't left at the gate value.
        if gated:
            last_strength = None

    total = time.time() - gen_start
    print(f"\nGenerated {len(todo)-failed}/{len(todo)} clips in {total:.1f} s "
          f"({total/max(len(todo)-failed,1):.1f} s/clip average).")
    print(f"Total wall time including model load: {time.time()-t0:.1f} s")
    if failed:
        print(f"Failures: {failed}")
    print(f"\nRate: python sweep_rater/rate_server.py --folder {out_dir}")


if __name__ == "__main__":
    main()
