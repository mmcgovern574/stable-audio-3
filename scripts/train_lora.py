"""
Simple LoRA fine-tuning for Stable Audio 3.

Two dataset modes (exactly one required):

  --data_dir    Raw audio + caption pairs. Each clip needs a matching .txt file:
    data_dir/
      clip1.wav   (or .flac, .mp3, .ogg)
      clip1.txt   ← text prompt for clip1
      clip2.wav
      clip2.txt

  --encoded_dir Pre-encoded latents from pre_encode_dataset.py. Captions are
                already embedded in the .json metadata — no .txt files needed:
    encoded_dir/
      000000000000.npy
      000000000000.json
      000000000001.npy
      000000000001.json

Saves .safetensors LoRA checkpoints compatible with the inference model and run_gradio.py.

Usage:
  uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data --save_dir ./lora_out
  uv run python scripts/train_lora.py --model medium-base --encoded_dir ./latents_out --save_dir ./lora_out
  uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data --steps 500 --rank 8
"""

# Show HuggingFace download progress by default — silence is a debugging
# nightmare on a slow first run.  Opt out with:
#   HF_HUB_DISABLE_PROGRESS_BARS=1 uv run python scripts/train_lora.py ...
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

# Use the legacy HuggingFace downloader by default.  The newer chunked
# downloader (`hf-xet`) has two known issues: its tqdm bar can render once
# at startup and never refresh again (so a working download looks frozen),
# and the download itself can stall mid-transfer with no recovery.  The
# legacy downloader is slightly slower at peak but has a working progress
# bar and resumes partial downloads cleanly.  Opt back in with:
#   HF_HUB_DISABLE_XET=0 uv run python scripts/train_lora.py ...
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import itertools
import json
from pathlib import Path
import torch
import pytorch_lightning as pl

from stable_audio_3.data.dataset import (
    LatentDatasetConfig,
    LocalDatasetConfig,
    PreEncodedDataset,
    SampleDataset,
    collation_fn,
)
from safetensors.torch import load_file
from stable_audio_3.loading_utils import copy_state_dict
from stable_audio_3.model_configs import base_models
from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3.models.lora.utils import load_lora_checkpoint
from stable_audio_3.training.diffusion import (
    DiffusionCondTrainingWrapper,
    DiffusionCondInpaintDemoCallback,
)
from stable_audio_3._device import AUTOCAST_DEVICE


def load_model(model_name: str, device: torch.device, base_dtype: torch.dtype):
    if model_name not in base_models:
        raise ValueError(
            f"LoRA training requires a base model. Got '{model_name}', valid: {list(base_models)}"
        )
    model_cfg = base_models[model_name]
    local_config, local_ckpt = model_cfg.resolve()
    with open(local_config) as f:
        model_config = json.load(f)
    model = create_diffusion_cond_from_config(model_config)
    copy_state_dict(model, load_file(local_ckpt))

    # On MPS, splitting "cast to dtype on CPU" from "move to device" is much
    # faster than doing both at once — the combined `.to()` casts and ships
    # each tensor separately, lazily JITting an MPS kernel per (shape, dtype)
    # combo. Doing the dtype cast first on CPU avoids that per-tensor JIT.
    if device.type == "mps":
        print(f"[load_model] casting to {base_dtype} on CPU...")
        model.to(dtype=base_dtype)
        print(f"[load_model] moving to {device}...")
        model.to(device=device)
    else:
        model.to(device=device, dtype=base_dtype)
    model.eval().requires_grad_(False)

    if model.pretransform is not None:
        model.pretransform.enable_grad = False
    return model, model_config


def caption_metadata_fn(info, audio):
    # NOTE: import inside the function so dataloader workers on macOS (spawn)
    # have access — they don't inherit the parent script's globals.
    from pathlib import Path
    txt = Path(info["path"]).with_suffix(".txt")
    if not txt.exists():
        return {"__reject__": True}
    return {"prompt": txt.read_text().strip()}


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


def train(args):
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision("high")

    # Pick device-appropriate precision. bf16 is great on CUDA; on MPS it's
    # still gappy in PyTorch so we use fp16 there. CPU stays in fp32.
    if AUTOCAST_DEVICE == "cuda":
        trainer_precision, base_dtype = "bf16-mixed", torch.bfloat16
    elif AUTOCAST_DEVICE == "mps":
        trainer_precision, base_dtype = "16-mixed", torch.float16
    else:
        trainer_precision, base_dtype = "32-true", torch.float32
    if args.base_precision is None:
        args.base_precision = "bf16" if AUTOCAST_DEVICE == "cuda" else "fp16"
    print(f"[train_lora] device={AUTOCAST_DEVICE} precision={trainer_precision} base_dtype={base_dtype} base_precision={args.base_precision}")

    seed = args.seed

    pl.seed_everything(seed, workers=True)

    model, model_config = load_model(
        args.model, torch.device(AUTOCAST_DEVICE), base_dtype
    )

    print(f"[train_lora] model loaded")

    sample_rate = model.sample_rate
    ds_ratio = model.pretransform.downsampling_ratio

    # Align to downsampling ratio
    sample_size = (int(args.duration * sample_rate) // ds_ratio) * ds_ratio

    # Extract tokenizers from conditioners for pre-tokenization in DataLoader workers
    tokenizers = {}
    if hasattr(model, "conditioner"):
        for key, cond in model.conditioner.conditioners.items():
            if hasattr(cond, "tokenizer") and hasattr(cond, "max_length"):
                tokenizers[key] = (cond.tokenizer, cond.max_length)

    if args.encoded_dir:
        dataset = PreEncodedDataset(
            [LatentDatasetConfig(id="train", path=args.encoded_dir)],
            latent_crop_length=sample_size // ds_ratio,
            random_crop=True,
        )
    else:
        dataset = SampleDataset(
            [
                LocalDatasetConfig(
                    id="train",
                    path=args.data_dir,
                    custom_metadata_fn=caption_metadata_fn,
                )
            ],
            sample_size=sample_size,
            sample_rate=sample_rate,
            force_channels="stereo",
            max_crop_offset_sec=args.max_crop_offset_sec,
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collation_fn,
        # No worker_init_fn — `pl.seed_everything(seed, workers=True)` above
        # already seeds each worker as seed+worker_id. A lambda here would
        # also break macOS multiprocessing (spawn can't pickle local lambdas).
    )
    print(f"[train_lora] dataset+dataloader ready ({len(dataset)} samples, num_workers={args.num_workers})")

    lora_state_dict = None
    if args.lora_checkpoint:
        lora_state_dict, _ = load_lora_checkpoint(args.lora_checkpoint)

    lora_config = {
        "rank": args.rank,
        "alpha": args.lora_alpha if args.lora_alpha is not None else args.rank,
        "adapter_type": args.adapter_type,
        "dropout": args.dropout,
        "include": args.include,
        "exclude": args.exclude,
    }
    optimizer_config = {
        "diffusion": {
            "optimizer": {
                "type": "AdamW",
                "config": {
                    "lr": args.lr,
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.95],
                },
            }
        }
    }

    training_wrapper = DiffusionCondTrainingWrapper(
        model,
        mask_loss_weight=1.0,
        mask_padding_attention=True,
        silence_extension_scale_seconds=4.0,
        use_ema=False,
        log_loss_info=False,
        optimizer_configs=optimizer_config,
        pre_encoded=bool(args.encoded_dir),
        timestep_sampler="trunc_logit_normal",
        timestep_sampler_options={},
        inpainting_config={"mask_kwargs": {"mask_type_probabilities": [0.1, 0.8, 0.1]}},
        use_effective_length_for_schedule=True,
        sample_rate=model_config.get("sample_rate", 44100),
        sample_size=model_config.get("sample_size"),
        lora_config=lora_config,
        lora_state_dict=lora_state_dict,
        svd_bases_path=args.svd_bases_path,
        log_every_n_steps=args.log_every,
        ot_coupling=True,
        base_precision=args.base_precision,
    )

    print(f"[train_lora] training wrapper ready (LoRA adapter={args.adapter_type}, rank={args.rank})")

    exc_callback = ExceptionCallback()

    if args.logger == "wandb":
        logger = pl.loggers.WandbLogger(project=args.name)
        logger.watch(training_wrapper)

        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = os.path.join(
                args.save_dir,
                logger.experiment.project,
                logger.experiment.id,
                "checkpoints",
            )
        else:
            checkpoint_dir = None
    elif args.logger == "comet":
        logger = pl.loggers.CometLogger(project=args.name)
        if args.save_dir and isinstance(logger.version, str):
            checkpoint_dir = os.path.join(
                args.save_dir, logger.name, logger.version, "checkpoints"
            )
        else:
            print(
                f"No save_dir specified, using {args.save_dir if args.save_dir else None}."
            )
            checkpoint_dir = args.save_dir if args.save_dir else None
    elif args.logger == "csv":
        logger = pl.loggers.CSVLogger(args.save_dir)
        checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1
    )

    demo_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=collation_fn,
    )

    # Pre-fetch the first batch and cycle it so demos always use the same samples
    demo_batch = next(iter(demo_dl))
    _, metadata = demo_batch
    for j in range(min(4, len(metadata))):
        md = metadata[j]
        print(
            f"Demo sample {j}: prompt={md.get('prompt', '')} seconds_total={md.get('seconds_total', '')}"
        )
    demo_dl = itertools.cycle([demo_batch])

    # Build text-to-music demo conditioning so we get *real* prompt-driven demos
    # at each checkpoint, not just inpaint demos.
    #
    # Source of demo prompts (in priority order):
    #   1. --demo_prompts_file: one prompt per line, blank lines ignored.
    #   2. Otherwise the first --num_t2m_demos prompts from the training dataset
    #      (whatever order SampleDataset returns them in — not necessarily sorted).
    if args.demo_prompts_file:
        with open(args.demo_prompts_file) as f:
            demo_prompts = [line.strip() for line in f if line.strip()]
    else:
        demo_prompts = [
            md.get("prompt", "")
            for md in metadata[: args.num_t2m_demos]
            if md.get("prompt")
        ]

    t2m_conditioning = [
        {"prompt": p, "seconds_total": int(args.duration)}
        for p in demo_prompts
    ]
    if t2m_conditioning:
        src = args.demo_prompts_file or "training dataset"
        print(f"[train_lora] t2m demo prompts ({len(t2m_conditioning)}, source: {src}):")
        for j, c in enumerate(t2m_conditioning):
            print(f"  {j}: {c['prompt']}")

    # Important: the callback's sample_size controls the length of t2m demo
    # generations only (inpaint demos use the training audio's actual length).
    # Defaulting to model_config["sample_size"] forces t2m demos to generate at
    # the model's MAX duration (e.g., 380 s for medium-base), which is ~12×
    # slower than necessary if you're training at --duration 30. Align it to
    # the chosen training duration so t2m demos run in roughly the same time
    # as inpaint demos.
    sr = model_config.get("sample_rate")
    ds_ratio = model_config.get("model", {}).get("pretransform", {}).get(
        "config", {}
    ).get("downsampling_ratio", 1)
    # Align to downsampling ratio so the latent dims work out cleanly.
    # Demo length is decoupled from training --duration via --demo_duration, so
    # demos can be short (fast) even when training on long crops.
    demo_dur = args.demo_duration if args.demo_duration is not None else args.duration
    demo_sample_size = (int(demo_dur * sr) // ds_ratio) * ds_ratio
    if demo_sample_size <= 0:
        demo_sample_size = model_config.get("sample_size")
    print(f"[train_lora] demo sample_size: {demo_sample_size} samples "
          f"({demo_sample_size / sr:.1f} s @ {sr} Hz)")

    demo_callback = DiffusionCondInpaintDemoCallback(
        demo_every=args.demo_every,
        sample_size=demo_sample_size,
        sample_rate=model_config.get("sample_rate"),
        demo_steps=args.demo_steps,
        num_demos=args.num_inpaint_demos,
        demo_cfg_scales=args.demo_cfg_scales,
        demo_dl=demo_dl,
        demo_conditioning=t2m_conditioning,
        demo_dir=os.path.join(args.save_dir, "demos") if args.save_dir else None,
    )

    callbacks = [ckpt_callback, exc_callback, demo_callback]

    # Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})

    if args.logger == "comet":
        logger.log_hyperparams(args_dict)

    if not hasattr(args, "gradient_clip_val") or args.gradient_clip_val == 0:
        args.gradient_clip_val = None

    summary = pl.callbacks.ModelSummary(max_depth=2)
    callbacks.append(summary)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy="auto",
        precision=trainer_precision,
        accumulate_grad_batches=1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        max_steps=args.steps,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,  # If you need to debug validation, change this line
    )
    print(f"[train_lora] trainer ready — starting fit ({args.steps} steps)")

    trainer.fit(training_wrapper, dataloader)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Simple LoRA fine-tuning for Stable Audio 3"
    )
    p.add_argument("--model", choices=list(base_models), default="medium-base")
    p.add_argument(
        "--data_dir",
        default=None,
        help="Folder with audio files and matching .txt captions",
    )
    p.add_argument(
        "--encoded_dir",
        default=None,
        help="Pre-encoded latent directory from pre_encode_dataset.py (.npy/.json pairs; captions embedded in .json, no .txt needed)",
    )
    p.add_argument("--rank", type=int, default=16)
    p.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha scaling factor (default: same as rank)",
    )
    p.add_argument(
        "--adapter_type",
        choices=[
            "lora",
            "dora",
            "dora-rows",
            "dora-cols",
            "bora",
            "lora-xs",
            "dora-rows-xs",
            "dora-cols-xs",
            "bora-xs",
        ],
        default="dora-rows",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability applied to LoRA inputs",
    )
    p.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Only apply LoRA to modules whose name contains one of these substrings",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Skip modules whose name contains one of these substrings",
    )
    p.add_argument(
        "--svd_bases_path",
        default=None,
        help="Path to pre-computed SVD bases (.pt) for -XS adapter types",
    )
    p.add_argument(
        "--base_precision",
        choices=["bf16", "bfloat16", "fp16", "float16"],
        default=None,
        help="Cast frozen base weights to lower precision (LoRA params stay fp32). "
             "Defaults to bf16 on CUDA, fp16 on MPS.",
    )
    p.add_argument(
        "--lora_checkpoint",
        default=None,
        help="Path to an existing LoRA .safetensors checkpoint to resume from",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--duration",
        type=float,
        default=380.0,
        help="Maximum clip duration in seconds (default 380)",
    )
    p.add_argument(
        "--max_crop_offset_sec",
        type=float,
        default=None,
        help="Cap the random crop start offset to [0, N] seconds. Keeps crops "
             "near the start of each file (e.g. the melody body, excluding "
             "trailing step-outs). Default: None (crop anywhere).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logger", choices=["wandb", "comet", "csv", "none"], default="csv")
    p.add_argument("--name", type=str, default="lora-finetune")
    p.add_argument("--save_dir", type=str, default="./lora_checkpoints")
    p.add_argument("--checkpoint_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--demo_every", type=int, default=500)
    p.add_argument(
        "--demo_duration",
        type=float,
        default=None,
        help="Length of demo clips in seconds. Defaults to --duration. Set shorter "
             "(e.g. 12) to keep demos fast when training on long crops.",
    )
    p.add_argument(
        "--num_inpaint_demos",
        type=int,
        default=4,
        help="Number of inpaint demos (reconstructions of training audio) per "
             "checkpoint. Set 0 to skip them — faster, and removes the seeded-from-"
             "training-audio tail from the demo files.",
    )
    p.add_argument(
        "--demo_steps",
        type=int,
        default=50,
        help="Diffusion sampling steps per demo. 50 is decent quality; bump to 100 for better.",
    )
    p.add_argument(
        "--demo_cfg_scales",
        type=int,
        nargs="+",
        default=[2, 4, 7],
        help="CFG scales to render at each demo checkpoint. Each scale renders "
             "the full prompt set, so demos = prompts x scales. Use a single "
             "value (e.g. --demo_cfg_scales 4) to keep demo time down.",
    )
    p.add_argument(
        "--num_t2m_demos",
        type=int,
        default=4,
        help="Number of text-to-music (prompt-only) demos per checkpoint, "
             "using the first N training-set prompts. Set to 0 to disable t2m demos "
             "and keep only inpaint demos (the old behavior). Ignored if "
             "--demo_prompts_file is set.",
    )
    p.add_argument(
        "--demo_prompts_file",
        type=str,
        default=None,
        help="Optional path to a text file containing demo prompts (one per "
             "line). When provided, these are used for t2m demos instead of "
             "the first N training-set prompts.",
    )
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader workers. Defaults to 8 on CUDA, 2 on MPS/CPU.")
    args = p.parse_args()
    if args.num_workers is None:
        args.num_workers = 8 if AUTOCAST_DEVICE == "cuda" else 2
    if not args.encoded_dir and not args.data_dir:
        p.error("one of --data_dir or --encoded_dir is required")
    train(args)


if __name__ == "__main__":
    main()
