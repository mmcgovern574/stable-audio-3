"""Device-detection helpers so training works on CUDA, MPS, or CPU.

Why this file exists
--------------------
SA3 training was written for CUDA — `torch.amp.autocast("cuda", ...)` and
`torch.cuda.empty_cache()` are sprinkled across the codebase. The first
raises (or silently runs on the wrong device) on Mac MPS; the second always
raises on MPS. This module centralizes the cuda/mps/cpu fork into one place.
"""

from __future__ import annotations

import torch


def get_accelerator_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' for the current machine."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Autocast decorators are evaluated at import time, so they need a constant
# they can read without a function call.
AUTOCAST_DEVICE: str = get_accelerator_device()


def safe_empty_cache() -> None:
    """Free cached accelerator memory. No-op on CPU; dispatches correctly on MPS."""
    if AUTOCAST_DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif AUTOCAST_DEVICE == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
