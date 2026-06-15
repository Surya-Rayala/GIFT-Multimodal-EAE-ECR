"""Best-effort release of Python + torch allocator memory.

Dropping the last Python reference to a model frees its tensors, but torch's
caching allocators (CUDA, MPS) keep the freed blocks reserved for reuse — the
memory stays unavailable to the rest of the process and the OS until
``empty_cache()`` runs. Call :func:`free_torch_memory` after a model's last
use (ASR/alignment/denoiser models, the vision backend) so the next phase of
the pipeline starts from a clean allocator.

Kept dependency-free at module level: importing this never pulls in torch.
"""
from __future__ import annotations

import gc
import logging
from typing import Optional


def free_torch_memory(device: Optional[str] = None) -> None:
    """Run ``gc.collect()`` and release torch's cached allocator blocks.

    ``device`` is the device string the freed model lived on ("cuda",
    "cuda:0", "mps", "cpu", or None). CPU needs no cache release; unknown /
    missing torch installs are tolerated silently — this is a best-effort
    cleanup that must never break the pipeline.
    """
    gc.collect()
    if not device:
        return
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        logging.debug("free_torch_memory: cache release skipped", exc_info=True)
