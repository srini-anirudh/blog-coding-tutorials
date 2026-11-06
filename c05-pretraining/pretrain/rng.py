"""Capture and restore every random number generator the run touches.

Python's `random`, NumPy, the torch CPU generator, and the accelerator generator are
separate streams. Restoring only one of them is the most common reason a "resumed" run
quietly differs from the original.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def rng_state() -> dict:
    kind, keys, pos, has_gauss, gauss = np.random.get_state()
    state = {
        "python": random.getstate(),
        # Stored as a tensor so the checkpoint loads with torch.load(weights_only=True).
        "numpy": (kind, torch.from_numpy(keys.astype(np.int64)), pos, has_gauss, gauss),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    kind, keys, pos, has_gauss, gauss = state["numpy"]
    np.random.set_state((kind, keys.numpy().astype(np.uint32), pos, has_gauss, gauss))
    torch.set_rng_state(state["torch_cpu"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])
