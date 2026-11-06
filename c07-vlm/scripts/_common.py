"""Shared setup for the step scripts: make `tinyvlm` importable and keep the laptop responsive."""

import sys
from pathlib import Path

import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)
torch.manual_seed(0)  # a fresh projector starts from the same weights on every run
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()


def stats(name: str, tensor: torch.Tensor) -> str:
    """One line: shape, mean, std and range of a tensor."""
    t = tensor.detach().float()
    return (f"{name:<22} {str(tuple(t.shape)):<20} mean {t.mean():+.3f}  std {t.std():.3f}  "
            f"min {t.min():+.3f}  max {t.max():+.3f}")
