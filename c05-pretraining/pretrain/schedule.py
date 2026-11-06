"""Learning-rate schedule: linear warmup, cosine decay, floor at min_lr.

    step < warmup:            lr = peak * (step + 1) / warmup
    warmup <= step < total:   lr = min + (peak - min) * 0.5 * (1 + cos(pi * progress))
    step >= total:            lr = min

Written as a multiplier for LambdaLR, so its state lives in the scheduler's state_dict
and is restored with the rest of the checkpoint.
"""

from __future__ import annotations

import math

import torch


def lr_multiplier(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Multiplier on the peak LR at optimiser step `step` (0-based)."""
    if step < warmup:
        return (step + 1) / warmup
    if step >= total:
        return min_ratio
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def build_scheduler(
    opt: torch.optim.Optimizer, warmup: int, total: int, peak_lr: float, min_lr: float
) -> torch.optim.lr_scheduler.LambdaLR:
    ratio = min_lr / peak_lr
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_multiplier(s, warmup, total, ratio))
