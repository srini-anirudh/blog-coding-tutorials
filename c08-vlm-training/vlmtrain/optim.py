"""Differential learning rates: one AdamW parameter group per module (and per decay setting).

A typical Stage 2 setting:

    vision     1e-6    pretrained and fragile; it should barely move
    lm         2e-5    pretrained, needs to adapt to visual tokens
    projector  1e-4    small and new (or freshly aligned); it can move fast
    lora       2e-4    LoRA factors start at zero and need a larger step to matter

The scheduler multiplies every group's own base LR by the same factor, so the ratios between
modules hold for the whole run.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .freeze import MODULES


def _no_decay(name: str, p: nn.Parameter) -> bool:
    # Biases and norm gains are not weights that overfit by growing. Embeddings are excluded too:
    # decoupled decay would shrink rows that receive no gradient (all the old vocabulary).
    return p.ndim < 2 or "norm" in name or "embed" in name


def param_groups(model: nn.Module, lrs: dict[str, float], weight_decay: float = 0.0) -> list[dict]:
    """Group trainable parameters by top-level module. Frozen parameters never reach the optimizer."""
    groups: dict[tuple[str, bool], dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        module = name.split(".")[0]
        if module not in MODULES:
            raise ValueError(f"parameter {name} is outside vision/projector/lm")
        key = "lora" if "lora_" in name and "lora" in lrs else module
        nd = _no_decay(name, p)
        g = groups.setdefault((key, nd), {"name": f"{key}{'/no_decay' if nd else ''}", "params": [],
                                          "lr": lrs[key], "weight_decay": 0.0 if nd else weight_decay})
        g["params"].append(p)
    return list(groups.values())


def build_optimizer(model: nn.Module, lrs: dict[str, float], weight_decay: float = 0.0,
                    betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.AdamW:
    return torch.optim.AdamW(param_groups(model, lrs, weight_decay), betas=betas, eps=1e-8)


def warmup_cosine(warmup: int, total: int, min_ratio: float = 0.1):
    """LR multiplier: linear warmup to 1, cosine decay to min_ratio at `total`."""
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    return f


def describe(opt: torch.optim.Optimizer) -> str:
    lines = [f"{'group':<22}{'tensors':>8}{'params':>14}{'lr':>10}{'wd':>6}"]
    for g in opt.param_groups:
        n = sum(p.numel() for p in g["params"])
        lines.append(f"{g['name']:<22}{len(g['params']):>8}{n:>14,}{g['lr']:>10.1e}{g['weight_decay']:>6}")
    return "\n".join(lines)
