"""AdamW with two parameter groups: decay and no-decay.

Weight decay pulls weights toward zero. That is a useful prior for the big matrices
(projections, MLPs, embeddings). It is not useful for RMSNorm gains, whose natural value
is 1, or for biases. So every parameter with fewer than two dimensions goes in a group
with weight_decay = 0. (The derivation of AdamW and decoupled decay is in the theory post.)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pretrain.config import TrainConfig


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():  # tied weights appear once here
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay, "name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "name": "no_decay"},
    ]


def build_optimizer(model: nn.Module, cfg: TrainConfig, device: torch.device) -> torch.optim.AdamW:
    groups = param_groups(model, cfg.weight_decay)
    return torch.optim.AdamW(
        groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        # One fused kernel for all parameters instead of a loop of small ones.
        fused=device.type in ("cuda", "mps"),
    )
