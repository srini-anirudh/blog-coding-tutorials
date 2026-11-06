"""Weight initialisation: small normal weights, with residual output projections scaled by depth."""

from __future__ import annotations

import math

from torch import nn

from .config import ModelConfig
from .norm import RMSNorm

# Layers whose output is added straight onto the residual stream.
RESIDUAL_PROJECTIONS = ("o_proj", "down_proj")


def init_weights(model: nn.Module, config: ModelConfig) -> None:
    """GPT-2 / Llama-style init.

    Every Linear and Embedding weight ~ N(0, std) with std = initializer_range (0.02).
    o_proj and down_proj use std / sqrt(2 * num_layers): each block adds two branches to the
    residual stream, so 2L random contributions accumulate. Scaling each by 1/sqrt(2L) keeps
    the variance of the final residual stream about the same at any depth.
    RMSNorm gains start at 1, so every norm begins as a pure rescale.
    """
    std = config.initializer_range
    residual_std = std / math.sqrt(2 * config.num_layers)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            is_residual = name.split(".")[-1] in RESIDUAL_PROJECTIONS
            nn.init.normal_(module.weight, mean=0.0, std=residual_std if is_residual else std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
