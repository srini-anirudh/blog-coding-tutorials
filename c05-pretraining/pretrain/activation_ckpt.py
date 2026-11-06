"""Activation checkpointing: keep block inputs, recompute block internals in backward.

Without it, autograd keeps every intermediate tensor of every block alive until the
backward pass reaches it. With it, each block keeps only its input; during backward
the block's forward runs again to rebuild what it needs. Memory drops, compute rises
by roughly one extra forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from pretrain.model import decoder_layers


class CheckpointWrapper(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        if not torch.is_grad_enabled():
            return self.block(*args, **kwargs)
        # use_reentrant=False: the supported mode. It handles keyword arguments and
        # inputs that do not require grad, and it nests with other autograd hooks.
        return checkpoint(self.block, *args, use_reentrant=False, **kwargs)


def apply_activation_checkpointing(model: nn.Module) -> nn.Module:
    layers = decoder_layers(model)
    for i, block in enumerate(layers):
        if not isinstance(block, CheckpointWrapper):
            layers[i] = CheckpointWrapper(block)
    return model


def saved_tensor_bytes(fn) -> int:
    """Bytes of tensors autograd saves for backward while running fn().

    Uses saved_tensors_hooks: every tensor autograd stores passes through `pack`.
    Storages are de-duplicated, because several saved tensors can share memory.
    """
    seen: dict[int, int] = {}

    def pack(t: torch.Tensor):
        s = t.untyped_storage()
        seen[s.data_ptr()] = s.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return sum(seen.values())
