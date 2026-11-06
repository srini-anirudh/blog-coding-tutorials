"""FSDP2: shard parameters, gradients and optimizer state across ranks.

DDP keeps a full copy of the model on every rank. FSDP keeps 1/world_size of every wrapped
block. Just before a block runs, its weights are gathered with an all-gather. Just after,
they are freed again. In backward the gradient is reduced back to the local shard.
So per-rank parameter memory falls by world_size, and each block pays extra communication.

Written for NCCL on GPUs, where it is meant to be used. scripts/fsdp_train.py also runs on
gloo on CPU when PyTorch allows it. That demonstrates the shard sizes, not the performance.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from pretrain.model import decoder_layers


def wrap_fsdp(model: nn.Module, precision: str = "fp32") -> nn.Module:
    """Shard every decoder block on its own, then whatever is left at the root."""
    device = next(model.parameters()).device
    # An explicit mesh keeps FSDP from probing for an accelerator it will not use.
    mesh = init_device_mesh(device.type, (dist.get_world_size(),))
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16 if precision == "bf16" else None,
        reduce_dtype=torch.float32 if precision == "bf16" else None,
    )
    for block in decoder_layers(model):
        fully_shard(block, mesh=mesh, mp_policy=policy)  # the unit of gather and free
    return fully_shard(model, mesh=mesh, mp_policy=policy)


def local_param_bytes(model: nn.Module) -> int:
    """Bytes of parameters this rank actually stores. Works for full and sharded tensors."""
    total = 0
    for p in model.parameters():
        data = p.to_local() if hasattr(p, "to_local") else p
        total += data.numel() * data.element_size()
    return total
