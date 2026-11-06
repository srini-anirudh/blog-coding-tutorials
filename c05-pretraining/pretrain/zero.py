"""ZeRO: stop keeping redundant copies of optimizer state, then of gradients and weights.

    Stage 1  optimizer state sharded       (torch ZeroRedundancyOptimizer, runs on gloo here)
    Stage 2  also gradients sharded        (DeepSpeed, GPU only)
    Stage 3  also parameters sharded       (DeepSpeed, GPU only; the same idea as FSDP)

AdamW keeps two moment tensors per weight. So optimizer state is twice the model, and in DDP
every rank stores all of it even though only one rank needs to update each parameter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer


def wrap_zero1(model: nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.95)) -> ZeroRedundancyOptimizer:
    """Each parameter is owned by one rank. Only the owner keeps moments and computes the update.

    The owner then broadcasts the new weight to the others.
    """
    return ZeroRedundancyOptimizer(
        model.parameters(), optimizer_class=torch.optim.AdamW,
        lr=lr, weight_decay=weight_decay, betas=betas,
    )


def optimizer_state_bytes(opt: torch.optim.Optimizer) -> int:
    """Bytes of optimizer tensors held on this rank."""
    inner = getattr(opt, "optim", opt)  # ZeroRedundancyOptimizer keeps its local AdamW in .optim
    total = 0
    for state in inner.state.values():
        for v in state.values():
            if torch.is_tensor(v) and v.dim() > 0:
                local = v.to_local() if hasattr(v, "to_local") else v  # DTensor under FSDP
                total += local.numel() * local.element_size()
    return total


def deepspeed_config(stage: int, micro_batch: int, accum: int, lr: float, grad_clip: float) -> dict:
    """A DeepSpeed config for the same run. Stage 3 also needs a sharded checkpoint on save."""
    return {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": accum,
        "gradient_clipping": grad_clip,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": stage},
        "optimizer": {"type": "AdamW", "params": {"lr": lr, "betas": [0.9, 0.95], "weight_decay": 0.1}},
    }
