"""Model FLOPs utilisation: useful training FLOPs per second divided by the hardware peak.

Per token, a decoder with N matmul parameters, n_layers layers, width d and context T does

    forward:   2N  (one multiply-add per weight)  +  4 * n_layers * d * T  (QK^T and AV)
    backward:  twice the forward (gradients w.r.t. activations and w.r.t. weights)
    total:     6N + 12 * n_layers * d * T

N counts every weight a matmul touches: the blocks plus the output projection to the
vocabulary. The input embedding is a lookup, not a matmul, so it is left out (with tied
embeddings the same matrix still costs 2*V*d per token in the output projection).
Recomputation from activation checkpointing is not useful work, so MFU does not count it.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from pretrain.config import ModelArgs
from pretrain.model import count_params
from pretrain.utils import synchronize


def matmul_params(model: nn.Module, a: ModelArgs) -> int:
    """Block parameters plus the V x d output projection."""
    return count_params(model, non_embedding=True) + a.vocab_size * a.hidden_size


def flops_per_token(model: nn.Module, a: ModelArgs, seq_len: int) -> float:
    return 6 * matmul_params(model, a) + 12 * a.num_layers * a.hidden_size * seq_len


@torch.no_grad()
def measure_matmul_peak(device: torch.device, dtype: torch.dtype, n: int = 4096, iters: int = 20) -> float:
    """Best FLOP/s of a large square matmul on this device.

    Vendors publish a peak for datacenter GPUs; use that number there. Elsewhere we
    measure what the hardware actually reaches on the friendliest possible workload,
    and report MFU against that. It is an upper bound we can defend, not a datasheet value.
    """
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    for _ in range(3):
        a @ b
    synchronize(device)
    best = 0.0
    for _ in range(iters):
        t = time.perf_counter()
        a @ b
        synchronize(device)
        best = max(best, 2 * n**3 / (time.perf_counter() - t))
    return best


def mfu(tokens_per_sec: float, flops_per_tok: float, peak_flops: float) -> float:
    return tokens_per_sec * flops_per_tok / peak_flops
