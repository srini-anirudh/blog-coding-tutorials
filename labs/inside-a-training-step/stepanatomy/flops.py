"""Count the FLOPs autograd actually issues, with PyTorch's FlopCounterMode.

FlopCounterMode sees every aten op that runs, including the ones the autograd engine launches during
backward. So the forward/backward split the post derives on paper can be read off a real run.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode


def count_flops(fn: Callable[[], object]) -> tuple[int, object]:
    with FlopCounterMode(display=False) as fc:
        out = fn()
    return fc.get_total_flops(), out


def step_flops(forward: Callable[[], torch.Tensor]) -> dict[str, int]:
    """forward() returns a scalar loss. Returns the FLOPs of the forward and of loss.backward()."""
    fwd, loss = count_flops(forward)
    bwd, _ = count_flops(loss.backward)
    return {"forward": fwd, "backward": bwd, "total": fwd + bwd}


def linear_flops(s: int, h_in: int, h_out: int, *, input_grad: bool, weight_grad: bool) -> dict[str, int]:
    """Y = XW for X of shape [S, H] and W of shape [H, H'], with either gradient switched off.

    input_grad=False is the first layer of a network (nothing upstream needs dX).
    weight_grad=False is a frozen layer in the middle of a network.
    """
    torch.manual_seed(0)
    lin = nn.Linear(h_in, h_out, bias=False)
    lin.weight.requires_grad_(weight_grad)
    x = torch.randn(s, h_in, requires_grad=input_grad)
    return step_flops(lambda: lin(x).sum())


def model_flops(model: nn.Module, input_ids: torch.Tensor) -> dict[str, int]:
    return step_flops(lambda: model(input_ids, labels=input_ids).loss)


def matmul_params(model: nn.Module) -> int:
    """Parameters that take part in a GEMM: everything except the embedding table and norm gains."""
    n = 0
    for name, p in model.named_parameters():
        if p.dim() == 2 and "embed_tokens" not in name:
            n += p.numel()
    return n
