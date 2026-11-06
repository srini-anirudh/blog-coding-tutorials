"""Time an Adam step and turn the time into bytes per second.

The useful traffic of FP32 AdamW is 28 bytes per parameter: read the gradient, weight, m and v
(16 bytes) and write the weight, m and v back (12 bytes). The post's mixed recipe also comes to 28
(see accounting.ADAM_BYTES_PER_PARAM). About 12 FLOPs per parameter against 28 bytes is 0.43 FLOP/byte.
"""

from __future__ import annotations

import time

import torch

from .mixed import MasterWeightAdam

FP32_ADAM_BYTES_PER_PARAM = 4 * 4 + 3 * 4


def layer_like_tensors(n_params: float, hidden: int, device: str, dtype=torch.float32) -> list[torch.Tensor]:
    """Weight-shaped tensors adding up to about n_params: per block, 4 of H x H and 3 of H x 2.69H, and norms."""
    inter = int(hidden * 11008 / 4096)
    per_block = 4 * hidden * hidden + 3 * hidden * inter + 2 * hidden
    blocks = max(1, round(n_params / per_block))
    shapes = []
    for _ in range(blocks):
        shapes += [(hidden, hidden)] * 4 + [(inter, hidden)] * 3 + [(hidden,)] * 2
    return [torch.randn(s, device=device, dtype=dtype) * 0.02 for s in shapes]


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_adam(variant: str, n_params: float, hidden: int, device: str, iters: int = 5) -> dict[str, float]:
    """variant: 'for-loop', 'foreach', 'fused' (torch.optim.AdamW on FP32) or 'master' (MasterWeightAdam on BF16)."""
    if variant == "master":
        params = [p.requires_grad_() for p in layer_like_tensors(n_params, hidden, device, torch.bfloat16)]
    else:
        params = [p.requires_grad_() for p in layer_like_tensors(n_params, hidden, device)]
    for p in params:
        p.grad = torch.randn_like(p) * 1e-3
    n = sum(p.numel() for p in params)
    if variant == "master":
        opt = MasterWeightAdam(params, lr=1e-4)
    else:
        kw = {"for-loop": {"foreach": False}, "foreach": {"foreach": True}, "fused": {"fused": True}}[variant]
        opt = torch.optim.AdamW(params, lr=1e-4, **kw)
    opt.step()  # first step allocates m and v; do not time it
    times = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        opt.step()
        _sync(device)
        times.append(time.perf_counter() - t0)
    t = sorted(times)[len(times) // 2]
    return {"params": n, "tensors": len(params), "ms": t * 1e3,
            "useful GB/s": n * FP32_ADAM_BYTES_PER_PARAM / t / 1e9}
