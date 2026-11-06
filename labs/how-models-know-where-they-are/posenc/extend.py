"""Inference-time context extension for RoPE. Each method returns new rotation speeds and a logit scale.

pi    Position Interpolation (Chen et al., 2023): divide every position by s.
ntk   NTK-aware scaling (bloc97, 2023): raise the base to base * s^(d / (d - 2)).
yarn  YaRN (Peng et al., 2023): per plane, interpolate slow planes, keep fast planes, ramp in between,
      and scale the logits by (0.1 ln s + 1)^2.

The YaRN ramp follows the reference code (and Hugging Face): it is linear in the plane index between
the planes that make beta_fast and beta_slow full turns inside the original context.
"""

from __future__ import annotations

import math

import torch

from .rope import inv_freq


def scaled_inv_freq(
    method: str, head_dim: int, base: float, factor: float, original_len: int,
    beta_fast: float = 32.0, beta_slow: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Returns (inv_freq (d/2,), logit_scale). logit_scale multiplies q.k before the softmax."""
    theta = inv_freq(head_dim, base)
    if method == "none" or factor == 1.0:
        return theta, 1.0
    if method == "pi":
        return theta / factor, 1.0
    if method == "ntk":
        return inv_freq(head_dim, base * factor ** (head_dim / (head_dim - 2))), 1.0
    if method == "yarn":
        def plane_for(turns: float) -> float:
            # The (fractional) plane index whose wavelength fits `turns` times into original_len.
            return head_dim * math.log(original_len / (turns * 2 * math.pi)) / (2 * math.log(base))

        lo = max(math.floor(plane_for(beta_fast)), 0)
        hi = min(math.ceil(plane_for(beta_slow)), head_dim - 1)
        if lo == hi:
            hi += 0.001
        ramp = ((torch.arange(head_dim // 2, dtype=torch.float32) - lo) / (hi - lo)).clamp(0, 1)
        keep = 1 - ramp  # 1 = extrapolate (fast planes), 0 = interpolate (slow planes)
        new = theta / factor * (1 - keep) + theta * keep
        mscale = 0.1 * math.log(factor) + 1.0
        return new, mscale**2
    raise ValueError(f"unknown method {method!r}")


def out_of_range_planes(freqs: torch.Tensor, train_freqs: torch.Tensor, train_len: int, eval_len: int) -> int:
    """Planes whose largest angle at eval_len exceeds anything seen in training.

    A plane that already completed a full turn in training has seen every angle, so it never counts.
    """
    seen = train_freqs * train_len
    reached = freqs * eval_len
    return int(((reached > seen * (1 + 1e-6)) & (seen < 2 * math.pi)).sum())
