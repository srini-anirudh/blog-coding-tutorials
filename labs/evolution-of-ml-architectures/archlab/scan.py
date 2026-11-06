"""Linear recurrences h_t = a_t * h_{t-1} + b_t, three ways.

1. `sequential_scan`: one step per position. N dependent steps, like an RNN.
2. `parallel_scan`: Hillis-Steele associative scan. ceil(log2 N) dependent steps, each a full-tensor op.
   It works because (a1, b1) then (a2, b2) composes to (a1 a2, a2 b1 + b2), and that is associative.
3. `lti_kernel` + `causal_conv`: when a, B, C do not depend on t, the recurrence is a convolution.

`semiseparable_matrix` is the Mamba-2 view: the same scan written as y = M x with a masked N x N matrix.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sequential_scan(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor | None = None) -> torch.Tensor:
    """a, b: (..., N, ...) with time on dim 1. Returns every h_t, same shape as b."""
    # unbind once: indexing a[:, t] inside the loop would make backward allocate a full-size gradient per step.
    h = torch.zeros_like(b[:, 0]) if h0 is None else h0
    out = []
    for a_t, b_t in zip(a.unbind(1), b.unbind(1)):
        h = a_t * h + b_t
        out.append(h)
    return torch.stack(out, dim=1)


def parallel_scan(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Inclusive scan over dim 1 in ceil(log2 N) rounds. Returns (h, number of rounds)."""
    n = b.shape[1]
    rounds = 0
    shift = 1
    while shift < n:
        # Combine each position with the one `shift` steps earlier. Positions < shift have nothing to combine.
        a_prev = torch.cat([torch.ones_like(a[:, :shift]), a[:, :-shift]], dim=1)
        b_prev = torch.cat([torch.zeros_like(b[:, :shift]), b[:, :-shift]], dim=1)
        b = a * b_prev + b
        a = a * a_prev
        shift *= 2
        rounds += 1
    return b, rounds


def lti_kernel(a_bar: torch.Tensor, b_bar: torch.Tensor, c: torch.Tensor, n: int) -> torch.Tensor:
    """Time-invariant SSM with diagonal A: K_k = sum_s C_s a_s^k B_s, one kernel per channel. Shapes (d, s) -> (d, n)."""
    powers = a_bar[..., None] ** torch.arange(n, dtype=a_bar.dtype)  # (d, s, n)
    return (c[..., None] * powers * b_bar[..., None]).sum(1)


def causal_conv(x: torch.Tensor, kernel: torch.Tensor, fft: bool = False) -> torch.Tensor:
    """y_t = sum_k K_k x_{t-k}. x: (B, N, d), kernel: (d, N). Direct O(N^2) or FFT O(N log N)."""
    n = x.shape[1]
    if fft:
        m = 2 * n
        y = torch.fft.irfft(torch.fft.rfft(x, m, dim=1) * torch.fft.rfft(kernel.T, m, dim=0), m, dim=1)
        return y[:, :n]
    xp = F.pad(x.transpose(1, 2), (n - 1, 0))  # (B, d, 2N-1)
    return F.conv1d(xp, kernel.flip(-1)[:, None], groups=x.shape[2]).transpose(1, 2)


def semiseparable_matrix(a: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Scalar decay per step. a: (N,), B, C: (N, s). M_ij = C_i . B_j * prod_{r=j+1..i} a_r for j <= i, else 0."""
    log_a = torch.log(a)
    cum = torch.cumsum(log_a, 0)
    decay = torch.exp(cum[:, None] - cum[None, :])  # prod_{r=j+1..i} a_r
    mask = torch.ones(len(a), len(a), dtype=torch.bool).tril()
    return torch.where(mask, decay, torch.zeros_like(decay)) * (C @ B.T)


def log2_rounds(n: int) -> int:
    return math.ceil(math.log2(n)) if n > 1 else 0
