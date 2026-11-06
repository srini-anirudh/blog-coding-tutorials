"""Linear attention: replace exp(q.k) with phi(q).phi(k) and reorder the matmuls.

softmax(Q K^T) V costs O(N^2 d). With a feature map phi,
    phi(Q) [phi(K)^T V]
costs O(N d^2): the (d, dv) matrix phi(K)^T V is built once and shared by every query.
In the causal case that matrix becomes a running state, which is an RNN:
    S_t = S_{t-1} + phi(k_t) v_t^T        z_t = z_{t-1} + phi(k_t)
    y_t = phi(q_t) S_t / (phi(q_t) . z_t)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import AttentionConfig
from .heads import merge_heads, split_heads


def elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    """phi(x) = elu(x) + 1 > 0 (Katharopoulos et al., 2020). Positive, so the normaliser never hits zero."""
    return F.elu(x) + 1.0


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Non-causal. q, k: (B, H, N, d) already feature-mapped; v: (B, H, N, dv) -> (B, H, N, dv)."""
    kv = k.transpose(-2, -1) @ v  # (B, H, d, dv): the whole sequence summarised once
    z = k.sum(dim=-2)  # (B, H, d): normaliser
    num = q @ kv  # (B, H, N, dv)
    den = (q * z[:, :, None]).sum(-1, keepdim=True)  # (B, H, N, 1)
    return num / (den + eps)


def causal_linear_attention_recurrent(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None, eps: float = 1e-6
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Causal, one step at a time. Returns outputs (B, H, N, dv) and the final state (S, z).

    S: (B, H, d, dv)   z: (B, H, d). Constant size, however long the sequence gets.
    """
    b, h, n, d = q.shape
    if state is None:
        s = q.new_zeros(b, h, d, v.size(-1))
        z = q.new_zeros(b, h, d)
    else:
        s, z = state
    outs = []
    for t in range(n):
        k_t, v_t, q_t = k[:, :, t], v[:, :, t], q[:, :, t]  # (B, H, d), (B, H, dv), (B, H, d)
        s = s + k_t[..., :, None] * v_t[..., None, :]  # outer product phi(k_t) v_t^T
        z = z + k_t
        num = (q_t[..., None, :] @ s).squeeze(-2)  # (B, H, dv)
        den = (q_t * z).sum(-1, keepdim=True)
        outs.append(num / (den + eps))
    return torch.stack(outs, dim=2), (s, z)


def causal_linear_attention_parallel(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Causal, all positions at once, by masking phi(Q) phi(K)^T. O(N^2) again, but parallel: good for training."""
    n = q.size(-2)
    a = (q @ k.transpose(-2, -1)) * torch.ones(n, n, device=q.device, dtype=q.dtype).tril()  # (B, H, N, N), no softmax
    return (a @ v) / (a.sum(-1, keepdim=True) + eps)


def performer_features(x: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    """FAVOR+ positive random features (Choromanski et al., 2021). Advanced, for illustration.

    x: (..., N, d), projection: (m, d) with rows ~ N(0, I). Returns (..., N, m) with
    E[phi(q) . phi(k)] = exp(q . k / sqrt(d)). The approximation improves as m grows.
    """
    d = x.size(-1)
    x = x / d**0.25  # split the 1/sqrt(d) temperature between q and k
    wx = x @ projection.T  # (..., N, m)
    sq = (x**2).sum(-1, keepdim=True) / 2
    # Subtracting the global max is a constant factor that cancels in the normaliser; it prevents overflow.
    return torch.exp(wx - sq - wx.amax().detach()) / math.sqrt(projection.size(0))


class LinearStateCache:
    """Per-layer recurrent state (S, z) for decoding. Its size does not grow with the sequence."""

    def __init__(self) -> None:
        self.states: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, layer_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self.states.get(layer_idx)

    def set(self, state: tuple[torch.Tensor, torch.Tensor], layer_idx: int = 0) -> None:
        self.states[layer_idx] = state

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for s in self.states.values() for t in s)


class LinearAttention(nn.Module):
    """A drop-in attention layer using the elu+1 feature map. hidden (B, N, D) -> (B, N, D).

    Training uses the parallel form; decoding with a LinearStateCache uses the recurrence.
    """

    def __init__(self, config: AttentionConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        d, h, dh = config.hidden_size, config.num_heads, config.head_dim
        self.q_proj = nn.Linear(d, h * dh, bias=False)
        self.k_proj = nn.Linear(d, h * dh, bias=False)
        self.v_proj = nn.Linear(d, h * dh, bias=False)
        self.o_proj = nn.Linear(h * dh, d, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, position_ids=None, kv_cache: LinearStateCache | None = None) -> torch.Tensor:
        h = self.config.num_heads
        q = elu_feature_map(split_heads(self.q_proj(hidden_states), h))
        k = elu_feature_map(split_heads(self.k_proj(hidden_states), h))
        v = split_heads(self.v_proj(hidden_states), h)
        if not self.config.causal:
            out = linear_attention(q, k, v)
        elif kv_cache is not None:
            out, state = causal_linear_attention_recurrent(q, k, v, kv_cache.get(self.layer_idx))
            kv_cache.set(state, self.layer_idx)
        else:
            out = causal_linear_attention_parallel(q, k, v)
        return self.o_proj(merge_heads(out))
