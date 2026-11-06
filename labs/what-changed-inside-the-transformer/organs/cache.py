"""The KV cache and the mask: a sliding-window cache that stops growing, MLA's absorption and why RoPE breaks
it, and how far a stack of local and global layers can see.
"""

from __future__ import annotations

import math

import torch

from . import _paths  # noqa: F401
from attnzoo.masks import causal_mask, sliding_window_mask
from tinyllm.rope import RotaryEmbedding, rotate


def masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """q, k, v: (T, d). mask: (T, T) bool, True = may attend."""
    s = (q @ k.T / math.sqrt(q.shape[-1])).masked_fill(~mask, float("-inf"))
    return s.softmax(-1) @ v


class RollingKVCache:
    """A sliding-window layer's cache: a ring buffer of `window` slots. Memory is min(t, window), never t."""

    def __init__(self, window: int, dim: int, dtype: torch.dtype = torch.float32) -> None:
        self.window, self.t = window, 0
        self.k = torch.zeros(window, dim, dtype=dtype)
        self.v = torch.zeros(window, dim, dtype=dtype)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slot = self.t % self.window  # overwrite the oldest entry
        self.k[slot], self.v[slot] = k, v
        self.t += 1
        n = min(self.t, self.window)
        return self.k[:n], self.v[:n]  # order does not matter to softmax attention

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() + self.v.numel() * self.v.element_size()


def decode_with_window(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> tuple[torch.Tensor, int]:
    """Token-by-token decoding through a RollingKVCache. Returns the outputs (T, d) and the cache's peak bytes."""
    cache = RollingKVCache(window, k.shape[-1], k.dtype)
    outs = []
    for t in range(q.shape[0]):
        ks, vs = cache.append(k[t], v[t])
        p = (ks @ q[t] / math.sqrt(q.shape[-1])).softmax(-1)
        outs.append(p @ vs)
    return torch.stack(outs), cache.nbytes()


def full_window_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
    return masked_attention(q, k, v, sliding_window_mask(q.shape[0], window))


def mla_absorption_check(d: int = 64, r: int = 16, dh: int = 32, seed: int = 0) -> dict[str, float]:
    """Logit q^T k for k = W_UK c, computed three ways, and what happens when RoPE rotates k after W_UK.

    absorbed:          (W_UK^T q)^T c, without ever building k. Exact.
    rope_materialized: q_m^T R_n W_UK c, the true logit when the key is rotated at position n.
    rope_absorbed:     the absorbed matrix built for one position (n=5), reused at another (n=9). Wrong.
    """
    g = torch.Generator().manual_seed(seed)
    h, wq = torch.randn(d, generator=g, dtype=torch.float64), torch.randn(dh, d, generator=g, dtype=torch.float64)
    w_dkv = torch.randn(r, d, generator=g, dtype=torch.float64)
    w_uk = torch.randn(dh, r, generator=g, dtype=torch.float64)
    q, c = wq @ h, w_dkv @ h
    out = {"materialized": (q @ (w_uk @ c)).item(), "absorbed": ((w_uk.T @ q) @ c).item()}

    rope = RotaryEmbedding(dh, 16)
    cos, sin = rope(torch.arange(16)[None])

    def rot(x: torch.Tensor, pos: int) -> torch.Tensor:
        return rotate(x.view(1, 1, 1, -1), cos[:, pos : pos + 1].double(), sin[:, pos : pos + 1].double()).flatten()

    m, n_built, n_used = 12, 5, 9
    rotated_wuk = lambda n: torch.stack([rot(w_uk[:, j], n) for j in range(r)], 1)  # R_n W_UK, (dh, r)
    out["rope_materialized"] = (rot(q, m) @ (rotated_wuk(n_used) @ c)).item()
    out["rope_absorbed_wrong_pos"] = ((rotated_wuk(n_built).T @ rot(q, m)) @ c).item()
    out["commutator_norm"] = (rotated_wuk(n_used) - w_uk).norm().item() / w_uk.norm().item()
    return out


def receptive_field(layer_kinds: list[str], seq: int, window: int) -> list[int]:
    """How far back the last token can see after each layer. kinds: "local" or "global".

    Reachability composes like a matrix product: after layers 1..l, token i depends on token j
    iff there is a path through the masks. Returns, per layer, the distance to the farthest visible token.
    """
    reach = torch.eye(seq, dtype=torch.float64)
    out = []
    for kind in layer_kinds:
        mask = causal_mask(seq) if kind == "global" else sliding_window_mask(seq, window)
        reach = ((mask.double() @ reach) > 0).double()
        seen = torch.nonzero(reach[-1]).flatten()
        out.append(int(seq - 1 - seen.min()))
    return out


def interleave(layers: int, local_per_global: int | None) -> list[str]:
    """None = all local. 0 = all global. k = k local layers, then one global, repeated."""
    if local_per_global is None:
        return ["local"] * layers
    if local_per_global == 0:
        return ["global"] * layers
    return ["global" if (i + 1) % (local_per_global + 1) == 0 else "local" for i in range(layers)]
