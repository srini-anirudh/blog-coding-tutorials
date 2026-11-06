"""Multi-head latent attention: absorption works, RoPE breaks it, decoupled RoPE restores it.

Per head h, keys come from a cached latent: k_j = W_UK c_j, with c_j in R^{d_c}. The score is
    q_i . k_j = q_i^T W_UK c_j = (W_UK^T q_i) . c_j,
so W_UK can be folded into the query and the per-head keys never exist. With RoPE on the key,
    (R_i q_i) . (R_j W_UK c_j) = q_i^T R_{j-i} W_UK c_j,
and R_{j-i} sits between q and W_UK. It changes with the key's position, so no single fixed matrix
can be folded in. The decoupled form keeps the absorbable content part rotation-free and puts the
position in a separate small key k_pe = RoPE(W_KR h), shared by all heads and cached next to c.
The full module with decoupled RoPE is C3's `attnzoo.mla.MultiHeadLatentAttention`; we reuse it.
"""

from __future__ import annotations

import torch

from .common import use_c03


def rope_matrix(pos: int, dim: int, theta: float = 10_000.0) -> torch.Tensor:
    """The (dim, dim) rotation RoPE applies at one position: 2x2 blocks with angle pos * theta^(-2i/dim)."""
    freqs = theta ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    r = torch.zeros(dim, dim, dtype=torch.float64)
    for i, f in enumerate(freqs):
        c, s = torch.cos(pos * f), torch.sin(pos * f)
        r[2 * i, 2 * i], r[2 * i, 2 * i + 1], r[2 * i + 1, 2 * i], r[2 * i + 1, 2 * i + 1] = c, -s, s, c
    return r


def absorption_check(heads: int = 4, dh: int = 32, d_c: int = 16, n: int = 24, seed: int = 0) -> dict:
    """Explicit keys against absorbed queries, without RoPE and with RoPE on the full key."""
    g = torch.Generator().manual_seed(seed)
    w_uk = torch.randn(heads, dh, d_c, generator=g, dtype=torch.float64) / d_c**0.5
    c = torch.randn(n, d_c, generator=g, dtype=torch.float64)  # cached latents
    q = torch.randn(heads, n, dh, generator=g, dtype=torch.float64)
    rot = torch.stack([rope_matrix(p, dh) for p in range(n)])  # (n, dh, dh)

    k = torch.einsum("hdc,nc->hnd", w_uk, c)  # explicit per-head keys
    explicit = q @ k.transpose(-2, -1)
    absorbed = torch.einsum("hid,hdc->hic", q, w_uk) @ c.T  # W_UK folded into the query
    q_rot = torch.einsum("nde,hne->hnd", rot, q)
    true_rope = q_rot @ torch.einsum("nde,hne->hnd", rot, k).transpose(-2, -1)
    naive = torch.einsum("hid,hdc->hic", q_rot, w_uk) @ c.T  # absorb as if the key had no rotation
    return {
        "no RoPE: max |explicit - absorbed|": (explicit - absorbed).abs().max().item(),
        "RoPE on key: max |true - naive absorbed|": (true_rope - naive).abs().max().item(),
        "RoPE on key: score scale (std of true scores)": true_rope.std().item(),
    }


def best_fixed_matrix_error(dh: int = 32, d_c: int = 16, offsets: int = 128, seed: int = 0) -> float:
    """The best single matrix A to stand in for R_delta W_UK over all offsets 0..offsets-1.

    The least-squares best A is the mean over offsets. Returns its relative error, averaged over offsets.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(dh, d_c, generator=g, dtype=torch.float64)
    mats = torch.stack([rope_matrix(dlt, dh) @ w for dlt in range(offsets)])
    best = mats.mean(0)
    return ((mats - best).flatten(1).norm(dim=-1) / mats.flatten(1).norm(dim=-1)).mean().item()


def decoupled_check(seed: int = 0) -> dict:
    """C3's MLA with decoupled RoPE: absorbed and explicit forwards agree, with and without a cache."""
    use_c03()
    from attnzoo.config import AttentionConfig
    from attnzoo.kv_cache import KVCache
    from attnzoo.mla import MultiHeadLatentAttention

    torch.manual_seed(seed)
    cfg = AttentionConfig(
        "mla", hidden_size=128, num_heads=4, head_dim=32, kv_lora_rank=16, qk_rope_head_dim=8,
        v_head_dim=32, max_seq_len=256, backend="naive",
    )
    mla = MultiHeadLatentAttention(cfg).double().eval()
    x = torch.randn(2, 40, 128, dtype=torch.float64)
    with torch.no_grad():
        explicit, absorbed = mla(x, absorb=False), mla(x, absorb=True)
        cache = KVCache()
        steps = [mla(x[:, :30], kv_cache=cache, absorb=True)] + [mla(x[:, t : t + 1], kv_cache=cache, absorb=True) for t in range(30, 40)]
    return {
        "decoupled RoPE: max |explicit - absorbed|": (explicit - absorbed).abs().max().item(),
        "decoupled RoPE, absorbed and cached: max |cached - full|": (torch.cat(steps, 1) - explicit).abs().max().item(),
        "cached numbers per token (c_kv + k_pe)": cache.keys[0].size(-1) + cache.values[0].size(-1),
    }


def cache_per_token(heads: int, dh: int, kv_heads: int | None = None, d_c: int | None = None, d_r: int = 0) -> int:
    """Numbers cached per token per layer: 2 * Hkv * dh for MHA/GQA/MQA, d_c + d_r for MLA."""
    if d_c is not None:
        return d_c + d_r
    return 2 * (kv_heads or heads) * dh
