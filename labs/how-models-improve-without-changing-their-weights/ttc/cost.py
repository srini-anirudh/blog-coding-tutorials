"""The bill for K trajectories of T tokens versus one trajectory of K*T tokens.

Counts for a dense decoder. Weight FLOPs are 2N per token. Attention FLOPs for the token at
position t are about 4 * n_layers * d_model * t (scores Q K^T plus the weighted sum over V).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Bill:
    weight_flops: float
    attention_flops: float
    decode_steps: int
    kv_slots: int
    kv_bytes: float


def decode_bill(n_params: float, n_layers: int, d_model: int, kv_dim: int, width: int, depth: int,
                bytes_per_value: int = 2) -> Bill:
    """Cost of `width` independent trajectories, each `depth` generated tokens (prompt ignored).

    kv_dim is n_kv_heads * head_dim. Each cached token stores one key and one value per layer.
    """
    per_traj_attn = 4.0 * n_layers * d_model * depth * (depth + 1) / 2.0
    slots = width * depth
    return Bill(
        weight_flops=2.0 * n_params * width * depth,
        attention_flops=width * per_traj_attn,
        decode_steps=depth,
        kv_slots=slots,
        kv_bytes=2.0 * n_layers * kv_dim * bytes_per_value * slots,
    )


def wide_vs_deep(n_params, n_layers, d_model, kv_dim, K: int, T: int) -> dict:
    """Compare K x T (wide) with 1 x KT (deep). Returns both bills and the deep/wide ratios."""
    wide = decode_bill(n_params, n_layers, d_model, kv_dim, K, T)
    deep = decode_bill(n_params, n_layers, d_model, kv_dim, 1, K * T)
    ratio = {f: getattr(deep, f) / getattr(wide, f) for f in wide.__dataclass_fields__}
    return {"wide": wide, "deep": deep, "deep_over_wide": ratio}
