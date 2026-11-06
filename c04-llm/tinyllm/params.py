"""Parameter and KV-cache size from the config alone, with no model instantiated."""

from __future__ import annotations

from .config import ModelConfig


def count_parameters(c: ModelConfig) -> dict[str, int]:
    """Closed-form parameter count, split by component.

    With D = hidden_size, F = intermediate_size, V = vocab_size, L = num_layers,
    H = num_heads, H_kv = num_kv_heads, d = head_dim:
      embeddings  V*D
      attention   L * (D*H*d  [q]  + 2*D*H_kv*d  [k, v]  + H*d*D  [o])
      mlp         L * 3*D*F, or L * (E * 3*D*F + D*E) with E experts and a router
      norms       L * 2*D + D  (two per block, one final)
      lm_head     V*D if untied, 0 if tied to the embeddings
    """
    d_model, d = c.hidden_size, c.head_dim
    attention = c.num_layers * (d_model * c.num_heads * d + 2 * d_model * c.num_kv_heads * d + c.num_heads * d * d_model)
    ffn = 3 * d_model * c.intermediate_size
    if c.num_experts:
        mlp = c.num_layers * (c.num_experts * ffn + d_model * c.num_experts)
    else:
        mlp = c.num_layers * ffn
    counts = {
        "embeddings": c.vocab_size * d_model,
        "attention": attention,
        "mlp": mlp,
        "norms": c.num_layers * 2 * d_model + d_model,
        "lm_head": 0 if c.tie_word_embeddings else c.vocab_size * d_model,
    }
    counts["total"] = sum(counts.values())
    return counts


def active_parameters(c: ModelConfig) -> int:
    """Parameters touched by one token. Equal to the total for a dense model."""
    total = count_parameters(c)["total"]
    if not c.num_experts:
        return total
    unused_experts = c.num_experts - c.num_experts_per_tok
    return total - c.num_layers * unused_experts * 3 * c.hidden_size * c.intermediate_size


def kv_cache_bytes(c: ModelConfig, seq_len: int, batch_size: int = 1, bytes_per_value: int = 2) -> int:
    """K and V, for every layer, every kv head, every position: 2 * L * H_kv * d * T * B values."""
    return 2 * c.num_layers * c.num_kv_heads * c.head_dim * seq_len * batch_size * bytes_per_value
