"""What a picture costs at serving time: visual tokens are KV cache.

    KV bytes per token = 2 (K and V) * layers * kv_heads * head_dim * bytes_per_value

`measure_kv_bytes` checks the formula against a real Hugging Face Llama cache. The model has the
post's attention shape (32 layers, 8 KV heads, head dim 128) but a tiny hidden size and FFN, so it
fits in memory on a laptop. Only the attention shape decides the cache size.
"""

from __future__ import annotations

import torch


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bytes_per_value: int = 2) -> int:
    return 2 * layers * kv_heads * head_dim * bytes_per_value


def picture_cost(tokens: int, layers: int = 32, kv_heads: int = 8, head_dim: int = 128,
                 bytes_per_value: int = 2, context: int = 128_000) -> dict:
    """The three cards of the post's fifth explainer, for one visual prefix."""
    per = kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value)
    return {
        "visual_tokens": tokens,
        "kv_per_token_kib": per / 1024,
        "kv_mib": tokens * per / 2**20,
        "approx_words": round(tokens * 0.75),   # the post's rule of thumb, 0.75 words per token
        "images_in_context": context // tokens,
    }


def cache_nbytes(cache) -> int:
    """Total bytes held by a transformers Cache (or the legacy tuple-of-tuples format)."""
    total = 0
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            for name in ("keys", "values"):
                t = getattr(layer, name, None)
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
        return total
    for k, v in cache:
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


@torch.no_grad()
def measure_kv_bytes(tokens: int, layers: int = 32, kv_heads: int = 8, head_dim: int = 128,
                     dtype=torch.bfloat16) -> int:
    """Run a prefix of `tokens` positions through a Llama with this attention shape and return the
    bytes in its KV cache. Hidden size is small; the cache does not depend on it."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=kv_heads, num_key_value_heads=kv_heads, head_dim=head_dim,
                      max_position_embeddings=max(tokens, 64))
    model = LlamaForCausalLM(cfg).to(dtype).eval()
    out = model(torch.zeros(1, tokens, dtype=torch.long), use_cache=True)
    return cache_nbytes(out.past_key_values)
