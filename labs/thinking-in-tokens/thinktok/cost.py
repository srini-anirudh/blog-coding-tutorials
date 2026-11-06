"""The bill for K thinking tokens (section "The bill").

  C_gen ~ 2 N K + 4 L d (P K + K^2 / 2)      FLOPs       weights term + attention term
  M_KV  = 2 L n_kv d_head b K                 bytes       one K and one V per layer per kv head
  t_tok >= bytes_of_weights / HBM_bandwidth   seconds     batch-1 decode streams every weight once

N counts every weight a token is multiplied by (attention, MLP and the LM head; the embedding lookup is
not a matmul). d in the attention term is the width of the attended vectors, num_heads * head_dim.
`count_decode_flops` checks the formula against torch's FLOP counter on a real (small) decoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import _paths  # noqa: F401
from tinyllm import KVCache, ModelConfig, TinyLLM

# The post's "32B-class dense model": 64 layers, 8 KV heads of width 128. These are Qwen2.5-32B's published
# config values (hidden 5120, 40 query heads, MLP 27648, vocab 152064, untied embeddings).
DENSE_32B = ModelConfig(
    vocab_size=152_064, hidden_size=5120, num_layers=64, num_heads=40, num_kv_heads=8,
    intermediate_size=27_648, max_seq_len=131_072, tie_word_embeddings=False,
)
H100_SXM_BANDWIDTH = 3.35e12  # bytes/s, NVIDIA's H100 SXM5 spec; the post rounds it to 3.3 TB/s


def matmul_params(c: ModelConfig) -> int:
    """Weights each generated token is multiplied by: attention + MLP + LM head (dense models)."""
    d, hd = c.hidden_size, c.head_dim
    attn = c.num_layers * (d * c.num_heads * hd + 2 * d * c.num_kv_heads * hd + c.num_heads * hd * d)
    mlp = c.num_layers * 3 * d * c.intermediate_size
    return attn + mlp + c.vocab_size * d


def non_embedding_params(c: ModelConfig) -> int:
    """Attention + MLP weights. This is the N of the post's formula; it excludes the LM head."""
    return matmul_params(c) - c.vocab_size * c.hidden_size


@dataclass
class FlopBill:
    weights: float
    attention: float

    @property
    def total(self) -> float:
        return self.weights + self.attention


def gen_flops(n_params: float, layers: int, d_attn: int, prompt: int, k: int, exact: bool = False) -> FlopBill:
    """FLOPs to generate k tokens after a prompt of `prompt` tokens (prefill not included).

    The token written at step t (0-based) attends over prompt + t + 1 keys. Summed, that is
    P K + K (K + 1) / 2 query-key pairs, which the post writes as P K + K^2 / 2.
    """
    pairs = prompt * k + (k * (k + 1) / 2 if exact else k * k / 2)
    return FlopBill(weights=2.0 * n_params * k, attention=4.0 * layers * d_attn * pairs)


def per_token_attention_share(c: ModelConfig, context: int, n_params: float | None = None) -> float:
    """Attention FLOPs / weight FLOPs for one token written at the given context length."""
    n = non_embedding_params(c) if n_params is None else n_params
    return 4 * c.num_layers * c.num_heads * c.head_dim * context / (2 * n)


def kv_bytes_per_token(layers: int, n_kv: int, d_head: int, bytes_per_value: int = 2) -> int:
    return 2 * layers * n_kv * d_head * bytes_per_value


def decode_floor_seconds(n_weights: float, bytes_per_weight: int = 2, bandwidth: float = H100_SXM_BANDWIDTH) -> float:
    """Lower bound on one batch-1 decode step: every weight crosses HBM once."""
    return n_weights * bytes_per_weight / bandwidth


def bits_per_step(vocab: int) -> float:
    """Capacity of the token channel between two reasoning steps: log2 V."""
    return math.log2(vocab)


@torch.no_grad()
def count_decode_flops(model: TinyLLM, prompt: int, k: int, seed: int = 0) -> dict:
    """Prefill `prompt` random tokens, then count the FLOPs of k cached decode steps with torch's counter.

    Returns the measured count, the KV-cache bytes held afterwards, and both formula values.
    The model must use attn_backend="naive" so every attention product is a counted matmul.
    """
    from torch.utils.flop_counter import FlopCounterMode

    c = model.config
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, c.vocab_size, (1, prompt), generator=g)
    model.eval()
    cache = KVCache()
    out = model(ids, past_key_values=cache, use_cache=True)
    nxt = out.logits[:, -1:].argmax(-1)
    counter = FlopCounterMode(display=False)
    with counter:
        for _ in range(k):
            out = model(nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1:].argmax(-1)
    measured = counter.get_total_flops()
    d_attn = c.num_heads * c.head_dim
    exact = gen_flops(matmul_params(c), c.num_layers, d_attn, prompt, k, exact=True)
    approx = gen_flops(matmul_params(c), c.num_layers, d_attn, prompt, k, exact=False)
    bytes_per_value = next(model.parameters()).element_size()
    return {
        "prompt": prompt,
        "k": k,
        "measured_flops": measured,
        "formula_exact": exact.total,
        "formula_post": approx.total,
        "weights_term": exact.weights,
        "attention_term": exact.attention,
        "kv_bytes_measured": cache.nbytes(),
        "kv_bytes_formula": kv_bytes_per_token(c.num_layers, c.num_kv_heads, c.head_dim, bytes_per_value) * (prompt + k),
    }
