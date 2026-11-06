"""The post's arithmetic, one function per formula.

Every number the post derives on paper comes from one of these functions. Units: bytes, FLOPs, seconds.
GB means 1e9 bytes (to match bandwidth units). KiB means 1024 bytes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    """The handful of numbers that set a model's inference cost."""

    name: str
    params: float  # P, total parameters, embeddings included
    layers: int  # L
    d_model: int  # d
    n_q: int  # query heads
    n_kv: int  # key/value heads
    head_dim: int  # d_head
    vocab: int  # V
    tied: bool = False  # True if the LM head reuses the embedding matrix

    @property
    def group(self) -> int:
        """g = n_q / n_kv: 1 is MHA, n_q is MQA."""
        return self.n_q // self.n_kv

    @property
    def embed_params(self) -> int:
        return self.vocab * self.d_model


# Shapes from the models' public configs. P is the total the post uses.
MODELS = {
    "llama-2-7b": ModelShape("Llama-2-7B (MHA)", 6.74e9, 32, 4096, 32, 32, 128, 32_000),
    "llama-3-8b": ModelShape("Llama-3-8B (GQA)", 8.03e9, 32, 4096, 32, 8, 128, 128_256),
    "mistral-7b": ModelShape("Mistral-7B (GQA)", 7.24e9, 32, 4096, 32, 8, 128, 32_000),
    "llama-3-70b": ModelShape("Llama-3-70B (GQA)", 70.6e9, 80, 8192, 64, 8, 128, 128_256),
}


@dataclass(frozen=True)
class Accelerator:
    name: str
    peak_flops: float  # dense bf16 FLOP/s (pi)
    bandwidth: float  # HBM bytes/s (beta)
    memory: float  # bytes


# Vendor spec sheets, dense bf16. These are what the post uses; they are not measurements.
ACCELERATORS = {
    "h100": Accelerator("H100 SXM (spec)", 989e12, 3.35e12, 80e9),
    "a100": Accelerator("A100 SXM (spec)", 312e12, 2.039e12, 80e9),
    "l40s": Accelerator("L40S (spec)", 362e12, 0.864e12, 48e9),
}


# ---- KV cache (post sections 4, 5, 7) --------------------------------------------------------------------------


def kv_bytes_per_token(m: ModelShape, s: float = 2) -> float:
    """2 * L * n_kv * d_head * s. The 2 is one K and one V."""
    return 2 * m.layers * m.n_kv * m.head_dim * s


def kv_bytes(m: ModelShape, batch: int, context: int, s: float = 2) -> float:
    """M_KV = 2 * B * N * L * n_kv * d_head * s."""
    return batch * context * kv_bytes_per_token(m, s)


def mla_bytes_per_token(layers: int, d_c: int, d_rope: int, s: float = 2) -> float:
    """Multi-head latent attention caches one latent of width d_c plus a small RoPE key, per layer."""
    return layers * (d_c + d_rope) * s


def crossover_tokens(m: ModelShape, s_w: float = 2, s_c: float = 2) -> float:
    """Total cached tokens (B * N) at which the cache weighs as much as the weights."""
    return m.params * s_w / kv_bytes_per_token(m, s_c)


# ---- Work in each phase (section 3) ----------------------------------------------------------------------------


def prefill_flops(m: ModelShape, n_prompt: int, params: float | None = None) -> float:
    """2 P N_p (linear layers) + 2 L N_p^2 d (causal attention)."""
    p = m.params if params is None else params
    return 2 * p * n_prompt + 2 * m.layers * n_prompt**2 * m.d_model


def decode_flops(m: ModelShape, context: int) -> float:
    """One token against N cached positions: 2 P + 4 L N d."""
    return 2 * m.params + 4 * m.layers * context * m.d_model


def attention_equals_linear(m: ModelShape, params: float | None = None) -> float:
    """Prompt length at which prefill's attention term equals its linear term: N_p = P / (L d)."""
    p = m.params if params is None else params
    return p / (m.layers * m.d_model)


def gemm_intensity(rows: int, d_in: int, d_out: int, s: float = 2) -> float:
    """FLOP per byte of one (rows x d_in) @ (d_in x d_out) matmul, counting each operand once at HBM."""
    return 2 * rows * d_in * d_out / (s * (rows * d_in + d_in * d_out + rows * d_out))


def no_cache_vs_cache_flops(params: float, n_prompt: int, n_out: int) -> tuple[float, float]:
    """Linear-layer FLOPs to generate n_out tokens, recomputing the prefix each step vs caching it."""
    without = 2 * params * (n_out * n_prompt + n_out**2 / 2)
    with_cache = 2 * params * n_prompt + 2 * params * n_out
    return without, with_cache


# ---- The decode wall and batching (sections 10, 11) ------------------------------------------------------------


def decode_step_bytes(m: ModelShape, batch: int, context: int, s_w: float = 2, s_c: float = 2) -> float:
    """Q = s_w P + B * (KV bytes of one sequence)."""
    return s_w * m.params + kv_bytes(m, batch, context, s_c)


def decode_step_floor(m: ModelShape, acc: Accelerator, batch: int, context: int, s_w: float = 2,
                      s_c: float = 2) -> float:
    """t_step >= Q / beta, in seconds. A floor: free arithmetic, perfect kernels."""
    return decode_step_bytes(m, batch, context, s_w, s_c) / acc.bandwidth


def b_star(acc: Accelerator, s_w: float = 2) -> float:
    """Batch at which the weight read alone becomes compute-bound: s_w * pi / (2 beta)."""
    return s_w * acc.peak_flops / (2 * acc.bandwidth)


def b_cross(m: ModelShape, context: int, s_w: float = 2, s_c: float = 2) -> float:
    """Batch at which cache bytes overtake weight bytes in one decode step."""
    return m.params * s_w / kv_bytes(m, 1, context, s_c)


def b_max(m: ModelShape, acc: Accelerator, context: int, s_w: float = 2, s_c: float = 2,
          workspace: float = 2e9) -> float:
    """How many sequences fit next to the weights and a workspace allowance."""
    return (acc.memory - m.params * s_w - workspace) / kv_bytes(m, 1, context, s_c)


def decode_intensity(m: ModelShape, batch: float, context: int, s_w: float = 2, s_c: float = 2) -> float:
    """I(B) = B (2P + 4 L N d) / (s_w P + B * KV/seq)."""
    return batch * decode_flops(m, context) / decode_step_bytes(m, batch, context, s_w, s_c)


def decode_intensity_limit(m: ModelShape, context: int, s_c: float = 2) -> float:
    """I at B -> infinity: 2P / (KV/seq) + 2g / s_c. Tends to 2g / s_c as N grows."""
    return decode_flops(m, context) / kv_bytes(m, 1, context, s_c)


# ---- Speculation, MoE, tensor parallelism, scheduling (sections 15, 16, 19) ------------------------------------


def spec_expected_tokens(alpha: float, gamma: int) -> float:
    """Expected tokens per target pass, i.i.d. acceptance alpha: (1 - alpha^(gamma+1)) / (1 - alpha)."""
    if alpha == 1.0:
        return gamma + 1.0
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def spec_speedup(alpha: float, gamma: int, draft_cost: float, verify_cost: float = 1.0) -> float:
    """E[tokens] / (verify_cost + gamma * draft_cost), both in units of one plain target step."""
    return spec_expected_tokens(alpha, gamma) / (verify_cost + gamma * draft_cost)


def moe_distinct_experts(n_experts: int, top_k: int, batch: int) -> float:
    """Expected number of distinct experts one decode step touches, uniform routing."""
    return n_experts * (1 - (1 - top_k / n_experts) ** batch)


def tp_decode_step(single_gpu_step: float, n: int, layers: int, allreduce_latency: float = 6e-6) -> float:
    """Tensor parallel: bytes split n ways, plus 2 all-reduces per layer that do not split."""
    return single_gpu_step / n + (2 * layers * allreduce_latency if n > 1 else 0.0)


def static_batch_utilisation(output_lengths: list[int]) -> float:
    """Share of slot-steps doing real work when the whole batch runs to its longest member."""
    return sum(output_lengths) / (len(output_lengths) * max(output_lengths))
