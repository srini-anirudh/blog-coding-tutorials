"""Back-of-envelope accounting for one training step, from six architecture numbers.

Every figure in the post is arithmetic on a Llama-2-7B-shaped model. This module redoes that
arithmetic in code, so each number can be checked and re-derived for another shape.
Nothing here runs a model. The measured counterparts live in the other modules.
"""

from __future__ import annotations

from dataclasses import dataclass

# Datasheet numbers for an H100 SXM, the GPU the post assumes. Dense, no sparsity.
H100_BF16_FLOPS = 989e12
H100_FP32_FLOPS = 67e12
H100_HBM_BYTES_PER_S = 3.35e12
H100_RIDGE = H100_BF16_FLOPS / H100_HBM_BYTES_PER_S  # FLOP per byte where the roofline bends


@dataclass(frozen=True)
class Shape:
    layers: int = 32
    hidden: int = 4096
    heads: int = 32
    intermediate: int = 11008
    vocab: int = 32000
    seq: int = 4096
    batch: int = 1

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def tokens(self) -> int:
        return self.batch * self.seq


LLAMA2_7B = Shape()


def params_per_layer(s: Shape) -> int:
    """Q, K, V, O are H x H each; gate, up, down are H x I each. Norm weights are ignored."""
    return 4 * s.hidden**2 + 3 * s.hidden * s.intermediate


def param_count(s: Shape) -> int:
    """32 x (4H^2 + 3HI) + 2VH: blocks, plus an untied embedding and LM head."""
    return s.layers * params_per_layer(s) + 2 * s.vocab * s.hidden


@dataclass(frozen=True)
class Op:
    name: str
    flops: float
    hbm_bytes: float  # minimal traffic: read every input and weight once, write every output once

    @property
    def intensity(self) -> float:
        return self.flops / self.hbm_bytes

    @property
    def bound(self) -> str:
        return "compute" if self.intensity > H100_RIDGE else "memory"


def block_ops(s: Shape, bytes_per_el: int = 2, causal: bool = True) -> list[Op]:
    """One decoder block, op by op, for one microbatch.

    FLOP counts: a GEMM is 2mnk. RMSNorm is about 4 FLOPs per element (square, sum, scale, weight),
    SiLU-multiply about 5, a residual add 1. Attention uses FlashAttention-style traffic: Q, K, V are
    read once and the S x S scores never reach HBM. Causal masking halves the attention FLOPs.
    """
    t, h, i = s.tokens, s.hidden, s.intermediate
    e = bytes_per_el
    act_h, act_i = t * h * e, t * i * e

    def gemm(name: str, n_in: int, n_out: int) -> Op:
        return Op(name, 2 * t * n_in * n_out, t * n_in * e + n_in * n_out * e + t * n_out * e)

    attn_flops = 4 * s.batch * s.seq**2 * h * (0.5 if causal else 1.0)
    return [
        Op("rmsnorm (attn)", 4 * t * h, 2 * act_h),
        gemm("qkv proj", h, 3 * h),
        Op("attention (flash)", attn_flops, 4 * act_h),
        gemm("out proj", h, h),
        Op("residual add", t * h, 3 * act_h),
        Op("rmsnorm (mlp)", 4 * t * h, 2 * act_h),
        gemm("gate proj", h, i),
        gemm("up proj", h, i),
        Op("silu * mul", 5 * t * i, 3 * act_i),
        gemm("down proj", i, h),
        Op("residual add", t * h, 3 * act_h),
    ]


PROJECTIONS = {"qkv proj", "out proj", "gate proj", "up proj", "down proj"}


def projection_share(s: Shape, causal: bool = True) -> float:
    ops = block_ops(s, causal=causal)
    return sum(o.flops for o in ops if o.name in PROJECTIONS) / sum(o.flops for o in ops)


def tensor_sizes(s: Shape, bytes_per_el: int = 2) -> dict[str, tuple[tuple[int, ...], int, int]]:
    """name -> (shape, elements, bytes) for the tensors in the post's shape table."""
    b, t, h = s.batch, s.seq, s.hidden
    shapes = {
        "block input X": (b, t, h),
        "Q, K, V (packed)": (b, t, 3 * h),
        "per-head view": (b, s.heads, t, s.head_dim),
        "attention scores": (b, s.heads, t, t),
        "MLP intermediate": (b, t, s.intermediate),
        "logits": (b, t, s.vocab),
    }
    out = {}
    for k, shp in shapes.items():
        n = 1
        for d in shp:
            n *= d
        out[k] = (shp, n, n * bytes_per_el)
    return out


def attention_flops_per_layer(s: Shape) -> float:
    """QK^T and PV, no causal saving: 4 B S^2 H."""
    return 4 * s.batch * s.seq**2 * s.hidden


def projection_flops_per_layer(s: Shape) -> float:
    return 2 * s.tokens * params_per_layer(s)


def attention_crossover_seq(s: Shape) -> float:
    """S at which 4 B S^2 H equals 2 B S P_layer: S* = P_layer / 2H."""
    return params_per_layer(s) / (2 * s.hidden)


def train_flops(s: Shape, tokens: float | None = None, recompute: bool = False) -> float:
    """6ND, or 8ND when every block's forward runs a second time in backward."""
    d = s.tokens if tokens is None else tokens
    return (8 if recompute else 6) * param_count(s) * d


# Mixed-precision recipe from the post: BF16 compute weights and grads, FP32 master and Adam moments.
LEDGER_BYTES = {"params (bf16)": 2, "grads (bf16)": 2, "master (fp32)": 4, "adam m (fp32)": 4, "adam v (fp32)": 4}


def model_state_bytes(s: Shape) -> dict[str, float]:
    n = param_count(s)
    out = {k: v * n for k, v in LEDGER_BYTES.items()}
    out["total"] = sum(out.values())
    return out


def activation_bytes(s: Shape, elems_per_token_layer_in_h: float = 16, bytes_per_el: int = 2) -> float:
    """M_act = L * B * S * 16H * 2 bytes. 16H is the post's count of saved elements with FlashAttention."""
    return s.layers * s.tokens * elems_per_token_layer_in_h * s.hidden * bytes_per_el


def checkpointed_activation_bytes(s: Shape, bytes_per_el: int = 2) -> float:
    """Full checkpointing: one H-wide block input per layer, plus one block's full 16H while it is recomputed."""
    return (s.layers * 1 + 16) * s.tokens * s.hidden * bytes_per_el


# Adam, per parameter: read grad (bf16, 2), read master (4), read+write m (8), read+write v (8),
# write master (4), write the bf16 copy (2).
ADAM_BYTES_PER_PARAM = 2 + 4 + 8 + 8 + 4 + 2
ADAM_FLOPS_PER_PARAM = 12


def optimizer_step_seconds(s: Shape, bytes_per_s: float = 2.5e12) -> float:
    return param_count(s) * ADAM_BYTES_PER_PARAM / bytes_per_s


def run_seconds(n_params: float, tokens: float, mfu: float, peak: float = H100_BF16_FLOPS) -> float:
    return 6 * n_params * tokens / (mfu * peak)


def bf16_unit_roundoff() -> float:
    """BF16 keeps 8 significant bits (7 stored plus an implicit one). The gap above 1.0 is 2^-7, so any
    relative change below half of it, 2^-8, rounds back to the old value."""
    return 2.0**-8
