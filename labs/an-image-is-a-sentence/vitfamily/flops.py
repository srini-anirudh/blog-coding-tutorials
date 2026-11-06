"""The post's cost formulas, and a counter that measures the same quantities on real modules.

Conventions, as in the post: FLOPs = 2 x multiply-accumulates (MACs), matmuls and convolutions only
(no softmax, LayerNorm, GELU or bias adds). Swin's paper and the post's Swin table count MACs of the
attention sublayer only, without the MLP; swin_* functions follow that convention.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import torch
from torch.utils.flop_counter import FlopCounterMode


# ---- formulas, per block, per sample -------------------------------------------------------------
def vit_block_flops(n: int, d: int) -> dict[str, int]:
    """Pre-norm ViT block: QKVO projections 8ND^2, MLP 16ND^2, QK^T and AV 4N^2 D."""
    return {"linear": 24 * n * d * d, "attention": 4 * n * n * d}


def crossover_tokens(d: int) -> int:
    """24 N D^2 = 4 N^2 D  =>  N = 6 D."""
    return 6 * d


def swin_msa_macs(h: int, w: int, d: int) -> dict[str, int]:
    """Swin eq. (1): global MSA, attention sublayer only, in MACs."""
    return {"linear": 4 * h * w * d * d, "attention": 2 * (h * w) ** 2 * d}


def swin_wmsa_macs(h: int, w: int, d: int, m: int) -> dict[str, int]:
    """Swin eq. (2): window MSA. hw / M^2 windows of M^2 tokens each."""
    return {"linear": 4 * h * w * d * d, "attention": 2 * m * m * h * w * d}


def sra_block_flops(n: int, d: int, r: int) -> dict[str, int]:
    """PVT block, counted op by op: Q and O on N tokens, K and V on N / R^2, the R x R strided conv, MLP."""
    reduced = n // (r * r)
    conv = 2 * reduced * (r * r * d) * d if r > 1 else 0
    return {"linear": 4 * n * d * d + 4 * reduced * d * d + 16 * n * d * d, "conv": conv, "attention": 4 * n * reduced * d}


def sra_block_flops_post(n: int, d: int, r: int) -> dict[str, int]:
    """The post's complexity-bench row for SRA: 4N^2 D / R^2 attention, 24 N D^2 + N D^2 / R^2 the rest."""
    return {"rest": 24 * n * d * d + n * d * d // (r * r), "attention": 4 * n * n * d // (r * r)}


def divided_block_flops(t: int, n: int, d: int) -> dict[str, int]:
    """Divided space-time block: two attention sublayers (8 T N D^2 each) and one MLP (16 T N D^2)."""
    return {"linear": 32 * t * n * d * d, "attention": 4 * t * n * (t + n) * d}


def joint_block_flops(t: int, n: int, d: int) -> dict[str, int]:
    return vit_block_flops(t * n, d)


# ---- score-matrix and activation memory, in elements ---------------------------------------------
def score_elems_global(n: int, heads: int) -> int:
    return heads * n * n


def score_elems_window(n: int, heads: int, m: int) -> int:
    return heads * n * m * m


def vit_train_activation_elems(n: int, d: int, depth: int) -> int:
    """The post's 13 N D per block: input, LN, Q, K, V, attention output, MLP hidden (4ND), GELU input (4ND)."""
    return 13 * n * d * depth


def st_pairs(t: int, n: int) -> dict[str, int]:
    """Query-key pairs per head per block for joint and divided space-time attention."""
    return {"joint": (t * n) ** 2, "divided": t * n * (t + n)}


# ---- measurement ---------------------------------------------------------------------------------
def _kind(op_name: str) -> str:
    if "scaled_dot_product" in op_name or "bmm" in op_name:
        return "attention"
    if "convolution" in op_name:
        return "conv"
    return "linear"  # mm, addmm


@torch.no_grad()
def measure_flops(fn: Callable[[], object]) -> dict[str, int]:
    """Run fn() under torch's FlopCounterMode and sort the counted FLOPs into linear / attention / conv."""
    with FlopCounterMode(display=False) as counter:
        fn()
    out: dict[str, int] = defaultdict(int)
    for op, flops in counter.get_flop_counts()["Global"].items():
        out[_kind(str(op))] += int(flops)
    out["total"] = sum(v for k, v in out.items())
    return dict(out)
