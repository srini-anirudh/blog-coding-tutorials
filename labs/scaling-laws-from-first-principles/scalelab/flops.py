"""Counting training FLOPs: the 6ND identity, what it leaves out, and three published conventions.

A matmul of a (T, n) activation with an (n, m) weight costs 2*T*n*m FLOPs. Forward does one such product per weight
matrix. Backward does two: one for the input gradient, one for the weight gradient. So a weight matrix with P entries
costs 2P + 4P = 6P FLOPs per token, and the whole linear part of a model costs 6 * N * D.
"""

from __future__ import annotations

import torch
from torch.utils.flop_counter import FlopCounterMode


def per_token(d: int, layers: int, vocab: int, seq: int, mlp: int | None = None) -> dict[str, float]:
    """Training FLOPs per token (forward + backward), split by where they go. Full multi-head attention.

    mlp is the number of MLP weights per layer. The default 8d^2 is the classic two-matrix MLP with d_ffn = 4d, which
    makes the linear part 72 L d^2. A SwiGLU MLP of width f has 3*d*f.
    """
    mlp = 8 * d * d if mlp is None else mlp
    return {
        "linear": 6.0 * layers * (4 * d * d + mlp),
        "head": 6.0 * vocab * d,  # the output projection onto the vocabulary
        "attn_unmasked": 12.0 * layers * d * seq,  # QK^T and AV, 2*S*d each forward, x3 for training
        "attn_causal": 6.0 * layers * d * seq,  # about half the score matrix is masked out
    }


def conventions(d: int, layers: int, vocab: int, seq: int, tied: bool = False) -> dict[str, float]:
    """FLOPs per token under each published convention, for a dense model with d_ffn = 4d (the post's Figure 3).

    reference  : linear + head + causal attention, what the hardware executes in matmuls.
    kaplan     : 6 * N_non-embedding. Drops the head and attention.
    chinchilla : 6 * N_total. Counts embedding parameters, which do no matmul when untied.
    deepseek_M : 72 L d^2 + 12 L d S. Non-embedding FLOPs, drops the head, adds unmasked attention.
    """
    f = per_token(d, layers, vocab, seq)
    n_nonemb = 12 * layers * d * d
    n_total = n_nonemb + vocab * d * (1 if tied else 2)
    return {
        "reference": f["linear"] + f["head"] + f["attn_causal"],
        "kaplan": 6.0 * n_nonemb,
        "chinchilla": 6.0 * n_total,
        "deepseek_M": f["linear"] + f["attn_unmasked"],
    }


def measure(model: torch.nn.Module, batch: int, seq: int, vocab: int) -> dict[str, float]:
    """Count the FLOPs PyTorch actually dispatches for one forward and one backward pass, per token.

    Splits each pass into weight matmuls (aten.mm / addmm) and fused attention (scaled_dot_product_*).
    """
    x = torch.randint(0, vocab, (batch, seq))
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(display=False) as fwd:
        logits = model(x).logits
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), x.flatten())
    with FlopCounterMode(display=False) as bwd:
        loss.backward()
    tokens = batch * seq
    out = {}
    for name, counter in (("fwd", fwd), ("bwd", bwd)):
        ops = counter.get_flop_counts()["Global"]
        attn = sum(v for k, v in ops.items() if "scaled_dot_product" in str(k))
        out[f"{name}_attn"] = attn / tokens
        out[f"{name}_matmul"] = (sum(ops.values()) - attn) / tokens
    return out
