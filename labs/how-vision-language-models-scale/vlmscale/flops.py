"""Count the FLOPs of one training step per tower, with PyTorch's FlopCounterMode.

FlopCounterMode records every matmul in the forward and backward pass and attributes it to the
module that ran it. We split each tower's count into weight matmuls (`mm`, `addmm`: activations
times a weight matrix) and attention-score matmuls (`bmm`: QK^T and AV), then divide the weight
part by (matmul parameters x tokens) to get k, the FLOPs per parameter per token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.flop_counter import FlopCounterMode

from .model import TwoTowerVLM, matmul_params, set_trainable

WEIGHT_OPS = ("mm", "addmm")
ATTN_OPS = ("bmm",)

SETTINGS = {
    # name: (vision trains, connector trains, llm trains)
    "all trained": (True, True, True),
    "frozen encoder": (False, True, True),
    "frozen encoder + LLM (connector only)": (False, True, False),
    "frozen LLM, encoder trains": (True, True, False),
}


@dataclass
class TowerCount:
    weight: float
    attention: float
    params: int
    tokens: int

    @property
    def k(self) -> float:
        return self.weight / (self.params * self.tokens)


def _op_total(counts: dict, ops: tuple[str, ...]) -> float:
    return float(sum(v for op, v in counts.items() if str(op).split(".")[1] in ops))


def count_step(model: TwoTowerVLM, pixels: torch.Tensor, text: torch.Tensor, targets: torch.Tensor,
               vision: bool, connector: bool, llm: bool) -> dict[str, TowerCount]:
    """FLOPs of forward + backward for one batch, per tower, under a freezing setting."""
    set_trainable(model, vision, connector, llm)
    model.zero_grad(set_to_none=True)
    b, t = text.shape
    cfg = model.cfg
    tokens = {"vision": b * cfg.patches, "connector": b * cfg.visual_tokens, "llm": b * (cfg.visual_tokens + t)}
    with FlopCounterMode(display=False) as fc:
        loss = model.loss(pixels, text, targets)
        if loss.requires_grad:
            loss.backward()
    per_module = fc.get_flop_counts()
    out = {}
    for name in ("vision", "connector", "llm"):
        counts = per_module.get(f"TwoTowerVLM.{name}", {})
        out[name] = TowerCount(_op_total(counts, WEIGHT_OPS), _op_total(counts, ATTN_OPS),
                               matmul_params(getattr(model, name)), tokens[name])
    set_trainable(model, True, True, True)
    return out


def measured_encoder_share(counts: dict[str, TowerCount], include_attention: bool = False) -> float:
    def tot(c: TowerCount) -> float:
        return c.weight + (c.attention if include_attention else 0.0)
    return tot(counts["vision"]) / (tot(counts["vision"]) + tot(counts["llm"]))
