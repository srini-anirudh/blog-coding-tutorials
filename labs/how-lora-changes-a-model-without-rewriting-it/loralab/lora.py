"""LoRA and DoRA as one small nn.Module, with every option the post discusses.

    h = W x + s * B (A x)          W frozen (out, in), A (r, in), B (out, r)
    s = alpha / r                  the LoRA paper's scaling ("standard")
    s = alpha / sqrt(r)            rank-stabilized scaling ("rslora", Kalajdzievski 2023)

DoRA (Liu et al. 2024) rescales each output row of W + s B A to a trainable length m:

    W' = m * (W + s B A) / ||W + s B A||_row

tests/test_lora.py checks both against Hugging Face PEFT with the same factors.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .quant import NF4Linear

ALL_LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
INITS = ("b_zero", "a_zero", "both_zero", "both_random")


def lora_scale(alpha: float, r: int, scaling: str = "standard") -> float:
    if scaling == "standard":
        return alpha / r
    if scaling == "rslora":
        return alpha / math.sqrt(r)
    raise ValueError(f"unknown scaling {scaling!r}")


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a parallel rank-r branch.

    init picks which factor starts at zero. Only "b_zero" (PEFT's default) and "a_zero" give an exact
    copy of the base model at step 0 with a live gradient path; the post explains why.
    """

    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0, scaling: str = "standard",
                 init: str = "b_zero", dora: bool = False, dropout: float = 0.0):
        super().__init__()
        if init not in INITS:
            raise ValueError(f"init must be one of {INITS}")
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scaling = r, alpha, scaling
        self.scale = lora_scale(alpha, r, scaling)
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
        if init in ("b_zero", "both_random"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # PEFT's default for A
        if init in ("a_zero", "both_random"):
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dora = dora
        if dora:
            # One number per output row, initialised to the row norm of W so W' = W at step 0.
            self.magnitude = nn.Parameter(base.weight.detach().norm(dim=1).clone())
        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        """s * B A, the (out, in) update this adapter represents (before DoRA's rescaling)."""
        return self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.base(x)
        base_out = self.base(x)
        lora_out = self.scale * F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        if not self.dora:
            return base_out + lora_out
        # DoRA. Like the paper (Sec. 4.3) and PEFT, the row norm is treated as a constant in the backward pass.
        w = self.base.weight
        norm = (w + self.delta_weight().detach()).norm(dim=1).detach()
        g = (self.magnitude / norm).view(*([1] * (x.dim() - 1)), -1)
        bias = self.base.bias if self.base.bias is not None else 0.0
        return g * (base_out - bias + lora_out) + bias

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """The single matrix this layer is equivalent to. Computed in fp32, returned in W's dtype."""
        w = self.base.weight.float() + self.delta_weight().float()
        if self.dora:
            w = self.magnitude.float().view(-1, 1) * w / w.norm(dim=1, keepdim=True)
        return w.to(self.base.weight.dtype)

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into W in place. Exactly reversible only for plain LoRA in exact arithmetic."""
        if self.merged:
            return
        if self.dora:
            raise NotImplementedError("DoRA merges with merged_weight(); unmerge is not a subtraction")
        self.base.weight += self.delta_weight().to(self.base.weight.dtype)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base.weight -= self.delta_weight().to(self.base.weight.dtype)
        self.merged = False


def apply_lora(model: nn.Module, targets: tuple[str, ...] = ALL_LINEAR, **kw) -> list[str]:
    """Freeze every parameter, then wrap each nn.Linear whose attribute name is in targets. Returns wrapped names."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, (nn.Linear, NF4Linear)):
                setattr(module, child_name, LoRALinear(child, **kw))
                wrapped.append(f"{name}.{child_name}" if name else child_name)
    return wrapped


def lora_layers(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the trainable tensors. This is the whole LoRA checkpoint."""
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def merge_all(model: nn.Module) -> nn.Module:
    """Replace every LoRALinear by a plain nn.Linear holding the merged weight."""
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                child.base.weight.data = child.merged_weight()
                setattr(module, child_name, child.base)
    return model


def manual_grads(W: torch.Tensor, A: torch.Tensor, B: torch.Tensor, s: float, x: torch.Tensor, g: torch.Tensor):
    """The post's backward pass, written out for a batch. x: (n, in), g = dL/dh: (n, out).

        dL/dB = s g (A x)^T     dL/dA = s B^T g x^T     dL/dx = W^T g + s A^T B^T g
    """
    ax = x @ A.T                      # (n, r)
    dB = s * g.T @ ax                 # (out, r)
    dA = s * (g @ B).T @ x            # (r, in)
    dx = g @ W + s * (g @ B) @ A      # (n, in)
    return dB, dA, dx


def multi_adapter_forward(x: torch.Tensor, W: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor,
                          ids: torch.Tensor, scale: float) -> torch.Tensor:
    """One shared base GEMM for a mixed batch, plus a gathered low-rank branch per request (S-LoRA's layout).

    x: (n, in). W: (out, in). As: (k, r, in). Bs: (k, out, r). ids: (n,) adapter index of each request.
    """
    base = x @ W.T                                         # one GEMM over the whole batch
    ax = torch.einsum("ni,nri->nr", x, As[ids])            # each request reads through its own A
    return base + scale * torch.einsum("nr,nor->no", ax, Bs[ids])
