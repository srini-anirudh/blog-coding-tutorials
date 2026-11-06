"""PEFT LoRA: wrap a model, inspect what changed, check the math by hand, merge.

For why a low-rank update works, see the LoRA theory post. Here we check that PEFT computes
    y = W x + (alpha / r) · B (A x)
with A: (r, d_in) random, B: (d_out, r) zeros at the start, and that merging folds B A into W.
"""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn

ATTENTION_PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def add_lora(model: nn.Module, r: int = 8, alpha: int = 16, targets: list[str] | None = None) -> PeftModel:
    """Freeze `model` and insert rank-`r` adapters into every module whose name ends in one of `targets`."""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=targets or ATTENTION_PROJECTIONS,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def lora_shapes(peft_model: PeftModel) -> list[tuple[str, tuple[int, ...]]]:
    """Name and shape of every trainable tensor. These are the only tensors the optimizer sees."""
    return [(n, tuple(p.shape)) for n, p in peft_model.named_parameters() if p.requires_grad]


def manual_lora_forward(layer: nn.Module, x: torch.Tensor, adapter: str = "default") -> torch.Tensor:
    """Recompute a PEFT `lora.Linear` output from its parts. x: (..., d_in) → (..., d_out)."""
    base = layer.base_layer(x)  # W x (+ bias)
    A = layer.lora_A[adapter].weight  # (r, d_in)
    B = layer.lora_B[adapter].weight  # (d_out, r)
    scale = layer.scaling[adapter]  # alpha / r
    return base + scale * (x @ A.T @ B.T)


def adapter_size_bytes(peft_model: PeftModel, directory: str | Path) -> int:
    """Save only the adapter and return the size of its weights file."""
    peft_model.save_pretrained(directory, save_embedding_layers=False)  # only adapters; no Hub lookup
    return (Path(directory) / "adapter_model.safetensors").stat().st_size


@torch.no_grad()
def merge_matches(peft_model: PeftModel, input_ids: torch.Tensor) -> tuple[nn.Module, float]:
    """Merge adapters into the base weights and return (merged model, max |logit difference|)."""
    before = peft_model(input_ids=input_ids).logits
    merged = peft_model.merge_and_unload()  # W ← W + (alpha / r) B A; the lora modules disappear
    after = merged(input_ids=input_ids).logits
    return merged, (before - after).abs().max().item()
