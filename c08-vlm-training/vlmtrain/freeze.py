"""Which parameters train in each stage, and a report of trainable parameters per module.

Presets, from cheapest to most expensive:

    align     projector only                       Stage 1 (LLaVA-style alignment)
    lm        projector + LM, vision frozen        the most common Stage 2
    partial   + the last k vision layers           lets the encoder adapt to new kinds of images
    full      everything                           joint training, most memory, most risk
    none      nothing (LoRA adds its own trainable parameters afterwards)

The model only needs `.vision`, `.projector` and `.lm`, so the same code drives C7's TinyVLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MODULES = ("vision", "projector", "lm")
PRESETS = ("align", "lm", "partial", "full", "none")


def vision_layers(model: nn.Module) -> nn.ModuleList:
    """The vision tower's block list, found by shape so any ViT-like tower works."""
    best = None
    for _, m in model.vision.named_modules():
        if isinstance(m, nn.ModuleList) and (best is None or len(m) > len(best)):
            best = m
    if best is None:
        raise ValueError("no layer list found in model.vision")
    return best


def set_trainable(model: nn.Module, preset: str, vision_layers_to_train: int = 0) -> None:
    if preset not in PRESETS:
        raise ValueError(f"unknown freeze preset {preset!r}, pick one of {PRESETS}")
    on = {"align": {"projector"}, "lm": {"projector", "lm"}, "partial": {"projector", "lm"},
          "full": set(MODULES), "none": set()}[preset]
    for name in MODULES:
        getattr(model, name).requires_grad_(name in on)
    if preset == "partial":
        if vision_layers_to_train <= 0:
            raise ValueError("preset 'partial' needs vision_layers_to_train > 0")
        for layer in list(vision_layers(model))[-vision_layers_to_train:]:
            layer.requires_grad_(True)
        # The norm after the last block feeds the projector directly, so it trains with the top layers.
        for n, m in model.vision.named_modules():
            if n.endswith("post_layernorm"):
                m.requires_grad_(True)


def train_new_token_rows(model: nn.Module, first_new_id: int):
    """Let only the embedding rows of added tokens (<loc_k>, <image>) learn, the rest stay frozen.

    Needed when the LM is frozen or LoRA-adapted but the grounding format uses new tokens: without
    it the <loc_k> rows keep their initial values and the model cannot learn to emit them. The
    gradient of the old rows is zeroed by a hook. AdamW then leaves them untouched as long as their
    weight decay is 0, which optim.param_groups guarantees for embeddings.
    """
    emb = model.lm.get_input_embeddings().weight
    emb.requires_grad_(True)
    keep = torch.zeros(emb.shape[0], 1, device=emb.device, dtype=emb.dtype)
    keep[first_new_id:] = 1.0
    return emb.register_hook(lambda g: g * keep.to(g.device, g.dtype))


def trainable_report(model: nn.Module) -> dict[str, dict[str, int]]:
    """{module: {"trainable": n, "total": n}} counting each shared (tied) tensor once."""
    out: dict[str, dict[str, int]] = {}
    for name in MODULES:
        seen: set[int] = set()
        tr = tot = 0
        for p in getattr(model, name).parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            tot += p.numel()
            tr += p.numel() if p.requires_grad else 0
        out[name] = {"trainable": tr, "total": tot}
    return out


def format_report(rep: dict[str, dict[str, int]]) -> str:
    lines = [f"{'module':<10}{'trainable':>14}{'total':>14}{'share':>8}"]
    for k, v in rep.items():
        share = v["trainable"] / v["total"] if v["total"] else 0.0
        lines.append(f"{k:<10}{v['trainable']:>14,}{v['total']:>14,}{share:>8.1%}")
    tr = sum(v["trainable"] for v in rep.values())
    tot = sum(v["total"] for v in rep.values())
    lines.append(f"{'all':<10}{tr:>14,}{tot:>14,}{tr / tot:>8.1%}")
    return "\n".join(lines)
