"""Choose which parts of the VLM learn: the freeze ablation.

    llm             LoRA on the decoder.                       Vision and connector frozen.
    projector+llm   LoRA on the decoder, connector fully trained.  Vision frozen.
    all             LoRA on decoder and vision encoder, connector fully trained.

If "llm" matches "all" after RL, the gain did not need new visual features.
"""

from __future__ import annotations

from torch import nn

from .lora import add_lora

TEXT_LINEARS = r"model\.text_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
VISION_LINEARS = r"model\.vision_model\.encoder\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|mlp\.fc(1|2))$"
CONNECTOR = "model.connector."
MODES = ("llm", "projector+llm", "all")


def configure_trainable(model: nn.Module, mode: str, r: int = 16, alpha: int = 32) -> dict[str, int]:
    """Freeze everything, then open up the parts `mode` names. Returns trainable counts per part."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    for p in model.parameters():
        p.requires_grad_(False)
    add_lora(model, TEXT_LINEARS, r, alpha)
    if mode == "all":
        add_lora(model, VISION_LINEARS, r, alpha)
    if mode in ("projector+llm", "all"):
        for p in model.model.connector.parameters():
            p.requires_grad_(True)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
    return trainable_counts(model)


def trainable_counts(model: nn.Module) -> dict[str, int]:
    parts = {"vision": "model.vision_model.", "connector": CONNECTOR, "llm": "model.text_model.", "lm_head": "lm_head."}
    out = {k: 0 for k in parts}
    for n, p in model.named_parameters():
        if p.requires_grad:
            for k, prefix in parts.items():
                if n.startswith(prefix):
                    out[k] += p.numel()
    out["total"] = sum(out.values())
    return out
