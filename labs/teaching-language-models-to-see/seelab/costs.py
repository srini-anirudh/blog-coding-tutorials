"""Freeze or train: the rule-of-thumb cost model of the post's Figure 4, and a measurement to check it against.

Rule of thumb, per token that passes through a part with P parameters:
    forward                         2 P
    activation gradients            2 P   if the backward pass must travel through the part to reach a trained
                                          part upstream of it
    weight gradients                2 P   if the part is trained
"""

from __future__ import annotations

import statistics
import time

import torch

from . import common  # noqa: F401  (puts C7 on sys.path)

PARTS = ("vision", "projector", "lm")
PATTERNS = {  # (vision, connector, language model) trained?  -- the four rows of the post's freezing table
    "align: connector only (LLaVA stage 1)": (False, True, False),
    "adapt language side (LLaVA stage 2)": (False, True, True),
    "adapt vision side (Qwen-VL stage 1)": (True, True, False),
    "joint: train everything": (True, True, True),
}


def predicted_flops(params: dict[str, int], tokens: dict[str, int], trained: tuple[bool, bool, bool]) -> float:
    """FLOPs for one example under the rule above. Parts are in forward order: vision, connector, LM."""
    total = 0.0
    for i, part in enumerate(PARTS):
        p, n = params[part], tokens[part]
        total += 2 * p * n
        if any(trained[:i]):  # something upstream needs the gradient, so it must pass through this part
            total += 2 * p * n
        if trained[i]:
            total += 2 * p * n
    return total


def predicted_bytes(params: dict[str, int], trained: tuple[bool, bool, bool], trained_bytes: int = 16,
                    frozen_bytes: int = 2) -> float:
    """Weights + gradients + Adam state. Mixed precision: 16 and 2. Plain float32: 16 and 4."""
    return sum(params[p] * (trained_bytes if t else frozen_bytes) for p, t in zip(PARTS, trained))


def tiny_cost_vlm(image_size: int = 384, lm_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
    """C7's TinyVLM with random weights: SigLIP-base at 384 px (576 patches), MLP connector, SmolLM2-135M's shape.

    Weights do not change the cost, so nothing is downloaded but the language model's config.
    """
    from transformers import AutoConfig, SiglipVisionConfig

    from tinyvlm.configuration import VLMConfig
    from tinyvlm.modeling import TinyVLM

    vision_config = SiglipVisionConfig(image_size=image_size, patch_size=16, vision_use_head=False)
    text_config = AutoConfig.from_pretrained(lm_id)
    config = VLMConfig(vision_config, text_config, image_token_id=text_config.vocab_size - 1,
                       projector_type="mlp", pixel_shuffle_factor=1)
    torch.manual_seed(0)
    return TinyVLM(config)


def param_counts(model) -> dict[str, int]:
    return {p: sum(x.numel() for x in getattr(model, p).parameters()) for p in PARTS}


def state_bytes(model, optimizer) -> int:
    """Bytes actually held: every weight, every gradient, every optimizer state tensor."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
    for state in optimizer.state.values():
        total += sum(v.numel() * v.element_size() for v in state.values() if torch.is_tensor(v) and v.dim() > 0)
    return total


def make_batch(model, batch_size: int, n_text: int, device) -> dict:
    cfg = model.config
    n_img, size = cfg.image_seq_len, cfg.vision_config.image_size
    g = torch.Generator().manual_seed(0)
    text = torch.randint(0, cfg.image_token_id, (batch_size, n_text), generator=g)
    ids = torch.cat([torch.full((batch_size, n_img), cfg.image_token_id), text], 1)
    labels = torch.cat([torch.full((batch_size, n_img), -100), text], 1)
    return {"input_ids": ids.to(device), "labels": labels.to(device),
            "attention_mask": torch.ones_like(ids).to(device),
            "pixel_values": torch.randn(batch_size, 3, size, size, generator=g).to(device)}


def time_step(model, batch, trained: tuple[bool, bool, bool], repeats: int = 3, warmup: int = 1) -> dict:
    """Median seconds for forward + backward + optimizer step, and the bytes held after a step."""
    for part, t in zip(PARTS, trained):
        getattr(model, part).requires_grad_(t)
    model.zero_grad(set_to_none=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    cuda = batch["input_ids"].is_cuda
    times = []
    for i in range(warmup + repeats):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = model(**batch).loss
        loss.backward()
        opt.step()
        if cuda:
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
        held = state_bytes(model, opt)
        opt.zero_grad(set_to_none=False)  # keep the gradient buffers: they are part of what training holds
    out = {"seconds": statistics.median(times), "all_seconds": times, "bytes": held}
    model.zero_grad(set_to_none=True)
    del opt
    return out
