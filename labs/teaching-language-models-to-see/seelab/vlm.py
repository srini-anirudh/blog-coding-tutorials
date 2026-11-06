"""Thin helpers around C7's TinyVLM: build it, train the connector, caption held-out pictures."""

from __future__ import annotations

import math

import torch

from . import common  # noqa: F401  (puts C7 on sys.path)
from tinyvlm.data import make_example, score  # noqa: E402

TRAIN_SEED, EVAL_SEED = 0, 1


def build(projector: str = "mlp", seed: int = 0):
    """SigLIP-base + fresh projector + SmolLM2-135M-Instruct (C7's build), with a fixed projector seed."""
    import transformers

    from tinyvlm import build_tinyvlm

    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    torch.manual_seed(seed)
    return build_tinyvlm(projector_type=projector, pixel_shuffle_factor=2)


def batch_of(processor, start: int, n: int, seed: int = TRAIN_SEED) -> tuple[dict, list[dict]]:
    examples = [make_example(start + i, seed=seed) for i in range(n)]
    return processor.collate([processor.build_example(e["messages"], e["images"]) for e in examples]), examples


def to(batch: dict, device) -> dict:
    return {k: batch[k].to(device) for k in ("input_ids", "pixel_values", "attention_mask", "labels")}


def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak (C7's schedule)."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def caption_eval(model, processor, n: int, device, batch_size: int = 16) -> tuple[list[str], dict]:
    """Greedy captions for n held-out pictures, scored by parsing count, colour and shape."""
    examples = [make_example(i, seed=EVAL_SEED) for i in range(n)]
    preds = []
    for s in range(0, n, batch_size):
        chunk = examples[s:s + batch_size]
        prompts = [processor.apply_chat_template(e["messages"][:1], add_generation_prompt=True) for e in chunk]
        b = processor(prompts, [e["images"] for e in chunk])
        new = model.generate(b["input_ids"].to(device), b["pixel_values"].to(device), b["attention_mask"].to(device),
                             max_new_tokens=12, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
        preds += [p.strip() for p in processor.batch_decode(new)]
    return preds, score(preds, [e["target"] for e in examples])
