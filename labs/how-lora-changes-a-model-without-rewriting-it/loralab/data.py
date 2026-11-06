"""Text data for the real-model experiments: WikiText-2 (source domain) and a cipher of it (target domain).

The target domain writes every word backwards ("the cat sat" -> "eht tac tas"). The base model has never seen this
"language", so learning it is adding a capability, not re-weighting one. Plain WikiText measures what is forgotten.
"""

from __future__ import annotations

import re

import torch

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
_WORD = re.compile(r"[A-Za-z]+")


def reverse_words(text: str) -> str:
    return _WORD.sub(lambda m: m.group(0)[::-1], text)


def wikitext(split: str) -> str:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return "".join(t for t in ds["text"] if t.strip())


def chunks(tokenizer, text: str, seq: int, limit: int | None = None) -> torch.Tensor:
    """Tokenize once, cut into non-overlapping rows of `seq` tokens. Returns (n, seq) int64."""
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // seq
    if limit:
        n = min(n, limit)
    return ids[: n * seq].view(n, seq)


@torch.no_grad()
def mean_loss(model, rows: torch.Tensor, batch: int = 8, device: str = "cpu") -> float:
    """Mean next-token cross entropy over all rows (nats per token)."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(rows), batch):
        x = rows[i:i + batch].to(device)
        out = model(input_ids=x, labels=x)
        k = x.numel() - x.shape[0]
        total += out.loss.item() * k
        count += k
    return total / count
