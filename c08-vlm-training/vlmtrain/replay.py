"""Text replay and the forgetting check.

Replay mixes plain text into multimodal training. The forgetting check measures what replay is
for: perplexity of the language model on held-out text, before and after multimodal training.
Train and test passages come from different WikiText-2 splits, so the check never sees replayed text.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import text_sample

# Used by the offline tests and as a last resort when the Hub is unreachable.
FALLBACK_PASSAGES = [
    "The river runs north through the valley before it turns east toward the sea. Farmers on both "
    "banks grow wheat and barley, and the old stone bridge at the centre of the town is still in use.",
    "A library keeps more than books. It keeps maps, letters and newspapers, and it keeps them in an "
    "order that lets a stranger find a single page among millions.",
    "The committee met on Tuesday and agreed to repair the roof before winter. The cost will be shared "
    "between the school and the town council.",
    "Bees visit thousands of flowers in a single day. As they move from one flower to the next, they "
    "carry pollen that lets the plants produce seeds and fruit.",
]


def load_passages(split: str = "train", min_chars: int = 300, limit: int | None = None) -> list[str]:
    """Paragraphs from WikiText-2 (raw). Headings and short lines are dropped."""
    try:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        out = [t.strip() for t in ds["text"] if len(t.strip()) >= min_chars and not t.strip().startswith("=")]
    except Exception:  # offline: keep the pipeline runnable, and say so in the log
        print(f"[replay] WikiText-2 unavailable, using {len(FALLBACK_PASSAGES)} fallback passages")
        out = list(FALLBACK_PASSAGES)
    return out[:limit] if limit else out


class ReplayCorpus:
    """Indexable text-only samples in the unified format."""

    def __init__(self, passages: list[str]):
        self.passages = passages

    def __len__(self) -> int:
        return len(self.passages)

    def __getitem__(self, i: int) -> dict:
        return text_sample(self.passages[i % len(self.passages)])


@torch.no_grad()
def perplexity(lm: torch.nn.Module, tokenizer, passages: list[str], device: torch.device,
               max_tokens: int = 128, batch_size: int = 16) -> float:
    """Token-weighted perplexity of a causal LM on plain text (the language side of the VLM only)."""
    was_training = lm.training
    lm.eval()
    nll, count = 0.0, 0
    for i in range(0, len(passages), batch_size):
        enc = tokenizer(passages[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_tokens).to(device)
        logits = lm(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits.float()
        target = enc.input_ids[:, 1:].masked_fill(enc.attention_mask[:, 1:] == 0, -100)
        nll += F.cross_entropy(logits[:, :-1].transpose(1, 2), target, ignore_index=-100, reduction="sum").item()
        count += int((target != -100).sum())
    lm.train(was_training)
    return math.exp(nll / count)
