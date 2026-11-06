"""Data and loss tips from "How the families were trained", small enough to check by hand.

    capped_weights      cap each source (Cambrian-1), then sample in proportion or by square root (Molmo).
                        Square-root sampling is C8's temperature mixture with T = 2; we import it.
    token_shares        a mixture written in samples is not the mixture the loss sees, which is per token
    sft_loss            three normalisations of the SFT loss: per token, per sample, and the square-root
                        weighting of InternVL 2.5
    modality_warmup     DeepSeek-VL's schedule: start with 100% text and decay linearly to the target share
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "c08-vlm-training"))
from vlmtrain.mixture import mixture_weights  # noqa: E402  (C8: proportional, uniform, temperature)


def capped_weights(sizes: dict[str, int], cap: int | None = None, strategy: str = "proportional") -> dict[str, float]:
    """Sampling probabilities after capping every source at `cap` samples."""
    eff = {k: min(n, cap) if cap else n for k, n in sizes.items()}
    if strategy == "sqrt":
        return mixture_weights(eff, "temperature", temperature=2.0)
    return mixture_weights(eff, strategy)


def token_shares(samples: dict[str, float], tokens_per_sample: dict[str, float]) -> dict[str, float]:
    """Share of supervised tokens each source contributes, given its share of samples."""
    raw = {k: samples[k] * tokens_per_sample[k] for k in samples}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def sft_loss(logits: torch.Tensor, labels: torch.Tensor, mode: str = "token") -> torch.Tensor:
    """Cross entropy over supervised tokens (label != -100) of a batch of responses.

    token   sum over all tokens / number of tokens. A sample's weight grows with its length L.
    sample  mean within each sample, then mean over samples. Every sample weighs the same.
    sqrt    each token weighted 1 / sqrt(L), then normalised by the sum of weights. A sample weighs sqrt(L):
            the middle ground InternVL 2.5 uses so that neither long nor short answers dominate.
    """
    per_tok = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
    mask = (labels != -100).float()
    lengths = mask.sum(1, keepdim=True).clamp(min=1)
    if mode == "token":
        w = mask
    elif mode == "sample":
        w = mask / lengths
    elif mode == "sqrt":
        w = mask / lengths.sqrt()
    else:
        raise ValueError(mode)
    return (per_tok * w).sum() / w.sum() if mode != "sample" else (per_tok * w).sum() / labels.shape[0]


def sample_weights(lengths: list[int], mode: str) -> list[float]:
    """The share of the total loss weight each sample receives, in closed form."""
    raw = {"token": [float(n) for n in lengths], "sample": [1.0] * len(lengths),
           "sqrt": [n ** 0.5 for n in lengths]}[mode]
    s = sum(raw)
    return [x / s for x in raw]


def modality_warmup(step: int, warmup_steps: int, target_text_share: float) -> float:
    """Text share of the batch at `step`: 1.0 at step 0, linear down to the target, then flat."""
    if step >= warmup_steps:
        return target_text_share
    return 1.0 - (1.0 - target_text_share) * step / warmup_steps
