"""Sections 11 and 12: the RL step in shapes, masks, loss normalisation, and what is resident in memory."""

from __future__ import annotations

import torch


def response_mask(actions: torch.Tensor, eos: int) -> torch.Tensor:
    """1 on response tokens up to and including the first EOS, 0 after it: (B, T)."""
    is_eos = (actions == eos).int()
    after = (is_eos.cumsum(1) - is_eos) > 0  # strictly after the first EOS
    return (~after).float()


def masked_loss(per_token: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """seq: mean within each sequence, then over sequences.  token: one mean over every real token in the batch."""
    m = mask.to(per_token.dtype)
    if mode == "seq":
        return ((per_token * m).sum(1) / m.sum(1).clamp(min=1)).mean()
    if mode == "token":
        return (per_token * m).sum() / m.sum().clamp(min=1)
    raise ValueError(mode)


def per_token_weight(lengths: list[int], mode: str) -> list[float]:
    """The weight one token of each sequence gets in the loss, from the gradient of `masked_loss`."""
    T = max(lengths)
    mask = torch.tensor([[1.0] * n + [0.0] * (T - n) for n in lengths])
    x = torch.zeros_like(mask, requires_grad=True)
    masked_loss(x, mask, mode).backward()
    return [float(x.grad[i, 0]) for i in range(len(lengths))]


def chunked_token_logprobs(hidden: torch.Tensor, head: torch.nn.Linear, targets: torch.Tensor,
                           chunk: int = 256) -> torch.Tensor:
    """log softmax(head(hidden))[target] without materialising (B, T, |V|) logits at once: (B, T).

    Only (B, chunk, |V|) logits exist at any time. The result is identical to the full computation.
    """
    out = []
    for s in range(0, hidden.shape[1], chunk):
        logits = head(hidden[:, s:s + chunk]).float()
        lp = logits.log_softmax(-1).gather(-1, targets[:, s:s + chunk].unsqueeze(-1)).squeeze(-1)
        out.append(lp)
    return torch.cat(out, dim=1)


def logits_bytes(n_seq: int, T: int, vocab: int, bytes_per: int = 2) -> int:
    return n_seq * T * vocab * bytes_per


# Bytes per parameter: bf16 weights + bf16 grads + fp32 master weights, momentum and variance = 2 + 2 + 12 = 16.
TRAINABLE_BYTES, FROZEN_BYTES = 16, 2


def resident_gb(n_params: float, components: dict[str, bool]) -> dict[str, float]:
    """Weights and optimiser state only (no activations, no KV cache), in GB, for the post's four-model table."""
    sizes = {"policy": TRAINABLE_BYTES, "critic": TRAINABLE_BYTES, "reference": FROZEN_BYTES,
             "reward_model": FROZEN_BYTES, "inference_copy": FROZEN_BYTES}
    out = {k: n_params * b / 1e9 for k, b in sizes.items() if components.get(k, False)}
    out["total"] = sum(out.values())
    return out
