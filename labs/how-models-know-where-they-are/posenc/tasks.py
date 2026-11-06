"""Synthetic data. Every batch function returns (tokens, targets, mask) and draws from a torch.Generator."""

from __future__ import annotations

from functools import partial

import torch


def position_batch(gen: torch.Generator, batch: int = 64, length: int = 64, vocab: int = 64):
    """Random tokens; the target at every slot is the slot's own index. Pure position, no content signal."""
    tokens = torch.randint(vocab, (batch, length), generator=gen)
    targets = torch.arange(length).expand(batch, length)
    return tokens, targets, torch.ones(batch, length, dtype=torch.bool)


def induction_batch(gen: torch.Generator, batch: int = 64, length: int = 64, vocab: int = 128, seg: int = 12):
    """Random tokens with one segment copied later in the sequence. Next-token targets.

    Only the copied positions are scored: at slot dst + j the model must find the earlier occurrence of the
    current token and emit the token that followed it. This needs content matching at any distance plus a
    precise +1 offset, the two things RoPE extension methods trade against each other.
    """
    tokens = torch.randint(vocab, (batch, length + 1), generator=gen)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    src = torch.randint(0, length + 1 - 2 * seg + 1, (batch,), generator=gen)
    for b in range(batch):
        s = int(src[b])
        d = int(torch.randint(s + seg, length + 1 - seg + 1, (1,), generator=gen))
        tokens[b, d : d + seg] = tokens[b, s : s + seg]
        mask[b, d : d + seg - 1] = True
    return tokens[:, :-1], tokens[:, 1:], mask


def make(kind: str, **kw):
    fn = {"position": position_batch, "induction": induction_batch}[kind]
    return partial(fn, **kw)
