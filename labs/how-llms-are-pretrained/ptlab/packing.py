"""Packed sequences: does a document in a packed row see the same thing it would see alone?

The helpers that build document ids, reset positions and the block-diagonal mask come from
C5 (`c05-pretraining/pretrain/packing.py`). This module adds the check the post describes:
run a real pretrained model on a packed row under four settings and compare each document's
per-token loss with the loss of the same document run on its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_C5 = Path(__file__).resolve().parents[3] / "c05-pretraining"
if str(_C5) not in sys.path:
    sys.path.insert(0, str(_C5))

from pretrain.packing import block_causal_mask, document_ids, position_ids  # noqa: E402

SETTINGS = {
    "causal mask, positions keep counting": (False, False),
    "causal mask, positions reset": (False, True),
    "document mask, positions keep counting": (True, False),
    "document mask, positions reset": (True, True),
}


def per_token_nll(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """(L, V) logits and (L,) ids -> (L-1,) NLL of token t+1 given tokens up to t."""
    return F.cross_entropy(logits[:-1].float(), ids[1:], reduction="none")


@torch.no_grad()
def alone_losses(model, docs: list[list[int]]) -> list[torch.Tensor]:
    """Each document run by itself, positions from 0."""
    out = []
    for d in docs:
        ids = torch.tensor(d)
        out.append(per_token_nll(model(input_ids=ids[None]).logits[0], ids))
    return out


@torch.no_grad()
def packed_losses(model, docs: list[list[int]], eos_id: int, doc_mask: bool, reset_pos: bool) -> list[torch.Tensor]:
    """All documents in one row. Each document already ends in EOS.

    Returns, for each document, the NLL of its own tokens 2..n given its earlier tokens
    (the first token of a document has no in-document context, so it is left out, as in `alone_losses`).
    """
    ids = torch.tensor([t for d in docs for t in d])
    L = len(ids)
    doc = document_ids(ids, eos_id)
    pos = position_ids(ids, eos_id) if reset_pos else torch.arange(L)
    allowed = block_causal_mask(doc) if doc_mask else torch.ones(L, L, dtype=torch.bool).tril()
    mask4d = torch.zeros(1, 1, L, L, dtype=torch.float32).masked_fill(~allowed, torch.finfo(torch.float32).min)
    logits = model(input_ids=ids[None], attention_mask=mask4d, position_ids=pos[None]).logits[0]
    nll = per_token_nll(logits, ids)
    out, start = [], 0
    for d in docs:
        out.append(nll[start : start + len(d) - 1])
        start += len(d)
    return out


def compare(model, docs: list[list[int]], eos_id: int) -> dict[str, list[float]]:
    """Largest per-token difference from the alone run, for each document after the first."""
    ref = alone_losses(model, docs)
    res = {}
    for name, (dm, rp) in SETTINGS.items():
        got = packed_losses(model, docs, eos_id, dm, rp)
        res[name] = [float((g - r).abs().max()) for g, r in zip(got[1:], ref[1:])]
    return res
