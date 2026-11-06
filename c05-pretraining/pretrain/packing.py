"""Turning documents into fixed-length training sequences: padding versus packing.

Every function here works on document lengths (each length includes the EOS token), so
we can measure the real corpus without building any tensors. `pad_batch` and
`pack_stream` build the actual tensors for the small examples in the tutorial.
"""

from __future__ import annotations

import bisect
import math

import numpy as np
import torch

IGNORE = -100  # label value that F.cross_entropy skips


# ---------------------------------------------------------------- tensors, for small examples
def pad_batch(docs: list[list[int]], seq_len: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One document per row, truncated or right-padded to seq_len.

    Returns input_ids (B, L) and attention_mask (B, L) with 1 for real tokens.
    """
    ids = torch.full((len(docs), seq_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(docs), seq_len), dtype=torch.long)
    for i, d in enumerate(docs):
        d = d[:seq_len]
        ids[i, : len(d)] = torch.tensor(d)
        mask[i, : len(d)] = 1
    return ids, mask


def pack_stream(docs: list[list[int]], seq_len: int) -> torch.Tensor:
    """Concatenate documents (each already ends in EOS) and cut windows of seq_len + 1.

    Consecutive windows overlap by one token: window k holds tokens
    [k*L, k*L + L + 1), so its input is the first L and its targets are the last L.
    Returns (num_windows, L + 1). The tail that does not fill a window is dropped.
    """
    stream = torch.tensor([t for d in docs for t in d], dtype=torch.long)
    n = (len(stream) - 1) // seq_len
    idx = torch.arange(n)[:, None] * seq_len + torch.arange(seq_len + 1)[None, :]
    return stream[idx]


def document_ids(window: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Which document each position of a packed window belongs to. (L,) -> (L,).

    The EOS token closes its document, so the id increments on the token after it.
    """
    is_eos = (window == eos_id).long()
    return torch.cumsum(is_eos, 0) - is_eos


def position_ids(window: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Positions that restart at 0 at every document start inside a packed window."""
    doc = document_ids(window, eos_id)
    starts = torch.ones_like(doc)
    starts[1:] = (doc[1:] != doc[:-1]).long()
    first = torch.cummax(torch.where(starts.bool(), torch.arange(len(doc)), 0), 0).values
    return torch.arange(len(doc)) - first


def block_causal_mask(doc: torch.Tensor) -> torch.Tensor:
    """(L,) document ids -> (L, L) bool mask. True where query i may attend to key j:
    j <= i and both tokens are in the same document."""
    L = doc.numel()
    causal = torch.ones(L, L, dtype=torch.bool).tril()
    return causal & (doc[:, None] == doc[None, :])


# ---------------------------------------------------------------- utilisation on real lengths
def padding_utilization(doc_lens: np.ndarray, seq_len: int) -> dict:
    """One document per row, truncate at seq_len, pad the rest."""
    kept = np.minimum(doc_lens, seq_len)
    return {
        "utilization": kept.sum() / (len(doc_lens) * seq_len),
        "truncated_frac": 1 - kept.sum() / doc_lens.sum(),
        "rows": len(doc_lens),
    }


def dynamic_padding_utilization(
    doc_lens: np.ndarray, seq_len: int, batch_size: int, seed: int = 0
) -> dict:
    """Shuffle, batch, pad each batch only to its own longest document (capped at seq_len)."""
    lens = np.minimum(np.random.default_rng(seed).permutation(doc_lens), seq_len)
    n = len(lens) // batch_size * batch_size
    batches = lens[:n].reshape(-1, batch_size)
    slots = (batches.max(1) * batch_size).sum()
    return {"utilization": batches.sum() / slots, "truncated_frac": 1 - lens.sum() / doc_lens.sum()}


def greedy_packing_utilization(doc_lens: np.ndarray, seq_len: int) -> dict:
    """Concatenate everything and cut every seq_len tokens (what our loader does).

    `multi_doc_frac` is the share of windows that contain a document boundary,
    which is where tokens can attend across unrelated documents.
    """
    total = int(doc_lens.sum())
    windows = math.ceil(total / seq_len)
    ends = np.cumsum(doc_lens)
    starts = ends - doc_lens
    windows_with_boundary = len(np.unique(starts[1:] // seq_len))
    return {
        "utilization": total / (windows * seq_len),
        "truncated_frac": 0.0,
        "multi_doc_frac": windows_with_boundary / windows,
        "split_docs_frac": float(np.mean(starts // seq_len != (ends - 1) // seq_len)),
    }


def best_fit_packing_utilization(doc_lens: np.ndarray, seq_len: int) -> dict:
    """Best-fit decreasing bin packing: never split a document that fits in a window.

    Documents longer than seq_len are first cut into seq_len pieces. Each piece goes into
    the fullest window that still has room for it. This is the idea behind
    "Fewer Truncations Improve Language Modeling" (Ding et al., 2024).
    """
    pieces = []
    for n in doc_lens.tolist():
        pieces += [seq_len] * (n // seq_len) + ([n % seq_len] if n % seq_len else [])
    caps: list[int] = []  # remaining capacity of each open window, kept sorted
    bins = 0
    for n in sorted(pieces, reverse=True):
        i = bisect.bisect_left(caps, n)
        if i == len(caps):
            bins += 1
            rest = seq_len - n
        else:
            rest = caps.pop(i) - n
        if rest > 0:
            bisect.insort(caps, rest)
    return {
        "utilization": doc_lens.sum() / (bins * seq_len),
        "truncated_frac": 0.0,
        "split_docs_frac": float(np.mean(doc_lens > seq_len)),
    }
