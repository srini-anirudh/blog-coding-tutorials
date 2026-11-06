"""Padding vs packing for SFT batches.

Padding: one conversation per row, short rows filled with pad tokens. Simple, but the pad tokens cost compute.
Packing: several conversations share one row. It wastes almost nothing, but it is only safe if
(1) position ids restart at 0 for each conversation and (2) attention cannot cross conversation boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .chat import IGNORE_INDEX, ChatExample


def round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def pad_batch(examples: list[ChatExample], pad_id: int, multiple: int = 32) -> dict[str, torch.Tensor]:
    """Right-pad to the longest example, rounded up to `multiple`.

    Rounding matters on MPS: every new sequence length compiles new kernels, which costs seconds.
    Returns input_ids, labels, attention_mask, each (B, L) with L = round_up(L_max, multiple).
    """
    L = round_up(max(len(e.input_ids) for e in examples), multiple)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), L), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((len(examples), L), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e.input_ids)
        ids[i, :n] = torch.tensor(e.input_ids)
        labels[i, :n] = torch.tensor(e.labels)
        attn[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": attn}


@dataclass
class PackedRow:
    input_ids: list[int]
    labels: list[int]
    position_ids: list[int]
    seq_ids: list[int]  # which conversation each token belongs to; -1 marks trailing padding


def pack_examples(examples: list[ChatExample], max_len: int) -> list[PackedRow]:
    """First-fit-decreasing bin packing of whole conversations into rows of at most `max_len` tokens.

    Conversations are never split: a split conversation would train its second half without its prompt.
    """
    order = sorted(range(len(examples)), key=lambda i: -len(examples[i].input_ids))
    bins: list[list[int]] = []
    space: list[int] = []
    for i in order:
        n = len(examples[i].input_ids)
        if n > max_len:
            raise ValueError(f"example {i} has {n} tokens > max_len={max_len}")
        for b, free in enumerate(space):
            if n <= free:
                bins[b].append(i)
                space[b] -= n
                break
        else:
            bins.append([i])
            space.append(max_len - n)

    rows = []
    for members in bins:
        row = PackedRow([], [], [], [])
        for k, i in enumerate(members):
            e = examples[i]
            row.input_ids += e.input_ids
            # The first token of a conversation must not be predicted from the previous conversation.
            row.labels += [IGNORE_INDEX] + e.labels[1:]
            row.position_ids += list(range(len(e.input_ids)))
            row.seq_ids += [k] * len(e.input_ids)
        rows.append(row)
    return rows


def block_causal_mask(seq_ids: torch.Tensor) -> torch.Tensor:
    """(B, L) segment ids -> (B, 1, L, L) bool mask, True where query i may attend to key j.

    Allowed iff j <= i (causal) and both tokens are in the same conversation. Padding (-1) attends to itself
    only, so no row of the mask is all False (an all-False row makes softmax produce NaN).
    """
    L = seq_ids.shape[1]
    causal = torch.ones(L, L, dtype=torch.bool, device=seq_ids.device).tril()
    same = seq_ids[:, :, None] == seq_ids[:, None, :]
    pad_self = torch.eye(L, dtype=torch.bool, device=seq_ids.device) & (seq_ids[:, :, None] < 0)
    return ((causal & same) | pad_self)[:, None]


def collate_packed(rows: list[PackedRow], max_len: int, pad_id: int) -> dict[str, torch.Tensor]:
    """Stack packed rows into tensors.

    input_ids, labels, position_ids, seq_ids: (B, max_len).  attention_mask: (B, 1, max_len, max_len) bool.
    """
    B = len(rows)
    ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
    pos = torch.zeros((B, max_len), dtype=torch.long)
    seq = torch.full((B, max_len), -1, dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r.input_ids)
        ids[i, :n] = torch.tensor(r.input_ids)
        labels[i, :n] = torch.tensor(r.labels)
        pos[i, :n] = torch.tensor(r.position_ids)
        pos[i, n:] = torch.arange(max_len - n)
        seq[i, :n] = torch.tensor(r.seq_ids)
    return {
        "input_ids": ids,
        "labels": labels,
        "position_ids": pos,
        "seq_ids": seq,
        "attention_mask": block_causal_mask(seq),
    }


def padding_stats(lengths: list[int], batch_size: int, max_len: int) -> dict[str, float]:
    """Fraction of tokens that are real (not padding) when padding vs when packing the same examples."""
    real = sum(lengths)
    padded = 0
    for s in range(0, len(lengths), batch_size):
        chunk = lengths[s : s + batch_size]
        padded += round_up(max(chunk), 32) * len(chunk)  # what pad_batch actually allocates
    fake = [ChatExample([0] * n, [0] * n, [""] * n) for n in lengths]
    n_rows = len(pack_examples(fake, max_len))
    return {
        "real_tokens": real,
        "padded_slots": padded,
        "packed_slots": n_rows * max_len,
        "pad_efficiency": real / padded,
        "pack_efficiency": real / (n_rows * max_len),
        "packed_rows": n_rows,
    }
