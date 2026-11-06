"""Multimodal packing: fill fixed token budgets with several samples instead of padding.

Padding wastes compute on tokens that carry nothing. In a VLM the waste is large, because lengths
vary a lot: a chart costs 49 visual tokens, a document with 2 x 2 tiles plus a thumbnail costs 245.

Packing concatenates samples into rows of at most `budget` tokens. Three things must then hold so that
a packed row computes exactly what the separate samples would:

    1. attention is block-diagonal causal: a token sees only earlier tokens of its own sample
    2. position ids restart at 0 for every sample
    3. the first token of each sample is not a target for the last token of the previous one

Visual tokens stay correct for free: pixel tiles are concatenated in the same order as the <image>
slots, and the model scatters them in order.
"""

from __future__ import annotations

import torch

from .processor import IGNORE, Encoded


def first_fit_decreasing(lengths: list[int], budget: int) -> list[list[int]]:
    """Bin packing: longest samples first, each into the first row with room. Returns index lists."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    rows: list[list[int]] = []
    room: list[int] = []
    for i in order:
        if lengths[i] > budget:
            raise ValueError(f"sample {i} has {lengths[i]} tokens > budget {budget}")
        for r, free in enumerate(room):
            if lengths[i] <= free:
                rows[r].append(i)
                room[r] -= lengths[i]
                break
        else:
            rows.append([i])
            room.append(budget - lengths[i])
    return rows


def block_causal_mask(seg_ids: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """(B, L) segment ids (-1 = padding) -> additive mask (B, 1, L, L): 0 where allowed, min where blocked."""
    L = seg_ids.shape[1]
    same = seg_ids.unsqueeze(2) == seg_ids.unsqueeze(1)                    # (B, L, L) same sample
    causal = torch.ones(L, L, dtype=torch.bool, device=seg_ids.device).tril()
    real = (seg_ids >= 0).unsqueeze(1)                                     # keys that are not padding
    allowed = same & causal & real
    # A padding query row would be all-blocked, which makes softmax NaN; let it see itself.
    allowed |= torch.eye(L, dtype=torch.bool, device=seg_ids.device)
    mask = torch.zeros(allowed.shape, dtype=dtype, device=seg_ids.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min).unsqueeze(1)


def pack(items: list[Encoded], budget: int, pad_id: int) -> dict:
    """Pack encoded samples into rows of exactly `budget` tokens."""
    rows = first_fit_decreasing([len(e.input_ids) for e in items], budget)
    B = len(rows)
    input_ids = torch.full((B, budget), pad_id, dtype=torch.long)
    labels = torch.full((B, budget), IGNORE, dtype=torch.long)
    position_ids = torch.zeros((B, budget), dtype=torch.long)
    seg_ids = torch.full((B, budget), -1, dtype=torch.long)
    pix, tasks, meta = [], [], []
    for b, row in enumerate(rows):
        pos = 0
        for s, i in enumerate(row):
            e = items[i]
            L = len(e.input_ids)
            input_ids[b, pos:pos + L] = torch.tensor(e.input_ids)
            lab = torch.tensor(e.labels)
            lab[0] = IGNORE  # rule 3: never predict across a sample boundary
            labels[b, pos:pos + L] = lab
            position_ids[b, pos:pos + L] = torch.arange(L)
            seg_ids[b, pos:pos + L] = s
            if e.pixel_values is not None:
                pix.append(e.pixel_values)
            tasks.append(e.task)
            meta.append(e.meta)
            pos += L
    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": position_ids,
        "seg_ids": seg_ids,
        "attention_mask": block_causal_mask(seg_ids),
        "pixel_values": torch.cat(pix) if pix else None,
        "rows": rows,
        "tasks": tasks,
        "meta": meta,
    }


def padding_fraction(lengths: list[int], batch_size: int, sort: bool = False) -> float:
    """Share of positions that are padding when batching `batch_size` samples to their longest."""
    lens = sorted(lengths) if sort else list(lengths)
    pad = total = 0
    for i in range(0, len(lens), batch_size):
        chunk = lens[i:i + batch_size]
        total += max(chunk) * len(chunk)
        pad += max(chunk) * len(chunk) - sum(chunk)
    return pad / total
