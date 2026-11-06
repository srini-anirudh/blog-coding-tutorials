"""Prefix reuse on a real transformer: keep the KV cache of the last request.

`PrefixKV.forward(ids)` finds the longest common prefix with the previous request,
crops the stored cache to that length, and runs the model only on the new suffix.
If the new request extends the old one, only the new tokens are computed. If a token
near the front changed, almost everything is computed again. This is the mechanism
that makes the append-only rule pay.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .prefix_cache import first_divergence


@dataclass
class Step:
    tokens: int        # length of the request
    computed: int      # tokens run through the model
    seconds: float


class PrefixKV:
    def __init__(self, model, device: str = "cpu"):
        self.model, self.device = model, device
        self.ids: list[int] = []
        self.cache = None

    @torch.inference_mode()
    def forward(self, ids: list[int]) -> tuple[torch.Tensor, int]:
        """Last-position logits for `ids`, and how many tokens had to be computed."""
        from transformers import DynamicCache

        keep = min(first_divergence(self.ids, ids), len(ids) - 1)   # always compute at least one token
        if self.cache is None or keep == 0:
            cache, keep = DynamicCache(), 0
        else:
            cache = self.cache
            drop = cache.get_seq_length() - keep
            if drop > 0:
                cache.crop(-drop)          # transformers 5: a negative count removes tokens from the end
        x = torch.tensor([ids[keep:]], device=self.device)
        out = self.model(input_ids=x, past_key_values=cache, use_cache=True)
        self.ids, self.cache = list(ids), out.past_key_values
        return out.logits[0, -1].float(), len(ids) - keep


@torch.inference_mode()
def full_forward(model, ids: list[int], device: str = "cpu") -> torch.Tensor:
    """Last-position logits with no cache: the reference answer."""
    return model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0, -1].float()


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_lane(model, contexts: list[list[int]], lane: str, device: str = "cpu") -> list[Step]:
    """Serve a turn's requests in order. lane is "no reuse" or "reuse"."""
    kv = PrefixKV(model, device)
    steps = []
    for ids in contexts:
        _sync(device)
        t0 = time.perf_counter()
        if lane == "no reuse":
            full_forward(model, ids, device)
            computed = len(ids)
        elif lane == "reuse":
            _, computed = kv.forward(ids)
        else:
            raise ValueError(lane)
        _sync(device)
        steps.append(Step(len(ids), computed, time.perf_counter() - t0))
    return steps


def agent_contexts(prefix: list[int], chunks: list[list[int]], edit_at: int | None = None,
                   edit_pos: int = 64) -> list[list[int]]:
    """Request k is the prefix plus the first k chunks.

    With edit_at set, from request `edit_at` on, one token at `edit_pos` in the
    prefix is replaced: a state change written in place instead of appended.
    """
    out, hist = [], []
    for k, chunk in enumerate(chunks, 1):
        hist = hist + chunk
        pre = list(prefix)
        if edit_at is not None and k >= edit_at:
            pre[edit_pos] = (pre[edit_pos] + 1) % 1000 + 1
        out.append(pre + hist)
    return out
