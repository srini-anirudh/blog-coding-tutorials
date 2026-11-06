"""Fitting a video into the budget: spatial pooling, slow-fast allocation, a merging memory bank, and
patch-level pruning of unchanged regions.

These work on pixels so their losses can be measured directly. Real models do the same on encoder features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def pool_to_grid(frames: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    """Average-pool each frame to a gh x gw grid: one token per cell. (T, C, H, W) -> (T, C, gh, gw)."""
    return F.adaptive_avg_pool2d(frames, (gh, gw))


def reconstruct(tokens: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Nearest upsampling back to pixels, to see what detail a grid kept."""
    return F.interpolate(tokens, size=size, mode="nearest")


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(1.0 / mse)).item()


@dataclass
class Allocation:
    name: str
    views: list[tuple[int, int, int]]  # (frame index, grid h, grid w)

    @property
    def tokens(self) -> int:
        return sum(h * w for _, h, w in self.views)

    @property
    def frames(self) -> list[int]:
        return sorted({i for i, _, _ in self.views})


def equal_allocation(n_frames: int, grid: tuple[int, int]) -> Allocation:
    return Allocation(f"equal {n_frames} x {grid[0]}x{grid[1]}", [(i, *grid) for i in range(n_frames)])


def slowfast_allocation(n_frames: int, fast_grid: tuple[int, int], slow_every: int, slow_grid: tuple[int, int]):
    """Every frame gets a coarse fast view; every `slow_every`-th frame also gets a detailed slow view."""
    views = [(i, *fast_grid) for i in range(n_frames)]
    views += [(i, *slow_grid) for i in range(0, n_frames, slow_every)]
    return Allocation(f"slow-fast: {n_frames} x {fast_grid[0]}x{fast_grid[1]} + every {slow_every}th at "
                      f"{slow_grid[0]}x{slow_grid[1]}", views)


def best_contrast_kept(frames: torch.Tensor, alloc: Allocation, left, right) -> float:
    """How much of a small left-right contrast survives in the most detailed view of any frame.

    left(i) and right(i) give (channel, row, column) slices of the two halves of the detail in frame i.
    1.0 = the contrast is fully kept; 0.0 = pooling averaged it away.
    """
    best = 0.0
    H, W = frames.shape[-2:]
    for i, gh, gw in alloc.views:
        rec = reconstruct(pool_to_grid(frames[i:i + 1], gh, gw), (H, W))[0]
        true = float(frames[i][right(i)].mean() - frames[i][left(i)].mean())
        kept = float(rec[right(i)].mean() - rec[left(i)].mean())
        best = max(best, kept / true)
    return best


# ---- merging memory bank (MA-LMM style) ---------------------------------------------------------------

class MemoryBank:
    """A fixed number of slots. When full, average the two most similar adjacent slots into one.

    Each slot remembers the time span it covers, so the blur in timing can be read out.
    """

    def __init__(self, size: int):
        self.size = size
        self.feats: list[torch.Tensor] = []
        self.spans: list[tuple[float, float]] = []
        self.counts: list[int] = []

    def add(self, feat: torch.Tensor, t: float) -> None:
        self.feats.append(feat.flatten().float())
        self.spans.append((t, t))
        self.counts.append(1)
        if len(self.feats) > self.size:
            sims = [float(F.cosine_similarity(self.feats[i], self.feats[i + 1], dim=0))
                    for i in range(len(self.feats) - 1)]
            i = max(range(len(sims)), key=sims.__getitem__)
            n1, n2 = self.counts[i], self.counts[i + 1]
            self.feats[i] = (self.feats[i] * n1 + self.feats[i + 1] * n2) / (n1 + n2)
            self.spans[i] = (self.spans[i][0], self.spans[i + 1][1])
            self.counts[i] = n1 + n2
            del self.feats[i + 1], self.spans[i + 1], self.counts[i + 1]


# ---- differential frame pruning (VideoLLaMA 3 idea) ----------------------------------------------------

def changed_patches(frames: torch.Tensor, patch: int, threshold: float) -> torch.Tensor:
    """Keep mask (T, H/p, W/p): a patch survives if its mean absolute pixel change since the previous frame
    exceeds the threshold. The first frame keeps every patch."""
    diff = (frames[1:] - frames[:-1]).abs().mean(1, keepdim=True)
    d = F.avg_pool2d(diff, patch)[:, 0]
    keep = d > threshold
    first = torch.ones(1, *keep.shape[1:], dtype=torch.bool)
    return torch.cat([first, keep])
