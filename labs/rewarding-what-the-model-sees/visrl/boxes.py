"""A box policy trained with GRPO: does the reward make boxes grow?

The post argues that a soft IoU reward pays an oversized box more than an honest
miss, and that under group-relative advantages this pulls the policy toward
bigger boxes on hard targets. GUI-G1 measured that growth. This file builds the
smallest policy where the effect can happen.

    target box (centre c, size w x h) in the unit image
    rollout k: centre   = c + sigma_d * eps_k        sigma_d: how well the model can locate (fixed)
               log size = m_d + tau * xi_k           m_d: learned, one scalar per difficulty d
    box_k = centre +- exp(log size) * (w, h) / 2, clipped to the image

Only the size is learned, so any change in box size comes from the reward alone.
A ratio exp(2 m_d) > 1 means the policy outputs boxes larger than the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .grpo import group_advantages, zero_advantage

DIFFICULTIES = {"easy": 0.01, "hard": 0.15}   # location noise sigma_d, in image widths


# ---------- vectorised rewards on (..., 4) tensors of (x0, y0, x1, y1) ----------

def _area(b):
    return (b[..., 2] - b[..., 0]).clamp_min(0) * (b[..., 3] - b[..., 1]).clamp_min(0)


def _inter(a, b):
    w = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp_min(0)
    h = (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp_min(0)
    return w * h


def iou_t(pred, gold):
    inter = _inter(pred, gold)
    return inter / (_area(pred) + _area(gold) - inter).clamp_min(1e-12)


def giou_t(pred, gold):
    inter = _inter(pred, gold)
    union = (_area(pred) + _area(gold) - inter).clamp_min(1e-12)
    c = torch.stack([torch.minimum(pred[..., 0], gold[..., 0]), torch.minimum(pred[..., 1], gold[..., 1]),
                     torch.maximum(pred[..., 2], gold[..., 2]), torch.maximum(pred[..., 3], gold[..., 3])], -1)
    area_c = _area(c).clamp_min(1e-12)
    return inter / union - (area_c - union) / area_c


def coverage_t(pred, gold):
    return _inter(pred, gold) / _area(gold).clamp_min(1e-12)


def hit_t(pred, gold):
    cx, cy = (pred[..., 0] + pred[..., 2]) / 2, (pred[..., 1] + pred[..., 3]) / 2
    return ((gold[..., 0] <= cx) & (cx <= gold[..., 2]) & (gold[..., 1] <= cy) & (cy <= gold[..., 3])).float()


def size_match_t(pred, gold):
    a, b = _area(pred), _area(gold)
    return torch.minimum(a, b) / torch.maximum(a, b).clamp_min(1e-12)


REWARDS = {
    "iou": iou_t,
    "iou>=0.5": lambda p, g: (iou_t(p, g) >= 0.5).float(),
    "giou": giou_t,
    "coverage": coverage_t,
    "hit": hit_t,
    "iou+size": lambda p, g: 0.5 * iou_t(p, g) + 0.5 * size_match_t(p, g),
}


# ---------- data and policy ----------

def sample_targets(n: int, g: torch.Generator) -> torch.Tensor:
    """n target boxes: sides in [0.15, 0.35], centres in [0.25, 0.75]."""
    wh = 0.15 + 0.2 * torch.rand(n, 2, generator=g)
    c = 0.25 + 0.5 * torch.rand(n, 2, generator=g)
    return torch.cat([c - wh / 2, c + wh / 2], -1)


def make_boxes(gold: torch.Tensor, log_size: torch.Tensor, sigma: float, g: torch.Generator) -> torch.Tensor:
    """gold (P, 4), log_size (P, K) -> boxes (P, K, 4): noisy centre, scaled size, clipped to [0, 1]."""
    c = (gold[:, None, :2] + gold[:, None, 2:]) / 2
    c = c + sigma * torch.randn(*log_size.shape, 2, generator=g)
    half = torch.exp(log_size)[..., None] * (gold[:, None, 2:] - gold[:, None, :2]) / 2
    return torch.cat([c - half, c + half], -1).clamp(0, 1)


@dataclass
class BoxRun:
    reward: str
    area_ratio: dict[str, list[float]]     # per difficulty, exp(2 m_d) after each step
    useful: dict[str, list[float]]         # per difficulty, share of groups with non-zero advantage
    final_iou: dict[str, float]            # mean IoU of the policy's own samples at the end


def train_box_policy(reward: str, steps: int = 300, prompts: int = 64, k: int = 8, tau: float = 0.25,
                     lr: float = 0.05, seed: int = 0, scale_by_std: bool = True) -> BoxRun:
    """GRPO on the log-size of the box, one learned mean per difficulty, starting at the target size."""
    g = torch.Generator().manual_seed(seed)
    fn = REWARDS[reward]
    m = {d: torch.zeros(1, requires_grad=True) for d in DIFFICULTIES}
    opt = torch.optim.Adam(list(m.values()), lr=lr)
    ratio = {d: [] for d in DIFFICULTIES}
    useful = {d: [] for d in DIFFICULTIES}
    for _ in range(steps):
        opt.zero_grad()
        for d, sigma in DIFFICULTIES.items():
            gold = sample_targets(prompts, g)
            z = (m[d].detach() + tau * torch.randn(prompts, k, generator=g))     # sampled log sizes
            r = fn(make_boxes(gold, z, sigma, g), gold[:, None, :])              # (P, K)
            adv = group_advantages(r, scale_by_std=scale_by_std)
            logp = -((z - m[d]) ** 2) / (2 * tau ** 2)                           # Gaussian log-prob, up to a constant
            (-(adv * logp).mean()).backward()
            useful[d].append(float((~zero_advantage(r)).float().mean()))
        opt.step()
        for d in DIFFICULTIES:
            ratio[d].append(float(torch.exp(2 * m[d].detach())))
    final = {}
    with torch.no_grad():
        for d, sigma in DIFFICULTIES.items():
            gold = sample_targets(2000, g)
            z = m[d] + tau * torch.randn(2000, 1, generator=g)
            final[d] = float(iou_t(make_boxes(gold, z, sigma, g), gold[:, None, :]).mean())
    return BoxRun(reward, ratio, useful, final)
