"""The small formulas the post writes down, as functions."""
from __future__ import annotations

import torch


def multimodal_gain(s_v: float, s_wv: float) -> float:
    """MG = S_v - S_wv (MMStar, Chen et al., 2024): what the image adds."""
    return s_v - s_wv


def multimodal_leakage(s_wv: float, s_t: float) -> float:
    """ML = max(0, S_wv - S_t): how much better the VLM does blind than its own language model."""
    return max(0.0, s_wv - s_t)


def restoration(p_patched: float, p_corrupt: float, p_clean: float) -> float:
    """(P_patched - P_corrupt) / (P_clean - P_corrupt). Near 1: this activation alone restores the answer."""
    return (p_patched - p_corrupt) / (p_clean - p_corrupt)


def relative_change(p_new: float, p_old: float) -> float:
    """Relative change in the probability of the correct answer, as Zhang et al. (2024) report knockout."""
    return (p_new - p_old) / p_old


def choice(logits: torch.Tensor, candidate_ids: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Forced choice: index of the best candidate, and the distribution over candidates."""
    probs = torch.softmax(logits[candidate_ids], -1)
    return int(probs.argmax()), probs


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """Half the L1 distance between two answer distributions: 0 means the image changed nothing."""
    return 0.5 * float((p - q).abs().sum())
