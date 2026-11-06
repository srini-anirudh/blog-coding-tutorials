"""Stage 5: Group Relative Policy Optimization, written out by hand.

For each prompt we sample G completions, score them with an exact reward, and use the group itself as the
baseline:  A_i = (r_i - mean(r)) / (std(r) + eps).  The update is PPO's clipped surrogate on per-token ratios
pi / pi_old, plus a KL penalty to a frozen reference, averaged over completion tokens only.

tests/test_grpo.py feeds the same tensors to TRL's GRPOTrainer._compute_loss and checks the losses match.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def group_advantages(rewards: torch.Tensor, group_size: int, scale: str = "group", eps: float = 1e-4) -> torch.Tensor:
    """rewards: (P * G,) laid out prompt-major (all G samples of prompt 0 first). Returns (P * G,).

    scale="group": divide by the group's std (GRPO).  scale="none": only subtract the mean (Dr. GRPO).
    A group where every sample gets the same reward has advantage 0 everywhere: it teaches nothing.
    """
    r = rewards.view(-1, group_size)  # (P, G)
    adv = r - r.mean(dim=1, keepdim=True)
    if scale == "group":
        adv = adv / (r.std(dim=1, keepdim=True) + eps)  # unbiased std, as in TRL
    return adv.reshape(-1)


@dataclass
class GRPOLossConfig:
    clip_eps: float = 0.2
    beta: float = 0.04  # KL coefficient; 0 disables the reference model entirely
    loss_type: str = "grpo"  # "grpo": mean per sequence, then over sequences. "dapo": mean over all tokens.


def grpo_loss(
    logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    ref_logps: torch.Tensor | None = None,
    cfg: GRPOLossConfig = GRPOLossConfig(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped surrogate + KL, on completion tokens only.

    logps:      (N, C) log pi_theta of each sampled token, with grad.
    old_logps:  (N, C) log pi_old, the policy that sampled the data (no grad).
    advantages: (N,)   one scalar per completion, broadcast to its tokens.
    mask:       (N, C) 1 on real completion tokens (up to and including <|im_end|>), 0 on padding.
    ref_logps:  (N, C) log pi_ref of the frozen reference (no grad), needed when beta > 0.
    """
    mask = mask.to(logps.dtype)
    ratio = torch.exp(logps - old_logps)  # 1.0 on the first inner step, by construction
    adv = advantages.unsqueeze(1)  # (N, 1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
    per_token = -torch.min(unclipped, clipped)  # pessimistic bound: never profit from moving too far

    kl = torch.zeros_like(per_token)
    if cfg.beta > 0:
        assert ref_logps is not None, "beta > 0 needs reference log-probs"
        d = ref_logps - logps
        kl = torch.exp(d) - d - 1  # k3 estimator of KL(pi || ref): >= 0, unbiased, low variance
        per_token = per_token + cfg.beta * kl

    if cfg.loss_type == "grpo":
        loss = ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
    elif cfg.loss_type == "dapo":
        loss = (per_token * mask).sum() / mask.sum().clamp(min=1)
    else:
        raise ValueError(cfg.loss_type)

    with torch.no_grad():
        n = mask.sum().clamp(min=1)
        is_clipped = ((ratio < 1 - cfg.clip_eps) & (adv < 0)) | ((ratio > 1 + cfg.clip_eps) & (adv > 0))
        stats = {
            "clip_frac": float((is_clipped.to(mask.dtype) * mask).sum() / n),
            "kl": float((kl * mask).sum() / n),
            "ratio_mean": float((ratio * mask).sum() / n),
        }
    return loss, stats
