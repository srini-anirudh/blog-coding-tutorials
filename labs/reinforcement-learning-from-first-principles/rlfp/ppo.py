"""PPO's clipped surrogate, the entropy term and the sampled KL estimators.

    r_t = pi_theta(a_t | s_t) / pi_old(a_t | s_t)
    L_CLIP = E_t [ min( r_t A_t,  clip(r_t, 1 - eps, 1 + eps) A_t ) ]

The min makes the objective flat (zero gradient) in two of four regions: A > 0 with r > 1 + eps, and A < 0 with
r < 1 - eps. In the other two the gradient is active, including A < 0 with r > 1 + eps, so a bad token whose
probability rose can always be pulled back down.
"""

from __future__ import annotations

import torch


def clipped_surrogate(logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
                      eps: float = 0.2, clip: bool = True) -> tuple[torch.Tensor, dict[str, float]]:
    """Returns (loss to minimise, stats). logp (N, T) with grad; old_logp (N, T); adv (N,) or (N, T); mask (N, T).

    Token-level mean over all real tokens. `clip=False` gives the naive importance-weighted surrogate.
    """
    m = mask.to(logp.dtype)
    a = adv.unsqueeze(1) if adv.dim() == 1 else adv
    ratio = torch.exp(logp - old_logp.detach())
    surr = ratio * a
    if clip:
        surr = torch.min(surr, ratio.clamp(1 - eps, 1 + eps) * a)
    loss = -(surr * m).sum() / m.sum().clamp(min=1)
    with torch.no_grad():
        n = m.sum().clamp(min=1)
        clipped = ((ratio > 1 + eps) & (a > 0)) | ((ratio < 1 - eps) & (a < 0))
        outside = (ratio > 1 + eps) | (ratio < 1 - eps)
        stats = {
            "clip_frac": float((clipped.to(m.dtype) * m).sum() / n),
            "outside_frac": float((outside.to(m.dtype) * m).sum() / n),
            "ratio_max": float(ratio[m.bool()].max()) if m.any() else 1.0,
            "ratio_min": float(ratio[m.bool()].min()) if m.any() else 1.0,
        }
    return loss, stats


def clip_region_table(eps: float = 0.2) -> list[dict]:
    """Value and d/dr of min(r A, clip(r) A) at one ratio in each of the post's four regions, by autograd."""
    rows = []
    for A, r0 in [(1.0, 1.1), (1.0, 1.4), (-1.0, 0.9), (-1.0, 0.6), (-1.0, 1.4)]:
        r = torch.tensor(r0, requires_grad=True)
        obj = torch.min(r * A, r.clamp(1 - eps, 1 + eps) * A)
        obj.backward()
        rows.append({"A": A, "ratio": r0, "objective": round(obj.item(), 4), "d_obj_d_ratio": r.grad.item()})
    return rows


def token_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean entropy of pi(. | s_t) over real tokens. The bonus in L = -L_CLIP + c1 L_value - c2 H."""
    logp = torch.log_softmax(logits, dim=-1)
    H = -(logp.exp() * logp).sum(-1)
    m = mask.to(H.dtype)
    return (H * m).sum() / m.sum().clamp(min=1)


def kl_estimators(logp: torch.Tensor, ref_logp: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-token samples of KL(pi || ref) from tokens drawn from pi (Schulman's k1, k2, k3).

    k1 = log(pi/ref)              unbiased, can be negative on a single token
    k2 = 0.5 * log(pi/ref)^2      biased, always >= 0
    k3 = (ref/pi) - 1 - log(ref/pi)  unbiased, always >= 0
    """
    d = ref_logp - logp
    return {"k1": -d, "k2": 0.5 * d.pow(2), "k3": d.exp() - 1 - d}
