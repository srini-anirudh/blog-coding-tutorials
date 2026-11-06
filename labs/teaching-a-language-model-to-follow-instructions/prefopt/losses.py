"""The offline preference family as losses on one margin axis, plus the DPO loss with its two common patches.

`h` is the log-ratio margin  [log pi(y_w) - log ref(y_w)] - [log pi(y_l) - log ref(y_l)]  for the methods that use
a reference, and the length-normalised log-prob margin for SimPO. Drawing all of them against the same h, as the
post's Figure 9 does, makes the differences visible in the gradient, not in the loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo(h: torch.Tensor, beta: float) -> torch.Tensor:
    """Rafailov et al. 2023: logistic on beta * h."""
    return -F.logsigmoid(beta * h)


def ipo(h: torch.Tensor, beta: float) -> torch.Tensor:
    """Azar et al. 2023: regress h onto 1 / (2 tau), with tau = beta. The gradient never vanishes."""
    return (h - 1.0 / (2.0 * beta)) ** 2


def slic(h: torch.Tensor, delta: float) -> torch.Tensor:
    """Zhao et al. 2023 (SLiC-HF, calibration term): hinge with margin delta. Exactly zero once h > delta."""
    return torch.clamp(delta - h, min=0.0)


def rrhf(h: torch.Tensor) -> torch.Tensor:
    """Yuan et al. 2023 (ranking term): hinge at zero, so it stops as soon as the pair is ordered."""
    return torch.clamp(-h, min=0.0)


def simpo(h: torch.Tensor, beta: float, gamma: float) -> torch.Tensor:
    """Meng et al. 2024: logistic with a target margin gamma. h is a length-normalised log-prob margin."""
    return -F.logsigmoid(beta * h - gamma)


def kto_desirable(h: torch.Tensor, beta: float, z0: float = 0.0) -> torch.Tensor:
    """Ethayarajh et al. 2024, the desirable branch: 1 - sigmoid(beta (r - z0)). Pointwise, not pairwise.

    Here r is the chosen response's own log-ratio, drawn on the same axis; z0 is the batch KL reference point.
    """
    return 1.0 - torch.sigmoid(beta * (h - z0))


def orpo_ratio(h: torch.Tensor) -> torch.Tensor:
    """Hong et al. 2024, the odds-ratio term: -log sigmoid(h), where h is a log-odds margin, log p/(1-p).

    ORPO adds this (times lambda) to the SFT loss and uses no reference. On a common axis it is DPO with beta = 1.
    """
    return -F.logsigmoid(h)


FAMILY = {
    "DPO": lambda h, beta, target: dpo(h, beta),
    "IPO": lambda h, beta, target: ipo(h, beta),
    "SLiC hinge": lambda h, beta, target: slic(h, target),
    "RRHF hinge": lambda h, beta, target: rrhf(h),
    "SimPO": lambda h, beta, target: simpo(h, beta, beta * target),
    "KTO (desirable)": lambda h, beta, target: kto_desirable(h, beta),
    "ORPO odds term": lambda h, beta, target: orpo_ratio(h),
}


def loss_and_grad(name: str, h: torch.Tensor, beta: float = 1.0, target: float = 1.6):
    """Loss and dLoss/dh at every point of h, by autograd."""
    x = h.detach().clone().double().requires_grad_(True)
    loss = FAMILY[name](x, beta, target)
    (g,) = torch.autograd.grad(loss.sum(), x)
    return loss.detach(), g


def dpo_sequence_loss(pol_w, pol_l, ref_w, ref_l, beta: float, dpop_lambda: float = 0.0, nll_alpha: float = 0.0,
                      len_w=None):
    """DPO on summed sequence log-probs (B,), with two optional patches for likelihood displacement:

    dpop_lambda > 0: DPO-Positive (Pal et al. 2024) adds lambda * max(0, log ref(y_w) - log pi(y_w)) inside the
        sigmoid, so the chosen response is penalised as soon as it falls below the reference.
    nll_alpha > 0: an RPO-style negative log-likelihood term on the chosen response (per token if len_w is given).
    """
    margin = (pol_w - ref_w) - (pol_l - ref_l)
    if dpop_lambda:
        margin = margin - dpop_lambda * torch.clamp(ref_w - pol_w, min=0.0)
    loss = -F.logsigmoid(beta * margin).mean()
    if nll_alpha:
        nll = -pol_w if len_w is None else -pol_w / len_w
        loss = loss + nll_alpha * nll.mean()
    return loss
