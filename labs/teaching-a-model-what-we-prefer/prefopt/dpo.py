"""DPO on policies small enough to inspect: a softmax over a finite set of whole responses.

    implicit reward   r_hat(y) = beta * log( pi(y) / ref(y) )
    loss              -log sigmoid( r_hat(y_w) - r_hat(y_l) )
    gradient          -beta * sigmoid(r_hat_l - r_hat_w) * ( grad log pi(y_w) - grad log pi(y_l) )

A "response" here is one item of a finite set, so log pi(y) is one entry of a log-softmax. The language-model
version, with log pi(y | x) summed over tokens, is in C6 (c06-post-training/posttrain/dpo.py) and in
scripts/08_displacement_lm.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def implicit_reward(logp: torch.Tensor, ref_logp: torch.Tensor, beta: float) -> torch.Tensor:
    return beta * (logp - ref_logp)


def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta: float, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Mean DPO loss over pairs. `weight` (optional) turns hard labels into soft or counted ones."""
    margin = implicit_reward(logp_w, ref_w, beta) - implicit_reward(logp_l, ref_l, beta)
    losses = -F.logsigmoid(margin)
    return losses.mean() if weight is None else (weight * losses).sum() / weight.sum()


def dpo_weight(logp_w, logp_l, ref_w, ref_l, beta: float) -> torch.Tensor:
    """The self-annealing scalar sigmoid(r_hat_l - r_hat_w): near 1 when the pair is ranked backwards."""
    return torch.sigmoid(implicit_reward(logp_l, ref_l, beta) - implicit_reward(logp_w, ref_w, beta))


class SoftmaxPolicy(nn.Module):
    """pi(y) = softmax(base + E h)[y] over a finite set of responses.

    E (V, d) are fixed response embeddings and h (d,) is trained. E = identity gives a tabular policy, where every
    response has its own free logit. A low-dimensional E makes responses share parameters, as tokens in a real model
    share the final hidden state, so pushing one response moves its neighbours.
    """

    def __init__(self, base_logits: torch.Tensor, emb: torch.Tensor | None = None):
        super().__init__()
        v = base_logits.numel()
        self.register_buffer("base", base_logits.double())
        self.register_buffer("emb", torch.eye(v, dtype=torch.float64) if emb is None else emb.double())
        self.h = nn.Parameter(torch.zeros(self.emb.shape[1], dtype=torch.float64))

    def log_probs(self) -> torch.Tensor:
        return torch.log_softmax(self.base + self.emb @ self.h, -1)


def train_dpo(policy: SoftmaxPolicy, winners, losers, beta: float, steps: int, lr: float,
              weight: torch.Tensor | None = None, variant: str = "dpo", lam: float = 1.0, log_every: int = 0,
              opt: str = "adam", ref: torch.Tensor | None = None):
    """Full-batch DPO on index pairs. variant: "dpo", "dpop" (DPO-Positive) or "rpo" (DPO + NLL on the winner).

    DPO-Positive adds lam * max(0, log ref(y_w) - log pi(y_w)) inside the sigmoid's argument, so the loss rises
    whenever the winner falls below the reference. RPO adds lam * (-log pi(y_w)) to the loss.
    `ref` is the reference's log-probs; by default, the policy as it is before this call.
    Returns a list of per-step logs if log_every > 0.
    """
    ref = policy.log_probs().detach() if ref is None else ref
    optim = (torch.optim.Adam if opt == "adam" else torch.optim.SGD)(policy.parameters(), lr=lr)
    logs = []
    for step in range(steps + 1):
        lp = policy.log_probs()
        lw, ll = lp[winners], lp[losers]
        margin = beta * ((lw - ref[winners]) - (ll - ref[losers]))
        if variant == "dpop":
            margin = margin - beta * lam * torch.relu(ref[winners] - lw)
        losses = -F.logsigmoid(margin)
        loss = losses.mean() if weight is None else (weight * losses).sum() / weight.sum()
        if variant == "rpo":
            loss = loss - lam * lw.mean()
        if log_every and step % log_every == 0:
            logs.append({"step": step, "loss": loss.item(), "logp_w": lw.mean().item(), "logp_l": ll.mean().item(),
                         "weight": torch.sigmoid(-margin).mean().item()})
        if step == steps:
            break
        optim.zero_grad()
        loss.backward()
        optim.step()
    return logs


# ---- the offline preference family, each written as a function of one margin ---------------------------------------
# m is the method's own margin: a log-ratio margin for DPO and IPO, a log-prob margin for SLiC and RRHF, a
# length-normalised log-prob margin for SimPO, a log-odds margin for ORPO. KTO is pointwise: m is one response's
# implicit reward, scored against a reference point z0. Drawing all of them on one axis is what Figure 09 does.

def loss_dpo(m, beta=1.0):
    return -F.logsigmoid(beta * m)


def loss_ipo(m, tau=1.0):
    return (m - 1.0 / (2 * tau)) ** 2  # regress the margin onto 1 / (2 tau) instead of pushing it to infinity


def loss_slic(m, delta=1.0):
    return torch.relu(delta - m)  # hinge: zero gradient once the margin passes delta


def loss_rrhf(m):
    return torch.relu(-m)  # hinge at zero, applied to every inverted pair


def loss_simpo(m, beta=1.0, gamma=1.0):
    return -F.logsigmoid(beta * m - gamma)  # logistic with a target margin gamma / beta


def loss_orpo(m):
    return -F.logsigmoid(m)  # the same logistic, applied to a log-odds margin (plus an SFT term not drawn here)


def loss_kto_desirable(m, beta=1.0, z0=0.0):
    return 1 - torch.sigmoid(beta * (m - z0))  # bounded: a value function that saturates in both directions


FAMILY = {
    "DPO": loss_dpo, "IPO": loss_ipo, "SLiC-HF": loss_slic, "RRHF": loss_rrhf, "SimPO": loss_simpo,
    "ORPO": loss_orpo, "KTO (desirable)": loss_kto_desirable,
}
