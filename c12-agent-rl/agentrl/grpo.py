"""The policy update: GRPO-style clipped surrogate over action tokens of whole episodes.

    ratio_j = exp(log pi_theta(a_j) - log pi_old(a_j))                  for every sampled action token j
    L_pg    = - mean_j min(ratio_j * A_j, clip(ratio_j, 1 - eps, 1 + eps) * A_j)
    L       = L_pg + beta * KL(pi_theta || pi_ref) + c_v * (V - R)^2     (the last term only for action level)

The mean runs over action tokens in the minibatch (token-level normalisation, as in DAPO), so a long episode
does not get a smaller per-token gradient than a short one. Environment tokens never enter the loss.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .advantages import gae, group_relative, to_tokens
from .batching import collate
from .policy import token_logprobs
from .rewards import REWARDS
from .trajectory import Trajectory


@dataclass
class GRPOConfig:
    lr: float = 2e-4
    clip: float = 0.2
    kl_beta: float = 0.0          # KL to the reference (BC) policy; k3 estimator
    epochs: int = 2               # passes over each batch of rollouts
    minibatch: int = 64           # trajectories per gradient step
    temperature: float = 1.0      # sampling temperature, also used to compute log-probs
    credit: str = "trajectory"    # "trajectory" (GRPO) or "action" (critic + GAE)
    reward: str = "outcome"       # see rewards.py
    scale_std: bool = True        # divide by the group std (GRPO) or not (Dr. GRPO)
    gamma: float = 1.0
    lam: float = 0.95
    value_coef: float = 0.5
    max_grad_norm: float = 1.0


def step_advantages(trajs: list[Trajectory], cfg: GRPOConfig,
                    values: list[list[float]] | None = None) -> tuple[list[list[float]], list[list[float]]]:
    """Per-step advantages and value targets for each trajectory."""
    rewards = [REWARDS[cfg.reward](t) for t in trajs]
    if cfg.credit == "trajectory":
        R = torch.tensor([sum(r) for r in rewards])
        A = group_relative(R, torch.tensor([t.group for t in trajs]), scale=cfg.scale_std)
        return [[float(a)] * t.n_steps for a, t in zip(A, trajs)], [[] for _ in trajs]
    assert values is not None, "action-level credit needs V(s_t) from the value head"
    adv = [gae(r, v, cfg.gamma, cfg.lam) for r, v in zip(rewards, values)]
    targets = [[a + v for a, v in zip(ad, vs)] for ad, vs in zip(adv, values)]
    flat = torch.tensor([a for ad in adv for a in ad])
    mu, sd = flat.mean(), flat.std() + 1e-4          # whiten over the batch, as PPO does
    return [[(a - mu.item()) / sd.item() for a in ad] for ad in adv], targets


@torch.no_grad()
def state_values(policy, trajs: list[Trajectory], pad: int, device: str) -> list[list[float]]:
    """V(s_t) under the current (old) policy, read at the <act> token before each step."""
    out = []
    for i in range(0, len(trajs), 128):
        part = trajs[i: i + 128]
        v = policy(collate(part, pad).tokens.to(device)).values.cpu()
        out += [[float(v[j, p]) for p in t.state_positions] for j, t in enumerate(part)]
    return out


def grpo_loss(policy, batch, adv_tok: torch.Tensor, cfg: GRPOConfig, ref_policy=None,
              value_targets: torch.Tensor | None = None, value_mask: torch.Tensor | None = None):
    out = policy(batch.tokens)
    logp = token_logprobs(out.logits, batch.tokens, cfg.temperature)          # (B, T)
    mask = batch.action_mask
    n = mask.sum().clamp(min=1)
    ratio = (logp - batch.old_logprobs).exp()
    surr = torch.min(ratio * adv_tok, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv_tok)
    loss = -(surr * mask).sum() / n
    stats = {"clip_frac": (((ratio - 1).abs() > cfg.clip).float() * mask).sum().item() / n.item()}
    if ref_policy is not None and cfg.kl_beta > 0:
        with torch.no_grad():
            ref = token_logprobs(ref_policy(batch.tokens).logits, batch.tokens, cfg.temperature)
        kl = (ref - logp).exp() - (ref - logp) - 1                             # k3, >= 0
        loss = loss + cfg.kl_beta * (kl * mask).sum() / n
        stats["kl"] = ((kl * mask).sum() / n).item()
    if value_targets is not None:
        v_loss = (((out.values - value_targets) ** 2) * value_mask).sum() / value_mask.sum().clamp(min=1)
        loss = loss + cfg.value_coef * v_loss
        stats["value_loss"] = v_loss.item()
    return loss, stats


def update(policy, opt, trajs: list[Trajectory], cfg: GRPOConfig, *, pad: int, device: str,
           ref_policy=None, seed: int = 0) -> dict:
    """One policy update on a batch of rollouts (several minibatch steps). Returns averaged stats."""
    values = state_values(policy, trajs, pad, device) if cfg.credit == "action" else None
    step_adv, targets = step_advantages(trajs, cfg, values)
    rng = random.Random(seed)
    order = list(range(len(trajs)))
    logs: dict[str, list[float]] = {}
    policy.train()
    for _ in range(cfg.epochs):
        rng.shuffle(order)
        for k in range(0, len(order), cfg.minibatch):
            idx = order[k: k + cfg.minibatch]
            part = [trajs[i] for i in idx]
            batch = collate(part, pad)
            adv_tok = to_tokens([step_adv[i] for i in idx], batch.step_of_token)
            vt = vm = None
            if cfg.credit == "action":
                vt, vm = torch.zeros(batch.tokens.shape), torch.zeros(batch.tokens.shape)
                for j, i in enumerate(idx):
                    for t, p in enumerate(trajs[i].state_positions):
                        vt[j, p], vm[j, p] = targets[i][t], 1.0
                vt, vm = vt.to(device), vm.to(device)
            loss, stats = grpo_loss(policy, batch.to(device), adv_tok.to(device), cfg, ref_policy, vt, vm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            opt.step()
            for key, val in {"loss": loss.item(), **stats}.items():
                logs.setdefault(key, []).append(val)
    policy.eval()
    return {k: sum(v) / len(v) for k, v in logs.items()}
