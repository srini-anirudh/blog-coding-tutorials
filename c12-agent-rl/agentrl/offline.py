"""Offline RL: every update uses trajectories stored once, by the policy that existed at the time.

Two variants, both with zero new environment steps:

    offline GRPO   the same clipped loss, on stored groups, with the stored (behaviour-policy) log-probs.
                   As the policy moves away from the behaviour policy, ratios leave the clip range and the
                   gradient of those tokens is zero. Clipping is what keeps it honest, and what stops it learning.
    filtered BC    keep only the stored trajectories that succeeded and clone them (RFT, STaR).
                   No advantage, no ratio. It can only reinforce what the behaviour policy already did.
"""

from __future__ import annotations

import random

import torch

from .bc import bc_loss
from .batching import collate
from .codec import Codec
from .grpo import GRPOConfig, update
from .trajectory import Trajectory


def offline_grpo(policy, codec: Codec, buffer: list[Trajectory], cfg: GRPOConfig, *, iterations: int,
                 tasks_per_iter: int, device: str = "cpu", eval_fn=None, eval_every: int = 10, seed: int = 0,
                 log=print) -> list[dict]:
    """Each iteration samples ``tasks_per_iter`` stored groups, so one iteration costs what an online one does."""
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    rng = random.Random(seed)
    groups: dict[int, list[Trajectory]] = {}
    for t in buffer:
        groups.setdefault(t.group, []).append(t)
    keys = sorted(groups)
    history = []
    for it in range(1, iterations + 1):
        batch = [t for g in rng.sample(keys, min(tasks_per_iter, len(keys))) for t in groups[g]]
        row = {"iter": it, **update(policy, opt, batch, cfg, pad=codec.pad, device=device, seed=it)}
        if eval_fn is not None and (it % eval_every == 0 or it == iterations):
            row["eval"] = eval_fn(policy)
            log(f"it {it:3d} clip {row['clip_frac']:.3f} eval {row['eval']}")
        history.append(row)
    return history


def filtered_bc(policy, codec: Codec, buffer: list[Trajectory], *, steps: int, batch_size: int = 64,
                lr: float = 2e-4, device: str = "cpu", eval_fn=None, eval_every: int = 200, seed: int = 0,
                log=print) -> list[dict]:
    kept = [t for t in buffer if t.success]
    rng = random.Random(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    history = []
    policy.train()
    for step in range(1, steps + 1):
        batch = collate(rng.sample(kept, min(batch_size, len(kept))), codec.pad).to(device)
        loss = bc_loss(policy, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if eval_fn is not None and (step % eval_every == 0 or step == steps):
            policy.eval()
            row = {"step": step, "loss": loss.item(), "eval": eval_fn(policy)}
            policy.train()
            log(f"step {step:4d} loss {loss.item():.3f} eval {row['eval']}")
            history.append(row)
    policy.eval()
    return history
