"""Behaviour cloning: L = - sum_t log pi(a*_t | s_t), summed over action tokens only.

This is SFT with an assistant-only loss mask (C6), applied to whole multi-turn episodes. The environment's tokens
are context; only the expert's action tokens are targets.
"""

from __future__ import annotations

import random

import torch

from .batching import Batch, collate
from .codec import Codec, encode_trajectory
from .env import FileSystemEnv, run_episode
from .expert import ScriptedExpert
from .policy import token_logprobs
from .tasks import sample_task
from .trajectory import Trajectory


def expert_demos(n: int, levels: tuple[int, ...], codec: Codec, seed: int = 0) -> list[Trajectory]:
    """Clean expert episodes. Seeds start at 10**6 so they never collide with evaluation seeds."""
    rng = random.Random(seed)
    demos = []
    for k in range(n):
        task = sample_task(rng.choice(levels), seed=10**6 + seed * n + k)
        demos.append(encode_trajectory(run_episode(FileSystemEnv(task), ScriptedExpert(task.text)), codec))
    return demos


def bc_loss(policy, batch: Batch) -> torch.Tensor:
    logp = token_logprobs(policy(batch.tokens).logits, batch.tokens)       # (B, T)
    return -(logp * batch.action_mask).sum() / batch.action_mask.sum()


def train_bc(policy, demos: list[Trajectory], *, steps: int, batch_size: int = 64, lr: float = 1e-3,
             device: str = "cpu", pad: int = 0, seed: int = 0, log_every: int = 100) -> list[tuple[int, float]]:
    rng = random.Random(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    history = []
    policy.train()
    for step in range(1, steps + 1):
        batch = collate(rng.sample(demos, batch_size), pad).to(device)
        loss = bc_loss(policy, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == 1:
            history.append((step, loss.item()))
    policy.eval()
    return history
