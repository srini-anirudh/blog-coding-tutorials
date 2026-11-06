"""One training loop, six advantage estimators. The loop is the post's "Back to tensors" listing, made small:

    prompts -> rollout (G samples each) -> score -> log-probs (and values) -> advantages -> loss -> step

Methods:
    reinforce   R as the weight, no baseline
    batch       R minus the batch mean
    rloo        R minus the mean of the other G-1 samples of the same prompt
    grpo        (R - group mean) / group std
    critic_gae  a learned critic V_phi(s_t) and GAE(gamma=1, lambda) advantages per token
    ppo         rloo advantages, several clipped epochs on each batch
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .advantages import gae, returns, terminal_rewards
from .estimators import baseline_weights, pg_loss
from .exact import ExactTask, kl
from .policy import TinyLM, prompt_batch, sample, token_logprobs, token_values
from .ppo import clipped_surrogate
from .tasks import SUM_N_ACTIONS, SUM_T, SUM_TARGETS, sum_prompt_ids, sum_reward


@dataclass
class TrainConfig:
    method: str = "rloo"
    steps: int = 200
    group: int = 8
    lr: float = 3e-3
    hidden: int = 64
    ppo_epochs: int = 4
    clip_eps: float = 0.2
    lam: float = 0.95
    critic_lr: float = 1e-2
    eval_every: int = 20
    seed: int = 0
    track_step_kl: bool = False


def sum_task() -> ExactTask:
    return ExactTask(SUM_N_ACTIONS, SUM_T, sum_reward, sum_prompt_ids(), SUM_TARGETS)


def make_policy(task: ExactTask, hidden: int, seed: int, n_out: int | None = None) -> TinyLM:
    torch.manual_seed(seed)
    return TinyLM(task.n_actions + len(task.prompt_ids), n_out or task.n_actions, hidden)


def rollout(policy: TinyLM, task: ExactTask, group: int, gen: torch.Generator):
    """G samples for every prompt. Returns prompts (N, 1), actions (N, T), rewards (N,), prompt-major."""
    prompts = prompt_batch(task.prompt_ids, group)
    actions = sample(policy, prompts, task.T, gen)
    R = torch.cat([task.reward_fn(actions[i * group:(i + 1) * group], arg)
                   for i, arg in enumerate(task.prompt_args)])
    return prompts, actions, R


def train(cfg: TrainConfig, task: ExactTask | None = None, policy: TinyLM | None = None, log=None) -> dict:
    task = task or sum_task()
    policy = policy or make_policy(task, cfg.hidden, cfg.seed)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    critic = critic_opt = None
    if cfg.method == "critic_gae":
        critic = make_policy(task, cfg.hidden, cfg.seed + 1000, n_out=1)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    history = []
    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.steps:
            row = {"step": step, **task.metrics(policy)}
            history.append(row)
            if log:
                log(row)
        if step == cfg.steps:
            break
        before = [lp.detach() for lp in task.logps(policy)] if cfg.track_step_kl else None
        prompts, actions, R = rollout(policy, task, cfg.group, gen)
        mask = torch.ones_like(actions, dtype=torch.float32)
        stats = {}
        if cfg.method in ("reinforce", "batch", "rloo", "grpo"):
            kind = {"reinforce": "none", "batch": "batch", "rloo": "group", "grpo": "group_std"}[cfg.method]
            loss = pg_loss(token_logprobs(policy, prompts, actions), baseline_weights(R, kind, cfg.group))
            opt.zero_grad()
            loss.backward()
            opt.step()
        elif cfg.method == "critic_gae":
            values = token_values(critic, prompts, actions)
            r = terminal_rewards(R, task.T)
            adv = gae(r, values.detach(), gamma=1.0, lam=cfg.lam)
            loss = pg_loss(token_logprobs(policy, prompts, actions), adv)
            v_loss = (values - returns(r)).pow(2).mean()
            opt.zero_grad()
            critic_opt.zero_grad()
            (loss + v_loss).backward()
            opt.step()
            critic_opt.step()
        elif cfg.method == "ppo":
            with torch.no_grad():
                old = token_logprobs(policy, prompts, actions)
            adv = baseline_weights(R, "group", cfg.group)
            for _ in range(cfg.ppo_epochs):
                loss, stats = clipped_surrogate(token_logprobs(policy, prompts, actions), old, adv, mask,
                                                cfg.clip_eps)
                opt.zero_grad()
                loss.backward()
                opt.step()
        else:
            raise ValueError(cfg.method)
        if before is not None:
            with torch.no_grad():
                after = task.logps(policy)
                step_kl = float(torch.stack([kl(a, b) for a, b in zip(after, before)]).mean())
            history.append({"step": step + 1, "step_kl": step_kl, **{k: v for k, v in stats.items()}})
    return {"config": asdict(cfg), "history": history}


def first_step_reaching(history: list[dict], key: str, threshold: float):
    for row in history:
        if key in row and row[key] >= threshold:
            return row["step"]
    return None
