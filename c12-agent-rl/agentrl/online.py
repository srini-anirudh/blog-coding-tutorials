"""Online RL: sample tasks, roll out G episodes per task with the current policy, score, update, repeat."""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field

import torch

from .codec import Codec
from .env import FileSystemEnv
from .evaluate import Episode
from .grpo import GRPOConfig, update
from .rewards import REWARDS
from .rollout import rollout_batch
from .shift import perturbations
from .tasks import sample_task
from .trajectory import Trajectory


@dataclass
class TaskMix:
    levels: dict[int, float]          # level -> sampling weight
    p_perturb: float = 0.0            # chance that a level 1-3 task starts with 1-2 injected mistakes


class TaskSampler:
    """Diverse tasks and diverse initial states. Training seeds start at 10**6, away from the eval seeds."""

    def __init__(self, mix: TaskMix, seed: int = 0):
        self.mix, self.rng, self.next_seed = mix, random.Random(seed), 10**6 + 10**5 * seed

    def sample(self, n: int) -> list[Episode]:
        levels, weights = zip(*self.mix.levels.items())
        out = []
        for _ in range(n):
            level, seed = self.rng.choices(levels, weights)[0], self.next_seed
            self.next_seed += 1
            k = self.rng.choice([1, 2]) if level <= 3 and self.rng.random() < self.mix.p_perturb else 0
            out.append(Episode(level, seed, perturbations(sample_task(level, seed), k, seed) if k else []))
        return out


@dataclass
class Curriculum:
    """Move to the next stage when the rolling training success on the current stage passes ``threshold``."""

    stages: list[TaskMix]
    threshold: float = 0.7
    window: int = 5
    stage: int = 0
    recent: list[float] = field(default_factory=list)

    @property
    def mix(self) -> TaskMix:
        return self.stages[self.stage]

    def record(self, success: float) -> bool:
        self.recent = (self.recent + [success])[-self.window:]
        ready = len(self.recent) == self.window and sum(self.recent) / self.window >= self.threshold
        if ready and self.stage < len(self.stages) - 1:
            self.stage, self.recent = self.stage + 1, []
            return True
        return False


def group_rollouts(policy, codec: Codec, episodes: list[Episode], G: int, cfg: GRPOConfig, device: str,
                   version: int = 0, group_offset: int = 0) -> list[Trajectory]:
    """G trajectories per task. Members share the task and the start; their tool-failure seeds differ."""
    envs, forced, groups = [], [], []
    for g, ep in enumerate(episodes):
        task = sample_task(ep.task_level, ep.seed)
        for j in range(G):
            envs.append(FileSystemEnv(task, seed=ep.seed * 31 + j))
            forced.append(ep.forced)
            groups.append(group_offset + g)
    trajs = rollout_batch(policy, codec, envs, forced=forced, temperature=cfg.temperature, device=device,
                          version=version)
    for t, g in zip(trajs, groups):
        t.group = g
    return trajs


def group_stats(trajs: list[Trajectory], reward: str) -> dict:
    R = [sum(REWARDS[reward](t)) for t in trajs]
    by_group: dict[int, list[float]] = {}
    for t, r in zip(trajs, R):
        by_group.setdefault(t.group, []).append(r)
    zero_var = sum(max(v) == min(v) for v in by_group.values()) / len(by_group)
    return {"success": sum(t.success for t in trajs) / len(trajs), "return": sum(R) / len(R),
            "zero_var_groups": zero_var, "steps": sum(t.n_steps for t in trajs) / len(trajs),
            "submitted": sum(t.submitted for t in trajs) / len(trajs)}


def train_online(policy, codec: Codec, sampler, cfg: GRPOConfig, *, iterations: int, tasks_per_iter: int = 32,
                 G: int = 8, device: str = "cpu", ref_policy=None, curriculum: Curriculum | None = None,
                 eval_fn=None, eval_every: int = 10, log=print) -> list[dict]:
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    ref_policy = ref_policy or (copy.deepcopy(policy).eval() if cfg.kl_beta > 0 else None)
    history = []
    for it in range(1, iterations + 1):
        t0 = time.time()
        if curriculum is not None:
            sampler.mix = curriculum.mix
        trajs = group_rollouts(policy, codec, sampler.sample(tasks_per_iter), G, cfg, device, version=it)
        t1 = time.time()
        stats = update(policy, opt, trajs, cfg, pad=codec.pad, device=device, ref_policy=ref_policy, seed=it)
        row = {"iter": it, **group_stats(trajs, cfg.reward), **stats,
               "t_rollout": t1 - t0, "t_update": time.time() - t1}
        if curriculum is not None:
            row["stage"] = curriculum.stage
            curriculum.record(row["success"])
        if eval_fn is not None and (it % eval_every == 0 or it == iterations):
            row["eval"] = eval_fn(policy)
        history.append(row)
        log(_fmt(row))
    return history


def _fmt(row: dict) -> str:
    s = (f"it {row['iter']:3d}  success {row['success']:.2f}  return {row['return']:.2f}  "
         f"zero-var {row['zero_var_groups']:.2f}  steps {row['steps']:.1f}  "
         f"clip {row.get('clip_frac', 0):.3f}  {row['t_rollout']:.1f}s+{row['t_update']:.1f}s")
    if "stage" in row:
        s += f"  stage {row['stage']}"
    if "eval" in row:
        s += "  eval " + " ".join(f"{k}={v:.2f}" for k, v in row["eval"].items())
    return s
