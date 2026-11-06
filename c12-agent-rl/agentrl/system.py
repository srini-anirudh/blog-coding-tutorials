"""The full agent-RL system: every box of the pipeline as its own small component.

    task sampler -> env workers -> agent rollouts -> trajectory store -> verifier -> batches -> update -> new policy

The components run in one process here, in sequence, so that each one can be read and timed. A production system
(see the theory post) separates them: rollouts on inference servers, tools in sandboxes, the learner on GPUs. The
interfaces stay the same, because everything crosses a boundary as text or as a JSON line.
"""

from __future__ import annotations

import copy
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from .codec import Codec
from .env import FileSystemEnv
from .evaluate import Episode
from .grpo import GRPOConfig, update
from .online import TaskSampler, group_stats
from .rollout import rollout_batch
from .tasks import sample_task
from .trajectory import Trajectory


class SlowTool(FileSystemEnv):
    """The same environment with tool latency, as if every call ran a real command."""

    latency: float = 0.0

    def step(self, action: str):
        if self.latency:
            time.sleep(self.latency)
        return super().step(action)


class EnvWorkers:
    """Run tool calls concurrently. Threads are enough: a real tool call waits on a subprocess, not on Python."""

    def __init__(self, n: int):
        self.pool = ThreadPoolExecutor(n) if n > 0 else None

    def __call__(self, envs: list[FileSystemEnv], actions: list[str]) -> list[tuple[str, float, bool]]:
        if self.pool is None:
            return [e.step(a) for e, a in zip(envs, actions)]
        return list(self.pool.map(lambda pair: pair[0].step(pair[1]), zip(envs, actions)))


class TrajectoryStore:
    """Append-only JSON lines. Everything the learner uses can be replayed, audited or trained on again."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trajs: list[Trajectory]) -> None:
        with self.path.open("a") as f:
            for t in trajs:
                f.write(t.to_json() + "\n")

    def load(self, min_version: int = 0) -> list[Trajectory]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            trajs = [Trajectory.from_json(line) for line in f]
        return [t for t in trajs if t.policy_version >= min_version]


class Verifier:
    """Re-run the recorded actions on a fresh copy of the task and decide success from the final files.

    The reward never comes from what the agent or the rollout worker claims. A disagreement means a bug or a
    tampered record, so the verifier's answer wins and the mismatch is counted.
    """

    def __init__(self) -> None:
        self.checked = 0
        self.mismatches = 0

    def verify(self, traj: Trajectory) -> Trajectory:
        env = FileSystemEnv(sample_task(traj.level, traj.seed), seed=traj.meta.get("env_seed"))
        env.reset()
        for step in traj.steps:
            if env.done():
                break
            env.step(step.action)
        verdict = env.reward() == 1.0
        self.checked += 1
        self.mismatches += int(verdict != traj.success)
        traj.success = verdict
        if traj.steps:                       # outcome reward lives on the last step
            traj.steps[-1].reward = float(verdict)
        return traj


def make_batches(trajs: list[Trajectory], current_version: int, max_lag: int = 0) -> list[Trajectory]:
    """Drop trajectories from policies older than ``max_lag`` versions, and groups left with fewer than two."""
    fresh = [t for t in trajs if current_version - t.policy_version <= max_lag]
    groups: dict[int, list[Trajectory]] = defaultdict(list)
    for t in fresh:
        groups[t.group].append(t)
    return [t for g in groups.values() if len(g) >= 2 for t in g]


@dataclass
class SystemConfig:
    tasks_per_iter: int = 16
    G: int = 8
    workers: int = 0
    max_lag: int = 0
    tool_latency: float = 0.0


class AgentRLSystem:
    def __init__(self, policy, codec: Codec, sampler: TaskSampler, cfg: GRPOConfig, sys_cfg: SystemConfig,
                 store_path: str | Path, device: str = "cpu"):
        self.policy, self.codec, self.sampler, self.cfg, self.sys = policy, codec, sampler, cfg, sys_cfg
        self.store = TrajectoryStore(store_path)
        self.verifier = Verifier()
        self.workers = EnvWorkers(sys_cfg.workers)
        self.opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
        self.device = device
        self.version = 0
        self.next_group = 0

    # -- one pass through the pipeline ---------------------------------------------------------------------
    def rollouts(self, episodes: list[Episode]) -> list[Trajectory]:
        envs, forced, groups = [], [], []
        for ep in episodes:
            task = sample_task(ep.task_level, ep.seed)
            for j in range(self.sys.G):
                env = SlowTool(task, seed=ep.seed * 31 + j)
                env.latency = self.sys.tool_latency
                envs.append(env)
                forced.append(ep.forced)
                groups.append(self.next_group)
            self.next_group += 1
        self.policy.eval()
        trajs = rollout_batch(self.policy, self.codec, envs, forced=forced, temperature=self.cfg.temperature,
                              device=self.device, version=self.version, step_fn=self.workers)
        for t, g in zip(trajs, groups):
            t.group = g
        return trajs

    def iteration(self) -> dict:
        clock = {}
        t = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t
            now = time.perf_counter()
            clock[name] = now - t
            t = now

        episodes = self.sampler.sample(self.sys.tasks_per_iter)
        lap("sample")
        trajs = self.rollouts(episodes)
        lap("rollout")
        self.store.append(trajs)
        lap("store")
        trajs = [self.verifier.verify(tr) for tr in trajs]
        lap("verify")
        batch = make_batches(trajs, self.version, self.sys.max_lag)
        lap("batch")
        stats = update(self.policy, self.opt, batch, self.cfg, pad=self.codec.pad, device=self.device,
                       seed=self.version)
        lap("update")
        self.version += 1
        return {"version": self.version, **group_stats(trajs, self.cfg.reward), **stats, "clock": clock}

    # -- resume --------------------------------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"policy": copy.deepcopy(self.policy.state_dict()), "opt": copy.deepcopy(self.opt.state_dict()),
                "version": self.version, "next_group": self.next_group, "next_seed": self.sampler.next_seed,
                "sampler_rng": self.sampler.rng.getstate(), "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate()}

    def load_state_dict(self, state: dict) -> None:
        self.policy.load_state_dict(state["policy"])
        self.opt.load_state_dict(state["opt"])
        self.version, self.next_group = state["version"], state["next_group"]
        self.sampler.next_seed = state["next_seed"]
        self.sampler.rng.setstate(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, weights_only=False))
