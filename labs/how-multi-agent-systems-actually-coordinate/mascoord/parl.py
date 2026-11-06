"""Serial collapse and staged reward shaping, in a toy orchestrator bandit.

The post's Learned orchestration section describes PARL (Kimi K2.5): a trainable orchestrator,
frozen subagents, and the reward

    r = r_perf + lambda1 * r_parallel + lambda2 * r_finish

with lambda1 annealed to zero. This file is NOT that system. It is the smallest environment
that has the same two traps, so the role of each term can be watched:

- The orchestrator picks how many subagents to spawn, k in ACTIONS (0 = do it serially).
- Serial work succeeds with probability P_SERIAL.
- Parallel work needs a decomposition skill s in [0, 1] that only improves with practice:
  every parallel episode moves s toward 1. With s = 0, parallel is worse than serial.
  With s = 1, parallel is much better, because it is both more accurate and faster.
  That is the serial basin: early on, the policy gradient pushes away from the only action
  that would teach it the skill.
- The task has N_SUBTASKS real subtasks. Subagents beyond that are spurious: they do no
  work, hurt success a little, and never "finish" a subtask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIONS = np.array([0, 1, 2, 4, 8])
N_SUBTASKS = 4
P_SERIAL = 0.55
F_PARALLEL = 0.8          # parallelisable fraction of the task
COORD = 0.03              # wall-clock cost per subagent
TIME_WEIGHT = 0.3         # r_perf = success - TIME_WEIGHT * wall_clock
SKILL_RATE = 0.0005       # how fast practice builds the decomposition skill
SPURIOUS_PENALTY = 0.05   # success lost per spurious subagent
SERIAL_PRIOR = 2.0        # initial logit advantage of serial work


def wall_clock(k: np.ndarray) -> np.ndarray:
    kk = np.maximum(k, 1)
    par = np.where(k == 0, 1.0, (1 - F_PARALLEL) + F_PARALLEL / np.minimum(kk, N_SUBTASKS))
    return par + COORD * k


def success_prob(k: np.ndarray, skill: float) -> np.ndarray:
    covered = np.minimum(k, N_SUBTASKS) / N_SUBTASKS            # share of subtasks with an agent
    spurious = np.maximum(k - N_SUBTASKS, 0)
    par = (0.2 + 0.75 * skill) * (0.5 + 0.5 * covered) - SPURIOUS_PENALTY * spurious
    return np.clip(np.where(k == 0, P_SERIAL, par), 0, 1)


@dataclass
class Shaping:
    name: str
    lam_parallel: float = 0.0     # weight of r_parallel = k / max(ACTIONS)
    anneal_steps: int = 0         # linear anneal of lam_parallel to 0; 0 = never anneal
    lam_finish: float = 0.0       # weight of r_finish = finished real subtasks / k


def lam_at(sh: Shaping, step: int) -> float:
    if sh.anneal_steps <= 0:
        return sh.lam_parallel
    return sh.lam_parallel * max(0.0, 1 - step / sh.anneal_steps)


def train(sh: Shaping, steps: int = 1500, batch: int = 32, lr: float = 1.0,
          serial_prior: float = SERIAL_PRIOR, skill_rate: float = SKILL_RATE, seed: int = 0) -> dict:
    """REINFORCE with a batch-mean baseline on a softmax over ACTIONS.

    The initial policy prefers serial work by `serial_prior` logits, as a model trained to
    work alone would. Returns the final policy, the final skill, and per-step traces of
    P(serial), mean k, spurious subagents per episode, and the true objective r_perf.
    """
    rng = np.random.default_rng(seed)
    logits = np.zeros(len(ACTIONS))
    logits[0] = serial_prior
    skill = 0.0
    trace = {"p_serial": [], "mean_k": [], "spurious": [], "r_perf": []}
    for step in range(steps):
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        idx = rng.choice(len(ACTIONS), size=batch, p=probs)    # (batch,) sampled actions
        k = ACTIONS[idx]
        succ = rng.random(batch) < success_prob(k, skill)
        r_perf = succ.astype(float) - TIME_WEIGHT * wall_clock(k)
        r_par = k / ACTIONS.max()
        finished = np.minimum(k, N_SUBTASKS) * succ            # real subtasks done
        r_fin = np.where(k > 0, finished / np.maximum(k, 1), 0.0)
        reward = r_perf + lam_at(sh, step) * r_par + sh.lam_finish * r_fin

        adv = reward - reward.mean()
        grad = np.zeros_like(logits)
        for a, g in zip(idx, adv):                              # d log pi(a) / d logits = onehot - probs
            grad += g * (np.eye(len(ACTIONS))[a] - probs)
        logits += lr * grad / batch

        n_par = int((k > 0).sum())                              # practice builds the skill
        skill = 1 - (1 - skill) * (1 - skill_rate) ** n_par

        trace["p_serial"].append(float(probs[0]))
        trace["mean_k"].append(float((probs * ACTIONS).sum()))
        trace["spurious"].append(float(np.maximum(k - N_SUBTASKS, 0).mean()))
        trace["r_perf"].append(float(r_perf.mean()))
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return {"probs": probs, "skill": skill, "trace": trace}


def expected_r_perf(probs: np.ndarray, skill: float) -> float:
    """The true objective of a policy, computed exactly."""
    return float((probs * (success_prob(ACTIONS, skill) - TIME_WEIGHT * wall_clock(ACTIONS))).sum())


CONDITIONS = [
    Shaping("r_perf only"),
    Shaping("+ r_parallel, never annealed", lam_parallel=0.6),
    Shaping("+ r_parallel, annealed", lam_parallel=0.6, anneal_steps=600),
    Shaping("+ r_parallel, annealed, + r_finish", lam_parallel=0.6, anneal_steps=600, lam_finish=0.3),
]
