"""Credit over many steps: trajectory-level advantages versus anchor-state (step-level) grouping.

The world. An n x n grid, a start cell and a goal cell. Actions: up, down, left, right, and `reload`, which leaves
the state unchanged (a wasted step, like reloading a page). Walking into a wall is also a wasted step. Binary
reward: 1 if the goal is reached within the step budget.

Trajectory-level credit (the GRPO default). Every action of rollout i gets A_i = (R_i - mean_G) / (std_G + eps).
A reload inside a winning rollout gets a positive advantage.

Anchor-state grouping (GiGPO, Feng et al., 2025). Rollouts in a group start from the same state, so they pass through
the same cells. Every visit (i, t) to cell s has a discounted return g_{i,t} = gamma^(steps left until reward) * R_i.
Group all visits to the same cell across the group (the anchor group of s) and normalise g inside it. A wasted
step at s is followed by a later visit to the same s with one fewer step to go, so its return is lower by a factor
gamma and its step advantage is negative, even inside a winning rollout. The final advantage is

    A(s_t, a_t) = A_episode(tau) + lam * A_step(s_t, a_t).

The policy is a table of logits per cell (softmax over 5 actions), trained by policy gradient with autograd.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

ACTIONS = ["up", "down", "left", "right", "reload"]
DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]


def transition_table(n: int) -> torch.Tensor:
    """next_state[s, a] for an n x n grid with walls on the border."""
    nxt = torch.zeros(n * n, len(ACTIONS), dtype=torch.long)
    for s in range(n * n):
        r, c = divmod(s, n)
        for a, (dr, dc) in enumerate(DELTAS):
            r2, c2 = r + dr, c + dc
            nxt[s, a] = r2 * n + c2 if 0 <= r2 < n and 0 <= c2 < n else s
    return nxt


@dataclass
class Rollouts:
    states: torch.Tensor     # (B, T) state before each step
    actions: torch.Tensor    # (B, T)
    alive: torch.Tensor      # (B, T) bool: step t was taken
    length: torch.Tensor     # (B,) steps taken
    reward: torch.Tensor     # (B,) 1 if the goal was reached


def rollout(logits: torch.Tensor, nxt: torch.Tensor, starts: torch.Tensor, goal: int, T: int,
            gen: torch.Generator) -> Rollouts:
    B = starts.shape[0]
    s = starts.clone()
    states = torch.zeros(B, T, dtype=torch.long)
    actions = torch.zeros(B, T, dtype=torch.long)
    alive = torch.zeros(B, T, dtype=torch.bool)
    done = s == goal
    probs = torch.softmax(logits.detach(), -1)
    for t in range(T):
        a = torch.multinomial(probs[s], 1, generator=gen).squeeze(-1)
        states[:, t], actions[:, t], alive[:, t] = s, a, ~done
        s = torch.where(done, s, nxt[s, a])
        done = done | (s == goal)
    return Rollouts(states, actions, alive, alive.sum(1), done.float())


def episode_advantages(reward: torch.Tensor, group: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    adv = torch.zeros_like(reward)
    for g in group.unique():
        m = group == g
        r = reward[m]
        adv[m] = (r - r.mean()) / (r.std() + eps) if m.sum() > 1 else 0.0
    return adv


def step_returns(ro: Rollouts, gamma: float) -> torch.Tensor:
    """g[i, t] = gamma^(length_i - 1 - t) * R_i on alive steps, 0 elsewhere."""
    t = torch.arange(ro.states.shape[1])[None, :]
    left = (ro.length[:, None] - 1 - t).clamp(min=0).float()
    return torch.where(ro.alive, gamma ** left * ro.reward[:, None], torch.zeros(()))


def anchor_advantages(ro: Rollouts, group: torch.Tensor, gamma: float, eps: float = 1e-6
                      ) -> tuple[torch.Tensor, float]:
    """Step-level advantage from anchor groups. Returns (A_step (B, T), fraction of steps in an anchor group > 1)."""
    g = step_returns(ro, gamma)
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for i, t in ro.alive.nonzero().tolist():
        buckets[(int(group[i]), int(ro.states[i, t]))].append((i, t))
    adv = torch.zeros_like(g)
    shared = 0
    for visits in buckets.values():
        if len(visits) < 2:
            continue
        shared += len(visits)
        idx = torch.tensor(visits)
        vals = g[idx[:, 0], idx[:, 1]]
        adv[idx[:, 0], idx[:, 1]] = (vals - vals.mean()) / (vals.std() + eps)
    return adv, shared / max(1, int(ro.alive.sum()))


@dataclass
class CreditConfig:
    n: int = 5
    T: int = 16                      # step budget; the shortest path from corner to corner is 2 (n - 1)
    groups: int = 16                 # groups per iteration, one start cell each
    G: int = 8
    iters: int = 150
    lr: float = 5.0
    gamma: float = 0.9
    lam: float = 1.0
    mode: str = "trajectory"         # trajectory | trajectory_discounted | anchor
    seed: int = 0


def wasted(ro: Rollouts, nxt: torch.Tensor) -> torch.Tensor:
    """(B, T) bool: the step left the state unchanged (reload or wall bump)."""
    return ro.alive & (nxt[ro.states, ro.actions] == ro.states)


def train(cfg: CreditConfig) -> list[dict]:
    gen = torch.Generator().manual_seed(cfg.seed)
    nxt = transition_table(cfg.n)
    goal = cfg.n * cfg.n - 1
    logits = torch.zeros(cfg.n * cfg.n, len(ACTIONS), requires_grad=True)
    opt = torch.optim.SGD([logits], lr=cfg.lr)
    group = torch.arange(cfg.groups).repeat_interleave(cfg.G)
    rows = []
    for it in range(cfg.iters):
        starts = torch.zeros(cfg.groups * cfg.G, dtype=torch.long)       # every group starts at the top-left
        ro = rollout(logits, nxt, starts, goal, cfg.T, gen)
        if cfg.mode == "trajectory_discounted":
            outcome = ro.reward * cfg.gamma ** (ro.length.float() - 1)
        else:
            outcome = ro.reward
        adv = episode_advantages(outcome, group)[:, None].expand_as(ro.states).clone()
        shared = 0.0
        if cfg.mode == "anchor":
            a_step, shared = anchor_advantages(ro, group, cfg.gamma)
            adv = adv + cfg.lam * a_step
        logp = torch.log_softmax(logits, -1)[ro.states, ro.actions]
        loss = -(adv * logp * ro.alive).sum() / ro.states.shape[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
        win = ro.reward > 0
        rows.append({"iter": it, "success": float(ro.reward.mean()),
                     "steps_when_won": float(ro.length[win].float().mean()) if win.any() else float("nan"),
                     "wasted_frac": float(wasted(ro, nxt).sum() / ro.alive.sum()),
                     "shared_frac": shared})
    return rows


def worked_example(gamma: float = 0.9) -> list[dict]:
    """Figure 7 in miniature: four rollouts on a 1 x 4 corridor (cells 0..3, goal 3), one with a wasted reload.

    Rollout A: right, reload, right, right   (wins, 4 steps, one wasted)
    Rollout B: right, right, right           (wins, 3 steps)
    Rollout C: left, left, left, left        (loses: bumps the wall until the budget of 4 steps ends)
    Rollout D: reload, left, reload, left    (loses)

    Cell 1 is the anchor: A visits it twice (reload, then right) and B once (right). The losing rollouts never
    reach it. If they did, their returns of 0 would lower the anchor mean, and the reload would score above them
    but still below the move to the right: step credit ranks the actions taken at one state.
    """
    seqs = {"A": [0, 1, 1, 2], "B": [0, 1, 2], "C": [0, 0, 0, 0], "D": [0, 0, 0, 0]}
    acts = {"A": ["right", "reload", "right", "right"], "B": ["right", "right", "right"],
            "C": ["left", "left", "left", "left"], "D": ["reload", "left", "reload", "left"]}
    won = {"A": 1.0, "B": 1.0, "C": 0.0, "D": 0.0}
    T = 4
    names = list(seqs)
    states = torch.zeros(len(names), T, dtype=torch.long)
    alive = torch.zeros(len(names), T, dtype=torch.bool)
    for i, k in enumerate(names):
        states[i, :len(seqs[k])] = torch.tensor(seqs[k])
        alive[i, :len(seqs[k])] = True
    ro = Rollouts(states, torch.zeros_like(states), alive, alive.sum(1), torch.tensor([won[k] for k in names]))
    group = torch.zeros(len(names), dtype=torch.long)
    a_ep = episode_advantages(ro.reward, group)
    a_step, _ = anchor_advantages(ro, group, gamma)
    g = step_returns(ro, gamma)
    rows = []
    for i, k in enumerate(names):
        for t in range(int(ro.length[i])):
            rows.append({"rollout": k, "t": t, "cell": int(states[i, t]), "action": acts[k][t],
                         "return": round(float(g[i, t]), 4), "A_traj": round(float(a_ep[i]), 3),
                         "A_step": round(float(a_step[i, t]), 3),
                         "A_two_level": round(float(a_ep[i] + a_step[i, t]), 3)})
    return rows
