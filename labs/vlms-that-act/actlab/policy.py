"""A pixel policy for MiniDesk, trained three ways: behavioural cloning, DAgger, and RL from the task outcome.

The policy reads the 96 x 96 screenshot and the category of the current report (the harness keeps the plan). It
outputs one of 145 actions: a click at the centre of one of the 12 x 12 grid cells (one per visual token), or
"type the amount". This is the pixel-level action space with a GUI-Actor style pointer over visual tokens.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .observe import elements
from .desk import CATEGORIES, EXPERT_STEPS_PER_REPORT, SIZE, DeskEnv, random_task, render

CELL = 8
GRID = SIZE // CELL  # 12
N_ACTIONS = GRID * GRID + 1
TYPE = GRID * GRID


def index_to_action(i: int, env: DeskEnv):
    if i == TYPE:
        return ("type", env.subtask[1])
    return ("click", (i % GRID + 0.5) * CELL, (i // GRID + 0.5) * CELL)


def action_to_index(a) -> int:
    if a[0] == "type":
        return TYPE
    return int(a[2] // CELL) * GRID + int(a[1] // CELL)


def obs_of(env: DeskEnv):
    return render(env.state), CATEGORIES.index(env.subtask[0])


class PixelPolicy(nn.Module):
    def __init__(self, ch: int = 48):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(5, 32, 3, 2, 1), nn.GELU(), nn.Conv2d(32, ch, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(ch, ch, 3, 2, 1), nn.GELU())  # 96 -> 12: one cell per token
        self.task = nn.Embedding(len(CATEGORIES), 2 * ch)
        self.glob = nn.Sequential(nn.Linear(2 * ch, ch), nn.GELU())
        self.mix = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU(), nn.Conv2d(ch, ch, 3, 1, 1), nn.GELU())
        self.point = nn.Conv2d(ch, 1, 1)
        self.type_head = nn.Linear(ch, 1)

    def forward(self, img_u8: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        x = img_u8.permute(0, 3, 1, 2).float().div(255.0)
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        f = self.stem(torch.cat([x, xx.expand(b, 1, h, w), yy.expand(b, 1, h, w)], 1))
        gamma, beta = self.task(cat)[:, :, None, None].chunk(2, 1)
        f = f * (1 + gamma) + beta
        g = self.glob(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))  # what is on the whole screen
        f = self.mix(f + g[:, :, None, None])
        return torch.cat([self.point(f).flatten(1), self.type_head(f.mean((2, 3)))], 1)


@dataclass
class Rollout:
    obs: list
    cats: list
    acts: list
    expert: list
    success: bool
    steps: int


@torch.no_grad()
def run_episodes(policy, envs, greedy=True, temperature=1.0, record=False, explore=0.0):
    """Run a batch of environments to completion. With ``record``, keep every state and the expert's label.
    ``explore`` replaces a sampled action, with that probability, by a click on a random on-screen element (taken from
    the accessibility tree, as element-level explorers do) or by typing."""
    trajs = [Rollout([], [], [], [], False, 0) for _ in envs]
    live = list(range(len(envs)))
    while live:
        obs = [obs_of(envs[i]) for i in live]
        imgs = torch.from_numpy(np.stack([o[0] for o in obs]))
        cats = torch.tensor([o[1] for o in obs])
        logits = policy(imgs, cats)
        if greedy:
            acts = logits.argmax(1)
        else:
            acts = torch.distributions.Categorical(logits=logits / temperature).sample()
        nxt = []
        for j, i in enumerate(live):
            env, a = envs[i], int(acts[j])
            if explore and random.random() < explore:
                options = [("click", *e["box"].center) for e in elements(env.state)] + [("type", "")]
                a = action_to_index(random.choice(options))
            if record:
                t = trajs[i]
                t.obs.append(obs[j][0]), t.cats.append(obs[j][1]), t.acts.append(a)
                t.expert.append(action_to_index(env.expert_action()))
            env.step(index_to_action(a, env))
            if not env.done():
                nxt.append(i)
        live = nxt
    for t, env in zip(trajs, envs):
        t.success, t.steps = env.success(), env.steps
    return trajs


def make_envs(n, seed, n_reports=(1, 3), popup_prob=0.0, budget_factor=2.0):
    rng = random.Random(seed)
    envs = []
    for k in range(n):
        reports = rng.randint(*n_reports) if isinstance(n_reports, tuple) else n_reports
        budget = int(budget_factor * EXPERT_STEPS_PER_REPORT * reports)
        envs.append(DeskEnv(random_task(rng, reports), seed=seed * 100003 + k, popup_prob=popup_prob, budget=budget))
    return envs


def expert_demos(n, seed, n_reports=(1, 3), popup_prob=0.0):
    """Clean human-style demonstrations: the expert acts, nothing goes wrong, no pop-ups by default."""
    obs, cats, acts = [], [], []
    for env in make_envs(n, seed, n_reports, popup_prob, budget_factor=1e6):
        while not env.done():
            o, c = obs_of(env)
            a = env.expert_action()
            obs.append(o), cats.append(c), acts.append(action_to_index(a))
            env.step(a)
        assert env.success()
    return np.stack(obs), np.array(cats), np.array(acts)


def fit(policy, obs, cats, acts, steps=1500, batch=128, lr=1e-3, seed=0, log=None):
    """Behavioural cloning: cross entropy on the expert's action at each recorded state."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    O, C, A = torch.from_numpy(obs), torch.from_numpy(cats), torch.from_numpy(acts)
    policy.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(A), (batch,), generator=g)
        loss = F.cross_entropy(policy(O[idx], C[idx]), A[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log and step % 500 == 0:
            log(f"  bc step {step:5d}  loss {loss.item():.4f}  ({len(A)} examples)")
    return policy.eval()


def dagger(policy, obs, cats, acts, iters=4, episodes=96, seed=0, popup_prob=0.5, steps=800, log=None):
    """DAgger: run the learner, let the expert label the states the learner actually visits, retrain on the union."""
    for it in range(iters):
        envs = make_envs(episodes, seed=10_000 + 97 * it + seed, popup_prob=popup_prob)
        trajs = run_episodes(policy, envs, greedy=True, record=True)
        new_o = np.stack([o for t in trajs for o in t.obs])
        obs = np.concatenate([obs, new_o])
        cats = np.concatenate([cats, [c for t in trajs for c in t.cats]])
        acts = np.concatenate([acts, [a for t in trajs for a in t.expert]])
        if log:
            log(f"  dagger iter {it + 1}: learner success {np.mean([t.success for t in trajs]):.2f}, "
                f"+{len(new_o)} relabelled states")
        policy = fit(policy, obs, cats, acts, steps=steps, seed=seed + it)
    return policy


def reinforce(policy, iters=60, episodes=64, seed=0, popup_prob=0.5, lr=3e-5, explore=0.1, positive_only=False,
              kl_coef=0.0, log=None):
    """RL from the task outcome. Reward = 1 if the checker finds the right records, else 0. The advantage is the
    reward minus the batch mean (group-relative, as in GRPO, without clipping). No expert is used. ``explore`` mixes
    in clicks on random on-screen elements so that the policy can find actions it gives almost no probability, such as
    the pop-up's close button; the update ignores this small off-policy mismatch."""
    random.seed(seed)  # the explorer's choices
    torch.manual_seed(seed)  # the policy's samples
    ref = copy.deepcopy(policy).eval()  # the starting policy, for a KL penalty that keeps updates small
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for it in range(iters):
        envs = make_envs(episodes, seed=50_000 + it + 1000 * seed, n_reports=(1, 2), popup_prob=popup_prob)
        trajs = run_episodes(policy, envs, greedy=False, record=True, explore=explore)
        r = np.array([float(t.success) for t in trajs])
        history.append(r.mean())
        if r.std() == 0:
            continue
        adv = r - r.mean()
        if positive_only:  # learn only from rollouts that beat the batch: filtered imitation of its own successes
            adv = np.maximum(adv, 0.0)
        adv = adv / np.array([len(t.acts) for t in trajs])  # each episode counts once, however long it ran
        O = torch.from_numpy(np.stack([o for t in trajs for o in t.obs]))
        C = torch.tensor([c for t in trajs for c in t.cats])
        A = torch.tensor([a for t in trajs for a in t.acts])
        W = torch.tensor([adv[k] for k, t in enumerate(trajs) for _ in t.acts], dtype=torch.float32)
        policy.train()
        opt.zero_grad()
        for s in range(0, len(A), 256):  # chunks keep memory small; gradients add up
            logits = policy(O[s:s + 256], C[s:s + 256]).log_softmax(1)
            loss = -(W[s:s + 256] * logits.gather(1, A[s:s + 256, None])[:, 0]).sum() / len(trajs)
            if kl_coef:
                with torch.no_grad():
                    ref_lp = ref(O[s:s + 256], C[s:s + 256]).log_softmax(1)
                kl = (ref_lp.exp() * (ref_lp - logits)).sum(1)  # KL(reference || policy) per state
                loss = loss + kl_coef * kl.sum() / len(A)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        policy.eval()
        if log and (it + 1) % 10 == 0:
            log(f"  rl iter {it + 1:3d}: mean sampled success over last 10 iters {np.mean(history[-10:]):.2f}")
    return policy, history


@torch.no_grad()
def static_accuracy(policy, obs, cats, acts) -> float:
    """Offline, step-wise evaluation: given the expert's history, does the next action match the recorded one?"""
    pred = torch.cat([policy(torch.from_numpy(obs[i:i + 512]), torch.from_numpy(cats[i:i + 512])).argmax(1)
                      for i in range(0, len(acts), 512)])
    return float((pred.numpy() == acts).mean())


def interactive_success(policy, n=200, seed=777, n_reports=1, popup_prob=0.5, budget_factor=2.0) -> float:
    """Online evaluation: run the policy in the live app and check the saved records."""
    trajs = run_episodes(policy, make_envs(n, seed, n_reports, popup_prob, budget_factor), greedy=True)
    return float(np.mean([t.success for t in trajs]))


def noisy_expert_success(eps: float, n=1000, n_reports=6, budget_factor=1.0, seed=5) -> float:
    """The expert, but with probability eps each step is replaced by a uniformly random action. With
    budget_factor=1 there is no spare step, so any slip is fatal: the no-recovery model (1 - eps)^T."""
    rng = random.Random(seed)
    wins = 0
    for env in make_envs(n, seed, n_reports, 0.0, budget_factor):
        while not env.done():
            a = index_to_action(rng.randrange(N_ACTIONS), env) if rng.random() < eps else env.expert_action()
            env.step(a)
        wins += env.success()
    return wins / n
