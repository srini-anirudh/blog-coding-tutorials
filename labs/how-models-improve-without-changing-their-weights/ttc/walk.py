"""A toy model of 'more thinking hurts': a reasoning trace as a walk between three states.

U = still working, C = holds the correct conclusion, D = captured by a distractor.
At step t the model reaches C from U with chance skill * exp(-t / tau): early steps apply the skill
it has, later steps have less of it left. A distractor pulls U and C towards D with chance `pull`
(C is pulled at half that rate). C is not absorbing, so a correct conclusion can be abandoned.
With pull = 0, C is absorbing and accuracy only rises.
"""

from __future__ import annotations

import numpy as np


def step_matrix(t: int, skill: float, pull: float, tau: float) -> np.ndarray:
    """Row-stochastic 3x3 transition matrix over (U, C, D) at step t."""
    s = skill * np.exp(-t / tau)
    return np.array([
        [1.0 - s - pull, s, pull],
        [0.0, 1.0 - pull / 2.0, pull / 2.0],
        [0.0, s / 2.0, 1.0 - s / 2.0],
    ])


def accuracy_curve(T: int, skill: float, pull: float, tau: float = 30.0) -> np.ndarray:
    """P(state = C) after 1..T steps, computed exactly by propagating the distribution."""
    dist = np.array([1.0, 0.0, 0.0])
    acc = np.empty(T)
    for t in range(T):
        dist = dist @ step_matrix(t, skill, pull, tau)
        acc[t] = dist[1]
    return acc


def sample_walks(n: int, T: int, skill: float, pull: float, tau: float, rng: np.random.Generator) -> np.ndarray:
    """Sample n trajectories. Returns an (n, T) array of states 0 = U, 1 = C, 2 = D."""
    states = np.zeros((n, T), dtype=np.int8)
    cur = np.zeros(n, dtype=np.int64)
    for t in range(T):
        P = step_matrix(t, skill, pull, tau)
        u = rng.random(n)
        cum = np.cumsum(P[cur], axis=1)
        cur = (u[:, None] > cum).sum(axis=1)
        states[:, t] = cur
    return states


def negative_flips(states: np.ndarray) -> float:
    """Fraction of trajectories that held C at some step and ended outside it."""
    held = (states == 1).any(axis=1)
    return float((held & (states[:, -1] != 1)).mean())
