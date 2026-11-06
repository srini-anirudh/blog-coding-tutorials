"""Why cloning is not enough: the O(eps T^2) versus O(eps T) gap, measured on a one-dimensional lane.

The world. A lane of integer positions. The agent starts at 0 and should stay there. Each step it moves left, stays,
or moves right. Cost is 1 for every step spent away from 0, so the expert's cost is 0.

The expert. Stay at 0. Off the lane, step back toward 0.

Two versions of the world. In the forgiving lane the agent can always walk back, so a lost learner doing a random walk
eventually returns to 0 by chance. In the cliff lane, positions beyond +-cliff are absorbing: once over the edge the
agent stays there and pays cost 1 for the rest of the episode. The cliff lane is the worst case the bound assumes.

The learner. A lookup table of expert labels. At a state it has a label for, it takes the labelled action with
probability 1 - eps and a wrong action otherwise. This is the hypothesis of the imitation-learning bound: error at most
eps on the states the learner was trained on. At a state it has no label for, it acts uniformly at random, because
nothing in the training set describes that state.

Three ways to fill the table:
    bc            clean expert demonstrations. They only ever visit position 0.
    bc_recovery   expert demonstrations with injected noise (the expert is pushed off the lane and steps back), so the
                  demonstrations contain recovery behaviour.
    dagger        roll out the learner, ask the expert to label every state it visited, add them, repeat.
                  States now come from the learner's own distribution.

Everything is vectorised over episodes in numpy, so 20,000 episodes of length 200 take well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEFT, STAY, RIGHT = 0, 1, 2
MOVES = np.array([-1, 0, 1])


def expert_action(x: np.ndarray) -> np.ndarray:
    """Step toward 0; stay at 0."""
    return np.where(x < 0, RIGHT, np.where(x > 0, LEFT, STAY))


@dataclass
class Table:
    """Which positions have an expert label. Positions run from -width to +width."""
    width: int

    def __post_init__(self) -> None:
        self.labelled = np.zeros(2 * self.width + 1, dtype=bool)

    def add(self, xs: np.ndarray) -> None:
        xs = np.clip(xs, -self.width, self.width)
        self.labelled[xs + self.width] = True

    def has(self, xs: np.ndarray) -> np.ndarray:
        return self.labelled[np.clip(xs, -self.width, self.width) + self.width]

    @property
    def n_states(self) -> int:
        return int(self.labelled.sum())


def learner_action(table: Table, x: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Labelled state: expert action w.p. 1 - eps, else one of the two wrong actions. Unlabelled: uniform."""
    good = expert_action(x)
    wrong = (good + rng.integers(1, 3, size=x.shape)) % 3          # one of the other two actions
    labelled_act = np.where(rng.random(x.shape) < eps, wrong, good)
    random_act = rng.integers(0, 3, size=x.shape)
    return np.where(table.has(x), labelled_act, random_act)


def rollout(policy, T: int, n: int, rng: np.random.Generator, cliff: int | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """Run n episodes of length T. Returns (states visited (n, T), total cost per episode (n,)).

    With ``cliff`` set, a position with |x| > cliff is absorbing.
    """
    x = np.zeros(n, dtype=np.int64)
    visited = np.zeros((n, T), dtype=np.int64)
    cost = np.zeros(n)
    for t in range(T):
        visited[:, t] = x
        step = MOVES[policy(x)]
        if cliff is not None:
            step = np.where(np.abs(x) > cliff, 0, step)
        x = x + step
        cost += (x != 0)
    return visited, cost


def expert_demos(T: int, n: int, rng: np.random.Generator, noise: float = 0.0, cliff: int | None = None
                 ) -> np.ndarray:
    """States the expert visits. With noise > 0, a random action replaces the expert's w.p. noise (DART-style)."""
    def act(x):
        a = expert_action(x)
        return np.where(rng.random(x.shape) < noise, rng.integers(0, 3, size=x.shape), a)
    visited, _ = rollout(act, T, n, rng, cliff)
    return visited


def fit_bc(T: int, width: int, rng: np.random.Generator, n_demos: int = 200, noise: float = 0.0,
           cliff: int | None = None) -> Table:
    table = Table(width)
    table.add(expert_demos(T, n_demos, rng, noise, cliff).ravel())
    return table


def fit_dagger(T: int, width: int, eps: float, rng: np.random.Generator, iters: int = 5,
               n_rollouts: int = 200, cliff: int | None = None) -> Table:
    table = fit_bc(T, width, rng, cliff=cliff)                      # iteration 0 is plain cloning
    for _ in range(iters):
        visited, _ = rollout(lambda x: learner_action(table, x, eps, rng), T, n_rollouts, rng, cliff)
        table.add(visited.ravel())                                  # the expert labels where the learner went
    return table


def regret(table: Table, T: int, eps: float, rng: np.random.Generator, n: int = 20000,
           cliff: int | None = None) -> tuple[float, float]:
    """Mean total cost of the learner (the expert's is 0) and its standard error."""
    _, cost = rollout(lambda x: learner_action(table, x, eps, rng), T, n, rng, cliff)
    return float(cost.mean()), float(cost.std() / np.sqrt(n))


def bc_bound(eps: float, T: int) -> float:
    """The post's first-deviation bound: sum_t eps (T - t) = eps T (T - 1) / 2."""
    return eps * T * (T - 1) / 2


def loglog_slope(Ts, ys) -> float:
    return float(np.polyfit(np.log(Ts), np.log(ys), 1)[0])
