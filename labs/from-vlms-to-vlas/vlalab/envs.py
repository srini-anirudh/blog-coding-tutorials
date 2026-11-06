"""Two tiny control tasks with scripted experts.

LaneTask: keep a point on a curving lane. Used to show compounding error and the DAgger remedy.
ObstacleTask: reach a goal behind an obstacle. Demonstrators pass it on the left or on the right, so the demos are
bimodal. Used to show why a mean-squared-error head drives into the obstacle.
"""

from __future__ import annotations

import math

import numpy as np


class LaneTask:
    """State: time step t and lateral offset y from the lane centre. Action: lateral velocity a.

    The lane drifts by c(t) per step, so y_{t+1} = y_t + dt * (a_t - c(t)). The expert cancels the drift and steers
    back: a* = c(t) - k * y. A step is a "mistake" when |y| exceeds `band`.
    """

    def __init__(self, dt: float = 0.1, gain: float = 3.0, band: float = 0.1, period: int = 40):
        self.dt, self.gain, self.band, self.period = dt, gain, band, period

    def drift(self, t: np.ndarray) -> np.ndarray:
        ph = 2 * math.pi * t / self.period
        return 1.0 * np.sin(ph) + 0.5 * np.sin(2 * ph + 1.0)

    def features(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        ph = 2 * math.pi * t / self.period
        return np.stack([np.sin(ph), np.cos(ph), np.sin(2 * ph), np.cos(2 * ph), y], -1).astype(np.float32)

    def expert(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.drift(t) - self.gain * y

    def step(self, t, y, a):
        return y + self.dt * (a - self.drift(t))

    def rollout(self, policy, n: int, horizon: int, rng: np.random.Generator, y0_std: float = 0.0,
                act_noise: float = 0.0):
        """Run `policy(t, y) -> a` on n lanes. Returns the visited states (t, y), shape (n, horizon) each."""
        y = rng.normal(0, y0_std, n) if y0_std else np.zeros(n)
        ts, ys = [], []
        for t in range(horizon):
            tt = np.full(n, t, dtype=np.float64)
            ts.append(tt)
            ys.append(y.copy())
            a = policy(tt, y) + (act_noise * rng.standard_normal(n) if act_noise else 0.0)
            y = self.step(tt, y, a)
        return np.stack(ts, 1), np.stack(ys, 1)


class ObstacleTask:
    """Reach the goal at (0, 1) from near (0, 0). A disk of radius 0.15 sits at (0, 0.5).

    The expert picks a side at random for each episode and steers through a waypoint at (+-0.3, 0.5). Action: a
    2-D step of length at most 0.05.
    """

    start, goal = np.array([0.0, 0.0]), np.array([0.0, 1.0])
    centre, radius, speed = np.array([0.0, 0.5]), 0.15, 0.05

    def expert(self, pos: np.ndarray, side: np.ndarray) -> np.ndarray:
        way = np.stack([0.3 * side, np.full_like(side, 0.5)], -1)
        target = np.where((pos[:, 1] < 0.45)[:, None], way, self.goal)
        d = target - pos
        dist = np.linalg.norm(d, axis=-1, keepdims=True)
        return d / np.maximum(dist, 1e-9) * np.minimum(dist, self.speed)

    def demos(self, n: int, steps: int, rng: np.random.Generator, chunk: int = 1):
        """Observation-action pairs from n expert episodes. Start positions get a little noise.

        With chunk = H the target is the next H actions, flattened to 2H numbers.
        """
        side = rng.choice([-1.0, 1.0], n)
        pos = self.start + rng.normal(0, 0.01, (n, 2))
        obs, act = [], []
        for _ in range(steps + chunk - 1):
            a = self.expert(pos, side)
            obs.append(pos.copy())
            act.append(a)
            pos = pos + a
        obs, act = np.stack(obs[:steps], 1), np.stack(act, 1)  # (n, steps, 2), (n, steps + H - 1, 2)
        act = np.concatenate([act[:, i:i + steps] for i in range(chunk)], -1)  # (n, steps, 2H)
        return obs.reshape(-1, 2).astype(np.float32), act.reshape(-1, 2 * chunk).astype(np.float32)

    def evaluate(self, policy, n: int, steps: int, rng: np.random.Generator) -> dict:
        """Roll out `policy(pos) -> step`. A step is clipped to `speed`. Collision ends the episode."""
        pos = self.start + rng.normal(0, 0.01, (n, 2))
        hit = np.zeros(n, bool)
        min_x_at_obstacle = np.zeros(n)
        for _ in range(steps):
            a = policy(pos)
            norm = np.linalg.norm(a, axis=-1, keepdims=True)
            a = a * np.minimum(1.0, self.speed / np.maximum(norm, 1e-9))
            pos = np.where(hit[:, None], pos, pos + a)
            hit |= np.linalg.norm(pos - self.centre, axis=-1) < self.radius
            near = np.abs(pos[:, 1] - 0.5) < 0.05
            min_x_at_obstacle = np.where(near & ~hit, pos[:, 0], min_x_at_obstacle)
        reached = (np.linalg.norm(pos - self.goal, axis=-1) < 0.05) & ~hit
        return {"collision_rate": float(hit.mean()), "success_rate": float(reached.mean()),
                "went_left": float((min_x_at_obstacle < -0.1).mean()),
                "went_right": float((min_x_at_obstacle > 0.1).mean())}
