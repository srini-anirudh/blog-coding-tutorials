"""Demonstrations disagree: half go left of the obstacle, half go right.

Train heads on the same demos and roll them out: MSE regression, 256 bins per dimension, and flow matching with
pi0's time distribution and ten Euler steps. Two of them predict a chunk of 8 actions and play it open loop.
The MSE head predicts the average of left and right.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlalab.envs import ObstacleTask  # noqa: E402
from vlalab.heads import BinHead, FlowHead, RegressionHead, train  # noqa: E402
from vlalab.report import Report  # noqa: E402
from vlalab.tokenizers import UniformBinner  # noqa: E402


CHUNK = 8


def build_heads(act1: np.ndarray, act8: np.ndarray):
    """(name, chunk size, head). Regression and bins use a 3-layer, 128-wide MLP; flow uses 4 layers, 256 wide."""
    scale = lambda a: torch.as_tensor(np.abs(a).max(0) + 1e-6, dtype=torch.float32)  # noqa: E731
    return [("MSE regression", 1, RegressionHead(2, 2)),
            (f"MSE regression, chunk of {CHUNK}", CHUNK, RegressionHead(2, 2 * CHUNK)),
            ("256 bins per dimension", 1, BinHead(2, 2, UniformBinner(256).fit(act1))),
            ("flow matching, 10 steps", 1, FlowHead(2, 2, scale=scale(act1))),
            (f"flow matching, 10 steps, chunk of {CHUNK}", CHUNK, FlowHead(2, 2 * CHUNK, scale=scale(act8)))]


class ChunkExecutor:
    """Ask the head for a chunk of H actions every H steps and play it open loop in between."""

    def __init__(self, head, H: int, gen, norm):
        self.head, self.H, self.gen, self.norm, self.t, self.chunk = head, H, gen, norm, 0, None

    def __call__(self, pos):
        if self.t % self.H == 0:
            obs = torch.as_tensor(self.norm(pos), dtype=torch.float32)
            out = self.head.act(obs, generator=self.gen).numpy()
            self.chunk = out.reshape(len(pos), self.H, 2)
        a = self.chunk[:, self.t % self.H]
        self.t += 1
        return a


def run(steps: int, seed: int, n_demos: int = 200, n_eval: int = 500):
    task = ObstacleTask()
    data = {H: task.demos(n_demos, 30, np.random.default_rng(seed), chunk=H) for H in (1, CHUNK)}
    # Standardise positions. Without this the 2-D input is tiny next to the noise input of the flow head.
    mu, sd = data[1][0].mean(0), data[1][0].std(0)
    norm = lambda p: ((p - mu) / sd).astype(np.float32)  # noqa: E731
    out = {}
    for name, H, head in build_heads(data[1][1], data[CHUNK][1]):
        obs, act = data[H]
        log = train(head, norm(obs), act, steps=steps, seed=seed)
        gen = torch.Generator().manual_seed(seed)
        res = task.evaluate(ChunkExecutor(head, H, gen, norm), n_eval, 40, np.random.default_rng(seed + 1))
        # What each head predicts at the start, where the demos split: the spread of sampled lateral steps.
        start = torch.as_tensor(norm(np.zeros((1000, 2))))
        dx = head.act(start, generator=gen)[:, 0].numpy()
        res.update(final_loss=round(log[-1], 6), start_dx_mean=round(float(dx.mean()), 4),
                   start_dx_abs_mean=round(float(np.abs(dx).mean()), 4))
        out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "02_multimodal_demos"))
    args = ap.parse_args()
    rep = Report(args.out, " ".join(["scripts/02_multimodal_demos.py"] + sys.argv[1:]))
    task = ObstacleTask()
    rep.line(f"Obstacle task: start (0, 0), goal (0, 1), disk of radius {task.radius} at (0, 0.5).")
    rep.line("200 demos, each picks left or right at random. 500 test rollouts per head, 40 steps.")
    rep.line(f"Demo lateral step at the start: +-{task.speed * 0.3 / np.hypot(0.3, 0.5):.4f} (left or right, never 0).")
    res = run(args.steps, args.seed)
    rep.table([{"head": k, **v} for k, v in res.items()])
    rep.save_json(res)


if __name__ == "__main__":
    main()
