"""Every RL experiment in the tutorial. Each one starts from the same BC checkpoint and writes results/<name>.json.

    python c12-agent-rl/scripts/experiments.py main
    python c12-agent-rl/scripts/experiments.py credit
    python c12-agent-rl/scripts/experiments.py shaping
    python c12-agent-rl/scripts/experiments.py curriculum
    python c12-agent-rl/scripts/experiments.py offline
    python c12-agent-rl/scripts/experiments.py exploration
    python c12-agent-rl/scripts/experiments.py groupsize
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentrl.codec import WordCodec  # noqa: E402
from agentrl.evaluate import SUITES, evaluate  # noqa: E402
from agentrl.grpo import GRPOConfig  # noqa: E402
from agentrl.offline import filtered_bc, offline_grpo  # noqa: E402
from agentrl.online import Curriculum, TaskMix, TaskSampler, group_rollouts, train_online  # noqa: E402
from agentrl.policy import TinyPolicy  # noqa: E402

codec = WordCodec()
DEVICE = "cpu"
RUNS = ROOT / "runs"
FULL_MIX = TaskMix({1: 1, 2: 1, 3: 1, 4: 2, 5: 2}, p_perturb=0.5)


def fresh(seed: int = 0) -> TinyPolicy:
    torch.manual_seed(seed)
    return TinyPolicy.load(str(RUNS / f"{ARGS.bc}.pt")).eval()


def eval_fn(n: int = 150, suites=SUITES):
    return lambda policy: evaluate(policy, codec, device=DEVICE, n=n, suites=suites)


def run(name: str, mix: TaskMix, cfg: GRPOConfig, *, iterations: int, tasks: int = 24, G: int = 8, seed: int = 0,
        curriculum: Curriculum | None = None, save: str | None = None, suites=SUITES, eval_every: int = 10) -> dict:
    print(f"--- {name}", flush=True)
    policy = fresh(seed)
    t0 = time.time()
    start = evaluate(policy, codec, device=DEVICE, n=150, suites=suites)
    before = group_rollouts(policy, codec, TaskSampler(mix, seed=99).sample(4), 1, cfg, DEVICE)
    hist = train_online(policy, codec, TaskSampler(mix, seed=seed), cfg, iterations=iterations, tasks_per_iter=tasks,
                        G=G, device=DEVICE, curriculum=curriculum, eval_fn=eval_fn(150, suites),
                        eval_every=eval_every, log=lambda s: print(s, flush=True))
    if save:
        policy.save(RUNS / f"{save}.pt")
    samples = group_rollouts(policy, codec, TaskSampler(mix, seed=99).sample(4), 1, cfg, DEVICE)
    return {"name": name, "start": start, "seconds": time.time() - t0, "history": hist,
            "cfg": cfg.__dict__, "tasks": tasks, "G": G, "samples": [t.pretty() for t in samples],
            "before": [t.pretty() for t in before]}


def dump(name: str, payload) -> None:
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(payload, indent=0))


def main_run() -> None:
    dump("main", run("grpo", FULL_MIX, GRPOConfig(), iterations=ARGS.iters or 120, save="grpo", seed=ARGS.seed))


def credit() -> None:
    long_mix = TaskMix({4: 1, 5: 2})
    out = [run("trajectory", long_mix, GRPOConfig(credit="trajectory"), iterations=ARGS.iters or 80, suites=("failures", "long")),
           run("action", long_mix, GRPOConfig(credit="action"), iterations=ARGS.iters or 80, save="grpo_action", suites=("failures", "long")),
           run("action+potential", long_mix, GRPOConfig(credit="action", reward="potential"), iterations=ARGS.iters or 80, suites=("failures", "long"))]
    dump("credit", out)


def shaping() -> None:
    mix = TaskMix({2: 1, 3: 1})
    out = [run(r, mix, GRPOConfig(reward=r), iterations=ARGS.iters or 60, suites=("clean",), eval_every=5)
           for r in ("outcome", "activity", "potential")]
    dump("shaping", out)


def curriculum() -> None:
    stages = [TaskMix({3: 1}, p_perturb=0.5), TaskMix({2: 1, 3: 1, 4: 2}, p_perturb=0.5),
              TaskMix({2: 1, 3: 1, 4: 2, 5: 4}, p_perturb=0.5)]
    final = TaskMix({2: 1, 3: 1, 4: 2, 5: 4}, p_perturb=0.5)
    out = [run("hardest first", TaskMix({5: 1}), GRPOConfig(), iterations=ARGS.iters or 100),
           run("uniform", final, GRPOConfig(), iterations=ARGS.iters or 100),
           run("curriculum", final, GRPOConfig(), iterations=ARGS.iters or 100,
               curriculum=Curriculum(stages, threshold=0.75))]
    dump("curriculum", out)


def offline() -> None:
    """Same data budget, same gradient steps. Online collects as it goes; offline collects once, from BC."""
    iters, tasks, G = ARGS.iters or 80, 24, 8
    cfg = GRPOConfig()
    behaviour = fresh()
    sampler = TaskSampler(FULL_MIX, seed=0)
    t0 = time.time()
    buffer = group_rollouts(behaviour, codec, sampler.sample(iters * tasks), G, cfg, DEVICE)
    success = sum(t.success for t in buffer) / len(buffer)
    print(f"buffer: {len(buffer)} trajectories, behaviour success {success:.3f}, {time.time() - t0:.0f}s", flush=True)
    online = run("online", FULL_MIX, cfg, iterations=iters, tasks=tasks, G=G)
    pol = fresh()
    start = evaluate(pol, codec, device=DEVICE, n=150)
    hist = offline_grpo(pol, codec, buffer, cfg, iterations=iters, tasks_per_iter=tasks, device=DEVICE,
                        eval_fn=eval_fn(), log=lambda s: print(s, flush=True))
    off = {"name": "offline GRPO", "start": start, "history": hist}
    pol = fresh()
    steps = iters * (tasks * G // cfg.minibatch + 1) * cfg.epochs
    fb = filtered_bc(pol, codec, buffer, steps=steps, device=DEVICE, eval_fn=eval_fn(),
                     eval_every=max(steps // 8, 1), log=lambda s: print(s, flush=True))
    dump("offline", {"buffer_success": success, "buffer_size": len(buffer), "gradient_steps": steps,
                     "online": online, "offline": off, "filtered": {"name": "filtered BC", "start": start, "history": fb}})


def exploration() -> None:
    mix = FULL_MIX
    out = [run(f"T={t}", mix, GRPOConfig(temperature=t), iterations=ARGS.iters or 60) for t in (0.5, 1.0, 1.5)]
    dump("exploration", out)


def groupsize() -> None:
    """Same rollouts per iteration (192), split into groups of different sizes."""
    out = [run(f"G={g}", FULL_MIX, GRPOConfig(), iterations=ARGS.iters or 60, tasks=192 // g, G=g) for g in (2, 4, 8, 16)]
    dump("groupsize", out)


EXPERIMENTS = {"main": main_run, "credit": credit, "shaping": shaping, "curriculum": curriculum, "offline": offline,
               "exploration": exploration, "groupsize": groupsize}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(EXPERIMENTS))
    ap.add_argument("--bc", default="bc")
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ARGS = ap.parse_args()
    torch.set_num_threads(2)
    EXPERIMENTS[ARGS.name]()
