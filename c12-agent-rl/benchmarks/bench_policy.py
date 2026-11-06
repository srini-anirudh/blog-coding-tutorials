"""Why the tiny policy: the same GRPO iteration timed with each policy that fits the interface.

    python c12-agent-rl/benchmarks/bench_policy.py --tasks 4 --G 4 --hardware "your CPU or GPU name"

The time per iteration decides what learning curve you can measure on a laptop. SmolLM2 also runs on a GPU when one
is visible.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentrl.codec import WordCodec  # noqa: E402
from agentrl.env import FileSystemEnv  # noqa: E402
from agentrl.evaluate import pick_device  # noqa: E402
from agentrl.grpo import GRPOConfig, update  # noqa: E402
from agentrl.hf_policy import smollm_policy, tiny_llama_policy  # noqa: E402
from agentrl.policy import PolicyConfig, TinyPolicy  # noqa: E402
from agentrl.rollout import rollout_batch  # noqa: E402
from agentrl.tasks import sample_task  # noqa: E402


def timed_iteration(policy, codec, device: str, tasks: int, G: int, max_action_tokens: int = 6) -> tuple:
    envs, groups = [], []
    for g in range(tasks):
        task = sample_task(1 + g % 3, g)
        for j in range(G):
            envs.append(FileSystemEnv(task, seed=j))
            groups.append(g)
    policy.eval()
    t = time.perf_counter()
    trajs = rollout_batch(policy, codec, envs, temperature=1.0, device=device, max_action_tokens=max_action_tokens)
    roll = time.perf_counter() - t
    for tr, g in zip(trajs, groups):
        tr.group = g
    trajs[0].success = True  # one winner: the update then does real work even for an untrained policy
    trajs[0].steps[-1].reward = 1.0
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=1e-4)
    t = time.perf_counter()
    update(policy, opt, trajs, GRPOConfig(minibatch=len(trajs), epochs=1), pad=codec.pad, device=device)
    upd = time.perf_counter() - t
    tokens = sum(len(tr.tokens) for tr in trajs)
    return roll, upd, tokens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--skip-smollm", action="store_true")
    ap.add_argument("--hardware", default="unnamed hardware")
    args = ap.parse_args()
    codec = WordCodec()
    torch.manual_seed(0)
    n = args.tasks * args.G
    print(f"one GRPO iteration: {args.tasks} tasks x G={args.G} = {n} episodes, untrained policies, {args.hardware}, {torch.get_num_threads()} CPU threads\n")
    print(f"{'policy':<30}{'device':<7}{'params':>13}{'trainable':>12}{'rollout s':>11}{'update s':>10}{'total s':>9}")
    rows = [("TinyPolicy d=128 L=3", lambda d: (TinyPolicy(PolicyConfig(codec.vocab_size, n_layers=3, max_len=384)).to(d), codec), "cpu"),
            ("LlamaForCausalLM d=128 L=3", lambda d: (tiny_llama_policy(codec.vocab_size, 128, 3, 4).to(d), codec), "cpu")]
    if not args.skip_smollm:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        rows += [("SmolLM2-135M + LoRA r=16", lambda d: smollm_policy(device=d), d)
                 for d in dict.fromkeys(["cpu", pick_device()])]
    for name, make, device in rows:
        policy, cdc = make(device)
        roll, upd, _ = timed_iteration(policy, cdc, device, args.tasks, args.G)
        total = sum(p.numel() for p in policy.parameters())
        train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"{name:<30}{device:<7}{total:>13,}{train:>12,}{roll:>11.2f}{upd:>10.2f}{roll + upd:>9.2f}", flush=True)


if __name__ == "__main__":
    main()
