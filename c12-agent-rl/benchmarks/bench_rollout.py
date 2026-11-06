"""Where does a rollout spend its time?

    python c12-agent-rl/benchmarks/bench_rollout.py

Prints: naive vs batched rollouts, the split between policy forwards and tool calls, and what concurrent env workers
buy when tools are slow.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentrl.codec import WordCodec  # noqa: E402
from agentrl.env import FileSystemEnv  # noqa: E402
from agentrl.policy import PolicyConfig, TinyPolicy  # noqa: E402
from agentrl.rollout import rollout_batch  # noqa: E402
from agentrl.rollout_naive import rollout_one  # noqa: E402
from agentrl.system import EnvWorkers, SlowTool  # noqa: E402
from agentrl.tasks import sample_task  # noqa: E402


class Counting(torch.nn.Module):
    """Wrap a policy and accumulate the time spent inside forward."""

    def __init__(self, policy):
        super().__init__()
        self.policy, self.seconds, self.calls = policy, 0.0, 0

    def forward(self, x):
        t = time.perf_counter()
        out = self.policy(x)
        self.seconds += time.perf_counter() - t
        self.calls += 1
        return out


def tasks(n: int) -> list:
    return [sample_task(1 + i % 5, i) for i in range(n)]


def load_policy(codec, device: str) -> TinyPolicy:
    path = ROOT / "runs" / "bc.pt"
    if path.exists():
        return TinyPolicy.load(str(path)).to(device).eval()
    return TinyPolicy(PolicyConfig(codec.vocab_size, n_layers=3, max_len=384)).to(device).eval()


def main() -> None:
    torch.set_num_threads(2)
    codec = WordCodec()
    torch.manual_seed(0)
    n = 64
    print(f"episodes={n}, levels 1-5, greedy decoding, Apple M5 CPU, 2 threads\n")

    print(f"{'rollout':<20}{'device':<8}{'batch':>6}{'seconds':>10}{'episodes/s':>12}{'forwards':>10}")
    policy = Counting(load_policy(codec, "cpu"))
    t = time.perf_counter()
    for task in tasks(n):
        rollout_one(policy, codec, FileSystemEnv(task), temperature=0)
    naive = time.perf_counter() - t
    print(f"{'naive':<20}{'cpu':<8}{1:>6}{naive:>10.2f}{n / naive:>12.1f}{policy.calls:>10}")
    naive_forward = policy.seconds

    for batch in (8, 64, 256):
        count = batch if batch > n else n
        policy = Counting(load_policy(codec, "cpu"))
        t = time.perf_counter()
        for i in range(0, count, batch):
            rollout_batch(policy, codec, [FileSystemEnv(x) for x in tasks(count)[i: i + batch]], temperature=0)
        sec = time.perf_counter() - t
        print(f"{'batched':<20}{'cpu':<8}{batch:>6}{sec:>10.2f}{count / sec:>12.1f}{policy.calls:>10}")
        if batch == 64:
            batched, batched_forward = sec, policy.seconds

    if torch.backends.mps.is_available():
        for batch in (64, 256):
            count = batch if batch > n else n
            policy = Counting(load_policy(codec, "mps"))
            t = time.perf_counter()
            rollout_batch(policy, codec, [FileSystemEnv(x) for x in tasks(count)], temperature=0, device="mps")
            sec = time.perf_counter() - t
            print(f"{'batched':<20}{'mps':<8}{batch:>6}{sec:>10.2f}{count / sec:>12.1f}{policy.calls:>10}")

    print(f"\ntime inside policy forward: naive {100 * naive_forward / naive:.0f}%, "
          f"batched {100 * batched_forward / batched:.0f}%")

    print("\nslow tools: every call sleeps as if it ran a command (batched rollout, batch 64)")
    print(f"{'tool latency':<16}{'workers':>8}{'seconds':>10}{'episodes/s':>12}")
    policy = load_policy(codec, "cpu")
    for latency in (0.0, 0.02):
        for workers in (0, 8, 32):
            envs = []
            for x in tasks(n):
                e = SlowTool(x)
                e.latency = latency
                envs.append(e)
            t = time.perf_counter()
            rollout_batch(policy, codec, envs, temperature=0, step_fn=EnvWorkers(workers))
            sec = time.perf_counter() - t
            print(f"{1000 * latency:>6.0f} ms{'':<7}{workers:>8}{sec:>10.2f}{n / sec:>12.1f}")


if __name__ == "__main__":
    main()
