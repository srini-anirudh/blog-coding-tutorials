"""Stage 7: a small rollout engine.

    prompts -> generation workers -> rollouts -> reward workers -> training batch -> optimizer

Each generation worker owns a policy replica and a `version` number. The trainer pushes new weights with
`sync()`. Every rollout records the version that produced it, so the trainer can see how stale its data is:
the same bookkeeping an agent-RL system needs when environments are slow and generation runs asynchronously.
"""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import torch

from .chat import encode_prompt
from .generation import generate, left_pad
from .grpo import group_advantages
from .tasks import Problem, to_messages


@dataclass
class Rollout:
    problem: Problem
    prompt_ids: list[int]
    completion_ids: list[int]  # up to and including <|im_end|>
    text: str
    truncated: bool
    policy_version: int
    worker_id: int
    reward: float = 0.0


class GenerationWorker:
    """Generates `group_size` samples per prompt with its own copy of the policy."""

    def __init__(self, model, tok, worker_id: int, share_weights: bool, max_new_tokens: int, temperature: float):
        # share_weights=True: the worker *is* the trainer's model (always current, no copy, no parallelism).
        self.model = model if share_weights else copy.deepcopy(model).eval()
        self.tok, self.worker_id, self.share = tok, worker_id, share_weights
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.version = 0

    def sync(self, policy: torch.nn.Module, version: int) -> None:
        if not self.share:
            with torch.no_grad():
                for dst, src in zip(self.model.parameters(), policy.parameters()):
                    dst.copy_(src)
        self.version = version

    def run(self, problems: list[Problem], group_size: int) -> list[Rollout]:
        prompts = [encode_prompt(self.tok, to_messages(p, with_answer=False)) for p in problems]
        flat = [q for q in prompts for _ in range(group_size)]  # prompt-major: G copies of prompt 0 first
        gen = generate(self.model, self.tok, flat, self.max_new_tokens, self.temperature, do_sample=True)
        out = []
        for i, q in enumerate(flat):
            n = int(gen.completion_mask[i].sum())
            out.append(Rollout(problems[i // group_size], q, gen.completion_ids[i, :n].tolist(), gen.texts[i],
                               bool(gen.truncated[i]), self.version, self.worker_id))
        return out


class RolloutEngine:
    def __init__(
        self,
        policy,
        tok,
        reward_fn: Callable[[str, int], float],
        num_workers: int = 1,
        group_size: int = 8,
        max_new_tokens: int = 48,
        temperature: float = 1.0,
    ):
        share = num_workers == 1
        self.workers = [GenerationWorker(policy, tok, i, share, max_new_tokens, temperature) for i in range(num_workers)]
        self.reward_fn, self.group_size, self.tok = reward_fn, group_size, tok
        self.pool = ThreadPoolExecutor(max_workers=num_workers)
        self.timings: dict[str, float] = {}

    def sync(self, policy, version: int) -> None:
        t = time.time()
        for w in self.workers:
            w.sync(policy, version)
        self.timings["sync_sec"] = time.time() - t

    def collect(self, problems: list[Problem]) -> list[Rollout]:
        """Shard prompts across workers (whole groups stay on one worker), generate, then score."""
        t = time.time()
        shards = [problems[i :: len(self.workers)] for i in range(len(self.workers))]
        futures = [self.pool.submit(w.run, s, self.group_size) for w, s in zip(self.workers, shards) if s]
        rollouts = [r for f in futures for r in f.result()]
        self.timings["generate_sec"] = time.time() - t
        t = time.time()
        for r in rollouts:  # verifiers are cheap here; for code or agents this becomes its own worker pool
            r.reward = self.reward_fn(r.text, r.problem.answer)
        self.timings["reward_sec"] = time.time() - t
        return rollouts

    def build_batch(self, rollouts: list[Rollout], scale: str = "group") -> dict[str, torch.Tensor]:
        """Pad rollouts into the tensors the GRPO loss consumes.

        prompt_ids, prompt_mask: (N, P) left-padded.  completion_ids, completion_mask: (N, C) right-padded.
        rewards, advantages, versions: (N,).
        """
        pad = self.tok.pad_token_id
        prompt_ids, prompt_mask = left_pad([r.prompt_ids for r in rollouts], pad)
        C = -(-max(len(r.completion_ids) for r in rollouts) // 32) * 32  # few distinct shapes for MPS
        completion_ids = torch.full((len(rollouts), C), pad, dtype=torch.long)
        completion_mask = torch.zeros((len(rollouts), C), dtype=torch.long)
        for i, r in enumerate(rollouts):
            completion_ids[i, : len(r.completion_ids)] = torch.tensor(r.completion_ids)
            completion_mask[i, : len(r.completion_ids)] = 1
        rewards = torch.tensor([r.reward for r in rollouts])
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "rewards": rewards,
            "advantages": group_advantages(rewards, self.group_size, scale=scale),
            "versions": torch.tensor([r.policy_version for r in rollouts]),
        }
