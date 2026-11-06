"""Shared pieces of the two rollout implementations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .env import FileSystemEnv
from .trajectory import Trajectory


def sample_token(logits: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """logits: (B, V) -> token ids (B,), log-probs (B,) under the distribution we sampled from.

    temperature 0 means greedy; the log-prob is then reported at temperature 1.
    """
    if temperature == 0:
        tok = logits.argmax(-1)
        logp = F.log_softmax(logits.float(), -1)
    else:
        logp = F.log_softmax(logits.float() / temperature, -1)
        tok = torch.multinomial(logp.exp(), 1).squeeze(-1)
    return tok, logp.gather(-1, tok.unsqueeze(-1)).squeeze(-1)


def new_trajectory(env: FileSystemEnv, prompt: list[int], version: int = 0) -> Trajectory:
    tr = Trajectory(env.task.text, env.task.level, env.task.seed, policy_version=version,
                    meta={"env_seed": env.seed})
    tr.tokens = list(prompt)
    tr.action_mask = [0] * len(prompt)
    tr.logprobs = [0.0] * len(prompt)
    tr.step_of_token = [-1] * len(prompt)
    return tr


def append(tr: Trajectory, ids: list[int], logprobs: list[float] | None, step: int) -> None:
    """Append action tokens (``logprobs`` given) or environment tokens (``logprobs`` None)."""
    sampled = logprobs is not None
    tr.tokens += ids
    tr.action_mask += [int(sampled)] * len(ids)
    tr.logprobs += logprobs if sampled else [0.0] * len(ids)
    tr.step_of_token += [step if sampled else -1] * len(ids)


def finish(tr: Trajectory, env: FileSystemEnv) -> Trajectory:
    tr.steps = list(env.history)
    tr.success = env.reward() == 1.0
    tr.submitted = env.submitted
    tr.truncated = not env.submitted
    return tr
