"""Reward functions. Each takes completions (B, T) and one per-prompt argument, and returns (B,) floats.

None of them is differentiable, and nothing in the lab asks them to be: the reward is a checker, as in the post.
"""

from __future__ import annotations

import torch


def sum_reward(actions: torch.Tensor, target: int) -> torch.Tensor:
    """1 if the tokens, read as digits, add up to `target`. Many completions are correct, some targets are rare."""
    return (actions.sum(-1) == target).float()


def key_token_reward(actions: torch.Tensor, key: tuple[int, int]) -> torch.Tensor:
    """1 if the token at position key[0] equals key[1]. Every other token is irrelevant to the reward."""
    pos, tok = key
    return (actions[:, pos] == tok).float()


def exact_match_reward(actions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 only for one specific completion. From a uniform policy the pass rate is |A|^-T."""
    return (actions == target.to(actions.device)).all(-1).float()


# The default training task: 6 digits, 5 tokens, five targets from common to rare.
SUM_N_ACTIONS, SUM_T = 6, 5
SUM_TARGETS = [4, 8, 12, 16, 20]


def sum_prompt_ids(n_actions: int = SUM_N_ACTIONS, n_prompts: int = len(SUM_TARGETS)) -> list[int]:
    """Prompt tokens live after the action ids in the input vocabulary."""
    return [n_actions + i for i in range(n_prompts)]
