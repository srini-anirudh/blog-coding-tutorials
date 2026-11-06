"""TRL trainers on the arithmetic task: the dataset format each one expects, and a tiny run of each.

    Trainer
     ├── SFTTrainer      dataset: {"messages": [...]}                         loss: next-token NLL
     ├── DPOTrainer      dataset: {"prompt", "chosen", "rejected"}            loss: preference margin vs reference
     ├── RewardTrainer   dataset: {"chosen", "rejected"} (prompt optional)    loss: Bradley-Terry on scalar scores
     └── GRPOTrainer     dataset: {"prompt"} + reward function              loss: group-relative policy gradient

The theory lives in the SFT, preference and RL posts; C6 builds each loss by hand. Here we only
check the data contract and that each trainer runs.
"""

from __future__ import annotations

import re

from datasets import Dataset

from .data import make_arithmetic_rows, to_messages, to_preference, to_prompt_only


def sft_dataset(n: int, seed: int = 0) -> Dataset:
    return Dataset.from_list([to_messages(r) for r in make_arithmetic_rows(n, seed)])


def preference_dataset(n: int, seed: int = 0) -> Dataset:
    return Dataset.from_list([to_preference(r) for r in make_arithmetic_rows(n, seed)])


def prompt_dataset(n: int, seed: int = 0) -> Dataset:
    return Dataset.from_list([to_prompt_only(r) for r in make_arithmetic_rows(n, seed)])


def exact_answer_reward(completions: list[list[dict]], answer: list[str], **kwargs) -> list[float]:
    """GRPO reward: 1.0 if the first integer in the reply equals the answer, else 0.0.

    TRL passes every extra dataset column (here `answer`) as a keyword argument, one value per completion.
    """
    rewards = []
    for completion, target in zip(completions, answer):
        match = re.search(r"-?\d+", completion[0]["content"])
        rewards.append(1.0 if match and match.group() == target else 0.0)
    return rewards
