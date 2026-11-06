"""Greedy accuracy on held-out arithmetic prompts, broken down by problem kind."""

from __future__ import annotations

from collections import defaultdict

import torch
from transformers import PreTrainedTokenizerBase

from .chat import encode_prompt
from .generation import generate
from .tasks import Problem, exact_reward, extract_answer, to_messages


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    problems: list[Problem],
    batch_size: int = 64,
    max_new_tokens: int = 48,
) -> dict[str, float]:
    """Returns accuracy overall and per kind, plus the fraction of completions with a parseable answer."""
    was_training = model.training
    model.eval()
    hits: dict[str, list[float]] = defaultdict(list)
    formatted = []
    for s in range(0, len(problems), batch_size):
        chunk = problems[s : s + batch_size]
        prompts = [encode_prompt(tok, to_messages(p, with_answer=False)) for p in chunk]
        gen = generate(model, tok, prompts, max_new_tokens=max_new_tokens, do_sample=False)
        for p, text in zip(chunk, gen.texts):
            r = exact_reward(text, p.answer)
            hits[p.kind].append(r)
            hits["all"].append(r)
            formatted.append(extract_answer(text) is not None)
    model.train(was_training)
    result = {k: sum(v) / len(v) for k, v in hits.items()}
    result["format"] = sum(formatted) / len(formatted)
    return result
