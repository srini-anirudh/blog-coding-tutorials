"""Budgets and stopping (post, section 14).

Turn length: at each step the model asks for another tool call with probability q,
so L is geometric: E[L] = 1 / (1 - q) and P(L > B) = q^B. A cap B bounds the bill:
E[min(L, B)] = (1 - q^B) / (1 - q).

Termination: "no tool call" is a statement about the output format. A simulated agent
works on a task with a hidden state (is the fix in? are the plan items done?). It
sometimes declares done early. Two harnesses judge that claim:

    naive        done = no tool call
    conjunctive  done = no tool call AND verifier green AND no open plan item
                 otherwise the harness appends the failing check and the loop goes on
"""

from __future__ import annotations

import random
from dataclasses import dataclass


def mean_length(q: float) -> float:
    return 1 / (1 - q)


def tail(q: float, B: int) -> float:
    return q ** B


def mean_capped(q: float, B: int) -> float:
    return (1 - q ** B) / (1 - q)


def sample_length(q: float, rng: random.Random, cap: int | None = None) -> int:
    n = 1
    while rng.random() < q and (cap is None or n < cap):
        n += 1
    return n


@dataclass
class Episode:
    status: str        # "solved", "false victory", "max turns"
    steps: int


def run_episode(harness: str, rng: random.Random, plan_items: int = 3, p_progress: float = 0.35,
                p_early_done: float = 0.08, max_turns: int = 40) -> Episode:
    """One task. Each step finishes a plan item with probability p_progress.

    The last plan item is the fix that turns the tests green. At any step with work
    left, the model claims done with probability p_early_done; once all items are
    done it always claims done.
    """
    done_items = 0
    for step in range(1, max_turns + 1):
        if done_items < plan_items and rng.random() < p_progress:
            done_items += 1
        finished = done_items == plan_items
        claims_done = finished or rng.random() < p_early_done
        if not claims_done:
            continue
        if harness == "naive":
            return Episode("solved" if finished else "false victory", step)
        if harness == "conjunctive":
            verifier_green = finished                 # the fix is the last item
            no_open_items = done_items == plan_items
            if verifier_green and no_open_items:
                return Episode("solved", step)
            continue                                  # the failing check is appended; keep going
        raise ValueError(harness)
    return Episode("max turns", max_turns)
