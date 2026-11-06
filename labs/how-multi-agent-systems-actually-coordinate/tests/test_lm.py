"""The real-model agent: one tiny cached model, so these are marked network and slow."""

import random

import pytest

from mascoord import haystack as H


@pytest.mark.network
@pytest.mark.slow
def test_agent_answers_short_lookup():
    from mascoord.lm import Agent

    agent = Agent()
    inst = H.make_lookup(8, random.Random(3))
    r = H.solo(inst, lambda ps: agent.ask(ps, max_new_tokens=12))
    assert r["answer"] is None or r["answer"].isdigit()
    assert agent.bill()["calls"] == 1 and agent.bill()["prompt_tokens"] > 50
    reps = agent.ask(["What is 2 + 2? Reply with only the number."] * 2, system=["You are a teacher.", None])
    assert len(reps) == 2
