import random

import pytest

from agentcraft.state_layers import HarnessConfig, make_todo, run_episode
from agentcraft.tool_retrieval import TfidfIndex, catalogue, evaluate


def todo():
    return make_todo(random.Random(0))


def test_compaction_causes_duplicate_writes():
    assert run_episode(todo(), HarnessConfig()).duplicates > 0
    assert run_episode(todo(), HarnessConfig(context_budget=1000)).duplicates == 0


@pytest.mark.parametrize("cfg", [HarnessConfig(write_log=True), HarnessConfig(check_world=True),
                                 HarnessConfig(idempotent=True)])
def test_each_fix_removes_duplicates(cfg):
    r = run_episode(todo(), cfg)
    assert r.duplicates == 0 and r.missing == 0 and r.finished


def test_expired_token_needs_the_harness():
    stuck = run_episode(todo(), HarnessConfig(token_ttl=5, write_log=True))
    fixed = run_episode(todo(), HarnessConfig(token_ttl=5, write_log=True, refresh_token=True))
    assert not stuck.finished and stuck.errors > 0
    assert fixed.finished and fixed.missing == 0


def test_retrieval_exact_name_and_recall_grows_with_k():
    tools = catalogue(500)
    index = TfidfIndex(tools)
    assert index.top_k(tools[7].name.replace("_", " ") + " " + tools[7].description(), 1) == [7]
    small, large = evaluate(index, 3, False, 100, 0), evaluate(index, 32, False, 100, 0)
    assert small["all_three_retrieved"] <= large["all_three_retrieved"]
    assert large["all_three_retrieved"] <= large["per_tool_recall"]


@pytest.mark.network
def test_schema_token_count_with_real_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    n = len(tok(catalogue(10)[0].schema())["input_ids"])
    assert 60 < n < 200
