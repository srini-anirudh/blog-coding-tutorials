import random

import pytest

from agentmem.budget import Block, BudgetError, ContextBudgeter, Section, truncate_left
from agentmem.compress import compact_history, edge_order, extract_lines, extractive_summary
from agentmem.tokens import count_tokens


def blocks(rng):
    out = [Block("instructions", "You are a deploy agent. Be careful.", required=True, compressible=False),
           Block("task", "goal: deploy atlas\n[ ] fix", required=True),
           Block("request", "Deploy atlas.", required=True, compressible=False)]
    for i in range(rng.randint(0, 12)):
        out.append(Block("memory", "- memory " + "word " * rng.randint(2, 40), value=rng.random(), order=i,
                         id=f"m{i}"))
    for i in range(rng.randint(0, 10)):
        out.append(Block("events", "event " * rng.randint(2, 30), value=rng.random(), order=i, id=f"e{i}"))
    out.append(Block("observation", "\n".join(f"log line {j} ok" for j in range(rng.randint(1, 80))),
                     value=0.9, id="obs"))
    return out


@pytest.mark.parametrize("seed", range(40))
def test_pack_never_exceeds_budget_and_keeps_required(seed):
    rng = random.Random(seed)
    bs = blocks(rng)
    budget = rng.randint(40, 900)
    packed = ContextBudgeter().pack(bs, budget)
    assert packed.total <= budget
    assert count_tokens(packed.render()) <= budget
    assert {"instructions", "task", "request"} <= {b.section for b in packed.blocks}
    for sec in ContextBudgeter().sections:
        if sec.max_tokens:
            assert packed.tokens_by_section.get(sec.name, 0) <= sec.max_tokens
    placed = len(packed.blocks) + len(packed.dropped)
    assert placed == len(bs)


def test_required_blocks_that_do_not_fit_raise():
    with pytest.raises(BudgetError):
        ContextBudgeter().pack([Block("instructions", "word " * 100, required=True, compressible=False)], 50)


def test_density_beats_raw_value():
    short = Block("memory", "- atlas region: eu-west-1", value=0.5, id="short")
    long = Block("memory", "- " + "filler " * 300 + "atlas region", value=0.9, compressible=False, id="long")
    b = ContextBudgeter([Section("memory", "Memory")])
    packed = b.pack([long, short], 60)
    assert [x.id for x in packed.blocks] == ["short"]


def test_oversized_observation_is_compressed_not_dropped():
    obs = Block("observation", "\n".join([f"[{i}/9] ok" for i in range(400)] + ["AssetError: missing"]),
                value=0.9, id="obs")
    packed = ContextBudgeter().pack([obs], 200)
    assert packed.compressed == ["obs"] and packed.blocks[0].text.endswith("AssetError: missing")


def test_render_order_is_by_section_then_order_not_by_score():
    bs = [Block("request", "REQ", required=True), Block("memory", "- low", value=0.1, order=0),
          Block("memory", "- high", value=0.9, order=1), Block("instructions", "INS", required=True)]
    text = ContextBudgeter().pack(bs, 200).render()
    assert text.index("INS") < text.index("- low") < text.index("- high") < text.index("REQ")


def test_truncate_left_keeps_the_newest():
    bs = [Block("events", f"turn {i}", order=i) for i in range(10)]
    kept = truncate_left(bs, 9)
    assert [b.text for b in kept] == ["turn 7", "turn 8", "turn 9"]


def test_extract_lines_keeps_the_failure():
    log = "\n".join([f"collected test_{i} PASSED" for i in range(300)] +
                    ["test_api.py::test_login FAILED", "AssertionError: 401 != 200"] +
                    [f"teardown {i}" for i in range(50)])
    out = extract_lines(log, 60)
    assert "AssertionError: 401 != 200" in out and "test_login FAILED" in out
    assert count_tokens(out) <= 60


def test_extractive_summary_keeps_latest_fact_and_drops_retracted():
    turns = ["The owner for atlas is priya.", "Heads up: we moved the owner for atlas to chen.",
             "The database for ember is redis.", "Forget the database for ember.", "Thanks, that helps."]
    s = extractive_summary(turns, 200)
    assert "atlas owner: chen" in s and "priya" not in s.split("\n")[0] and "ember database" not in s


def test_compact_history_and_edge_order():
    evs = [Block("events", f"The owner for p{i} is x{i}.", value=0.5, order=i) for i in range(6)]
    out = compact_history(evs, keep_recent=2, max_summary_tokens=80)
    assert len(out) == 3 and out[0].text.startswith("(summary of 4 earlier events)")
    assert edge_order([1, 2, 3, 4, 5], key=float) == [5, 3, 1, 2, 4]
