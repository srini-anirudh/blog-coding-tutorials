from harness_lab.loop_cost import (compaction_turn, edit_item_cost, effective_input, plain_turn, state_change_turn,
                                   tool_loading_turn, tool_order_turn, tool_tax, total_input)
from harness_lab.prefix_cache import PrefixCache, chain_hashes, first_divergence


def test_chain_hash_depends_on_everything_before():
    a = chain_hashes(list(range(64)), 16)
    b = chain_hashes([999] + list(range(1, 64)), 16)
    assert len(a) == 4 and all(x != y for x, y in zip(a, b))


def test_cache_reuses_whole_blocks_only():
    c = PrefixCache(16)
    c.insert(list(range(40)))
    assert c.lookup(list(range(40)) + [7, 8]) == 32
    assert first_divergence([1, 2, 3], [1, 2, 4]) == 2


def test_sent_tokens_match_closed_form():
    for K in (1, 7, 30):
        assert plain_turn(K, 1600, 320).sent == total_input(K, 1600, 320)


def test_full_cache_computes_only_new_tokens():
    assert plain_turn(30, 1600, 320).computed == 30 * 320
    assert plain_turn(30, 1600, 320, warm=False).computed == 1600 + 30 * 320


def test_effective_input_matches_at_the_ends():
    for h in (0.0, 1.0):
        assert plain_turn(20, 1600, 320, h=h).billed(0.1) == effective_input(20, 1600, 320, h, 0.1)


def test_edit_in_place_costs_more_than_append():
    app, edit = state_change_turn("append"), state_change_turn("edit")
    assert edit[19].computed > 10 * app[19].computed
    assert sum(q.computed for q in edit) > sum(q.computed for q in app)


def test_editing_item_six_recomputes_the_rest():
    q = edit_item_cost(40, 128, 6)
    assert q.computed == 35 * 128 and q.cached == 5 * 128


def test_tool_order_changes_break_the_cache():
    names = [f"t{i}" for i in range(8)]
    fixed = tool_order_turn([names] * 6)
    moving = tool_order_turn([names[i:] + names[:i] for i in range(6)])
    assert sum(q.computed for q in moving) > 3 * sum(q.computed for q in fixed)


def test_compaction_bounds_the_context():
    t = compaction_turn(100, 2000, 500, 12_000, 1000)
    assert t.compactions > 0 and max(q.total for q in t.requests) <= 12_000 + 500


def test_tool_tax():
    assert tool_tax(40, 300, 120) == 1_440_000
    assert tool_loading_turn("eager")[1] == 1_440_000
    assert tool_loading_turn("deferred")[1] < 20_000
