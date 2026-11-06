import random

import pytest

from toollab.loop import fold
from toollab.verify import gates, reward
from toollab.world import (Example, Vocab, encode_result, execute, make_example, make_tools, parse_call,
                           render_call, split_names)


def pool(n=32, seed=0):
    train, _ = split_names(seed=0)
    return make_tools(n, train, random.Random(seed))


def test_vocab_has_no_collisions_and_round_trips():
    v = Vocab()
    ex = make_example(pool(), random.Random(1))
    toks = ex.prompt_tokens() + ex.target_tokens()
    assert v.decode(v.encode(toks)) == toks


def test_heldout_names_never_appear_in_training_tools():
    train, held = split_names(seed=0)
    assert not set(train) & set(held)
    tools = make_tools(1000, train, random.Random(3))
    used = {t.name for t in tools} | {f for t in tools for f, _ in t.fields}
    assert not used & set(held)


def test_parse_inverts_render():
    rng = random.Random(2)
    for _ in range(50):
        ex = make_example(pool(), rng)
        call = parse_call(ex.target_tokens())
        if ex.gold is None:
            assert call == {"none": True}
        else:
            assert call == {"name": ex.gold.name_str, "args": ex.args}
            assert execute(call, ex.tools).startswith(ex.gold.purpose)


def test_parse_rejects_malformed_spans():
    assert parse_call(["ka", "zo", "(", "be"]) is None
    assert parse_call(["paris", "(", ")"]) is None


def test_execute_raises_on_unknown_tool_and_wrong_type():
    t = pool()[0]
    with pytest.raises(KeyError):
        execute({"name": "nope", "args": {}}, [t])
    bad = {f: "#3" if typ != "count" else "paris" for f, typ in t.field_types().items()}
    with pytest.raises(TypeError):
        execute({"name": t.name_str, "args": bad}, [t])


def test_fold_keeps_all_state_in_the_context():
    seen = []
    replies = iter(["CALL a", "CALL b", "done"])

    def model(ctx):
        seen.append(ctx)
        return next(replies)

    tr = fold(model, "x0|", lambda s: {"n": s[5:]} if s.startswith("CALL") else None,
              lambda c: c["n"].upper(), lambda r: f"[{r}]")
    assert tr.context == "x0|CALL a[A]CALL b[B]done"
    assert seen == ["x0|", "x0|CALL a[A]", "x0|CALL a[A]CALL b[B]"]  # each pass sees only the string
    assert [s.call is None for s in tr.steps] == [False, False, True]


def test_fold_writes_errors_back_as_text():
    tr = fold(lambda c: "go" if "error" not in c else "stop", "", lambda s: {} if s == "go" else None,
              lambda c: 1 / 0, lambda r: f"<{r}>", max_steps=3)
    assert "error: division by zero" in tr.context and tr.context.endswith("stop")


def test_gates_and_reward():
    rng = random.Random(4)
    ex = next(e for e in (make_example(pool(), rng) for _ in range(100)) if e.gold is not None)
    good = ex.target_tokens()
    assert gates(ex, good) == "kept" and reward(ex, good)["r"] == 3.0
    assert gates(ex, ["ka", "("]) == "format" and reward(ex, ["ka", "("])["r"] == 0.0
    wrong_type = dict(ex.args)
    f, typ = next(iter(ex.gold.field_types().items()))
    wrong_type[f] = "#3" if typ != "count" else "paris"
    out = render_call(ex.gold, wrong_type) + ["<eos>"]
    assert gates(ex, out) == "execution" and reward(ex, out)["r"] == 1.0
    other = dict(ex.args)
    from toollab.world import VALUES
    other[f] = next(v for v in VALUES[typ] if v != ex.args[f])
    out = render_call(ex.gold, other) + ["<eos>"]
    assert gates(ex, out) == "semantic" and reward(ex, out)["r"] == 2.0
    none_ex = Example(ex.tools, ["weather"], None)
    assert gates(none_ex, ["NONE", "<eos>"]) == "kept"
    assert encode_result("x") == "<result> x </result>"
