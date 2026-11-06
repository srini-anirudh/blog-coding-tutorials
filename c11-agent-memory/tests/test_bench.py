"""The evaluation has to be reproducible, and its baselines have to behave as claimed."""

from agentmem import AgentMemory
from agentmem.bench.deploy_eval import DeployPolicy, recovery_fix, run_deploy, summarise as sum_deploy
from agentmem.bench.deploy_world import QUIRKS, make_world
from agentmem.bench.facts_eval import (CorruptedMemory, FactsEvaluator, RuleReader, is_correct, run_facts,
                                       summarise)
from agentmem.bench.facts_world import make_script


def test_scripts_are_deterministic_and_have_every_probe_kind():
    a, b = make_script(5), make_script(5)
    assert [t.text for t in a.turns] == [t.text for t in b.turns]
    assert [(p.entity, p.attribute, p.gold) for p in a.probes] == [(p.entity, p.attribute, p.gold) for p in b.probes]
    kinds = {p.kind for s in range(6) for p in make_script(s).probes}
    assert kinds == {"stated", "updated", "rumor", "deleted", "today", "after_today", "unseen"}


def test_gold_never_depends_on_a_later_turn():
    """Causality of the benchmark: gold is computed from the turns a probe can see."""
    script = make_script(2)
    for p in script.probes:
        assert p.after_turn <= len(script.turns)
        assert all(t.session <= p.session for t in script.turns[:p.after_turn])


def test_reader_is_perfect_with_the_oracle_context():
    scripts = [make_script(s) for s in range(3)]
    rows = summarise(run_facts(scripts, 512, conditions=("oracle",)))
    assert rows["oracle"]["accuracy"] == 1.0


def test_reader_handles_each_phrasing():
    r = RuleReader()
    q = ("atlas", "owner", 3)
    assert r("[s1] user: The owner for atlas is priya.", *q) == "priya"
    assert r("[s1] user: The owner for atlas is priya.\n[s2] user: Forget the owner for atlas.", *q) == "unknown"
    assert r("[s1] user: The owner for atlas is priya.\n[s2] user: I think the owner for atlas might be chen.", *q) == "priya"
    assert r("[s1] user: The owner for atlas is priya.\n[s2] user: Just for today, the owner for atlas is chen.", *q) == "priya"
    assert r("[s1] user: The owner for atlas is priya.\n[s3] user: Just for today, the owner for atlas is chen.", *q) == "chen"
    assert r("- atlas owner: priya", *q) == "priya"
    assert r("nothing here", *q) == "unknown"


def test_answer_matching():
    assert is_correct("`pytest -x`.", "pytest -x") and is_correct("Unknown.", "unknown")
    assert not is_correct("priya", "chen") and not is_correct("priya", "unknown")


def test_transcript_respects_budget():
    script = make_script(0)
    from agentmem import count_tokens
    for probe, ctx in FactsEvaluator(512).contexts(script):
        assert count_tokens(ctx["full_transcript"]) <= 512


def test_memory_causes_accuracy_on_the_fact_suite():
    scripts = [make_script(s) for s in range(4)]
    rows = summarise(run_facts(scripts, 512))
    assert rows["retrieved"]["accuracy"] > rows["no_memory"]["accuracy"] + 0.2
    assert rows["retrieved"]["accuracy"] > rows["full_transcript"]["accuracy"]
    assert rows["oracle"]["accuracy"] == 1.0
    # Swap only the stored values: if accuracy holds, the reader is not using memory.
    bad = summarise(run_facts(scripts, 512, memory_factory=CorruptedMemory, conditions=("retrieved",)))
    assert bad["retrieved"]["accuracy"] < rows["retrieved"]["accuracy"] - 0.3


def test_deploy_oracle_is_the_lower_bound_and_memory_beats_nothing():
    rows = {}
    for cond in ("no_memory", "retrieved", "oracle"):
        res = []
        for s in range(3):
            res += run_deploy(make_world(s), cond, 1024)
        rows[cond] = sum_deploy(res)[cond]
    assert rows["oracle"]["success"] == 1.0
    assert rows["oracle"]["steps"] < rows["retrieved"]["steps"] < rows["no_memory"]["steps"]


def test_every_quirk_needs_its_own_fix():
    world = make_world(0)
    project, err = next((p, e) for p, e in world.quirks.items() if e)
    assert world.step(f"deploy {project}")[1] == -1.0
    world.step(f"clear_cache {project}")
    assert world.step(f"deploy {project}")[1] == -1.0
    world.step(f"{QUIRKS[err][0]} {project}")
    assert world.step(f"deploy {project}")[1] == 1.0
    world.new_session()                                # fixes expire with the session
    assert world.step(f"deploy {project}")[1] == -1.0


def test_policy_uses_past_only_through_context():
    policy = DeployPolicy()
    assert policy("", "s2t1", "atlas") == "deploy atlas"
    ctx = "episode s1t1 (session 1): `deploy atlas` -> AuthError; `login atlas` -> ok; `deploy atlas` -> ok"
    assert policy(ctx, "s2t1", "atlas") == "login atlas"
    assert recovery_fix([("deploy", "x", "AuthError"), ("login", "x", "ok"), ("deploy", "x", "ok")]) == ("AuthError", "login")
    rule = "rule: when `deploy {entity}` fails with AuthError, run `login {entity}` and retry"
    assert policy(rule + "\n[s2t1] `deploy dune` -> AuthError", "s2t1", "dune") == "login dune"


def test_memory_object_can_run_consolidation_through_the_world():
    mem = AgentMemory()
    run_deploy(make_world(0), "retrieved", 1024, memory=mem)
    assert len(mem.procedures) >= 1
