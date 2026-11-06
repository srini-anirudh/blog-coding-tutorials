from agentmem import AgentMemory, Episode, MemoryEvent, Op, TaskState, count_tokens


def feed(mem, *texts):
    return [mem.observe(MemoryEvent(t, "user", float(i)))[1] for i, t in enumerate(texts)]


def test_supersede_removes_the_old_value_from_retrieval():
    mem = AgentMemory()
    ops = feed(mem, "The deploy region for atlas is eu-west-1.", "We moved the deploy region for atlas to us-east-2.")
    assert ops[0][0][0] is Op.ADD and ops[1][0][0] is Op.SUPERSEDE
    texts = [h.item.text for h in mem.retrieve("deploy region for atlas", 5.0)]
    assert "atlas deploy region: us-east-2" in texts and "atlas deploy region: eu-west-1" not in texts
    assert [f.status for f in mem.facts.history("atlas", "deploy region")] == ["superseded", "active"]


def test_append_only_ablation_keeps_both_values():
    mem = AgentMemory(supersede=False)
    feed(mem, "The deploy region for atlas is eu-west-1.", "We moved the deploy region for atlas to us-east-2.")
    texts = [h.item.text for h in mem.retrieve("deploy region for atlas", 5.0)]
    assert "atlas deploy region: us-east-2" in texts and "atlas deploy region: eu-west-1" in texts


def test_session_notes_live_for_one_session():
    mem = AgentMemory()
    feed(mem, "Just for today, the owner for atlas is marco.")
    assert mem.session_notes and not mem.facts.facts
    ctx = mem.build_context("I", "Who owns atlas?", 1.0, 200).render()
    assert "marco" in ctx
    mem.end_session()
    assert "marco" not in mem.build_context("I", "Who owns atlas?", 2.0, 200).render()


def test_retraction_passes_the_novelty_gate_and_leaves_the_index():
    mem = AgentMemory()
    feed(mem, "Forget the owner for atlas.", "The owner for atlas is priya.", "Forget the owner for atlas.")
    assert mem.facts.current("atlas", "owner") is None
    assert not any("priya" in it.text for it in mem.retriever.items.values())


def test_save_and_load_give_identical_retrieval(tmp_path):
    mem = AgentMemory()
    mem.working = TaskState("x")
    feed(mem, "The deploy region for atlas is eu-west-1.", "The owner for atlas is priya.", "Remember: never deploy on Fridays.")
    mem.record_step(Episode("deploy atlas", "deploy atlas", "AuthError: no", -1.0, 3.0, "t1", 1, "atlas"))
    mem.end_task("t1")
    mem.save(tmp_path / "m.json")
    back = AgentMemory.load(tmp_path / "m.json")
    q = "where does atlas deploy"
    assert [(h.item.id, round(h.score, 5)) for h in mem.retrieve(q, 5.0)] == [(h.item.id, round(h.score, 5)) for h in back.retrieve(q, 5.0)]
    assert len(back.episodes) == 1
    # A restored store still applies edits to the right fact.
    op, _ = back.observe(MemoryEvent("We moved the deploy region for atlas to us-east-2.", "user", 6.0))[1][0], None
    assert op[0] is Op.SUPERSEDE
    assert all("eu-west-1" not in h.item.text for h in back.retrieve(q, 7.0))


def test_end_session_promotes_a_recurring_recovery_to_a_rule():
    mem = AgentMemory()
    for i, proj in enumerate(("atlas", "cobalt")):
        task = f"t{i}"
        mem.record_step(Episode(f"deploy {proj}", f"deploy {proj}", "AuthError: rejected", -1.0, 1.0 + i, task, 1, proj))
        mem.record_step(Episode(f"deploy {proj}", f"login {proj}", "ok", 0.0, 1.1 + i, task, 1, proj))
        mem.record_step(Episode(f"deploy {proj}", f"deploy {proj}", "healthy", 1.0, 1.2 + i, task, 1, proj))
        mem.end_task(task)
    new = mem.end_session()
    assert [p.steps for p in new] == [["login {entity}"]]
    assert mem.procedures.lookup("deploy {entity}", "AuthError").support == 2
    # It stays one rule, re-indexed, when the next session consolidates again.
    assert mem.end_session() == []
    assert sum(it.kind == "procedural" for it in mem.retriever.items.values()) == 1
    assert any(h.item.kind == "procedural" for h in mem.retrieve("deploy dune AuthError", 3.0, k=3))


def test_build_context_stays_under_budget_for_every_budget():
    mem = AgentMemory()
    feed(mem, *[f"The owner for project{i} is person{i}." for i in range(30)])
    log = "\n".join(f"line {i} status=200" for i in range(200))
    for budget in range(40, 800, 20):
        packed = mem.build_context("Be brief.", "Who owns project3?", 9.0, budget,
                                   recent_events=[f"event {i}" for i in range(20)], observation=log)
        assert count_tokens(packed.render()) <= budget, budget
