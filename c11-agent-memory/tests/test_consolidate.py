from agentmem.consolidate import consolidate, error_signature, find_recoveries
from agentmem.episodic import EpisodicStore
from agentmem.procedural import Procedure, ProcedureStore
from agentmem.records import Episode


def traj(task, entity, steps, session=1):
    return [Episode(f"deploy {entity}", a, o, r, session + i / 100, task, session, entity)
            for i, (a, o, r) in enumerate(steps)]


def failing(entity, err, fix):
    return [(f"deploy {entity}", f"...\n{err}: boom", -1.0), (f"clear_cache {entity}", "ok", 0.0),
            (f"deploy {entity}", f"{err}: boom", -1.0), (f"{fix} {entity}", "ok", 0.0),
            (f"deploy {entity}", "deployed", 1.0)]


def test_error_signature():
    assert error_signature("[7/7] rollout\nMigrationError: schema 41 != 42") == "MigrationError"
    assert error_signature("deployed atlas: healthy") is None


def test_find_recoveries_keeps_only_the_fix_that_worked():
    [rec] = find_recoveries(traj("t1", "atlas", failing("atlas", "MigrationError", "migrate")))
    assert (rec.trigger, rec.error, rec.fix) == ("deploy {entity}", "MigrationError", ("migrate {entity}",))


def test_consolidation_needs_several_entities():
    store = EpisodicStore()
    for e in traj("t1", "atlas", failing("atlas", "AuthError", "login")) + \
            traj("t2", "atlas", failing("atlas", "AuthError", "login"), session=2):
        store.add(e)
    assert consolidate(store) == []  # two episodes, one project: a quirk, not a rule
    for e in traj("t3", "cobalt", failing("cobalt", "AuthError", "login"), session=3):
        store.add(e)
    [proc] = consolidate(store)
    assert proc.support == 3 and proc.entities == ["atlas", "cobalt"]
    assert proc.instantiate("dune") == ["login dune"]
    assert "AuthError" in proc.text()


def test_procedure_store_merges_and_tracks_outcomes():
    ps = ProcedureStore()
    p = ps.add(Procedure("deploy {entity}", "AuthError", ["login {entity}"], 2, ["a", "b"]))
    q = ps.add(Procedure("deploy {entity}", "AuthError", ["login {entity}"], 3, ["c"]))
    assert p is q and len(ps) == 1 and p.support == 3 and p.entities == ["a", "b", "c"]
    assert p.success_rate == 0.5
    ps.record_outcome(p.id, True)
    ps.record_outcome(p.id, True)
    assert p.success_rate == 0.75
    assert ps.lookup("deploy {entity}", "AuthError") is p
