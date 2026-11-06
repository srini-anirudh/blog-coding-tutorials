from agentmem.embed import HashingEmbedder
from agentmem.writer import MemoryEvent, MemoryWriter


def decide(writer, text, source="user", **kw):
    return writer.should_store(MemoryEvent(text, source, 1.0, **kw))


def test_decisions_on_a_fixed_set_of_events():
    w = MemoryWriter(HashingEmbedder())
    fact = decide(w, "The deploy region for atlas is eu-west-1.", entities=["atlas"])
    assert fact.store and fact.scores.scope == "project"
    w.remember("The deploy region for atlas is eu-west-1.")
    assert not decide(w, "Thanks, that helps.").store                         # chatter: unstable
    assert not decide(w, "I think the owner for atlas might be chen.").store  # hedge: low confidence
    temp = decide(w, "Just for today, the release branch for atlas is trunk.", entities=["atlas"])
    assert not temp.store and temp.scores.scope == "session"
    again = decide(w, "The deploy region for atlas is eu-west-1.", entities=["atlas"])
    assert not again.store and "not novel" in again.reason
    rule = decide(w, "Remember: never deploy on Fridays.")
    assert rule.store and rule.scores.importance > 0.5


def test_an_edit_is_never_rejected_as_not_novel():
    w = MemoryWriter(HashingEmbedder())
    w.remember("Forget the owner for atlas.")
    text = "Forget the owner for atlas."
    assert not decide(w, text).store
    assert w.should_store(MemoryEvent(text, "user", 2.0), edits_memory=True).store


def test_tool_logs_score_low_on_stability():
    w = MemoryWriter(HashingEmbedder())
    log = decide(w, "12:01:33 INFO api request id=9f3a2b77c1d0 status=200 latency=41ms", source="tool")
    assert log.scores.stability < 0.2 and not log.store
