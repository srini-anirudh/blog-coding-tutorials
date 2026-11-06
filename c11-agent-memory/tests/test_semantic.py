from agentmem.semantic import Claim, FactStore, LLMFactExtractor, Op, RuleFactExtractor


def c(value, t, conf=0.9, kind="assert"):
    return Claim("atlas", "deploy region", value, t, conf, kind=kind)


def test_add_update_supersede_delete_noop():
    s = FactStore()
    assert s.apply(c("eu-west-1", 1))[0] is Op.ADD
    op, f = s.apply(c("eu-west-1", 2))
    assert op is Op.UPDATE and f.support == 2 and f.confidence > 0.9 and f.t == 2
    assert s.apply(c("ap-south-1", 3, conf=0.4))[0] is Op.NOOP           # hedge loses to a confident fact
    assert s.current("atlas", "deploy region").value == "eu-west-1"
    assert len(s.conflicts()) == 1
    op, new = s.apply(c("us-east-2", 4))
    assert op is Op.SUPERSEDE and s.current("atlas", "deploy region") is new
    old = s.history("atlas", "deploy region")[0]
    assert old.status == "superseded" and old.valid_to == 4 and old.superseded_by == new.id
    assert s.apply(c(None, 5, kind="delete"))[0] is Op.DELETE
    assert s.current("atlas", "deploy region") is None
    assert s.apply(c(None, 6, kind="delete"))[0] is Op.NOOP                # nothing left to delete


def test_as_of_answers_with_what_was_true_then():
    s = FactStore()
    s.apply(c("eu-west-1", 1))
    s.apply(c("us-east-2", 4))
    assert s.as_of("atlas", "deploy region", 2).value == "eu-west-1"
    assert s.as_of("atlas", "deploy region", 4).value == "us-east-2"
    assert s.as_of("atlas", "deploy region", 0.5) is None


def test_store_json_roundtrip():
    s = FactStore()
    s.apply(c("eu-west-1", 1))
    s.apply(c("us-east-2", 4))
    t = FactStore.from_json(s.to_json())
    assert [f.value for f in t.facts] == ["eu-west-1", "us-east-2"]
    assert t.current("atlas", "deploy region").value == "us-east-2"


def test_rule_extractor_phrasings():
    ex = RuleFactExtractor()
    [a] = ex("The test command for dune is `tox -e py312`.", 1.0)
    assert (a.subject, a.attribute, a.value, a.confidence) == ("dune", "test command", "tox -e py312", 0.9)
    [b] = ex("Heads up: we moved the deploy region for atlas to us-east-2.", 1.0)
    assert b.value == "us-east-2"
    [h] = ex("I think the owner for atlas might be chen, not sure.", 1.0)
    assert h.confidence < 0.5
    [d] = ex("Forget the database for ember, we no longer use it.", 1.0)
    assert d.kind == "delete" and d.value is None
    [s] = ex("Just for today, the release branch for iris is trunk.", 1.0)
    assert s.scope == "session"
    [p] = ex("lumen's owner is amara.", 1.0)
    assert (p.subject, p.attribute, p.value) == ("lumen", "owner", "amara")
    assert ex("Thanks, that helps.", 1.0) == []


def test_llm_extractor_validates_model_output():
    good = LLMFactExtractor(lambda p: '```json\n[{"subject": "Atlas", "attribute": "owner", "value": "priya", '
                                     '"op": "assert", "confidence": 0.8}, {"subject": 3}]\n```')
    [claim] = good("atlas is owned by priya", 1.0)
    assert (claim.subject, claim.value, claim.confidence) == ("atlas", "priya", 0.8)
    assert LLMFactExtractor(lambda p: "not json at all")("x", 1.0) == []
    retract = LLMFactExtractor(lambda p: '[{"subject": "atlas", "attribute": "owner", "value": null, "op": "delete"}]')
    assert retract("forget atlas owner", 1.0)[0].kind == "delete"
