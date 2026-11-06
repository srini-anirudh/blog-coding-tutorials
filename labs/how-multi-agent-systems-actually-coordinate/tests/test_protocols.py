"""Record stores, coordinator protocols with a perfect reader, and the toy PARL bandit."""

import random

import numpy as np
import pytest

from mascoord import haystack as H
from mascoord import parl as P


@pytest.fixture
def rng():
    return random.Random(0)


def test_lookup_instance_is_unambiguous(rng):
    inst = H.make_lookup(64, rng)
    hits = [r for r in inst.records if f"Project {inst.target}:" in r]
    assert len(hits) == 1 and inst.answer in hits[0]
    assert len(set(inst.records)) == 64


def test_two_hop_instance_has_both_facts(rng):
    inst = H.make_two_hop(64, rng)
    assert any(f"Project {inst.target} is led by {inst.bridge}." == r for r in inst.records)
    assert any(r == f"{inst.bridge} works in room {inst.answer}." for r in inst.records)


def test_shards_partition_everything():
    recs = [str(i) for i in range(37)]
    for k in [1, 4, 5, 37]:
        sl = H.shards(recs, k)
        assert len(sl) == k and sum(sl, []) == recs


def test_parsers():
    assert H.parse_number("The code is 4821.") == "4821"
    assert H.parse_number("NONE") is None
    assert H.parse_name("It is Mira Okafor.", H.all_people()) == "Mira Okafor"


def test_aggregate_majority_and_conflict():
    assert H.aggregate([None, "12", "12", "7"]) == ("12", 2)
    assert H.aggregate([None, None]) == (None, 0)


def test_evidence_check_drops_hallucinated_values():
    sl = [["Project a-b: access code 1111."], ["Project c-d: access code 2222."]]
    assert H.checked(["1111", "9999"], sl, "a-b") == ["1111", None]


def test_oracle_protocols(rng):
    """A perfect reader: lookup works every way; two-hop needs one context or two rounds."""
    one_round = []
    for _ in range(30):
        li, th = H.make_lookup(64, rng), H.make_two_hop(64, rng)
        assert H.solo(li, H.oracle_ask)["correct"]
        assert H.split_one_round(li, H.oracle_ask, 4)["correct"]
        assert H.split_one_round(li, H.oracle_ask, 4, verify=True)["correct"]
        assert H.solo(th, H.oracle_ask)["correct"]
        two = H.split_two_round(th, H.oracle_ask, 4, H.all_people())
        assert two["correct"] and two["bridge_ok"] and two["rounds"] == 2
        one_round.append(H.split_one_round(th, H.oracle_ask, 4)["correct"])
    # one round only succeeds when both facts land in the same slice (about 1 in 4)
    assert 0 < np.mean(one_round) < 0.6


def test_parl_reward_landscape():
    r0 = P.success_prob(P.ACTIONS, 0) - P.TIME_WEIGHT * P.wall_clock(P.ACTIONS)
    r1 = P.success_prob(P.ACTIONS, 1) - P.TIME_WEIGHT * P.wall_clock(P.ACTIONS)
    assert r0.argmax() == 0                              # untrained: serial is best
    assert P.ACTIONS[r1.argmax()] == P.N_SUBTASKS         # trained: one subagent per subtask


def test_parl_anneal():
    sh = P.Shaping("x", lam_parallel=0.6, anneal_steps=100)
    assert P.lam_at(sh, 0) == 0.6 and P.lam_at(sh, 50) == pytest.approx(0.3) and P.lam_at(sh, 200) == 0


def test_serial_collapse_and_its_fix():
    perf_only = P.train(P.CONDITIONS[0], steps=600, seed=0)
    shaped = P.train(P.CONDITIONS[3], steps=1200, seed=0)
    never = P.train(P.CONDITIONS[1], steps=1200, seed=0)
    assert perf_only["probs"][0] > 0.9                   # serial collapse
    assert P.ACTIONS[shaped["probs"].argmax()] == 4       # learns to parallelise properly
    assert P.ACTIONS[never["probs"].argmax()] == 8        # a bonus that is never annealed is gamed
    assert P.expected_r_perf(shaped["probs"], shaped["skill"]) > P.expected_r_perf(never["probs"], never["skill"])
