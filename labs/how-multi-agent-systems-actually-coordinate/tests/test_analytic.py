"""Voting, topology, scaling and the split lab: closed forms against the post and each other."""

import math
import random

import numpy as np
import pytest

from mascoord import scaling as S
from mascoord import topology as T
from mascoord import voting as V
from mascoord.splitlab import PRESETS, evaluate, verdict


def test_independent_search_numbers_from_post():
    assert V.p_any(5, 0.4) == pytest.approx(0.92, abs=0.005)


def test_vote_ceiling_matches_post_example():
    # p = 0.6, rho = 0.5: "no number of agents gets you past 80%"
    assert V.vote_ceiling(0.6, 0.5) == pytest.approx(0.8)
    assert V.vote_accuracy(1001, 0.6, 0.5) == pytest.approx(0.8, abs=1e-6)
    assert all(V.vote_accuracy(n, 0.6, 0.5) < 0.8 for n in range(1, 200, 2))


def test_rho_zero_is_plain_binomial_and_rho_one_is_one_agent():
    for n in [1, 3, 7]:
        assert V.vote_accuracy(n, 0.7, 0.0) == pytest.approx(V.p_majority(n, 0.7))
        assert V.vote_accuracy(n, 0.7, 1.0) == pytest.approx(0.7)


def test_p_majority_against_brute_force():
    for n, p in [(3, 0.6), (4, 0.6), (5, 0.3)]:
        brute = sum(np.prod([p if b else 1 - p for b in bits])
                    for bits in np.ndindex(*(2,) * n) if sum(bits) > n / 2)
        assert V.p_majority(n, p) == pytest.approx(brute)


def test_monte_carlo_matches_closed_form():
    sim = V.simulate(5, 0.6, 0.35, trials=60_000, seed=1)
    assert sim["strict_majority"] == pytest.approx(V.vote_accuracy(5, 0.6, 0.35), abs=0.01)
    assert sim["best_of_n"] == pytest.approx(V.best_of_n_accuracy(5, 0.6, 0.35), abs=0.01)
    assert sim["plurality"] >= sim["strict_majority"]


def test_voting_a_weak_agent_hurts_but_a_verifier_helps():
    assert V.vote_accuracy(25, 0.3, 0.2) < V.vote_accuracy(1, 0.3, 0.2)
    assert V.best_of_n_accuracy(25, 0.3, 0.2) > 0.75
    assert V.best_of_n_accuracy(10_000, 0.3, 0.2) == pytest.approx(V.vote_ceiling(0.3, 0.2), abs=1e-6)


def test_implied_rho_inverts_ceiling():
    assert V.implied_rho(V.vote_ceiling(0.65, 0.4), 0.65) == pytest.approx(0.4)


def test_mixed_team_reduces_to_binomial():
    assert V.mixed_team_accuracy(5, 0, 0.7, 0.5, 0.0, 0.0) == pytest.approx(V.p_majority(5, 0.7))
    assert V.mixed_team_accuracy(3, 2, 0.6, 0.6, 0.0, 0.0) == pytest.approx(V.p_majority(5, 0.6))


def test_table1_counts():
    post = {"solo": (0, 0, 1), "pipeline": (6, 6, 7), "star": (6, 12, 3), "hierarchy": (6, 12, 5),
            "full mesh": (21, 42, 2), "blackboard": (7, 14, "async")}
    for t in T.all_topologies(7):
        assert (t.channels, t.msgs_per_round, t.depth) == post[t.name]


def test_mesh_channels_quadratic_and_hierarchy_bounds_inbox():
    for n in [4, 8, 31]:
        assert T.mesh(n).channels == math.comb(n, 2)
        assert T.star(n).max_inbox() == n - 1
        assert T.hierarchy(n).max_inbox() <= 3


def test_comm_bill_post_numbers():
    b = T.comm_bill(8, 4, 1500)
    assert b["mesh_full"] == 336_000      # "roughly 340,000 tokens"
    assert b["blackboard"] < b["mesh_structured"] < b["mesh_full"]


def test_blast_radius_ordering():
    rng = np.random.default_rng(0)
    assert T.spread(T.pipeline(7), 6, 5, 1.0, rng) == 1           # last stage cannot reach upstream
    assert T.spread(T.star(7), 6, 2, 1.0, rng) == 7               # a worker reaches all via the coordinator
    assert T.spread(T.mesh(7), 0, 1, 1.0, rng) == 7
    assert T.spread(T.blackboard(7), 3, 2, 1.0, rng) == 7         # write, then everyone reads


def test_amdahl():
    assert S.amdahl_limit(0.8) == pytest.approx(5.0)
    assert S.amdahl_speedup(1e9, 0.8) == pytest.approx(5.0, rel=1e-6)
    nb = S.best_n(0.8, 0.01, 1.6)
    assert abs(nb - S.best_n_continuous(0.8, 0.01, 1.6)) < 1
    assert S.wall_time(nb, 0.8, 0.01, 1.6) < S.wall_time(nb + 5, 0.8, 0.01, 1.6)
    be = S.break_even_n(0.8, 0.01, 1.6)
    assert S.wall_time(be, 0.8, 0.01, 1.6) > 1.0 >= S.wall_time(be - 1, 0.8, 0.01, 1.6)


def test_split_lab_presets_match_captions():
    v = {s.name: verdict(evaluate(s), s) for s in PRESETS}
    assert v["the task fits"] == "Stay solo."
    assert v["multi-hop QA, same model x4"] == "Do not split."
    assert v["six-agent debate"] == "Do not split."
    assert v["frontier planner, cheap workers"] == "Split on economics."
    assert v["twenty agents, serial spine"] == "Past the optimum."
    assert v["good topology, bad messages"] == "Fix the protocol."
    assert evaluate(PRESETS[3])["quality"] > 1 and evaluate(PRESETS[4])["quality"] > 1


def test_split_lab_single_agent_is_neutral():
    r = evaluate(PRESETS[0])
    assert r["quality"] == pytest.approx(1.0) and r["depth"] == 0
