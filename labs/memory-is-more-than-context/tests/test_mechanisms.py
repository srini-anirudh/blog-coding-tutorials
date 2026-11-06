"""Offline tests: every mechanism except the real-model reader."""

from __future__ import annotations

import json
import random
from datetime import date as D

import pytest
import torch

from memlab.allocation import SELECTORS, Item, Pool, figure3_pool, random_pool, select_marginal, select_optimal
from memlab.bitemporal import AppendStore, BitemporalStore, ReplaceStore
from memlab.ladder import CONDITIONS, decompose, make_world, memory_for, prompt_for
from memlab.poisoning import e_star, iterate, simulate, wrong_fraction
from memlab.position import kv_case, positions_for
from memlab.scoring import World, fig11_rank, fit_logistic, naive_bayes_weights, retrieval_episodes
from memlab.staleness import precision_at_k, rho_append, rho_managed, simulate_store


# allocation
def test_marginal_skips_duplicate_and_contradiction():
    pool = figure3_pool()
    chosen = select_marginal(pool, 2300)
    assert 1 in chosen and 2 not in chosen and 3 not in chosen
    assert {5, 6} <= set(chosen)


def test_post_ranking_of_rules_at_default_budget():
    pool = figure3_pool()
    u = {n: pool.utility(f(pool, 2300)) for n, f in SELECTORS.items()}
    assert u["top-k relevance"] < u["density u/l"] < u["marginal gain"] <= u["exact optimum"]


def test_figure3_hole_and_fix():
    # As drawn, the optimum avoids the contradiction penalty by swapping #1 for its restatement #2.
    assert set(select_optimal(figure3_pool(), 2300)) == {0, 2, 3, 5, 6, 8}
    fixed = figure3_pool(consistent=True)
    assert sorted(select_marginal(fixed, 2300)) == sorted(select_optimal(fixed, 2300))


def test_additive_pool_greedy_respects_budget():
    pool = Pool([Item("a", 1.0, 10), Item("b", 0.5, 10), Item("c", 2.0, 100)])
    for f in SELECTORS.values():
        assert pool.tokens(f(pool, 25)) <= 25
    assert set(select_optimal(pool, 25)) == {0, 1}


def test_optimum_dominates_on_random_pools():
    rng = random.Random(1)
    for _ in range(20):
        p = random_pool(rng, n=8)
        best = p.utility(select_optimal(p, 300))
        for f in SELECTORS.values():
            assert p.utility(f(p, 300)) <= best + 1e-9


# staleness
def test_rho_append_limits():
    assert rho_append(0, 0.1) == 1.0
    assert rho_append(1000, 0.1) == pytest.approx(1 / 100, rel=1e-3)
    assert rho_managed(100, 0.12, 0.0) == pytest.approx(rho_append(100, 0.12))
    assert rho_managed(100, 0.12, 1.0) == pytest.approx(1.0)


def test_simulation_matches_closed_form():
    runs = [simulate_store(50, 6, 0.12, seed=s)[-1]["rho"] for s in range(10)]
    assert sum(runs) / len(runs) == pytest.approx(rho_append(50, 0.12), abs=0.02)
    runs = [simulate_store(50, 6, 0.12, 0.7, seed=s)[-1]["rho"] for s in range(10)]
    assert sum(runs) / len(runs) == pytest.approx(rho_managed(50, 0.12, 0.7), abs=0.03)


def test_precision_falls_with_store_size():
    small = precision_at_k(100, 5, 5, 1.0, 0.0, 0.4, trials=50)
    big = precision_at_k(5000, 5, 5, 1.0, 0.0, 0.4, trials=50)
    assert big < small


# poisoning
@pytest.mark.parametrize("rho,phi", [(0.25, 0.9), (0.5, 0.8), (2.0, 0.8), (4.0, 0.9)])
def test_closed_form_is_a_fixed_point(rho, phi):
    e = e_star(rho, phi, 0.15)
    assert e == pytest.approx(phi * wrong_fraction(e, rho) + (1 - phi) * 0.15, abs=1e-10)
    assert e == pytest.approx(iterate(rho, phi, 0.15), abs=1e-6)


def test_post_worked_numbers():
    assert e_star(1.0, 0.9, 0.15) == 0.15
    assert round(e_star(0.25, 0.9, 0.15), 2) == 0.02
    assert round(e_star(4.0, 0.9, 0.15), 2) == 0.89


def test_simulation_tracks_closed_form_away_from_one():
    for rho in (0.25, 4.0):
        sims = [simulate(rho, 0.8, 0.15, tasks=20000, seed=s)["error"] for s in range(4)]
        assert sum(sims) / 4 == pytest.approx(e_star(rho, 0.8, 0.15), abs=0.02)


# scoring
def test_logistic_fit_recovers_log_likelihood_ratios():
    w = World()
    g = torch.Generator().manual_seed(0)
    x, y = w.sample(60000, g)
    fit_w, fit_b = fit_logistic(x, y)
    nb_w, _ = naive_bayes_weights(x, y)
    assert torch.allclose(fit_w, w.true_weights(), rtol=0.15)
    assert torch.allclose(nb_w, w.true_weights(), rtol=0.1)
    assert fit_b == pytest.approx(w.true_bias(), rel=0.1)


def test_threshold_abstains_where_topk_cannot():
    w = World()
    g = torch.Generator().manual_seed(1)
    st = retrieval_episodes(w, w.true_weights(), w.true_bias(), 300, 30, 3, 0.0, 0.5, g)
    assert st["thresh_empty_on_unanswerable"] / st["unanswerable"] > 0.6
    assert st["topk_returned"] == 300 * 3


def test_fig11_mmr_demotes_duplicate():
    plain = [i for i, _ in fig11_rank(0.6, 0.25, 0.3, False)]
    mmr = [i for i, _ in fig11_rank(0.6, 0.25, 0.3, True)]
    assert plain.index(3) == 1 and mmr.index(3) > 4
    assert [i for i, _ in fig11_rank(1, 0, 0, False)][1] == 1  # semantic only: the stale fact is second


# bitemporal
def _filled():
    st = BitemporalStore()
    a = st.tell("user", "city", "Hyderabad", valid_from=D(2023, 1, 1), recorded_at=D(2023, 1, 1))
    st.tell("user", "city", "Bangalore", valid_from=D(2026, 7, 15), recorded_at=D(2026, 9, 10))
    return st, a


def test_supersede_keeps_both_clocks():
    st, _ = _filled()
    assert st.current("user", "city") == ["Bangalore"]
    assert st.as_of("user", "city", D(2026, 8, 1)) == ["Bangalore"]
    assert st.as_of("user", "city", D(2026, 8, 1), known_at=D(2026, 8, 1)) == ["Hyderabad"]
    assert st.as_of("user", "city", D(2024, 1, 1)) == ["Hyderabad"]


def test_append_and_replace_fail_differently():
    ap, rp = AppendStore(), ReplaceStore()
    for s in (ap, rp):
        s.tell("user", "city", "Hyderabad")
        s.tell("user", "city", "Bangalore")
    assert ap.current("user", "city") == ["Hyderabad", "Bangalore"]
    assert rp.current("user", "city") == ["Bangalore"]


def test_invalidate_reopens_previous_value():
    st = BitemporalStore()
    st.tell("p", "image", "v1", valid_from=D(2026, 1, 1), recorded_at=D(2026, 1, 1))
    bad = st.tell("p", "image", "v2", valid_from=D(2026, 3, 1), recorded_at=D(2026, 3, 1), source="model-inferred")
    st.invalidate(bad, at=D(2026, 4, 1))
    assert st.current("p", "image") == ["v1"]
    assert st.as_of("p", "image", D(2026, 3, 15), known_at=D(2026, 3, 15)) == ["v2"]
    assert st.as_of("p", "image", D(2026, 3, 15)) == ["v1"]


def test_delete_must_chase_derived():
    st, a = _filled()
    st.derive("user used to live in Hyderabad", [a])
    st.delete(a, chase_derived=False)
    assert "Hyderabad" in st.visible_text()
    st2, a2 = _filled()
    st2.derive("user used to live in Hyderabad", [a2])
    st2.delete(a2)
    assert "Hyderabad" not in st2.visible_text()


# position and ladder (no model)
def test_kv_case_places_needle():
    rng = random.Random(0)
    c = kv_case(10, 3, rng)
    body = c.prompt.split("JSON data:\n", 1)[1].split("\n\n", 1)[0]
    pairs = json.loads(body)
    assert list(pairs.values())[3] == c.answer and c.prompt.count(c.answer) == 1
    assert kv_case(10, 3, random.Random(0), restate=True).prompt.count(c.answer) == 2
    assert positions_for(30) == [0, 7, 14, 22, 29]


def test_ladder_conditions():
    notes, qs = make_world(0)
    rng = random.Random(0)
    q = next(q for q in qs if q.moved)
    assert memory_for("no_memory", q, notes, 3, rng) == []
    oracle = memory_for("oracle", q, notes, 3, rng)
    assert len(oracle) == 1 and q.gold in oracle[0].text
    assert all(n.project != q.project for n in memory_for("random", q, notes, 3, rng))
    retrieved = memory_for("retrieved", q, notes, 3, rng)
    assert len(retrieved) == 3 and sum(n.project == q.project for n in retrieved) == 2  # stale and current
    assert prompt_for(q, []).startswith("Question:")


def test_decompose_adds_up():
    acc = dict(zip(CONDITIONS, [0.1, 0.05, 0.6, 0.9, 0.5]))
    g = decompose(acc)
    assert g["value of right memories A4-A1"] == pytest.approx(g["delivered lift A3-A1"] + g["retrieval headroom A4-A3"])
    assert g["distraction cost A2-A1"] == pytest.approx(-0.05)
