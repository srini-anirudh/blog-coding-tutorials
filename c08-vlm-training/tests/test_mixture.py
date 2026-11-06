import math
import random

import pytest

from vlmtrain.data import TASK_SIZES
from vlmtrain.mixture import MixtureSampler, epochs_per_task, mixture_weights, with_replay

SIZES = {"big": 900, "small": 100}


def test_proportional_and_uniform():
    assert mixture_weights(SIZES, "proportional") == {"big": 0.9, "small": 0.1}
    assert mixture_weights(SIZES, "uniform") == {"big": 0.5, "small": 0.5}


def test_temperature_formula_by_hand():
    w = mixture_weights(SIZES, "temperature", temperature=2.0)
    assert w["big"] == pytest.approx(30 / 40)  # sqrt(900) = 30, sqrt(100) = 10
    assert w["small"] == pytest.approx(10 / 40)


def test_temperature_limits():
    t1 = mixture_weights(TASK_SIZES, "temperature", temperature=1.0)
    prop = mixture_weights(TASK_SIZES, "proportional")
    assert all(t1[k] == pytest.approx(prop[k]) for k in prop)
    hot = mixture_weights(TASK_SIZES, "temperature", temperature=1e6)
    assert all(v == pytest.approx(1 / len(TASK_SIZES), abs=1e-4) for v in hot.values())


def test_higher_temperature_helps_the_smallest_task():
    shares = [mixture_weights(TASK_SIZES, "temperature", temperature=t)["doc"] for t in (1, 2, 4, 8)]
    assert shares == sorted(shares)


def test_manual_is_normalised_and_checked():
    assert mixture_weights(SIZES, "manual", manual={"big": 3, "small": 1}) == {"big": 0.75, "small": 0.25}
    with pytest.raises(ValueError):
        mixture_weights(SIZES, "manual", manual={"big": 1})


def test_replay_share_is_exact_and_rest_keeps_ratios():
    w = with_replay({"a": 0.75, "b": 0.25}, 0.2)
    assert w["text"] == pytest.approx(0.2)
    assert w["a"] / w["b"] == pytest.approx(3)
    assert sum(w.values()) == pytest.approx(1)


def test_sampler_matches_target_and_visits_every_index_once_per_epoch():
    w = {"a": 0.8, "b": 0.2}
    s = MixtureSampler(w, {"a": 1000, "b": 50}, seed=1)
    drawn = [s.draw() for _ in range(5000)]
    assert abs(s.observed()["a"] - 0.8) < 0.02
    first_b = [i for t, i in drawn if t == "b"][:50]
    assert sorted(first_b) == list(range(50))  # no repeat before the epoch ends


def test_sampler_resume_is_exact():
    s = MixtureSampler({"a": 0.5, "b": 0.5}, {"a": 97, "b": 89}, seed=3)
    [s.draw() for _ in range(40)]
    state = s.state_dict()
    next_ten = [s.draw() for _ in range(10)]
    r = MixtureSampler({"a": 0.5, "b": 0.5}, {"a": 97, "b": 89}, seed=99)
    r.load_state_dict(state)
    assert [r.draw() for _ in range(10)] == next_ten


def test_epochs_per_task():
    e = epochs_per_task({"a": 0.5}, {"a": 100}, 400)
    assert e["a"] == 2.0
    assert not math.isnan(e["a"])
