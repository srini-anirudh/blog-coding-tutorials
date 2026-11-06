import random

import numpy as np
import pytest

from actlab.desk import CATEGORIES, DeskEnv, Layout, random_task, render
from actlab.observe import elements
from actlab.policy import action_to_index, index_to_action


def run_expert(env):
    while not env.done():
        env.step(env.expert_action())
    return env


@pytest.mark.parametrize("seed", range(20))
def test_expert_solves_with_and_without_popups(seed):
    rng = random.Random(seed)
    env = run_expert(DeskEnv(random_task(rng, 3), seed=seed, popup_prob=1.0))
    assert env.success()
    assert env.steps >= 15


def test_expert_takes_five_steps_per_report():
    env = run_expert(DeskEnv([("Meals", "7.25")], seed=1))
    assert env.steps == 5 and env.state.records == [("Meals", "7.25")]


def test_screenshot_never_shows_records():
    env = DeskEnv([("Travel", "42.50")], seed=3, clear_after_submit=False)
    run_expert(env)
    a = render(env.state)
    env.step(("click", *env.state.layout.submit.center))
    assert len(env.state.records) == 2 and not env.success()
    assert np.array_equal(a, render(env.state))


def test_popup_blocks_clicks_until_closed():
    env = DeskEnv([("Travel", "42.50")], seed=0)
    env.state.popup = True
    env.step(("click", *env.state.layout.amount.center))
    assert env.state.focus is None and env.state.popup
    env.step(("click", *env.state.layout.popup_close.center))
    assert not env.state.popup


def test_type_without_focus_does_nothing():
    env = DeskEnv([("Travel", "42.50")], seed=0)
    env.step(("type", "42.50"))
    assert env.state.amount == ""


def test_early_submit_opens_error_dialog():
    env = DeskEnv([("Travel", "42.50")], seed=0)
    env.step(("click", *env.state.layout.submit.center))
    assert env.state.dialog == "error" and not env.state.records


@pytest.mark.parametrize("seed", range(30))
def test_expert_labels_survive_grid_quantisation(seed):
    """Snapping each expert click to its 8 px grid cell centre must still hit the intended element."""
    rng = random.Random(seed)
    env = DeskEnv(random_task(rng, 2), seed=seed, popup_prob=1.0)
    while not env.done():
        a = env.expert_action()
        if a[0] == "click":
            snapped = index_to_action(action_to_index(a), env)
            want = [e["id"] for e in elements(env.state) if e["box"].contains(a[1], a[2])]
            got = [e["id"] for e in elements(env.state) if e["box"].contains(snapped[1], snapped[2])]
            assert want == got or (not want and env.state.popup), (a, snapped)
            a = snapped
        env.step(a)
    assert env.success()


def test_redesign_changes_name_id_and_place():
    a, b = Layout.sample(random.Random(0), False), Layout.sample(random.Random(0), False, redesign=True)
    assert (a.submit_name, a.submit_id) != (b.submit_name, b.submit_id)
    assert a.submit.center != b.submit.center
    assert set(CATEGORIES) == set(a.items)
