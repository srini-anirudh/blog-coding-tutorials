import random

import numpy as np
import pytest
import torch

from actlab import grounding as G
from actlab import policy as P


def test_screen_has_unique_target():
    img, box, g = G.make_screen(random.Random(0))
    x, y, w, h = box
    patch = img[y:y + h, x:x + w]
    assert img.shape == (G.SCREEN, G.SCREEN, 3) and (patch != 238).any()


def test_label_roundtrip_and_null():
    box = (100, 200, 12, 12)
    idx, (dx, dy) = G.label_for(box, (0, 0, 384), 96)
    g = 96 // G.PATCH
    cx, cy = (idx % g + 0.5 + dx) * 384 / g, (idx // g + 0.5 + dy) * 384 / g
    assert abs(cx - 106) < 1e-6 and abs(cy - 206) < 1e-6
    assert G.label_for(box, (200, 0, 96), 48)[0] == (48 // G.PATCH) ** 2


def test_tiles_cover_every_icon():
    for side in (96, 192):
        ts = G.tiles(side)
        for x in range(0, G.SCREEN - G.ICON + 1, 7):
            assert any(tx <= x and x + G.ICON <= tx + side for tx, _, _ in ts)


def test_grounder_shapes_any_resolution():
    m = G.Grounder(ch=16)
    for r in (48, 96):
        logits, off = m(torch.rand(2, 3, r, r), torch.tensor([0, 1]))
        assert logits.shape == (2, (r // 8) ** 2 + 1) and off.shape == (2, (r // 8) ** 2, 2)


def test_grounder_learns_a_little():
    """A short run must beat chance on full screens at the largest budget (offline, tiny)."""
    torch.manual_seed(0)
    rng = random.Random(0)
    m = G.Grounder(ch=16)
    x, q, y, off = G.sample_batch(rng, 32, 384, 48, 1.0)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    first = None
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(m(x, q)[0], y)
        first = first or loss.item()
        opt.zero_grad(), loss.backward(), opt.step()
    assert loss.item() < 0.5 * first  # it can at least fit one batch


def test_policy_overfits_demos_and_runs_live():
    torch.manual_seed(0)
    O, C, A = P.expert_demos(20, seed=1, n_reports=1)
    pol = P.fit(P.PixelPolicy(ch=16), O, C, A, steps=300, batch=64, lr=3e-3)
    assert P.static_accuracy(pol, O, C, A) > 0.4  # chance is 1/145
    trajs = P.run_episodes(pol, P.make_envs(4, 3, 1), greedy=False, record=True)
    assert all(len(t.obs) == len(t.expert) == t.steps for t in trajs)


def test_noisy_expert_matches_no_recovery_formula():
    s = P.noisy_expert_success(0.05, n=400, n_reports=4, budget_factor=1.0)
    assert abs(s - 0.95 ** 20) < 0.08
    assert P.noisy_expert_success(0.05, n=100, n_reports=4, budget_factor=2.0) > 0.95


def test_index_action_roundtrip():
    env = P.make_envs(1, 0, 1)[0]
    for i in (0, 13, 143):
        assert P.action_to_index(P.index_to_action(i, env)) == i
    assert P.index_to_action(P.TYPE, env)[0] == "type"


@pytest.mark.slow
def test_reinforce_step_runs():
    pol = P.PixelPolicy(ch=16)
    _, hist = P.reinforce(pol, iters=2, episodes=8)
    assert len(hist) == 2
