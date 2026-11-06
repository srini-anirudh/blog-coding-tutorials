import random
from collections import Counter

import numpy as np

from vlmalign.blind import fit_text_probe
from vlmalign.preference import VALUES, build_pairs, wrong_value
from vlmalign.scenes import OFF_PALETTE, PALETTE, render, sample_scene
from vlmalign.tasks import KINDS, ask


def test_scene_is_deterministic_and_identities_unique():
    a, b = sample_scene(7), sample_scene(7)
    assert a.objects == b.objects and a.word == b.word
    for s in range(50):
        scene = sample_scene(s)
        ids = [(o.shape, o.color) for o in scene.objects]
        assert len(ids) == len(set(ids))
    assert np.array_equal(np.asarray(render(a)), np.asarray(render(b)))


def test_render_draws_the_object_colour_at_its_centre():
    scene = sample_scene(3)
    img = render(scene)
    for o in scene.objects:
        cx, cy = map(int, o.center)
        # triangles are thin at the top, so probe slightly below the centre
        assert img.getpixel((cx, cy + 10)) == PALETTE[o.color]


def test_answers_match_the_scene():
    rng = random.Random(0)
    for s in range(100):
        scene = sample_scene(s)
        for kind in KINDS:
            q = ask(scene, kind, rng)
            if q is None:
                continue
            if kind == "count":
                shape = q.prompt.split()[2][:-1]
                assert int(q.answer) == scene.count(shape)
            if kind == "ocr":
                assert q.answer == scene.word


def test_negatives_are_always_false():
    rng = random.Random(0)
    for p in build_pairs(100, "balanced", seed=1) + build_pairs(100, "biased", seed=2):
        assert p.chosen != p.rejected
        assert wrong_value(p.question, rng, p.mode) != p.question.answer


def test_balanced_pairs_are_exchangeable():
    """Every value appears about equally often on the chosen and the rejected side."""
    pairs = build_pairs(600, "balanced", seed=3, kinds=("count",))
    chosen = Counter(p.question.answer for p in pairs)
    rejected = Counter(p.rejected.rstrip(".").split()[-1] for p in pairs)
    for v in VALUES["count"]:  # 200 expected each; the difference has std ~16
        assert abs(chosen[v] - rejected[v]) < 50


def test_biased_colour_negatives_use_colours_never_drawn():
    for p in build_pairs(50, "biased", seed=4, kinds=("color",)):
        assert p.rejected.rstrip(".").split()[-1] in OFF_PALETTE


def test_text_probe_sees_bias_but_not_balance():
    biased = fit_text_probe(build_pairs(400, "biased", seed=5), build_pairs(200, "biased", seed=6))
    balanced = fit_text_probe(build_pairs(400, "balanced", seed=5), build_pairs(200, "balanced", seed=6))
    assert biased.test_acc > 0.85
    assert abs(balanced.test_acc - 0.5) < 0.12
