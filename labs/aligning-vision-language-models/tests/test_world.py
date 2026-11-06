import random

import torch

from groundpref.metrics import score_response
from groundpref.world import (OBJECTS, clean_image, crop, decode, edit_colour, encode, hallucinate,
                              null_image, render, sample_scene, style_for)


def test_encode_decode_round_trip_ignores_style():
    rng = random.Random(0)
    for _ in range(50):
        s = sample_scene(rng)
        claims = s.claims()
        back, n_style = decode(encode(claims, style_for(len(claims), rng, 0.5)))
        assert back == claims
        assert n_style <= len(claims)


def test_each_corruption_adds_exactly_one_error_of_its_kind():
    rng = random.Random(1)
    for kind, key in [("object", "object"), ("object_swap", "object"), ("colour", "colour"), ("count", "count")]:
        for _ in range(30):
            s = sample_scene(rng)
            bad = hallucinate(s, s.claims(), kind, rng)
            errs = score_response(s, bad)
            assert errs[key] == 1
            assert sum(errs[k] for k in ("object", "colour", "count")) == 1
            if kind != "object":
                assert len(bad) == len(s.claims())


def test_crop_removes_the_evidence_of_one_slot_only():
    s = sample_scene(random.Random(2))
    gen = torch.Generator().manual_seed(0)
    img = render(s, gen, noise=0.0)
    k = OBJECTS.index(next(iter(s.facts)))
    out = crop(img, [k], gen, noise=0.0)
    assert out[k].abs().sum() == 0
    keep = [i for i in range(len(OBJECTS)) if i != k]
    assert torch.equal(out[keep], img[keep])
    assert torch.equal(null_image(), torch.zeros_like(img))


def test_edit_colour_changes_one_fact():
    s = sample_scene(random.Random(3))
    obj = next(iter(s.facts))
    new = "blue" if s.facts[obj].colour != "blue" else "red"
    e = edit_colour(s, obj, new)
    diff = (clean_image(e) - clean_image(s)).abs()
    assert diff.sum() == 2 and diff[OBJECTS.index(obj)].sum() == 2
