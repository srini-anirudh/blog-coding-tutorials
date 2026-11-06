import random

import torch

from groundpref.metrics import expected_calibration_error, risk_coverage, score_response, summarise
from groundpref.pairs import build_pairs
from groundpref.prefopt import VARIANTS, corrupt_images, train_pref
from groundpref.reward import audit, train_rm
from groundpref.model import TinyVLM
from groundpref.world import Claim, Scene


def test_chair_and_recall_on_a_hand_example():
    s = Scene({"car": Claim("car", "red", "one"), "person": Claim("person", "white", "two")}, "street")
    rows = [score_response(s, [Claim("car", "red", "one"), Claim("dog", "white", "one")]),
            score_response(s, [Claim("person", "black", "two")])]
    r = summarise(rows)
    assert r["chair_i"] == 1 / 3 and r["chair_s"] == 0.5 and r["recall"] == 0.5 and r["colour_err"] == 0.5
    assert summarise([score_response(s, [])])["chair_i"] == 0.0  # silence scores perfectly


def test_effective_reliability():
    conf = torch.tensor([0.9, 0.9, 0.2, 0.6])
    correct = torch.tensor([True, False, False, True])
    r = risk_coverage(conf, correct, tau=0.5, c=4.0)
    assert r["coverage"] == 0.75 and abs(r["eff_reliability"] - (1 - 4 + 1) / 4) < 1e-6
    assert expected_calibration_error(torch.tensor([1.0, 1.0]), torch.tensor([True, True])) == 0.0


def test_pairs_and_corruptions_have_the_promised_shape():
    p = build_pairs(40, 0, "fact")
    assert all(len(a) == len(b) for a, b in zip(p.yw, p.yl))
    s = build_pairs(40, 0, "style")
    assert all(len(b) > len(a) for a, b in zip(s.yw, s.yl))
    for mode in ("crop_small", "crop_large", "unrelated"):
        assert corrupt_images(p, mode, 0).shape == p.imgs.shape


def test_preference_training_and_audit_run_end_to_end():
    torch.manual_seed(0)
    ref = TinyVLM(d=16, layers=1, heads=2).eval()
    pairs = build_pairs(32, 1, "style")
    pol, rows = train_pref(ref, pairs, VARIANTS["mdpo"], steps=3, bs=8, log=lambda *a: None)
    assert rows[-1]["step"] == 2
    rm = train_rm(build_pairs(32, 2, "fact"), steps=3, bs=8)
    a = audit(rm, build_pairs(30, 3, "fact"))
    assert 0 <= a["pair_acc"] <= 1 and a["sensitivity"] >= 0 and a["swap_pairs"] > 0
