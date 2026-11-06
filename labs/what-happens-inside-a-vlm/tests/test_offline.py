"""Offline tests: scenes, formulas, probes, and the intervention machinery on a tiny random Llama."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlmscope import scenes  # noqa: E402
from vlmscope import intervene as ivm  # noqa: E402
from vlmscope.intervene import Intervention, intervene  # noqa: E402
from vlmscope.metrics import choice, multimodal_gain, multimodal_leakage, relative_change, restoration, total_variation  # noqa: E402
from vlmscope.probe import knn_overlap, logistic_probe, ridge, ridge_predict  # noqa: E402


# ---- scenes ----------------------------------------------------------------------------------------------------
def test_token_mask_matches_cell():
    s = scenes.Shape("square", "red", 5 * 64, 2 * 64, 63)
    m = scenes.token_mask(s)
    assert m.sum() == 1 and m[2 * 8 + 5]


def test_dilate_adds_border_without_wrapping():
    m = np.zeros(64, bool)
    m[0] = True
    d = scenes.dilate(m)
    assert set(np.flatnonzero(d)) == {0, 1, 8, 9}


def test_random_scene_is_unambiguous():
    rng = random.Random(0)
    for _ in range(20):
        a, b = scenes.random_scene(rng, n=2)
        assert a.kind != b.kind and a.colour != b.colour
        assert not (scenes.pixel_mask(a) & scenes.pixel_mask(b)).any()


def test_minimal_pair_changes_only_the_object():
    rng = random.Random(1)
    shapes = scenes.random_scene(rng, n=2)
    new = next(c for c in scenes.COLOURS if c not in {s.colour for s in shapes})
    a = np.asarray(scenes.render(shapes))
    b = np.asarray(scenes.render(scenes.recolour(shapes, 0, new)))
    changed = (a != b).any(-1)
    assert changed.any()
    assert not (changed & ~scenes.pixel_mask(shapes[0])).any()


def test_overlap_and_count_scenes():
    rng = random.Random(2)
    labels = [scenes.circles_overlap(scenes.overlap_scene(rng, ov)) for ov in [True, False] * 20]
    assert 10 < sum(labels) < 30
    assert len(scenes.count_scene(rng, 4)) == 4


# ---- formulas ----------------------------------------------------------------------------------------------------
def test_mmstar_metrics():
    assert multimodal_gain(62, 44) == 18
    assert multimodal_leakage(44, 30) == 14
    assert multimodal_leakage(20, 30) == 0


def test_restoration_and_relative_change():
    assert restoration(0.9, 0.1, 0.9) == pytest.approx(1.0)
    assert restoration(0.1, 0.1, 0.9) == pytest.approx(0.0)
    assert relative_change(0.4, 0.8) == pytest.approx(-0.5)


def test_choice_and_tv():
    logits = torch.tensor([0.0, 5.0, 1.0, 9.0])
    pick, p = choice(logits, torch.tensor([0, 1, 2]))
    assert pick == 1 and p.sum() == pytest.approx(1.0)
    assert total_variation(p, p) == 0.0
    assert total_variation(torch.tensor([1.0, 0]), torch.tensor([0, 1.0])) == 1.0


# ---- probes ------------------------------------------------------------------------------------------------------
def test_logistic_probe_learns_and_control_does_not():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(400, 20, generator=g)
    y = (x[:, 0] > 0).long()
    assert logistic_probe(x[:300], y[:300], x[300:], y[300:], 2) > 0.95
    y_rand = torch.randint(0, 2, (400,), generator=g)
    assert logistic_probe(x[:300], y_rand[:300], x[300:], y_rand[300:], 2) < 0.7


def test_ridge_recovers_linear_map():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, 8, generator=g)
    w = torch.randn(8, 3, generator=g)
    y = x @ w + 2.0
    assert torch.allclose(ridge_predict(ridge(x, y, lam=1e-8), x), y, atol=1e-3)


def test_knn_overlap_identity_and_random():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(200, 16, generator=g)
    assert knn_overlap(a, a * 3).mean() == 1.0
    assert knn_overlap(a, torch.randn(200, 16, generator=g)).mean() < 0.3


# ---- interventions on a tiny random language model ---------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    ref = LlamaForCausalLM(cfg).eval()
    lab = LlamaForCausalLM(cfg).eval()
    lab.load_state_dict(ref.state_dict())
    lab.config._attn_implementation = ivm.NAME
    for sub in lab.modules():
        if hasattr(sub, "config"):
            sub.config._attn_implementation = ivm.NAME
    ivm.install(lab.model)
    emb = ref.model.embed_tokens(torch.randint(0, 50, (1, 12), generator=torch.Generator().manual_seed(1)))
    return ref, lab, emb.detach()


def run(model, emb, iv=None):
    with intervene(iv or Intervention()), torch.no_grad():
        return model(inputs_embeds=emb).logits[0]


def test_lab_attention_matches_reference(tiny):
    ref, lab, emb = tiny
    assert torch.allclose(run(ref, emb), run(lab, emb), atol=1e-5)


def test_recorded_attention_is_causal_and_normalised(tiny):
    _, lab, emb = tiny
    iv = Intervention(record_attention=True)
    run(lab, emb, iv)
    a = iv.attentions[2][0]
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)
    assert a.triu(1).abs().max() == 0


def test_knockout_of_future_positions_is_a_no_op(tiny):
    _, lab, emb = tiny
    ko = {l: [(torch.tensor([3]), torch.tensor([7, 8]))] for l in range(4)}
    assert torch.allclose(run(lab, emb), run(lab, emb, Intervention(knockout=ko)), atol=1e-6)


def test_knockout_changes_only_later_positions(tiny):
    _, lab, emb = tiny
    ko = {l: [(torch.tensor([6]), torch.tensor([1, 2]))] for l in range(4)}
    base, cut = run(lab, emb), run(lab, emb, Intervention(knockout=ko))
    assert torch.allclose(base[:6], cut[:6], atol=1e-6)
    assert not torch.allclose(base[6], cut[6], atol=1e-4)


def test_full_patch_restores_clean_run(tiny):
    _, lab, emb = tiny
    corrupt = emb + 0.5 * torch.randn_like(emb)
    iv = Intervention(capture_residual=True)
    clean = run(lab, emb, iv)
    every = torch.arange(emb.shape[1])
    patched = run(lab, corrupt, Intervention(patch={1: (every, iv.residual[1][0])}))
    assert torch.allclose(clean, patched, atol=1e-5)
    patched_emb = run(lab, corrupt, Intervention(embed_patch=(every, emb[0])))
    assert torch.allclose(clean, patched_emb, atol=1e-5)
