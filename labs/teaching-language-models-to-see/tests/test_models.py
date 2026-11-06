"""Gated cross-attention, the cost model, the gap metrics and the gradient-share tools, on C7's tiny random VLM."""

import copy

import pytest
import torch
from PIL import Image

from seelab.costs import PATTERNS, param_counts, predicted_bytes, predicted_flops, state_bytes
from seelab.gap import image_stats, nearest_rows, object_mask, object_token, text_reference
from seelab.gated import FlamingoLite
from seelab.signal import grad_patch_loss, shares, two_object_scene
from tinyvlm.data import draw_shape
from tinyvlm.testing import tiny_vlm


@pytest.fixture
def tiny():
    return tiny_vlm()


def text_batch(processor, n=3):
    texts = ["Two red circles.", "One blue square.", "Three green triangles."][:n]
    b = processor.tokenizer(texts, return_tensors="pt", padding=True)
    return b.input_ids, b.attention_mask


def test_zero_gates_leave_the_language_model_unchanged(tiny):
    vlm, processor = tiny
    lm = copy.deepcopy(vlm.lm)
    ids, mask = text_batch(processor)
    with torch.no_grad():
        before = lm(input_ids=ids, attention_mask=mask).logits
    fl = FlamingoLite(vlm.vision, lm, every=1, shuffle=1, heads=4)
    px = torch.randn(ids.shape[0], 3, 32, 32)
    with torch.no_grad():
        after = fl(ids, mask, pixel_values=px).logits
        assert torch.equal(before, after)
        fl.gated[0].xattn.alpha_attn.fill_(0.5)
        opened = fl(ids, mask, pixel_values=px).logits
        blind = fl(ids, mask, pixel_values=None).logits
    assert not torch.allclose(before, opened)
    assert torch.equal(before, blind)  # no picture: the gated blocks are skipped


def test_only_the_gates_get_gradient_at_step_zero(tiny):
    vlm, processor = tiny
    fl = FlamingoLite(vlm.vision, vlm.lm, every=1, shuffle=1, heads=4)
    ids, mask = text_batch(processor)
    fl(ids, mask, labels=ids, pixel_values=torch.randn(ids.shape[0], 3, 32, 32)).loss.backward()
    for g in fl.gated:
        assert g.xattn.alpha_attn.grad.abs() > 0
        for name, p in g.xattn.named_parameters():
            if not name.startswith("alpha"):
                assert p.grad is None or p.grad.abs().max() == 0
    assert all(p.grad is None for p in fl.lm.parameters() if not any(p is q for q in fl.trainable()))


def test_cost_rule_connector_only_still_pays_for_the_language_model():
    params = {"vision": 400, "projector": 20, "lm": 7000}
    tokens = {"vision": 576, "projector": 576, "lm": 676}
    full = predicted_flops(params, tokens, (True, True, True))
    conn = predicted_flops(params, tokens, PATTERNS["align: connector only (LLaVA stage 1)"])
    forward = sum(2 * params[p] * tokens[p] for p in params)
    assert forward < conn < full
    # vision: forward only. connector: forward + weight gradients (nothing upstream needs its input gradient).
    # LM: forward + activation gradients, no weight gradients -> 4/6 of the LM's full cost
    assert conn == pytest.approx(2 * 400 * 576 + 4 * 20 * 576 + 4 * 7000 * 676)
    assert predicted_bytes(params, (False, True, False)) == 2 * 400 + 16 * 20 + 2 * 7000


def test_measured_state_bytes_match_the_float32_rule(tiny):
    vlm, processor = tiny
    vlm.requires_grad_(False)
    vlm.projector.requires_grad_(True)
    opt = torch.optim.AdamW(vlm.projector.parameters())
    for p in vlm.projector.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    params = param_counts(vlm)
    assert state_bytes(vlm, opt) == predicted_bytes(params, (False, True, False), 16, 4)


def test_gap_metrics_on_text_rows():
    table = torch.randn(50, 16)
    st = image_stats(table[:10], table)
    assert st["nearest_cos"] == pytest.approx(1.0)  # every row is its own nearest neighbour
    ref = text_reference(table, n=20)
    assert ref["nearest_cos"] < 1.0
    _, idx = nearest_rows(table[:5] * 7.0, table)
    assert idx[:, 0].tolist() == [0, 1, 2, 3, 4]  # cosine ignores scale


def test_object_token_and_mask_find_the_shape():
    img = Image.new("RGB", (224, 224), (245, 245, 245))
    from PIL import ImageDraw

    draw_shape(ImageDraw.Draw(img), "square", 176, 16, 12, (40, 80, 220))  # top right cell of a 7 x 7 grid
    assert object_token(img) == 5
    m = object_mask(img, 7)
    assert m[5] and m.sum() <= 4


def test_two_object_scene_masks_are_separate():
    s = two_object_scene(3)
    a, b = s["masks"]
    assert a.any() and b.any() and not (a & b).any()
    assert s["objects"][0][0] != s["objects"][1][0]


def test_patch_loss_gradient_is_local_and_shares_sum_to_one():
    h = torch.randn(1, 196, 768, requires_grad=True)
    px = torch.randn(1, 3, 224, 224)
    dec = torch.nn.Linear(768, 768)
    g1 = grad_patch_loss(h, px, dec)
    px2 = px.clone()
    px2[:, :, :16, :16] += 1.0  # change only patch 0's target
    g2 = grad_patch_loss(h, px2, dec)
    diff = (g1 - g2).abs().sum(-1)[0]
    assert diff[0] > 0 and diff[1:].max() == 0
    masks = [torch.zeros(196, dtype=torch.bool), torch.zeros(196, dtype=torch.bool)]
    masks[0][:10] = True
    masks[1][100:110] = True
    sh = shares(g1[0], masks)
    assert sh["obj0"] + sh["obj1"] + sh["background"] == pytest.approx(1.0)
