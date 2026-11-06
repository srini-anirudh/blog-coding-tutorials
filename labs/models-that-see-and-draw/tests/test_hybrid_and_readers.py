"""The Transfusion mask and losses, and the probes used to compare readers."""

import torch

from seedraw import data, hybrid
from seedraw.readers import CaptionHead, Judge, conv_encoder, linear_probe


def test_transfusion_mask_is_causal_outside_and_full_inside_the_image():
    m = hybrid.transfusion_mask(8, [(2, 5)])
    assert m[2, 4] and m[4, 2]  # patches see each other both ways
    assert not m[1, 2]  # text before the image cannot see it
    assert m[6, 3] and not m[3, 6]  # text after the image sees it; the image cannot see later text
    causal = hybrid.transfusion_mask(8, [(2, 5)], bidirectional_images=False)
    assert torch.equal(causal, torch.ones(8, 8, dtype=torch.bool).tril())


def _patch_outputs(model, xt):
    cap = data.caption_tokens(torch.tensor([1]), torch.tensor([0]))
    return model.forward_draw(cap, xt, torch.tensor([0.5]))[1]


def test_bidirectional_patches_see_later_patches_and_causal_ones_do_not():
    torch.manual_seed(0)
    xt = torch.randn(1, 16, 16)
    later = xt.clone()
    later[:, -1] += 1.0
    for bidi, should_change in [(True, True), (False, False)]:
        torch.manual_seed(0)
        model = hybrid.HybridModel(d=32, layers=1, bidirectional_images=bidi).eval()
        with torch.no_grad():
            changed = not torch.allclose(_patch_outputs(model, xt)[:, 0], _patch_outputs(model, later)[:, 0], atol=1e-6)
        assert changed == should_change


def test_read_layout_image_cannot_see_its_caption():
    torch.manual_seed(0)
    model = hybrid.HybridModel(d=32, layers=1).eval()
    x0 = torch.randn(2, 16, 16)
    t = torch.zeros(2)
    a = model.forward_read(data.caption_tokens(torch.tensor([0, 1]), torch.tensor([0, 1])), x0, t)
    b = model.forward_read(data.caption_tokens(torch.tensor([3, 2]), torch.tensor([1, 0])), x0, t)
    assert torch.allclose(a[1], b[1], atol=1e-6)  # patch outputs ignore the caption that follows
    assert torch.allclose(a[0][:, 0], b[0][:, 0], atol=1e-6)  # the shape prediction comes before the caption


def test_hybrid_losses_and_sampling_shapes():
    torch.manual_seed(0)
    d = data.shapes(8, seed=0)
    model = hybrid.HybridModel(d=32, layers=1)
    losses = hybrid.hybrid_losses(model, d["images"], d["shape"], d["side"], read_t_max=0.5)
    (losses["lm"] + 5 * losses["flow"]).backward()
    assert all(torch.isfinite(v) for v in losses.values())
    assert model.patch_in.weight.grad.abs().sum() > 0 and model.text_in.weight.grad.abs().sum() > 0
    imgs = hybrid.draw(model, torch.tensor([0, 1]), torch.tensor([1, 0]), steps=2)
    assert imgs.shape == (2, 1, 16, 16)
    shape, side = hybrid.read(model, d["images"][:3])
    assert shape.shape == (3,) and int(shape.max()) < 4 and int(side.max()) < 2


def test_linear_probe_on_separable_features():
    torch.manual_seed(0)
    y = torch.randint(0, 4, (400,))
    x = torch.nn.functional.one_hot(y, 4).float() + 0.1 * torch.randn(400, 4)
    assert linear_probe(x[:200], y[:200], x[200:], y[200:], 4, steps=200) > 0.95


def test_reader_modules_shapes():
    imgs = data.shapes(4, seed=0)["images"]
    f = conv_encoder()(imgs)
    assert f.shape == (4, 64, 4, 4)
    assert CaptionHead()(f).shape == (4, 4)
    s, sd = Judge()(imgs)
    assert s.shape == (4, 4) and sd.shape == (4, 2)
