"""Preprocessing, pixel shuffle, contrastive losses and Patch n' Pack."""

import pytest
import torch
import torch.nn.functional as F

from vencoder import contrastive as C
from vencoder import packing as P
from vencoder import preprocess as pre
from vencoder.shuffle import avg_pool_tokens, pixel_shuffle, pixel_unshuffle


# Preprocessing
def test_no_antialias_aliases_thin_strokes():
    rep = pre.aliasing_report(pre.stroke_page(2000, 3), 168)
    assert rep["torch bicubic antialias=False"]["output_std"] > 0.2
    assert rep["PIL bicubic (antialiased)"]["output_std"] < 0.02


def test_norm_mismatch_is_identity_for_same_constants():
    pix = torch.rand(3, 4, 4)
    for used, exp in [("ImageNet", "SigLIP (0.5)"), ("OpenAI CLIP", "ImageNet")]:
        seen = pre.normalise(pix, *pre.NORMS[used])
        want = pre.normalise(pix, *pre.NORMS[exp])
        for c, (g, b) in enumerate(pre.norm_mismatch(used, exp)):
            assert torch.allclose(seen[c], g * want[c] + b, atol=1e-5)


def test_exif_and_alpha():
    data = pre.exif_rotated_jpeg(64, 32, 6)
    assert pre.load(data, False).size == (64, 32) and pre.load(data, True).size == (32, 64)
    chart = pre.transparent_chart()
    assert pre.line_contrast(pre.flatten(chart, None), chart) < 0.05
    assert pre.line_contrast(pre.flatten(chart, (255, 255, 255)), chart) > 0.95


# Pixel shuffle
@pytest.mark.parametrize("side,r", [(32, 2), (32, 4), (27, 3)])
def test_pixel_shuffle_round_trip(side, r):
    x = torch.randn(2, side * side, 8)
    y = pixel_shuffle(x, r)
    assert y.shape == (2, side * side // r**2, 8 * r * r)
    assert torch.equal(pixel_unshuffle(y, r), x)


def test_pixel_shuffle_is_a_permutation_avgpool_is_not():
    x = torch.randn(1, 16, 3)
    assert torch.equal(pixel_shuffle(x, 2).flatten().sort().values, x.flatten().sort().values)
    assert avg_pool_tokens(x, 2).numel() == x.numel() // 4


def test_pixel_shuffle_groups_2x2_neighbours():
    x = torch.arange(16.0).reshape(1, 16, 1)  # a 4 x 4 grid, value = index
    first = set(pixel_shuffle(x, 2)[0, 0].tolist())
    assert first == {0.0, 1.0, 4.0, 5.0}


# Contrastive
def test_softmax_couples_cells_sigmoid_does_not():
    c = C.coupling(n=8)
    assert c["siglip"] == 0 and c["clip"] == 14  # the bumped cell's row and column


def test_chunked_siglip_equals_full():
    img = F.normalize(torch.randn(64, 16), dim=-1)
    txt = F.normalize(torch.randn(64, 16), dim=-1)
    t, b = torch.tensor(10.0), torch.tensor(-10.0)
    assert torch.allclose(C.siglip_loss(img, txt, t, b), C.siglip_loss_chunked(img, txt, t, b, 4), atol=1e-4)


def test_clip_loss_matches_manual():
    img = F.normalize(torch.randn(8, 4), dim=-1)
    txt = F.normalize(torch.randn(8, 4), dim=-1)
    logits = 5 * img @ txt.T
    manual = -(logits.log_softmax(1).diag().mean() + logits.log_softmax(0).diag().mean()) / 2
    assert torch.allclose(C.clip_loss(img, txt, torch.tensor(5.0)), manual, atol=1e-6)


def test_toy_training_learns():
    r = C.train("siglip", 64, 6400, C.ToyData(), eval_n=500)
    assert r["top1"] > 0.05  # chance is 0.002


# Patch n' Pack
def test_packed_equals_separate_and_unmasked_differs():
    torch.manual_seed(0)
    m = P.TinyViT().eval()
    imgs = [torch.rand(3, 16, 24), torch.rand(3, 32, 16)]
    with torch.no_grad():
        p = P.pack(m, imgs)
        y = m(p["x"], p["rows"], p["cols"], p["mask"])[0]
        z = m(p["x"], p["rows"], p["cols"], None)[0]
        for i, img in enumerate(imgs):
            ref = m.encode_one(img)
            assert torch.allclose(y[p["segment"] == i], ref, atol=1e-5)
            assert not torch.allclose(z[p["segment"] == i], ref, atol=1e-3)


def test_pad_mask_and_pooling():
    torch.manual_seed(0)
    m = P.TinyViT().eval()
    imgs = [torch.rand(3, 16, 16), torch.rand(3, 32, 32)]
    with torch.no_grad():
        b = P.pad_batch(m, imgs)
        y = m(b["x"], b["rows"], b["cols"], b["mask"])
        ref = m.encode_one(imgs[0])
        assert torch.allclose(y[0, :4], ref, atol=1e-5)
        assert torch.allclose(P.mean_pool(y, b["valid"])[0], ref.mean(0), atol=1e-5)
        assert not torch.allclose(P.mean_pool(y, None)[0], ref.mean(0), atol=1e-3)


def test_rope_2d_relative():
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def dot(a, b):
        qr = P.rope_2d(q, torch.tensor([[a[0]]]), torch.tensor([[a[1]]]))
        kr = P.rope_2d(k, torch.tensor([[b[0]]]), torch.tensor([[b[1]]]))
        return (qr * kr).sum()

    assert torch.allclose(dot((1, 2), (4, 6)), dot((11, 32), (14, 36)), atol=1e-4)
    assert not torch.allclose(dot((1, 2), (4, 6)), dot((1, 2), (6, 4)), atol=1e-3)


def test_first_fit_and_cost():
    lengths = [5, 4, 3, 3, 2, 1]
    bins = P.first_fit(lengths, 6)
    assert sorted(i for b in bins for i in b) == list(range(6))
    assert all(sum(lengths[i] for i in b) <= 6 for b in bins)
    c = P.packing_cost(lengths, batch=3, capacity=6)
    assert c["packed_waste"] < c["padded_waste"]


def test_pos_embed_interpolation_identity():
    pos = torch.randn(16 * 16, 8)
    assert torch.allclose(P.interpolate_pos_embed(pos, (16, 16)), pos, atol=1e-5)
    assert P.interpolate_pos_embed(pos, (20, 12)).shape == (240, 8)
