import torch
import torch.nn.functional as F

from pixsearch import gap
from pixsearch import index as I
from pixsearch import pages as P
from pixsearch import train as T
from pixsearch.models import maxsim


def test_infonce_gradient_matches_closed_form():
    g = torch.Generator().manual_seed(0)
    t = torch.randn(16, generator=g, requires_grad=True)
    v = torch.randn(10, 16, generator=g)
    tau = 0.1
    loss = F.cross_entropy((v @ t / tau)[None], torch.tensor([3]))
    loss.backward()
    assert torch.allclose(t.grad, T.info_nce_grad(t.detach(), v, 3, tau), atol=1e-5)


def test_easy_negatives_carry_almost_no_weight():
    t = F.normalize(torch.tensor([1.0, 0.0]), dim=0)
    v = torch.tensor([[1.0, 0.0], [0.9, 0.436], [-1.0, 0.0]])  # positive, hard, easy
    p = torch.softmax(v @ t / 0.05, 0)
    assert p[2] < 1e-15 < p[1]


def test_false_negative_mask():
    has = torch.zeros(3, P.N_COMBOS, dtype=torch.bool)
    has[0, [1, 2]] = True
    has[1, [1, 2]] = True   # page 1 also answers query 0
    has[2, [1]] = True
    qc = torch.tensor([[1, 2], [1, 2], [1, 5]])
    m = T.false_negative_mask(has, qc)
    assert m[0, 1] and m[1, 0] and not m[0, 0] and not m[0, 2]


def test_short_training_lowers_loss():
    torch.manual_seed(0)
    model = T.build("late", 16)
    hist = T.train_dual(model, steps=40, batch=16, lr=3e-3, warmup=5, log_every=10, log=lambda s: None)
    assert hist[-1]["loss"] < hist[0]["loss"]


def test_cross_training_runs():
    torch.manual_seed(0)
    model = T.build("cross")
    hist = T.train_cross(model, steps=4, batch=16, group=4, warmup=1, log_every=2, log=lambda s: None)
    assert len(hist) == 2


def test_storage_formula_reproduces_colpali():
    # 1,030 vectors x 128 dims x 2 bytes = 263,680 bytes = 257.5 KiB per page.
    assert I.storage_bytes(1, 1030, 128, 2) == 263_680
    assert I.storage_bytes(1, 1030, 128, 2) / 1024 == 257.5
    assert round(I.storage_bytes(1, 1030, 128, 2) / I.storage_bytes(1, 1, 2048, 2), 1) == 64.4


def test_merge_pool_and_prune_shapes():
    g = torch.Generator().manual_seed(0)
    x = F.normalize(torch.randn(4, 16, 8, generator=g), dim=-1)
    assert I.merge_pool(x, 3).shape == (4, 6, 8)
    assert I.prune(x, 4, "random").shape == (4, 4, 8)
    assert I.prune(x, 4, "norm", raw_norms=torch.rand(4, 16, generator=g)).shape == (4, 4, 8)


def test_merging_duplicates_loses_nothing():
    base = F.normalize(torch.randn(4, 8), dim=-1)
    x = base.repeat_interleave(4, dim=0)[None]       # 16 vectors, 4 distinct directions
    q = F.normalize(torch.randn(2, 3, 8), dim=-1)
    pooled = I.merge_pool(x, 4)
    assert torch.allclose(maxsim(q, pooled), maxsim(q, x), atol=1e-5)


def test_quantisers():
    x = F.normalize(torch.randn(10, 16, 32), dim=-1)
    assert (I.to_int8(x) - x).abs().max() < 0.01
    b = I.to_binary(x)
    assert torch.allclose(b.norm(dim=-1), torch.ones(10, 16))


def test_binary_rescore_with_full_depth_equals_float():
    g = torch.Generator().manual_seed(0)
    q = F.normalize(torch.randn(3, 4, 16, generator=g), dim=-1)
    d = F.normalize(torch.randn(12, 16, 16, generator=g), dim=-1)
    out = I.binary_then_rescore(q, d, k=12)
    assert torch.allclose(out, maxsim(q, d), atol=1e-5)


def test_shared_offset_cannot_change_a_single_modality_ranking():
    g = torch.Generator().manual_seed(0)
    q, v = torch.randn(5, 8, generator=g), torch.randn(20, 8, generator=g)
    assert gap.offset_preserves_ranking(q, v, 3 * torch.randn(8, generator=g))


def test_gap_stats_and_wrong_modality():
    g = torch.Generator().manual_seed(0)
    x = F.normalize(torch.randn(10, 8, generator=g), dim=-1)
    s = gap.gap_stats(x, x)
    assert abs(s["gap"]) < 1e-6 and abs(s["matched_cos"] - 1) < 1e-6
    # Text candidates on the query's side win when images carry an offset away from the queries.
    off = torch.zeros(8)
    off[0] = 5
    imgs = x + off
    txt = F.normalize(torch.randn(10, 8, generator=g), dim=-1) * 0.5 + x * 0.5
    assert gap.wrong_modality_rate(x, imgs, txt[torch.randperm(10, generator=g)]) > 50
    assert gap.image_pool_top1_accuracy(x, x) == 100
