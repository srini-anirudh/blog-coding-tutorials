import pytest
import torch

from seeingtime.allocate import MemoryBank, best_contrast_kept, changed_patches, equal_allocation, slowfast_allocation
from seeingtime.controls import accuracy_under_controls, apply_control, t_grpo_bonus
from seeingtime.models import BagOfFrames, SpaceTimeCNN, TinyVideoTransformer
from seeingtime.synth import event_batch, qa_batch, render_times, speed_batch


def test_allocations_and_detail():
    eq, sf = equal_allocation(16, (8, 8)), slowfast_allocation(16, (4, 4), 4, (12, 16))
    assert eq.tokens == sf.tokens == 1024 and len(sf.frames) == 16
    frames = torch.zeros(1, 3, 8, 8)
    frames[..., :4] = 1.0
    left = lambda i: (slice(None), slice(None), slice(0, 4))
    right = lambda i: (slice(None), slice(None), slice(4, 8))
    assert best_contrast_kept(frames, equal_allocation(1, (8, 8)), left, right) == pytest.approx(1.0)
    assert best_contrast_kept(frames, equal_allocation(1, (1, 1)), left, right) == pytest.approx(0.0, abs=1e-6)


def test_memory_bank_stays_fixed_and_covers_time():
    bank = MemoryBank(4)
    for i in range(20):
        bank.add(torch.randn(10) if i % 5 == 0 else torch.ones(10), float(i))
    assert len(bank.feats) == 4 and sum(bank.counts) == 20
    assert bank.spans[0][0] == 0.0 and bank.spans[-1][1] == 19.0
    assert all(a[1] < b[0] for a, b in zip(bank.spans, bank.spans[1:]))


def test_pruning_drops_static_patches():
    frames = render_times([0.0, 1.0, 2.0])  # nothing moves before 8 s
    keep = changed_patches(frames, 6, 0.01)
    assert keep[0].all() and not keep[1:].any()


def test_bag_is_order_blind_and_transformer_is_not():
    torch.manual_seed(0)
    x, _ = qa_batch(4)
    heads = {"direction": 2}
    bag, vid = BagOfFrames(432, heads), TinyVideoTransformer(432, heads, "index")
    for m in (bag, vid):
        m.eval()
    rev = x.flip(1)
    assert torch.allclose(bag(x)["direction"], bag(rev)["direction"], atol=1e-5)
    assert not torch.allclose(vid(x)["direction"], vid(rev)["direction"], atol=1e-4)


@pytest.mark.parametrize("mode", ["none", "index", "abstime", "text"])
def test_transformer_time_modes_run_with_padding(mode):
    f, t, m, d = event_batch(3, (1, 4))
    out = TinyVideoTransformer(64, {"dur": 1}, mode)(f, t, m)["dur"]
    assert out.shape == (3, 1) and torch.isfinite(out).all()


def test_text_stamp_digits():
    model = TinyVideoTransformer(64, {"dur": 1}, "text")
    a = model.stamp(torch.tensor([12.5]))
    b = sum(e(torch.tensor([d])) for e, d in zip(model.digits, (1, 2, 5)))
    assert torch.allclose(a, b)


def test_synthetic_sets():
    x, y = speed_batch(6, (4, 8), "fps")
    assert x.shape == (6, 8, 6, 32) and y.max() < 3
    assert SpaceTimeCNN(3)(x).shape == (6, 3)
    x, y = qa_batch(500, bias=1.0)
    assert (y["direction"][y["colour"] == 0] == 1).all() and (y["direction"][y["colour"] == 2] == 0).all()
    f, t, m, d = event_batch(50, (2,))
    assert ((t[:, 1] - t[:, 0]) == 0.5).all() and d.min() >= 1 and d.max() <= 6


def test_controls():
    x = torch.arange(8.0).reshape(1, 8, 1)
    assert apply_control(x, "reversed")[0, :, 0].tolist() == list(range(7, -1, -1))
    assert (apply_control(x, "one_frame") == 4).all()
    assert sorted(apply_control(x, "shuffled")[0, :, 0].tolist()) == list(range(8))
    xs, ys = qa_batch(20)
    s = accuracy_under_controls(BagOfFrames(432, {"colour": 3}), xs, {"colour": ys["colour"]})
    assert set(s) == {"full", "one_frame", "shuffled", "reversed"}


def test_t_grpo_gate():
    good = t_grpo_bonus(torch.tensor([1, 1, 1, 0]), torch.tensor([1, 0, 0, 0]))
    assert torch.allclose(good, torch.tensor([0.3, 0.3, 0.3, 0.0]))
    assert t_grpo_bonus(torch.tensor([1, 1]), torch.tensor([1, 1])).sum() == 0
