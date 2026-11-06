import numpy as np
import pytest
import torch

from pointlab.models import DecodeGrounder, SelectGrounder, TypedGrounder, bce_dice
from pointlab.toy import SIZE, connected_components, make_dataset, make_scene, proposals, subset
from pointlab.toy import _touch


def test_scene_masks_are_disjoint_and_separated():
    rng = np.random.default_rng(0)
    for _ in range(10):
        s = make_scene(rng, separate=True)
        assert len(s.masks) == 3 and min(m.sum() for m in s.masks) >= 25
        for i in range(3):
            for j in range(i + 1, 3):
                assert not (s.masks[i] & s.masks[j]).any()
                assert not _touch(s.masks[i], s.masks[j])


def test_proposer_finds_three_blobs_when_apart():
    d = make_dataset(20, seed=3, touching_share=0.0)
    props = proposals(d.image)
    assert all(len(p) == 3 for p in props)


def test_connected_components():
    fg = torch.zeros(1, 8, 8, dtype=torch.bool)
    fg[0, 0:2, 0:2] = True
    fg[0, 5:8, 5:8] = True
    ids = connected_components(fg)
    assert len(ids.unique()) == 3          # background + two blobs


def test_typed_tokens_round_trip():
    for fmt in ("loc", "digits"):
        m = TypedGrounder(fmt)
        box = torch.tensor([[3.2, 10.0, 40.5, 63.9]])
        k = m.to_bins(m.to_tokens(box))
        assert k.tolist() == [[50, 156, 632, 998]]


def test_models_run_and_learn():
    torch.manual_seed(0)
    d = make_dataset(8, seed=1)
    b = subset(d, torch.arange(8))
    m = DecodeGrounder()
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    first = None
    for _ in range(25):
        loss = bce_dice(m(b.image, b.query), b.mask)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first
    assert m(b.image, b.query).shape == (8, SIZE, SIZE)
    s = SelectGrounder().scores(b.image, b.query, proposals(b.image))
    assert s.shape[0] == 8
    box, logp = TypedGrounder("digits").predict(b.image, b.query, sample=True)
    assert box.shape == (8, 4) and logp.shape == (8,)


@pytest.mark.gpu
def test_training_loops_run_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from pointlab import train as T
    T.DEVICE = torch.device("cuda")
    d = make_dataset(64, seed=2)
    for fn in (lambda: T.train_typed("loc", d, 3), lambda: T.train_select(d, 3), lambda: T.train_decode(d, 3)):
        assert next(fn().parameters()).is_cuda
