import math

import pytest
import torch
import torch.nn.functional as F

from organs.block import BlockConfig, Decoder, sinusoid_table
from organs.train import TrainConfig, train


@pytest.mark.parametrize("placement", ["post", "pre", "sandwich", "postbranch"])
@pytest.mark.parametrize("pos", ["sinusoid", "learned", "rope", "nope"])
def test_every_variant_runs(placement, pos):
    torch.manual_seed(0)
    model = Decoder(BlockConfig(d=32, layers=2, heads=2, max_seq=16, placement=placement, pos=pos,
                                ffn="gelu" if pos == "learned" else "swiglu", norm="layer" if pos == "nope" else "rms"))
    x = torch.randint(0, 256, (2, 16))
    loss = F.cross_entropy(model(x).flatten(0, 1), x.flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_post_ln_pins_the_stream_norm():
    torch.manual_seed(0)
    model = Decoder(BlockConfig(d=64, layers=6, heads=4, placement="post", norm="layer"))
    trace = []
    with torch.no_grad():
        model(torch.randint(0, 256, (2, 32)), trace=trace)
    for t in trace[1:]:
        assert t.norm(dim=-1).mean().item() == pytest.approx(math.sqrt(64), rel=1e-3)


def test_pre_ln_stream_grows_with_depth():
    torch.manual_seed(0)
    model = Decoder(BlockConfig(d=64, layers=12, heads=4, placement="pre"))
    trace = []
    with torch.no_grad():
        model(torch.randint(0, 256, (2, 32)), trace=trace)
    norms = [t.norm(dim=-1).mean().item() for t in trace]
    assert norms[-1] > 3 * norms[1]


def test_sinusoid_table_values():
    t = sinusoid_table(10, 8)
    assert t[0, 0] == 0 and t[0, 1] == 1
    assert t[3, 2] == pytest.approx(math.sin(3 / 10_000 ** (2 / 8)))


def test_qk_norm_caps_logits():
    torch.manual_seed(0)
    for qk in (False, True):
        model = Decoder(BlockConfig(d=32, layers=1, heads=2, max_seq=16, qk_norm=qk))
        with torch.no_grad():
            model.blocks[0].attn.qkv.weight.mul_(50)
            model(torch.randint(0, 256, (1, 16)))
        logit = model.blocks[0].attn.last_logit_max
        if qk:
            assert logit <= math.sqrt(16) + 1e-4  # |q.k| / sqrt(d_h) <= d_h / sqrt(d_h) with unit-RMS q, k
        else:
            assert logit > 20  # grows as 50^2 without the norm


def test_train_loop_reduces_loss():
    text = (b"the quick brown fox jumps over the lazy dog. " * 400)
    _, log = train(BlockConfig(d=32, layers=2, heads=2, max_seq=32), TrainConfig(steps=30, batch=8, seq=32, lr=3e-3,
                                                                              eval_every=30), text)
    assert log.losses[-1] < log.losses[0] - 1.0
    assert len(log.val) == 1 and len(log.first_layer_grad) == 30
