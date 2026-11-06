import pytest
import torch
import torch.nn.functional as F

from seeingtime.controls import accuracy_under_controls, train
from seeingtime.models import SpaceTimeCNN, TinyVideoTransformer
from seeingtime.synth import qa_batch, speed_batch


@pytest.mark.slow
def test_video_model_needs_order_for_direction():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    heads = {"colour": 3, "direction": 2, "order": 2}
    model = TinyVideoTransformer(432, heads, "index")

    def batch_fn(step):
        x, y = qa_batch(128, g=g)
        return (x,), y

    train(model, batch_fn, 200, lr=2e-3, loss_fn=lambda o, y: sum(F.cross_entropy(o[q], y[q]) for q in heads))
    x, y = qa_batch(1000, g=torch.Generator().manual_seed(1))
    s = accuracy_under_controls(model, x, y)
    assert s["full"]["direction"] > 0.9 and s["shuffled"]["direction"] < 0.65
    assert s["reversed"]["direction"] < 0.2 and s["shuffled"]["colour"] > 0.9


@pytest.mark.slow
def test_uniform_sampling_breaks_on_long_videos():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    accs = {}
    for rule in ("uniform", "fps"):
        model = SpaceTimeCNN(3)

        def batch_fn(step, rule=rule):
            x, y = speed_batch(128, (4, 8), rule, g=g)
            return (x,), y

        train(model, batch_fn, 150, loss_fn=F.cross_entropy)
        x, y = speed_batch(1000, (16, 32), rule, g=torch.Generator().manual_seed(1))
        with torch.no_grad():
            accs[rule] = float((model(x).argmax(-1) == y).float().mean())
    assert accs["fps"] > 0.9 and accs["uniform"] < 0.5
