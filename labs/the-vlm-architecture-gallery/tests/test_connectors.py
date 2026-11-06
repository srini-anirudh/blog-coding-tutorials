import importlib.util
from pathlib import Path

import pytest
import torch

from gallery.connectors import CONNECTORS, build_connector, pixel_shuffle

REPO = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name,kw,expected", [
    ("linear", {}, 64), ("mlp", {}, 64), ("pixel_shuffle", {"s": 2}, 16), ("pixel_shuffle", {"s": 4}, 4),
    ("avg_pool", {"k": 2}, 16), ("attn_pool", {"k": 2}, 16), ("resampler", {"n_queries": 5}, 5),
])
def test_shapes_and_token_count(name, kw, expected):
    c = build_connector(name, 32, 48, **kw)
    y = c(torch.randn(3, 64, 32), (8, 8))
    assert y.shape == (3, expected, 48)
    assert c.out_tokens(8, 8) == expected


def test_every_connector_registered_and_trainable():
    for name in CONNECTORS:
        c = build_connector(name, 16, 16)
        y = c(torch.randn(2, 16, 16), (4, 4))
        y.sum().backward()
        assert all(p.grad is not None for p in c.parameters())


def test_unknown_connector():
    with pytest.raises(ValueError):
        build_connector("nope", 4, 4)


def test_pixel_shuffle_rectangular_grid_and_layout():
    # patch (r, c) holds the value 10 r + c; the first token must hold the top-left 2x2 block
    h, w = 4, 6
    x = torch.tensor([[10.0 * r + c] for r in range(h) for c in range(w)]).unsqueeze(0)
    y = pixel_shuffle(x, (h, w), 2)
    assert y.shape == (1, 6, 4)
    assert y[0, 0].tolist() == [0, 1, 10, 11]
    assert y[0, 1].tolist() == [2, 3, 12, 13]
    assert y[0, 3].tolist() == [20, 21, 30, 31]


def test_pixel_shuffle_matches_c07():
    path = REPO / "c07-vlm" / "tinyvlm" / "projector.py"
    if not path.exists():
        pytest.skip("C7 not present")
    spec = importlib.util.spec_from_file_location("c07_projector", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    x = torch.randn(2, 196, 12)
    assert torch.equal(pixel_shuffle(x, (14, 14), 2), mod.pixel_shuffle(x, 2))


def test_resampler_output_size_is_fixed():
    r = build_connector("resampler", 16, 16, n_queries=7)
    for h, w in [(2, 2), (8, 8), (6, 10)]:
        assert r(torch.randn(1, h * w, 16), (h, w)).shape[1] == 7


def test_avg_pool_cannot_see_position_inside_window_but_shuffle_can():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 8)
    g = x.view(1, 4, 4, 8)
    sw = g.clone()
    sw[:, 0::2, 0::2], sw[:, 0::2, 1::2] = g[:, 0::2, 1::2], g[:, 0::2, 0::2]
    sw = sw.reshape(1, 16, 8)
    pool, shuf = build_connector("avg_pool", 8, 8), build_connector("pixel_shuffle", 8, 8)
    with torch.no_grad():
        assert torch.allclose(pool(x, (4, 4)), pool(sw, (4, 4)), atol=1e-6)
        assert not torch.allclose(shuf(x, (4, 4)), shuf(sw, (4, 4)), atol=1e-3)
