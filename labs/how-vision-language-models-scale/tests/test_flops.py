import pytest
import torch

from vlmscale.flops import SETTINGS, count_step, measured_encoder_share
from vlmscale.budget import encoder_share
from vlmscale.model import TwoTowerConfig, TwoTowerVLM, pixel_shuffle, pool_grid


@pytest.fixture(scope="module")
def counts():
    torch.manual_seed(0)
    cfg = TwoTowerConfig(image_size=32, patch_size=4, v_dim=64, merge=2, l_dim=96, vocab=50, max_text=8)
    model = TwoTowerVLM(cfg)
    px, txt = torch.randn(2, 3, 32, 32), torch.randint(0, 50, (2, 8))
    return cfg, {name: count_step(model, px, txt, txt, *flags) for name, flags in SETTINGS.items()}


def test_k_values(counts):
    _, c = counts
    assert c["all trained"]["llm"].k == pytest.approx(6)
    assert 5.8 < c["all trained"]["vision"].k < 6  # patch embedding skips the pixel gradient
    assert c["frozen encoder"]["vision"].k == pytest.approx(2)
    assert c["frozen encoder + LLM (connector only)"]["llm"].k == pytest.approx(4)
    assert c["frozen LLM, encoder trains"]["llm"].k == pytest.approx(4)


def test_attention_not_saved_by_freezing_llm(counts):
    _, c = counts
    assert c["frozen encoder + LLM (connector only)"]["llm"].attention == c["all trained"]["llm"].attention
    assert c["frozen encoder"]["vision"].attention == pytest.approx(c["all trained"]["vision"].attention / 3)


def test_encoder_share_formula(counts):
    cfg, c = counts
    both = c["all trained"]
    f = encoder_share(both["vision"].params, both["llm"].params, cfg.merge, 8 / cfg.visual_tokens)
    assert measured_encoder_share(both) == pytest.approx(f, abs=0.01)


def test_pixel_shuffle_and_pool_shapes():
    x = torch.randn(2, 64, 8)
    assert pixel_shuffle(x, 2).shape == (2, 16, 32)
    assert pool_grid(x, 4).shape == (2, 4, 8)
    assert torch.allclose(pool_grid(x, 1)[:, 0], x.mean(1), atol=1e-6)
    assert torch.equal(pool_grid(x, 64), x)


def test_pixel_shuffle_matches_c07():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[3] / "c07-vlm" / "tinyvlm" / "projector.py"
    if not path.exists():
        pytest.skip("C7 is not in this checkout")
    spec = importlib.util.spec_from_file_location("c07_projector", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    x = torch.randn(3, 196, 5)
    assert torch.equal(pixel_shuffle(x, 2), mod.pixel_shuffle(x, 2))
