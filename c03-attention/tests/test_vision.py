"""Vision attention: patches, windows, shifted windows and 2D positions."""

import torch

from attnzoo.config import AttentionConfig
from attnzoo.gqa import GroupedQueryAttention
from attnzoo.pos2d import RelativePositionBias2D, Rope2D, sincos_2d_embedding
from attnzoo.vision import WindowAttention, patchify, shifted_window_mask, vit_attention_config, window_merge, window_partition


def test_patchify_shape_and_content():
    img = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)
    p = patchify(img, 4)
    assert p.shape == (2, 4, 48)
    torch.testing.assert_close(p[0, 1].view(3, 4, 4), img[0, :, 0:4, 4:8])  # patch 1 = top-right


def test_vit_attention_is_bidirectional():
    torch.manual_seed(0)
    attn = GroupedQueryAttention(vit_attention_config(32, 4)).eval()
    x = torch.randn(1, 16, 32)
    y = x.clone()
    y[:, -1] += 1.0  # change the last patch
    assert not torch.allclose(attn(x)[:, 0], attn(y)[:, 0])  # the first patch notices


def test_window_partition_roundtrip():
    x = torch.randn(2, 8, 12, 5)
    w = window_partition(x, 4)
    assert w.shape == (2 * 6, 16, 5)
    torch.testing.assert_close(window_merge(w, 4, 8, 12), x)


def test_one_window_covering_the_image_equals_global_attention():
    torch.manual_seed(0)
    win = WindowAttention(32, 4, window=4).eval()  # relative bias starts at zero
    glob = GroupedQueryAttention(vit_attention_config(32, 4)).eval()
    glob.load_state_dict(win.attn.state_dict())
    x = torch.randn(2, 4, 4, 32)
    torch.testing.assert_close(win(x), glob(x.view(2, 16, 32)).view(2, 4, 4, 32), atol=1e-5, rtol=1e-4)


def test_windows_are_independent_without_shift():
    torch.manual_seed(0)
    win = WindowAttention(16, 2, window=2).eval()
    x = torch.randn(1, 4, 4, 16)
    y = x.clone()
    y[:, 0, 0] += 1.0  # top-left window only
    diff = (win(x) - win(y)).abs().sum(-1)[0]
    assert diff[:2, :2].min() > 0 and diff[2:, :].max() == 0 and diff[:, 2:].max() == 0


def test_shifted_window_mask_blocks_wrapped_regions():
    m = shifted_window_mask(8, 8, window=4, shift=2)
    assert m.shape == (4, 16, 16)
    assert m[0].all()  # the top-left window holds no wrapped pixels
    assert not m[-1].all()  # the bottom-right window mixes four regions


def test_shifted_window_attention_runs_and_differs():
    torch.manual_seed(0)
    plain, shifted = WindowAttention(16, 2, window=2).eval(), WindowAttention(16, 2, window=2, shift=1).eval()
    shifted.load_state_dict(plain.state_dict())
    x = torch.randn(1, 4, 4, 16)
    assert shifted(x).shape == x.shape and not torch.allclose(plain(x), shifted(x))


def test_rope_2d_depends_on_relative_offset():
    rope = Rope2D(16, max_side=32)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(qr, qc, kr, kc):
        a = rope(q, torch.tensor([qr]), torch.tensor([qc]))
        b = rope(k, torch.tensor([kr]), torch.tensor([kc]))
        return (a * b).sum().item()

    assert abs(score(5, 3, 2, 1) - score(15, 13, 12, 11)) < 1e-4  # same (dy, dx) = (3, 2)
    assert abs(score(5, 3, 2, 1) - score(5, 3, 1, 2)) > 1e-3


def test_relative_bias_and_sincos_shapes():
    assert RelativePositionBias2D(4, 3)().shape == (1, 3, 16, 16)
    assert sincos_2d_embedding(4, 6, 32).shape == (24, 32)
