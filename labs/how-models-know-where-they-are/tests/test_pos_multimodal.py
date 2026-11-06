"""M-RoPE ids and layouts, 2D grids, and Fourier features."""

from types import SimpleNamespace

import torch

from posenc import fourier, mrope, rope, vision2d
from posenc.c03 import Rope2D


def test_mrope_ids_match_transformers_qwen2_vl():
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

    segs = [("text", 5), ("image", 6, 3), ("text", 4), ("image", 2, 7), ("text", 1)]
    types, grids = [], []
    for s in segs:
        types += [0] * s[1] if s[0] == "text" else [1] * (s[1] * s[2])
        if s[0] == "image":
            grids.append([1, 2 * s[1], 2 * s[2]])
    fake = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    fake.get_vision_position_ids = lambda *a, **k: Qwen2VLModel.get_vision_position_ids(fake, *a, **k)
    ref, _ = Qwen2VLModel.get_rope_index(fake, torch.zeros(1, len(types), dtype=torch.long), torch.tensor([types]),
                                         image_grid_thw=torch.tensor(grids))
    assert torch.equal(mrope.position_ids(segs), ref[:, 0])


def test_text_only_mrope_is_1d_rope():
    f = rope.inv_freq(64, 1e6)
    x = torch.randn(1, 20, 64)
    for layout, sec in (("chunked", [8, 12, 12]), ("interleaved", [12, 10, 10])):
        out = mrope.apply_mrope(x, mrope.position_ids([("text", 20)]), f, mrope.axis_of_plane(sec, layout))
        torch.testing.assert_close(out, rope.apply_half(x, torch.arange(20), f))


def test_interleaved_layout_matches_qwen3_vl_recomposition():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    sec = [24, 20, 20]
    freq = torch.stack([torch.full((1, 1, 64), float(a)) for a in range(3)])  # each axis fills its own value
    fake = SimpleNamespace(mrope_section=sec)
    ref = Qwen3VLTextRotaryEmbedding.recomposition_frequencies(fake, freq)[0, 0, :64]
    assert torch.equal(ref.long(), mrope.axis_of_plane(sec, "interleaved"))


def test_bookkeeping_rules():
    segs = [("text", 4), ("image", 3, 4), ("text", 3)]
    good = mrope.position_ids(segs, "max")
    assert good[:, 16].tolist() == [8, 8, 8] and int(good[:, :16].max()) == 7
    bad = mrope.position_ids(segs, "last")
    assert bad[:, 16].tolist() == [5, 5, 5]


def test_absolute_time_ids_do_not_depend_on_fps():
    ids = {}
    for fps in (1, 4):
        t = mrope.vision_ids(0, 8 * fps // 2, 1, 1, mrope.qwen25_time_interval(fps))[0]
        ids[fps] = int(t[6 * fps // 2])
    assert ids[1] == ids[4]


def test_chunked_layout_starves_time_of_slow_planes():
    chunked = mrope.axis_wavelength_range([16, 24, 24], "chunked", 1e6)
    inter = mrope.axis_wavelength_range([24, 20, 20], "interleaved", 1e6)
    assert chunked["t"][1] < 1000 < inter["t"][1]


def test_axial_rope_depends_only_on_offset():
    torch.manual_seed(0)
    r = Rope2D(16)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def s(a, b):
        return float((r(q, torch.tensor([a[0]]), torch.tensor([a[1]])) * r(k, torch.tensor([b[0]]), torch.tensor([b[1]]))).sum())

    assert abs(s((1, 2), (4, 0)) - s((10, 9), (13, 7))) < 1e-4


def test_resize_pos_table_keeps_cls_and_checks_shape():
    t = torch.randn(1 + 16, 8)
    out = vision2d.resize_pos_table(t, 4, 6)
    assert out.shape == (37, 8) and torch.equal(out[0], t[0])
    try:
        vision2d.resize_pos_table(t, 4, 6, has_cls=False)
    except ValueError:
        pass
    else:
        raise AssertionError("a 17-row table is not a 4x4 grid")


def test_fourier_kernel_is_stationary_and_gaussian():
    B = fourier.gaussian_B(8192, 2, 2.0, seed=0)
    a, d = torch.tensor([[0.1, 0.2]]), torch.tensor([[0.05, 0.03]])
    k1, k2 = fourier.kernel(a, a + d, B), fourier.kernel(a + 0.4, a + 0.4 + d, B)
    torch.testing.assert_close(k1, k2, atol=1e-4, rtol=1e-4)
    assert abs(float(k1) - float(fourier.expected_kernel(d, 2.0))) < 0.03


def test_fourier_features_beat_raw_coordinates():
    img = fourier.test_image(32)
    _, raw, _ = fourier.fit_image(img, None, steps=150)
    _, ff, _ = fourier.fit_image(img, fourier.gaussian_B(64, 2, 5.0), steps=150)
    assert ff > raw
