import types

import torch

from seeingtime.positions import (frequency_axes, inv_frequencies, mrope_angles, rotate, temporal_interval,
                                  text_then_video_then_text, timestamp_text, v2pe_positions, vision_position_ids)
from seeingtime.tubelet import FramePatchEmbed, TubeletEmbed, merge_2x2, token_count


def test_figure4_counts():
    v = torch.rand(4, 3, 56, 56)
    assert token_count(FramePatchEmbed(14, 8)(v)) == 64
    t = TubeletEmbed(14, 2, 8)(v)
    assert token_count(t) == 32 and token_count(merge_2x2(t)) == 8
    assert merge_2x2(t).shape[-1] == 32


def test_merge_groups_neighbours():
    x = torch.arange(16.0).reshape(4, 4, 1)
    m = merge_2x2(x)
    assert m[0, 0].tolist() == [0, 1, 4, 5] and m[1, 1].tolist() == [10, 11, 14, 15]


@torch.no_grad()
def test_image_as_two_frames_is_summed_kernel():
    tube = TubeletEmbed(14, 2, 16)
    img = torch.rand(3, 28, 42)
    a = tube.embed_image(img)[0]
    b = tube.as_frame_conv()(img[None])[0].permute(1, 2, 0)
    assert torch.allclose(a, b, atol=1e-5)


@torch.no_grad()
def test_tubelet_matches_hf_qwen2vl_patch_embed():
    from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchEmbed

    p, dim = 14, 16
    ours = TubeletEmbed(p, 2, dim)
    hf = PatchEmbed(patch_size=p, temporal_patch_size=2, in_channels=3, embed_dim=dim)
    hf.proj.weight.data = ours.proj.weight.data.clone()
    v = torch.rand(4, 3, 28, 42)
    # HF takes flattened tubes (C, T, p, p), ordered by temporal group, then row, then column.
    tubes = v.reshape(2, 2, 3, 2, p, 3, p).permute(0, 3, 5, 2, 1, 4, 6).reshape(-1, 3 * 2 * p * p)
    assert torch.allclose(ours(v).reshape(-1, dim), hf(tubes), atol=1e-5)


def test_vision_ids_match_hf_qwen25vl():
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    for fps in (1.0, 2.0, 4.0):
        interval = temporal_interval("abstime", fps)  # tokens_per_second 2 * (2 frames / fps)
        hf = Qwen2_5_VLModel.get_vision_position_ids(None, 7, torch.tensor([3, 4, 6]), 1, 2, interval)
        assert torch.equal(vision_position_ids(7, 3, 2, 3, interval), hf)
    assert temporal_interval("abstime", 2.0) == 2.0 and temporal_interval("index", 0.5) == 1.0


def test_text_tokens_reduce_to_1d_and_resume_after_video():
    pos = text_then_video_then_text(3, (2, 2, 2), 2, 1.0)
    assert torch.equal(pos[:, :3], torch.arange(3).expand(3, -1))
    assert (pos[:, -2:] == torch.tensor([5, 6])).all()


def _hf_layout(cls_attr, sections, head_dim=128):
    axes_tag = torch.stack([torch.full((head_dim // 2,), float(a)) for a in range(3)])[:, None, None]
    fake = types.SimpleNamespace(mrope_section=list(sections))
    return cls_attr(fake, axes_tag.clone())[0, 0, : head_dim // 2].long()


def test_frequency_axes_match_hf():
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLRotaryEmbedding
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    hf2 = _hf_layout(Qwen2VLRotaryEmbedding.recomposition_frequencies, (16, 24, 24))
    hf3 = _hf_layout(Qwen3VLTextRotaryEmbedding.recomposition_frequencies, (24, 20, 20))
    assert torch.equal(frequency_axes(128, (16, 24, 24), False), hf2)
    assert torch.equal(frequency_axes(128, (24, 20, 20), True), hf3)


def test_mrope_on_text_equals_1d_rope_and_depends_on_gap():
    axes, inv = frequency_axes(64, (8, 12, 12), False), inv_frequencies(64, 1e4)
    p = torch.arange(5)
    ang3 = mrope_angles(p.expand(3, -1), axes, inv)
    assert torch.allclose(ang3, p[:, None].float() * inv[None])
    q, k = torch.randn(64), torch.randn(64)

    def score(a, b):
        ang = mrope_angles(torch.tensor([[a, b], [0, 0], [0, 0]]), axes, inv)
        return float(rotate(q[None], ang[:1]) @ rotate(k[None], ang[1:]).T)

    assert abs(score(7, 4) - score(13, 10)) < 1e-4
    assert abs(score(7, 4) - score(8, 4)) > 1e-4


def test_timestamps_and_v2pe():
    assert timestamp_text(3) == "<3.0 seconds>"
    assert timestamp_text(3725.5, "hms") == "<01:02:05.5>"
    pos = v2pe_positions(torch.tensor([False, True, True, False]), 0.5)
    assert pos.tolist() == [0.0, 1.0, 1.5, 2.0]
