from types import SimpleNamespace

import torch

from vlmfam import positions as P


def _hf(segs, cls, tps=2):
    types, img, vid, secs = [], [], [], []
    for s in segs:
        if s[0] == "text":
            types += [0] * s[1]
            continue
        t, h, w = s[1]
        types += [1 if s[0] == "image" else 2] * (t * h * w // 4)
        (img if s[0] == "image" else vid).append([t, h, w])
        if s[0] == "video":
            secs.append(s[2] / tps)
    cfg = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2, tokens_per_second=tps))
    fake = SimpleNamespace(config=cfg, get_vision_position_ids=lambda *a, **k: cls.get_vision_position_ids(None, *a, **k))
    kw = {"second_per_grid_ts": torch.tensor(secs)} if "2_5" in cls.__module__ and secs else {}
    pos, _ = cls.get_rope_index(fake, torch.zeros(1, len(types), dtype=torch.long), torch.tensor([types]),
                                torch.tensor(img) if img else None, torch.tensor(vid) if vid else None, **kw)
    return pos[:, 0]


def test_mrope_matches_hf_qwen2_and_qwen25():
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

    segs = [("text", 3), ("image", (1, 4, 6)), ("text", 2), ("video", (4, 2, 2), 1), ("text", 1)]
    assert torch.equal(P.mrope_ids(segs), _hf(segs, Qwen2VLModel))
    segs = [("text", 3), ("video", (4, 4, 2), 4), ("text", 2)]
    assert torch.equal(P.mrope_ids(segs), _hf(segs, Qwen2_5_VLModel))


def test_image_uses_few_positions():
    ids = P.mrope_ids([("text", 2), ("image", (1, 8, 12)), ("text", 1)])
    assert ids.shape == (3, 2 + 24 + 1)
    assert P.positions_used(ids) == 2 + 6 + 1  # max(h, w) / merge = 6


def test_interleaving_matches_hf_qwen3():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    freq = torch.stack([torch.full((1, 1, 64), float(a)) for a in range(3)])
    hf = Qwen3VLTextRotaryEmbedding.recomposition_frequencies(SimpleNamespace(mrope_section=[24, 20, 20]), freq)
    assert hf[0, 0, :64].long().tolist() == P.axis_per_freq([24, 20, 20], interleaved=True)
    assert P.axis_per_freq([16, 24, 24], interleaved=False)[:17] == [0] * 16 + [1]


def test_v2pe():
    segs = [("text", 3), ("image", 8), ("text", 2)]
    assert torch.equal(P.v2pe_positions(segs, 1.0), torch.arange(13, dtype=torch.float64))
    pos = P.v2pe_positions(segs, 0.25)
    assert pos[-1].item() == 3 + 8 * 0.25 + 1
