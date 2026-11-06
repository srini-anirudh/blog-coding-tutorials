import pytest
import torch

from gallery.data import CELL, GRID, IMG, K, VOCAB, GlyphPages, glyph_templates
from gallery.fusion import TinyLM, attention_cost
from gallery.model import TinyVLM
from gallery.train import evaluate, fresh_copy, set_trainable, train_vlm


@pytest.fixture
def lm():
    torch.manual_seed(0)
    return TinyLM(VOCAB, d=32, layers=2, heads=2)


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    return GlyphPages().visual_batch(4, "fine", g)


def test_data_answers_are_right():
    t = glyph_templates()
    assert len({tuple(x.flatten().tolist()) for x in t}) == K
    data = GlyphPages(noise=0.0)
    g = torch.Generator().manual_seed(1)
    ids = data.layout(3, "fine", g)
    img = data.render(ids, g)
    assert img.shape == (3, 1, IMG, IMG)
    cells = img.view(3, GRID, CELL, GRID, CELL).permute(0, 1, 3, 2, 4).reshape(3, GRID * GRID, CELL * CELL)
    decoded = torch.cdist(cells, t.view(K, -1)).argmin(-1)
    assert torch.equal(decoded, ids)
    q, ans = data.question(ids, "fine", g)
    r, c = q[:, 1] - 8, q[:, 2] - 8
    assert torch.equal(ans, ids[torch.arange(3), r * GRID + c])
    major_ids = data.layout(64, "coarse", g)
    _, major = data.question(major_ids, "coarse", g)
    assert ((major_ids == major[:, None]).sum(1) >= GRID * GRID // 2).all()


def test_gated_xattn_is_exact_copy_at_init_even_with_image(lm, batch):
    img, q, _ = batch
    vlm = TinyVLM(fresh_copy(lm), connector="mlp", fusion="xattn")
    with torch.no_grad():
        assert torch.equal(vlm(img, q), lm(q)[:, -1])


def test_visual_expert_keeps_text_path(lm, batch):
    img, q, _ = batch
    for rank in (0, 4):
        vlm = TinyVLM(fresh_copy(lm), connector="mlp", fusion="expert", expert_rank=rank)
        with torch.no_grad():
            assert torch.equal(vlm.lm(q), lm(q))


def test_plora_at_init_equals_prefix(lm, batch):
    img, q, _ = batch
    torch.manual_seed(3)
    prefix = TinyVLM(fresh_copy(lm), connector="mlp", fusion="prefix")
    torch.manual_seed(3)
    plora = TinyVLM(fresh_copy(lm), connector="mlp", fusion="expert", expert_rank=4)
    with torch.no_grad():
        assert torch.allclose(prefix(img, q), plora(img, q), atol=1e-6)


def test_full_expert_doubles_projection_params(lm):
    base = sum(p.numel() for p in lm.parameters())
    vlm = TinyVLM(fresh_copy(lm), fusion="expert")
    proj = sum(p.numel() for b in lm.blocks for m in b.projections() for p in m.base.parameters())
    assert sum(p.numel() for p in vlm.lm.parameters()) == base + proj


def test_encoder_free_and_sequence_lengths(lm, batch):
    img, q, _ = batch
    enc = TinyVLM(fresh_copy(lm), fusion="encfree")
    assert enc.encoder is None and enc(img, q).shape == (4, VOCAB)
    assert enc.lm_sequence_length(3) == GRID * GRID + 3
    x = TinyVLM(fresh_copy(lm), connector="pixel_shuffle", fusion="xattn")
    assert x.lm_sequence_length(3) == 3


def test_frozen_schedule_leaves_lm_unchanged(lm):
    vlm = TinyVLM(fresh_copy(lm), connector="mlp", fusion="expert", expert_rank=4)
    before = {k: v.clone() for k, v in vlm.lm.state_dict().items() if "lora" not in k}
    enc_before = {k: v.clone() for k, v in vlm.encoder.state_dict().items()}
    info = train_vlm(vlm, ("fine",), "connector", steps=3, batch=8)
    after = vlm.lm.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(torch.equal(enc_before[k], v) for k, v in vlm.encoder.state_dict().items())
    assert info["trainable_params"] == sum(p.numel() for p in vlm.connector.parameters()) + sum(
        p.numel() for n, p in vlm.lm.named_parameters() if "lora" in n)
    assert 0.0 <= evaluate(vlm, "fine", True, n=64, batch=32) <= 1.0


def test_all_schedule_trains_lm_but_not_encoder(lm):
    vlm = TinyVLM(fresh_copy(lm), connector="avg_pool", fusion="prefix")
    params = set_trainable(vlm, "all")
    assert not any(p.requires_grad for p in vlm.encoder.parameters())
    assert len(params) == len(list(vlm.connector.parameters())) + len(list(vlm.lm.parameters()))


def test_attention_cost():
    p = attention_cost(64, 576, 128, 2, "prefix")
    x = attention_cost(64, 576, 128, 2, "xattn")
    assert p["seq_len"] == 640 and x["seq_len"] == 64
    assert x["attn_macs"] < p["attn_macs"]
    assert x["kv_positions"] == p["kv_positions"]
