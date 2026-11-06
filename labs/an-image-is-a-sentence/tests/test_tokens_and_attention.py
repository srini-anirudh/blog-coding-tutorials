import torch
import torch.nn.functional as F

from vitfamily.attention import GPSA, MHSA, PoolingAttention, SRAttention
from vitfamily.blocks import DividedSTBlock, SwinBlock, ViTBlock
from vitfamily.data import augment
from vitfamily.models import TinyCNN, TinyViT
from vitfamily.patch import PatchEmbed, TubeletEmbed, merge_patches, patch_embed_matmul, pixel_unshuffle_order, resize_pos_table
from vitfamily.stats import mean_attention_distance, patch_distances


def test_patchify_is_a_strided_conv():
    pe = PatchEmbed(16, 32)
    x = torch.randn(2, 3, 64, 64)
    assert torch.allclose(pe(x), patch_embed_matmul(x, pe.proj), atol=1e-5)
    assert pe(x).shape == (2, 16, 32)


def test_patch_merging_is_pixel_unshuffle_reordered():
    x = torch.randn(1, 4, 6, 5)
    unshuf = F.pixel_unshuffle(x.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
    assert torch.equal(merge_patches(x)[..., pixel_unshuffle_order(5)], unshuf)


def test_tubelet_token_counts():
    clip = torch.zeros(1, 3, 4, 32, 32)
    assert TubeletEmbed(1, 16, 4)(clip).shape[1] == 16
    assert TubeletEmbed(2, 16, 4)(clip).shape[1] == 8


def test_resize_pos_table_holds_out_special_rows():
    table = torch.randn(1, 2 + 16, 8)
    good = resize_pos_table(table, 6, n_special=2)
    assert good.shape == (1, 2 + 36, 8)
    assert torch.equal(good[:, :2], table[:, :2])
    same = resize_pos_table(table, 4, n_special=2)  # resizing to the same grid is the identity
    assert torch.allclose(same, table, atol=1e-5)
    bug = resize_pos_table(table, 4, n_special=2, hold_out=False)
    assert torch.equal(bug[:, :2], table[:, -2:])  # the last patches became the special tokens
    assert torch.allclose(bug[:, 2:], table[:, :16], atol=1e-5)  # every patch read 2 slots late


def test_mhsa_fast_and_explicit_paths_agree():
    m = MHSA(32, 4)
    x = torch.randn(2, 10, 32)
    out, a = m(x, return_attn=True)
    assert torch.allclose(out, m(x), atol=1e-5)
    assert torch.allclose(a.sum(-1), torch.ones(2, 4, 10), atol=1e-5)


def test_sra_scores_are_n_by_n_over_r2():
    _, a = SRAttention(16, 2, 4)(torch.randn(1, 64, 16), (8, 8), return_attn=True)
    assert a.shape == (1, 2, 64, 4)


def test_pooling_attention_query_stride_downsamples():
    out, hw = PoolingAttention(16, 2, stride_q=2, stride_kv=4)(torch.randn(1, 64, 16), (8, 8))
    assert hw == (4, 4) and out.shape == (1, 16, 16)
    out, hw = PoolingAttention(16, 2, stride_q=1, stride_kv=2)(torch.randn(1, 64, 16), (8, 8))
    assert hw == (8, 8) and out.shape == (1, 64, 16)


def test_gpsa_positional_init_is_local():
    g = GPSA(36, 9, side=8)
    with torch.no_grad():
        g.gate.fill_(20.0)
        _, a = g(torch.randn(1, 64, 36), return_attn=True)
    assert torch.allclose(a.sum(-1), torch.ones(1, 9, 64), atol=1e-5)
    centre_head = a[0, 4, 3 * 8 + 3].view(8, 8)  # head with offset (0, 0), query (3, 3)
    assert centre_head.argmax() == 3 * 8 + 3
    assert centre_head[2:5, 2:5].sum() > 0.9
    assert mean_attention_distance(a, 8, 1).max() < 2.0


def test_vit_block_is_permutation_equivariant():
    blk = ViTBlock(16, 2).eval()
    x, perm = torch.randn(1, 12, 16), torch.randperm(12)
    assert torch.allclose(blk(x[:, perm]), blk(x)[:, perm], atol=1e-5)


def test_swin_block_shapes_and_shift():
    x = torch.randn(1, 14, 14, 24)
    assert SwinBlock(24, 2, 7, 0)(x).shape == x.shape
    assert SwinBlock(24, 2, 7, 3)(x).shape == x.shape


def test_divided_block_zero_init_equals_image_block():
    blk = DividedSTBlock(16, 2).eval()
    x = torch.randn(2, 3, 5, 16)
    assert torch.equal(blk(x), blk.spatial_only(x))


def test_tinyvit_variants_and_cnn():
    x = torch.randn(2, 3, 32, 32)
    assert TinyViT(dim=16, depth=2, heads=4)(x)["logits"].shape == (2, 10)
    assert TinyViT(dim=16, depth=2, heads=4, readout="gap")(x)["logits"].shape == (2, 10)
    o = TinyViT(dim=16, depth=2, heads=4, distill=True)(x)
    assert torch.allclose(o["logits"], (o["logits_cls"] + o["logits_dist"]) / 2)
    assert TinyViT(dim=16, depth=2, heads=4, gpsa_layers=2)(x)["logits"].shape == (2, 10)
    assert TinyViT(dim=16, depth=3, heads=4, gpsa_layers=2, distill=True)(x)["dist_feat"].shape == (2, 16)
    assert TinyCNN(width=8)(x)["logits"].shape == (2, 10)


def test_mean_attention_distance_limits():
    side = 4
    eye = torch.eye(side * side)[None, None]
    assert mean_attention_distance(eye, side, 16).item() == 0.0
    uniform = torch.full((1, 1, side * side, side * side), 1 / side**2)
    assert torch.isclose(mean_attention_distance(uniform, side, 16), patch_distances(side, 16).mean()).all()


def test_augment_keeps_shape_and_content_range():
    x = torch.randn(4, 3, 32, 32)
    y = augment(x)
    assert y.shape == x.shape
    assert y.min() >= x.min() and y.max() <= x.max()
