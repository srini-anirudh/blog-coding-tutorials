import pytest
import torch
import torch.nn.functional as F

from twomod import Attention, Block, BlockConfig, SmallCNN, TextTransformer, ViT
from twomod.front import count_params, patchify, unpatchify
from twomod.masks import causal


@pytest.mark.parametrize("n_kv", [None, 2, 1])
@pytest.mark.parametrize("is_causal", [False, True])
def test_attention_matches_sdpa(n_kv, is_causal):
    torch.manual_seed(0)
    attn = Attention(BlockConfig(dim=32, n_heads=4, n_kv_heads=n_kv))
    x = torch.randn(2, 7, 32)
    ours = attn(x, causal(7) if is_causal else None)
    b, n, h, dh = 2, 7, 4, 8
    q = attn.q(x).view(b, n, h, dh).transpose(1, 2)
    k = attn.k(x).view(b, n, -1, dh).transpose(1, 2).repeat_interleave(h // attn.n_kv, 1)
    v = attn.v(x).view(b, n, -1, dh).transpose(1, 2).repeat_interleave(h // attn.n_kv, 1)
    ref = attn.out(F.scaled_dot_product_attention(q, k, v, is_causal=is_causal).transpose(1, 2).reshape(b, n, -1))
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


def test_block_is_permutation_equivariant_without_mask():
    block = Block(BlockConfig(dim=32, n_heads=4)).eval()
    x, perm = torch.randn(1, 10, 32), torch.randperm(10)
    torch.testing.assert_close(block(x[:, perm]), block(x)[:, perm], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cfg", [BlockConfig(dim=32, n_heads=4, prenorm=False),
                                 BlockConfig(dim=32, n_heads=4, norm="rms", mlp="swiglu", bias=False, n_kv_heads=2),
                                 BlockConfig(dim=32, n_heads=4, layerscale=1e-4)])
def test_block_variants_run(cfg):
    assert Block(cfg)(torch.randn(2, 5, 32)).shape == (2, 5, 32)


def test_layerscale_starts_near_identity():
    block = Block(BlockConfig(dim=32, n_heads=4, layerscale=1e-6))
    x = torch.randn(2, 5, 32)
    torch.testing.assert_close(block(x), x, atol=1e-3, rtol=0)


def test_causal_model_ignores_the_future_and_bidirectional_does_not():
    torch.manual_seed(0)
    lm = TextTransformer(100, BlockConfig(dim=32, n_heads=4), depth=2, causal=True).eval()
    enc = TextTransformer(100, BlockConfig(dim=32, n_heads=4), depth=2, causal=False).eval()
    enc.load_state_dict(lm.state_dict())
    ids = torch.randint(0, 100, (1, 12))
    other = ids.clone()
    other[:, 8:] = (other[:, 8:] + 1) % 100
    assert torch.equal(lm(ids)[:, :8], lm(other)[:, :8])
    assert not torch.allclose(enc(ids)[:, :8], enc(other)[:, :8])


def test_kv_cache_matches_full_pass():
    torch.manual_seed(0)
    lm = TextTransformer(100, BlockConfig(dim=32, n_heads=4, n_kv_heads=2), depth=2).eval()
    ids = torch.randint(0, 100, (2, 10))
    caches = [{} for _ in lm.blocks]
    first = lm(ids[:, :4], caches=caches)
    steps = [lm(ids[:, t : t + 1], caches=caches, start=t) for t in range(4, 10)]
    torch.testing.assert_close(torch.cat([first] + steps, 1), lm(ids), atol=1e-5, rtol=1e-5)


def test_tied_head_and_loss():
    lm = TextTransformer(50, BlockConfig(dim=16, n_heads=2), depth=1)
    assert lm.head.weight is lm.embed.weight
    loss = lm.loss(torch.randint(0, 50, (2, 9)))
    assert abs(loss.item() - torch.log(torch.tensor(50.0)).item()) < 0.5


def test_same_middle_transplants():
    cfg = BlockConfig(dim=32, n_heads=4)
    text, vit = TextTransformer(80, cfg, depth=2), ViT(cfg, depth=2, image_size=16, patch=4)
    vit.blocks.load_state_dict(text.blocks.state_dict())  # strict: same names and shapes
    x = torch.randn(1, 17, 32)
    a, b = x, x
    for bt, bv in zip(text.blocks, vit.blocks):
        a, b = bt(a), bv(b)
    assert torch.equal(a, b)


def test_vit_shapes_registers_and_position_free_shuffle_invariance():
    cfg = BlockConfig(dim=32, n_heads=4)
    vit = ViT(cfg, depth=2, image_size=16, patch=4, registers=3)
    x = torch.randn(2, 3, 16, 16)
    assert vit.tokens(x).shape == (2, 1 + 3 + 16, 32) and vit(x).shape == (2, 10)
    free = ViT(cfg, depth=2, image_size=16, patch=4, pos="none").eval()
    shuffled = unpatchify(patchify(x, 4)[:, torch.randperm(16)], 4, 3, 16, 16)
    torch.testing.assert_close(free(x), free(shuffled), atol=1e-5, rtol=1e-5)
    sincos = ViT(cfg, depth=2, image_size=16, patch=4, pos="sincos2d")
    assert "pos" not in dict(sincos.named_parameters())  # fixed, not learned


def test_swiglu_at_8_thirds_matches_gelu_params():
    d = 96
    gelu = count_params(Block(BlockConfig(dim=d, n_heads=4, bias=False)).mlp)
    swiglu = count_params(Block(BlockConfig(dim=d, n_heads=4, bias=False, mlp="swiglu", mlp_hidden=8 * d // 3)).mlp)
    assert gelu == swiglu


def test_cnn_forward():
    assert SmallCNN()(torch.randn(2, 3, 32, 32)).shape == (2, 10)
