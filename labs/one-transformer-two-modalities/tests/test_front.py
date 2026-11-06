import torch

from twomod.bpe import ByteBPE
from twomod.front import PatchEmbed, onehot_lookup, patchify, unpatchify

TEXT = "the transformer reads the tokens; the vision transformer reads the patches. " * 20


def test_bpe_round_trip_and_merges():
    bpe = ByteBPE.train(TEXT, num_merges=50)
    assert bpe.vocab_size == 256 + len(bpe.merges) <= 306
    for s in [TEXT, "unseen words, ÜNICODE ✓ 東京", "  spaces\n\nand\tTABS "]:
        assert bpe.decode(bpe.encode(s)) == s
    assert len(bpe.encode(" the transformer")) < len(" the transformer".encode())  # it compresses


def test_bpe_is_deterministic():
    assert ByteBPE.train(TEXT, 30).merges == ByteBPE.train(TEXT, 30).merges


def test_lookup_is_onehot_matmul_and_gradient_is_sparse():
    E = torch.nn.Parameter(torch.randn(100, 8))
    ids = torch.tensor([[3, 3, 7], [50, 7, 3]])
    assert torch.equal(E[ids], onehot_lookup(ids, E))
    E[ids].sum().backward()
    touched = set(torch.nonzero(E.grad.abs().sum(1)).flatten().tolist())
    assert touched == {3, 7, 50}


def test_patchify_inverse_and_shape():
    x = torch.randn(2, 3, 32, 48)
    p = patchify(x, 8)
    assert p.shape == (2, 4 * 6, 3 * 8 * 8)
    assert torch.equal(unpatchify(p, 8, 3, 32, 48), x)
    assert torch.equal(p[0, 1], x[0, :, 0:8, 8:16].flatten())  # raster order, (channel, row, column) layout


def test_patch_linear_equals_conv_and_gradient_is_dense():
    embed = PatchEmbed(4, 3, 16)
    x = torch.randn(2, 3, 16, 16)
    out = embed(x)
    conv = embed.as_conv()(x).flatten(2).transpose(1, 2)
    torch.testing.assert_close(out, conv, atol=1e-5, rtol=1e-5)
    out.square().sum().backward()
    assert (embed.proj.weight.grad != 0).all()
