import torch
import torch.nn.functional as F

from vlmfam import mixing as M
from vlmfam import pages as G


def test_deepencoder_modes():
    assert {m: G.deepencoder_tokens(s) for m, s in G.MODES.items()} == {"Tiny": 64, "Small": 100, "Base": 256,
                                                                          "Large": 400}


def test_smaller_font_fits_more_text():
    text = " ".join(["word"] * 5000)
    big, _ = G.fill_page(text, 32, side=256)
    small, page = G.fill_page(text, 12, side=256)
    assert len(small.split()) > len(big.split()) > 0 and page.size == (256, 256)


def test_clean_wikitext():
    assert G.clean_wikitext([" = Title = \n", " a @-@ b , c . \n"]) == "a-b, c."


def test_caps_and_sqrt():
    w = M.capped_weights({"a": 900, "b": 100}, strategy="sqrt")
    assert abs(w["a"] - 0.75) < 1e-12
    w = M.capped_weights({"a": 900, "b": 100}, cap=100)
    assert w == {"a": 0.5, "b": 0.5}


def test_token_shares():
    s = M.token_shares({"x": 0.5, "y": 0.5}, {"x": 1, "y": 3})
    assert s == {"x": 0.25, "y": 0.75}


def test_loss_modes_by_hand():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 7)
    labels = torch.tensor([[1, -100, -100, -100], [2, 3, 4, 5]])
    tok = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
    a, b = tok[0, 0], tok[1].mean()
    assert torch.allclose(M.sft_loss(logits, labels, "sample"), (a + b) / 2)
    assert torch.allclose(M.sft_loss(logits, labels, "token"), (tok[0, 0] + tok[1].sum()) / 5)
    w = torch.tensor([1.0, 0.5])  # 1 / sqrt(1), 1 / sqrt(4)
    want = (w[0] * tok[0, 0] + w[1] * tok[1].sum()) / (w[0] * 1 + w[1] * 4)
    assert torch.allclose(M.sft_loss(logits, labels, "sqrt"), want)
    assert [round(x, 6) for x in M.sample_weights([1, 4], "sqrt")] == [round(1 / 3, 6), round(2 / 3, 6)]


def test_modality_warmup():
    assert M.modality_warmup(0, 10, 0.7) == 1.0
    assert abs(M.modality_warmup(5, 10, 0.7) - 0.85) < 1e-12
    assert M.modality_warmup(50, 10, 0.7) == 0.7
