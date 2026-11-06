import pytest
import torch

from vlmtrain import data as D
from vlmtrain.model import build_vlm, tiny_config
from vlmtrain.packing import block_causal_mask, first_fit_decreasing, pack, padding_fraction
from vlmtrain.processor import IGNORE, TinyVLMProcessor
from vlmtrain.tokenizer import offline_tokenizer
from vlmtrain.train import per_sample_loss


def test_first_fit_decreasing_respects_the_budget_and_keeps_everything():
    lens = [70, 60, 50, 40, 30, 20, 10]
    rows = first_fit_decreasing(lens, 100)
    assert sorted(i for r in rows for i in r) == list(range(7))
    assert all(sum(lens[i] for i in r) <= 100 for r in rows)
    assert len(rows) == 3
    with pytest.raises(ValueError):
        first_fit_decreasing([101], 100)


def test_block_causal_mask_by_hand():
    seg = torch.tensor([[0, 0, 1, 1, -1]])
    allowed = block_causal_mask(seg)[0, 0] == 0
    want = torch.tensor([[1, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 1]], dtype=torch.bool)
    assert torch.equal(allowed, want)


def test_padding_fraction():
    assert padding_fraction([10, 10, 20, 20], 2, sort=False) == 0.0
    assert padding_fraction([10, 20], 2) == pytest.approx(10 / 40)
    assert padding_fraction([10, 20, 10, 20], 2, sort=True) == 0.0


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    tok = offline_tokenizer()
    cfg = tiny_config()
    model = build_vlm(cfg, tok).eval()
    return model, TinyVLMProcessor(tok, cfg.tokens_per_tile, cfg.image_size)


@torch.no_grad()
def test_packed_loss_equals_separate_samples(setup):
    """The contract of packing: same per-sample losses as running each sample alone."""
    model, proc = setup
    samples = [D.make_example(t, 1) for t in ("vqa", "doc", "ocr", "caption")] + [D.text_sample("plain text here")]
    enc = [proc.encode(s) for s in samples]
    alone = []
    for e in enc:
        b = proc.collate([e])
        out = model(b["input_ids"], b["pixel_values"], b["attention_mask"], b["labels"])
        alone.append(out.loss.item())
    b = pack(enc, 400, proc.pad_id)
    assert b["input_ids"].shape[0] < len(enc)  # really packed
    out = model(b["input_ids"], b["pixel_values"], attention_mask=b["attention_mask"], position_ids=b["position_ids"])
    offsets = torch.tensor([0] + [len(r) for r in b["rows"]]).cumsum(0)[:-1]
    index = torch.where(b["seg_ids"] >= 0, b["seg_ids"] + offsets[:, None], -1)
    sums, counts = per_sample_loss(out.logits, b["labels"], index, len(enc))
    packed = (sums / counts).tolist()
    order = [i for row in b["rows"] for i in row]
    for k, i in enumerate(order):
        assert packed[k] == pytest.approx(alone[i], rel=1e-4), samples[i]["task"]


@torch.no_grad()
def test_break_it_without_the_block_mask_samples_leak(setup):
    """Same packed row with a plain causal mask: the second sample now sees the first, and the loss moves."""
    model, proc = setup
    enc = [proc.encode(D.make_example("vqa", 1)), proc.encode(D.make_example("ocr", 1))]
    b = pack(enc, 200, proc.pad_id)
    ok = model(b["input_ids"], b["pixel_values"], attention_mask=b["attention_mask"], position_ids=b["position_ids"]).logits
    leaky = model(b["input_ids"], b["pixel_values"], position_ids=b["position_ids"],
                  attention_mask=block_causal_mask((b["seg_ids"] >= 0).long() * 0)).logits
    first = len(enc[b["rows"][0][0]].input_ids)
    assert torch.allclose(ok[0, :first], leaky[0, :first], atol=1e-4)   # the first sample cannot tell
    assert not torch.allclose(ok[0, first:first + 20], leaky[0, first:first + 20], atol=1e-3)


def test_boundary_targets_are_ignored(setup):
    _, proc = setup
    enc = [proc.encode(D.text_sample("aaaa")), proc.encode(D.text_sample("bbbbbb"))]
    b = pack(enc, 64, proc.pad_id)
    seg = b["seg_ids"][0]
    start_of_second = int((seg == 1).nonzero()[0])
    assert b["labels"][0, start_of_second] == IGNORE
