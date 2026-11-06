"""The contract on real pretrained weights. These download SigLIP and SmolLM2 from the Hub."""

import pytest
import torch

from vlmtrain import data as D
from vlmtrain import grounding as G
from vlmtrain.c07_adapter import build_c07
from vlmtrain.evaluate import GROUNDING_NEW_TOKENS
from vlmtrain.model import VLMConfig, build_vlm
from vlmtrain.packing import pack
from vlmtrain.processor import TinyVLMProcessor
from vlmtrain.tokenizer import build_tokenizer
from vlmtrain.train import per_sample_loss

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def c07():
    tok, model, proc = build_c07()
    return tok, model.eval(), proc


@torch.no_grad()
def test_our_fallback_model_computes_the_same_logits_as_c07(c07):
    """Two assemblies of one architecture. Same weights in, same logits out."""
    _, theirs, proc = c07
    tok = build_tokenizer()
    cfg = VLMConfig()
    ours = build_vlm(cfg, tok).eval()
    assert tok.convert_tokens_to_ids("<image>") == proc.image_token_id
    ours.projector.net[0].load_state_dict(theirs.projector.fc1.state_dict())
    ours.projector.net[2].load_state_dict(theirs.projector.fc2.state_dict())
    v = ours.lm.get_input_embeddings().num_embeddings  # C7 pads its vocabulary to a multiple of 64
    ours.lm.get_input_embeddings().weight.copy_(theirs.lm.get_input_embeddings().weight[:v])
    ours_proc = TinyVLMProcessor(tok, cfg.tokens_per_tile, cfg.image_size, proc.strategy, proc.max_tiles)
    samples = [D.make_example("vqa", 1), D.make_example("doc", 1)]
    b = proc(samples)
    assert torch.equal(b["input_ids"], ours_proc(samples)["input_ids"])
    a = ours(b["input_ids"], b["pixel_values"], b["attention_mask"]).logits
    c = theirs(b["input_ids"], b["pixel_values"], attention_mask=b["attention_mask"]).logits[..., :v]
    real = b["attention_mask"].bool()
    assert torch.allclose(a[real], c[real], atol=1e-3)


@torch.no_grad()
def test_packing_on_real_weights_matches_running_each_sample_alone(c07):
    _, model, proc = c07
    samples = [D.make_example(t, 2) for t in ("vqa", "ocr", "caption")]
    enc = [proc.encode(s) for s in samples]
    alone = []
    for e in enc:
        b = proc.collate([e])
        alone.append(model(b["input_ids"], b["pixel_values"], attention_mask=b["attention_mask"],
                           labels=b["labels"]).loss.item())
    b = pack(enc, 512, proc.pad_id)
    assert b["input_ids"].shape[0] == 1
    out = model(b["input_ids"], b["pixel_values"], attention_mask=b["attention_mask"], position_ids=b["position_ids"])
    index = b["seg_ids"]
    sums, counts = per_sample_loss(out.logits, b["labels"], index, len(enc))
    for k, i in enumerate(b["rows"][0]):
        assert (sums[k] / counts[k]).item() == pytest.approx(alone[i], rel=1e-3)


@torch.no_grad()
def test_real_generation_returns_decodable_text(c07):
    tok, model, proc = c07
    b = proc([D.make_example("vqa", 3)], for_generation=True)
    out = model.generate(b["input_ids"], b["pixel_values"], b["attention_mask"], max_new_tokens=4, do_sample=False,
                         pad_token_id=proc.pad_id, eos_token_id=proc.eos_id)
    assert out.shape[1] <= 4 and isinstance(tok.decode(out[0]), str)


@pytest.mark.parametrize("fmt", ["tokens", "text"])
def test_the_generation_limit_fits_the_longest_grounding_target(fmt):
    tok = build_tokenizer()
    longest = 0
    for i in range(400):
        answer = D.make_example("grounding", i, "test", fmt)["meta"]["answer"]
        longest = max(longest, len(tok(answer, add_special_tokens=False)["input_ids"]))
    worst = G.encode_polygon([(200.0, 200.0)] * 12, 224, 224, fmt)  # every coordinate has two digits
    assert max(longest, len(tok(worst, add_special_tokens=False)["input_ids"])) <= GROUNDING_NEW_TOKENS[fmt]
