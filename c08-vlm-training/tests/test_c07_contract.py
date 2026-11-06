"""C8 trains C7's model. These tests pin the contract, and check our fallback computes the same thing."""

import pytest
import torch

from vlmtrain import data as D
from vlmtrain.c07_adapter import build_c07_tiny, c07_available
from vlmtrain.freeze import set_trainable, trainable_report
from vlmtrain.model import Projector, pixel_shuffle
from vlmtrain.packing import pack
from vlmtrain.train import TrainConfig, per_sample_loss, run

pytestmark = pytest.mark.skipif(not c07_available(), reason="../c07-vlm not present")


@pytest.fixture()
def built():
    return build_c07_tiny()


def test_c07_has_the_three_modules(built):
    _, model, _ = built
    assert {"vision", "projector", "lm"} <= set(dict(model.named_children()))


def test_processor_and_c07_model_agree_on_visual_tokens(built):
    _, model, proc = built
    b = proc([D.make_example("doc", 0), D.make_example("vqa", 0)])
    out = model(b["input_ids"], b["pixel_values"], attention_mask=b["attention_mask"], labels=b["labels"])
    assert out.logits.shape[:2] == b["input_ids"].shape
    assert out.loss.isfinite()


@torch.no_grad()
def test_c07_accepts_a_packed_batch_and_matches_separate_samples(built):
    _, model, proc = built
    enc = [proc.encode(D.make_example("vqa", 1)), proc.encode(D.make_example("caption", 1))]
    alone = []
    for e in enc:
        b = proc.collate([e])
        alone.append(model(b["input_ids"], b["pixel_values"], attention_mask=b["attention_mask"],
                           labels=b["labels"]).loss.item())
    b = pack(enc, 400, proc.pad_id)
    out = model(b["input_ids"], b["pixel_values"], attention_mask=b["attention_mask"], position_ids=b["position_ids"])
    sums, counts = per_sample_loss(out.logits, b["labels"], b["seg_ids"], len(enc))
    for k, i in enumerate(b["rows"][0]):
        assert (sums[k] / counts[k]).item() == pytest.approx(alone[i], rel=1e-4)


def test_c07_understands_our_freeze_presets(built):
    _, model, _ = built
    set_trainable(model, "align")
    rep = trainable_report(model)
    assert rep["projector"]["trainable"] > 0 and rep["lm"]["trainable"] == 0


def test_our_pixel_shuffle_matches_c07():
    from tinyvlm.projector import pixel_shuffle as c07_shuffle

    x = torch.randn(3, 196, 8)
    assert torch.equal(pixel_shuffle(x, 2), c07_shuffle(x, 2))


def test_our_projector_matches_c07():
    from tinyvlm.projector import MLPProjector, pixel_shuffle as c07_shuffle

    ours = Projector(8, 6, shuffle=2, kind="mlp")
    theirs = MLPProjector(8 * 4, 6)
    theirs.fc1.load_state_dict(ours.net[0].state_dict())
    theirs.fc2.load_state_dict(ours.net[2].state_dict())
    x = torch.randn(2, 16, 8)
    assert torch.allclose(ours(x), theirs(c07_shuffle(x, 2)))


def test_the_trainer_runs_c07():
    r = run(TrainConfig(model="tiny_c07", steps=2, batch_size=4, eval_n=1, ppl_passages=0, results_dir=None,
                        device="cpu", log_every=1, warmup=1))
    assert r["log"][-1]["loss"] == r["log"][-1]["loss"]
