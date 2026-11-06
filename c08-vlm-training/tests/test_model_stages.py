"""Freezing, parameter groups, staged training and resume, on a random tiny model (no downloads)."""

import pytest
import torch

from vlmtrain import data as D
from vlmtrain.freeze import set_trainable, train_new_token_rows, trainable_report, vision_layers
from vlmtrain.model import build_vlm, load_partial, pixel_shuffle, save_trainable, tiny_config
from vlmtrain.optim import build_optimizer, warmup_cosine
from vlmtrain.processor import IGNORE, TinyVLMProcessor
from vlmtrain.tokenizer import offline_tokenizer
from vlmtrain.train import TrainConfig, per_sample_loss, run

LRS = {"vision": 1e-6, "projector": 1e-4, "lm": 2e-5, "lora": 2e-4}


@pytest.fixture()
def setup():
    torch.manual_seed(0)
    tok = offline_tokenizer()
    cfg = tiny_config()
    return build_vlm(cfg, tok), TinyVLMProcessor(tok, cfg.tokens_per_tile, cfg.image_size)


def test_pixel_shuffle_folds_blocks():
    x = torch.arange(16.0).reshape(1, 16, 1)
    y = pixel_shuffle(x, 2)
    assert y.shape == (1, 4, 4)
    assert y[0, 0].tolist() == [0, 1, 4, 5]    # top-left 2 x 2 block
    assert y[0, 3].tolist() == [10, 11, 14, 15]


def test_forward_shapes_and_token_check(setup):
    model, proc = setup
    b = proc([D.make_example("vqa", 0), D.make_example("doc", 0)])
    out = model(b["input_ids"], b["pixel_values"], b["attention_mask"], b["labels"])
    assert out.logits.shape[:2] == b["input_ids"].shape
    assert out.loss.isfinite()
    with pytest.raises(ValueError, match="<image> tokens"):
        model(b["input_ids"], b["pixel_values"][:-1], b["attention_mask"])


@pytest.mark.parametrize("preset,on", [("align", {"projector"}), ("lm", {"projector", "lm"}),
                                       ("full", {"vision", "projector", "lm"}), ("none", set())])
def test_presets(setup, preset, on):
    model, _ = setup
    set_trainable(model, preset)
    rep = trainable_report(model)
    for name, v in rep.items():
        assert v["trainable"] == (v["total"] if name in on else 0)


def test_partial_unfreezes_only_the_top(setup):
    model, _ = setup
    set_trainable(model, "partial", vision_layers_to_train=1)
    layers = list(vision_layers(model))
    assert all(p.requires_grad for p in layers[-1].parameters())
    assert not any(p.requires_grad for p in layers[0].parameters())
    with pytest.raises(ValueError):
        set_trainable(model, "partial", 0)


def test_gradients_reach_only_the_trainable_modules(setup):
    model, proc = setup
    set_trainable(model, "align")
    b = proc([D.make_example("caption", 0)])
    model(b["input_ids"], b["pixel_values"], b["attention_mask"], b["labels"]).loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.projector.parameters())
    assert all(p.grad is None for p in model.lm.parameters())
    assert all(p.grad is None for p in model.vision.parameters())


def test_frozen_modules_do_not_change(setup):
    model, proc = setup
    set_trainable(model, "align")
    before = {n: p.clone() for n, p in model.named_parameters()}
    opt = build_optimizer(model, {**LRS, "projector": 1e-2})
    b = proc([D.make_example("caption", 0)])
    model(b["input_ids"], b["pixel_values"], b["attention_mask"], b["labels"]).loss.backward()
    opt.step()
    for n, p in model.named_parameters():
        assert torch.equal(p, before[n]) == (not n.startswith("projector")), n


def test_each_trainable_parameter_is_in_exactly_one_group(setup):
    model, _ = setup
    set_trainable(model, "partial", 1)
    opt = build_optimizer(model, LRS, weight_decay=0.01)
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    by_name = {g["name"]: g["lr"] for g in opt.param_groups}
    assert by_name["vision"] == 1e-6 and by_name["projector"] == 1e-4 and by_name["lm"] == 2e-5


def test_scheduler_keeps_the_lr_ratios(setup):
    model, _ = setup
    set_trainable(model, "lm")
    opt = build_optimizer(model, LRS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(2, 10))
    for _ in range(5):
        opt.step()
        sched.step()
    lr = {g["name"]: g["lr"] for g in opt.param_groups}
    assert lr["projector"] / lr["lm"] == pytest.approx(5)


def test_warmup_cosine_shape():
    f = warmup_cosine(10, 100, 0.1)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(100) == pytest.approx(0.1)


def test_new_token_rows_learn_and_old_rows_do_not(setup):
    model, proc = setup
    set_trainable(model, "none")
    first_new = proc.image_token_id
    train_new_token_rows(model, first_new)
    emb = model.lm.get_input_embeddings().weight
    before = emb.detach().clone()
    opt = torch.optim.AdamW([emb], lr=0.1, weight_decay=0.0)
    ids = torch.tensor([[first_new + 1, 40, first_new + 2, 41]])
    model.lm(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    assert torch.equal(emb[:first_new], before[:first_new])
    assert not torch.equal(emb[first_new:], before[first_new:])


def test_per_sample_loss_matches_hf_mean(setup):
    model, proc = setup
    b = proc([D.make_example("vqa", 0), D.make_example("ocr", 0), D.text_sample("short text")])
    out = model(b["input_ids"], b["pixel_values"], b["attention_mask"], b["labels"])
    idx = torch.arange(3)[:, None].expand_as(b["input_ids"])
    sums, counts = per_sample_loss(out.logits, b["labels"], idx, 3)
    assert (sums.sum() / counts.sum()).item() == pytest.approx(out.loss.item(), rel=1e-5)
    assert counts[2] == (b["labels"][2, 1:] != IGNORE).sum()


def test_loss_mask_ignores_prompt_tokens(setup):
    model, proc = setup
    s = D.make_example("vqa", 0)
    b = proc([s])
    base = model(b["input_ids"], b["pixel_values"], b["attention_mask"], b["labels"]).loss.item()
    wrong = b["labels"].clone()
    wrong[wrong == IGNORE] = 5  # the classic bug: the model is graded on the prompt too
    wrong[0, 0] = IGNORE
    assert model(b["input_ids"], b["pixel_values"], b["attention_mask"], wrong).loss.item() != pytest.approx(base)


def test_save_and_load_trainable_round_trip(setup, tmp_path):
    model, _ = setup
    set_trainable(model, "align")
    n = save_trainable(model, tmp_path / "p.safetensors")
    assert n == trainable_report(model)["projector"]["total"]
    saved = {k: v.clone() for k, v in model.projector.state_dict().items()}
    for p in model.projector.parameters():
        p.data.zero_()
    assert load_partial(model, tmp_path / "p.safetensors")
    for k, v in model.projector.state_dict().items():
        assert torch.equal(v, saved[k])


def _tiny_run(**kw):
    cfg = dict(model="tiny", steps=3, batch_size=4, eval_n=2, ppl_passages=0, results_dir=None, device="cpu",
               log_every=1, warmup=1)
    cfg.update(kw)
    return run(TrainConfig(**cfg))


def test_stage1_trains_only_the_projector():
    r = _tiny_run(freeze="align", tasks=["caption"])
    assert r["trainable"]["lm"]["trainable"] == 0 and r["trainable"]["vision"]["trainable"] == 0
    assert r["trainable"]["projector"]["trainable"] > 0


@pytest.mark.parametrize("kw", [dict(freeze="lm"), dict(freeze="partial", vision_layers_to_train=1),
                                dict(freeze="full"), dict(freeze="none", lora=["lm", "projector"], train_new_tokens=True),
                                dict(pack_budget=512), dict(strategy="resize"), dict(grounding_format="text"),
                                dict(mixture="uniform"), dict(mixture="proportional"), dict(loss_norm="sample")])
def test_stage2_variants_run(kw):
    r = _tiny_run(**kw)
    assert all(rec["loss"] == rec["loss"] for rec in r["log"])  # no NaN
    assert set(r["metrics"]) == set(D.TASKS)


def test_replay_reaches_the_target_share():
    r = _tiny_run(steps=30, batch_size=8, replay_ratio=0.25, eval_n=1, tasks=["vqa"])
    assert abs(r["mixture_observed"]["text"] - 0.25) < 0.1
    assert "text" in r["supervised_token_share"]


def test_a_tiny_vlm_can_overfit_one_task():
    r = _tiny_run(steps=60, batch_size=8, tasks=["vqa"], task_sizes={"vqa": 8}, eval_n=1, freeze="full",
                  lrs={"vision": 1e-3, "projector": 3e-3, "lm": 3e-3}, log_every=5)
    assert r["log"][-1]["loss"] < r["log"][0]["loss"] * 0.75
