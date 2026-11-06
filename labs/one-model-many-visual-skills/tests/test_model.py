import random

import torch

from skillmix.chat import VOCAB, serialize
from skillmix.model import N_VISUAL, LoRALinear, TinyVLM
from skillmix.scenes import Scene, Item, render
from skillmix.train import RunConfig, collate, generate, new_model, param_groups, sample_batch, sft_loss, token_weights, train


def _ids(text):
    return torch.tensor([serialize([("user", text)], N_VISUAL, add_generation_prompt=True).ids])


def test_image_reaches_only_later_positions():
    m = new_model()
    ids = _ids("what color <image> is the cup ?")
    a = render(Scene((Item("cup", "red", 0),)))[None]
    b = render(Scene((Item("cup", "blue", 8),)))[None]
    la, lb = m(ids, a), m(ids, b)
    first_img = (ids[0] == 6).nonzero()[0, 0].item()
    assert torch.allclose(la[0, :first_img], lb[0, :first_img])  # causal: before the image, nothing changes
    assert not torch.allclose(la[0, -1], lb[0, -1])


def test_loss_normalizations_by_hand():
    logits = torch.zeros(2, 4, len(VOCAB))  # uniform: every target costs log V
    ids = torch.zeros(2, 4, dtype=torch.long)
    mask = torch.tensor([[0, 1, 0, 0], [0, 1, 1, 1.0]])
    logv = torch.log(torch.tensor(float(len(VOCAB))))
    assert torch.allclose(sft_loss(logits, ids, mask, "token"), logv)
    assert torch.allclose(sft_loss(logits, ids, mask, "example"), logv)
    logits[1, :, 0] = 5.0  # make row 1 cheaper: the two averages now disagree
    tok, ex = sft_loss(logits, ids, mask, "token"), sft_loss(logits, ids, mask, "example")
    assert tok < ex  # per-token weights the long, cheap row by 3/4
    assert torch.allclose(token_weights(mask, "token"), torch.tensor([0.25, 0.75]))
    assert torch.allclose(token_weights(mask, "example"), torch.tensor([0.5, 0.5]))


def test_lora_starts_as_identity_and_freezes_base():
    base = torch.nn.Linear(8, 8)
    lo = LoRALinear(base, r=2)
    x = torch.randn(3, 8)
    assert torch.allclose(lo(x), base(x))
    assert not base.weight.requires_grad


def test_strategies_train_the_right_parameters():
    def trainable(strategy):
        m = new_model()
        groups = param_groups(m, RunConfig(mixture={"vqa": 1}, strategy=strategy))
        return m, sum(p.numel() for g in groups for p in g["params"])
    full = trainable("full")[1]
    m, frozen = trainable("frozen_llm")
    assert frozen == sum(p.numel() for p in [*m.encoder.parameters(), *m.connector.parameters()])
    lora = trainable("lora")[1]
    assert frozen < lora < full
    groups = param_groups(new_model(), RunConfig(mixture={"vqa": 1}, strategy="low_llm_lr", lr=1.0))
    assert [g["lr"] for g in groups] == [1.0, 0.1]


def test_short_training_lowers_loss_and_generates():
    m = new_model()
    cfg = RunConfig(mixture={"vqa": 0.5, "text": 0.5}, steps=40, batch=16, log_every=39)
    curve = train(m, cfg)["curve"]
    assert curve[-1][1] < curve[0][1]
    outs = generate(m, [([("user", "reverse : apple tree")], None),
                        ([("user", "<image> describe the image .")], Scene((Item("cup", "red", 0),)))], max_new=5)
    assert len(outs) == 2 and all(isinstance(o, str) for o in outs)


def test_collate_marks_text_only_rows_without_images():
    b = collate(sample_batch(RunConfig(mixture={"text": 1.0, "vqa": 1.0}, batch=16), random.Random(0)))
    for src, row, img in zip(b["sources"], b["ids"], b["images"]):
        assert ((row == 6).sum().item() == 0) == (src == "text")
        if src == "text":
            assert img.abs().sum() == 0


def test_eval_suites_run_in_every_image_mode():
    from skillmix import evaluate as ev
    m = new_model()
    for image in ("real", "blank", "swap"):
        assert set(ev.eval_vqa(m, n=4, image=image)) == {"vqa_exact"}
        assert 0 <= ev.eval_fork_colour(m, n=4, image=image)["fork_colour_exact"] <= 1
        assert ev.eval_reason(m, n=4, image=image)["rationale_unchanged"] >= 0


def test_swap_keeps_the_object_the_question_names():
    from skillmix.evaluate import EMPTY, shown
    rng = random.Random(0)
    s = Scene((Item("cup", "red", 0),))
    assert shown(s, "real", rng) is s and shown(s, "blank", rng) == EMPTY
    assert all(shown(s, "swap", rng, "fork").has("fork") for _ in range(20))
