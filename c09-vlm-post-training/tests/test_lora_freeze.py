import pytest
import torch
from torch import nn

from vlmalign.freeze import configure_trainable
from vlmalign.logprobs import build_batch, image_features, sequence_logprobs
from vlmalign.lora import LoRALinear, add_lora, lora_disabled
from vlmalign.tiny import tiny_pixels, tiny_prompt, tiny_vlm


def _batch():
    return build_batch([tiny_prompt()], [torch.tensor([7, 8, 9])], [0], tiny_pixels(1), pad_id=0)


def test_fresh_lora_is_an_exact_no_op():
    model, b = tiny_vlm(), _batch()
    before, _ = sequence_logprobs(model, b)
    configure_trainable(model, "all")
    after, _ = sequence_logprobs(model, b)
    torch.testing.assert_close(before, after)


def test_lora_disabled_recovers_the_reference():
    model, b = tiny_vlm(), _batch()
    ref, _ = sequence_logprobs(model, b)
    configure_trainable(model, "llm")
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B)  # pretend training moved the adapters
    moved, _ = sequence_logprobs(model, b)
    with lora_disabled(model):
        back, _ = sequence_logprobs(model, b)
    assert not torch.allclose(moved, ref)
    torch.testing.assert_close(back, ref)


def test_lora_matches_peft():
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(16, 8))
    ours = add_lora(nn.Sequential(nn.Linear(16, 8)), r"^0$", r=4, alpha=8)
    assert ours == ["0"]
    mine = nn.Sequential(nn.Linear(16, 8))
    mine.load_state_dict(base.state_dict())
    add_lora(mine, r"^0$", r=4, alpha=8)
    theirs = peft.get_peft_model(base, peft.LoraConfig(r=4, lora_alpha=8, target_modules=["0"], lora_dropout=0.0))
    with torch.no_grad():
        layer = theirs.base_model.model[0]
        layer.lora_A["default"].weight.copy_(torch.randn(4, 16))
        layer.lora_B["default"].weight.copy_(torch.randn(8, 4))
        mine[0].lora_A.copy_(layer.lora_A["default"].weight)
        mine[0].lora_B.copy_(layer.lora_B["default"].weight)
    x = torch.randn(5, 16)
    torch.testing.assert_close(mine(x), theirs(x))


@pytest.mark.parametrize("mode,vision,connector", [("llm", False, False), ("projector+llm", False, True),
                                                   ("all", True, True)])
def test_freeze_modes_control_what_moves(mode, vision, connector):
    model, b = tiny_vlm(), _batch()
    counts = configure_trainable(model, mode)
    assert (counts["vision"] > 0) == vision
    assert (counts["connector"] > 0) == connector
    assert counts["llm"] > 0 and counts["lm_head"] == 0
    feats_before = image_features(model, b.pixel_values).detach().clone()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    for _ in range(2):  # step 1 moves lora_B off zero, step 2 moves lora_A's effect
        lp, _ = sequence_logprobs(model, b)
        opt.zero_grad()
        (-lp.sum()).backward()
        opt.step()
    feats_after = image_features(model, b.pixel_values)
    # the image features change only if the vision tower or the connector learns
    assert torch.allclose(feats_before, feats_after) == (not (vision or connector))


def test_saved_trainable_set_reloads_exactly(tmp_path):
    """Save the way train_grpo.py does, load the way perception_probe.py does.

    The load uses strict=False, which would ignore a wrong key in silence, so the keys are checked too.
    """
    model, b = tiny_vlm(), _batch()
    configure_trainable(model, "projector+llm")
    for m in model.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B)  # pretend training moved the adapters
    moved, _ = sequence_logprobs(model, b)
    path = tmp_path / "run.pt"
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, path)

    fresh = tiny_vlm()
    torch.manual_seed(1)  # a different random lora_A, so only the load can make the two models agree
    configure_trainable(fresh, "projector+llm")
    before_load, _ = sequence_logprobs(fresh, b)
    state = torch.load(path)
    assert fresh.load_state_dict(state, strict=False).unexpected_keys == []
    after_load, _ = sequence_logprobs(fresh, b)
    assert not torch.allclose(before_load, moved)
    torch.testing.assert_close(after_load, moved)
