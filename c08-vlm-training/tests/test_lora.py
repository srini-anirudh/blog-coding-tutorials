import pytest
import torch
import torch.nn as nn

from vlmtrain.freeze import set_trainable, trainable_report
from vlmtrain.lora import LoRALinear, apply_lora, merge_lora
from vlmtrain.model import build_vlm, tiny_config
from vlmtrain.tokenizer import offline_tokenizer


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return build_vlm(tiny_config(), offline_tokenizer()).eval()


def test_fresh_lora_is_the_identity():
    base = nn.Linear(8, 4)
    x = torch.randn(3, 8)
    assert torch.equal(LoRALinear(base, r=2)(x), base(x))


def test_rank_and_trainable_count():
    lora = LoRALinear(nn.Linear(8, 4), r=2)
    assert not any(p.requires_grad for p in lora.base.parameters())
    assert sum(p.numel() for p in lora.parameters() if p.requires_grad) == 2 * (8 + 4)


def test_merge_matches_the_wrapped_module():
    lora = LoRALinear(nn.Linear(8, 4), r=2)
    nn.init.normal_(lora.lora_B)
    x = torch.randn(3, 8)
    assert torch.allclose(lora.merged()(x), lora(x), atol=1e-6)


def test_matches_peft_output():
    """Trusted implementation: the same A, B and W through PEFT must give the same numbers."""
    peft = pytest.importorskip("peft")

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(x)

    torch.manual_seed(0)
    m = M()
    x = torch.randn(5, 8)
    ours = LoRALinear(m.fc, r=2, alpha=4.0)
    nn.init.normal_(ours.lora_B)
    wrapped = peft.get_peft_model(M(), peft.LoraConfig(r=2, lora_alpha=4, target_modules=["fc"]))
    layer = wrapped.base_model.model.fc
    with torch.no_grad():
        layer.base_layer.weight.copy_(m.fc.weight)
        layer.base_layer.bias.copy_(m.fc.bias)
        layer.lora_A["default"].weight.copy_(ours.lora_A)
        layer.lora_B["default"].weight.copy_(ours.lora_B)
    wrapped.eval()
    assert torch.allclose(ours(x), wrapped(x), atol=1e-6)


@pytest.mark.parametrize("place,on", [(["lm"], {"lm"}), (["vision"], {"vision"}), (["projector"], {"projector"}),
                                      (["lm", "vision", "projector"], {"lm", "vision", "projector"})])
def test_placements_touch_only_their_module(model, place, on):
    set_trainable(model, "none")
    names = apply_lora(model, place, r=2, alpha=4)
    assert names
    assert {n.split(".")[0] for n in names} == on
    rep = trainable_report(model)
    for k, v in rep.items():
        assert (v["trainable"] > 0) == (k in on)


def test_vision_placement_does_not_wrap_embeddings(model):
    names = apply_lora(model, ["vision"], r=2)
    assert not any("embed" in n or "head" in n for n in names)


def test_merge_lora_restores_plain_linears_and_output(model):
    set_trainable(model, "none")
    apply_lora(model, ["lm", "projector"], r=2)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            nn.init.normal_(p, std=0.01)
    ids = torch.tensor([[1, 2, 3, 4]])
    a = model.lm(input_ids=ids).logits
    assert merge_lora(model) > 0
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert torch.allclose(a, model.lm(input_ids=ids).logits, atol=1e-5)
