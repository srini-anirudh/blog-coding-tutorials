import pytest
import torch
from torch import nn

from loralab.lora import (LoRALinear, adapter_state_dict, apply_lora, manual_grads, merge_all, multi_adapter_forward)


class Block(nn.Module):
    def __init__(self, d=24, h=40):
        super().__init__()
        self.q_proj = nn.Linear(d, h)
        self.down_proj = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.q_proj(x)))


def filled(layer: LoRALinear, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        layer.lora_A.copy_(torch.randn(layer.lora_A.shape, generator=g) * 0.1)
        layer.lora_B.copy_(torch.randn(layer.lora_B.shape, generator=g) * 0.1)
    return layer


@pytest.mark.parametrize("init", ["b_zero", "a_zero", "both_zero"])
def test_zero_inits_start_at_the_base_model(init):
    base = nn.Linear(16, 12)
    x = torch.randn(5, 16)
    layer = LoRALinear(base, r=4, init=init)
    assert torch.equal(layer(x), base(x))


def test_both_random_does_not_start_at_the_base_model():
    base = nn.Linear(16, 12)
    x = torch.randn(5, 16)
    assert not torch.allclose(LoRALinear(base, r=4, init="both_random")(x), base(x))


def test_both_zero_is_dead_and_b_zero_only_moves_b():
    x = torch.randn(8, 16)
    dead = LoRALinear(nn.Linear(16, 12), r=4, init="both_zero")
    dead(x).pow(2).sum().backward()
    assert dead.lora_A.grad.abs().max() == 0 and dead.lora_B.grad.abs().max() == 0
    live = LoRALinear(nn.Linear(16, 12), r=4, init="b_zero")
    live(x).pow(2).sum().backward()
    assert live.lora_A.grad.abs().max() == 0 and live.lora_B.grad.abs().max() > 0


def test_base_is_frozen_and_checkpoint_is_adapter_only():
    model = Block()
    wrapped = apply_lora(model, targets=("q_proj", "down_proj"), r=4)
    assert wrapped == ["q_proj", "down_proj"]
    sd = adapter_state_dict(model)
    assert set(sd) == {"q_proj.lora_A", "q_proj.lora_B", "down_proj.lora_A", "down_proj.lora_B"}


def test_manual_backward_matches_autograd():
    torch.manual_seed(0)
    W = torch.randn(10, 7, dtype=torch.float64)
    A = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
    B = torch.randn(10, 3, dtype=torch.float64, requires_grad=True)
    x = torch.randn(4, 7, dtype=torch.float64, requires_grad=True)
    s = 0.7
    h = x @ W.T + s * x @ A.T @ B.T
    g = torch.randn_like(h)
    h.backward(g)
    dB, dA, dx = manual_grads(W, A.detach(), B.detach(), s, x.detach(), g)
    assert torch.allclose(dB, B.grad) and torch.allclose(dA, A.grad) and torch.allclose(dx, x.grad)


@pytest.mark.parametrize("scaling", ["standard", "rslora"])
def test_scaling_rules(scaling):
    layer = LoRALinear(nn.Linear(8, 8), r=16, alpha=32, scaling=scaling)
    assert layer.scale == (2.0 if scaling == "standard" else 8.0)


def test_merge_unmerge_and_merge_all():
    base = nn.Linear(16, 12)
    layer = filled(LoRALinear(base, r=4))
    w0 = base.weight.detach().clone()
    x = torch.randn(6, 16)
    y = layer(x).detach()
    layer.merge()
    assert torch.allclose(layer(x), y, atol=1e-6)
    layer.unmerge()
    assert torch.allclose(base.weight, w0, atol=1e-6)
    model = Block()
    apply_lora(model, targets=("q_proj",), r=4, dora=True)
    filled(model.q_proj)
    x2 = torch.randn(6, 24)
    y = model(x2).detach()
    merge_all(model)
    assert isinstance(model.q_proj, nn.Linear) and torch.allclose(model(x2), y, atol=1e-5)


def test_multi_adapter_batch_equals_per_request_merge():
    torch.manual_seed(0)
    W = torch.randn(12, 16)
    As, Bs = torch.randn(3, 4, 16), torch.randn(3, 12, 4)
    ids = torch.tensor([0, 2, 1, 2, 0])
    x = torch.randn(5, 16)
    y = multi_adapter_forward(x, W, As, Bs, ids, 0.5)
    ref = torch.stack([x[i] @ (W + 0.5 * Bs[ids[i]] @ As[ids[i]]).T for i in range(5)])
    assert torch.allclose(y, ref, atol=1e-5)


def _peft_pair(use_dora=False, use_rslora=False):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    ours, ref = Block(), Block()
    ref.load_state_dict(ours.state_dict())
    apply_lora(ours, targets=("q_proj",), r=4, alpha=8, dora=use_dora, scaling="rslora" if use_rslora else "standard")
    filled(ours.q_proj)
    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"], use_dora=use_dora, use_rslora=use_rslora,
                     lora_dropout=0.0)
    ref = get_peft_model(ref, cfg)
    q = ref.base_model.model.q_proj
    with torch.no_grad():
        q.lora_A["default"].weight.copy_(ours.q_proj.lora_A)
        q.lora_B["default"].weight.copy_(ours.q_proj.lora_B)
        if use_dora:
            m = torch.rand(40) + 0.5
            ours.q_proj.magnitude.copy_(m)
            q.lora_magnitude_vector["default"].weight.copy_(m)
    return ours, ref


@pytest.mark.parametrize("use_dora,use_rslora", [(False, False), (False, True), (True, False)])
def test_matches_peft(use_dora, use_rslora):
    ours, ref = _peft_pair(use_dora, use_rslora)
    x = torch.randn(7, 24)
    y_ours, y_ref = ours(x), ref(x)
    assert torch.allclose(y_ours, y_ref, atol=1e-5)
    y_ours.pow(2).sum().backward()
    y_ref.pow(2).sum().backward()
    q = ref.base_model.model.q_proj
    assert torch.allclose(ours.q_proj.lora_A.grad, q.lora_A["default"].weight.grad, atol=1e-5)
    assert torch.allclose(ours.q_proj.lora_B.grad, q.lora_B["default"].weight.grad, atol=1e-5)
    if use_dora:
        assert torch.allclose(ours.q_proj.magnitude.grad, q.lora_magnitude_vector["default"].weight.grad, atol=1e-5)
