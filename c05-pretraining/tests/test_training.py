"""Optimiser groups, schedule, accumulation, clipping, checkpointing and exact resume."""

import copy
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from pretrain.activation_ckpt import apply_activation_checkpointing, saved_tensor_bytes
from pretrain.checkpoint import latest_valid_checkpoint, verify
from pretrain.config import ModelArgs
from pretrain.model import build_model
from pretrain.optim import param_groups
from pretrain.packing import IGNORE
from pretrain.schedule import build_scheduler, lr_multiplier
from pretrain.step import train_step
from pretrain.trainer import train

ARGS = ModelArgs(vocab_size=300, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                 intermediate_size=64, max_seq_len=32)
CPU = torch.device("cpu")


def tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    return build_model(ARGS)


def batch(n: int, L: int = 32, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, ARGS.vocab_size, (n, L), generator=g)
    return x, torch.roll(x, -1, 1)


# ------------------------------------------------------------------ optimiser and schedule
def test_param_groups_cover_every_parameter_once():
    m = tiny_model()
    decay, no_decay = param_groups(m, 0.1)
    ids = [id(p) for g in (decay, no_decay) for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    assert all(p.dim() == 1 for p in no_decay["params"])  # RMSNorm gains
    assert no_decay["weight_decay"] == 0.0


def test_schedule_matches_transformers():
    from transformers import get_cosine_with_min_lr_schedule_with_warmup

    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    hf = get_cosine_with_min_lr_schedule_with_warmup(opt, 10, 100, min_lr_rate=0.1)
    hf_values = []
    for _ in range(120):
        hf_values.append(opt.param_groups[0]["lr"])
        opt.step()
        hf.step()
    for s in range(100):  # past the end HF's cosine turns back up; ours holds min_lr
        ours = lr_multiplier(s, 10, 100, 0.1)
        # HF starts warmup at lr = 0 on step 0; ours starts at 1/warmup (no wasted step).
        expected = hf_values[s + 1] if s < 10 else hf_values[s]
        assert math.isclose(ours, expected, abs_tol=1e-9), s
    assert lr_multiplier(9, 10, 100, 0.1) == 1.0 and lr_multiplier(500, 10, 100, 0.1) == 0.1


# ------------------------------------------------------------------ gradient accumulation
def step_params(model, micro, grad_clip=float("inf")):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    train_step(model, opt, None, micro, CPU, "fp32", grad_clip)
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_accumulation_equals_big_batch():
    x, y = batch(8)
    big = step_params(tiny_model(), [(x, y)])
    accum = step_params(tiny_model(), [(x[i : i + 2], y[i : i + 2]) for i in range(0, 8, 2)])
    torch.testing.assert_close(accum, big, rtol=0, atol=2e-6)


def test_mean_of_means_is_wrong_with_unequal_token_counts():
    x, y = batch(8)
    y = y.clone()
    y[:2, 4:] = IGNORE  # first micro-batch has far fewer target tokens
    micro = [(x[i : i + 2], y[i : i + 2]) for i in range(0, 8, 2)]
    big = step_params(tiny_model(), [(x, y)])
    torch.testing.assert_close(step_params(tiny_model(), micro), big, rtol=0, atol=2e-6)

    m = tiny_model()  # the common bug: average each micro-batch's mean loss
    for xb, yb in micro:
        F.cross_entropy(m(xb).logits.reshape(-1, ARGS.vocab_size), yb.reshape(-1),
                        ignore_index=IGNORE).div(len(micro)).backward()
    torch.optim.SGD(m.parameters(), lr=0.1).step()
    naive = torch.cat([p.detach().flatten() for p in m.parameters()])
    assert (naive - big).abs().max() > 1e-4


def test_clip_reports_preclip_norm_and_clips():
    m = tiny_model()
    x, y = batch(4)
    ref = copy.deepcopy(m)
    F.cross_entropy(ref(x).logits.reshape(-1, ARGS.vocab_size), y.reshape(-1)).backward()
    manual = torch.sqrt(sum(p.grad.pow(2).sum() for p in ref.parameters())).item()
    res = train_step(m, torch.optim.SGD(m.parameters(), lr=0.0), None, [(x, y)], CPU, "fp32", grad_clip=0.01)
    assert math.isclose(res.grad_norm, manual, rel_tol=1e-5)


def test_nonfinite_gradient_skips_update():
    m = tiny_model()
    before = [p.detach().clone() for p in m.parameters()]
    with torch.no_grad():
        m.model.embed_tokens.weight[5] = float("nan")
    x, y = batch(2)
    x[0, 0] = 5
    res = train_step(m, torch.optim.AdamW(m.parameters(), lr=1e-2), None, [(x, y)], CPU, "fp32")
    assert res.skipped
    for p, b in zip(list(m.parameters())[1:], before[1:]):
        assert torch.equal(p, b)


# ------------------------------------------------------------------ activation checkpointing
def test_activation_checkpointing_same_grads_less_saved_memory():
    x, y = batch(4)

    def loss_fn(model):
        return F.cross_entropy(model(x).logits.reshape(-1, ARGS.vocab_size), y.reshape(-1))

    plain, ckpt = tiny_model(), apply_activation_checkpointing(tiny_model())
    saved_plain = saved_tensor_bytes(lambda: loss_fn(plain).backward())
    saved_ckpt = saved_tensor_bytes(lambda: loss_fn(ckpt).backward())
    assert saved_ckpt < saved_plain
    for p, q in zip(plain.parameters(), ckpt.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------ checkpoint and resume
def test_exact_resume_matches_uninterrupted_run(tiny_cfg):
    straight = copy.deepcopy(tiny_cfg)
    straight.out_dir = tiny_cfg.out_dir + "_straight"
    full = [r["loss"] for r in train(straight) if "loss" in r]

    first = [r["loss"] for r in train(tiny_cfg, stop_after=4) if "loss" in r]  # checkpoint at step 4
    second = [r["loss"] for r in train(tiny_cfg) if "loss" in r]  # new model, optimiser, loader
    assert len(first) == 4 and len(second) == 4
    assert first + second == full  # bitwise equal on CPU


def test_corrupt_checkpoint_falls_back(tiny_cfg):
    train(tiny_cfg)
    out = Path(tiny_cfg.out_dir)
    ckpts = sorted(out.glob("ckpt_*.pt"))
    assert [c.name for c in ckpts] == ["ckpt_000004.pt", "ckpt_000008.pt"]
    data = ckpts[-1].read_bytes()
    ckpts[-1].write_bytes(data[: len(data) // 2])  # simulate a crash mid-write on a non-atomic FS
    assert not verify(ckpts[-1])
    assert latest_valid_checkpoint(out) == ckpts[0]


def test_config_change_is_refused_on_resume(tiny_cfg):
    train(tiny_cfg, stop_after=4)
    tiny_cfg.lr = 1e-2
    with pytest.raises(AssertionError, match="config changed"):
        train(tiny_cfg)
