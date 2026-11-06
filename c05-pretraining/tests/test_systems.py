"""Guards, MFU, DDP, ZeRO, FSDP and the C4 integration."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from pretrain.config import ModelArgs
from pretrain.guards import (BadBatchError, NonFiniteGuard, SpikeDetector, TrainingDiverged, check_batch,
                             step_with_oom_retry)
from pretrain.mfu import flops_per_token, matmul_params, mfu
from pretrain.model import C4_DIR, build_model, count_params
from pretrain.step import lm_loss_sum, train_step
from pretrain.zero import deepspeed_config

ROOT = Path(__file__).resolve().parents[1]
ARGS = ModelArgs(vocab_size=300, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                 intermediate_size=64, max_seq_len=32)
CPU = torch.device("cpu")


def batch(n=4, L=32):
    x = torch.randint(0, ARGS.vocab_size, (n, L))
    return x, torch.roll(x, -1, 1)


def test_check_batch_rejects_bad_batches():
    x, y = batch()
    check_batch(x, y, ARGS.vocab_size)
    bad = x.clone()
    bad[0, 0] = ARGS.vocab_size
    with pytest.raises(BadBatchError, match="input ids"):
        check_batch(bad, y, ARGS.vocab_size)
    with pytest.raises(BadBatchError, match="shape"):
        check_batch(x, y[:, :-1], ARGS.vocab_size)


def test_spike_detector_flags_jump_and_keeps_stats_clean():
    det = SpikeDetector(k=6, warmup=20)
    for i in range(50):
        assert not det.update(3.0 + 0.01 * (-1) ** i)
    mean = det.mean
    assert det.update(4.0)
    assert det.mean == mean
    assert det.update(float("nan"))


def test_nonfinite_guard_escalates_only_when_consecutive():
    g = NonFiniteGuard(patience=3)
    for skipped in (True, True, False, True, True):
        g.update(skipped)
    with pytest.raises(TrainingDiverged):
        g.update(True)


def test_oom_retry_gives_same_update_as_no_oom():
    x, y = batch(8)

    def model_and_opt():
        torch.manual_seed(0)
        m = build_model(ARGS)
        return m, torch.optim.SGD(m.parameters(), lr=0.1)

    ref, opt = model_and_opt()
    train_step(ref, opt, None, [(x, y)], CPU, "fp32", float("inf"))

    m, opt = model_and_opt()
    calls = []

    def flaky(micro):
        calls.append(len(micro))
        if micro[0][0].size(0) > 2:
            raise torch.OutOfMemoryError("fake out of memory")
        return train_step(m, opt, None, micro, CPU, "fp32", float("inf"))

    _, n = step_with_oom_retry(flaky, [(x, y)], opt, lambda: None)
    assert calls == [1, 2, 4] and n == 4
    for p, q in zip(ref.parameters(), m.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=2e-6)


def test_flop_formula_close_to_flop_counter():
    a = ModelArgs(vocab_size=512, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=4,
                  intermediate_size=172, max_seq_len=64)
    m = build_model(a)
    x = torch.randint(0, a.vocab_size, (1, 64))
    with FlopCounterMode(display=False) as fc:
        lm_loss_sum(m(x).logits, torch.roll(x, -1, 1)).backward()
    counted = fc.get_total_flops() / 64  # the counter reports 2 FLOPs per multiply-add, like the formula
    assert 0.85 < flops_per_token(m, a, 64) / counted < 1.15


def test_mfu_arithmetic():
    assert mfu(1000.0, 1e6, 1e10) == pytest.approx(0.1)
    m = build_model(ARGS)
    assert matmul_params(m, ARGS) == count_params(m)  # tied: V x D counted once, for the output


def test_ddp_two_processes_match_one_big_batch(tmp_path):
    """Real two-process DDP with gloo: same weights as one process with the whole batch."""
    out = tmp_path / "ddp.json"
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_port", "29465",
         str(ROOT / "scripts/ddp_equivalence.py"), "--out", str(out)],
        check=True, capture_output=True, timeout=120,
    )
    r = json.loads(out.read_text())
    assert r["ranks_identical"]
    assert r["ddp_vs_single"] < 2e-6
    assert r["hand_vs_ddp"] < 2e-6


def test_deepspeed_config_shape():
    cfg = deepspeed_config(3, 16, 2, 3e-3, 1.0)
    assert cfg["zero_optimization"]["stage"] == 3 and cfg["bf16"]["enabled"]


@pytest.mark.skipif(not C4_DIR.exists(), reason="C4 not present")
def test_tinyllm_trains_with_the_same_step():
    a = ModelArgs(vocab_size=300, hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                  intermediate_size=64, max_seq_len=32)
    torch.manual_seed(0)
    tiny, llama = build_model(a, "tinyllm"), build_model(a, "llama")
    assert count_params(tiny) == count_params(llama)  # same architecture, same parameter count
    x, y = batch(4)
    opt = torch.optim.AdamW(tiny.parameters(), lr=3e-3)
    losses = [train_step(tiny, opt, None, [(x, y)], CPU, "fp32", 1.0).loss for _ in range(25)]
    assert losses[-1] < losses[0] - 0.5


@pytest.mark.skipif(not C4_DIR.exists(), reason="C4 not present")
def test_tinyllm_activation_checkpointing_same_grads():
    from pretrain.activation_ckpt import apply_activation_checkpointing

    torch.manual_seed(0)
    plain = build_model(ARGS, "tinyllm")
    torch.manual_seed(0)
    ckpt = apply_activation_checkpointing(build_model(ARGS, "tinyllm"))
    x, y = batch(4)
    lm_loss_sum(plain(x).logits, y).backward()
    lm_loss_sum(ckpt(x).logits, y).backward()
    for p, q in zip(plain.parameters(), ckpt.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.gpu
def test_fsdp_on_gpus_halves_local_params():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA GPUs")
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2",
         str(ROOT / "scripts/fsdp_train.py"), "--steps", "5"],
        check=True, timeout=300,
    )
