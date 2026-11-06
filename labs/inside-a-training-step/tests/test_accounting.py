"""The post's arithmetic, checked number by number."""

import pytest

from stepanatomy import accounting as A
from stepanatomy import flops, models

S = A.LLAMA2_7B


def test_param_count_matches_post():
    assert A.param_count(S) == pytest.approx(6.74e9, rel=1e-3)


def test_param_formula_matches_a_real_model():
    s = models.SMALL
    m = models.build(s)
    norms = (2 * s.layers + 1) * s.hidden  # the formula ignores norm gains
    assert sum(p.numel() for p in m.parameters()) == A.param_count(s) + norms


def test_block_shares():
    assert A.projection_share(S) == pytest.approx(0.92, abs=0.005)
    silu = next(o for o in A.block_ops(S) if o.name == "silu * mul")
    assert silu.hbm_bytes / 1e6 == pytest.approx(270, rel=0.01)
    assert {o.bound for o in A.block_ops(S) if o.name in A.PROJECTIONS} == {"compute"}
    assert {o.bound for o in A.block_ops(S) if "add" in o.name or "norm" in o.name} == {"memory"}


def test_tensor_sizes():
    t = A.tensor_sizes(S)
    assert t["attention scores"][2] / 1e9 == pytest.approx(1.07, abs=0.01)
    assert t["logits"][2] / 1e6 == pytest.approx(262, rel=0.01)


def test_crossover_and_time_wall():
    assert A.attention_crossover_seq(S) == pytest.approx(24_700, rel=0.01)
    assert 6 * A.param_count(S) * 2e12 == pytest.approx(8.1e22, rel=0.01)
    assert A.run_seconds(A.param_count(S), 2e12, 0.45) / 86400 / 365 == pytest.approx(5.8, abs=0.1)


def test_ledger_and_activations():
    st = A.model_state_bytes(S)
    assert st["total"] / 1e9 == pytest.approx(108, rel=0.01)
    assert st["params (bf16)"] / 1e9 == pytest.approx(13.5, rel=0.01)
    assert A.activation_bytes(S) / 1e9 == pytest.approx(17.2, rel=0.01)
    assert A.activation_bytes(S) / A.checkpointed_activation_bytes(S) == pytest.approx(10.7, abs=0.1)


def test_optimizer_numbers():
    assert A.ADAM_BYTES_PER_PARAM == 28
    assert A.ADAM_FLOPS_PER_PARAM / A.ADAM_BYTES_PER_PARAM == pytest.approx(0.43, abs=0.01)
    assert A.optimizer_step_seconds(S) * 1e3 == pytest.approx(75, rel=0.02)


def test_matmul_params_excludes_embedding_only():
    s = models.SMALL
    m = models.build(s)
    assert flops.matmul_params(m) == A.param_count(s) - s.vocab * s.hidden
