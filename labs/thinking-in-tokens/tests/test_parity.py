import torch

from thinktok.parity import SEP, ParityConfig, answer_len, make_batch, train


def test_cot_targets_are_the_running_parity():
    g = torch.Generator().manual_seed(0)
    inputs, targets, answers = make_batch(6, 32, "cot", g)
    bits = inputs[:, :6]
    assert (inputs[:, 6] == SEP).all()
    running = torch.cumsum(bits, 1) % 2
    assert torch.equal(targets[:, 6:], running)  # after '=' the model writes p_1 ... p_n
    assert (targets[:, :6] == -100).all()  # no loss on the input bits
    assert torch.equal(answers, running[:, -1])


def test_direct_format_has_one_answer_token():
    g = torch.Generator().manual_seed(1)
    inputs, targets, answers = make_batch(5, 16, "direct", g)
    assert inputs.shape == (16, 6) and targets.shape == (16, 6)
    assert torch.equal(targets[:, -1], answers)
    assert (targets[:, :-1] == -100).all()
    assert torch.equal(answers, inputs[:, :5].sum(1) % 2)
    assert answer_len(5, "direct") == 1 and answer_len(5, "cot") == 5


def test_cot_parity_is_learned_quickly():
    r = train(ParityConfig(n=6, mode="cot", steps=80, eval_size=200, hidden=64), log_every=80, log=lambda s: None)
    assert r["final_acc"] > 0.95
    assert r["serial_depth_bound"] == 2 * 6
