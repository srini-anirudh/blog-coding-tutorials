import random

from harness_lab.permissions import (MODE_FIRST, SDK_ORDER, Action, attention, decide, mean_attention,
                                     team_policy)
from harness_lab.stopping import mean_capped, run_episode


def test_deny_rule_outranks_bypass_mode():
    a = Action("Read", "~/.aws/credentials")
    assert decide(team_policy(), a, "bypassPermissions", SDK_ORDER)[0] == "deny"
    assert decide(team_policy(), a, "bypassPermissions", MODE_FIRST)[0] == "allow"


def test_ask_rule_outranks_bypass_mode():
    assert decide(team_policy(), Action("Bash", "git push origin main"), "bypassPermissions")[0] == "ask"


def test_mean_attention_closed_form_and_decreasing():
    for N in (1, 7, 50):
        direct = sum(attention(n, 0.9, 10) for n in range(1, N + 1)) / N
        assert abs(direct - mean_attention(N, 0.9, 10)) < 1e-12
    assert mean_attention(10, 1, 10) > mean_attention(100, 1, 10)


def test_capped_mean():
    q, B = 0.9, 15
    assert abs(mean_capped(q, B) - sum(q ** n for n in range(B))) < 1e-12


def test_conjunctive_harness_never_declares_false_victory():
    rng = random.Random(0)
    assert all(run_episode("conjunctive", rng).status != "false victory" for _ in range(2000))
    rng = random.Random(0)
    assert any(run_episode("naive", rng).status == "false victory" for _ in range(2000))
