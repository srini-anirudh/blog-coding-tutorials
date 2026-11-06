import random

import pytest

from agentrl.env import FileSystemEnv, parse_action, run_episode
from agentrl.expert import ScriptedExpert, parse_task
from agentrl.failures import INVALID, MISSING, TIMEOUT, FailureConfig
from agentrl.shift import perturbations
from agentrl.tasks import Subtask, Task, sample_task


def small_task(**kw) -> Task:
    files = {"db/main": {"port": "5", "host": "7", "mode": "1"}, "app/dev": {"ttl": "30", "rate": "2", "size": "9"}}
    return Task(1, files, [Subtask("direct", "port", "db/main", "42")], **kw)


def test_reset_observe_step_reward_done():
    env = FileSystemEnv(small_task())
    assert env.reset() == "set port in db/main to 42" == env.observe()
    assert not env.done() and env.reward() == 0.0
    assert env.step("read db/main") == ("port = 5 ; host = 7 ; mode = 1", 0.0, False)
    assert env.step("edit db/main port 42") == ("ok", 0.0, False)
    obs, r, done = env.step("submit")
    assert (obs, r, done) == ("submitted", 1.0, True) and env.reward() == 1.0


def test_tools():
    env = FileSystemEnv(small_task())
    env.reset()
    assert env.step("list db")[0] == "db/main"
    assert env.step("search ttl")[0] == "found app/dev"
    assert env.step("search depth")[0] == "found none"
    assert env.step("read web/main")[0] == MISSING
    assert env.step("edit db/main depth 3")[0] == "error no such key"
    assert env.step("edit db/main port 420")[0] == INVALID
    assert env.step("edit db/main")[0] == INVALID           # malformed: wrong arg count
    assert env.step("delete db/main")[0] == INVALID         # unknown tool


def test_wrong_edit_fails_and_step_budget_ends_episode():
    env = FileSystemEnv(small_task(), max_steps=3)
    env.reset()
    env.step("edit db/main port 42")
    env.step("edit db/main host 1")                         # an extra change breaks equality with the target
    assert env.step("submit")[1] == 0.0
    env = FileSystemEnv(small_task(), max_steps=2)
    env.reset()
    env.step("read db/main")
    _, r, done = env.step("read db/main")
    assert done and r == 0.0 and not env.submitted


def test_parse_action():
    assert parse_action("edit a/b k 1") == ("edit", ["a/b", "k", "1"])
    assert parse_action("submit") == ("submit", [])
    assert parse_action("submit now") is None and parse_action("") is None


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_expert_solves_every_level(level):
    tasks = [sample_task(level, s) for s in range(200)]
    wins = [run_episode(FileSystemEnv(t), ScriptedExpert(t.text)).success for t in tasks]
    assert sum(wins) >= 199  # level 4 can, very rarely, exhaust its step budget on repeated failures


def test_expert_steps_equal_optimal_on_clean_levels():
    for level in (1, 2, 3):
        for s in range(50):
            task = sample_task(level, s)
            assert run_episode(FileSystemEnv(task), ScriptedExpert(task.text)).n_steps == task.optimal_steps


def test_tasks_are_deterministic_and_parse_back():
    a, b = sample_task(5, 11), sample_task(5, 11)
    assert a.text == b.text and a.files == b.files
    for level in range(1, 6):
        for s in range(30):
            assert len(parse_task(sample_task(level, s).text)) == len(sample_task(level, s).subtasks)


def test_failure_injection_is_seeded_and_recoverable():
    task = small_task(failures=FailureConfig(p_timeout=1.0))
    env = FileSystemEnv(task, seed=3)
    env.reset()
    assert env.step("search port")[0] == TIMEOUT             # search always times out at p=1
    assert env.step("edit db/main port 42")[0] == "ok"      # edit is not affected by timeouts
    task = sample_task(4, 0)
    runs = [run_episode(FileSystemEnv(task, seed=9), ScriptedExpert(task.text)) for _ in range(2)]
    assert [s.observation for s in runs[0].steps] == [s.observation for s in runs[1].steps]


def test_perturbed_start_is_harmless_for_the_expert():
    for s in range(100):
        task = sample_task(1 + s % 3, s)
        tr = run_episode(FileSystemEnv(task), ScriptedExpert(task.text), forced=perturbations(task, 2, s))
        # the expert reads only the task text, so it continues as if nothing happened
        assert tr.steps[0].forced and tr.success


def test_progress_is_zero_at_the_end_and_rises_with_milestones():
    task = sample_task(3, 4)
    tr = run_episode(FileSystemEnv(task), ScriptedExpert(task.text))
    p = [s.progress for s in tr.steps]
    assert p[-1] == 0.0 and all(b >= a for a, b in zip(p[:-2], p[1:-1])) and p[-2] == 1.0


def test_random_agent_rarely_succeeds():
    rng = random.Random(0)
    words = ["list", "read", "search", "edit", "submit", "db/main", "port", "42"]
    wins = 0
    for s in range(100):
        wins += run_episode(FileSystemEnv(sample_task(2, s)),
                            lambda _: " ".join(rng.choices(words, k=rng.randint(1, 4)))).success
    assert wins == 0
