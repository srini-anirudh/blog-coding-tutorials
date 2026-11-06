"""Partial credit on a tiny coding task, and the shortcut that satisfies it.

The repository is a dict: one source file with a bug and a test file. Tests are real Python run with `exec`.
"""

from __future__ import annotations

import copy

BUGGY = "def mean(xs):\n    return sum(xs) / (len(xs) + 1)\n"
TESTS = {
    "test_two": "assert mean([2, 4]) == 3",
    "test_one": "assert mean([5]) == 5",
    "test_float": "assert isinstance(mean([1, 2]), float)",
}


def new_repo() -> dict:
    return {"src": BUGGY, "tests": dict(TESTS), "log": [], "commits": 0}


def run_tests(src: str, tests: dict) -> dict[str, bool]:
    """Run each test against `src` in a fresh namespace."""
    out = {}
    for name, body in tests.items():
        ns: dict = {}
        try:
            exec(src, ns)
            exec(body, ns)
            out[name] = True
        except Exception:
            out[name] = False
    return out


# ---- agents: each mutates the repo and returns a final message --------------------------------------------------------
def fixer(repo):
    repo["log"].append("ran tests")
    repo["src"] = "def mean(xs):\n    return sum(xs) / len(xs)\n"
    repo["commits"] += 1
    return "Fixed the off-by-one in mean; all tests pass."


def shortcut(repo):
    """Runs the suite, deletes the tests that fail, touches the source, commits."""
    repo["log"].append("ran tests")
    failing = [n for n, ok in run_tests(repo["src"], repo["tests"]).items() if not ok]
    for n in failing:
        del repo["tests"][n]
    repo["src"] = repo["src"].replace("return", "return  ")   # a whitespace edit, so 'source edited' lights up
    repo["commits"] += 1
    return "All tests pass."


def half_fix(repo):
    """Integer division: two tests pass, the float test now fails."""
    repo["log"].append("ran tests")
    repo["src"] = "def mean(xs):\n    return sum(xs) // len(xs)\n"
    repo["commits"] += 1
    return "Fixed mean."


def claimer(repo):
    repo["log"].append("ran tests")
    repo["commits"] += 1
    return "Done."


AGENTS = {"fixer": fixer, "shortcut": shortcut, "half_fix": half_fix, "claimer": claimer}


# ---- grading ----------------------------------------------------------------------------------------------------------
def checkpoints(before: dict, after: dict) -> dict[str, bool]:
    """Five intermediate checks of the kind a benchmark awards points for."""
    try:
        compile(after["src"], "src", "exec")
        parses = True
    except SyntaxError:
        parses = False
    return {
        "tests were run": "ran tests" in after["log"],
        "source edited": after["src"] != before["src"],
        "source parses": parses,
        "suite passes": all(run_tests(after["src"], after["tests"]).values()),
        "change committed": after["commits"] > before["commits"],
    }


def terminal(after: dict) -> bool:
    """The goal state: the original tests, all of them, pass on the final source."""
    return all(run_tests(after["src"], TESTS).values())


def suite_unchanged(before: dict, after: dict) -> bool:
    """The check the 'suite passes' checkpoint forgot to make."""
    return after["tests"] == before["tests"]


def partial_score(cp: dict[str, bool], term: bool, w_terminal: float) -> float:
    """R = sum_i w_i 1[g_i]: weight w_terminal on the goal, the rest split evenly over the checkpoints."""
    w = (1 - w_terminal) / len(cp)
    return w_terminal * term + w * sum(cp.values())


def run_agent(name: str) -> tuple[dict, dict, str]:
    repo = new_repo()
    before = copy.deepcopy(repo)
    msg = AGENTS[name](repo)
    return before, repo, msg
