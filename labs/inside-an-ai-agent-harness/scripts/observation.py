"""Section 9: one large output, six observation policies, one line that matters.

    python labs/inside-an-ai-agent-harness/scripts/observation.py

Writes results/observation.txt.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_lab.observation import POLICIES, apply_policy, build_output, grep_output, pytest_output  # noqa: E402
from harness_lab.report import Report  # noqa: E402
from harness.tokens import count_tokens  # noqa: E402  (path set by harness_lab)

BUDGET = 2000


def main() -> None:
    rep = Report("observation.txt")
    rep(f"Token budget per observation: {BUDGET} (C10 estimate). 'yes' = the explaining line is in the context.")
    rep("'spill to handle' includes one follow-up query; its tokens are counted.")
    rep()
    cases = []
    for at in (0.0, 0.5, 1.0):
        with tempfile.TemporaryDirectory() as d:
            text, key = grep_output(Path(d), 2000, at)
        cases.append((f"grep, answer at {at:.0%}", "grep", text, key, r"security"))
    with tempfile.TemporaryDirectory() as d:
        text, key = pytest_output(Path(d), 400, 200)
    cases.append(("pytest -s, failure in test 200 of 400", "pytest", text, key, r"^E\s"))
    for at in (0.5, 0.999):
        text, key = build_output(20000, at)
        cases.append((f"build log, error at {at:.1%}", "build", text, key, r"error"))

    rep(f"{'output':>38} {'lines':>7} {'tokens':>9} | " + " | ".join(f"{p:>15}" for p in POLICIES))
    for name, kind, text, key, query in cases:
        cells = []
        for p in POLICIES:
            obs = apply_policy(p, text, kind, BUDGET, query)
            cells.append(f"{'yes' if key in obs else 'no':>3} {count_tokens(obs):>11,}")
        rep(f"{name:>38} {len(text.splitlines()):>7,} {count_tokens(text):>9,} | " + " | ".join(cells))
    rep()
    rep("Each cell: survived?, then tokens the observation adds to the context.")
    rep.save()


if __name__ == "__main__":
    main()
