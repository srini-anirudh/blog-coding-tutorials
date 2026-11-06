"""Section "Vote": when plurality voting helps, simulated from stated answer marginals.

These are simulations of the estimator, not model outputs. Each case fixes q(y) and draws N answers
20,000 times. Script 04 measures the same curves on real model samples.

  python labs/thinking-in-tokens/scripts/03_vote_theory.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from thinktok.util import cpu_name  # noqa: E402
from thinktok.vote import simulate_vote  # noqa: E402

NS = [1, 3, 5, 11, 21, 41, 101]
CASES = {
    "right 40%, the other 60% spread thinly over 20 wrong answers (3% each)":
        {"right": 0.40, **{f"w{i}": 0.03 for i in range(20)}},
    "right 40%, one specific wrong answer at 45%, 15% spread over 5 others":
        {"right": 0.40, "w0": 0.45, **{f"w{i}": 0.03 for i in range(1, 6)}},
    "right 60%, the other 40% on one wrong answer":
        {"right": 0.60, "w0": 0.40},
    "systematic error: the same dropped minus sign 70%, right 20%, 10% scattered":
        {"right": 0.20, "w0": 0.70, **{f"w{i}": 0.02 for i in range(1, 6)}},
}


def main() -> None:
    lines = [f"# hardware: {cpu_name()}, 1 thread (pure Python)",
             "# command: python labs/thinking-in-tokens/scripts/03_vote_theory.py",
             "# simulated: P(plurality of N draws from q is the right answer), 20,000 trials per cell", "",
             "| answer marginal q | is right the mode? | " + " | ".join(f"N={n}" for n in NS) + " |",
             "|---|---|" + "---|" * len(NS)]
    data = {}
    for name, q in CASES.items():
        accs = [simulate_vote(q, "right", n, seed=n) for n in NS]
        data[name] = dict(zip(map(str, NS), accs))
        is_mode = "yes" if q["right"] == max(q.values()) else "no"
        lines.append(f"| {name} | {is_mode} | " + " | ".join(f"{a:.3f}" for a in accs) + " |")
    text = "\n".join(lines) + "\n"
    print(text)
    (LAB / "results" / "03_vote_theory.md").write_text(text)
    (LAB / "results" / "03_vote_theory.json").write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
