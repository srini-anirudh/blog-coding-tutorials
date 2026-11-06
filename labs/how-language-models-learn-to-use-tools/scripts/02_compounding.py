"""Why single-call accuracy lies: p**K, the requirement target**(1/K), and what recovery buys.

    python labs/how-language-models-learn-to-use-tools/scripts/02_compounding.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from toollab.hw import header  # noqa: E402
from toollab.compound import required_p, success_exact, success_monte_carlo  # noqa: E402


def main() -> None:
    header()
    print("Task success p**K, and the approximation exp(-K * eps)")
    print(f"{'p':>6} {'K':>4} {'p**K':>8} {'exp(-K eps)':>12}")
    for p in (0.95, 0.99):
        for k in (1, 5, 10, 20, 50):
            print(f"{p:>6} {k:>4} {p ** k:>8.4f} {math.exp(-k * (1 - p)):>12.4f}")

    print("\nPer-step accuracy needed to hold task success at 0.9: p >= 0.9 ** (1/K)")
    for k in (10, 20, 50):
        print(f"  K = {k:>2}: p >= {100 * required_p(0.9, k):.4f}%")

    print("\nRecovery: success after K steps of the T/E/D chain. Exact (matrix power) vs Monte Carlo (100,000 runs).")
    print(f"{'p':>6} {'q':>5} {'K':>4} {'exact':>8} {'MC':>8}")
    for p, q in ((0.95, 0.0), (0.96, 0.0), (0.95, 0.5), (0.95, 0.8), (0.99, 0.0)):
        for k in (10, 50):
            print(f"{p:>6} {q:>5} {k:>4} {success_exact(p, q, k):>8.4f} {success_monte_carlo(p, q, k, 100_000):>8.4f}")

    k = 50
    base = success_exact(0.95, 0.0, k)
    print(f"\nAt K = {k}, starting from p = 0.95 with no recovery (success {base:.4f}):")
    print(f"  one more point of precision, p = 0.96: {success_exact(0.96, 0.0, k):.4f}")
    print(f"  recover half of all errors, q = 0.5:   {success_exact(0.95, 0.5, k):.4f}")


if __name__ == "__main__":
    main()
