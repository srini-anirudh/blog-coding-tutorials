"""Section 14: the speculative sampling rule. CPU only, toy distributions over a 32-token vocabulary.

1. Exactness: the first emitted token follows the target distribution p, not the draft q.
2. Tokens per target pass: measured against (1 - alpha^(gamma+1)) / (1 - alpha), where alpha is the
   acceptance rate sum_x min(p, q). The same p and q are used at every position so alpha is constant.
The batch-size cost of verification (why speculation stops paying at large batch) is measured on GPU in
01_roofline.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from servekit.metrics import Report  # noqa: E402
from servekit.spec import acceptance_rate, expected_tokens, speculative_step  # noqa: E402


def main() -> None:
    rep = Report("09_speculative.txt")
    g = torch.Generator().manual_seed(0)
    V, trials = 32, 20000
    base = torch.randn(V, generator=g) * 2
    rows = []
    for noise in (0.3, 1.0, 3.0):  # how far the draft is from the target
        p1 = torch.softmax(base, -1)
        q1 = torch.softmax(base + torch.randn(V, generator=g) * noise, -1)
        alpha = acceptance_rate(p1, q1)
        for gamma in (2, 4, 8):
            p, q = p1.expand(gamma + 1, V), q1.expand(gamma, V)
            counts = torch.zeros(V)
            emitted = 0
            for _ in range(trials):
                out = speculative_step(p, q, g)
                counts[out[0]] += 1
                emitted += len(out)
            tv = 0.5 * (counts / trials - p1).abs().sum().item()
            rows.append({"draft_noise": noise, "alpha": alpha, "gamma": gamma, "tokens/pass": emitted / trials,
                         "formula": expected_tokens(alpha, gamma), "TV(first token, p)": tv,
                         "TV(q, p)": 0.5 * (q1 - p1).abs().sum().item()})
    rep(f"{trials} speculative steps per row; TV = total variation distance")
    rep.table(rows, {"draft_noise": "{:.1f}", "alpha": "{:.3f}", "tokens/pass": "{:.3f}", "formula": "{:.3f}",
                     "TV(first token, p)": "{:.4f}", "TV(q, p)": "{:.3f}"})
    rep(f"Post's example: alpha = 0.8, gamma = 4 -> {expected_tokens(0.8, 4):.4f} tokens per target pass")
    rep.save({"rows": rows})


if __name__ == "__main__":
    main()
