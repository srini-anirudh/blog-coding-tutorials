"""GRPO from first principles: baselines are free, the group mean costs a factor (G-1)/G, RLOO is exact.

    python labs/how-reinforcement-learning-teaches-models-to-reason/scripts/05_baselines.py
"""

import _path  # noqa: F401
import torch

from rlvrlab.bandit import estimator_stats, exact_gradient
from rlvrlab.common import Report

THETA = torch.tensor([0.5, 1.0, 0.0, -0.5, 0.3, -1.0], dtype=torch.float64)
C = torch.tensor([0.9, 0.6, 0.3, 0.1, 0.05, 0.0], dtype=torch.float64)  # verifier acceptance rate per response
ESTIMATORS = [("reinforce", None), ("constant b=0.5", 0.5), ("mean", None), ("rloo", None), ("grpo", None)]


def main() -> None:
    rep = Report("05_baselines")
    pi = THETA.softmax(-1)
    rep.print("One prompt, 6 whole responses, 0/1 verifier. J =", round(float(pi @ C), 4))
    rep.print("exact gradient on the logits:", [round(float(x), 4) for x in exact_gradient(THETA, C)])
    rep.print("200000 groups per cell. scale = projection of the mean estimate on the exact gradient (1 = unbiased).")
    rep.print("variance = trace of the covariance of one group's estimate / |exact gradient|^2.")
    rep.print()
    data = []
    for g in (2, 4, 8, 16, 64):
        rows = []
        for name, b in ESTIMATORS:
            kind = name if b is None else "reinforce"
            s = estimator_stats(THETA, C, kind, g, baseline=b)
            rows.append([name, s["scale"], s["cosine"], s["rel_bias"], s["rel_variance"]])
            data.append({"G": g, "estimator": name, **s})
        rep.print(f"G = {g}   (mean-centred prediction: scale = (G-1)/G = {(g - 1) / g:.4f})")
        rep.table(["estimator", "scale", "cosine", "rel. bias", "variance"], rows)
        rep.print()
    rep.data = {"theta": THETA.tolist(), "c": C.tolist(), "rows": data}
    rep.save()


if __name__ == "__main__":
    main()
