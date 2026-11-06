"""The verifier era: best-of-N with an oracle checker and with an imperfect scorer (a simulation).

    python labs/how-reinforcement-learning-teaches-models-to-reason/scripts/02_best_of_n.py
"""

import _path  # noqa: F401

from rlvrlab.common import Report
from rlvrlab.verifier import best_of_n

NS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def main() -> None:
    rep = Report("02_best_of_n")
    settings = {"heavy-tailed scorer (fool rate 0.004)": 0.004, "same scorer, no confident errors": 0.0}
    curves = {}
    for label, fool in settings.items():
        c = best_of_n(NS, fool_rate=fool)
        curves[label] = c
        rep.print(label)
        rep.table(["N", "scorer picks", "oracle (pass@N)"], [[n, s, o] for n, s, o in zip(NS, c["scorer"], c["oracle"])])
        best = max(range(len(NS)), key=lambda i: c["scorer"][i])
        rep.print(f"peak: N={NS[best]}, accuracy {c['scorer'][best]:.3f}; at N={NS[-1]}: {c['scorer'][-1]:.3f}")
        rep.print()
    rep.print("2000 simulated problems, solve rate per problem ~ U(0.02, 0.5), scorer = 2*correct + N(0,1),")
    rep.print("plus +4 on a wrong candidate with the stated fool rate. Parameters are chosen, not fitted to any paper.")
    rep.data = {"ns": NS, "curves": curves}
    rep.save()


if __name__ == "__main__":
    main()
