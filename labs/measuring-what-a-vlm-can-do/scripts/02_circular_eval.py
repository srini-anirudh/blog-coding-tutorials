"""CircularEval on simulated models (post Figure 7). The real-model version is in 11_analyse.py.

Three simulated answerers on 1,500 four-option questions:
  guesser     picks a position uniformly at random
  biased      always picks the first position it does not know the answer for
  knower      knows a fraction k of the answers, guesses the rest uniformly

Single pass gives the guesser 25%. CircularEval asks each question four times with the options
rotated and gives credit only for four correct passes, so chance falls to (1/4)^4 = 1/256.

    python labs/measuring-what-a-vlm-can-do/scripts/02_circular_eval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from benchlab.mc import LETTERS, MCItem, circular_correct, circular_variants  # noqa: E402
from benchlab.report import Report  # noqa: E402

N, K = 1500, 4
g = torch.Generator().manual_seed(0)
items = [MCItem(f"q{i}", [f"opt{j}" for j in range(K)], int(torch.randint(0, K, (1,), generator=g)))
         for i in range(N)]
knows = {}  # question -> does the model know it (fixed per question, not per pass)


def guesser(v: MCItem, q: int) -> str:
    return LETTERS[int(torch.randint(0, K, (1,), generator=g))]


def first_position(v: MCItem, q: int) -> str:
    return "A"


def make_knower(k: float, fallback):
    def answer(v: MCItem, q: int) -> str:
        if q not in knows:
            knows[q] = bool(torch.rand(1, generator=g) < k)
        return v.gold_letter if knows[q] else fallback(v, q)
    return answer


def evaluate(model) -> tuple[float, float]:
    single, circ = 0, 0
    for q, it in enumerate(items):
        vs = circular_variants(it)
        picks = [model(v, q) for v in vs]
        single += picks[0] == vs[0].gold_letter
        circ += circular_correct(picks, vs)
    return single / N, circ / N


r = Report("02_circular_eval")
r(f"{N} simulated questions, {K} options, gold letter uniform. Exact chance under CircularEval: "
  f"(1/4)^4 = {0.25 ** 4:.4f}")
r(f"{'model':38s} {'single pass':>12s} {'CircularEval':>13s}")
rows = [("uniform guesser", guesser), ("always 'A' (position bias)", first_position)]
for k in (0.4, 0.7):
    rows.append((f"knows {k:.0%}, guesses the rest", make_knower(k, guesser)))
    rows.append((f"knows {k:.0%}, picks 'A' otherwise", make_knower(k, first_position)))
for name, model in rows:
    knows.clear()
    s, c = evaluate(model)
    r(f"{name:38s} {s:12.3f} {c:13.3f}")
r("\nA model that knows k of the answers keeps about k under CircularEval, plus (1/4)^4 of the rest")
r("when it guesses. A pure position bias scores about 1/4 on one pass and exactly 0 under CircularEval.")
r.save()
