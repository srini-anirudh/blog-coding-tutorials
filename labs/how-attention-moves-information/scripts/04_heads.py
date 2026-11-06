"""Why more than one head. Section "Why more than one head". About 1 minute on 8 CPU threads.

Same width (d = 32) and the same parameter count for 1, 2 and 4 heads. Task: output the tokens at i - 1 and i - 3.
"""

import _setup  # noqa: F401

from attnmoves.common import Report
from attnmoves.heads import ablate, train

rep = Report("04_heads")
rep("Train 2000 Adam steps, 12 positions, 12 tokens (chance accuracy 0.083), seed 0")
rows, models = [], {}
for h in (1, 2, 4):
    models[h], res = train(h)
    rows.append([h, 32 // h, res["params"], res["acc(i-1)"], res["acc(i-3)"],
                 str(res["weight on i-1 per head"]), str(res["weight on i-3 per head"])])
rep.table(["heads", "d_h", "params", "acc(i-1)", "acc(i-3)", "weight on i-1, per head", "weight on i-3, per head"], rows)
rep("One head learns one lookup and is at chance on the other. Two heads split the job, one lookup each.")

rep("\nPrune the trained 4-head model: keep a subset of heads, zero the rest, no retraining")
rows = []
for keep in (1, 2, 3):
    for r in ablate(models[4], keep):
        rows.append([str(r["heads kept"]), r["acc(i-1)"], r["acc(i-3)"]])
rep.table(["heads kept", "acc(i-1)", "acc(i-3)"], rows)
rep("Two heads that cover both lookups keep the task. This toy echoes Michel et al. (2019) on redundant")
rep("heads. It is not a reproduction of their pruning results on trained translation or BERT models.")
rep.save()
