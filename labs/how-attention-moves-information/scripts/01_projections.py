"""Why three projections. Section "Why three projections" and the permutation note in the opening section.

Trains small one-layer routers on two directional tasks with three score rules. About 1 minute on 8 CPU threads.
"""

import _setup  # noqa: F401

from attnmoves.common import Report
from attnmoves.projections import identifiability_gap, permutation_check, raw_self_attention_stats, train_router

rep = Report("01_projections")
rep("Part 1. softmax(X X^T) X with no parameters, 64 random vectors in R^64")
rows = []
for scaled in (False, True):
    s = raw_self_attention_stats(scaled=scaled)
    rows.append(["X X^T / sqrt(d)" if scaled else "X X^T", s["max |S - S^T|"], s["rows whose argmax is self"], s["mean self weight"]])
rep.table(["scores", "max |S - S^T|", "rows whose argmax is self", "mean self weight"], rows)

rep("\nPart 2. Train one attention layer on directional tasks (mean of 2 seeds, 1500 Adam steps, d = 32)")
rep("cycle3: A reads B's payload, B reads C's, C reads A's. 8 payloads, so chance is 0.125.")
rep("previous: position i copies the token at i - 1, cyclic over 12 positions. Chance is 1/12.")
rows = []
for task in ("cycle3", "previous"):
    for rule in ("none", "tied", "separate"):
        res = [train_router(rule, task, seed=s) for s in (0, 1)]
        acc = sum(r["accuracy"] for r in res) / 2
        extra = res[0]["accuracy by type"] if task == "cycle3" else f"w(i-1) {res[0]['weight on i-1']:.2f}, w(self) {res[0]['weight on self']:.2f}"
        rows.append([task, rule, acc, str(extra)])
rep.table(["task", "score rule", "test accuracy", "seed 0 detail"], rows)
rep("Symmetric rules (none, tied) cannot hold a directed cycle of preferences. For cycle3 that caps them")
rep("at 2 of 3 rows. For previous they can get 11 of 12 rows, plus chance on the last (about 0.924).")

rep("\nPart 3. Separate projections with a narrower d_k, previous-token task, seed 0")
rows = [[d_k, train_router("separate", "previous", d_k=d_k)["accuracy"]] for d_k in (1, 2, 4, 32)]
rep.table(["d_k", "test accuracy"], rows)
rep("Rank 2 is enough here: a 2D rotation can map each position to the previous one.")

rep("\nPart 4. Only the product W_Q W_K^T is identifiable")
for k, v in identifiability_gap().items():
    rep(f"  {k}: {v if isinstance(v, int) else f'{v:.2e}'}")

rep("\nPart 5. Attention is permutation-equivariant until positions are added")
for k, v in permutation_check().items():
    rep(f"  {k}: {v:.2e}")
rep.save()
