"""Linear attention: a fixed-size state against exact recall. Section "Linear attention". About 10 s on CPU.

No training. Store n random (key, value) pairs, query each stored key once, and read out the nearest value.
The reordering Q (K^T V) = (Q K^T) V and the recurrence are checked in C3: c03-attention/results/linear_demo.txt.
"""

import _setup  # noqa: F401

from attnmoves.common import Report
from attnmoves.recall import capacity, recall_table

rep = Report("06_linear_recall")
rep("Recall accuracy, d = 64, mean of 3 seeds. Chance is 1 / n. Softmax uses beta = 32 on unit keys.")
rows = [[r["n"], r["softmax"], r["linear"], r["elu+1"], r["elu+1 (32k)"], r["softmax floats"], r["linear floats"]]
        for r in recall_table(64, [16, 32, 64, 128, 256, 512, 1024, 4096])]
rep.table(["n pairs", "softmax", "linear (phi = id)", "elu+1", "elu+1 on 32k", "softmax cache floats", "linear state floats"], rows)
rep("The softmax cache grows with n and recall stays exact. The linear state stays at d^2 + d floats and")
rep("recall falls once n passes a few times d. The positive elu+1 map, on these untrained unit keys,")
rep("returns close to the mean value: every phi(k) . phi(k') is large. Learned projections would do better;")
rep("this shows the fixed-state limit, not a trained linear-attention model.")

rep("\nLargest n with linear (phi = id) recall >= 0.9, mean of 3 seeds")
rep.table(["d", "largest n", "n / d", "state floats"], [[d, (c := capacity(d)), c / d, d * d + d] for d in (32, 64, 128)])
rep.save()
