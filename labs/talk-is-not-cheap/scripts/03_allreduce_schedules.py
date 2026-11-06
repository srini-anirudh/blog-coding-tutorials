"""Sections 10 and 11: build every AllReduce algorithm, check it, count its bytes and rounds, and price it.

Runs on numpy in one process. The byte counts are exact. The times are the alpha-beta model, not measurements.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from commcost import alphabeta as ab  # noqa: E402
from commcost import schedules as S  # noqa: E402
from commcost import simulate as M  # noqa: E402
from commcost.report import Report, human_bytes, table  # noqa: E402

R = Report("03_allreduce_schedules", note="numpy simulator, exact byte counts, model times")

R("1. Every algorithm gives the exact sum on every rank. Bytes are the busiest rank's sent bytes, in units of n.")
rows, data = [], []
for p in (4, 8, 16, 64):
    n = p * 6  # divisible by p, so chunk sizes are equal and the counts are exact
    for name, f in S.ALLREDUCE.items():
        st = M.check_allreduce(p, n, f, dtype="float64")
        sent = st.max_sent / (n * 8)
        rows.append([p, name, st.rounds, f"{sent:.4f}", f"{ab.allreduce_lower_bound_bytes(1, p):.4f}"])
        data.append({"p": p, "algo": name, "rounds": st.rounds, "max_sent_over_n": sent})
for line in table(rows, ["p", "algorithm", "rounds", "bytes sent / n", "lower bound / n"]):
    R(line)
R("Ring and Rabenseifner sit exactly on the 2(p-1)/p bound. Naive is (p-1)n at the root, tree and recursive doubling")
R("are n log2 p. Rabenseifner also needs only 2 log2 p rounds, the latency floor's order; the ring needs 2(p-1).")

R()
R("2. The naive root's load at p = 64, n = 100 MB")
p, n = 64, 100e6
R(f"rank 0 moves {2 * (p - 1) * n / 1e9:.1f} GB, every other rank {2 * n / 1e9:.1f} GB")
R(f"rounds: naive {2 * (p - 1)}, tree {2 * 6}")

R()
R("3. The simulator's time equals the closed form (alpha = 12 us, 45 GB/s, p = 16, n = 64 MiB)")
alpha, beta, p, n = 12e-6, 1 / 45e9, 16, 64 * 2**20
closed = {"naive": ab.t_naive, "tree": ab.t_tree, "ring": ab.t_ring, "rabenseifner": ab.t_rhd}
rows = []
for name, f in closed.items():
    _, st = M.run(S.ALLREDUCE[name](p, p * 4), [np.zeros(p * 4) for _ in range(p)],
                  link=M.uniform(alpha, beta), bytes_per_element=n / (p * 4))
    rows.append([name, f"{st.time * 1e3:.3f}", f"{f(n, p, alpha, beta) * 1e3:.3f}"])
for line in table(rows, ["algorithm", "simulated ms", "closed form ms"]):
    R(line)

R()
R("4. Ring versus binomial tree: where does the ring start to win? (closed forms from the post)")
rows = []
cross = {}
for label, a, bw, pp in [("EFA-like", 12e-6, 45e9, 64), ("IB-like", 2e-6, 45e9, 64), ("EFA-like", 12e-6, 45e9, 1024),
                         ("EFA-like, half bandwidth", 12e-6, 22.5e9, 64)]:
    c = ab.crossover_size(ab.t_tree, ab.t_ring, pp, a, 1 / bw)
    cross[f"{label}, p={pp}"] = c
    rows.append([label, f"{a * 1e6:.0f} us", f"{bw / 1e9:.1f}", pp, human_bytes(c),
                 f"{2 * (pp - 1) * a * 1e3:.2f}"])
for line in table(rows, ["fabric", "alpha", "GB/s", "p", "ring wins above", "ring alpha term ms"]):
    R(line)
R("The post's text says the tree wins below roughly 100 KiB at p = 64, alpha = 12 us, 45 GB/s. Its own formulas,")
R("which its Figure 14 also uses, put the crossover near 5.9 MiB. Lowering alpha does move it left, as the post says.")

R()
R("5. Rabenseifner against both at p = 64, alpha = 12 us, 45 GB/s")
rows = []
for n in (4 << 10, 64 << 10, 1 << 20, 16 << 20, 256 << 20):
    ts = {k: f(n, 64, 12e-6, 1 / 45e9) for k, f in closed.items()}
    best = min(ts, key=ts.get)
    rows.append([human_bytes(n)] + [f"{ts[k] * 1e3:.3f}" for k in closed] + [best])
for line in table(rows, ["n", *[f"{k} ms" for k in closed], "fastest"]):
    R(line)

R()
R("6. In-network reduction moves 2n per rank; the ring moves 2(p-1)/p n")
for pp in (8, 64, 512):
    R(f"p = {pp:4d}: ring {ab.bytes_per_rank('ring', 1, pp):.3f} n, in-network 2.000 n; "
      f"alpha terms {2 * (pp - 1)} vs 2 messages (at 3 us: {2 * (pp - 1) * 3e-3:.2f} ms vs 0.006 ms)")
R.save({"table": data, "crossovers_bytes": cross})
