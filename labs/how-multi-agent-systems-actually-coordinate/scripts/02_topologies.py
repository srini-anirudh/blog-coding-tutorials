"""Topology is the architecture, and talking has a bill.

Recomputes Table 1 of the post from graph definitions, the Figure 5 token bill, and how far a
prompt injection spreads along each graph.

    python labs/how-multi-agent-systems-actually-coordinate/scripts/02_topologies.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mascoord import topology as T  # noqa: E402
from mascoord.report import Report  # noqa: E402

out = Report("02_topologies")
POST = {"solo": (0, 0, "1"), "pipeline": (6, 6, "7"), "star": (6, 12, "3"), "hierarchy": (6, 12, "5"),
        "full mesh": (21, 42, "2"), "blackboard": (7, 14, "async")}

out("\n1. Table 1 at N = 7, computed from the graphs (post's values in brackets)")
out(f"{'topology':<11} {'channels':>12} {'msgs/round':>12} {'depth':>11} {'max inbox':>10}  failure")
for t in T.all_topologies(7):
    ch, msg, d = POST[t.name]
    out(f"{t.name:<11} {t.channels:>5} [{ch:>3}] {t.msgs_per_round:>5} [{msg:>3}] {str(t.depth):>5} [{d:>3}] "
        f"{t.max_inbox():>10}  {t.note}")
out("   max inbox = most messages one agent reads per round. The star coordinator reads 6;")
out("   a binary hierarchy caps it at 3 (two children and a parent).")

out("\n2. Channels as N grows: mesh is N(N-1)/2, star and hierarchy N-1, blackboard N")
out(f"{'N':>4} {'mesh':>6} {'star':>6} {'hier':>6} {'board':>6}")
for n in [4, 8, 16, 32, 64]:
    out(f"{n:>4} {T.mesh(n).channels:>6} {T.star(n).channels:>6} {T.hierarchy(n).channels:>6} "
        f"{T.blackboard(n).channels:>6}")

out("\n3. The token bill, N = 8 agents, R = 4 rounds, L = 1500 tokens per message")
b = T.comm_bill(8, 4, 1500)
out(f"   all-to-all, full traces      {b['mesh_full']:>9,} tokens")
out(f"   all-to-all, 180-token msgs   {b['mesh_structured']:>9,} tokens")
out(f"   blackboard, write + read     {b['blackboard']:>9,} tokens")
out(f"   the work itself (N R L)      {b['work']:>9,} tokens")
out(f"   useful share with full traces: {b['useful_fraction_mesh']:.1%}")

out("\n4. Prompt infection: agents compromised after 3 rounds, each message carries the payload")
out("   with probability beta (2000 trials). Start from a random agent, or from the last agent")
out("   (a pipeline's final stage, a star worker, a hierarchy leaf).")
for beta in [0.5, 0.2]:
    out(f"   beta = {beta}")
    out(f"   {'topology':<11} {'from random':>12} {'from leaf':>10}   of 7")
    for t in T.all_topologies(7)[1:]:
        r = T.blast_radius(t, rounds=3, beta=beta)
        out(f"   {t.name:<11} {r['from_random_agent']:>12.2f} {r['from_leaf']:>10.2f}")
