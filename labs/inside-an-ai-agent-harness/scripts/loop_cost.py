"""Sections 3, 5, 16 and 17 of the post: what a turn costs, and what breaks the cache.

    python labs/inside-an-ai-agent-harness/scripts/loop_cost.py

Writes results/loop_cost.txt. Token counts only: no model runs here.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness_lab.loop_cost import (compaction_turn, edit_item_cost, effective_input, plain_turn,  # noqa: E402
                                   state_change_turn, tool_loading_turn, tool_order_turn, tool_tax,
                                   total_input)
from harness_lab.report import Report  # noqa: E402

P0, DELTA, R = 14_000, 2_000, 0.1
TOOLS = ["github__create_issue", "github__list_prs", "github__get_file", "slack__post", "slack__search",
         "jira__create", "jira__transition", "fs__read", "fs__write", "fs__list", "shell__run", "web__fetch",
         "web__search", "db__query", "db__schema", "browser__click", "browser__open", "notes__append",
         "calendar__list", "mail__send"]


def set_orders(n: int) -> list[list[str]]:
    """Iterate the same set of tool names in n fresh Python processes.

    String hashing is salted per process, so a set's order changes between runs.
    This is the kind of order bug the post describes for MCP tool lists.
    """
    code = f"print('\\n'.join(set({TOOLS!r})))"
    out = []
    for seed in range(n):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
        out.append(res.stdout.split())
    return out


def main() -> None:
    rep = Report("loop_cost.txt")
    rep(f"Prefix P0 = {P0} tokens, growth delta = {DELTA} tokens per step, cached tokens billed at r = {R}.")
    rep("Block-hashed prefix cache, 16-token blocks. Prefix warm from an earlier turn unless noted.")
    rep()
    rep("1. The naive loop is quadratic (section 3)")
    rep(f"{'K':>4} {'closed-form T':>14} {'sent (sim)':>12} {'computed, cache on':>19} {'billed, r=0.1':>14} {'T / |C_1|':>10}")
    for K in (1, 10, 20, 40, 60):
        t = plain_turn(K, P0, DELTA)
        rep(f"{K:>4} {total_input(K, P0, DELTA):>14,} {t.sent:>12,} {t.computed:>19,} {t.billed(R):>14,.0f} "
            f"{t.sent / (P0 + DELTA):>10.1f}")
    cold = plain_turn(60, P0, DELTA, warm=False)
    rep(f"K = 60 with a cold cache: computed {cold.computed:,} = P0 + K delta = {P0 + 60 * DELTA:,}")
    rep(f"K = 60: T / delta = {total_input(60, P0, DELTA) / DELTA:,.0f}")
    rep()

    rep("2. Cache hit rate h, K = 60, r = 0.1: closed-form T_eff against the simulation, 400 seeds")
    rep("   (1000-token blocks here: P0 and delta are multiples of 1000, so the counts are the same, only faster)")
    rep(f"{'h':>5} {'closed form':>12} {'simulated mean':>15} {'std error':>10} {'sim / closed':>13}")
    for h in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        sims = [plain_turn(60, P0, DELTA, h=h, seed=s, block_size=1000).billed(R) for s in range(400)]
        cf = effective_input(60, P0, DELTA, h, R)
        se = statistics.stdev(sims) / len(sims) ** 0.5
        rep(f"{h:>5.2f} {cf:>12,.0f} {statistics.mean(sims):>15,.0f} {se:>10,.0f} {statistics.mean(sims) / cf:>13.3f}")
    rep()

    rep("3. A state change at step 20 of 40: append a message, or edit the block in place (section 5)")
    for lane in ("append", "edit"):
        reqs = state_change_turn(lane)
        rep(f"  {lane:>6}: computed at step 20 = {reqs[19].computed:>6,} tokens; "
            f"whole turn computed = {sum(q.computed for q in reqs):,}; sent = {sum(q.total for q in reqs):,}")
    q = edit_item_cost()
    rep(f"  Edit item 6 of a 40-item list (128 tokens each): recomputed {q.computed:,} tokens "
        f"= {q.computed // 128} items (items 6 to 40), reused {q.cached // 128} items")
    rep()

    rep("4. Tool schemas in a fixed order, or in set-iteration order from a fresh process each call (section 5)")
    orders = set_orders(30)
    distinct = len({tuple(o) for o in orders})
    for name, ords in (("sorted", [sorted(TOOLS)] * 30), ("set order", orders)):
        reqs = tool_order_turn(ords)
        sent, cached = sum(q.total for q in reqs), sum(q.cached for q in reqs)
        rep(f"  {name:>9}: {len({tuple(o) for o in ords}):>2} distinct orders in 30 calls; "
            f"cached {cached / sent:6.1%} of {sent:,} sent; computed {sent - cached:,}")
    rep(f"  ({distinct} distinct set orders from 30 hash seeds, {len(TOOLS)} tools of 160 tokens each)")
    rep()

    rep("5. Compaction is a scheduled cache miss (section 16): K = 200, summary = 4000 tokens")
    rep(f"{'threshold':>10} {'compactions':>12} {'max context':>12} {'computed':>12} {'cached':>13} "
        f"{'billed r=0.1':>13} {'summary out':>12}")
    for thr in (None, 200_000, 100_000, 50_000, 30_000, 22_000):
        t = compaction_turn(200, P0, DELTA, thr, 4_000)
        rep(f"{str(thr or 'none'):>10} {t.compactions:>12} {max(q.total for q in t.requests):>12,} "
            f"{t.computed:>12,} {t.cached:>13,} {t.billed(R):>13,.0f} {t.output_tokens:>12,}")
    rep()

    rep("6. The tool-context tax (section 17): M = 300 tools, tau = 120 tokens each, K = 40 steps")
    rep(f"  closed form K M tau = {tool_tax(40, 300, 120):,} schema tokens")
    for mode in ("eager", "deferred"):
        t, schema = tool_loading_turn(mode)
        rep(f"  {mode:>8}: schema tokens sent {schema:>10,}; turn sent {t.sent:>11,}; computed {t.computed:>8,}; "
            f"billed r=0.1 {t.billed(R):>11,.0f}")
    rep.save()


if __name__ == "__main__":
    main()
