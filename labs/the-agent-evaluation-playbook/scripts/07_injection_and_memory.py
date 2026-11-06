"""Data is not instruction: a provenance monitor stops an injected exfiltration, charges a utility bill, and a
poisoned memory note fires r / (1 - s) times.

The agent is scripted. When a fetched page contains an injected instruction it obeys with probability q.

    python labs/the-agent-evaluation-playbook/scripts/07_injection_and_memory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agenteval.harness import run_episode  # noqa: E402
from agenteval.report import Report  # noqa: E402
from agenteval.security import (attack_succeeded, contact_vendor_task, k_sends_agent, k_sends_task,  # noqa: E402
                                poison_firings, provenance_monitor, summarise_task, susceptible)
from agenteval.world import initial_state  # noqa: E402

TRIALS = 2000


def main() -> None:
    r = Report("07_injection_and_memory")
    rng = np.random.default_rng(0)
    s0 = initial_state()
    tasks = [summarise_task("https://shop.example/faq"), summarise_task("https://shop.example/review"),
             contact_vendor_task()]

    r.add(f"1. Attack success and utility, {TRIALS} harness episodes per cell")
    rows = []
    for q in (0.0, 0.3, 0.8):
        for mon_name, mon in (("none", None), ("provenance", provenance_monitor)):
            for t in tasks:
                ok = att = 0
                for i in range(TRIALS):
                    ep = run_episode(susceptible(q), t, s0, rng=rng, monitor=mon, episode=i)
                    ok += t.goal(ep.sT)
                    att += attack_succeeded(ep.sT)
                rows.append([f"{q:g}", mon_name, t.id, ok / TRIALS, att / TRIALS])
    r.table(["q (obeys injection)", "monitor", "task", "utility", "attack success"], rows)
    ep = run_episode(susceptible(1.0), tasks[1], s0, rng=rng, monitor=provenance_monitor)
    blocked = [s for s in ep.trace if s.monitor != "allow"]
    r.add(f"   hijacked episode under the monitor: blocked {blocked[0].tool}(to={blocked[0].args['to']!r}) "
          f"with rule '{blocked[0].monitor}'")
    ep = run_episode(susceptible(0.0), tasks[2], s0, rng=rng, monitor=provenance_monitor)
    r.add(f"   legitimate 'contact vendor' under the monitor: {ep.trace[-1].monitor}")
    r.add()

    r.add("2. The monitor's bill: K sensitive sends per task, each recipient comes from data with probability b,")
    r.add("   base success S = 0.9. Measured over 4,000 harness episodes per cell against S (1 - b)^K.")
    rows = []
    for b in (0.05, 0.15):
        for k in (1, 2, 4, 8):
            ok = 0
            for i in range(4000):
                t = k_sends_task(k, list(rng.random(k) < b))
                ep = run_episode(k_sends_agent(0.9), t, s0, rng=rng, monitor=provenance_monitor, max_calls=20)
                ok += t.goal(ep.sT)
            rows.append([f"{b:g}", str(k), ok / 4000, 0.9 * (1 - b) ** k])
    r.table(["b", "K", "defended success", "S (1-b)^K"], rows)

    r.add("3. One poisoned memory write. Retrieved with probability r per session, survives each session with")
    r.add("   probability s. 20,000 trials of 500 sessions each.")
    rows = []
    for s in (0.5, 0.9, 0.99, 1.0):
        f = poison_firings(s, 0.3, 500, 20_000, rng)
        f_filt = poison_firings(s, 0.3, 500, 20_000, rng, filter_untrusted=True)
        rows.append([f"{s:g}", float(f.mean()), 0.3 / (1 - s) if s < 1 else float("inf"),
                     0.3 * (1 - s ** 500) / (1 - s) if s < 1 else 0.3 * 500, float(f_filt.mean())])
    r.table(["s", "mean firings", "r/(1-s)", "r(1-s^T)/(1-s), T=500", "with write filter"], rows, fmt="{:.2f}")
    r.add("   At s = 1 (append-only) the count grows with the number of sessions instead of converging.")
    r.save()


if __name__ == "__main__":
    main()
