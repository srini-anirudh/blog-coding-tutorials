"""One task at three levels of action: pixel clicks, element ids, and an API call.

Post section: "Choosing the action space" (Figure 7, Table 2). The pixel agent makes the expert's decisions, but
each click lands with Gaussian error of sigma pixels around the target centre, so pointing is the only thing that
can go wrong. With no spare steps, task success should match (per-click hit rate)^4.
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actlab.desk import SIZE, DeskEnv, random_task  # noqa: E402
from actlab.harness import Rejected, validate  # noqa: E402
from actlab.observe import elements  # noqa: E402
from actlab.report import Report  # noqa: E402

N = 2000
rep = Report("05_action_levels")


def target_of(env, x, y):
    return next((e["id"] for e in elements(env.state) if e["box"].contains(x, y)), None)


def pixel_agent(env, rng, sigma):
    """Returns (decisions, pointing decisions, pointing hits)."""
    dec = pts = hits = 0
    while not env.done():
        a = env.expert_action()
        dec += 1
        if a[0] == "click":
            want = target_of(env, a[1], a[2])
            x, y = a[1] + rng.gauss(0, sigma), a[2] + rng.gauss(0, sigma)
            pts += 1
            try:
                a = validate(("click", x, y), SIZE, SIZE)
            except Rejected:
                env.steps += 1
                continue
            hits += target_of(env, x, y) == want
        env.step(a)
    return dec, pts, hits


def element_agent(env, hide_options=False):
    """Acts on element ids from the accessibility tree. The harness clicks the element's centre."""
    dec = 0
    cat, amt = env.subtask
    plan = [("set", "amount", amt), ("click", "category"), ("click", f"opt-{cat.lower()}"), ("click", "btn-submit")]
    for verb, eid, *text in plan:
        dec += 1
        els = {e["id"]: e for e in elements(env.state)}
        if hide_options:  # a canvas-drawn menu: the options are on screen but not in the tree
            els = {k: v for k, v in els.items() if not k.startswith("opt-")}
        try:
            validate(("element", eid), SIZE, SIZE, element_ids=els)
        except Rejected as err:
            return dec, str(err)
        env.step(("click", *els[eid]["box"].center))
        if verb == "set":
            env.step(("type", text[0]))
    return dec, None


def api_agent(env):
    """submit_expense(amount, category): one call, no perception. It exists only because someone wrote it."""
    cat, amt = env.subtask
    env.state.records.append((cat, amt))
    return 1


rows = []
rep(f"{N} single-report tasks per row, random layouts. Exact budget: 5 steps for the pixel agent, no retries.")
rep(f"{'agent':40s} {'decisions':>9s} {'with coords':>11s} {'hit/click':>9s} {'hit^4':>7s} {'success':>8s}")
for sigma in (1.0, 2.0, 3.0):
    for budget in (5, 10):
        rng, wins, stats = random.Random(int(sigma * 10)), 0, []
        for k in range(N):
            env = DeskEnv(random_task(rng, 1), seed=k, budget=budget)
            stats.append(pixel_agent(env, rng, sigma))
            wins += env.success()
        d, p, h = np.array(stats).sum(0)
        hr = h / p
        name = f"pixel, sigma {sigma:.0f}px, budget {budget}"
        rows.append(dict(agent=name, decisions=d / N, pointing=p / N, hit=hr, hit4=hr ** 4, success=wins / N))
        rep(f"{name:40s} {d / N:9.2f} {p / N:11.2f} {hr:9.4f} {hr ** 4:7.3f} {wins / N:8.3f}")
for hide in (False, True):
    rng, wins, decs, errs = random.Random(1), 0, [], set()
    for k in range(N):
        env = DeskEnv(random_task(rng, 1), seed=k)
        d, err = element_agent(env, hide)
        decs.append(d)
        wins += env.success()
        if err:
            errs.add(err.split("'")[1])
    name = "element ids" + (", options missing from tree" if hide else "")
    rows.append(dict(agent=name, decisions=float(np.mean(decs)), pointing=0, success=wins / N))
    rep(f"{name:40s} {np.mean(decs):9.2f} {0:11.2f} {'-':>9s} {'-':>7s} {wins / N:8.3f}"
        + (f"   harness rejected: {sorted(errs)[0]}..." if errs else ""))
wins = 0
for k in range(N):
    env = DeskEnv(random_task(random.Random(k), 1), seed=k)
    api_agent(env)
    wins += env.success()
rows.append(dict(agent="API call", decisions=1, pointing=0, success=wins / N))
rep(f"{'API call':40s} {1:9.2f} {0:11.2f} {'-':>9s} {'-':>7s} {wins / N:8.3f}")
rep("")
rep("Post's example: 0.98^4 =", f"{0.98 ** 4:.4f}")
rep.save({"n": N, "rows": rows})
