"""The computer-use loop on MiniDesk, and why a screenshot is not the state.

Post sections: "From answering to acting" (Figure 1 and the nuance on screenshots versus state).
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actlab.desk import DeskEnv, render  # noqa: E402
from actlab.observe import elements  # noqa: E402
from actlab.report import RESULTS, Report  # noqa: E402

rep = Report("01_loop_and_state")
task = [("Travel", "42.50")]

rep("Part 1. One episode of observe -> decide -> ground -> act. The expert plays the model.")
env = DeskEnv(task, seed=3)
frames = []
while not env.done():
    shot = render(env.state)
    frames.append(shot)
    a = env.expert_action()
    if a[0] == "click":
        hit = next((e["name"] for e in elements(env.state) if e["box"].contains(a[1], a[2])), "background")
        decision = f"click {hit!r}"
        grounded = f"at ({a[1]:.0f}, {a[2]:.0f})"
    else:
        decision, grounded = f"type {a[1]!r}", "into focused field"
    env.step(a)
    rep(f"  turn {env.steps}: screenshot {shot.shape[1]}x{shot.shape[0]} -> {decision:22s} {grounded:20s}"
        f" -> records on server: {len(env.state.records)}")
frames.append(render(env.state))
rep(f"  checker (reads records, not pixels): success = {env.success()}")

rep("")
rep("Part 2. Submit once versus submit twice, on a form that keeps its values after saving.")
shots, recs = [], []
for n_submits in (1, 2):
    env = DeskEnv(task, seed=3, clear_after_submit=False)
    while not env.state.records:
        env.step(env.expert_action())
    for _ in range(n_submits - 1):
        env.step(("click", *env.state.layout.submit.center))
    shots.append(render(env.state))
    recs.append(list(env.state.records))
    rep(f"  {n_submits} submit(s): records = {env.state.records}, checker success = {env.state.records == task},"
        f" action log length = {len(env.log)}")
same = bool(np.array_equal(shots[0], shots[1]))
rep(f"  final screenshots identical, pixel for pixel: {same}")
rep("  -> a grader that looks only at the final picture cannot tell these apart; the record count and the log can.")

strip = np.concatenate([np.pad(f, ((2, 2), (2, 2), (0, 0)), constant_values=255) for f in frames], 1)
Image.fromarray(strip).resize((strip.shape[1] * 2, strip.shape[0] * 2), Image.NEAREST).save(RESULTS / "01_episode.png")
rep("  wrote results/01_episode.png (the screenshots of part 1, left to right)")
rep.save({"identical_screens": same, "records": recs})
