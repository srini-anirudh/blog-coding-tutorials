"""The loss-chasing trap: allocate on the derivative of the loss, not its level (post section 10).

Five sources. Four are learnable. 'noise' has random answers, so its loss can never fall below about
4 * ln(10) / 5 = 1.84 nats per supervised token: a large irreducible part L_inf. Three schedules over 3,000 steps:

  static     uniform over all five sources (stratified sampling)
  level      every 100 steps, w_i proportional to source i's current held-out loss
  derivative every 100 steps, w_i proportional to the loss drop per unit of budget spent on i in the last window

Both controllers mix 10% uniform into their weights so that no source is starved to exactly zero.

    python labs/mixing-data-without-losing-capabilities/scripts/07_loss_chasing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mixlab.model import make_model  # noqa: E402
from mixlab.runs import Report, n_procs, parallel_map  # noqa: E402
from mixlab.train import evaluate, make_pools, normalise, scores, train  # noqa: E402

SOURCES = ("add", "rev", "sort", "copy", "noise")
REAL = SOURCES[:4]
STEPS, WINDOW, FLOOR = 3000, 100, 0.10
TR, EV = make_pools(SOURCES)
_, PROBE = make_pools(SOURCES, n_train=1, n_eval=256)  # small held-out set the controllers may look at


class Controller:
    def __init__(self, model, rule: str) -> None:
        self.model, self.rule = model, rule
        self.w = {s: 1 / len(SOURCES) for s in SOURCES}
        self.prev = None
        self.log = []

    def __call__(self, step, hist):
        if self.rule == "static" or (step - 1) % WINDOW:
            return self.w
        loss = {s: v["loss"] for s, v in evaluate(self.model, PROBE).items()}
        if self.rule == "level":
            raw = loss
        else:
            if self.prev is None:
                raw = {s: 1.0 for s in SOURCES}
            else:  # loss drop per unit of budget share spent on that source in the last window
                raw = {s: max(0.0, self.prev[s] - loss[s]) / self.w[s] for s in SOURCES}
                if sum(raw.values()) == 0:
                    raw = {s: 1.0 for s in SOURCES}
        self.prev = loss
        w = normalise(raw)
        self.w = {s: (1 - FLOOR) * w[s] + FLOOR / len(SOURCES) for s in SOURCES}
        self.log.append({"step": step, **{s: round(self.w[s], 3) for s in SOURCES}, "noise_loss": round(loss["noise"], 3)})
        return self.w


def run(cfg):
    rule, seed = cfg
    model = make_model(seed=seed)
    ctl = Controller(model, rule)
    h = train(model, TR, ctl, steps=STEPS, seed=seed)
    share = {s: h.examples_seen.get(s, 0) / (STEPS * 64) for s in SOURCES}
    return scores(evaluate(model, EV)), share, ctl.log


def main() -> None:
    rep = Report("07_loss_chasing", workers=n_procs())
    cfgs = [(r, s) for r in ("static", "level", "derivative") for s in (0, 1)]
    out = dict(zip(cfgs, parallel_map(run, cfgs)))
    rep(f"{STEPS} steps, batch 64, mean of 2 seeds. Budget share = fraction of all examples drawn from that source.")
    rows = []
    for rule in ("static", "level", "derivative"):
        r = [out[(rule, s)] for s in (0, 1)]
        acc = {t: sum(x[0][t] for x in r) / 2 for t in REAL}
        noise_share = sum(x[1]["noise"] for x in r) / 2
        rows.append([rule, f"{100 * noise_share:.1f}%", *(f"{acc[t]:.1f}" for t in REAL),
                     f"{sum(acc.values()) / 4:.1f}", f"{min(acc.values()):.1f}"])
        rep.data[rule] = [{"acc": x[0], "share": x[1], "weights": x[2]} for x in r]
    rep.table(["schedule", "budget to noise", *(f"{t} acc%" for t in REAL), "mean", "min"], rows)
    rep("")
    rep("weights chosen by the 'level' controller, seed 0 (every 500 steps)")
    log = out[("level", 0)][2]
    rep.table(["step", *SOURCES, "noise loss"], [[e["step"], *(e[s] for s in SOURCES), e["noise_loss"]] for e in log if (e["step"] - 1) % 500 == 0])
    rep("")
    rep("weights chosen by the 'derivative' controller, seed 0 (every 500 steps)")
    log = out[("derivative", 0)][2]
    rep.table(["step", *SOURCES, "noise loss"], [[e["step"], *(e[s] for s in SOURCES), e["noise_loss"]] for e in log if (e["step"] - 1) % 500 == 0])
    rep.save()


if __name__ == "__main__":
    main()
