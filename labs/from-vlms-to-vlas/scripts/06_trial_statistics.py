"""How many robot trials does a comparison need? Checks the post's numbers and verifies them by simulation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vlalab.evalstats import (interval_coverage, normal_interval, power, simulated_power,  # noqa: E402
                              trials_needed, wilson_interval)
from vlalab.report import Report  # noqa: E402


def pct(x):
    return round(100 * x, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "06_trial_statistics"))
    args = ap.parse_args()
    rep = Report(args.out, " ".join(["scripts/06_trial_statistics.py"] + sys.argv[1:]))
    res = {}

    rep.line("95% intervals for a measured success rate of 50%:")
    rows = []
    for n in (10, 20, 50, 100, 385):
        lo, hi = normal_interval(0.5, n)
        wl, wh = wilson_interval(0.5, n)
        rows.append({"trials": n, "normal half-width (pts)": pct((hi - lo) / 2),
                     "Wilson interval (%)": f"{pct(wl)}-{pct(wh)}",
                     "normal coverage": round(interval_coverage(0.5, n, "normal"), 3),
                     "Wilson coverage": round(interval_coverage(0.5, n, "wilson"), 3)})
    rep.table(rows)
    res["intervals"] = rows
    rep.line("Coverage: share of 20000 simulated experiments whose interval contains the true 50%.")

    rep.line("\nTrials per policy for 80% power, two-sided 5% test:")
    rows = [{"A": pa, "B": pb, "trials per policy": trials_needed(pa, pb),
             "simulated power at that n": round(simulated_power(pa, pb, trials_needed(pa, pb)), 3)}
            for pa, pb in ((0.5, 0.6), (0.5, 0.7), (0.5, 0.8), (0.8, 0.9), (0.9, 0.95))]
    rep.table(rows)
    res["sample_size"] = rows

    rep.line("\nChance to detect the gap with 20 trials per policy (A = 50%):")
    rows = [{"B": pb, "gap (pts)": pct(pb - 0.5), "power, normal approx": round(power(0.5, pb, 20), 3),
             "power, simulated": round(simulated_power(0.5, pb, 20), 3)} for pb in (0.6, 0.7, 0.8, 0.9)]
    rep.table(rows)
    res["power_n20"] = rows
    rep.save_json(res)


if __name__ == "__main__":
    main()
