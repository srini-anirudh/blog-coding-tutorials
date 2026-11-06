"""Train a tiny icon grounder, then compare one pass at three token budgets with zoom search at the smallest one.

Post sections: "Grounding becomes action" and "Resolution decides what can be pointed at" (ScreenSpot-Pro and
ScreenSeekeR). The paper numbers (18.9% to 48.1%) are not reproduced; this shows the same mechanism at toy scale.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actlab.grounding import evaluate, train_grounder  # noqa: E402
from actlab.report import RUNS, Report, seed_threads  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--eval", type=int, default=500)
ap.add_argument("--retrain", action="store_true")
args = ap.parse_args()
seed_threads()

rep = Report("04_grounding_zoom")
ckpt = RUNS / f"grounder_{args.steps}.pt"
t0 = time.time()
if ckpt.exists() and not args.retrain:
    from actlab.grounding import Grounder
    model = Grounder()
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    rep(f"loaded {ckpt.name}")
else:
    model = train_grounder(steps=args.steps, log=rep)
    RUNS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt)
rep(f"training: {args.steps} steps, {time.time() - t0:.0f} s")
rep("screen 384 x 384 px, icons 12 x 12 px, one target glyph among 8 kinds; a click counts if inside the icon box")
rep(f"{args.eval} held-out screens, the same weights for every row")
rep("")
rep(f"{'method':42s} {'visual tokens':>13s} {'px per token':>13s} {'click accuracy':>15s}")
res = evaluate(model, n=args.eval)
for k, v in res.items():
    rep(f"{k:42s} {v['tokens']:13d} {v['px_per_token']:13.1f} {v['accuracy']:15.3f}")
rep(f"total time {time.time() - t0:.0f} s")
rep.save({"steps": args.steps, "n_eval": args.eval, "results": res})
