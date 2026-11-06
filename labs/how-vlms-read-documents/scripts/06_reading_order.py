"""Reading order: reproduce Figure 7's scores, then compare four orderers on synthetic pages with a known order.

Post section: "Reading order and long documents", Figure 7. The orderers: sort by top edge; recursive XY-cut;
XY-cut after dropping blocks in the page margins; and a small learned pairwise model (in the spirit of LayoutReader,
which learned order from Word documents). The synthetic pages are rectangles, not real scans.
"""

import argparse
import random
import statistics

import _common
from docread.order import (FIG7_BLOCKS, FIG7_REFERENCE, FIG7_SIZE, LAYOUTS, drop_margin_furniture, order_score,
                           raster_order, synthetic_page, train_orderer, xy_cut)

ap = argparse.ArgumentParser()
ap.add_argument("--pages", type=int, default=300, help="test pages per layout")
ap.add_argument("--train-pages", type=int, default=300, help="training pages per layout for the learned orderer")
args = ap.parse_args()
log = _common.Report("06_reading_order")
data = {"figure7": {}, "synthetic": {}}

log("Figure 7 page (13 blocks: 11 body blocks, a running header and a page number):")
for name, seq in (("top to bottom", raster_order(FIG7_BLOCKS)), ("column-aware (XY-cut)", xy_cut(FIG7_BLOCKS)),
                  ("XY-cut, margin furniture dropped", xy_cut(drop_margin_furniture(FIG7_BLOCKS, FIG7_SIZE[1]))),
                  ("reference", FIG7_REFERENCE)):
    s = order_score(seq, FIG7_REFERENCE)
    data["figure7"][name] = {"order": seq, "score": s}
    log(f"  {name:<34} {s:.2f}  {' '.join(seq)}")
log("  (Figure 7's own script scores the same blocks with the same formula: lev / max(len), lower is better)\n")

rng = random.Random(0)
train = [synthetic_page(L, rng) for L in LAYOUTS for _ in range(args.train_pages)]
model = train_orderer(train, steps=6000)
rng = random.Random(1)
log(f"Synthetic pages: {args.pages} test pages per layout. Learned orderer trained on {args.train_pages} other pages per")
log("layout, on body blocks only. Score = normalized edit distance to the reference order (0 is perfect).")
log(f"{'layout':<30} {'top-to-bottom':>13} {'XY-cut':>7} {'XY-cut - margins':>17} {'learned':>8}   exact (XY - margins / learned)")
for layout in LAYOUTS:
    rows = {"top-to-bottom": [], "XY-cut": [], "XY-cut - margins": [], "learned": []}
    for _ in range(args.pages):
        blocks, ref = synthetic_page(layout, rng)
        body = drop_margin_furniture(blocks, 1400)
        rows["top-to-bottom"].append(order_score(raster_order(blocks), ref))
        rows["XY-cut"].append(order_score(xy_cut(blocks), ref))
        rows["XY-cut - margins"].append(order_score(xy_cut(body), ref))
        rows["learned"].append(order_score(model.order(body, 1000, 1400), ref))
    mean = {k: statistics.mean(v) for k, v in rows.items()}
    exact = {k: sum(x == 0 for x in v) / len(v) for k, v in rows.items()}
    data["synthetic"][layout] = {"mean": mean, "exact": exact}
    log(f"{layout:<30} {mean['top-to-bottom']:>13.3f} {mean['XY-cut']:>7.3f} {mean['XY-cut - margins']:>17.3f} "
        f"{mean['learned']:>8.3f}   {exact['XY-cut - margins']:.2f} / {exact['learned']:.2f}")
log("\nXY-cut cuts at the widest empty band. When loose boxes reach across the gutter there is no vertical band to")
log("cut, and it falls back to top-to-bottom order for that region. The learned model sees two boxes at a time.")
log("That is enough to ignore a loose box, but not to learn rules that depend on the rest of the page, such as")
log("where a full-width figure sits between two column bands. LayoutReader reads all blocks at once for this reason.")
log.save(data)
