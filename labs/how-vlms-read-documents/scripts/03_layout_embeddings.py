"""What 2D box embeddings buy a LayoutLM-style encoder: find the TOTAL amount on synthetic receipts when the OCR
hands over the words in row order, in column order, or in no order. Then misread "TOTAL" as "T0TAL".

Post section: "Reading OCR words with their positions", Figure 3. Small synthetic receipts, not FUNSD or CORD.
"""

import argparse
import statistics
import time

import _common
from docread.layout import ORDERS, accuracy, make_split, train_layout

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--n-train", type=int, default=6000)
args = ap.parse_args()
log = _common.Report("03_layout_embeddings")

train = make_split(args.n_train, seed=0)  # one third each of rows, columns, shuffled
tests = {o: make_split(1000, seed=1, orders=[o]) for o in ORDERS}
typo = {o: make_split(1000, seed=1, orders=[o], ocr_typo=True) for o in ORDERS}
chance = statistics.mean(1 / sum(1 for i in ex[0] if i == 2) for ex in tests["rows"])  # 2 = [NUM]
log(f"{args.n_train} training receipts (orders mixed), 1000 test receipts per order, {args.steps} steps, "
    f"{args.seeds} seeds. Chance (guess one amount at random): {chance:.3f}")

data = {"chance": chance, "steps": args.steps, "results": {}}
for use_2d in (False, True):
    name = "text + 1D + 2D boxes" if use_2d else "text + 1D position"
    accs = {o: [] for o in ORDERS}
    accs.update({f"{o}, T0TAL typo": [] for o in ORDERS})
    t0 = time.time()
    for seed in range(args.seeds):
        model = train_layout(use_2d, train, steps=args.steps, seed=seed, lr=2e-3)
        for o in ORDERS:
            accs[o].append(accuracy(model, tests[o]))
            accs[f"{o}, T0TAL typo"].append(accuracy(model, typo[o]))
    data["results"][name] = accs
    log(f"\n{name}  ({time.time() - t0:.0f} s for {args.seeds} seeds)")
    for k, v in accs.items():
        log(f"  {k:<22} accuracy {statistics.mean(v):.3f}  (seeds: {', '.join(f'{x:.3f}' for x in v)})")
log("\nrows: OCR reads line by line. columns: all left-half words, then all right-half amounts.")
log("shuffled: no order at all. T0TAL typo: the key word is misread, so it arrives as [UNK].")
log.save(data)
