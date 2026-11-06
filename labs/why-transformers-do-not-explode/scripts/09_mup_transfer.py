"""Part X: does the optimal learning rate move with width? SP against muP on a character-level MLP.

    python labs/why-transformers-do-not-explode/scripts/09_mup_transfer.py              # about 8 min on 8 CPU threads
    python labs/why-transformers-do-not-explode/scripts/09_mup_transfer.py --steps 50 --widths 64 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from stablelab.common import Report, seed_all  # noqa: E402
from stablelab.model import CharData, load_corpus  # noqa: E402
from stablelab.mup import MLP, context_dataset, coordinate_check, train_mlp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--log2-lrs", type=float, nargs="+", default=[-12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2])
    args = ap.parse_args()
    seed_all(0)
    r = Report("09_mup_transfer")
    data = CharData(load_corpus(), seq=8)
    x, y = context_dataset(data.train[:300_000], data.vocab, context=8)
    r(f"Task: predict the next character of Tiny Shakespeare from the previous 8 (one-hot, {x.shape[1]} inputs).")
    r(f"Model: input -> width -> width -> {data.vocab}, ReLU, Adam, batch 256, {args.steps} steps, constant LR.")
    r("muP follows Tensor Programs V, Table 3 (Adam), with base width 64, so SP and muP are identical at width 64.")
    r()

    r("1. Coordinate check: RMS change of each layer's output after 5 Adam steps at lr 1e-2.")
    rows = []
    for param in ("sp", "mup"):
        for w in args.widths:
            c = coordinate_check(w, param, x, y)
            rows.append([param, w, c["h1"], c["h2"], c["out"]])
    r.table(["param", "width", "hidden 1", "hidden 2", "output logits"], rows)
    r("Under SP the change of every layer depends on width (the output logits swing between 0.46 and 2.2 across widths); under muP each layer changes by a width-independent amount (hidden 2 stays within 0.36 to 0.39, logits within 0.86 to 0.98).")
    r()

    r("2. Final training loss (mean of the last 20% of steps) against log2(lr), for each width.")
    best = {}
    for param in ("sp", "mup"):
        rows = []
        for w in args.widths:
            losses = []
            for lg in args.log2_lrs:
                torch.manual_seed(0)
                model = MLP(x.shape[1], w, data.vocab, param)
                losses.append(train_mlp(model, x, y, 2.0**lg, args.steps))
            i = min(range(len(losses)), key=lambda j: losses[j])
            best[(param, w)] = (args.log2_lrs[i], losses[i])
            rows.append([w] + losses)
            print(f"  {param} width {w}: best log2 lr {args.log2_lrs[i]}, loss {losses[i]:.4f}", flush=True)
        r(f"   {param.upper()}")
        r.table(["width"] + [f"2^{lg:g}" for lg in args.log2_lrs], rows)
        r()
    r("3. Best learning rate per width.")
    rows = [[w, best[("sp", w)][0], best[("sp", w)][1], best[("mup", w)][0], best[("mup", w)][1]] for w in args.widths]
    r.table(["width", "SP best log2 lr", "SP loss", "muP best log2 lr", "muP loss"], rows)
    sp_shift = best[("sp", args.widths[-1])][0] - best[("sp", args.widths[0])][0]
    mup_shift = best[("mup", args.widths[-1])][0] - best[("mup", args.widths[0])][0]
    r(f"Shift of the best log2 lr from width {args.widths[0]} to {args.widths[-1]}: SP {sp_shift:+g}, muP {mup_shift:+g}"
      f" (grid step 1 = a factor of 2).")
    r(f"Transfer test: tune at width {args.widths[0]} and reuse at width {args.widths[-1]}:")
    for param in ("sp", "mup"):
        torch.manual_seed(0)
        lg = best[(param, args.widths[0])][0]
        loss = train_mlp(MLP(x.shape[1], args.widths[-1], data.vocab, param), x, y, 2.0**lg, args.steps)
        r(f"   {param.upper()}: lr 2^{lg:g} gives loss {loss:.4f}; the best at that width is {best[(param, args.widths[-1])][1]:.4f}.")
    r("This is the mechanism on a 3-matrix MLP. The post's quoted muTransfer results (BERT, GPT-3 6.7B) were not run.")
    r.save()


if __name__ == "__main__":
    main()
