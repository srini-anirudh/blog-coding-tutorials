"""One transformer, two losses: a tiny Transfusion on the shapes dataset.

Text tokens get a next-token loss. Image patches (16 patches of 4 x 4 pixels, no VAE) get a rectified-flow loss.
L = L_LM + 5 * L_flow. Every training batch holds both layouts: caption then image (drawing), and image then
caption (reading).

Three runs, same seed and steps:
  A. the Transfusion mask: causal, but bidirectional inside the image; read-side noise up to t = 1
  B. ablation: causal attention inside the image too
  C. Transfusion's fix for reading: cap the noise at t = 0.5 when the image comes before its caption

Drawing is graded by the judge CNN (shape and side both right). Reading is caption accuracy on clean test images.
"""

from __future__ import annotations

import argparse
import time

import torch

import _shared
from seedraw import hybrid
from seedraw.common import Report, ascii_image, seed_all
from seedraw.readers import judge_accuracy

RUNS = {
    "A_bidirectional_image": dict(bidirectional_images=True, read_t_max=1.0),
    "B_causal_image": dict(bidirectional_images=False, read_t_max=1.0),
    "C_bidirectional_noise_cap": dict(bidirectional_images=True, read_t_max=0.5),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=96, help="trunk width")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--runs", nargs="*", default=list(RUNS), help="subset of runs, by name")
    ap.add_argument("--name", default="06_hybrid", help="results file name")
    args = ap.parse_args()

    seed_all(0)
    rep = Report(args.name)
    train, test = _shared.splits()
    judge = _shared.get_judge(train)
    rep("Attention mask of the drawing layout (1 = may attend). Rows are queries.")
    rep("Positions: <bos> shape side <boi> p1..p16 <eoi>")
    m = hybrid.transfusion_mask(hybrid.SEQ, [(4, 20)]).int()
    for row in m:
        rep("  " + "".join("#" if v else "." for v in row.tolist()))
    rep()

    cond_shape = torch.arange(4).repeat_interleave(100)
    cond_side = torch.arange(2).repeat(200)
    ev_test = {k: test[k][:1000] for k in ("images", "shape", "side")}

    def evaluate(model):
        seed_all(5)
        imgs = hybrid.draw(model, cond_shape, cond_side, steps=30)
        d = judge_accuracy(judge, imgs, cond_shape, cond_side)
        shp, sd = hybrid.read(model, ev_test["images"])
        return {"draw_shape_acc": d["shape_acc"], "draw_both_acc": d["both_acc"],
                "read_shape_acc": float((shp == ev_test["shape"]).float().mean()),
                "read_both_acc": float(((shp == ev_test["shape"]) & (sd == ev_test["side"])).float().mean())}

    results = {}
    for name in args.runs:
        cfg = RUNS[name]
        seed_all(0)
        model = hybrid.HybridModel(d=args.width, layers=args.layers, bidirectional_images=cfg["bidirectional_images"])
        t0 = time.time()
        log = hybrid.train_hybrid(model, train, args.steps, lam=5.0, lr=args.lr, read_t_max=cfg["read_t_max"], seed=0,
                                  eval_fn=evaluate, eval_every=args.eval_every)
        rep(f"Run {name}: {args.steps} steps, lr {args.lr}, width {args.width}, {time.time() - t0:.0f} s, {sum(p.numel() for p in model.parameters()):,} parameters")
        rep(f"  {'step':>5s} {'L_LM':>6s} {'L_flow':>7s} {'draw: shape':>11s} {'draw: shape+side':>16s} "
            f"{'read: shape':>11s} {'read: shape+side':>16s}")
        for r in log:
            rep(f"  {r['step']:5d} {r['lm']:6.3f} {r['flow']:7.4f} {r['draw_shape_acc']:11.3f} {r['draw_both_acc']:16.3f} "
                f"{r['read_shape_acc']:11.3f} {r['read_both_acc']:16.3f}")
        seed_all(5)
        sample = hybrid.draw(model, torch.tensor([1]), torch.tensor([1]), steps=30)
        rep("  drawn for the caption 'disk right':")
        rep(ascii_image(sample[0, 0]))
        results[name] = {"config": cfg, "log": log}
        rep()
    rep("Drawing: 400 samples, 30 Euler steps, graded by the judge. Reading: 1,000 clean test images, greedy caption.")
    rep.data = {"steps": args.steps, "lr": args.lr, "width": args.width, "layers": args.layers, "lambda": 5.0, "runs": results}
    rep.save()


if __name__ == "__main__":
    main()
