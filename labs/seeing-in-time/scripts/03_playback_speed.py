"""Uniform sampling changes the playback speed (the post's Apollo insight), shown with a speed classifier.

A blob moves at 1, 2 or 4 pixels per second. The model sees 8 frames and must name the speed.
  uniform: 8 frames spread over the whole video, so the per-frame step depends on the video's length.
  fps:     a clip of 8 frames at 2 fps (Apollo-style), so the per-frame step is always speed / 2.
Train on short videos (4-8 s). Test on short videos and on long videos (16-32 s) the model never saw.
"""

from __future__ import annotations

import argparse

import torch

from _common import Report
from seeingtime.controls import train
from seeingtime.models import SpaceTimeCNN
from seeingtime.synth import speed_batch

TRAIN_DUR, LONG_DUR = (4.0, 8.0), (16.0, 32.0)


def run(rule: str, steps: int, seed: int, batch: int = 128):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = SpaceTimeCNN(3)

    def batch_fn(step):
        x, y = speed_batch(batch, TRAIN_DUR, rule, g=g)
        x = x + 0.05 * torch.randn(x.shape, generator=g)
        return (x,), y

    train(model, batch_fn, steps, lr=3e-3, loss_fn=torch.nn.functional.cross_entropy)
    model.eval()
    out = {}
    test_g = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for name, dur in (("short (4-8 s, as trained)", TRAIN_DUR), ("long (16-32 s, unseen)", LONG_DUR)):
            x, y = speed_batch(4000, dur, rule, g=test_g)
            pred = model(x).argmax(-1)
            out[name] = {"accuracy": float((pred == y).float().mean()),
                         "predicted_class_share": torch.bincount(pred, minlength=3).div(len(pred)).tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    rep = Report("03_playback_speed")
    rep.print("Speed classes: 1, 2, 4 pixels per second. Chance = 0.333.")
    rep.print(f"Training: {args.steps} steps, batch 128, videos of 4-8 s. {args.seeds} seeds.\n")
    rows = []
    for rule in ("uniform", "fps"):
        res = [run(rule, args.steps, s) for s in range(args.seeds)]
        rep.data[rule] = res
        for name in res[0]:
            accs = [r[name]["accuracy"] for r in res]
            share = [round(x, 2) for x in torch.tensor([r[name]["predicted_class_share"] for r in res]).mean(0).tolist()]
            rows.append([rule, name, f"{sum(accs) / len(accs):.3f}", f"{min(accs):.3f}-{max(accs):.3f}",
                         f"slow/mid/fast = {share}"])
    rep.table(["sampling", "test videos", "mean acc", "range", "predicted class share"], rows)
    rep.print("\nWith uniform sampling a long video's frames are further apart, so every ball looks fast.")
    rep.save()


if __name__ == "__main__":
    main()
