"""Telling the model when: frame-index ids, absolute-time ids, and text timestamps.

Part A (no training) builds the ids and the RoPE scores:
  - the post's Figure 5: two copies of one event at 1 fps and 2 fps, under frame-index and absolute-time ids
  - RoPE attention scores depend on the id gap, so frame-index ids give the same score to 3 s and 1.5 s gaps
  - which rotary frequencies each axis owns, contiguous (Qwen2-VL) against interleaved (Qwen3-VL)
  - how large the temporal ids get for an hour of video, and how V2PE shrinks the position range
Part B (training) asks a tiny transformer for the duration of an event, in seconds, from frames sampled
at an unknown frame rate. It compares no time signal, frame-index ids, absolute-time ids and text timestamps.
"""

from __future__ import annotations

import argparse

import torch

from _common import Report
from seeingtime.controls import train
from seeingtime.models import TinyVideoTransformer
from seeingtime.positions import (axis_wavelengths, frequency_axes, inv_frequencies, mrope_angles, rotate,
                                  temporal_interval, timestamp_text, v2pe_positions, vision_position_ids)
from seeingtime.synth import event_batch


def part_a(rep: Report) -> None:
    rep.print("== Part A: ids and RoPE scores ==")
    rep.print("Figure 5: one 4-second event, video A at 1 fps and video B at 2 fps, 1 frame per temporal group.")
    rows = []
    for name, fps in (("A, 1 fps", 1.0), ("B, 2 fps", 2.0)):
        secs = [k / fps for k in range(4)]
        idx = vision_position_ids(0, 4, 1, 1, temporal_interval("index", fps, temporal_patch=1))[0].tolist()
        ab = vision_position_ids(0, 4, 1, 1, temporal_interval("abstime", fps, temporal_patch=1))[0].tolist()
        rows.append([name, secs, idx, ab, [timestamp_text(s) for s in secs]])
    rep.table(["video", "frame times (s)", "frame-index ids", "absolute-time ids (2 per s)", "text timestamps"],
              rows)
    rep.data["figure5"] = rows

    # RoPE score between a fixed query and key, rotated by temporal ids only (height = width = 0).
    head_dim, theta = 128, 1e6
    axes = frequency_axes(head_dim, (16, 24, 24), interleaved=False)
    inv = inv_frequencies(head_dim, theta)
    g = torch.Generator().manual_seed(0)
    q, k = torch.randn(head_dim, generator=g), torch.randn(head_dim, generator=g)

    def score(t_q, t_k):
        pos = torch.tensor([[t_q, t_k], [0, 0], [0, 0]])
        ang = mrope_angles(pos, axes, inv)
        return float(rotate(q[None], ang[:1]) @ rotate(k[None], ang[1:]).T)

    rows = []
    for mode in ("index", "abstime"):
        ia, ib = temporal_interval(mode, 1.0, 1), temporal_interval(mode, 2.0, 1)
        rows.append([mode, f"{score(3 * ia, 0):.4f}", f"{score(3 * ib, 0):.4f}", f"{score(6 * ib, 0):.4f}"])
    rep.print("\nRoPE score q.k between frame 0 and a later frame (random q, k; head_dim 128, theta 1e6):")
    rep.table(["ids", "A: frame 3 (3.0 s)", "B: frame 3 (1.5 s)", "B: frame 6 (3.0 s)"], rows)
    rep.print("Frame-index ids: A's 3.0 s and B's 1.5 s get the same score. Absolute time: equal seconds, equal score.")
    rep.data["rope_scores"] = rows

    rep.print("\nWhich rotary frequencies each axis owns (wavelength in ids, head_dim 128, theta 1e6):")
    rows = []
    for name, sec, inter in (("contiguous [16,24,24] (Qwen2-VL)", (16, 24, 24), False),
                             ("interleaved [24,20,20] (Qwen3-VL)", (24, 20, 20), True)):
        wl = axis_wavelengths(frequency_axes(head_dim, sec, inter), inv)
        rows.append([name] + [f"{lo:.1f} to {hi:,.0f}" for lo, hi in wl.values()])
        rep.data.setdefault("wavelengths", {})[name] = wl
    rep.table(["layout", "time", "height", "width"], rows)
    rep.print("Contiguous: time gets only the fastest frequencies, so its longest wavelength is short and large\n"
              "temporal gaps alias. Interleaved: every axis owns slow and fast frequencies.")

    rep.print("\nTemporal ids for one hour of video at 2 fps, depth-2 groups (1 group per second):")
    rows = []
    for mode in ("index", "abstime"):
        top = vision_position_ids(0, 3600, 1, 1, temporal_interval(mode, 2.0, 2))[0].max().item()
        rows.append([mode, f"{top:,}"])
    rows.append(["text timestamps", "ids stay compact; the time is written as text, e.g. " + timestamp_text(3599, "hms")])
    rep.table(["ids", "largest temporal id"], rows)

    vis = torch.tensor([False] * 20 + [True] * 2048 + [False] * 20)
    rows = [[d, f"{float(v2pe_positions(vis, d)[-1]):.0f}"] for d in (1.0, 0.25, 1 / 16)]
    rep.print("\nV2PE: final position of 20 text + 2,048 visual + 20 text tokens:")
    rep.table(["delta per visual token", "last position"], rows)


def part_b(rep: Report, steps: int, seeds: int) -> None:
    rep.print("\n== Part B: predict an event's duration in seconds ==")
    rep.print("Videos of 8-24 s, one event of 1-6 s. Loss: mean squared error. Metric: mean absolute error (s).")
    rep.print(f"{steps} steps, batch 64, {seeds} seeds. Predicting the mean duration gives an error of about 1.25 s.\n")
    settings = (("mixed fps", (1, 2, 4), ((1, 2, 4),)),
                ("train 2 fps", (2,), ((2,), (1,), (4,))))
    rows = []
    for setting, train_fps, tests in settings:
        for mode in ("none", "index", "abstime", "text"):
            errs = {str(t): [] for t in tests}
            for seed in range(seeds):
                torch.manual_seed(seed)
                g = torch.Generator().manual_seed(seed)
                model = TinyVideoTransformer(64, {"dur": 1}, time_mode=mode)

                def batch_fn(step):
                    f, t, m, d = event_batch(64, train_fps, g=g)
                    return (f, t, m), d

                train(model, batch_fn, steps, lr=2e-3,
                      loss_fn=lambda out, d: torch.nn.functional.mse_loss(out["dur"][:, 0], d))
                model.eval()
                for t_fps in tests:
                    f, t, m, d = event_batch(2000, t_fps, g=torch.Generator().manual_seed(99))
                    with torch.no_grad():
                        pred = model(f, t, m)["dur"][:, 0]
                    errs[str(t_fps)].append(float((pred - d).abs().mean()))
            for t_fps, e in errs.items():
                rows.append([setting, mode, t_fps, f"{sum(e) / len(e):.2f}", f"{min(e):.2f}-{max(e):.2f}"])
                rep.data.setdefault("duration", []).append(
                    {"setting": setting, "mode": mode, "test_fps": t_fps, "mae": e})
    rep.table(["training", "time signal", "test fps", "mean abs error (s)", "range over seeds"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--skip-training", action="store_true")
    args = ap.parse_args()
    rep = Report("05_telling_when")
    part_a(rep)
    if not args.skip_training:
        part_b(rep, args.steps, args.seeds)
    rep.save()


if __name__ == "__main__":
    main()
