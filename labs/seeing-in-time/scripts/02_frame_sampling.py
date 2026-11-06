"""Choosing which frames to look at, on the post's one-minute ball-and-cup clip (post Figure 3).

1. Uniform N against fixed fps: frames, tokens, events seen, roll direction, and the same rule on an hour.
2. Apollo-style clip sampling keeps the gap inside each clip fixed whatever the video length.
3. Keyframes: drop frames that look like the last kept one (the LongVU idea, on pixels here).
4. Query-aware selection: top-k relevance collapses into one scene; adding coverage spreads it out.
"""

from __future__ import annotations

import torch

from _common import Report
from seeingtime.clip import (EVENTS, apollo_clip_times, coverage, fps_times, keyframes, relevance_with_coverage,
                             topk_relevance, uniform_times)
from seeingtime.synth import render_times

TAU = 256


def hour_readout(rule: str, k) -> str:
    if rule == "uniform":
        return f"{k * TAU:,} tokens, one frame every {3600 / k / 60:.1f} min"
    return f"{round(3600 * k) * TAU:,} tokens"


def main():
    rep = Report("02_frame_sampling")
    rep.print("Events: " + ", ".join(f"{e.name} {e.start:g}-{e.end:g} s" for e in EVENTS))
    rows = []
    for rule, settings in (("uniform", (8, 16, 32, 64)), ("fps", (0.25, 0.5, 1, 2))):
        for k in settings:
            ts = uniform_times(k) if rule == "uniform" else fps_times(k)
            c = coverage(ts)
            gap = 60 / k if rule == "uniform" else 1 / k
            rows.append([rule, k, c["frames"], f"{gap:.2f}", f"{c['frames'] * TAU:,}", f"{c['events_seen']} / 5",
                         "yes" if c["direction_recoverable"] else "no", c["roll_frames"], hour_readout(rule, k)])
            rep.data.setdefault("figure3", []).append({"rule": rule, "k": k, **c})
    rep.print(f"\nFigure 3 readouts, {TAU} tokens per frame:")
    rep.table(["rule", "N or fps", "frames", "gap (s)", "tokens", "events seen", "roll direction", "frames in roll",
               "same rule on 1 hour"], rows)
    missed = [e.name for e in EVENTS if e.name not in coverage(uniform_times(8))["seen"]]
    rep.print(f"8 uniform frames miss: {', '.join(missed)}")

    rep.print("\nApollo-style clips (8 frames per clip at 2 fps, at most 32 frames):")
    rows = []
    for dur in (30, 600, 3600):
        ts = apollo_clip_times(dur, 2.0, 8, 32)
        gaps = sorted({round(b - a, 3) for a, b in zip(ts, ts[1:]) if b - a <= 0.5 + 1e-9})
        rows.append([f"{dur} s", len(ts), gaps, f"{dur / 32:.2f}"])
    rep.table(["video", "frames", "gap inside clips (s)", "uniform-32 gap (s)"], rows)

    rep.print("\nKeyframes on the clip rendered at 2 fps (keep a frame if cosine similarity to the last kept < t):")
    times = fps_times(2)
    frames = render_times(times)
    rows = []
    for thr in (0.9999, 0.999, 0.995):
        kept = keyframes(frames, thr)
        kt = [times[i] for i in kept]
        c = coverage(kt)
        rows.append([thr, f"{len(kept)} / {len(times)}", f"{c['events_seen']} / 5", c["roll_frames"]])
        rep.data.setdefault("keyframes", []).append({"threshold": thr, "kept": len(kept), **c})
    rep.table(["threshold", "frames kept", "events seen", "frames in roll"], rows)
    rep.print("Static stretches collapse to one frame; frames inside events survive because the pixels change.")

    rep.print("\nQuery-aware selection, question about the cup, 8 frames from 120 (relevance = 1 inside cup events,")
    rep.print("0.3 while the person is visible, 0.05 elsewhere, plus small noise):")
    g = torch.Generator().manual_seed(0)
    rel = torch.tensor([1.0 if 40 <= t <= 53 else 0.3 if 38 <= t < 55 else 0.05 for t in times])
    rel = rel + 0.01 * torch.rand(len(times), generator=g)
    rows = []
    for name, idx in (("top-k relevance", topk_relevance(rel, 8)),
                      ("relevance + coverage (4 bins)", relevance_with_coverage(rel, 8, 4))):
        kt = [times[i] for i in idx]
        c = coverage(kt)
        rows.append([name, f"{min(kt):.2f}-{max(kt):.2f}", f"{c['events_seen']} / 5", ", ".join(c["seen"])])
    rep.table(["selector", "time span (s)", "events seen", "which"], rows)
    rep.save()


if __name__ == "__main__":
    main()
