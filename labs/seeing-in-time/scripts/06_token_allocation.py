"""Fitting a video into the budget, measured on pixels of the ball-and-cup clip.

1. Equal against slow-fast allocation at 1,024 tokens (post Figure 6): detail on the cup's label,
   and how many of the five events have a frame.
2. A merging memory bank (MA-LMM style) of 16 slots fed 120 frames: which slots survive, and how much
   time each merged slot smears together.
3. Differential pruning (the VideoLLaMA 3 idea): how many patches change between frames.
"""

from __future__ import annotations

from _common import Report
from seeingtime.allocate import (MemoryBank, best_contrast_kept, changed_patches, equal_allocation,
                                 slowfast_allocation)
from seeingtime.clip import EVENTS, coverage, fps_times, uniform_times
from seeingtime.synth import cup_y, render_times


def label_halves(times):
    """Left (dark) and right (light) halves of the cup's label in frame i; they move when the cup is lifted."""
    def half(cols):
        def region(i):
            cy = int(cup_y(times[i]))
            return (slice(None), slice(cy - 6, cy - 2), cols)
        return region
    return half(slice(8, 12)), half(slice(12, 16))


def main():
    rep = Report("06_token_allocation")

    rep.print("== 1. Spending 1,024 tokens on the one-minute clip ==")
    rep.print("Frames are 48 x 72 pixels; a g_h x g_w grid means one token per average-pooled cell.")
    plans = [("16 frames", uniform_times(16), equal_allocation(16, (8, 8))),
             ("16 frames", uniform_times(16), slowfast_allocation(16, (4, 4), 4, (12, 16))),
             ("16 frames", uniform_times(16), slowfast_allocation(16, (4, 4), 8, (16, 24))),
             ("5 frames", uniform_times(5), equal_allocation(5, (12, 16))),
             ("64 frames", uniform_times(64), equal_allocation(64, (4, 4)))]
    rows = []
    for label, times, alloc in plans:
        frames = render_times(times)
        cov = coverage([times[i] for i in alloc.frames])
        rows.append([alloc.name, label, alloc.tokens, f"{cov['events_seen']} / 5", cov["roll_frames"],
                     f"{best_contrast_kept(frames, alloc, *label_halves(times)):.2f}"])
        rep.data.setdefault("allocation", []).append(rows[-1])
    rep.table(["allocation", "sampled", "tokens", "events seen", "frames in roll", "label contrast kept, best view"],
              rows)
    rep.print("Label contrast: 1 = the 4-pixel dark half of the cup label is fully readable, 0 = averaged away.\n"
              "Slow-fast keeps equal's temporal coverage and gets the 5-frame plan's detail. Fewer, finer slow\n"
              "views keep more detail at the same budget. The 64-frame plan sees all events but blurs every frame.")

    rep.print("\n== 2. Memory bank: 16 slots, 120 frames at 2 fps ==")
    times = fps_times(2)
    frames = render_times(times)
    bank = MemoryBank(16)
    for f, t in zip(frames, times):
        bank.add(f, t)
    rows = []
    for (s, e), n in zip(bank.spans, bank.counts):
        inside = [ev.name for ev in EVENTS if s <= ev.end and e >= ev.start]
        rows.append([f"{s:.2f}-{e:.2f}", n, f"{e - s:.2f}", ", ".join(inside) or "-"])
    rep.table(["slot span (s)", "frames merged", "span (s)", "events overlapped"], rows)
    widths = [e - s for s, e in bank.spans]
    rep.print(f"Longest slot covers {max(widths):.1f} s; shortest {min(widths):.1f} s. Static stretches merge into one slot,\n"
              "and inside a merged slot the exact time of anything is lost.")
    rep.data["memory_bank"] = rows

    rep.print("\n== 3. Differential pruning, 6-pixel patches, mean |pixel change| > 0.01 ==")
    keep = changed_patches(frames, 6, 0.01)
    per_frame = keep.flatten(1).float().mean(1)
    rows = []
    for ev in EVENTS:
        idx = [i for i, t in enumerate(times) if ev.start <= t <= ev.end and i > 0]
        rows.append([ev.name, f"{float(per_frame[idx].mean()):.3f}"])
    still = [i for i, t in enumerate(times[1:], 1) if not any(e.start - 0.5 <= t <= e.end + 0.5 for e in EVENTS)
             and not (37.5 <= t <= 38.5 or 54.5 <= t <= 55.5)]
    rows.append(["no event (static)", f"{float(per_frame[still].mean()):.3f}"])
    rep.table(["stretch", "share of patches kept"], rows)
    total = float(keep.float().mean())
    rep.print(f"Overall: {total:.3f} of {keep.numel():,} patches kept ({keep.sum().item():,} tokens instead of {keep.numel():,}).")
    rep.data["pruning"] = {"rows": rows, "kept_share": total}
    rep.save()


if __name__ == "__main__":
    main()
