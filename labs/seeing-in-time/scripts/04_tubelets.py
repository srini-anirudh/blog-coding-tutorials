"""From 2D patches to space-time tubes (post Figure 4), with real modules.

1. Token counts: 4 frames, a 4 x 4 patch grid each -> 64 per-frame tokens -> 32 tubelet tokens -> 8 after 2 x 2 merge.
2. Qwen2-VL feeds an image as two identical frames: the 3D convolution then equals a 2D convolution whose
   kernel is the 3D kernel summed over time.
3. A tubelet sees motion inside the cube: forward and reversed frame pairs get different embeddings, while
   the average of per-frame embeddings is the same for both.
"""

from __future__ import annotations

import torch

from _common import Report
from seeingtime.synth import render_times
from seeingtime.tubelet import FramePatchEmbed, TubeletEmbed, merge_2x2, token_count


@torch.no_grad()
def main():
    rep = Report("04_tubelets")
    torch.manual_seed(0)
    p, dim = 14, 32
    video = torch.rand(4, 3, 4 * p, 4 * p)
    per_frame = FramePatchEmbed(p, dim)(video)
    tube = TubeletEmbed(p, 2, dim)
    tubes = tube(video)
    merged = merge_2x2(tubes)
    rows = [["per-frame patches", tuple(per_frame.shape[:-1]), token_count(per_frame)],
            ["tubelets, depth 2", tuple(tubes.shape[:-1]), token_count(tubes)],
            ["tubelets + 2 x 2 merge", tuple(merged.shape[:-1]), token_count(merged)]]
    rep.print("Figure 4: 4 frames of 56 x 56 pixels, 14-pixel patches")
    rep.table(["stage", "token grid (t, h, w)", "tokens"], rows)
    rep.data["figure4"] = rows

    image = torch.rand(3, 4 * p, 4 * p)
    a = tube.embed_image(image)[0]
    b = tube.as_frame_conv()(image[None])[0].permute(1, 2, 0)
    err = float((a - b).abs().max())
    rep.print(f"\nImage as two identical frames vs 2D conv with the time-summed kernel: max |diff| = {err:.2e}")
    rep.data["image_as_two_frames_max_diff"] = err

    # Two frames from the rolling ball (t = 12 s and 14 s), forward and reversed.
    frames = render_times([12.0, 14.0])[:, :, :42, :70]  # crop to a 3 x 5 patch grid of 14 px
    fwd, rev = frames, frames.flip(0)
    pf = FramePatchEmbed(p, dim)
    pf.proj.weight.data = tube.proj.weight.data[:, :, 0].clone()
    mean_fwd, mean_rev = pf(fwd).mean(0), pf(rev).mean(0)
    t_fwd, t_rev = tube(fwd)[0], tube(rev)[0]
    rows = [["mean of per-frame embeddings", f"{float((mean_fwd - mean_rev).norm()):.2e}"],
            ["depth-2 tubelet embedding", f"{float((t_fwd - t_rev).norm()):.3f}"],
            ["(scale: norm of the tubelet embedding)", f"{float(t_fwd.norm()):.3f}"]]
    rep.print("\nRolling ball, frames at 12 s and 14 s, played forward vs reversed: ||embedding(fwd) - embedding(rev)||")
    rep.table(["encoder", "difference"], rows)
    rep.print("Averaging per-frame tokens forgets which frame came first. A tubelet's kernel weights its two\n"
              "frames differently, so the same pair in the other order gives a different token.")
    rep.data["direction"] = rows
    rep.save()


if __name__ == "__main__":
    main()
