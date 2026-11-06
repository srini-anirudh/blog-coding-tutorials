"""Who can I see, and how far can information travel. Sections "Masks", "Local and sliding-window attention",
"Sparse attention" and "Attention over images". A few seconds on CPU."""

import _setup  # noqa: F401

from attnmoves.common import Report
from attnmoves.reach import (
    axial, block_diagonal, causal, farthest_reach, gradient_reach, grid_windows, hops_to_full, local_global, local_random,
    packing_leak, padding_and_causal, sliding_window, strided,
)

rep = Report("03_masks_and_reach")
rep("Part 1. Packed documents: change document 0, measure the change in documents 1 and 2 (one layer)")
for k, v in packing_leak([5, 7, 4]).items():
    rep(f"  {k:<26} max change in later documents = {v:.2e}")
rep(f"  padding AND causal: max |padded batch - each sequence alone| = {padding_and_causal([7, 3, 10], n=10):.2e}")

rep("\nPart 2. Stacked causal sliding windows, n = 64. Window w counts the query itself.")
rows = []
for w in (2, 4, 8):
    for depth in (1, 2, 4, 6):
        rows.append([w, depth, farthest_reach(64, w, depth), gradient_reach(64, w, depth), depth * (w - 1), depth * w])
rep.table(["w", "layers", "reach (mask powers)", "reach (autograd)", "L (w - 1)", "post: L w"], rows)
rep("The mask graph and the gradients of real stacked layers agree. Each layer adds w - 1 positions.")
rep("Mistral 7B: w = 4096, 32 layers. L w = 131,072. Counted exactly, L (w - 1) = 131,040. Not run at that size.")

rep("\nPart 3. Layers until every output can depend on every input (bidirectional, n = 256)")
n = 256
patterns = {
    "dense": lambda _: sliding_window(n, n, is_causal=False),
    "causal": lambda _: causal(n),
    "band w = 16": lambda _: sliding_window(n, 16, is_causal=False),
    "strided, block 16 + every 16th column": lambda _: strided(n, 16),
    "band w = 16 + 2 global tokens": lambda _: local_global(n, 16, 2),
    "band w = 16 + 3 random keys per query": lambda _: local_random(n, 16, 3),
    "block-diagonal, blocks of 16": lambda _: block_diagonal(n, 16),
}
rows = []
for name, p in patterns.items():
    m = p(0)
    rows.append([name, f"{m.float().mean().item():.3f}", hops_to_full(p) or "never"])
rep.table(["pattern", "density", "layers to full reach"], rows)

rep("\nPart 4. A 16 x 16 patch grid (256 tokens)")
row_m, col_m = axial(16, 16)
grid = {
    "full": lambda _: grid_windows(16, 16, 16),
    "4 x 4 windows, no shift": lambda _: grid_windows(16, 16, 4),
    "4 x 4 windows, shift by 2 every other layer": lambda i: grid_windows(16, 16, 4, shift=2 * (i % 2)),
    "axial: rows, then columns": lambda i: row_m if i % 2 == 0 else col_m,
}
rows = []
for name, p in grid.items():
    rows.append([name, f"{p(0).float().mean().item():.3f}", hops_to_full(p, max_layers=32) or "never in 32 layers"])
rep.table(["pattern", "density", "layers to full reach"], rows)
rep.save()
