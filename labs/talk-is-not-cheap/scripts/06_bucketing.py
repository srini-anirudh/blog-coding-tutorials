"""Section 14: overlap and gradient bucketing, as a timeline model.

Two gradient models. "post" is the post's Figure 18: 7e9 parameters in 32 equal per-layer blobs. "tensors" is the
291 parameter tensors of a Llama-style 7B model (embeddings, norms, q/k/v/o, gate/up/down, head), which lets buckets
smaller than a layer behave differently from one-message-per-layer. Both: bf16 gradients, 8,192 tokens per GPU, 45% of
989 TFLOP/s, ring AllReduce cost. These are model times, not measurements.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commcost import overlap as O  # noqa: E402
from commcost.report import Report, table  # noqa: E402

R = Report("06_bucketing", note="timeline model, no measurement")
MB = 1e6
BUCKETS = [b * MB for b in (1, 2, 5, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 14000)]
ALPHA = 12e-6

models = {"post": O.post_model_tensors(), "tensors": O.llama7b_tensors()}
R(f"'tensors' model: {len(models['tensors'])} tensors, {sum(g.params for g in models['tensors']) / 1e9:.2f} B params")
seq = O.simulate(models["post"], 0, 64, ALPHA, 1 / 45e9, overlap=False)
R(f"No overlap, p = 64, 45 GB/s: backward {seq['compute_s'] * 1e3:.0f} ms, then one 14 GB AllReduce: exposed "
  f"{seq['exposed_s'] * 1e3:.0f} ms")
R()

data = {}
for name, tensors in models.items():
    R(f"Exposed tail in ms, '{name}' model, alpha = 12 us")
    cols, curves = [], {}
    for p, bw in [(8, 45e9), (64, 45e9), (64, 22.5e9), (1024, 45e9)]:
        rows = O.sweep(tensors, BUCKETS, p, ALPHA, 1 / bw)
        curves[f"p{p}_bw{bw / 1e9:g}"] = rows
        cols.append((f"p={p} {bw / 1e9:g}GB/s", rows))
    out = []
    for i, b in enumerate(BUCKETS):
        out.append([f"{b / MB:g} MB", rows[i]["messages"]] + [f"{c[1][i]['exposed_s'] * 1e3:.0f}" for c in cols])
    out.append(["best", ""] + [f"{O.best_bucket(c[1])['bucket_bytes'] / MB:g} MB" for c in cols])
    for line in table(out, ["bucket", "messages", *[c[0] for c in cols]]):
        R(line)
    tiny = O.simulate(tensors, 1, 64, ALPHA, 1 / 45e9)
    R(f"One message per tensor/blob at p = 64: {tiny['messages']} messages, alpha alone "
      f"{tiny['messages'] * 2 * 63 * ALPHA * 1e3:.0f} ms, exposed {tiny['exposed_s'] * 1e3:.0f} ms")
    R()
    data[name] = curves

R("Notes")
R("- At 45 GB/s and p >= 64 the ring needs ~0.6 s to move 14 GB, longer than the ~0.5 s backward pass, so")
R("  communication cannot fully hide in this setting whatever the bucket size. The tail is what is left.")
R("- In the 'post' model every bucket below one layer (437.5 MB) is the same as one message per layer, so its")
R("  curve is flat there. Splitting the layer into its real tensors makes small buckets pay alpha visibly.")
R("- Compare the p = 64 columns at 45 and 22.5 GB/s for the bandwidth claim, and p = 64 against p = 1024 for the")
R("  rank-count claim (nuance 14).")
R.save(data)
