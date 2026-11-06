"""Write the real tensors that the post's figures draw to results/figure_data.json."""

import json
import math
from pathlib import Path

import _setup  # noqa: F401
import torch

from attnzoo.attention_tiled import tiled_attention
from attnzoo.core import scaled_dot_product_attention
from attnzoo.masks import causal_mask


def rounded(x: torch.Tensor, n: int = 3) -> list:
    return torch.round(x * 10**n).div(10**n).tolist()


torch.manual_seed(11)
data: dict = {}

# One head, six tokens: raw scores, masked scores and weights, for the tensor walk.
d = 8
q, k, v = torch.randn(1, 1, 6, d), torch.randn(1, 1, 6, d), torch.randn(1, 1, 6, d)
scores = (q @ k.transpose(-2, -1) / math.sqrt(d))[0, 0]
_, weights = scaled_dot_product_attention(q, k, v, mask=causal_mask(6))
data["walk"] = {"scores": rounded(scores, 2), "weights": rounded(weights[0, 0], 2)}

# One head, N = 12, tiles of 4: every tile the online softmax visits, for the tile stepper.
n, block, d = 12, 4, 8
q = torch.randn(1, 1, n, d)
k = torch.randn(1, 1, n, d) * 1.5
v = torch.randn(1, 1, n, d)
trace: list[dict] = []
out = tiled_attention(q, k, v, causal=True, block_q=block, block_k=block, trace=trace)
s = (q @ k.transpose(-2, -1) / math.sqrt(d))[0, 0]
s_masked = s.masked_fill(~causal_mask(n), float("-inf"))
data["tiles"] = {
    "n": n,
    "block": block,
    "scores": rounded(s, 2),
    "softmax": rounded(torch.softmax(s_masked, -1), 3),
    "trace": [
        {
            "q": t["q"],
            "k": t["k"],
            "skipped": t["skipped"],
            **({} if t["skipped"] else {"m": [round(x, 3) for x in t["m"]], "l": [round(x, 3) for x in t["l"]]}),
        }
        for t in trace
    ],
    "check_max_abs_diff": (out - scaled_dot_product_attention(q, k, v, mask=causal_mask(n))[0]).abs().max().item(),
}

path = Path(__file__).resolve().parent.parent / "results" / "figure_data.json"
path.write_text(json.dumps(data))
print(f"tiles traced: {len(trace)}, skipped: {sum(t['skipped'] for t in trace)}")
print(f"max |tiled - naive| = {data['tiles']['check_max_abs_diff']:.1e}")
print(f"wrote {path.name}, {path.stat().st_size} bytes")
