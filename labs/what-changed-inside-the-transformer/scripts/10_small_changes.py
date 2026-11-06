"""Small changes that aren't small: the logits tensor, chunked cross-entropy, weight tying, attention sinks.

Post section: "Small changes that aren't small". The GPU peak-memory measurement of chunked cross-entropy is in
benchmarks/gpu_bench.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from organs.arith import GIB, logits_bytes
from organs.report import Report
from organs.small import chunked_cross_entropy, softmax_with_sink


def main() -> None:
    rep = Report("10_small_changes")
    rep("1. The logits tensor for cross-entropy, fp32:")
    rows = []
    for name, v in [("GPT-2", 50257), ("Llama 3", 128256), ("Qwen3", 151936), ("Gemma 3", 262144)]:
        rows.append([name, f"{v:,}", f"{logits_bytes(8, 8192, v) / GIB:.1f}"])
    rep.table(["vocab", "V", "GiB for B = 8, S = 8192"], rows)
    rep("")

    rep("2. Chunked cross-entropy gives the same loss and gradients as the full one (N = 4096, d = 256, V = 32,000):")
    g = torch.Generator().manual_seed(0)
    h = torch.randn(4096, 256, generator=g, requires_grad=True)
    w = (torch.randn(32000, 256, generator=g) * 0.05).requires_grad_(True)
    y = torch.randint(0, 32000, (4096,), generator=g)
    full = F.cross_entropy(h @ w.T, y)
    gh_full, gw_full = torch.autograd.grad(full, (h, w))
    rows = []
    for chunk in (4096, 1024, 256):
        loss = chunked_cross_entropy(h, w, y, chunk)
        gh, gw = torch.autograd.grad(loss, (h, w))
        rows.append([chunk, f"{loss.item():.6f}", f"{(loss - full).abs().item():.1e}",
                     f"{max((gh - gh_full).abs().max().item(), (gw - gw_full).abs().max().item()):.1e}",
                     f"{chunk * 32000 * 4 / 2**20:.0f} MiB"])
    rep.table(["chunk", "loss", "|loss - full|", "max |grad diff|", "largest logits block"], rows)
    rep(f"   full loss {full.item():.6f}; the full logits block is {4096 * 32000 * 4 / 2**20:.0f} MiB.")
    rep("")

    rep("3. Weight tying saves V d parameters. How much that is depends on the model:")
    rows = []
    for name, v, d, total in [("Qwen3-0.6B (tied)", 151936, 1024, 0.6e9), ("Gemma 3 1B (tied)", 262144, 1152, 1.0e9),
                              ("Llama 3 8B", 128256, 4096, 8.03e9), ("DeepSeek-V3", 129280, 7168, 671e9)]:
        rows.append([name, f"{v * d / 1e6:.0f}M", f"{v * d / total:.1%}"])
    rep.table(["model", "V d", "share of reported total"], rows)
    rep("   Reported totals are the model names' sizes, rounded; treat the shares as approximate.")
    rep("")

    rep("4. Attention sinks. One query, 8 keys. Token 0 is the head's dumping ground (logit 6, value 0);")
    rep("   the other 7 keys are irrelevant (logits near 0, random values).")
    g = torch.Generator().manual_seed(0)
    z = torch.cat([torch.tensor([6.0]), 0.1 * torch.randn(7, generator=g)])
    v = torch.cat([torch.zeros(1, 16), torch.randn(7, 16, generator=g)])
    rows = []
    for name, sink in (("plain softmax", None), ("softmax + sink logit 6", 6.0)):
        if sink is None:
            p_all, p_evicted = z.softmax(-1), z[1:].softmax(-1)
            zz = z
        else:  # with a sink, the head does not need token 0 at all: give it an ordinary logit
            zz = z.clone()
            zz[0] = 0.0
            p_all, p_evicted = softmax_with_sink(zz, sink), softmax_with_sink(zz[1:], sink)
        out_all, out_evicted = p_all @ v, p_evicted @ v[1:]
        rows.append([name, f"{p_all.sum():.3f}", f"{out_all.norm():.3f}", f"{out_evicted.norm():.3f}"])
    rep.table(["", "weights sum", "||output||", "||output|| after evicting token 0"], rows)
    rep("   Without a sink, evicting the first token (streaming) removes the dumping ground and the 'say nothing'")
    rep("   output becomes an average of irrelevant values. With a learned sink logit, mass leaves the sequence")
    rep("   and eviction changes nothing important.")
    print(f"saved {rep.save()}")


if __name__ == "__main__":
    main()
