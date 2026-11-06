"""Where the position went: RoPE's relative-offset property, the base as a context-length dial, and NoPE.

Post sections: "The requirement, stated properly", "The base is a context-length dial", "Where this is going".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from organs.position import nope_attention, position_probe_r2, rope_logit, rope_wavelengths
from organs.report import Report


def main() -> None:
    rep = Report("03_position")
    torch.manual_seed(0)
    q, k = torch.randn(128), torch.randn(128)

    rep("1. <R_m q, R_n k> depends only on m - n. Same q, k (d_h = 128, base 1e4), different absolute positions:")
    rows = []
    for m, n in [(5, 2), (103, 100), (4003, 4000), (3, 0)]:
        rows.append([m, n, m - n, f"{rope_logit(q, k, m, n):+.6f}"])
    for m, n in [(10, 2), (2, 10)]:
        rows.append([m, n, m - n, f"{rope_logit(q, k, m, n):+.6f}"])
    rep.table(["m", "n", "m - n", "logit"], rows)
    rep("The first four rows share an offset of 3 and give the same logit up to float32 rounding.")
    rep("")

    rep("2. The base is a context-length dial. Wavelength of pair i = 2 pi b^(2i/d), d_h = 128.")
    rows = []
    for name, b in [("Llama 2", 1e4), ("Llama 3 / Olmo 3", 5e5), ("Qwen3, Gemma 3 global", 1e6)]:
        w = rope_wavelengths(128, b)
        rows.append([name, f"{b:.0e}", f"{w[0]:.2f}", f"{w[-1]:,.0f}", f"{2 * 3.141592653589793 * b:,.0f}",
                     int((w > 4096).sum()), int((w > 8192).sum())])
    rep.table(["model", "base b", "fastest pair", "slowest pair", "2 pi b", "pairs > 4k tokens", "pairs > 8k tokens"],
              rows)
    rep("The post rounds the slowest wavelength to 2 pi b. With d_h = 128 the exact slowest pair is 2 pi b^(126/128).")
    rep("A pair whose wavelength is longer than the training context never completes a turn during training;")
    rep("those are the dials that YaRN and NTK scaling stretch and leave the fast ones alone.")
    rep("")

    rep("3. NoPE: the causal mask alone carries position. One random attention head, no position signal,")
    rep("random byte embeddings (d = 64, T = 64, 64 sequences). A ridge probe reads log(1 + position) from the output.")
    d, t, b = 64, 64, 64
    g = torch.Generator().manual_seed(1)
    emb = torch.randn(256, d, generator=g, dtype=torch.float64)
    wq, wk, wv = (torch.randn(d, d, generator=g, dtype=torch.float64) / d**0.5 for _ in range(3))
    x = emb[torch.randint(0, 256, (b, t), generator=g)]
    target = torch.log1p(torch.arange(t, dtype=torch.float64)).repeat(b)
    perm = torch.randperm(t, generator=g)
    rows = []
    for causal in (False, True):
        out = nope_attention(x, wq, wk, wv, causal)
        shuffled = nope_attention(x[:, perm], wq, wk, wv, causal)
        equi = (shuffled - out[:, perm]).abs().max().item()
        feats = torch.cat([out, out.pow(2)], -1).reshape(b * t, -1)
        rows.append(["causal" if causal else "bidirectional", f"{equi:.2e}", f"{position_probe_r2(feats, target):.3f}"])
    rep.table(["mask", "max |f(Px) - P f(x)|", "probe R^2 for log(1+pos)"], rows)
    rep("Without a mask, shuffling the input shuffles the output exactly: position is invisible.")
    rep("With the causal mask, token m averages over m+1 values, so its output carries its position.")
    print(f"saved {rep.save()}")


if __name__ == "__main__":
    main()
