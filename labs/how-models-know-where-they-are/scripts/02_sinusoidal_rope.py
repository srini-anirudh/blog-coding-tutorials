"""Sections 03, 05, 06 and 13: the sinusoidal code, RoPE, and the numeric details that bite.

Sinusoidal: PE(p + k) = R_k PE(p), and PE(p) . PE(p + k) depends on k only.
RoPE: scores depend on m - n only; norms are preserved; the two pairing conventions differ by a
permutation, and mixing them up corrupts scores more the further apart two tokens are.
Decay: the RoFormer envelope, and a key that ignores it.
Bands: how many planes never finish one turn inside the context, for the bases models ship with.
Precision: angle error of a bf16 table at long positions.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posenc import rope, sinusoidal  # noqa: E402
from posenc.report import Report  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    rep = Report("02_sinusoidal_rope")
    d = 64

    rep.print("\nSinusoidal, d=64, positions 0..999")
    p = torch.arange(1000)
    pe = sinusoidal.sinusoidal(p, d)
    rows = []
    for k in (1, 7, 100):
        shifted = sinusoidal.sinusoidal(p + k, d)
        gap = float((shifted - pe @ sinusoidal.shift_matrix(k, d).T).abs().max())
        dots = (pe * shifted).sum(-1)
        rows.append([k, gap, float(dots.max() - dots.min())])
    rep.table(["k", "max |PE(p+k) - R_k PE(p)|", "spread of PE(p).PE(p+k) over p"], rows)
    rep.print("PE(0) . PE(k) as k grows (the maximum is d/2 = 32 at k = 0):")
    ks = [0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512]
    dots = [float(sinusoidal.sinusoidal(torch.tensor([0]), d) @ sinusoidal.sinusoidal(torch.tensor([k]), d).T) for k in ks]
    rep.table(["k", *ks], [["dot", *[round(x, 2) for x in dots]]])
    rep.data["sinusoidal_dot"] = dict(zip(ks, dots))

    rep.print("\nRoPE, head_dim=64, base 10000, float32")
    f = rope.inv_freq(d)
    q, k = torch.randn(256, d), torch.randn(256, d)

    def score(m, n, apply=rope.apply_half):
        return (apply(q[:, None], torch.tensor([m]), f) * apply(k[:, None], torch.tensor([n]), f)).sum(-1).squeeze(-1)

    base_s = score(10, 3)
    shift_gap = max(float((score(10 + s, 3 + s) - base_s).abs().max()) for s in (1, 100, 5000))
    norm_gap = float((rope.apply_half(q[:, None], torch.tensor([12345]), f).norm(dim=-1).squeeze() - q.norm(dim=-1)).abs().max())
    P = rope.half_to_interleaved(d)
    pos = torch.arange(50)
    x = torch.randn(3, 50, d)
    perm_gap = float((rope.apply_interleaved(x[..., P], pos, f) - rope.apply_half(x, pos, f)[..., P]).abs().max())
    rows = [
        ["score(m+s, n+s) - score(m, n), s in {1, 100, 5000}", shift_gap],
        ["| |R q| - |q| | at position 12345", norm_gap],
        ["interleaved(x[P]) - half(x)[P]", perm_gap],
    ]
    rep.table(["check", "max abs difference"], rows)
    rep.data["rope_checks"] = {r[0]: r[1] for r in rows}

    rep.print("\nConvention mismatch: q, k laid out for half-split, rotated with the interleaved kernel.")
    rep.print("Relative error = mean |wrong - right| / mean |right| over 256 random (q, k) pairs.")
    rows = []
    for dist in (0, 1, 2, 4, 8, 16, 32, 64, 256, 1024):
        right = score(dist, 0)
        wrong = score(dist, 0, rope.apply_interleaved)
        rows.append([dist, float((wrong - right).abs().mean() / right.abs().mean())])
    rep.table(["distance", "relative error"], rows)
    rep.data["mismatch_error"] = {r[0]: r[1] for r in rows}

    rep.print("\nDecay envelope (RoFormer), head_dim=128, base 10000, relative to distance 0")
    dist = torch.tensor([0, 1, 4, 16, 64, 256, 1024, 4096, 16384])
    env = rope.decay_envelope(128, 10_000, dist)
    f128 = rope.inv_freq(128)
    ones = torch.ones(1, 1, 128)
    smooth = [float((rope.apply_half(ones, torch.tensor([int(r)]), f128) * ones).sum()) / 128 for r in dist]
    # Random q, k: the score is a sum of terms with unrelated signs, so it has no reason to follow the envelope.
    qr, kr = torch.randn(512, 1, 128), torch.randn(512, 1, 128)
    q0 = rope.apply_half(qr, torch.tensor([0]), f128)
    rand = [float((q0 * rope.apply_half(kr, torch.tensor([int(r)]), f128)).sum(-1).abs().mean()) for r in dist]
    rand = [r / rand[0] for r in rand]
    rep.table(["distance", "envelope / envelope(0)", "q = k = ones: score / 128", "random q, k: mean |score| / at 0"],
              [[int(r), float(e / env[0]), s, a] for r, e, s, a in zip(dist, env, smooth, rand)])
    rep.data["decay"] = {"distance": dist.tolist(), "envelope": (env / env[0]).tolist(), "smooth": smooth, "random": rand}

    rep.print("\nPlanes (of 64 per 128-dim head) that never finish one turn inside the context")
    rows = []
    for base in (10_000, 500_000, 1_000_000):
        rows.append([f"{base:,}", *[rope.planes_without_full_turn(128, base, c) for c in (4096, 32768, 131072, 1_048_576)],
                     f"{float(rope.wavelengths(128, base)[-1]):.3g}"])
    rep.table(["base", "4k", "32k", "128k", "1M", "slowest wavelength"], rows)

    rep.print("\nRoPE table precision: wrapped |angle error| in radians, head_dim=128, base 500000")
    f5 = rope.inv_freq(128, 500_000, torch.float64)
    rows = []
    for position in (1_000, 32_768, 500_000):
        pos = torch.tensor([position])
        for dt in (torch.float32, torch.bfloat16):
            err = rope.wrapped_angle_error(pos, f5, dt)[0]
            rows.append([f"{position:,}", str(dt).replace("torch.", ""), float(err[0]), float(err.median()), float(err.max())])
    rep.table(["position", "table dtype", "fastest plane", "median plane", "worst plane"], rows)
    rep.data["precision"] = rows
    rep.save()


if __name__ == "__main__":
    main()
