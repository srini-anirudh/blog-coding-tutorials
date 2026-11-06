"""Section 09: Fourier features, the sigma knob, and SAM-style prompt encoding.

1. Fit a 128x128 image with a coordinate MLP, trained on every other pixel and tested on the rest.
   Raw (x, y) inputs, then random Fourier features with sigma = 1, 10 and 100.
2. The kernel gamma(a) . gamma(b) depends only on a - b, and for Gaussian B it is a Gaussian of width ~1/sigma.
3. A click at (0.413, 0.772) and a 64x64 grid encoded with one shared B: the similarity map peaks at the click.
   Two clicks inside one grid cell get different encodings; a lookup table would give them the same row.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from posenc import fourier  # noqa: E402
from posenc.report import Report, threads_from_env  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()
    threads_from_env()
    rep = Report("06_fourier")

    rep.print(f"\n1. Coordinate MLP, 3 x 256 ReLU, Adam lr 1e-3, {args.steps} full-batch steps, 256 Fourier features")
    img = fourier.test_image(128)
    rows = []
    for name, sigma in [("raw (x, y)", None), ("Fourier, sigma 1", 1.0), ("Fourier, sigma 10", 10.0),
                        ("Fourier, sigma 100", 100.0)]:
        B = None if sigma is None else fourier.gaussian_B(256, 2, sigma)
        tr, te, _ = fourier.fit_image(img, B, steps=args.steps)
        rows.append([name, round(tr, 2), round(te, 2)])
    rep.table(["input", "train PSNR (dB)", "held-out PSNR (dB)"], rows)
    rep.data["fit"] = rows

    rep.print("\n2. Empirical kernel vs exp(-2 pi^2 sigma^2 |delta|^2), sigma 1, 4096 features")
    B = fourier.gaussian_B(4096, 2, 1.0, seed=1)
    a = torch.tensor([[0.2, 0.3]])
    rows = []
    for dx in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        delta = torch.tensor([[dx, 0.0]])
        emp = float(fourier.kernel(a, a + delta, B))
        emp_shift = float(fourier.kernel(a + 0.37, a + 0.37 + delta, B))
        rows.append([dx, round(emp, 4), round(emp_shift, 4), round(float(fourier.expected_kernel(delta, 1.0)), 4)])
    rep.table(["|delta|", "kernel at a", "kernel at a + 0.37", "expected"], rows)

    rep.print("\n3. SAM-style: one B (128 features, sigma 1) for a 64x64 grid and for a click")
    B = fourier.gaussian_B(128, 2, 1.0, seed=2)
    cells = fourier.grid(64)
    click = torch.tensor([[0.772, 0.413]])  # (y, x): the post's click at x = 0.413, y = 0.772
    sim = fourier.kernel(click, cells, B)[0]
    best = cells[sim.argmax()]
    nearest = cells[((cells - click) ** 2).sum(-1).argmin()]
    cell = 1 / 64
    rep.print(f"   argmax cell centre {best.tolist()}, nearest cell centre {nearest.tolist()}, "
              f"distance between them {float((best - nearest).norm() / cell):.2f} cells")
    two = torch.tensor([[0.772, 0.4140], [0.772, 0.4210]])
    same_cell = bool(torch.equal(((two * 64).floor()[0]), (two * 64).floor()[1]))
    e = fourier.encode(two, B)
    cos = float(torch.nn.functional.cosine_similarity(e[:1], e[1:]))
    rep.print(f"   clicks x = 0.4140 and x = 0.4210: same grid cell {same_cell}; cosine of their encodings {cos:.4f}")
    rep.data["sam"] = {"argmax": best.tolist(), "nearest": nearest.tolist(), "same_cell": same_cell, "cosine": cos}
    rep.save()


if __name__ == "__main__":
    main()
