"""Part VIII: instability that lives in the number format. Limits, rounding stagnation, master weights,
underflow of real gradients, and a dynamic loss scaler.

    python labs/why-transformers-do-not-explode/scripts/08_number_formats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from stablelab.common import Report, seed_all  # noqa: E402
from stablelab.model import CharData, Config, TinyTransformer, load_corpus  # noqa: E402
from stablelab.numerics import DynamicLossScaler, format_table, fraction_lost  # noqa: E402

POST_PRECISION = {"FP32": 1.2e-7, "FP16": 9.8e-4, "BF16": 7.8e-3, "FP8 E4M3": 6.3e-2, "FP8 E5M2": 1.3e-1}


def regression(param_dtype: torch.dtype, master: bool, steps: int = 3000, lr: float = 1e-4) -> float:
    """Fit w to w* = 1 + 0.01 * noise from w = 1 with SGD. Updates are lr * grad, about 1e-6 per step."""
    g = torch.Generator().manual_seed(0)
    target = 1 + 0.01 * torch.randn(256, generator=g)
    w32 = torch.ones(256)
    w = w32.to(param_dtype)
    for _ in range(steps):
        grad = 2 * (w.float() - target)  # gradient computed in fp32 from the low-precision copy
        if master:
            w32 -= lr * grad
            w = w32.to(param_dtype)
        else:
            w = (w - (lr * grad).to(param_dtype)).to(param_dtype)
    return ((w.float() - target) ** 2).mean().item()


def main() -> None:
    seed_all(0)
    r = Report("08_number_formats")
    r("1. Format limits from torch.finfo, next to the post's 'relative precision' column.")
    rows = [[f["format"], f"1/{f['exp']}/{f['mant']}", f["max"], f["min_normal"], f["eps"], f["unit_roundoff"],
             POST_PRECISION[f["format"]]] for f in format_table()]
    r.table(["format", "S/E/M", "max", "min normal", "eps (spacing at 1)", "eps / 2", "post's value"], rows)
    r("The max and min-normal columns match the post. The precision column mixes two conventions: the FP32, FP16")
    r("and BF16 rows are eps (2^-23, 2^-10, 2^-7); the two FP8 rows are eps / 2, the unit roundoff.")
    r()

    r("2. Rounding stagnation: theta = 1, update 1e-5.")
    rows = []
    for name, dt in (("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)):
        one = torch.tensor(1.0, dtype=dt)
        rows.append([name, (one + torch.tensor(1e-5, dtype=dt)).item() == 1.0, torch.finfo(dt).eps])
    r.table(["format", "1 + 1e-5 == 1", "spacing above 1"], rows)
    r()

    r("3. Master weights. SGD on 256 weights near 1, lr 1e-4, 3,000 steps, per-step updates around 1e-6.")
    r(f"   Loss at the start (w = 1, target 1 + 0.01 * noise): {regression(torch.float32, False, steps=0):.3e}.")
    rows = []
    for name, dt, master in (("FP32 weights", torch.float32, False), ("BF16 weights only", torch.bfloat16, False),
                             ("FP16 weights only", torch.float16, False), ("BF16 + FP32 master", torch.bfloat16, True),
                             ("FP16 + FP32 master", torch.float16, True)):
        rows.append([name, regression(dt, master)])
    r.table(["setup", "final loss"], rows)
    r("Without a master copy every update rounds away and the loss stays at its starting value. With an FP32")
    r("master copy the 16-bit weights follow the FP32 run; the small gap is the 16-bit rounding of the weights.")
    r()

    data = CharData(load_corpus(), seq=64)
    torch.manual_seed(0)
    model = TinyTransformer(Config(vocab=data.vocab, layers=4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(0)
    for _ in range(200):  # train a little so the gradients are not just initialization gradients
        x, y = data.batch(32, gen)
        opt.zero_grad()
        model.loss(x, y).backward()
        opt.step()
    grads = torch.cat([p.grad.flatten() for p in model.parameters()])
    r(f"4. Real parameter gradients of a 4-layer, d = 128 model after 200 training steps ({grads.numel():,} values,")
    r(f"   median |g| = {grads.abs().median().item():.2e}), cast to a 16-bit format after multiplying by S.")
    rows = []
    for name, dt in (("FP16", torch.float16), ("BF16", torch.bfloat16)):
        for logs in ((0, 8, 16, 24) if dt == torch.float16 else (0,)):
            f = fraction_lost(grads, dt, 2.0**logs)
            rows.append([name, f"2^{logs}", f["zero"], f["subnormal"], f["inf"]])
    r.table(["format", "S", "flushed to zero", "subnormal", "inf"], rows)
    r("Only parameter gradients are measured here. Activation gradients inside the backward pass, which the post's")
    r("Micikevicius et al. citation measured, are not.")
    r()

    r("5. Dynamic loss scaling on the same run: 300 more steps; at each step check whether S * grad overflows FP16.")
    scaler = DynamicLossScaler(init_scale=2.0**32, growth_interval=100)
    history = []
    for step in range(300):
        x, y = data.batch(32, gen)
        opt.zero_grad()
        model.loss(x, y).backward()
        g = torch.cat([p.grad.flatten() for p in model.parameters()])
        found_inf = not torch.isfinite((g.double() * scaler.scale).to(torch.float16)).all().item()
        if scaler.update(found_inf):
            opt.step()
        history.append(scaler.scale)
    r(f"   S: start 2^32, after 10 steps 2^{torch.log2(torch.tensor(history[9])).item():.0f},"
      f" after 300 steps 2^{torch.log2(torch.tensor(history[-1])).item():.0f}; skipped steps {scaler.skipped}"
      f" ({scaler.skipped / 300:.1%}), most of them at the start while S comes down.")
    f = fraction_lost(g, torch.float16, scaler.scale)
    r(f"   At the final S, {f['zero']:.2%} of gradient values flush to zero in FP16 (against"
      f" {fraction_lost(g, torch.float16, 1.0)['zero']:.2%} at S = 1).")
    r.save()


if __name__ == "__main__":
    main()
