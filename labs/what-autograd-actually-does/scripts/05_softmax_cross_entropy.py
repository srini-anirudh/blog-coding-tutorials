"""The softmax and cross-entropy cancellation, and what happens without it.

The model is confidently wrong: logits are 0 for the true class and `margin` for one wrong class, over 8 classes.
Unfused: p = softmax(z), loss = -log p[true], and autograd differentiates through the division by p.
Fused: F.cross_entropy(z, true), whose backward rule returns p - y directly.
The exact gradient on the true logit is p_true - 1, which is close to -1 here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from autogradlab.report import Report  # noqa: E402

rep = Report("05_softmax_cross_entropy")
classes, true = 8, 0


def grads(margin: float, dtype, fused: bool):
    z = torch.zeros(1, classes, dtype=dtype)
    z[0, 1] = margin
    z.requires_grad_(True)
    target = torch.tensor([true])
    if fused:
        loss = F.cross_entropy(z, target)
        loss.backward()
        return loss.item(), z.grad[0, true].item(), float("nan")
    p = torch.softmax(z, -1)
    p.retain_grad()
    loss = -torch.log(p[0, true])
    loss.backward()
    return loss.item(), z.grad[0, true].item(), p.grad[0, true].item()


def fmt(v):
    return f"{v:.4g}"


rows = []
for dtype in [torch.float32, torch.bfloat16, torch.float16]:
    for margin in [5, 10, 20, 50, 90, 110]:
        exact_p = torch.softmax(torch.tensor([0.0] + [margin] + [0.0] * (classes - 2), dtype=torch.float64), -1)[0]
        lu, gu, dp = grads(margin, dtype, fused=False)
        lf, gf, _ = grads(margin, dtype, fused=True)
        rows.append([str(dtype).split(".")[1], margin, f"{exact_p.item() - 1:.4f}", fmt(dp), fmt(lu), fmt(gu), fmt(lf),
                     fmt(gf)])
rep("Gradient on the true logit. Exact value is p_true - 1.")
rep.table(["dtype", "margin", "exact", "unfused dL/dp", "unfused loss", "unfused dz", "fused loss", "fused dz"], rows)

rep("\nThe unfused path breaks before p_true reaches zero. At margin 90, p_true = e^-90 = 8e-40 is a subnormal")
rep("float32 number: the loss (90) is still right, but dL/dp = -1/p overflows to -inf, and the softmax")
rep("VJP turns into inf - inf = nan. At margin 110, p_true is 0 and the loss is inf as well.")
rep("float16 breaks first, at margin 20. float32 and bfloat16 share an exponent range and break together.")
rep("The fused rule returns p - y every time.")

# The interpretation: confident and right gives near zero, confident and wrong gives near one.
z = torch.tensor([[8.0, 0, 0, 0], [0, 8.0, 0, 0], [0, 0, 0, 0]], requires_grad=True)
F.cross_entropy(z, torch.tensor([0, 0, 0]), reduction="sum").backward()
rep("\nGradient of cross-entropy with respect to the logits is p - y (true class 0):")
for label, g in zip(["confident and right", "confident and wrong", "uniform"], z.grad):
    rep(f"  {label:<20} " + " ".join(f"{v:+.3f}" for v in g.tolist()))
rep.save()
