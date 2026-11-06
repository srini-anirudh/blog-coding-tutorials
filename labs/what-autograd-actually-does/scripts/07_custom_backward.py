"""Writing your own backward with torch.autograd.Function, and testing it.

1. A linear layer with a hand-written backward passes gradcheck and the dot-product test.
2. The same layer with a transposed weight gradient (a plausible bug) fails both.
3. Rounding has zero gradient almost everywhere. A straight-through estimator replaces it with the identity, and
   a model can then learn through it. This changes the gradient on purpose.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.autograd import gradcheck  # noqa: E402

from autogradlab.custom import BuggyLinear, MyLinear, RoundSTE, dot_test  # noqa: E402
from autogradlab.report import Report  # noqa: E402

rep = Report("07_custom_backward")
torch.manual_seed(0)
x = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
w = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)  # square, so the transposed bug has the right shape
rows = []
for name, fn in [("MyLinear", MyLinear.apply), ("BuggyLinear (dW transposed)", BuggyLinear.apply)]:
    try:
        ok = gradcheck(fn, (x, w), raise_exception=False)
    except Exception:  # noqa: BLE001
        ok = False
    a, b = dot_test(fn, x.detach(), w.detach())
    rows.append([name, "pass" if ok else "FAIL", f"{a:.8f}", f"{b:.8f}", f"{abs(a - b):.1e}"])
rep("1-2. gradcheck and the dot-product test, float64:")
rep.table(["function", "gradcheck", "<u, J v>", "<J^T u, v>", "gap"], rows)
rep("A square weight is the dangerous case: the wrong gradient has the right shape and training would run.")

rep("\n3. Learn a weight w so that round(w * x) matches round(2.7 * x), 300 SGD steps from w = 0.5:")
xs = torch.linspace(-5, 5, 101)
target = torch.round(2.7 * xs)
for name, rnd in [("true gradient of round", torch.round), ("straight-through estimator", RoundSTE.apply)]:
    wt = torch.tensor(0.5, requires_grad=True)
    opt = torch.optim.SGD([wt], lr=0.01)
    for _ in range(300):
        opt.zero_grad()
        loss = (rnd(wt * xs) - target).pow(2).mean()
        loss.backward()
        opt.step()
    rep(f"  {name:<27} w = {wt.item():.3f}, loss = {loss.item():.3f}, last grad = {wt.grad.item():.3f}")
rep("The true gradient of round is zero, so w never moves. The straight-through gradient is not the derivative of")
rep("the loss written down, but it points the right way, and w ends near 2.7.")
rep.save()
