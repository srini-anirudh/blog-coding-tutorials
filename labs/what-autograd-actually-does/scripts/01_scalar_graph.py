"""The post's one-unit residual network, differentiated by a 100-line scalar engine.

z = w x + b, h = tanh z, r = h + x, L = (r - y)^2 at w = 1.5, x = 2, b = -1, y = 2.
Shows the forward values, every edge of the backward pass, how much of dL/dx arrives along the skip, forward
mode needing one sweep per input, and what a wrong visit order does.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from autogradlab.report import Report  # noqa: E402
from autogradlab.scalar import Value, backward, forward_mode_gradient, post_example  # noqa: E402

rep = Report("01_scalar_graph")
inputs = dict(w=1.5, x=2.0, b=-1.0, y=2.0)
g = post_example(**inputs)

rep("Forward pass (values stored on each node):")
for k in "zhrL":
    rep(f"  {k} = {g[k].value:.4f}")

rep("\nBackward pass, edge by edge (child -> parent: adjoint x local derivative):")
edges = backward(g["L"])
for child, parent, c in edges:
    child, parent = (n.replace("*", "-y") for n in (child, parent))  # r - y is built as r + (-1 * y)
    if parent != "-1.0":  # the constant -1 gets an adjoint too; it is not a leaf we care about
        rep(f"  {child:>5} -> {parent:<5} {c:+.4f}")

rep("\nGradients on the leaves (reverse mode, one sweep):")
for k in "wxby":
    rep(f"  dL/d{k} = {g[k].grad:+.4f}")

# The two routes into x: the skip edge r -> x and the linear edge w*x -> x.
skip = [c for child, parent, c in edges if child == "r" and parent == "x"][0]
total = g["x"].grad
rep(f"\nTwo arrows arrive at x. Skip edge: {skip:.4f}. Through tanh: {total - skip:.4f}. Sum: {total:.4f}.")
rep(f"Share along the skip: {skip / total:.1%}  (the post says 1.93 of 2.13)")

# Check against PyTorch.
t = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in inputs.items()}
L = ((t["w"] * t["x"] + t["b"]).tanh() + t["x"] - t["y"]) ** 2
L.backward()
err = max(abs(t[k].grad.item() - g[k].grad) for k in inputs)
rep(f"Largest difference from torch.autograd (float64): {err:.1e}")

fwd, sweeps = forward_mode_gradient(inputs)
rep(f"\nForward mode: {sweeps} sweeps for {len(inputs)} partial derivatives (one per input). Reverse mode: 1 sweep.")
rep("  " + ", ".join(f"dL/d{k} = {v:+.4f}" for k, v in fwd.items()))

# A node with two children: s feeds both p = 2 s and q = exp(s). Visit s before q has reported in, and s
# passes on a partial sum. No error is raised.
a = Value(0.5, name="a")
s = a.tanh()
s.name = "s"
p = s * 2.0
p.name = "p"
q = s.exp()
q.name = "q"
out = p + q
out.name = "out"
backward(out)
right = a.grad
backward(out, order=[out, p, s, q])
rep("\nFan-out with a wrong visit order (out = 2 tanh(a) + exp(tanh(a)), a = 0.5):")
rep(f"  reverse topological order: d out/da = {right:.4f}")
rep(f"  s visited before q:         d out/da = {a.grad:.4f}  (only the p route arrived)")
rep("  No error is raised. This is why engines count dependencies before running a node.")

rep("\nA chain of 50 exchange rates of 0.9 each:", f"{0.9 ** 50:.4f}")
rep.save()
