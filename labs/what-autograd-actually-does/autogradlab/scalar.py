"""A scalar autograd engine in two flavours: reverse mode (`Value`) and forward mode (`Dual`).

`Value` records a graph while it computes. Each node stores its value, its parents and the local derivative
with respect to each parent. `backward` seeds the output with 1 and walks the graph in reverse topological
order, adding `adjoint * local` into each parent. That `+=` is the multivariable chain rule.

`Dual` carries (value, tangent) pairs. One forward sweep gives the derivative with respect to one input.
"""

from __future__ import annotations

import math


class Value:
    def __init__(self, value: float, parents: tuple = (), locals_: tuple = (), op: str = "", name: str = ""):
        self.value = float(value)
        self.parents = parents  # the nodes this one was computed from
        self.locals = locals_  # d(self)/d(parent) for each parent, evaluated at the forward values
        self.op = op
        self.name = name
        self.grad = 0.0  # the adjoint dL/d(self), filled by backward()

    def __repr__(self) -> str:
        return f"Value({self.name or self.op}={self.value:.4f}, grad={self.grad:.4f})"

    @staticmethod
    def lift(x) -> "Value":
        return x if isinstance(x, Value) else Value(x, name=f"{x}")

    def __add__(self, other):
        other = Value.lift(other)
        return Value(self.value + other.value, (self, other), (1.0, 1.0), "+")

    __radd__ = __add__

    def __mul__(self, other):
        other = Value.lift(other)
        return Value(self.value * other.value, (self, other), (other.value, self.value), "*")

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-Value.lift(other))

    def __pow__(self, k: float):
        return Value(self.value**k, (self,), (k * self.value ** (k - 1),), f"**{k}")

    def tanh(self):
        t = math.tanh(self.value)
        return Value(t, (self,), (1.0 - t * t,), "tanh")

    def exp(self):
        e = math.exp(self.value)
        return Value(e, (self,), (e,), "exp")


def topological_order(root: Value) -> list[Value]:
    """Every node reachable from `root`, each after all of its parents."""
    order, seen = [], set()

    def visit(v: Value) -> None:
        if id(v) in seen:
            return
        seen.add(id(v))
        for p in v.parents:
            visit(p)
        order.append(v)

    visit(root)
    return order


def backward(root: Value, order: list[Value] | None = None) -> list[tuple[str, str, float]]:
    """Reverse-mode sweep. Returns the edge contributions (child, parent, adjoint * local) in visit order.

    `order` defaults to the reverse topological order. Pass a different one to see what a wrong order does.
    """
    nodes = topological_order(root)
    for v in nodes:
        v.grad = 0.0
    root.grad = 1.0
    contributions = []
    for v in order if order is not None else reversed(nodes):
        for parent, local in zip(v.parents, v.locals):
            parent.grad += v.grad * local  # += is the multivariable chain rule
            contributions.append((v.name or v.op, parent.name or parent.op, v.grad * local))
    return contributions


def post_example(w=1.5, x=2.0, b=-1.0, y=2.0) -> dict[str, Value]:
    """The post's graph: z = w x + b, h = tanh z, r = h + x, L = (r - y)^2."""
    w, x, b, y = (Value(v, name=n) for v, n in ((w, "w"), (x, "x"), (b, "b"), (y, "y")))
    names = {}

    def named(v: Value, n: str) -> Value:
        v.name = n
        names[n] = v
        return v

    wx = named(w * x, "wx")
    z = named(wx + b, "z")
    h = named(z.tanh(), "h")
    r = named(h + x, "r")
    d = named(r - y, "r-y")
    L = named(d**2, "L")
    return dict(w=w, x=x, b=b, y=y, **names)


class Dual:
    """Forward mode: a value and its tangent d(value)/d(seed input)."""

    def __init__(self, value: float, tangent: float = 0.0):
        self.value, self.tangent = float(value), float(tangent)

    @staticmethod
    def lift(x) -> "Dual":
        return x if isinstance(x, Dual) else Dual(x)

    def __add__(self, o):
        o = Dual.lift(o)
        return Dual(self.value + o.value, self.tangent + o.tangent)

    __radd__ = __add__

    def __mul__(self, o):
        o = Dual.lift(o)
        return Dual(self.value * o.value, self.tangent * o.value + self.value * o.tangent)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-Dual.lift(o))

    def __pow__(self, k: float):
        return Dual(self.value**k, k * self.value ** (k - 1) * self.tangent)

    def tanh(self):
        t = math.tanh(self.value)
        return Dual(t, (1.0 - t * t) * self.tangent)


def post_loss(w, x, b, y):
    """The same function, written once, usable with `Value`, `Dual` or floats."""
    return ((w * x + b).tanh() + x - y) ** 2


def forward_mode_gradient(inputs: dict[str, float]) -> tuple[dict[str, float], int]:
    """All partial derivatives by forward mode: one sweep per input. Returns (gradient, sweeps)."""
    grad = {}
    for seed in inputs:
        duals = {k: Dual(v, 1.0 if k == seed else 0.0) for k, v in inputs.items()}
        grad[seed] = post_loss(**duals).tangent
    return grad, len(inputs)
