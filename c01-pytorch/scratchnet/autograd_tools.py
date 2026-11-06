"""Walk the graph autograd records during the forward pass.

Every non-leaf tensor has a `grad_fn`: the backward node that produced it.
Each node lists its inputs in `next_functions`. Leaf tensors that require
grad end in an `AccumulateGrad` node, which is what writes into `.grad`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Node:
    depth: int
    name: str
    leaf_shape: tuple[int, ...] | None  # set only for AccumulateGrad nodes


def walk_graph(t: torch.Tensor) -> list[Node]:
    """Depth-first list of the backward nodes reachable from `t.grad_fn`.

    A node reached twice (for example a weight used in two places) is listed
    once, at the first depth it is seen.
    """
    if t.grad_fn is None:
        return []
    seen: set[int] = set()
    nodes: list[Node] = []

    def visit(fn, depth: int) -> None:
        if fn is None or id(fn) in seen:
            return
        seen.add(id(fn))
        leaf = getattr(fn, "variable", None)  # only AccumulateGrad has .variable
        nodes.append(Node(depth, type(fn).__name__, tuple(leaf.shape) if leaf is not None else None))
        for next_fn, _ in fn.next_functions:
            visit(next_fn, depth + 1)

    visit(t.grad_fn, 0)
    return nodes


def format_graph(t: torch.Tensor) -> str:
    """The graph as an indented tree, one node per line."""
    lines = []
    for node in walk_graph(t):
        suffix = f"  -> .grad of leaf {list(node.leaf_shape)}" if node.leaf_shape is not None else ""
        lines.append("  " * node.depth + node.name + suffix)
    return "\n".join(lines)
