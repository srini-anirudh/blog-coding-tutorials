"""The device mesh as arithmetic: rank -> (pp, dp, cp, tp) with TP varying fastest, and the groups that follow.

Ranks that are adjacent in this order sit on the same node, so the fastest-varying axis gets the fastest links.
"""

from __future__ import annotations

from itertools import product

AXES = ("tp", "cp", "dp", "pp")  # fastest-varying first


def coords(rank: int, tp: int, cp: int, dp: int, pp: int) -> dict[str, int]:
    return {"tp": rank % tp, "cp": (rank // tp) % cp, "dp": (rank // (tp * cp)) % dp, "pp": rank // (tp * cp * dp)}


def rank_of(c: dict[str, int], tp: int, cp: int, dp: int, pp: int) -> int:
    return c["tp"] + tp * (c["cp"] + cp * (c["dp"] + dp * c["pp"]))


def groups(axis: str, tp: int, cp: int, dp: int, pp: int) -> list[list[int]]:
    """Every process group along one axis: ranks that share all other coordinates."""
    sizes = {"tp": tp, "cp": cp, "dp": dp, "pp": pp}
    others = [a for a in AXES if a != axis]
    out = []
    for fixed in product(*(range(sizes[a]) for a in others)):
        c = dict(zip(others, fixed))
        out.append([rank_of({**c, axis: i}, tp, cp, dp, pp) for i in range(sizes[axis])])
    return sorted(out)


def crosses_node(group: list[int], gpus_per_node: int = 8) -> bool:
    return len({r // gpus_per_node for r in group}) > 1


def placement_report(tp: int, cp: int, dp: int, pp: int, gpus_per_node: int = 8) -> dict[str, dict]:
    """For each axis: group size, how many groups there are, and how many of them leave a node."""
    rep = {}
    for axis in AXES:
        gs = groups(axis, tp, cp, dp, pp)
        rep[axis] = {"size": len(gs[0]), "groups": len(gs),
                     "cross_node": sum(crosses_node(g, gpus_per_node) for g in gs), "example": gs[0]}
    return rep
