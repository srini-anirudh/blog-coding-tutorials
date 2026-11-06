"""Tree edit distance (Zhang and Shasha, 1989) and TEDS, the table metric from PubTabNet.

TEDS(Ta, Tb) = 1 - EditDist(Ta, Tb) / max(|Ta|, |Tb|). Insert and delete cost 1. Substituting a cell costs 1 if its
spans differ, else the normalized edit distance of the two texts. TEDS-S ignores the text.

The tree is table -> tr -> td, as in Figure 6 of the post. PubTabNet's reference code also keeps thead and tbody
nodes and tokenizes cell text into characters; this version keeps the tree the figure draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import normalized_edit_distance
from .tables import Table


@dataclass
class Node:
    tag: str
    text: str = ""
    span: tuple[int, int] = (1, 1)
    children: list["Node"] = field(default_factory=list)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)


def table_tree(table: Table, structure_only: bool = False) -> Node:
    root = Node("table")
    for row in table.rows:
        tr = Node("tr")
        for cell in row:
            tr.children.append(Node("td", "" if structure_only else cell.text, (cell.colspan, cell.rowspan)))
        root.children.append(tr)
    return root


def rename_cost(a: Node, b: Node) -> float:
    if a.tag != b.tag:
        return 1.0
    if a.tag != "td":
        return 0.0
    if a.span != b.span:
        return 1.0
    return normalized_edit_distance(a.text, b.text)


def _postorder(root: Node):
    """Nodes in postorder, and for each the postorder index of its leftmost leaf."""
    nodes, lml = [], []

    def walk(n: Node) -> int:
        first = None
        for c in n.children:
            leaf = walk(c)
            first = leaf if first is None else first
        nodes.append(n)
        lml.append(len(nodes) - 1 if first is None else first)
        return lml[-1]

    walk(root)
    return nodes, lml


def tree_edit_distance(t1: Node, t2: Node, cost=rename_cost) -> float:
    """Zhang-Shasha: the cheapest insertions, deletions and relabels that turn t1 into t2."""
    n1, l1 = _postorder(t1)
    n2, l2 = _postorder(t2)
    # Key roots: the highest node for each distinct leftmost leaf.
    kr1 = sorted({max(i for i in range(len(n1)) if l1[i] == leaf) for leaf in set(l1)})
    kr2 = sorted({max(j for j in range(len(n2)) if l2[j] == leaf) for leaf in set(l2)})
    td = [[0.0] * len(n2) for _ in range(len(n1))]
    for i in kr1:
        for j in kr2:
            a0, b0 = l1[i], l2[j]
            fd = [[0.0] * (j - b0 + 2) for _ in range(i - a0 + 2)]
            for x in range(1, i - a0 + 2):
                fd[x][0] = fd[x - 1][0] + 1
            for y in range(1, j - b0 + 2):
                fd[0][y] = fd[0][y - 1] + 1
            for x in range(1, i - a0 + 2):
                for y in range(1, j - b0 + 2):
                    ii, jj = a0 + x - 1, b0 + y - 1
                    if l1[ii] == a0 and l2[jj] == b0:
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1, fd[x - 1][y - 1] + cost(n1[ii], n2[jj]))
                        td[ii][jj] = fd[x][y]
                    else:
                        p, q = l1[ii] - a0, l2[jj] - b0
                        fd[x][y] = min(fd[x - 1][y] + 1, fd[x][y - 1] + 1, fd[p][q] + td[ii][jj])
    return td[-1][-1]


def teds(pred: Table, gold: Table, structure_only: bool = False) -> float:
    a, b = table_tree(pred, structure_only), table_tree(gold, structure_only)
    return 1.0 - tree_edit_distance(a, b) / max(a.size(), b.size())
