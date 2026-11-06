"""Topologies as graphs: channel counts, message counts, latency depth, token bill, blast radius.

Every topology is a set of agents plus directed message channels. A blackboard is modelled
as one extra node, the shared store, that every agent writes to and reads from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STORE = -1  # the node id of the shared store in a blackboard


@dataclass
class Topology:
    name: str
    n: int                                   # number of agents
    edges: list[tuple[int, int]]             # undirected channels between agents (or agent-store)
    flows: list[tuple[int, int]] = field(default_factory=list)   # directed messages in one round
    depth: int | str = 1                     # agent activations on the critical path of one round
    note: str = ""

    @property
    def channels(self) -> int:
        return len(self.edges)

    @property
    def msgs_per_round(self) -> int:
        return len(self.flows)

    def max_inbox(self) -> int:
        """Most messages any single agent reads in one round: the context it must hold."""
        inbox = {i: 0 for i in range(self.n)}
        for _, dst in self.flows:
            if dst != STORE:
                inbox[dst] += 1
        return max(inbox.values()) if inbox else 0


def solo() -> Topology:
    return Topology("solo", 1, [], [], 1, "drift, context rot")


def pipeline(n: int = 7) -> Topology:
    e = [(i, i + 1) for i in range(n - 1)]
    return Topology("pipeline", n, e, list(e), n, "cascading error")


def star(n: int = 7) -> Topology:
    e = [(0, i) for i in range(1, n)]
    flows = e + [(j, i) for i, j in e]              # delegate down, return up
    return Topology("star", n, e, flows, 3, "coordinator overload")


def hierarchy(n: int = 7, branching: int = 2) -> Topology:
    e = [((i - 1) // branching, i) for i in range(1, n)]
    flows = e + [(j, i) for i, j in e]
    height, size, level = 0, 1, 1
    while size < n:                                 # number of levels below the root
        level *= branching
        size += level
        height += 1
    return Topology("hierarchy", n, e, flows, 2 * height + 1, "split-brain design")


def mesh(n: int = 7) -> Topology:
    e = [(i, j) for i in range(n) for j in range(i + 1, n)]
    flows = e + [(j, i) for i, j in e]
    return Topology("full mesh", n, e, flows, 2, "echo chamber")


def blackboard(n: int = 7) -> Topology:
    e = [(i, STORE) for i in range(n)]
    flows = [(i, STORE) for i in range(n)] + [(STORE, i) for i in range(n)]   # write once, read once
    return Topology("blackboard", n, e, flows, "async", "write contention")


def all_topologies(n: int = 7) -> list[Topology]:
    return [solo(), pipeline(n), star(n), hierarchy(n), mesh(n), blackboard(n)]


# --- the token bill ---------------------------------------------------------------------

def comm_bill(n: int, rounds: int, tokens_per_msg: int, structured_tokens: int = 180) -> dict:
    """Tokens agents spend reading each other, the four rows of the post's Figure 5.

    all-to-all full traces: every agent reads the other n-1 agents' L tokens each round.
    all-to-all structured:  the same, but each message is a bounded ~180-token artifact.
    blackboard:             each agent writes one structured message and reads one per round.
    work:                   the tokens the agents produce for the task itself.
    """
    mesh_full = rounds * n * (n - 1) * tokens_per_msg
    mesh_struct = rounds * n * (n - 1) * structured_tokens
    board = rounds * n * 2 * structured_tokens
    work = n * rounds * tokens_per_msg
    return {"mesh_full": mesh_full, "mesh_structured": mesh_struct, "blackboard": board,
            "work": work, "useful_fraction_mesh": work / (work + mesh_full)}


# --- prompt infection along the graph ---------------------------------------------------

def spread(topo: Topology, start: int, rounds: int, beta: float, rng: np.random.Generator) -> int:
    """Agents compromised after `rounds` rounds when one agent starts compromised.

    In every round each message from a compromised sender carries the payload with
    probability beta (the receiver treats it as instruction). For a blackboard the store
    is a node: a compromised write poisons it, and a poisoned store infects readers.
    Returns the number of compromised agents (the store itself is not counted).
    """
    bad = {start}
    for _ in range(rounds):
        new = set()
        for src, dst in topo.flows:
            if src in bad and dst not in bad and rng.random() < beta:
                new.add(dst)
        bad |= new
    bad.discard(STORE)
    return len(bad)


def blast_radius(topo: Topology, rounds: int = 3, beta: float = 0.5, trials: int = 2000,
                 seed: int = 0) -> dict:
    """Mean compromised agents, starting from a uniformly random agent and from agent n-1.

    Agent n-1 is the last pipeline stage, a star worker, a hierarchy leaf: the low-privilege
    end of the graph.
    """
    rng = np.random.default_rng(seed)
    rand = [spread(topo, int(rng.integers(topo.n)), rounds, beta, rng) for _ in range(trials)]
    leaf = [spread(topo, topo.n - 1, rounds, beta, rng) for _ in range(trials)]
    return {"from_random_agent": float(np.mean(rand)), "from_leaf": float(np.mean(leaf)),
            "n": topo.n}
