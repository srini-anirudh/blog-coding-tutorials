"""Run a schedule on numpy arrays: check the answer, count the bytes, and price it with the alpha-beta model."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .schedules import Schedule

# link(src, dst) -> (alpha seconds, beta seconds per byte)
LinkCost = Callable[[int, int], tuple[float, float]]


def uniform(alpha: float, beta: float) -> LinkCost:
    return lambda s, d: (alpha, beta)


def two_level(g: int, alpha_fast: float, beta_fast: float, alpha_slow: float, beta_slow: float) -> LinkCost:
    """Ranks are node-major with g per node. Same node: fast link. Different node: slow link."""
    return lambda s, d: (alpha_fast, beta_fast) if s // g == d // g else (alpha_slow, beta_slow)


@dataclass
class Stats:
    rounds: int = 0
    sent: dict = field(default_factory=lambda: defaultdict(int))  # bytes sent per rank
    received: dict = field(default_factory=lambda: defaultdict(int))
    cross_node_bytes: int = 0  # total bytes on links between nodes, if a node size was given
    cross_node_sent: dict = field(default_factory=lambda: defaultdict(int))  # per rank: what its NIC sends
    time: float = 0.0  # alpha-beta time, if a link cost was given
    round_times: list = field(default_factory=list)

    @property
    def max_sent(self) -> int:
        return max(self.sent.values(), default=0)

    @property
    def busiest_nic(self) -> int:
        return max(self.cross_node_sent.values(), default=0)

    @property
    def max_received(self) -> int:
        return max(self.received.values(), default=0)


def run(
    schedule: Schedule,
    inputs: list[np.ndarray],
    link: LinkCost | None = None,
    node_size: int | None = None,
    bytes_per_element: float | None = None,
) -> tuple[list[dict], Stats]:
    """Execute `schedule`. inputs[r] is rank r's flat vector. Returns each rank's buffers {"x", "y"} and stats.

    Pricing: each rank has one send port and one receive port per round. Messages through the same port in the same
    round are serialised, so a port's busy time is the sum of alpha + beta * bytes over its messages. A round lasts as
    long as its busiest port. A rank talking to itself is a local copy and costs nothing.
    bytes_per_element lets a short array stand in for a large one when only the price matters.
    """
    bufs = [{"x": np.array(x, copy=True), "y": np.zeros_like(x)} for x in inputs]
    itemsize = inputs[0].itemsize if bytes_per_element is None else bytes_per_element
    st = Stats()
    for rnd in schedule:
        if not rnd:
            continue
        st.rounds += 1
        payloads = [bufs[t.src]["x"][t.lo : t.hi].copy() for t in rnd]  # read the start-of-round state
        port = defaultdict(float)
        for t, data in zip(rnd, payloads):
            dst = bufs[t.dst][t.dst_buf]
            view = dst[t.at : t.at + t.count]
            if t.op == "add":
                view += data
            elif t.op == "copy":
                view[:] = data
            else:
                raise ValueError(t.op)
            if t.src == t.dst:
                continue
            nbytes = t.count * itemsize
            st.sent[t.src] += nbytes
            st.received[t.dst] += nbytes
            if node_size is not None and t.src // node_size != t.dst // node_size:
                st.cross_node_bytes += nbytes
                st.cross_node_sent[t.src] += nbytes
            if link is not None:
                a, b = link(t.src, t.dst)
                port[("out", t.src)] += a + b * nbytes
                port[("in", t.dst)] += a + b * nbytes
        if link is not None:
            rt = max(port.values(), default=0.0)
            st.round_times.append(rt)
            st.time += rt
    return bufs, st


def check_allreduce(p: int, n: int, make_schedule, seed: int = 0, dtype=np.float64) -> Stats:
    """Run an AllReduce schedule on random integers and assert every rank ends with the exact sum."""
    rng = np.random.default_rng(seed)
    xs = [rng.integers(-100, 100, size=n).astype(dtype) for _ in range(p)]
    out, st = run(make_schedule(p, n), xs)
    want = np.sum(xs, axis=0)
    for r in range(p):
        np.testing.assert_array_equal(out[r]["x"], want, err_msg=f"rank {r}")
    return st
