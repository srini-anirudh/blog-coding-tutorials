import numpy as np
import pytest

from commcost import alphabeta as ab
from commcost import schedules as S
from commcost import simulate as M


@pytest.mark.parametrize("name", ["naive", "tree", "ring"])
@pytest.mark.parametrize("p", [2, 3, 5, 6, 8])
@pytest.mark.parametrize("n", [1, 7, 40])
def test_any_p_allreduce_is_exact(name, p, n):
    if n < p and name == "ring":
        n = p  # a ring needs at least one element per chunk to be interesting; empty chunks are still legal
    M.check_allreduce(p, n, S.ALLREDUCE[name])


@pytest.mark.parametrize("name", ["recursive_doubling", "rabenseifner"])
@pytest.mark.parametrize("p", [2, 4, 8, 16])
def test_power_of_two_allreduce_is_exact(name, p):
    M.check_allreduce(p, 37, S.ALLREDUCE[name])


def test_power_of_two_required():
    with pytest.raises(ValueError):
        S.rabenseifner(6, 12)


@pytest.mark.parametrize("p", [2, 4, 8, 32])
def test_ring_and_rabenseifner_meet_the_lower_bound(p):
    n = p * 5
    for name in ("ring", "rabenseifner"):
        st = M.check_allreduce(p, n, S.ALLREDUCE[name])
        assert st.max_sent == ab.allreduce_lower_bound_bytes(n * 8, p)
        assert st.max_received == st.max_sent


def test_round_counts():
    p = 16
    assert M.check_allreduce(p, 32, S.naive).rounds == 2 * (p - 1)
    assert M.check_allreduce(p, 32, S.binomial_tree).rounds == 2 * 4
    assert M.check_allreduce(p, 32, S.ring).rounds == 2 * (p - 1)
    assert M.check_allreduce(p, 32, S.rabenseifner).rounds == 2 * 4


def test_naive_root_carries_everything():
    st = M.check_allreduce(8, 10, S.naive)
    assert st.sent[0] == 7 * 80 and st.received[0] == 7 * 80
    assert all(st.sent[r] == 80 for r in range(1, 8))


@pytest.mark.parametrize("nodes,g", [(2, 2), (2, 4), (4, 4), (3, 2)])
def test_hierarchical_is_exact_and_cuts_busiest_nic_by_about_g(nodes, g):
    p = nodes * g
    n = p * g * 3
    xs = [np.random.default_rng(r).integers(0, 9, n).astype(float) for r in range(p)]
    out, hier = M.run(S.hierarchical(nodes, g, n), xs, node_size=g)
    _, flat = M.run(S.ring(p, n), xs, node_size=g)
    for o in out:
        np.testing.assert_array_equal(o["x"], np.sum(xs, axis=0))
    want = g * (p - 1) / p * nodes / (nodes - 1)
    assert flat.busiest_nic / hier.busiest_nic == pytest.approx(want)


def test_alltoall_is_a_transpose():
    p, n = 4, 16
    xs = [np.arange(n, dtype=float) + 100 * r for r in range(p)]
    out, st = M.run(S.alltoall_pairwise(p, n), xs)
    for i in range(p):
        for j in range(p):
            np.testing.assert_array_equal(out[i]["y"][4 * j:4 * j + 4], xs[j][4 * i:4 * i + 4])
    assert st.max_sent == (p - 1) / p * n * 8


def test_chunks_cover_and_balance():
    c = S.chunks(10, 3)
    assert c == [(0, 4), (4, 7), (7, 10)]


def test_simulated_time_matches_closed_forms():
    a, b, p, n = 5e-6, 1 / 20e9, 8, 1 << 24
    elems = p * 2
    for name, f in [("naive", ab.t_naive), ("tree", ab.t_tree), ("ring", ab.t_ring), ("rabenseifner", ab.t_rhd)]:
        _, st = M.run(S.ALLREDUCE[name](p, elems), [np.zeros(elems)] * p, link=M.uniform(a, b),
                      bytes_per_element=n / elems)
        assert st.time == pytest.approx(f(n, p, a, b))
