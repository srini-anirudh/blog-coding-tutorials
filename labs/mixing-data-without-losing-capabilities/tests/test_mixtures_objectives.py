import math

from mixlab import mixtures as mx
from mixlab import objectives as O


def test_proportional_sees_every_source_equally():
    sizes = mx.log_spaced_sizes()
    ep = mx.epochs(mx.proportional(sizes), sizes, 1_000_000)
    assert max(ep) - min(ep) < 1e-12


def test_temperature_endpoints_and_neff():
    sizes = [10, 100, 1000]
    assert mx.temperature(sizes, 1.0) == mx.proportional(sizes)
    assert all(abs(p - 1 / 3) < 1e-12 for p in mx.temperature(sizes, 0.0))
    assert abs(mx.n_eff(mx.equal(sizes)) - 3) < 1e-9
    assert mx.n_eff([1.0, 0.0]) == 1.0


def test_posts_sixty_epoch_example():
    assert abs(mx.epochs(mx.equal([700] * 24), [700] * 24, 1_000_000)[0] - 59.52) < 0.01


def test_cap_is_proportional_below_cap():
    p = mx.capped([10, 20, 1000], cap=50)
    assert abs(p[1] / p[0] - 2) < 1e-12 and abs(p[2] / p[1] - 2.5) < 1e-12


def test_token_weights_and_inverse():
    p, lens = [0.3, 0.3, 0.4], [16, 5, 5]
    lam = mx.token_weights(p, lens)
    assert abs(lam[0] - 4.8 / 8.3) < 1e-12
    back = mx.example_weights_for_token_target(p, lens)
    assert all(abs(a - b) < 1e-12 for a, b in zip(mx.token_weights(back, lens), p))


def test_simplex_grid():
    g = O.simplex_grid(3, 0.25)
    assert len(g) == 15 and all(abs(sum(w) - 1) < 1e-12 for w in g)


def test_linear_scalarisation_misses_the_dent():
    front = [(1.0, 0.0), (0.45, 0.45), (0.0, 1.0)]  # below the chord from (1, 0) to (0, 1)
    assert 1 not in O.linear_scalarisation_hits(front)
    assert max(front, key=min) == (0.45, 0.45)


def test_constrained_objective():
    f = O.constrained_obj(0, [0, 50, 50])
    assert f([90, 40, 60]) == -math.inf and f([90, 60, 60]) == 90


def test_surrogate_recovers_a_known_law():
    true = O.Surrogate(2)
    ws = O.dirichlet_probes(2, 20, seed=0)

    def law(w):
        return [10 * w[0] + 5 * w[1] + 3 * math.log(w[0] + O.EPS), 20 * w[1] + 2 * math.log(w[1] + O.EPS)]

    sur = true.fit(ws, [law(w) for w in ws])
    for w in O.dirichlet_probes(2, 5, seed=1):
        assert all(abs(a - b) < 0.05 for a, b in zip(sur.predict(w), law(w)))
