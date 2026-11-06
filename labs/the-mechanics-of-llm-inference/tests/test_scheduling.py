from inferlab.scheduling import POST_REQUESTS, continuous_batching, static_batching


def test_static_matches_post():
    r = static_batching(POST_REQUESTS, 8)
    assert round(100 * r.utilisation, 1) == 29.5


def test_continuous_finishes_everything_and_wins_when_requests_wait():
    for slots in (2, 4):
        s, c = static_batching(POST_REQUESTS, slots), continuous_batching(POST_REQUESTS, slots)
        assert len(c.finish) == len(POST_REQUESTS) and c.useful == sum(r[2] for r in POST_REQUESTS)
        assert c.utilisation > s.utilisation
        assert c.mean_latency(POST_REQUESTS) < s.mean_latency(POST_REQUESTS)
