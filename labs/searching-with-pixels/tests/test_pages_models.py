import torch

from pixsearch import pages as P
from pixsearch.models import BiEncoder, CrossEncoder, LateInteraction, maxsim, patchify


def _corpus(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = P.make_corpus(n, g)
    return c, P.make_queries(c, g), g


def test_twins_share_shapes_and_colours_but_not_bindings():
    c, _, _ = _corpus()
    for p in range(0, len(c), 2):
        a, b = c.items[p], c.items[p + 1]
        assert sorted(a[:, 0].tolist()) == sorted(b[:, 0].tolist())
        assert sorted(a[:, 1].tolist()) == sorted(b[:, 1].tolist())
        ca = set((a[:, 0] * P.N_COLORS + a[:, 1]).tolist())
        cb = set((b[:, 0] * P.N_COLORS + b[:, 1]).tolist())
        assert ca != cb


def test_binding_queries_exclude_the_twin_and_include_the_source():
    c, q, _ = _corpus()
    rel = P.relevant_sets(c, q)
    for i, src in enumerate(q.source.tolist()):
        assert src in rel[i]
        assert (src ^ 1) not in rel[i]


def test_clones_are_relevant():
    g = torch.Generator().manual_seed(3)
    c = P.make_corpus(60, g, kind="random", clone_frac=1.0)
    q = P.make_queries(c, g, binding=False)
    rel = P.relevant_sets(c, q)
    # Every second page is a clone of the one before it, so it answers that page's query.
    assert all((p + 1) in rel[p] for p in range(0, 58, 2))


def test_render_puts_glyph_in_its_cell():
    g = torch.Generator().manual_seed(0)
    items = torch.tensor([[[3, 2, 5]] + [[0, 0, 15]] * (P.ITEMS - 1)])
    img = P.render(items, g, noise=0.0)
    patches = patchify(img)[0].view(P.N_PATCHES, 3, P.PATCH, P.PATCH)
    assert patches[5].sum() > 0 and patches[1].sum() == 0
    # The blue channel dominates for colour 2 (blue).
    assert patches[5][2].sum() > patches[5][0].sum()


def test_maxsim_matches_a_loop_and_respects_mask():
    g = torch.Generator().manual_seed(1)
    q = torch.randn(3, 4, 8, generator=g)
    d = torch.randn(5, 6, 8, generator=g)
    s = maxsim(q, d)
    for i in range(3):
        for j in range(5):
            ref = sum(max(float(q[i, t] @ d[j, m]) for m in range(6)) for t in range(4))
            assert abs(s[i, j].item() - ref) < 1e-4
    mask = torch.ones(5, 6, dtype=torch.bool)
    mask[:, 3:] = False
    assert torch.allclose(maxsim(q, d, mask), maxsim(q, d[:, :3]))


def test_shapes_and_precomputation():
    c, q, _ = _corpus(8)
    bi, late, cross = BiEncoder(16), LateInteraction(16), CrossEncoder()
    assert bi.encode_pages(c.images).shape == (8, 16)
    assert late.encode_pages(c.images).shape == (8, P.N_PATCHES, 16)
    assert late.encode_queries(q.tokens).shape == (8, P.QUERY_LEN, 16)
    assert bi.score(bi.encode_queries(q.tokens), bi.encode_pages(c.images)).shape == (8, 8)
    # A cross-encoder's page representation depends on the query: change the query, the score changes.
    s1 = cross(q.tokens[:1], c.images[:1])
    s2 = cross(q.tokens[1:2], c.images[:1])
    assert not torch.allclose(s1, s2)


def test_late_page_vectors_do_not_depend_on_other_pages():
    c, _, _ = _corpus(6)
    late = LateInteraction(16).eval()
    with torch.no_grad():
        alone = late.encode_pages(c.images[:1])
        batch = late.encode_pages(c.images)[:1]
    assert torch.allclose(alone, batch, atol=1e-5)
