"""Filters, MinHash/LSH and BPE: offline and fast."""

import numpy as np
import pytest
from ptlab import bpe, corpora, filters, minhash

PROSE = ("The river runs through the old town and the bridge that crosses it was built in the twelfth century. "
         "Many visitors come to see the stone arches, and the local museum has a room with drawings of the "
         "original design. The town council has worked with engineers to keep the bridge safe for walkers.\n") * 1
PROSE = PROSE + PROSE.replace("river", "canal").replace("twelfth", "tenth").replace("museum", "library")


CLEAN = ("The river runs through the old town and the bridge that crosses it was built in the twelfth century.\n"
         "Many visitors come to see the stone arches, and the local museum has a room with drawings of the design.\n"
         "The town council has worked with engineers to keep the bridge safe for people who walk to the market.\n"
         "In winter the water rises, so a small team checks the foundations after every storm that reaches the hills.")


def test_clean_prose_passes_rules():
    assert filters.quality_reasons(CLEAN) == []


def test_repeated_prose_fails_repetition_rule():
    assert "dup_5gram" in filters.gopher_reasons(PROSE)


def test_rules_catch_their_targets():
    assert "word_count" in filters.gopher_reasons("404 Not Found.")
    assert "stopwords" in filters.gopher_reasons(" ".join(["cheap", "buy", "deal", "sale"] * 30))
    assert "curly_bracket" in filters.c4_reasons(PROSE + "\nconfig = {'a': 1}")
    cut = "\n".join(" ".join(p.split()[:12]) + " ..." for p in [PROSE] * 6)
    assert "ellipsis_lines" in filters.gopher_reasons(cut)
    assert "mean_word_length" in filters.gopher_reasons("\n".join(corpora.math_docs(1)))


def test_exact_hash_ignores_whitespace_only():
    assert filters.exact_hash("a  b\nc") == filters.exact_hash("a b c")
    assert filters.exact_hash("a b c") != filters.exact_hash("a b d")


def test_contamination_needs_a_shared_13gram():
    ev = filters.ngram_set(PROSE)
    assert filters.contaminated("intro " + PROSE[:200], ev)
    assert not filters.contaminated("a completely different text with no overlap at all " * 3, ev)


def test_minhash_match_rate_is_jaccard():
    rng = np.random.default_rng(0)
    mh = minhash.MinHasher(256, seed=1)
    for j in [0.2, 0.5, 0.8]:
        a, b = minhash.sets_with_jaccard(j, 300, rng)
        assert abs(minhash.jaccard(a, b) - j) < 0.01
        est = np.mean(mh.signature(a) == mh.signature(b))
        assert abs(est - j) < 4 * np.sqrt(j * (1 - j) / 256)


def test_lsh_candidate_rate_matches_formula():
    rng = np.random.default_rng(0)
    mh = minhash.MinHasher(112, seed=0)
    hits = 0
    for _ in range(200):
        a, b = minhash.sets_with_jaccard(0.72, 100, rng)
        hits += bool(minhash.lsh_candidates(np.stack([mh.signature(a), mh.signature(b)]), 14, 8))
    j = 0.72
    assert abs(hits / 200 - minhash.candidate_prob(j, 14, 8)) < 0.12
    assert minhash.threshold(14, 8) == pytest.approx(0.719, abs=1e-3)


def test_near_dedup_clusters_copies():
    other = " ".join(f"word{i}" for i in range(300))
    edited = PROSE.replace("arches", "arch")
    kept, clusters = minhash.near_dedup([PROSE, other, edited, PROSE])
    assert kept == [0, 1]
    assert clusters == [[0, 2, 3]]


def test_bpe_round_trip_and_prefix_property():
    texts = [PROSE] * 3 + ["def f(x):\n    return x * 2\n"]
    big, small = bpe.train_bpe(texts, 80), bpe.train_bpe(texts, 30)
    assert big[:30] == small
    tok = bpe.BPETokenizer(big)
    s = PROSE + " ünïcödé 2024"
    assert tok.decode(tok.encode(s)) == s
    assert len(tok.encode(PROSE)) < len(PROSE.encode())
    assert tok.truncated(30).encode(PROSE) == bpe.BPETokenizer(small).encode(PROSE)


def test_fertility_cost_numbers():
    assert bpe.logits_bytes(8, 8192, 100_000) / 1e9 == pytest.approx(26.2, abs=0.1)
    assert bpe.training_flops(1e9, 1e12, 0.25) == pytest.approx(1.5e21)


def test_stdlib_code_domain_is_offline():
    tr, va = corpora.stdlib_code_docs("train"), corpora.stdlib_code_docs("val")
    assert len(tr) > 50 and len(va) > 5
    stream = corpora.as_bytes(va)
    assert (stream == corpora.EOD).sum() == len(va)


@pytest.mark.network
def test_wikitext_articles_load():
    docs = corpora.wikitext_docs("validation")
    assert len(docs) > 50
    assert " @-@ " not in docs[0]
