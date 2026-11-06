"""Figure 03 and "order changes the outcome": a small crawl through quality rules, dedup and decontamination.

The crawl is built from real text (WikiText-2 articles, standard-library Python files) plus
constructed page types: mirrors, scraper copies, exact copies, spam, error pages, tables,
arithmetic and pages that quote the evaluation set. The page types and their proportions are
made up for the demo. The filters and dedup are the real algorithms.
"""

import argparse
import random
from collections import Counter, defaultdict

import _common
from ptlab import corpora, filters, minhash

NAV = "Home | News | Sport | Contact"
FOOT = "Copyright 2024 the mirror site. All rights reserved. Privacy policy and cookie settings apply to this page."


def mirror(text: str, rng: random.Random) -> str:
    """A near-duplicate: site chrome added, a few words changed."""
    ws = text.split(" ")
    for _ in range(max(1, len(ws) // 200)):
        ws[rng.randrange(len(ws))] = rng.choice(["also", "very", "then", "now"])
    return NAV + "\n" + " ".join(ws) + "\n" + FOOT


def scraper_copy(text: str) -> str:
    """An excerpt scraper: every paragraph cut to 85% of its words and ended with '...'."""
    out = []
    for p in text.split("\n"):
        ws = p.split()
        out.append(" ".join(ws[: max(1, int(len(ws) * 0.85))]) + " ..." if len(ws) > 8 else p)
    return "\n".join(out)


def build_crawl(seed: int = 0):
    rng = random.Random(seed)
    arts = [a for a in corpora.wikitext_docs("train") if 300 <= len(a.split()) <= 4000]
    rng.shuffle(arts)
    test_arts = [a for a in corpora.wikitext_docs("test") if len(a.split()) > 300]
    eval_passages = [" ".join(a.split("\n")[1].split()[:60]) for a in test_arts[:20]]
    code = corpora.stdlib_code_docs("train")
    pages: list[tuple[str, str]] = []  # (kind, text)
    clean = arts[:300]
    pages += [("article", a) for a in clean]
    pages += [("mirror", mirror(a, rng)) for a in clean[:80]]
    pages += [("exact copy", a) for a in clean[80:130]]
    pages += [("code", c) for c in rng.sample(code, 40)]
    pages += [("arithmetic", m) for m in corpora.math_docs(20, seed)]
    spam_words = ["cheap", "best", "price", "buy", "online", "discount", "free", "shipping", "deal", "sale"]
    pages += [("spam", " ".join(rng.choice(spam_words) for _ in range(rng.randint(80, 300)))) for _ in range(30)]
    pages += [("error page", "404 Not Found. The page you requested could not be found. Return to the home page.")
              for _ in range(30)]
    pages += [("table", "\n".join(",".join(str(rng.randint(0, 999)) for _ in range(6)) for _ in range(40)))
              for _ in range(20)]
    pages += [("quotes eval set", a.split("\n")[0] + "\n" + p + "\n" + "\n".join(a.split("\n")[1:]))
              for a, p in zip(arts[300:320], eval_passages)]
    rng.shuffle(pages)
    # Scraper copies of 60 clean articles are crawled first, before the original.
    first = [("scraper copy", scraper_copy(a)) for a in clean[130:190]]
    pages = first + pages
    eval_ngrams = set().union(*(filters.ngram_set(p) for p in eval_passages))
    return pages, eval_ngrams, set(clean[130:190])


def stage_quality(pages, log=None):
    kept = []
    for kind, text in pages:
        why = filters.quality_reasons(text)
        if why:
            if log is not None:
                log[kind][why[0]] += 1
        else:
            kept.append((kind, text))
    return kept


def stage_exact(pages):
    seen, kept = set(), []
    for kind, text in pages:
        h = filters.exact_hash(text)
        if h not in seen:
            seen.add(h)
            kept.append((kind, text))
    return kept


def stage_near(pages):
    keep, _ = minhash.near_dedup([t for _, t in pages], b=14, r=8)
    return [pages[i] for i in keep]


def stage_decon(pages, eval_ngrams):
    return [(k, t) for k, t in pages if not filters.contaminated(t, eval_ngrams)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = _common.Report("02_pipeline_funnel")
    pages, eval_ngrams, scraped_originals = build_crawl(args.seed)
    kinds = sorted(Counter(k for k, _ in pages))
    reasons = defaultdict(Counter)
    stages = [("crawl", pages)]
    p = stage_quality(pages, reasons)
    stages.append(("quality rules", p))
    p = stage_exact(p)
    stages.append(("exact dedup", p))
    p = stage_near(p)
    stages.append(("MinHash dedup", p))
    p = stage_decon(p, eval_ngrams)
    stages.append(("decontamination", p))

    out("Survivors after each stage, by page type (order: quality -> exact dedup -> MinHash -> decontamination)\n")
    out(f"{'page type':<16}" + "".join(f"{s:>17}" for s, _ in stages))
    for k in kinds:
        out(f"{k:<16}" + "".join(f"{sum(kk == k for kk, _ in ps):>17}" for _, ps in stages))
    out(f"{'total':<16}" + "".join(f"{len(ps):>17}" for _, ps in stages))
    b0 = sum(len(t.encode()) for _, t in pages)
    b1 = sum(len(t.encode()) for _, t in p)
    out("Dedup keeps whichever copy was crawled first, so some originals go and their mirror or copy stays.")
    out(f"\nbytes kept: {b1:,} of {b0:,} ({100 * b1 / b0:.1f}%). This toy crawl is far cleaner than a real one,")
    out("where published pipelines keep roughly 1-10% of raw bytes.")

    out("\nFirst rule each page failed, by page type")
    for k in kinds:
        if reasons[k]:
            out(f"  {k:<16} " + ", ".join(f"{r} {c}" for r, c in reasons[k].most_common()))

    # Order changes the outcome: dedup first keeps the first-crawled copy, then the filters judge that copy.
    a =stage_near(stage_exact(stage_quality(pages)))
    b = stage_quality(stage_near(stage_exact(pages)))
    out("\nOrder changes the outcome. 60 articles were crawled first as an excerpt scraper's copy")
    out("(paragraphs cut short, ending in '...'), then later as the original.")
    out(f"  filter then dedup: {len(a)} pages kept; originals of the 60 scraped articles kept: "
        f"{sum(1 for k, t in a if k == 'article' and t in scraped_originals)}")
    out(f"  dedup then filter: {len(b)} pages kept; originals of the 60 scraped articles kept: "
        f"{sum(1 for k, t in b if k == 'article' and t in scraped_originals)}")
    out("  Dedup first keeps the scraper copy (it came first). The ellipsis rule then deletes it, and the article is gone.")
    out.save()


if __name__ == "__main__":
    main()
