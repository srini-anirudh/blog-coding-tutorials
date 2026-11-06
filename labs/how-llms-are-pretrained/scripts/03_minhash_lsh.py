"""Figure 04: why MinHash works, and the LSH S-curve.

1. Pr[min-hash values match] = Jaccard. Checked on random set pairs with known J.
2. The estimate from k hashes has standard error about sqrt(J(1-J)/k), at most 1/(2 sqrt k).
3. With b bands of r rows, Pr[candidate] = 1 - (1 - J^r)^b. Checked at FineWeb's b = 14, r = 8.
4. Near-duplicate text: edits of a real article, their true J, and whether LSH flags them.
"""

import argparse

import _common
import numpy as np
from ptlab import corpora, minhash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=400, help="set pairs per J value")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = _common.Report("03_minhash_lsh")
    rng = np.random.default_rng(args.seed)
    js = [0.1, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
    b, r = 14, 8
    mh = minhash.MinHasher(b * r, args.seed)

    out(f"Random set pairs, 200 elements each, {args.pairs} pairs per J. k = {b * r} hashes.\n")
    out(f"{'true J':>7} {'mean est.':>10} {'std of est.':>12} {'sqrt(J(1-J)/k)':>15} "
        f"{'candidate rate':>15} {'1-(1-J^r)^b':>12}")
    sweep = []
    for j in js:
        ests, cand, true_j = [], 0, []
        for _ in range(args.pairs):
            a_set, b_set = minhash.sets_with_jaccard(j, 200, rng)
            true_j.append(minhash.jaccard(a_set, b_set))
            sa, sb = mh.signature(a_set), mh.signature(b_set)
            ests.append(float(np.mean(sa == sb)))
            cand += bool(minhash.lsh_candidates(np.stack([sa, sb]), b, r))
        tj = float(np.mean(true_j))
        row = dict(j=tj, est=float(np.mean(ests)), std=float(np.std(ests)),
                   se=float(np.sqrt(tj * (1 - tj) / (b * r))), cand=cand / args.pairs,
                   theory=float(minhash.candidate_prob(tj, b, r)))
        sweep.append(row)
        out(f"{row['j']:7.3f} {row['est']:10.3f} {row['std']:12.4f} {row['se']:15.4f} {row['cand']:15.3f} {row['theory']:12.3f}")
    out(f"\nS-curve inflection (1/b)^(1/r) for b={b}, r={r}: {minhash.threshold(b, r):.3f}")

    out("\nMoving the threshold: bands x rows, all with about 112 hashes")
    out(f"{'b':>3} {'r':>3} {'threshold':>10} " + "".join(f"{'P@J=' + str(j):>9}" for j in [0.5, 0.7, 0.8, 0.9]))
    for bb, rr in [(28, 4), (16, 7), (14, 8), (8, 14), (7, 16)]:
        out(f"{bb:>3} {rr:>3} {minhash.threshold(bb, rr):10.3f} "
            + "".join(f"{float(minhash.candidate_prob(j, bb, rr)):9.3f}" for j in [0.5, 0.7, 0.8, 0.9]))

    out("\nEdited copies of one real WikiText-2 article (5-word shingles), against the original")
    art = max(corpora.wikitext_docs("validation")[:50], key=lambda a: len(a.split()))
    words = art.split()
    out(f"article: {art.splitlines()[0]!r} ({len(words)} words)")
    out(f"{'edit':<42} {'true J':>7} {'MinHash J':>10} {'LSH flags it':>13}")
    edits = {}
    for frac in [0.01, 0.03, 0.05, 0.1, 0.2]:
        ws = list(words)
        for i in rng.choice(len(ws), int(frac * len(ws)), replace=False):
            ws[i] = "XXX"
        edits[f"replace {int(frac * 100)}% of words at random"] = " ".join(ws)
    edits["append one word repeated 40 times"] = art + " " + " ".join(["footer"] * 40)
    edits["keep the first half"] = " ".join(words[: len(words) // 2])
    s0 = minhash.shingles(art)
    g0 = mh.signature(s0)
    for name, text in edits.items():
        s1 = minhash.shingles(text)
        g1 = mh.signature(s1)
        flag = bool(minhash.lsh_candidates(np.stack([g0, g1]), b, r))
        out(f"{name:<42} {minhash.jaccard(s0, s1):7.3f} {float(np.mean(g0 == g1)):10.3f} {str(flag):>13}")
    out("One changed word removes up to 5 shingles, so 5% random word edits already cut J to about 0.6.")
    _common.save_json("03_minhash_lsh", {"b": b, "r": r, "sweep": sweep})
    out.save()


if __name__ == "__main__":
    main()
