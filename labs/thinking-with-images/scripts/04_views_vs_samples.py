"""The sampling trap: k text samples from one look share its blind spot; k different views do not.

At 448 px, the model samples k answers at temperature 1 from the same full-photo encoding. Separately it answers once
on each of k random crops (a quarter to half of the photo's width, each at 448 px). pass@k asks whether any of the
first k answers is right, which measures how correlated the errors are. vote@k is the majority answer.
About 4 min on one H100.
    python labs/thinking-with-images/scripts/04_views_vs_samples.py --n 50 --k 16
"""
from _common import command, out_path, parser, reader

from lookagain.probes import blind_prior, chance_pass, pass_and_vote, samples_vs_views
from lookagain.scene import make_scene
from lookagain.util import write


def main():
    ap = parser(__doc__, 50)
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()
    vlm = reader(args)
    scenes = [make_scene(3000 + i) for i in range(args.n)]
    blind = blind_prior(vlm)
    samples, views, weights = samples_vs_views(vlm, scenes, k=args.k)
    ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= args.k]
    s, v = pass_and_vote(scenes, samples, ks), pass_and_vote(scenes, views, ks, weights)
    rows = [f"n={args.n} scenes; samples: temperature 1.0 on full @448 (192 visual tokens, shared);",
            "views: one greedy answer per random crop @448 (192 visual tokens each), crop width 25-50% of the photo",
            "wvote: the vote where each view counts for the model's own P(Yes) that the view shows opening hours", "",
            f"{'k':>3} {'chance pass':>11} {'samples pass':>12} {'samples vote':>12} {'views pass':>10} "
            f"{'views vote':>10} {'views wvote':>11} {'view tokens':>11}"]
    for k in ks:
        rows.append(f"{k:>3} {chance_pass(k):>11.2f} {s[k]['pass']:>12.2f} {s[k]['vote']:>12.2f} {v[k]['pass']:>10.2f} "
                    f"{v[k]['vote']:>10.2f} {v[k]['wvote']:>11.2f} {192 * k:>11}")
    rows += ["", "chance pass: pass@k of a reader guessing uniformly among the 10 closing times. pass@k alone mostly",
             "measures how many different guesses are made, so read vote@k for whether the answer is converging."]
    distinct = sum(len(set(a)) for a in samples) / len(samples)
    distinct_v = sum(len(set(a)) for a in views) / len(views)
    prior = sum(a.count(blind) for a in samples) / (len(samples) * args.k)
    rows += ["", f"distinct answers per scene over k={args.k}: samples {distinct:.1f}, views {distinct_v:.1f}",
             f"share of sampled answers equal to the no-image answer ({blind}): {prior:.2f}"]
    write(out_path(args, "04_views_vs_samples"), command("04_views_vs_samples.py"), rows,
          {"samples": s, "views": v, "distinct_samples": distinct, "distinct_views": distinct_v,
           "no_image_answer": blind, "sampled_prior_share": prior})


if __name__ == "__main__":
    main()
