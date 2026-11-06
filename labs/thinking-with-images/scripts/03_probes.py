"""Finding the stage that failed: the blind probe, the oracle swap, caption-then-reason and the crop probe.

Each probe changes one thing and is read as a change in accuracy over a set of scenes. At two encoder widths, every
wrong direct answer is then sorted into the post's three failures:
  1  never extracted: the transcript lacks the closing time, and the crop fixes the answer;
  2  extracted but not used: the model's own transcript contains the closing time, yet its direct answer is wrong;
  3  reasoning failed: given the facts as text (the oracle swap), the model still answers wrong.
About 4 min on one H100.
    python labs/thinking-with-images/scripts/03_probes.py --n 100
"""
from _common import command, out_path, parser, reader

from lookagain.probes import answer_region_view, ask_views, blind_prior, caption_then_reason, oracle_swap, score
from lookagain.scene import make_scene
from lookagain.util import write
from lookagain.views import full_view


def main():
    args = parser(__doc__, 100).parse_args()
    vlm = reader(args)
    scenes = [make_scene(2000 + i) for i in range(args.n)]
    prior = blind_prior(vlm)
    rows, data = [], {"blind_answer": prior}

    def line(name, preds, extra=""):
        r = score(scenes, preds, prior)
        data[name] = r
        rows.append(f"{name:<36} {r['acc']:>5.2f} {r['said_prior']:>6.2f} {r['said_bus_time']:>5.2f} "
                    f"{r['no_time']:>5.2f}  {extra}")
        return r

    rows.append(f"n={args.n} scenes, greedy decoding. With no image the model answers {prior!r}.")
    rows.append(f"{'probe':<36} {'acc':>5} {'prior':>6} {'bus':>5} {'none':>5}")
    line("blind: no image", ask_views(vlm, scenes, None))
    oracle = oracle_swap(vlm, scenes)
    line("oracle swap: facts as text, no image", oracle)
    clean = [make_scene(2000 + i, distractor=False) for i in range(args.n)]
    r = score(clean, oracle_swap(vlm, clean), prior)
    data["oracle swap, no bus sign"] = r
    rows.append(f"{'oracle swap, bus sign removed':<36} {r['acc']:>5.2f} {r['said_prior']:>6.2f} "
                f"{r['said_bus_time']:>5.2f} {r['no_time']:>5.2f}  the facts omit the bus sign too")
    crop = ask_views(vlm, scenes, [answer_region_view(s) for s in scenes])
    line("crop probe: plaque cell @448", crop)
    breakdown = {}
    for w in (448, 1344):
        direct = ask_views(vlm, scenes, full_view(w))
        line(f"direct: full @{w}", direct)
        ctr, has = caption_then_reason(vlm, scenes, full_view(w))
        line(f"caption, then reason: full @{w}", ctr, f"closing time in transcript: {sum(has) / len(has):.2f}")
        counts = {"correct": 0, "1 never extracted": 0, "2 extracted, not used": 0, "3 reasoning": 0, "unclear": 0}
        for s, d, h, c, o in zip(scenes, direct, has, crop, oracle):
            if d == s.close_time:
                counts["correct"] += 1
            elif o != s.close_time:
                counts["3 reasoning"] += 1
            elif h:
                counts["2 extracted, not used"] += 1
            elif c == s.close_time:
                counts["1 never extracted"] += 1
            else:
                counts["unclear"] += 1
        breakdown[w] = counts
    data["breakdown"] = breakdown
    rows += ["", "Each direct answer sorted into the post's failures (counts of scenes):"]
    rows.append(f"{'width':>6} " + " ".join(f"{k:>22}" for k in breakdown[448]))
    for w, c in breakdown.items():
        rows.append(f"{w:>6} " + " ".join(f"{v:>22}" for v in c.values()))
    rows += ["", "Rules, applied in order: correct; oracle wrong -> 3; own transcript has the time -> 2;",
             "crop right -> 1; otherwise unclear. A transcript is lossy, so 'not in transcript' is evidence for 1,",
             "not proof, as the post says."]
    write(out_path(args, "03_probes"), command("03_probes.py"), rows, data)


if __name__ == "__main__":
    main()
