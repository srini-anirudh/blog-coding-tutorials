"""Active perception and evidence search. The plaque now sits in a random cell of a 4 x 4 grid.

Compares, on the same scenes, how accuracy trades against visual tokens:
  one look at 448 px and at 2016 px;
  an attention-guided crop: look at 896 px, crop the 4 x 4 cell with the most attention, re-ask at 448 px;
  best-first search over a region tree, scored by the model's own P(Yes), with and without a verification step;
  reading all 16 cells and keeping the answer the verifier trusts most.
About 6 min on one H100.
    python labs/thinking-with-images/scripts/06_search.py --n 60
"""
from _common import command, out_path, parser, reader

from lookagain.answers import VERIFY, VERIFY_GENERIC
from lookagain.probes import all_cells, ask_views, blind_prior, score, verifier_separation
from lookagain.scene import make_scene
from lookagain.search import attention_cell, best_first, exhaustive
from lookagain.util import write
from lookagain.views import cell_view, full_view


def main():
    args = parser(__doc__, 60).parse_args()
    vlm = reader(args, attn="eager")
    scenes = [make_scene(5000 + i, cell=None) for i in range(args.n)]
    prior = blind_prior(vlm)
    rows = [f"n={args.n} scenes, plaque in a random cell of 16; tokens are Qwen2-VL visual tokens per question", "",
            f"{'method':<34} {'acc':>5} {'bus':>5} {'prior':>6} {'calls':>6} {'tokens':>7}"]
    data = {}

    def line(name, preds, calls, tokens):
        r = score(scenes, preds, prior)
        r.update(calls=calls, tokens=tokens)
        data[name] = r
        rows.append(f"{name:<34} {r['acc']:>5.2f} {r['said_bus_time']:>5.2f} {r['said_prior']:>6.2f} "
                    f"{calls:>6.1f} {tokens:>7.0f}")

    for w in (448, 2016):
        v = full_view(w)
        line(f"one look @{w}", ask_views(vlm, scenes, v), 1, (v.enc_w // 14) * (v.enc_h // 14) // 4)

    if hasattr(vlm, "attention_map"):
        cells = [attention_cell(vlm, s, full_view(896)) for s in scenes]
        hit = sum(c == s.plaque_cell for c, s in zip(cells, scenes)) / len(scenes)
        preds = ask_views(vlm, scenes, [cell_view(*c) for c in cells])
        line("attention-guided crop (896 -> 448)", preds, 2, 768 + 192)
        look = ask_views(vlm, scenes, full_view(896))
        wrong = [i for i, s in enumerate(scenes) if look[i] != s.close_time]
        hit_wrong = sum(cells[i] == scenes[i].plaque_cell for i in wrong) / max(1, len(wrong))
        data["attention_hit"] = {"all": hit, "when_896_answer_wrong": hit_wrong, "n_wrong": len(wrong),
                                 "acc_896": sum(a == s.close_time for a, s in zip(look, scenes)) / len(scenes)}

    for verify in (False, True):
        trs = [best_first(vlm, s, full_view(448), verify=verify) for s in scenes]
        name = "best-first search" + (" + verify" if verify else "")
        line(name, [t.answer for t in trs], sum(t.calls for t in trs) / len(trs),
             sum(t.visual_tokens for t in trs) / len(trs))
        data[name]["rejected_reads"] = sum(t.rejected for t in trs)
        data[name]["first_read_was_bus"] = sum(bool(t.reads) and t.reads[0][1] == s.bus_time
                                               for t, s in zip(trs, scenes))

    trs = [exhaustive(vlm, s, all_cells()) for s in scenes]
    line("read all 16 cells + verifier", [t.answer for t in trs], trs[0].calls, trs[0].visual_tokens)

    rows += ["", "calls: model calls per question (answer, score or verify). tokens: visual tokens summed over calls.",
             "bus: answered the bus-sign time, the distractor."]
    if "attention_hit" in data:
        a = data["attention_hit"]
        rows += [f"Attention picks the plaque cell in {a['all']:.2f} of scenes (chance 1/16 = 0.06); "
                 f"in {a['when_896_answer_wrong']:.2f} of the {a['n_wrong']} scenes where the 896 px answer was "
                 f"wrong (896 px accuracy {a['acc_896']:.2f})."]
    for name in ("best-first search", "best-first search + verify"):
        rows.append(f"{name}: first read was the bus sign in {data[name]['first_read_was_bus']} scenes; "
                    f"reads rejected by verification: {data[name]['rejected_reads']}")
    rows += ["", "Verifier prompts on the plaque cell and on the bus-sign cell (448 px each), threshold 0.5:",
             f"{'prompt':<10} {'P(Yes) plaque':>14} {'P(Yes) bus':>11} {'plaque > bus':>13} {'bus > 0.5':>10} "
             f"{'plaque <= 0.5':>14}"]
    for name, prompt in (("generic", VERIFY_GENERIC), ("OPEN-hours", VERIFY)):
        v = verifier_separation(vlm, scenes, prompt)
        data["verifier_" + name] = v
        rows.append(f"{name:<10} {v['p_yes_plaque']:>14.2f} {v['p_yes_bus']:>11.2f} {v['plaque_above_bus']:>13.2f} "
                    f"{v['bus_above_half']:>10.2f} {v['plaque_below_half']:>14.2f}")
    rows += ["The search above uses the OPEN-hours prompt. It was chosen on scenes 2000-2039, after the generic prompt",
             "scored the bus sign above 0.5 and so rejected nothing; the scenes here start at 5000."]
    write(out_path(args, "06_search"), command("06_search.py"), rows, data)


if __name__ == "__main__":
    main()
