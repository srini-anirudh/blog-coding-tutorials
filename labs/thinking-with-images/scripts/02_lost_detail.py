"""The lost-detail experiment: accuracy against encoder width, with and without longer reasoning, and the crop.

Qwen2-VL-2B answers "When does the bakery close?" on the full photo resized to each width. Then it answers on the
top-right crop at the smallest budget. A chain-of-thought prompt with up to 256 new tokens stands in for "more
reasoning tokens". About 4 min on one H100.
    python labs/thinking-with-images/scripts/02_lost_detail.py --n 100
"""
from _common import command, out_path, parser, reader

from lookagain.answers import COT, closing_time
from lookagain.probes import answer_region_view, ask_views, batched, blind_prior, score
from lookagain.scene import make_scene
from lookagain.util import write
from lookagain.views import full_view, render

WIDTHS = [448, 672, 896, 1120, 1344, 1512, 2016]


def main():
    ap = parser(__doc__, 100)
    ap.add_argument("--n-cot", type=int, default=40)
    args = ap.parse_args()
    vlm = reader(args)
    scenes = [make_scene(1000 + i) for i in range(args.n)]
    prior = blind_prior(vlm)
    rows, data = [], {"direct": {}, "cot": {}}
    head = f"{'condition':<22} {'text px':>7} {'tokens':>6} {'acc':>5} {'prior':>6} {'bus':>5} {'none':>5}"
    rows.append(f"Direct answer, max 24 new tokens, greedy, n={args.n}. With no image the model answers {prior!r}: that is the 'prior' column.")
    rows.append(head)
    conds = [(f"full @{w}", full_view(w)) for w in WIDTHS]
    conds.append(("top-right crop @448", [answer_region_view(s) for s in scenes]))
    for name, v in conds:
        r = score(scenes, ask_views(vlm, scenes, v), prior)
        v0 = v[0] if isinstance(v, list) else v
        tok = (v0.enc_w // 14) * (v0.enc_h // 14) // 4
        r.update(text_px=v0.text_px, tokens=tok)
        data["direct"][name] = r
        rows.append(f"{name:<22} {v0.text_px:>7.1f} {tok:>6} {r['acc']:>5.2f} {r['said_prior']:>6.2f} "
                    f"{r['said_bus_time']:>5.2f} {r['no_time']:>5.2f}")

    clean = [make_scene(1000 + i, distractor=False) for i in range(args.n)]
    rows += ["", f"Same photos with the bus sign removed, direct answer, n={args.n}: the resolution curve alone", head]
    data["no_distractor"] = {}
    for name, v in conds:
        r = score(clean, ask_views(vlm, clean, v if not isinstance(v, list) else [answer_region_view(s) for s in clean]), prior)
        v0 = v[0] if isinstance(v, list) else v
        data["no_distractor"][name] = r
        rows.append(f"{name:<22} {v0.text_px:>7.1f} {(v0.enc_w // 14) * (v0.enc_h // 14) // 4:>6} {r['acc']:>5.2f} "
                    f"{r['said_prior']:>6.2f} {r['said_bus_time']:>5.2f} {r['no_time']:>5.2f}")

    sub = scenes[:args.n_cot]
    rows += ["", f"Chain of thought, up to 256 new tokens, greedy, n={len(sub)}", head + f" {'gen tok':>7}"]
    for name, v in [("full @448", full_view(448)), ("full @2016", full_view(2016)),
                    ("top-right crop @448", [answer_region_view(s) for s in sub])]:
        vs = v if isinstance(v, list) else [v] * len(sub)
        outs = [o[0] for o in batched(vlm, [([render(s, x)], COT) for s, x in zip(sub, vs)], max_new_tokens=256)]
        r = score(sub, [closing_time(o) for o in outs], prior)
        tok = getattr(vlm, "tok", None)
        gen = sum(len(tok(o).input_ids) if tok else len(o.split()) for o in outs) / len(outs)
        r.update(mean_generated_tokens=gen, example=outs[0][:600])
        data["cot"][name] = r
        rows.append(f"{name:<22} {vs[0].text_px:>7.1f} {(vs[0].enc_w // 14) * (vs[0].enc_h // 14) // 4:>6} "
                    f"{r['acc']:>5.2f} {r['said_prior']:>6.2f} {r['said_bus_time']:>5.2f} {r['no_time']:>5.2f} "
                    f"{gen:>7.0f}")
    rows += ["", "acc: exact closing time. prior: gave the same answer as with no image. bus: answered the bus-sign time.",
             "The crop also removes the bus sign, which the post notes cropping can do; the second table separates that.",
             "none: no time in the answer. Chance over the 10 closing times is 0.10. gen tok: mean generated tokens.",
             "", "Example chain of thought at 448 px:", data["cot"]["full @448"]["example"]]
    write(out_path(args, "02_lost_detail"), command("02_lost_detail.py"), rows, data)


if __name__ == "__main__":
    main()
