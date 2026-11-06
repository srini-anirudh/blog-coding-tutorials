"""Figure 6 of the post: visual tokens per image under each family's preprocessing rule.

Offline. Prints one row per rule and one column per preset image, then the post's simplified formula next to the
exact rule for the three rules the post marks as approximations.
"""

import _path  # noqa: F401

from vlmfam import tokens as T
from vlmfam.report import Report

r = Report("01_tokens_per_image")
names = [p[0] for p in T.PRESETS]
r("Visual tokens per image. Columns are the post's Figure 6 presets:")
r("  " + ", ".join(f"{n} {w}x{h}" for n, w, h in T.PRESETS))
r("")
rows, data = [], {}
for name, rule in T.RULES.items():
    counts = [rule(w, h) for _, w, h in T.PRESETS]
    rows.append([name, counts[0].kind] + [c.tokens for c in counts])
    data[name] = {"kind": counts[0].kind, **{n: c.tokens for n, c in zip(names, counts)}}
r.table(["rule", "kind"] + names, rows)

r("")
r("How each rule spends tokens on the letter scan (1700 x 2200):")
for name, rule in T.RULES.items():
    r(f"  {name:<16} {rule(1700, 2200).detail}")

r("")
r("The post marks three bars as approximations. Exact rule against the post's simplified formula:")
rows = []
for name in ("LLaVA-NeXT", "Idefics3", "Pixtral"):
    for n, w, h in T.PRESETS:
        exact, post = T.RULES[name](w, h).tokens, T.post_formula(name, w, h)
        rows.append([name, n, exact, post, f"{(post - exact) / exact:+.0%}"])
r.table(["rule", "image", "exact", "post formula", "post vs exact"], rows)

r("")
r("Fixed rules cost the same for every image; native rules grow with area until the pixel cap:")
for side in (224, 448, 896, 1344, 2688, 4032):
    q, p = T.qwen2_vl(side, side), T.pixtral(side, side)
    r(f"  {side:>4} px square: Qwen2-VL {q.tokens:>6}  Pixtral {p.tokens:>5}  InternVL {T.internvl(side, side).tokens:>5}"
      f"  Gemma 3 {T.gemma3(side, side).tokens}")
r.save(data)
