"""Five views of one screen, their cost in tokens, and which ones survive a redesign of the page.

Post section: "What a screen looks like to a model" (Figure 4, Table 1). Token counts for text use 4 characters
per token; the screenshot cost assumes a 1920 x 1080 screen, 14-pixel patches and a 1024-token budget.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actlab.desk import AppState, Layout  # noqa: E402
from actlab import observe as ob  # noqa: E402
from actlab.report import Report  # noqa: E402

rep = Report("02_observations")
orig = AppState(Layout.sample(random.Random(0), jitter=False), amount="42.50", category="Travel")
new = AppState(Layout.sample(random.Random(0), jitter=False, redesign=True), amount="42.50", category="Travel")

rep("Accessibility tree (original page):")
for line in ob.a11y_tree(orig).splitlines():
    rep("  " + line)
rep(f"DOM: {len(ob.dom(orig))} characters, of which the visible elements are only part.")
rep("")
rep("Approximate prompt tokens per observation:")
for k, v in ob.observation_costs(orig).items():
    rep(f"  {k:12s} {v:6d}")

# Locators written once, against the original page.
mark = next(i for i, b in ob.set_of_mark(orig)[1].items() if b == orig.layout.submit)
fixed = orig.layout.submit.center
locators = {
    "screenshot, fixed coordinates": lambda s: ob.locate_fixed_point(s, fixed),
    "screenshot, visual cue (primary colour)": ob.locate_primary_colour,
    "accessibility tree, role + name 'Submit'": ob.locate_a11y_by_name,
    "DOM, id 'btn-submit'": ob.locate_dom_by_id,
    f"Set-of-Mark, mark {mark}": lambda s: ob.locate_mark(s, mark),
}
rep("")
rep("Redesign: Submit moves to the left, is renamed 'Send report', gets id 'action-primary-7', and moves to the top")
rep("of the document (so Set-of-Mark numbers shift). Does each locator still land on the submit button?")
rep(f"  {'locator':44s} {'original':>9s} {'redesign':>9s}")
rows = {}
for name, f in locators.items():
    res = []
    for s in (orig, new):
        p = f(s)
        res.append(p is not None and s.layout.submit.contains(*p))
    rows[name] = res
    rep(f"  {name:44s} {str(res[0]):>9s} {str(res[1]):>9s}")
rep("The visual cue is a hand-written stand-in for a learned grounder: it survives because the button keeps its")
rep("style. A renamed label breaks string matching, and a changed id breaks the DOM locator.")
rep.save({"costs": ob.observation_costs(orig), "locators": rows})
