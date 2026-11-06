"""Score tables as trees: reproduce Figure 6 (TEDS 0.83, TEDS-S 0.85, flattened text 0.97), then show the post's
claim that a cell moved to the wrong column keeps a high string score.

Post section: "Tables and the TEDS metric", Figure 6.
"""

import _common
from docread.metrics import normalized_edit_distance
from docread.tables import Table, flatten, from_markdown, to_markdown
from docread.teds import table_tree, teds, tree_edit_distance

log = _common.Report("05_teds")
data = {}


def report(name, pred, gold):
    a, b = table_tree(pred), table_tree(gold)
    flat = 1 - normalized_edit_distance(flatten(pred), flatten(gold))
    row = {"edit_cost": tree_edit_distance(a, b), "nodes": max(a.size(), b.size()), "teds": teds(pred, gold),
           "teds_s": teds(pred, gold, structure_only=True), "flattened_text": flat}
    log(f"{name:<38} cost {row['edit_cost']:.2f} / {row['nodes']} nodes   TEDS {row['teds']:.3f}   "
        f"TEDS-S {row['teds_s']:.3f}   flattened text {flat:.3f}")
    data[name] = row


gold = Table.from_lists([["Model", ("Score", 2)], ["A", "81.6", "82.2"], ["B", "65.4", "67.5"]])
fig6 = Table.from_lists([["Model", "Score", ""], ["A", "81.6", "82.2"], ["B", "65.4", "67.3"]])
log("Figure 6: the header lost its two-column span, an empty cell was added, and 67.5 became 67.3.")
report("Figure 6 prediction", fig6, gold)
log("Post: edit cost 1 + 1 + 0.25 = 2.25 over 13 nodes; TEDS 0.83, TEDS-S 0.85, flattened text 0.97.\n")

grid = Table.from_lists([["Item", "Net", "Tax"], ["Paper", "1,200.00", "216.00"], ["Toner", "300.00", "54.00"],
                         ["Ink", "80.00", ""]])
shifted = Table.from_lists([["Item", "Net", "Tax"], ["Paper", "1,200.00", "216.00"], ["Toner", "300.00", "54.00"],
                            ["Ink", "", "80.00"]])
log("A value in the wrong column: 'Ink 80.00' moved from Net to Tax. The table now says something false.")
report("one value in the wrong column", shifted, grid)
dropped = Table.from_lists([["Item", "Net", "Tax"], ["Paper", "1,200.00", "216.00"], ["Toner", "300.00", "54.00"]])
report("last row missing", dropped, grid)
merged = Table.from_lists([["Item", "Net", "Tax"], ["Paper", "1,200.00", "216.00"],
                           ["Toner Ink", "300.00 80.00", "54.00"]])
report("two rows merged into one", merged, grid)
log("The flattened string is identical when a value changes column, so a text metric gives it full marks.")
log("TEDS-S misses it too, because the grid is unchanged. Only TEDS, which compares each cell's text in place, drops.\n")

md_round_trip = from_markdown(to_markdown(gold))
log("Writing the Figure 6 ground truth as Markdown and reading it back loses the span:")
report("ground truth through Markdown", md_round_trip, gold)
log.save(data)
