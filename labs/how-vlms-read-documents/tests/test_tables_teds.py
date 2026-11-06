import pytest

from docread.tables import Table, flatten, from_html, from_markdown, from_otsl, to_html, to_markdown, to_otsl
from docread.teds import Node, teds, tree_edit_distance

GOLD = Table.from_lists([["Model", ("Score", 2)], ["A", "81.6", "82.2"], ["B", "65.4", "67.5"]])
PRED = Table.from_lists([["Model", "Score", ""], ["A", "81.6", "82.2"], ["B", "65.4", "67.3"]])


def t(label, *children):
    return Node(label, children=list(children))


def test_zhang_shasha_classic_example():
    a = t("f", t("d", t("a"), t("c", t("b"))), t("e"))
    b = t("f", t("c", t("d", t("a"), t("b"))), t("e"))
    assert tree_edit_distance(a, b, cost=lambda x, y: float(x.tag != y.tag)) == 2


def test_figure6_numbers():
    assert teds(PRED, GOLD) == pytest.approx(1 - 2.25 / 13)
    assert teds(PRED, GOLD, structure_only=True) == pytest.approx(1 - 2 / 13)
    assert 1 - len("3") / len(flatten(GOLD)) == pytest.approx(0.971, abs=1e-3)
    assert teds(GOLD, GOLD) == 1.0


def test_wrong_column_is_invisible_to_text_and_structure_scores():
    a = Table.from_lists([["x", "1", ""]])
    b = Table.from_lists([["x", "", "1"]])
    assert flatten(a) == flatten(b)
    assert teds(a, b, structure_only=True) == 1.0
    assert teds(a, b) < 1.0


def test_round_trips():
    span = Table.from_lists([[("Region", 1, 2), ("Sales", 2)], ["Q1", "Q2"], ["North", "5", "7"]])
    assert from_otsl(to_otsl(span)) == span
    assert from_html(to_html(span)) == span
    assert to_otsl(GOLD)[:6] == ["C", "Model", "C", "Score", "L", "NL"]
    assert "U" in to_otsl(span)
    md = from_markdown(to_markdown(GOLD))
    assert md != GOLD and teds(md, GOLD) == pytest.approx(1 - 2 / 13)  # Markdown loses the span
