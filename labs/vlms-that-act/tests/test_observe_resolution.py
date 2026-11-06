import math
import random

from actlab.desk import AppState, Layout
from actlab import observe as ob
from actlab.resolution import icon_strip, icons_per_token, pixels_per_token


def states():
    return (AppState(Layout.sample(random.Random(0), False)),
            AppState(Layout.sample(random.Random(0), False, redesign=True)))


def test_redesign_breaks_string_and_id_locators_but_not_visual_cue():
    orig, new = states()
    assert ob.locate_dom_by_id(orig) and ob.locate_dom_by_id(new) is None
    assert ob.locate_a11y_by_name(orig) and ob.locate_a11y_by_name(new) is None
    for s in (orig, new):
        assert s.layout.submit.contains(*ob.locate_primary_colour(s))


def test_dom_is_longer_than_tree_and_marks_cover_elements():
    orig, _ = states()
    assert len(ob.dom(orig)) > 3 * len(ob.a11y_tree(orig))
    _, marks = ob.set_of_mark(orig)
    assert len(marks) == len(ob.elements(orig))


def test_visual_tokens_respect_budget():
    assert ob.visual_tokens(1920, 1080, 14, 1024) <= 1024
    assert ob.visual_tokens(140, 140, 14, 1024) == 100


def test_pixels_per_token_matches_post():
    assert abs(pixels_per_token(3840, 2160, 1024) - 90.0) < 0.1
    assert math.isclose(pixels_per_token(100, 100, 100), 10.0)


def test_icons_share_tokens_at_low_budget():
    _, boxes = icon_strip(16, 24)
    assert max(icons_per_token(boxes, 90.0)) >= 3
    assert max(icons_per_token(boxes, 22.5)) == 1
