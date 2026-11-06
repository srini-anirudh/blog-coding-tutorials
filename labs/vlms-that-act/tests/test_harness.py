import pytest

from actlab.harness import Gate, Rejected, context_tokens, fit_to_budget, to_screen, validate


def test_fit_to_budget_keeps_aspect_and_budget():
    w, h, s = fit_to_budget(3840, 2160, 1024)
    assert (w / 14) * (h / 14) <= 1024 + 1e-6 and abs(w / h - 16 / 9) < 0.01
    assert fit_to_budget(200, 100, 1024) == (200, 100, 1.0)


@pytest.mark.parametrize("fmt,pt", [("norm01", (0.5, 0.25)), ("bin999", (499.5, 249.75)), ("screen_px", (960, 270))])
def test_coordinate_formats_agree(fmt, pt):
    assert to_screen(pt, fmt, 1920, 1080) == (960, 270)


def test_resized_px_needs_scale():
    _, _, s = fit_to_budget(2560, 1440, 1024)
    assert to_screen((1000 * s, 500 * s), "resized_px", 2560, 1440, s) == (1000, 500)


def test_validate_rejects_bad_actions():
    with pytest.raises(Rejected):
        validate(("click", 2000, 5), 1920, 1080)
    with pytest.raises(Rejected):
        validate(("element", "e9"), 10, 10, element_ids={"e1"})
    with pytest.raises(Rejected):
        validate(("shell", "rm"), 10, 10)
    assert validate(("click", 5, 5), 10, 10)


def test_gate_blocks_external_and_irreversible():
    g = Gate()
    assert g.risk({"verb": "click"}) == "safe"
    assert g.risk({"verb": "delete", "to": "https://expenses.example/r/1"}) == "irreversible"
    assert g.risk({"verb": "export", "to": "x@outside.example"}) == "external"
    assert not g.check({"verb": "export", "to": "x@outside.example"})
    assert Gate(confirm=lambda a: True).check({"verb": "delete"})


def test_context_compaction_is_bounded():
    full = [context_tokens(t, 1000, None) for t in range(30)]
    kept = [context_tokens(t, 1000, 3) for t in range(30)]
    assert full[-1] > 25_000 and kept[-1] < 5_000 and full[0] == kept[0]
