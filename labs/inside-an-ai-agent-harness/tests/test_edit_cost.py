import random

import pytest
from harness.editing import EditError

from harness_lab import C10
from harness_lab.edit_cost import FORMATS, apply, land_edit, payload, target_edit

TEXT = (C10 / "harness" / "planning.py").read_text()
LINE = next(i for i, ln in enumerate(TEXT.splitlines()) if ln.strip().startswith("def ") and i > 5)


@pytest.mark.parametrize("fmt", FORMATS)
def test_clean_attempt_lands_and_slip_is_refused(fmt):
    assert apply(fmt, TEXT, payload(fmt, TEXT, LINE, slip=False)) == target_edit(TEXT, LINE)
    with pytest.raises(EditError):
        apply(fmt, TEXT, payload(fmt, TEXT, LINE, slip=True))


def test_mean_attempts_is_one_over_p():
    rng = random.Random(0)
    tries = [land_edit("unified diff", TEXT, LINE, 0.5, rng).attempts for _ in range(600)]
    assert abs(sum(tries) / len(tries) - 2.0) < 0.2
