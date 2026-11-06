import pytest

from stats_lib import mean, median, summarize, variance


def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([4, 1, 3, 2]) == 2.5


def test_variance():
    assert variance([1, 2, 3, 4]) == 1.25


def test_empty():
    with pytest.raises(ValueError):
        median([])


def test_summarize():
    assert summarize([1, 2, 3, 4])["median"] == 2.5
