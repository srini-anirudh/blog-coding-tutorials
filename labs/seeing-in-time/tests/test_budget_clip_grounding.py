import pytest
import torch

from seeingtime.budget import seconds_that_fit, slowfast_tokens, tokens_per_frame, video_tokens
from seeingtime.clip import (EVENTS, apollo_clip_times, coverage, fps_times, keyframes, relevance_with_coverage,
                             topk_relevance, uniform_times)
from seeingtime.grounding import best_on_grid, snap_nearest, tiou, worst_case_floor


def test_post_token_numbers():
    assert tokens_per_frame(448, 448) == 256
    assert video_tokens(3600, 2, 256) == 1_843_200
    assert video_tokens(60, 2, 256) == 30_720
    assert video_tokens(10800, 1, 66) == 712_800
    assert slowfast_tokens(10, 288, 50, 16) == 3_680
    assert slowfast_tokens(4, 192, 16, 16) == 16 * 64
    assert seconds_that_fit(1_000_000, 1, 66) > 3 * seconds_that_fit(1_000_000, 1, 258)
    with pytest.raises(ValueError):
        tokens_per_frame(450, 448)


def test_sampling_matches_post_figure3():
    c = coverage(uniform_times(8))
    assert c["events_seen"] == 3 and c["direction_recoverable"]
    assert coverage(fps_times(0.5))["events_seen"] == 5
    assert len(fps_times(2)) == 120 and abs(fps_times(2)[0] - 0.25) < 1e-9


def test_apollo_clips_keep_gap_fixed():
    for dur in (30, 600, 3600):
        ts = apollo_clip_times(dur, 2.0, 8, 32)
        assert len(ts) == 32
        for k in range(0, 32, 8):
            gaps = {round(b - a, 6) for a, b in zip(ts[k:k + 8], ts[k + 1:k + 8])}
            assert gaps == {0.5}


def test_keyframes_drop_duplicates():
    f = torch.ones(5, 4)
    f[3] = torch.tensor([1.0, -1, 1, -1])
    assert keyframes(f, 0.99) == [0, 3, 4]


def test_coverage_spreads_selection():
    rel = torch.zeros(100)
    rel[60:70] = 1.0
    rel += torch.linspace(0, 0.01, 100)
    top = topk_relevance(rel, 8)
    spread = relevance_with_coverage(rel, 8, 4)
    assert all(60 <= i < 70 for i in top)
    assert len(spread) == 8 and min(spread) < 25 and max(spread) >= 75


def test_tiou_and_frame_ceiling():
    assert tiou((0, 2), (0, 2)) == 1.0
    assert tiou((0, 1), (2, 3)) == 0.0
    assert abs(tiou((1, 3), (2, 4)) - 1 / 3) < 1e-9
    assert abs(best_on_grid((13.4, 16.6), 2.0, 30) - 0.625) < 1e-9  # the post: 0.63 at 0.5 fps
    assert best_on_grid((13.4, 16.6), 2.0, 30) < 0.7 < best_on_grid((13.4, 16.6), 1.0, 30)
    assert worst_case_floor(4, 2) == 0.5 and worst_case_floor(4, 0.5) == 0.875
    assert snap_nearest((13.4, 16.6), 2.0) == (14.0, 16.0)


def test_floor_holds_for_snapped_oracle():
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        L, step = float(torch.empty(1).uniform_(3, 8, generator=g)), 1.0
        s = float(torch.empty(1).uniform_(0, 40, generator=g))
        phase = float(torch.rand(1, generator=g))
        assert tiou(snap_nearest((s, s + L), step, phase), (s, s + L)) >= worst_case_floor(L, step) - 1e-9


def test_events_match_post():
    assert [(e.start, e.end) for e in EVENTS] == [(8, 20), (20, 21.5), (40, 44), (44, 50), (52, 53)]
