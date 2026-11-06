import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visrl import boxes, charts  # noqa: E402
from visrl import verifiers as V  # noqa: E402
from visrl.grpo import group_advantages, p_zero_advantage, zero_advantage  # noqa: E402


# ---------- advantages ----------

def test_near_tie_amplified_by_std():
    adv = group_advantages(torch.tensor([[0.30] * 7 + [0.31]]))[0]
    assert float(adv[-1]) == pytest.approx(2.65, abs=0.01)
    assert float(adv[0]) == pytest.approx(-0.38, abs=0.01)
    no_std = group_advantages(torch.tensor([[0.30] * 7 + [0.31]]), scale_by_std=False)[0]
    assert float(no_std.abs().max()) < 0.01


def test_whole_image_box_gets_the_only_positive_advantage():
    r = torch.tensor([[0.0] * 7 + [0.2]])
    assert float(r.mean()) == pytest.approx(0.025)
    adv = group_advantages(r)[0]
    assert (adv[:7] < 0).all() and adv[7] > 0


def test_equal_group_has_exactly_zero_advantage():
    r = torch.tensor([[0.3] * 8, [1.0] * 8])
    assert (group_advantages(r) == 0).all()
    assert zero_advantage(r).all()


@pytest.mark.parametrize("p,expected", [(0.5, 0.0078), (0.9, 0.4305), (0.95, 0.6634), (0.05, 0.6634)])
def test_zero_advantage_probability_matches_post(p, expected):
    assert p_zero_advantage(p, 8) == pytest.approx(expected, abs=1e-4)


# ---------- box policy ----------

def test_vectorised_box_rewards_match_scalar_verifiers():
    g = torch.Generator().manual_seed(0)
    gold = boxes.sample_targets(20, g)
    pred = boxes.make_boxes(gold, torch.randn(20, 1, generator=g) * 0.5, 0.1, g)[:, 0]
    for i in range(20):
        p, t = tuple(pred[i].tolist()), tuple(gold[i].tolist())
        assert float(boxes.iou_t(pred[i], gold[i])) == pytest.approx(V.box_iou(p, t), abs=1e-5)
        assert float(boxes.giou_t(pred[i], gold[i])) == pytest.approx(V.giou(p, t), abs=1e-5)
        assert float(boxes.coverage_t(pred[i], gold[i])) == pytest.approx(V.coverage(p, t), abs=1e-5)
        assert float(boxes.hit_t(pred[i], gold[i])) == V.center_hit(p, t)


def test_soft_iou_grows_hard_boxes_and_size_term_holds_them():
    iou = boxes.train_box_policy("iou", steps=120, seed=0)
    fixed = boxes.train_box_policy("iou+size", steps=120, seed=0)
    assert iou.area_ratio["hard"][-1] > 1.4
    assert abs(iou.area_ratio["easy"][-1] - 1) < 0.15
    assert fixed.area_ratio["hard"][-1] < iou.area_ratio["hard"][-1]


# ---------- chart VLM ----------

def test_render_and_answers():
    g = torch.Generator().manual_seed(0)
    b = charts.make_batch(16, g)
    assert b.images.shape == (16, 1, charts.SIZE, charts.SIZE)
    assert ((b.ans <= 18) & (b.ans >= 0)).all()
    clean = charts.render(torch.tensor([3]), torch.tensor([7]), g, noise=0.0)[0, 0]
    heights = (clean > 0).sum(0)
    assert heights.max() == 2 * (7 + 1)


def test_step_checks_catch_cancelling_misreadings():
    b = charts.Batch(torch.zeros(1, 1, 24, 24), torch.tensor([charts.DIFF]), torch.tensor([4]), torch.tensor([7]),
                     torch.tensor([3]))
    tokens = torch.tensor([[3, 6, 3]])          # both readings off by one, answer still right
    assert charts.outcome(b, tokens).item() == 1
    assert charts.step_correct(b, tokens).tolist() == [[0, 0, 1]]


def test_freeze_modes_block_gradient():
    m = charts.ChartVLM()
    for mode, frozen in [("llm", {"encoder", "connector"}), ("connector+llm", {"encoder"}), ("full", set())]:
        charts.set_trainable(m, mode)
        for name, p in m.named_parameters():
            assert p.requires_grad == (m.part_of(name) not in frozen)


def test_oracle_forces_true_readings():
    m = charts.ChartVLM()
    b = charts.make_batch(8, torch.Generator().manual_seed(1))
    tokens = m.generate(b, sample=False, force_reads=True)
    assert (tokens[:, 0] == b.v1).all() and (tokens[:, 1] == b.v2).all()


def test_grpo_step_runs_and_only_trains_open_parts():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    m = charts.ChartVLM()
    enc = copy.deepcopy(m.encoder.state_dict())
    cfg = charts.RLConfig(trainable="connector+llm", prompts=4, k=4, steps=2, dynamic=True)
    charts.set_trainable(m, cfg.trainable)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    log = charts.grpo_step(m, opt, cfg, g)
    assert log["encoder"] == 0 and log["llm"] >= 0
    assert all(torch.equal(enc[k], v) for k, v in m.encoder.state_dict().items())
    ev = charts.evaluate(m, charts.make_batch(32, g))
    assert set(ev) == {"A_img", "A_blank", "A_oracle", "P"}


def test_chart_script_labels_are_distinct():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import train_chart_policy as t

    labels = [t.label(c) for exp in t.EXPERIMENTS.values() for c in exp]
    unique = set(labels)
    assert len(unique) == 6                     # baseline shared by the three experiments, five others
    assert any("96 prompts" in x for x in unique)
