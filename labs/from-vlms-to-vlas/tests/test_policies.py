import numpy as np
import torch

from vlalab.envs import LaneTask, ObstacleTask
from vlalab.heads import BinHead, FlowHead, RegressionHead, sample_tau, train
from vlalab.tokenizers import UniformBinner


def test_lane_expert_stays_in_band_and_open_loop_drifts():
    task = LaneTask()
    rng = np.random.default_rng(0)
    _, y = task.rollout(task.expert, 50, 200, rng, y0_std=0.02, act_noise=0.3)
    _, y_ff = task.rollout(lambda t, y: task.drift(t), 50, 200, rng, y0_std=0.02, act_noise=0.3)
    assert (np.abs(y) > task.band).mean() < 0.05
    assert np.abs(y_ff).mean() > 3 * np.abs(y).mean()  # no feedback: errors accumulate


def test_obstacle_expert_is_bimodal_and_safe():
    task = ObstacleTask()
    obs, act = task.demos(100, 30, np.random.default_rng(0))
    first = act.reshape(100, 30, 2)[:, 0, 0]
    assert set(np.sign(first)) == {-1.0, 1.0}
    rng = np.random.default_rng(1)
    side = rng.choice([-1.0, 1.0], 200)
    res = task.evaluate(lambda p: task.expert(p, side), 200, 40, rng)
    assert res["collision_rate"] == 0 and res["success_rate"] == 1


def test_chunked_demos_line_up():
    task = ObstacleTask()
    obs1, act1 = task.demos(4, 30, np.random.default_rng(0), chunk=1)
    obs8, act8 = task.demos(4, 30, np.random.default_rng(0), chunk=8)
    assert act8.shape == (120, 16) and np.allclose(obs1, obs8)
    assert np.allclose(act8[:, :2], act1)
    a = act8.reshape(4, 30, 16)
    assert np.allclose(a[:, 0, 2:4], act1.reshape(4, 30, 2)[:, 1])


def test_tau_distribution_favours_noise():
    torch.manual_seed(0)
    tau = sample_tau(20000)
    assert 0 <= tau.min() and tau.max() < 1
    assert (tau < 0.5).float().mean() > 0.6  # Beta(1.5, 1) mirrored: more weight on noisy inputs


def bimodal_data(n=2000):
    rng = np.random.default_rng(0)
    obs = np.zeros((n, 1), dtype=np.float32)
    act = (rng.choice([-1.0, 1.0], (n, 1)) + 0.05 * rng.standard_normal((n, 1))).astype(np.float32)
    return obs, act


def test_mse_averages_modes_but_bins_and_flow_keep_them():
    obs, act = bimodal_data()
    q = torch.zeros(1000, 1)
    reg = RegressionHead(1, 1, width=32)
    train(reg, obs, act, steps=300)
    assert abs(float(reg.act(q).mean())) < 0.3  # the mean of -1 and +1
    binh = BinHead(1, 1, UniformBinner(64).fit(act), width=32)
    train(binh, obs, act, steps=300)
    s = binh.act(q, generator=torch.Generator().manual_seed(0)).numpy()
    assert (np.abs(s) > 0.7).mean() > 0.9
    flow = FlowHead(1, 1, width=64, depth=3)
    train(flow, obs, act, steps=800)
    s = flow.act(q, generator=torch.Generator().manual_seed(0)).numpy()
    assert (np.abs(s) > 0.7).mean() > 0.8
    assert 0.3 < (s > 0).mean() < 0.7
