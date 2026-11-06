# Lab: Reinforcement Learning for Agents From First Principles

Small, exact experiments for the claims in the theory post. Each one builds a mechanism from scratch and runs it, so
you can watch the post's argument happen. Six of them:

- Behaviour cloning pays O(eps T^2) and DAgger pays O(eps T). Measured regret curves, with log-log slopes.
- A hand-shaped milestone reward gets farmed by a loop. Potential-based shaping does not.
- The informative fraction of a batch, `1 - p^G - (1-p)^G` with `p = q^H`, and the fifty-percent horizon.
- Group-relative RL on a combination lock: the batch goes empty at long horizons, and four fixes are tested.
- Trajectory-level credit against anchor-state (step-level) credit on a gridworld.
- The straggler tax of a synchronous rollout batch, and the staleness bill of an asynchronous queue.

Post: https://srini-anirudh.github.io/blogs/reinforcement-learning-for-agents-from-first-principles/

Everything is tabular or a few lines of numpy, and it runs on a CPU in about a minute. Nothing here is a language
model. The post's large-scale results (web agents, SWE agents, 7B models) are not reproduced. Where the post quotes
them, this lab shows the mechanism small and says so.

The coding tutorial [C12: Train an Agent With RL](https://srini-anirudh.github.io/blogs/train-an-agent-with-rl/)
([code](../../c12-agent-rl)) already builds the same ideas around a small transformer policy that drives a text tool
environment: cloning and distribution shift, GRPO with a critic, an activity reward hack, a potential reward, a task
curriculum, and an asynchronous trainer with a staleness limit. This lab does not rebuild that. It isolates the
arithmetic the post derives, in settings small enough that each number can be checked against a formula.

## Layout

```
labs/reinforcement-learning-for-agents-from-first-principles/
├── rlfirst/
│   ├── imitation.py   # the lane world, an eps-error tabular learner, BC, BC with recovery demos, DAgger
│   ├── shaping.py     # a six-page shopping graph, value iteration, Q-learning, the cycle test
│   ├── horizon.py     # P_informative, p = q^H, group size needed, graded reward, H_50, recovery
│   ├── lock.py        # combination lock, softmax table policy, group-relative policy gradient, curriculum
│   ├── credit.py      # gridworld, trajectory-level and anchor-state advantages, the worked example
│   ├── rollouts.py    # log-normal rollout times, synchronous barrier, asynchronous queue with window k
│   └── report.py      # writes results/<script>.txt and .json with a hardware line and the command
├── scripts/           # one script per experiment, listed below
├── results/           # the real output of every script
└── tests/             # 28 offline tests
```

## Setup

From the repository root, after `pip install -e ".[dev]"`. Only `torch` and `numpy` are needed. No downloads, no GPU.

## Run everything

```bash
python -m pytest labs/reinforcement-learning-for-agents-from-first-principles -m "not slow and not gpu and not network"
python labs/reinforcement-learning-for-agents-from-first-principles/scripts/01_imitation_regret.py   #  6 s
python labs/reinforcement-learning-for-agents-from-first-principles/scripts/02_reward_shaping.py     #  4 s
python labs/reinforcement-learning-for-agents-from-first-principles/scripts/03_horizon_arithmetic.py #  4 s
python labs/reinforcement-learning-for-agents-from-first-principles/scripts/04_lock_grpo.py          # 17 s
python labs/reinforcement-learning-for-agents-from-first-principles/scripts/05_step_credit.py        # 19 s
python labs/reinforcement-learning-for-agents-from-first-principles/scripts/06_rollout_engine.py     #  7 s
```

Times are wall clock on an AMD EPYC 7R13 with 8 threads. Each script writes `results/<name>.txt` and `.json`. Pass
`--no-write` to print only.

## Which script covers which section of the post

| Post section | Script | What you see |
|---|---|---|
| Why cloning is not enough; The quadratic | `01_imitation_regret.py` | regret vs T for BC, BC with recovery demos, DAgger |
| Dense reward; The one shaping construction that is safe | `02_reward_shaping.py` | the loop the policy finds; cycle gains; random potentials |
| Where the signal goes; Now connect it to the horizon | `03_horizon_arithmetic.py` | `P_informative` and `q^H`, formula against simulation |
| The one empirical regularity that does exist | `03_horizon_arithmetic.py` | `H_50`, and how error recovery stretches it |
| Which of the standard fixes actually address this; The horizon as a curriculum variable | `04_lock_grpo.py` | informative fraction and success under each fix |
| Credit over fifty steps; How the field got finer | `05_step_credit.py` | worked example of Figure 7; training with three estimators |
| The utilisation bound; The bill for it | `06_rollout_engine.py` | `E[T] / E[max T]`; async throughput and trajectory age |

## Results

All numbers below come from `results/`, on **AMD EPYC 7R13, 8 threads, CPU only, Python 3.12, torch 2.15.0, numpy
2.5.4**. The torch scripts (04, 05) cap torch at 2 threads. Every result is a count, a rate or a simulated quantity, so
none of them depends on machine load.

Tests: 28 passed, in about 4 s. There are no `slow`, `gpu` or `network` tests. See `results/tests.txt`.

### 1. Cloning against interaction (`01_imitation_regret.txt`)

A one-dimensional lane. Cost 1 per step away from position 0. The learner errs with probability eps on states it has
labels for, and acts at random on states it has none for. In the cliff lane, `|x| > 1` is absorbing. 20,000
evaluation episodes per cell.

| World, eps | Slope of log regret vs log T, T = 10 to 200: BC / BC with recovery demos / DAgger | BC / DAgger regret at T = 50 |
|---|---|---|
| cliff, eps = 0.002 | 1.93 / 1.00 / 1.05 | 1.31 / 0.10 (12.9x) |
| cliff, eps = 0.01 | 1.81 / 1.12 / 1.13 | 6.18 / 0.56 (11.0x) |
| forgiving, eps = 0.002 | 1.60 / 0.99 / 1.02 | 0.82 / 0.10 (8.1x) |
| forgiving, eps = 0.01 | 1.51 / 1.00 / 1.00 | 3.80 / 0.50 (7.6x) |

On the cliff, cloning's regret grows with slope 1.93, close to the quadratic. DAgger's grows with slope 1.05. The
measured BC regret stays under the post's bound `eps T (T-1) / 2` in every cell. Two things the bound does not show:

- The gap at T = 50 is about 11 to 13x, not the 50x ratio of the two bounds. The bounds are worst cases.
- In the forgiving lane, a lost learner can wander back to 0 by chance, and the slope drops to about 1.5 to 1.6. This
  is the post's first sanity check, measured: real worlds are often more forgiving than the bound.

Cloning from demonstrations with injected noise (the expert is pushed off the lane and walks back) gives the same
linear curve as DAgger. The clean demonstrations label one state. The noisy ones label five. This is the post's
"design instruction": recovery has to be in the data or in the environment.

### 2. Reward shaping (`02_reward_shaping.txt`)

A shopping task: home, search, results, product, cart, checkout. Back links and a reload on every page. gamma = 0.99,
step budget 30. The milestone reward pays +0.1 each time the agent enters `product` or `cart`.

| Reward | Optimal policy (value iteration) | Optimal return | Q-learning, 5 seeds: success / steps | Gain around product -> cart -> product |
|---|---|---|---|---|
| sparse | checkout in 5 steps | 0.961 | 1.0 / 5 | 0 |
| milestone | loops product <-> cart, checks out on step 29 | 3.009 | 0.0 / 30 | +0.2 |
| potential | checkout in 5 steps | 0.961 | 1.0 / 5 | 0 (undiscounted), -0.014 (discounted) |

The milestone reward has a loop that pays 0.2 per lap. The exact optimum circles for 24 steps. The Q-learner never
checks out at all, because a stationary policy that loops forever beats one that leaves. The potential reward has
gain 0 on every one of the 8 cycles in the graph, and the same optimal path as the sparse reward.

A nuance the post does not state: with 200 random potentials, 200 of 200 keep the sparse reward's optimal action at
every page, but only if `Phi` is set to 0 when the step budget runs out, as well as at the goal. With `Phi` zeroed at
the goal only, 17 of 200 keep it. A time-out leaves a term `gamma^H Phi(s_H)` behind, and that term is a new
objective. In a finite-horizon task, "any potential" means any potential that is 0 wherever the episode can end.

### 3. Where the signal goes (`03_horizon_arithmetic.txt`)

`q = 0.93`, `G = 8`. "Simulated" uses 50,000 groups of real H-step rollouts.

| H | p = q^H | P_inf exact | G q^H | P_inf simulated | Useful groups | G needed for P_inf >= 0.5 |
|---|---|---|---|---|---|---|
| 10 | 0.484 | 0.992 | 3.87 | 0.992 | 1 in 1.0 | 3 |
| 30 | 0.113 | 0.618 | 0.91 | 0.622 | 1 in 1.6 | 6 |
| 50 | 0.0266 | 0.194 | 0.21 | 0.192 | 1 in 5.2 | 26 |
| 100 | 0.00071 | 0.0056 | 0.0056 | 0.0052 | 1 in 178 | 983 |

The post's success rates (about one half at 10 steps, 11% at 30, 3% at 50) match. Two of its batch-fraction phrases
do not match the formula it states:

- 50 steps: the post says "roughly one useful group in four". The formula gives 1 in 5.2. The small-p approximation
  `G q^H` gives 1 in 4.7.
- 100 steps: the post says "one in forty". The formula gives 1 in 178. One in forty would need q of about 0.944.

With graded reward (the fraction of steps passed before the first error), a group of 8 has nonzero variance in more
than 99.7% of groups at every H from 10 up. The fifty-percent horizon `ln 0.5 / ln q` is 9.55 at q = 0.93, and the
approximation `0.69 / (1 - q)` gives 9.90. If an error is caught and retried with probability r, H_50 becomes 18.8
(r = 0.5), 46.4 (r = 0.8) and 92.4 (r = 0.9). The per-step error rate is 7% in every case. Simulated success at the
rounded H_50 is between 0.49 and 0.51.

### 4. Group-relative RL on a combination lock (`04_lock_grpo.txt`)

16 tasks. Each task has a secret code with K = 4 choices per position. A wrong pick ends the episode. The softmax table
starts with probability 0.8 on the right action everywhere, standing in for a cloned policy. SGD, lr 20, 300 optimiser
steps, mean over 3 seeds. "Success" is the exact expected success of the sampling policy on the full lock.

| Setting | Informative fraction at step 0, predicted / measured | Mean informative fraction | Final success | Steps to success 0.5 |
|---|---|---|---|---|
| H = 5, binary | 0.958 / 0.979 | 0.157 | 0.996 | 2 |
| H = 10, binary | 0.597 / 0.604 | 0.277 | 0.991 | 12 |
| H = 20, binary | 0.089 / 0.063 | 0.450 | 0.974 | 69 |
| H = 40, binary | 0.001 / 0 | 0.0015 | 0.0002 | never |

At H = 40, from the same start:

| Intervention | Mean informative fraction | Final success | Steps to 0.5 | Env steps | Rollouts through backward |
|---|---|---|---|---|---|
| none (G = 8) | 0.0015 | 0.0002 | never | 1.9e5 | 38,400 |
| G = 32 | 0.0087 | 0.0010 | never | 8.0e5 | 153,600 |
| drop zero-variance groups | 0.0015 | 0.0002 | never | 1.9e5 | 56 |
| graded reward | 0.81 | 0.938 | 126 | 1.2e6 | 38,400 |
| curriculum `h_t = min(40, 5 + 5 floor(t / 30))` | 0.82 | 0.907 | 215 | 9.3e5 | 38,400 |

The measured informative fraction at the first step matches the formula. At H = 40 the baseline learns nothing in 300
steps. Quadrupling G quadruples the rollout cost and still learns nothing, as the post predicts: G would have to grow
like `q^-H`. Dropping zero-variance groups gives exactly the same run (a test checks the weights match), with 56
instead of 38,400 rollouts through the backward pass. It saves compute. It does not add signal. Graded reward and the
horizon curriculum both refill the batch and solve the lock. The curriculum reaches the full horizon at step 210 and
passes success 0.5 five steps later. Here "curriculum" means shorter tasks (only the first h positions are scored),
which is the Horizon row of the post's curriculum table. It is not a cap on the number of turns.

### 5. Step-level credit (`05_step_credit.txt`)

The worked example: four rollouts on a four-cell corridor. Winning rollout A wastes one step with a reload. Its
trajectory-level advantage is +0.866 on every action, the reload included. Its anchor-state step advantage at the
reload is -1.155, so the two-level advantage of the reload is -0.289. The move to the right at the same cell gets
+1.443.

Training: a 7 x 7 grid, budget 30 steps, shortest path 12. Five actions, one of them a reload. 4 groups of 8, SGD lr 1,
150 steps, mean over 5 seeds. The outcome reward is binary in all three rows.

| Advantage | Steps to success 0.9 | Steps until wins take <= 13 moves | Final moves per win | Final wasted-step fraction |
|---|---|---|---|---|
| trajectory-level, binary R | 66 | never (5 of 5 seeds) | 16.0 | 0.109 |
| trajectory-level, R = gamma^(length - 1) | 66 | 96 | 12.2 | 0.006 |
| episode + anchor step, lambda = 1 | 40 | 53 | 12.1 | 0.003 |

Binary trajectory-level credit never learns to stop wasting steps, because a wasted step inside a win is rewarded.
Discounting the outcome fixes the final behaviour but not the speed. Anchor grouping reaches both thresholds first.
98% of all steps landed in an anchor group of size two or more, because a gridworld repeats states exactly. The post's
caveat applies: open-ended text observations rarely repeat, and this lab does not test that case. The post's
benchmark gains for anchor grouping (high single digits to low double digits) were not reproduced.

### 6. The rollout engine (`06_rollout_engine.txt`)

Simulated, not measured on a cluster. Rollout durations are log-normal with median 1 and shape sigma. Training-step
time is treated as zero.

Synchronous utilisation `E[T] / E[max T]`:

| sigma | N = 8 | N = 64 | N = 512 |
|---|---|---|---|
| 0.25 | 0.72 | 0.57 | 0.48 |
| 0.5 | 0.53 | 0.34 | 0.25 |
| 1.0 | 0.33 | 0.14 | 0.07 |
| 1.5 | 0.23 | 0.07 | 0.03 |

64 workers, sigma = 1.0, a trainer step every 64 finished trajectories:

| Mode | Usable trajectories per time unit | Dropped as stale | Mean age | Max age |
|---|---|---|---|---|
| synchronous barrier | 5.6 | 0 | 0 | 0 |
| async, window k = 0 | 14.8 | 62% | 0 | 0 |
| async, k = 1 | 31.3 | 19% | 0.53 | 1 |
| async, k = 4 | 38.1 | 2.3% | 0.86 | 4 |
| async, k = infinite | 38.7 | 0 | 1.0 | 41 |

Utilisation falls as the tail gets heavier and as N grows, as the bound says. The asynchronous queue with no window
reaches `N / E[T]`, 6.9 times the synchronous rate. The window k is the trade the post names: k = 1 keeps every
trajectory within one policy version but throws away 19% of the work. With no window, the oldest trajectory was 41
versions old. At sigma = 1.5 it was 256. C12's `agentrl/system.py` applies the same kind of window to a real policy.

## Not measured

- Anything at language-model scale. No LLM, no web or SWE environment, no 7B agent. The post's quoted results
  (GiGPO gains, the interaction-horizon curriculum's +10% on web navigation, environment-count scaling, the task-length
  doubling trend) are not reproduced.
- The granularity optimum (Figure 3), early experience, the echo trap, context folding, decoupling the agent from the
  trainer, irreversible actions and staging layers, and environment diversity. These are not built here.
- The importance-ratio correction for stale trajectories. Section 6 measures age and throughput only. C12's
  `make_batches(..., max_lag=k)` applies a staleness window during real training.
- Anchor grouping when states do not repeat exactly.
- GPU timings. Nothing here needs a GPU.
