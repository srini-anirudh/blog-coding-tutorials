# C12: Train an Agent With Reinforcement Learning

Code for the tutorial [Train an Agent With Reinforcement Learning](https://srini-anirudh.github.io/blogs/train-an-agent-with-rl/),
part 12 of the Coding Tutorials series. Package: `agentrl`.

The goal is to move from imitating successful agent trajectories to learning a strategy by interacting with an
environment. Everything is small enough to train on a CPU: a toy filesystem environment, a scripted expert, a tiny
decoder-only policy with a value head, behaviour cloning, a GRPO-style trainer, and the full agent-RL pipeline around it.
The same rollout and update code also runs a Hugging Face causal LM (tiny Llama, SmolLM2-135M with LoRA).

## What you build

| Step | Module | What it does |
|---|---|---|
| 1 Toy environment | `tasks.py`, `env.py`, `expert.py` | Five task levels: single tool, multiple tools, dependent tools, failure recovery, long horizon. Tools `list`, `read`, `search`, `edit`, `submit`. Outcome reward `1[final files == target]`. |
| 2 Environment API | `env.py` | `reset()`, `observe()`, `step(action)`, `reward()`, `done()`, all text in and text out. |
| 3 Trajectory | `trajectory.py`, `codec.py` | `tau = (s_0, a_0, r_0, ..., s_T)` stored twice: the text steps, and the exact token ids with an action mask, sampled log-probs and the step of each token. |
| 4 Behaviour cloning | `policy.py`, `batching.py`, `bc.py` | `L = -sum_t log pi(a*_t | s_t)` over action tokens only. |
| 5 Distribution shift | `shift.py` | Inject harmless mistakes, then measure success and how often the policy acts in a state no demonstration contains. |
| 6 Online rollouts | `sampling.py`, `rollout_naive.py`, `rollout.py` | Reference rollout (batch 1) and a batched rollout that must agree with it. |
| 7 Outcome reward | `rewards.py` | Final success only. |
| 8 GRPO update | `advantages.py`, `grpo.py`, `online.py` | G trajectories per task, group-relative advantages, clipped surrogate over action tokens, optional KL to the BC policy. |
| 9 Credit assignment | `advantages.py`, `grpo.py` | Trajectory-level advantage against action-level advantage (value head + GAE). |
| 10 Reward shaping | `rewards.py` | `activity` (a deliberate reward hack) and `potential` (potential-based shaping, which telescopes to the outcome). |
| 11 Exploration | `online.py` | Sampling temperature, diverse initial states (perturbed starts, failure seeds), diverse tasks. |
| 12 Curriculum | `online.py` | Advance through task mixes when rolling training success passes a threshold. |
| 13 Failure recovery | `failures.py` | Timeout, tool error, invalid input, missing file. Seeded per episode. |
| 14 Offline vs online | `offline.py` | Offline GRPO on a stored buffer, and filtered BC, against online GRPO with the same gradient-step budget. |
| 15 Training system | `system.py` | Task sampler, env workers, rollouts, trajectory store (JSON lines), verifier, batches, update, resume. |
| Trusted implementation | `hf_policy.py` | `LlamaForCausalLM` and SmolLM2-135M + LoRA behind the same `PolicyOutput` interface. |

```
c12-agent-rl/
├── README.md
├── conftest.py              # puts this folder on sys.path; tests use 2 threads
├── agentrl/                 # the library (modules above, plus evaluate.py: the four evaluation suites)
├── scripts/
│   ├── demo_env.py          # one expert episode per level, and the token view of one
│   ├── train_bc.py          # behaviour cloning -> runs/bc.pt
│   ├── distribution_shift.py
│   ├── experiments.py       # main, credit, shaping, curriculum, offline, exploration, groupsize, onpolicy
│   ├── recovery_demo.py
│   ├── export_figures.py    # replay.json for the post's replay figure
│   └── train_system.py
├── benchmarks/
│   ├── bench_rollout.py     # naive vs batched rollout, policy vs tool time, env workers
│   └── bench_policy.py      # one GRPO iteration per policy type
├── results/                 # measured outputs, as text and JSON
└── tests/                   # 51 tests
```

Checkpoints go to `runs/`, which git ignores.

## How to run

All commands run from the repository root with the shared virtual environment (`.venv/bin/python`, written `python`
below). Scripts use the thread count from `OMP_NUM_THREADS`. The runs below used 8.

```bash
# tests
python -m pytest c12-agent-rl -m "not slow and not gpu and not network"   # 50 tests, about 20 s
python -m pytest c12-agent-rl -m network                                # downloads SmolLM2-135M-Instruct

# environment and behaviour cloning
python c12-agent-rl/scripts/demo_env.py
python c12-agent-rl/scripts/train_bc.py --demos 30000 --steps 3000        # writes runs/bc.pt, about 2.5 min
python c12-agent-rl/scripts/distribution_shift.py

# benchmarks (pass a label for the hardware; they also print the thread count)
python c12-agent-rl/benchmarks/bench_rollout.py --hardware "AMD EPYC 7R13"
python c12-agent-rl/benchmarks/bench_policy.py --tasks 4 --G 4 --hardware "AMD EPYC 7R13"

# RL experiments: each starts from runs/bc.pt and writes results/<name>.json
python c12-agent-rl/scripts/experiments.py main          # saves runs/grpo.pt
python c12-agent-rl/scripts/recovery_demo.py             # needs runs/bc.pt and runs/grpo.pt
python c12-agent-rl/scripts/experiments.py credit        # saves runs/grpo_action.pt
python c12-agent-rl/scripts/export_figures.py            # needs runs/grpo_action.pt
python c12-agent-rl/scripts/experiments.py shaping
python c12-agent-rl/scripts/experiments.py curriculum
python c12-agent-rl/scripts/experiments.py offline
python c12-agent-rl/scripts/experiments.py exploration
python c12-agent-rl/scripts/experiments.py groupsize
python c12-agent-rl/scripts/experiments.py onpolicy
python c12-agent-rl/scripts/train_system.py --iters 10
```

`experiments.py` takes `--iters N` to shorten any run, and `--seed`. If the default Hugging Face cache is not
writable, set `HF_HOME` to a folder you own before the network test or `bench_policy.py`.

## Results

All numbers are measured. Success is greedy success on fresh evaluation tasks (seeds 0 to n-1; training seeds start at
10^6). Every RL run is a single seed, so differences of a few points are noise. CPU runs: AMD EPYC 7R13, 8 CPU threads,
torch 2.15.0. GPU rows: NVIDIA H100 80GB HBM3, torch 2.15.0+cu132, CUDA 13.2.

### Tests

| Selection | Result |
|---|---|
| `-m "not slow and not gpu and not network"` | 50 passed in 20 s |
| `-m network` | 1 passed (SmolLM2-135M + LoRA runs one episode) |
| `-m "slow or gpu"` | none in this folder |

### Behaviour cloning and distribution shift (`train_bc.txt`, `shift_bc.txt`)

686,465 parameters, 30,000 demonstrations at levels 1 to 3, 3,000 steps, 155 s. 300 tasks per suite.

| Suite | clean | perturbed | failures | long |
|---|---|---|---|---|
| BC success | 0.797 | 0.143 | 0.200 | 0.000 |

| Injected mistakes | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| BC success | 0.797 | 0.173 | 0.087 | 0.043 |
| Scripted expert from the same start | 1.000 | 1.000 | 1.000 | 1.000 |
| Decisions in a state absent from demos | 0.000 | 0.946 | 0.992 | 1.000 |

### Rollout and policy benchmarks (`bench_rollout.txt`, `bench_policy.txt`, `*_gpu.txt`)

64 episodes, levels 1 to 5, greedy.

| Rollout | Device | Batch | Episodes/s |
|---|---|---|---|
| naive | AMD EPYC 7R13 | 1 | 54.6 |
| batched | AMD EPYC 7R13 | 64 | 248.8 |
| batched | AMD EPYC 7R13 | 256 | 285.3 |
| batched | NVIDIA H100 | 64 | 147.9 (includes first-call CUDA warm-up) |
| batched | NVIDIA H100 | 256 | 822.7 |

About 91 to 94% of rollout time is spent inside the policy forward. With 20 ms tools at batch 64, 32 thread workers
take throughput from 12.6 to 117.3 episodes/s.

One GRPO iteration, 4 tasks x G=4 = 16 episodes, untrained policies:

| Policy | Device | Trainable params | Rollout + update (s) |
|---|---|---|---|
| TinyPolicy d=128 L=3 | AMD EPYC 7R13 | 686,465 | 0.43 |
| LlamaForCausalLM d=128 L=3 | AMD EPYC 7R13 | 832,001 | 0.66 |
| SmolLM2-135M + LoRA r=16 | AMD EPYC 7R13 | 1,843,777 | 32.69 |
| SmolLM2-135M + LoRA r=16 | NVIDIA H100 80GB HBM3 | 1,843,777 | 3.52 |

### RL experiments

Start for every run is the BC checkpoint: clean 0.78, perturbed 0.13, failures 0.21, long 0.00 (150 tasks per suite).

All rows: greedy success after training, 150 tasks per suite, single seed. "Train success" is the last iteration's
success on the sampled training tasks. Times are wall-clock on the 8-thread CPU and include periodic evaluation. The
curriculum and groupsize runs shared a busy node with other jobs, so their times are not comparable.

| Experiment | Setting | Iterations | clean | perturbed | failures | long | Train success | Time (s) |
|---|---|---|---|---|---|---|---|---|
| BC start | | | 0.78 | 0.13 | 0.21 | 0.00 | | |
| `main` | full mix, lr 2e-4, 2 epochs, 24 tasks x G=8 | 120 | 0.80 | 0.03 | 0.19 | 0.00 | 0.10 | 464 |
| `onpolicy` | full mix, lr 3e-4, 1 epoch, 96 tasks x G=8 | 60 | 0.79 | 0.31 | 0.22 | 0.00 | 0.22 | 759 |
| `credit` | level 4 and 5 mix, trajectory credit | 80 | | | 0.22 | 0.00 | | 466 |
| `credit` | action credit (value head + GAE) | 80 | | | 0.21 | 0.00 | | 110 |
| `credit` | action credit + potential reward | 80 | | | 0.23 | 0.00 | | 126 |
| `shaping` | outcome reward, levels 2 and 3 | 60 | 0.81 | | | | 0.75 | 80 |
| `shaping` | activity reward (the hack) | 60 | 0.73 | | | | 0.66 | 86 |
| `shaping` | potential reward | 60 | 0.81 | | | | 0.75 | 74 |
| `curriculum` | hardest first (level 5 only) | 100 | 0.78 | 0.13 | 0.21 | 0.00 | 0.00 | 165 |
| `curriculum` | uniform over levels 2 to 5 | 100 | 0.47 | 0.01 | 0.21 | 0.00 | 0.15 | 2589 |
| `curriculum` | staged, threshold 0.75 (still on stage 0 at the end) | 100 | 0.74 | 0.01 | 0.21 | 0.00 | 0.22 | 1043 |
| `exploration` | temperature 0.5 | 60 | 0.79 | 0.01 | 0.21 | 0.00 | 0.17 | 330 |
| `exploration` | temperature 1.0 | 60 | 0.79 | 0.01 | 0.19 | 0.00 | 0.17 | 209 |
| `exploration` | temperature 1.5 | 60 | 0.81 | 0.03 | 0.19 | 0.00 | 0.13 | 359 |
| `groupsize` | G=2, 96 groups (192 rollouts per iteration) | 60 | 0.80 | 0.00 | 0.21 | 0.00 | 0.17 | 320 |
| `groupsize` | G=4, 48 groups | 60 | 0.81 | 0.05 | 0.20 | 0.00 | 0.16 | 264 |
| `groupsize` | G=8, 24 groups | 60 | 0.79 | 0.01 | 0.19 | 0.00 | 0.17 | 536 |
| `groupsize` | G=16, 12 groups | 60 | 0.79 | 0.19 | 0.21 | 0.00 | 0.16 | 2230 |

`credit` and `shaping` only evaluate the suites they train on, so the other cells are empty.

Offline against online (`offline.json`). A buffer of 15,360 trajectories (success 0.247) was collected once from the BC
policy. Each method then made the same 480 gradient steps.

| Method | New environment steps | clean | perturbed | failures | long |
|---|---|---|---|---|---|
| Online GRPO | 15,360 (collected as it trains) | 0.81 | 0.02 | 0.20 | 0.00 |
| Offline GRPO on the stored buffer | 0 | 0.77 | 0.05 | 0.21 | 0.00 |
| Filtered BC (successful trajectories only) | 0 | 0.80 | 0.27 | 0.25 | 0.00 |

Offline GRPO's clipped fraction rises from 0.17 to about 0.5 to 0.6 during training. Online stays near 0.1.

Recovery (`recovery.txt`, greedy, 300 failure-suite episodes). Recovery means success among episodes where at least one
tool call returned an error.

| Policy | Success | Recovered / episodes with an error |
|---|---|---|
| BC | 0.200 | 5 / 218 |
| after `main` | 0.177 | 0 / 221 |

### Training system (`train_system.txt`, `system.json`)

10 iterations, 24 tasks x G=8, from the BC checkpoint. 1,920 trajectories stored (5.87 MB of JSON lines), 1,920
verified, 0 mismatches.

| Tools | Workers | Total (s) | Rollout share | Update share |
|---|---|---|---|---|
| instant | 0 | 40.0 | 68.6% | 30.4% |
| 20 ms per call | 0 | 318.2 | 96.4% | 3.5% |
| 20 ms per call | 32 | 44.9 | 76.3% | 22.7% |

## What the results say

- The environment, the data path, the losses and the system work. The tests check them against references: the batched
  rollout matches the naive one, the sampled log-probs match a training forward pass, group-relative advantages match
  a NumPy reference, and the same code runs a Hugging Face Llama and SmolLM2 with LoRA.
- Behaviour cloning is strong on clean tasks and fragile everywhere else. One harmless injected mistake takes success
  from 0.80 to 0.17, and the expert still solves every one of those starts. That is the distribution-shift gap.
- The default GRPO run did not improve held-out success. Over 120 iterations clean stayed at 0.80, perturbed fell from
  0.13 to 0.03, failures stayed at 0.19 and long at 0.00. Sampled training success fell from about 0.29 to 0.10 and the
  sampled policy drifted to longer episodes that never submit (submitted 0.98 to 0.35). Greedy evaluation does not show
  this. After RL the policy recovered from 0 of 221 error episodes, against 5 of 218 for BC.
- The cause is the reward signal, not a bug in the update. From the BC policy at temperature 1.0, 563 of 800 level-4
  rollouts hit a tool error and only 13 of those succeeded. On level 5, 0 of 782 succeeded. With a reward of 1 for a
  finished task, most groups have no variance (0.7 to 0.96 of groups late in the runs) and the update has little to
  reinforce. Lower learning rate, a KL penalty, Dr. GRPO scaling and temperature 0.7 were tried in 30-iteration checks
  with the same picture. Those checks are not saved in `results/`.
- What helped most was a bigger on-policy batch: one gradient step per iteration on 96 tasks x G=8. It raised perturbed
  success from 0.13 to 0.31 without hurting clean tasks. It did not move failures or long. Group size 16 also reached
  0.19 on perturbed. Both are single seeds.
- Action-level credit (value head + GAE) did not beat trajectory-level credit here. All three credit variants end
  within 0.02 of each other on the failures suite, which is within noise.
- Reward shaping shows the hack clearly. The `activity` reward reaches return 1.27 while success falls to 0.66 and
  clean evaluation falls to 0.73. The potential-based reward gives the same result as the outcome reward, as it must,
  because its shaping terms telescope to zero.
- Curriculum did not help. Hardest-first never saw a success, so it learned nothing. The staged curriculum was still on
  stage 0 after 100 iterations, because its 0.75 training-success threshold was never reached. Uniform sampling over
  levels 2 to 5 lowered clean success to 0.47.
- Filtered BC on the stored buffer matched or beat every GRPO variant (perturbed 0.27, failures 0.25) with zero new
  environment steps. Offline GRPO did not learn: its clipped fraction climbs as the policy leaves the behaviour policy.
- Level 5 (two subtasks with failures) stays at 0.00 for every method. The agent-RL tooling is complete, but a policy
  this small, cloned only on clean levels 1 to 3, does not reach it with outcome reward alone.
- Systems: the tool time dominates when tools are slow. With 20 ms per call, inline tools take 96% of an iteration and
  32 thread workers cut an iteration's total from 318 s to 45 s, close to the instant-tool figure of 40 s. The H100 runs
  SmolLM2-135M + LoRA about 9 times faster than 8 CPU threads (3.5 s against 32.7 s per iteration). It gives no
  advantage on the tiny policy at batch 64, where kernel launch and warm-up dominate.

## Notes

- Tests set 2 threads in `conftest.py`. Scripts do not set a thread count.
- `bench_policy.py` and `bench_rollout.py` add a CUDA (or MPS) row when one is visible. The GPU rows above came from
  that path. The CPU rows in the `*_gpu.txt` files ran on the GPU node's host, which has the same CPU model.
- Earlier versions of these results were measured on an Apple M5 laptop. Every file in `results/` has been replaced by
  the runs above.
